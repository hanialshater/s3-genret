"""GR1: a TIGER-style seq2seq retriever. A pretrained T5 reads the query (or a product's text, for the indexing task)
and its decoder writes the product's semantic id, one new token per (level, code). Decoding is the trie-constrained
beam search of genret.beam. Design: lab/predictions/genret-GR1.md.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch

from genret.beam import NEG, beam_search
from genret.sids import SidTrie
from genret.text import _clean

S_RATE = 0.3          # Substitutes sampled at 0.3 of the Exact rate per epoch (ESCI gains E 1.0, S 0.1)


def index_text(row) -> str:
    t = _clean(row.get("product_title"))
    b = _clean(row.get("product_brand"))
    return f"product: {t}" + (f" | brand: {b}" if b else "")


def query_text(q) -> str:
    return f"query: {_clean(q)}"


class SidVocab:
    """Token ids of the codes: level l, code c -> base + offsets[l] + c, appended after the text vocabulary."""

    def __init__(self, ks, base: int):
        self.ks, self.base = list(ks), base
        self.offsets = np.concatenate([[0], np.cumsum(self.ks)[:-1]]).astype(int).tolist()
        self.size = int(sum(self.ks))

    def tokens(self, codes) -> list:
        return [self.base + self.offsets[l] + int(c) for l, c in enumerate(codes)]


def build_examples(fit, catalogue, codes, ids, epoch: int, indexing: bool = True, seed: int = 0, s_rate=S_RATE):
    """One epoch's (input text, codes) examples: retrieval pairs from the fit split (+ indexing, if asked)."""
    rng = random.Random(seed * 1000 + epoch)
    pos = {p: i for i, p in enumerate(ids)}
    ex = []
    for q, p, l in zip(fit["query"], fit["product_id"], fit["esci_label"]):
        if l == "E" or (l == "S" and rng.random() < s_rate):
            i = pos.get(str(p))
            if i is not None:
                ex.append((query_text(q), codes[i]))
    if indexing:
        for i, r in enumerate(catalogue.to_dict("records")):
            ex.append((index_text(r), codes[i]))
    rng.shuffle(ex)
    return ex


def make_model(name: str, ks):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(name)
    model = T5ForConditionalGeneration.from_pretrained(name)
    base = model.config.vocab_size
    vocab = SidVocab(ks, base)
    model.resize_token_embeddings(base + vocab.size + 1)     # +1: an end token is not needed (fixed depth)
    return tok, model, vocab


def train(model, tok, vocab, epochs_examples, steps: int, batch: int = 256, lr: float = 1e-3, warmup: int = 1000,
          device: str = "cpu", max_len: int = 64, log_every: int = 200, log=print):
    """epochs_examples(epoch) -> list of examples. Runs `steps` optimizer steps, epoch after epoch."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, math.sqrt(warmup / max(s + 1, warmup))))
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    step, epoch, t0, hist = 0, 0, time.time(), []
    while step < steps:
        ex = epochs_examples(epoch)
        for s in range(0, len(ex) - batch + 1, batch):
            b = ex[s:s + batch]
            enc = tok([x for x, _ in b], max_length=max_len, truncation=True, padding=True, return_tensors="pt")
            labels = torch.tensor([vocab.tokens(c) for _, c in b])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device),
                            labels=labels.to(device))
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0 or step == steps:
                hist.append({"step": step, "epoch": epoch, "loss": float(out.loss), "s": round(time.time() - t0, 1)})
                log(f"gr1 step {step} epoch {epoch} loss {float(out.loss):.4f} ({time.time() - t0:.0f}s)")
            if step >= steps:
                break
        epoch += 1
    model.eval()
    return hist


@torch.no_grad()
def retrieve(model, tok, vocab, queries, trie: SidTrie, width: int = 100, batch: int = 32, device: str = "cpu",
             max_len: int = 64, unconstrained: bool = False):
    """For each query, the generated ids in score order: [(codes tuple, score), ...] (fewer when the trie runs out)."""
    model.to(device).eval()
    depth, ks = len(vocab.ks), vocab.ks
    out = []
    for s in range(0, len(queries), batch):
        qs = [query_text(q) for q in queries[s:s + batch]]
        enc = tok(qs, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
        hidden = model.get_encoder()(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
        B = len(qs)
        start = model.config.decoder_start_token_id

        def step(state, prefixes, l):
            h, m = state
            w = prefixes.shape[0] // B
            dec = torch.full((prefixes.shape[0], l + 1), start, dtype=torch.long)
            for j in range(l):
                dec[:, j + 1] = prefixes[:, j] + vocab.base + vocab.offsets[j]
            o = model(encoder_outputs=(h.repeat_interleave(w, 0),), attention_mask=m.repeat_interleave(w, 0),
                      decoder_input_ids=dec.to(device))
            lo = vocab.base + vocab.offsets[l]
            return o.logits[:, -1, lo:lo + ks[l]].float().log_softmax(-1), state

        codes, scores = beam_search(step, (hidden, enc.attention_mask), B, depth, ks, width, trie=trie,
                                    unconstrained=unconstrained)
        for b in range(B):
            out.append([(tuple(c.tolist()), float(v)) for c, v in zip(codes[b], scores[b]) if v > NEG / 2])
    return out


def to_ranked(results, trie: SidTrie, k: int = 100) -> list:
    """Run-file entries: a product id for a real id (TIGER's ids are unique), null for an id that does not exist."""
    ranked = []
    for codes, _ in results[:k]:
        items = trie.items(codes)
        ranked.append(items[0] if len(items) == 1 else ({"sid": "-".join(map(str, codes)), "pid": items[0]}
                                                         if items else None))
    return ranked
