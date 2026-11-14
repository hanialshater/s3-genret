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


def retrieval_pairs(fit, ids, codes, rng, s_rate=S_RATE):
    pos = {p: i for i, p in enumerate(ids)}
    out = []
    for q, p, l in zip(fit["query"], fit["product_id"], fit["esci_label"]):
        if l == "E" or (l == "S" and rng.random() < s_rate):
            i = pos.get(str(p))
            if i is not None:
                out.append((query_text(q), codes[i]))
    return out


class ExampleStream:
    """Epochs of examples: the retrieval pairs, mixed with indexing examples drawn from the catalogue so that indexing
    is a fixed share of every epoch (`index_share`, 0 for arm A2). The catalogue is walked in a shuffled cycle, so
    every product is indexed equally often and indexing is never front-loaded and then forgotten."""

    def __init__(self, fit, catalogue, codes, ids, index_share: float = 0.64, seed: int = 0, s_rate=S_RATE):
        self.fit, self.codes, self.ids, self.share, self.seed, self.s_rate = fit, codes, ids, index_share, seed, s_rate
        self.index = [index_text(r) for r in catalogue.to_dict("records")] if index_share > 0 else []
        self.order, self.cursor = [], 0
        self.rng = random.Random(seed)

    def _next_index(self, n):
        out = []
        while len(out) < n:
            if self.cursor >= len(self.order):
                self.order = list(range(len(self.index)))
                self.rng.shuffle(self.order)
                self.cursor = 0
            i = self.order[self.cursor]
            self.cursor += 1
            out.append((self.index[i], self.codes[i]))
        return out

    def __call__(self, epoch: int):
        rng = random.Random(self.seed * 1000 + epoch)
        ex = retrieval_pairs(self.fit, self.ids, self.codes, rng, self.s_rate)
        if self.share > 0:
            ex += self._next_index(int(round(len(ex) * self.share / (1 - self.share))))
        rng.shuffle(ex)
        return ex


def build_examples(fit, catalogue, codes, ids, epoch: int, indexing: bool = True, seed: int = 0, s_rate=S_RATE):
    """One epoch with all of the catalogue indexed once (kept for tests and small runs)."""
    rng = random.Random(seed * 1000 + epoch)
    ex = retrieval_pairs(fit, ids, codes, rng, s_rate)
    if indexing:
        ex += [(index_text(r), codes[i]) for i, r in enumerate(catalogue.to_dict("records"))]
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
