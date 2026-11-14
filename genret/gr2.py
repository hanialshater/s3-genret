"""GR2: a SID-LLM-style generative retriever (arXiv:2611.08804), scaled to one GPU. Design:
lab/predictions/genret-GR2.md.

A pretrained causal LM (Qwen3-0.6B-Base) gets one new token per (level, code) of the 3 x 8192 RQ-KMeans ids, plus a
few task tokens, appended after its text vocabulary and initialised at the mean of the existing rows. Training:
- stage 1, alignment: only the new rows train (the embedding is tied to the output head in Qwen3-0.6B), on
  attributes -> id and id -> title;
- stage 2, joint pre-training, all weights: attributes -> id, id -> title, query -> id (the fit judgments, stand-in 1)
  and co-judgment sequences (stand-in 2: the Exact products of one fit query as a "session");
- MTP: three heads on the hidden state at <sid_start>, each predicting one level from the prompt alone.
Decoding: MTP with a beam walk over the trie (genret.mtp), or autoregressive constrained beam search (arm B3).
Loss is on the completion only. Ids are built directly (no strings for the 24,576 code tokens).
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from genret.beam import NEG, beam_search
from genret.text import _clean

SPECIALS = ("sid_start", "sid_end", "seq", "title")
S_RATE = 0.3


class Vocab:
    """Token ids: text ids below `base`; code (l, c) at base + l*K + c; specials after the codes."""

    def __init__(self, base: int, ks):
        self.base, self.ks = base, list(ks)
        self.offsets = np.concatenate([[0], np.cumsum(self.ks)[:-1]]).astype(int).tolist()
        n = int(sum(self.ks))
        self.special = {s: base + n + i for i, s in enumerate(SPECIALS)}
        self.size = base + n + len(SPECIALS)

    def sid(self, codes) -> list:
        return [self.base + self.offsets[l] + int(c) for l, c in enumerate(codes)]


def add_sid_vocab(model, base: int, ks) -> Vocab:
    v = Vocab(base, ks)
    old = model.get_input_embeddings().weight.data[:base]
    mean = old.float().mean(0)
    model.resize_token_embeddings(v.size)
    emb = model.get_input_embeddings().weight.data
    emb[base:] = mean.to(emb.dtype)
    out = model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        out.weight.data[base:] = out.weight.data[:base].float().mean(0).to(out.weight.dtype)
    return v


def attr_text(row) -> str:
    t = _clean(row.get("product_title"))[:200]
    b = _clean(row.get("product_brand"))
    c = _clean(row.get("product_color"))
    return "product: " + t + (f" | brand: {b}" if b else "") + (f" | color: {c}" if c else "")


def _arr(x):
    return np.asarray(x, dtype=np.int32)            # millions of examples: arrays, not lists of ints


class Examples:
    """Builds (input ids, number of prompt tokens) examples. The loss is on the tokens after the prompt."""

    def __init__(self, encode, vocab: Vocab, catalogue, codes, ids, fit, eos: int, seed: int = 0):
        self.enc, self.v, self.codes, self.ids, self.fit, self.eos = encode, vocab, codes, ids, fit, eos
        self.rows = catalogue.to_dict("records")
        self.pos = {p: i for i, p in enumerate(ids)}
        self.rng = random.Random(seed)
        self._attr = None

    def attr(self, i):
        p = self.enc(attr_text(self.rows[i]))[:64]
        sp = self.v.special
        return p + [sp["sid_start"]], self.v.sid(self.codes[i]) + [sp["sid_end"]]

    def attr_to_sid(self, i):
        prompt, target = self.attr(i)
        return _arr(prompt + target), len(prompt)

    def sid_to_title(self, i):
        sp = self.v.special
        prompt = [sp["sid_start"]] + self.v.sid(self.codes[i]) + [sp["sid_end"], sp["title"]]
        target = self.enc(_clean(self.rows[i].get("product_title"))[:200])[:48] + [self.eos]
        return _arr(prompt + target), len(prompt)

    def query_pairs(self, epoch: int):
        rng = random.Random(epoch * 7919 + 1)
        out = []
        sp = self.v.special
        for q, p, l in zip(self.fit["query"], self.fit["product_id"], self.fit["esci_label"]):
            if l == "E" or (l == "S" and rng.random() < S_RATE):
                i = self.pos.get(str(p))
                if i is not None:
                    prompt = self.enc("query: " + _clean(q))[:32] + [sp["sid_start"]]
                    out.append((_arr(prompt + self.v.sid(self.codes[i]) + [sp["sid_end"]]), len(prompt)))
        return out

    def cojudged(self, epoch: int, max_items: int = 12):
        """Stand-in 2: the Exact products of each fit query, shuffled, as one sequence of ids (no query text)."""
        rng = random.Random(epoch * 104729 + 3)
        sp = self.v.special
        out = []
        ex = self.fit[self.fit["esci_label"] == "E"]
        for _, g in ex.groupby("query_id"):
            items = [self.pos[str(p)] for p in g["product_id"] if str(p) in self.pos]
            if len(items) < 2:
                continue
            rng.shuffle(items)
            seq = [sp["seq"]]
            for i in items[:max_items]:
                seq += [sp["sid_start"]] + self.v.sid(self.codes[i]) + [sp["sid_end"]]
            out.append((_arr(seq), 1 + len(self.v.ks) + 2))   # the first id is context, the rest is predicted
        return out


def collate(batch, pad: int, max_len: int = 96):
    n = min(max_len, max(len(x) for x, _ in batch))
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for r, (x, k) in enumerate(batch):
        x = x[:n]
        ids[r, :len(x)] = torch.as_tensor(np.asarray(x, dtype=np.int64))
        mask[r, :len(x)] = 1
        labels[r, k:len(x)] = torch.as_tensor(np.asarray(x[k:], dtype=np.int64))
    return ids, mask, labels


def _backbone(model):
    return getattr(model, getattr(model, "base_model_prefix", "model"), None) or model.model


def _run(model, ids, mask):
    pos = (mask.long().cumsum(1) - 1).clamp(min=0)        # positions count real tokens only (left padding)
    return _backbone(model)(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state


def lm_loss(model, ids, mask, labels):
    """Next-token loss on the completion only, applying the output head to the labelled positions alone (a full
    [batch, length, 176k-token vocabulary] logit tensor does not fit next to a 0.6B model)."""
    h = _run(model, ids, mask)[:, :-1]
    y = labels[:, 1:]
    sel = y != -100
    logits = model.get_output_embeddings()(h[sel]).float()
    return F.cross_entropy(logits, y[sel])


def last_logits(model, ids, mask):
    return model.get_output_embeddings()(_run(model, ids, mask)[:, -1])


def train_lm(model, examples, steps: int, batch: int, lr: float, device: str, pad: int, warmup: int = 200,
             only_new_rows_from: int = None, weight_decay: float = 0.1, decay_frac: float = 0.2, log=print,
             log_every: int = 200, max_len: int = 96, grad_ckpt: bool = False):
    """Warmup-stable-decay AdamW over `examples` (a list, cycled). With `only_new_rows_from`, only the embedding rows
    from that id on train (stage 1)."""
    model.to(device).train()
    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if only_new_rows_from is not None:
        for p in model.parameters():
            p.requires_grad_(False)
        emb = model.get_input_embeddings().weight
        emb.requires_grad_(True)
        keep = torch.zeros(emb.shape[0], 1, device=emb.device if emb.device.type != "meta" else "cpu")
        keep[only_new_rows_from:] = 1.0
        emb.register_hook(lambda g: g * keep.to(g.device, g.dtype))
        params = [emb]
        weight_decay = 0.0
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    stable_end = int(steps * (1 - decay_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) if s < stable_end else max(0.0, (steps - s) / max(1, steps - stable_end)))
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    lens = np.fromiter((len(x) for x, _ in examples), dtype=np.int32, count=len(examples))

    def epoch_batches(seed):
        """Shuffled, then length-sorted inside windows of 64 batches, then the batches shuffled: little padding."""
        r = np.random.default_rng(seed)
        order = r.permutation(len(examples))
        w = batch * 64
        out = []
        for s0 in range(0, len(order) - batch + 1, w):
            chunk = order[s0:s0 + w]
            chunk = chunk[np.argsort(lens[chunk], kind="stable")]
            out += [chunk[i:i + batch] for i in range(0, len(chunk) - batch + 1, batch)]
        r.shuffle(out)
        return out

    batches, bi, ep, t0, hist = epoch_batches(0), 0, 0, time.time(), []
    for step in range(1, steps + 1):
        if bi >= len(batches):
            ep += 1
            batches, bi = epoch_batches(ep), 0
        b = [examples[i] for i in batches[bi]]
        bi += 1
        ids, mask, labels = collate(b, pad, max_len)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = lm_loss(model, ids.to(device), mask.to(device), labels.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            hist.append({"step": step, "loss": float(loss), "s": round(time.time() - t0, 1)})
            log(f"lm step {step}/{steps} loss {float(loss):.4f} lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
    if grad_ckpt and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    return hist


def left_pad(prompts, pad: int):
    n = max(map(len, prompts))
    ids = torch.full((len(prompts), n), pad, dtype=torch.long)
    mask = torch.zeros((len(prompts), n), dtype=torch.long)
    for r, p in enumerate(prompts):
        ids[r, n - len(p):] = torch.tensor(p)
        mask[r, n - len(p):] = 1
    return ids, mask


def query_prompt(encode, vocab, q):
    return encode("query: " + _clean(q))[:32] + [vocab.special["sid_start"]]


def hidden_at_last(model, ids, mask):
    """The final hidden state (after the last norm) at the last position; prompts are left-padded, so that is
    <sid_start>."""
    return _run(model, ids, mask)[:, -1]


def train_mtp(model, heads, encode, vocab, pairs_fn, steps: int, batch: int, device: str, pad: int, lr_heads=1e-3,
              lr_backbone=2e-5, log=print, log_every: int = 200):
    """pairs_fn(epoch) -> list of (query text, codes). Cross-entropy per level on the heads; the backbone trains
    with them at a low rate."""
    model.to(device).train()
    heads.to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": heads.parameters(), "lr": lr_heads},
                             {"params": model.parameters(), "lr": lr_backbone}], weight_decay=0.0)
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    epoch, data, cur, t0, hist = 0, pairs_fn(0), 0, time.time(), []
    for step in range(1, steps + 1):
        if cur + batch > len(data):
            epoch += 1
            data, cur = pairs_fn(epoch), 0
        b = data[cur:cur + batch]
        cur += batch
        ids, mask = left_pad([query_prompt(encode, vocab, q) for q, _ in b], pad)
        y = torch.tensor(np.array([c for _, c in b]), dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            h = hidden_at_last(model, ids.to(device), mask.to(device))
        loss = heads.loss(h.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(heads.parameters()) + list(model.parameters()), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            hist.append({"step": step, "loss": float(loss), "s": round(time.time() - t0, 1)})
            log(f"mtp step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    heads.eval()
    return hist


@torch.no_grad()
def decode_mtp(model, heads, encode, vocab, queries, trie, width: int, device: str, pad: int, batch: int = 64):
    from genret.mtp import mtp_decode
    out = []
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    for s in range(0, len(queries), batch):
        ids, mask = left_pad([query_prompt(encode, vocab, q) for q in queries[s:s + batch]], pad)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            h = hidden_at_last(model, ids.to(device), mask.to(device))
        codes, scores = mtp_decode(heads, h.float(), trie, width=width)
        for b in range(len(codes)):
            out.append([(tuple(c.tolist()), float(v)) for c, v in zip(codes[b], scores[b]) if v > NEG / 2])
    return out


@torch.no_grad()
def decode_ar(model, encode, vocab, queries, trie, width: int, device: str, pad: int, batch: int = 8):
    """Autoregressive constrained beam search over the code tokens (arm B3); the prompt is recomputed each step."""
    out = []
    depth, ks = len(vocab.ks), vocab.ks
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    for s in range(0, len(queries), batch):
        prompts = [query_prompt(encode, vocab, q) for q in queries[s:s + batch]]
        B = len(prompts)

        def step(state, prefixes, l):
            w = prefixes.shape[0] // B
            seqs = []
            for r in range(prefixes.shape[0]):
                seqs.append(prompts[r // w] + [vocab.base + vocab.offsets[j] + int(prefixes[r, j]) for j in range(l)])
            ids, mask = left_pad(seqs, pad)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = last_logits(model, ids.to(device), mask.to(device))
            lo = vocab.base + vocab.offsets[l]
            return logits[:, lo:lo + ks[l]].float().log_softmax(-1), state

        codes, scores = beam_search(step, None, B, depth, ks, width, trie=trie)
        for b in range(B):
            out.append([(tuple(c.tolist()), float(v)) for c, v in zip(codes[b], scores[b]) if v > NEG / 2])
    return out


def expand(results, trie, static_rank: dict, cap: int):
    """Candidate ids -> (product, generator score), every product of a shared id in its static order, up to cap."""
    out = []
    for codes, sc in results:
        for p in sorted(trie.items(codes), key=lambda p: static_rank.get(p, math.inf)):
            out.append((p, codes, sc))
            if len(out) >= cap:
                return out
    return out


def ranked_entries(cands, k: int):
    """Run-file entries for (product, codes, score) in order: {"sid", "pid"}, so a shared id names its product."""
    return [{"sid": "-".join(map(str, c)), "pid": p} for p, c, _ in cands[:k]]
