"""Trie-constrained beam search for fixed-depth semantic ids, model-agnostic and batched over queries.

`step(state, prefixes)` gets the decoder state and a [B*W, l] tensor of the codes generated so far and returns
log-probabilities [B*W, K_l] over level l's codes (plus the new state). The trie masks every code that does not
continue a real id, so each finished beam is an id that exists (TIGER without its invalid ids; SID-LLM's STATIC).
`unconstrained=True` skips the mask (arm A4: the invalid-id rate).
"""
from __future__ import annotations

import numpy as np
import torch

NEG = -1e9


def trie_mask(trie, prefixes: np.ndarray, k: int) -> torch.Tensor:
    m = torch.full((len(prefixes), k), NEG)
    for i, p in enumerate(prefixes):
        a = trie.allowed(tuple(p))
        if len(a):
            m[i, torch.as_tensor(a)] = 0.0
    return m


def beam_search(step, state, batch: int, depth: int, ks, width: int, trie=None, unconstrained: bool = False):
    """Returns (codes [B, W, depth] int64, scores [B, W]); a beam with score <= NEG/2 is empty (fewer real ids than
    the width)."""
    prefixes = torch.zeros(batch, 1, 0, dtype=torch.long)
    scores = torch.zeros(batch, 1)
    for l in range(depth):
        w = prefixes.shape[1]
        flat = prefixes.reshape(batch * w, l)
        logp, state = step(state, flat, l)
        logp = logp.float().cpu().reshape(batch * w, -1)[:, :ks[l]]
        if trie is not None and not unconstrained:
            logp = logp + trie_mask(trie, flat.numpy(), ks[l])
        cand = (scores.reshape(batch * w, 1) + logp).reshape(batch, w * ks[l])
        top = cand.topk(min(width, cand.shape[1]), dim=1)
        parent, code = top.indices // ks[l], top.indices % ks[l]
        prefixes = torch.cat([torch.gather(prefixes, 1, parent[..., None].expand(-1, -1, l)), code[..., None]], 2)
        scores = top.values
        state = _reorder(state, parent, w)
    return prefixes, scores


def _reorder(state, parent, w):
    """Reorders a per-beam state [B*w, ...] by the chosen parents; states that are None or per-query pass through."""
    if state is None or not isinstance(state, torch.Tensor) or state.shape[0] != parent.shape[0] * w:
        return state
    b, nw = parent.shape
    idx = (parent + torch.arange(b)[:, None] * w).reshape(-1).to(state.device)
    return state.index_select(0, idx)
