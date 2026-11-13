"""GR2: multi-token prediction of a semantic id (SID-LLM, arXiv:2611.08804, section on MTP).

One prefill of the request; three heads on the final hidden state give the three codeword distributions, each
conditioned on the request only (not on the other codes). A beam walk over the trie of existing ids, with the summed
per-level log-probabilities, assembles up to W candidates. Head architecture is not given in the paper; here each
head is LayerNorm + Linear(d, K).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from genret.beam import beam_search


class MTPHeads(nn.Module):
    def __init__(self, d: int, ks):
        super().__init__()
        self.ks = list(ks)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, k)) for k in self.ks])

    def forward(self, h):                       # h [B, d] -> list of [B, K_l]
        return [head(h) for head in self.heads]

    def loss(self, h, codes):                   # codes [B, levels]
        return sum(F.cross_entropy(lo.float(), codes[:, l]) for l, lo in enumerate(self(h))) / len(self.ks)


def last_hidden(lm, input_ids, attention_mask):
    """The backbone's final hidden state at each row's last real token (left- or right-padded)."""
    out = lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    h = out.hidden_states[-1]
    last = attention_mask.long().cumsum(1).argmax(1)
    return h[torch.arange(h.shape[0], device=h.device), last]


@torch.no_grad()
def mtp_decode(heads: MTPHeads, h, trie, width: int = 250):
    """h [B, d] -> (codes [B, W, levels], scores [B, W]) over ids that exist."""
    logps = [lo.float().log_softmax(-1).cpu() for lo in heads(h)]
    B = h.shape[0]

    def step(state, prefixes, l):
        w = prefixes.shape[0] // B
        return logps[l].repeat_interleave(w, 0), state

    return beam_search(step, None, B, len(heads.ks), heads.ks, width, trie=trie)
