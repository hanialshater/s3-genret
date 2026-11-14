"""Residual k-means semantic ids (SID-LLM, arXiv:2611.08804, section 3: RQ-KMeans, 3 levels x 8192).

Level l fits k-means on the residuals left by levels < l and assigns each item its nearest centroid (greedy, no
balancing, as the paper's final tokenizer). No disambiguation token: collisions are left to the ranker.
"""
from __future__ import annotations

import numpy as np
import torch

from genret.kmeans import _assign, kmeans


class RQKMeans:
    def __init__(self, levels: int = 3, k=8192, iters: int = 20, seed: int = 0, device: str = "cpu",
                 deterministic: bool = True):
        self.levels = levels
        self.ks = list(k) if isinstance(k, (list, tuple)) else [k] * levels
        self.iters, self.seed, self.device, self.deterministic = iters, seed, device, deterministic
        self.codebooks = []

    def fit(self, x: np.ndarray):
        r = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.codebooks = []
        for l, k in enumerate(self.ks):
            c, a = kmeans(r, k, iters=self.iters, seed=self.seed + l, deterministic=self.deterministic)
            self.codebooks.append(c)
            r = r - c[a]
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        r = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        codes = []
        for c in self.codebooks:
            a = _assign(r, c)
            codes.append(a)
            r = r - c[a]
        return torch.stack(codes, 1).cpu().numpy()

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        out = 0
        for l, c in enumerate(self.codebooks):
            out = out + c[torch.as_tensor(codes[:, l], device=self.device)]
        return out.cpu().numpy()

    def state_dict(self):
        return {"levels": self.levels, "ks": self.ks, "codebooks": [c.cpu() for c in self.codebooks]}
