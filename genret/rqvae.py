"""RQ-VAE semantic ids as in TIGER (Rajput et al., NeurIPS 2023, section 3.1 and appendix).

Paper settings: encoder MLP 768 -> 512 -> 256 -> 128 -> 32 (ReLU), 3 residual levels of 256 codes of dim 32,
commitment beta 0.25, Adagrad lr 0.4, batch 1024, codebooks initialised by k-means on the first batch, a 4th code
appended to separate items that share the 3 codes (genret.sids.dedup_suffix).

Deviations, stated: the decoder mirrors the encoder (the paper does not give it); training is by steps, not the
paper's 20k epochs; an optional dead-code reset (off by default, as in the paper) re-seeds unused codes from the batch.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genret.kmeans import kmeans


def mlp(dims):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class RQVAE(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 256, 128), latent=32, levels=3, k=256, beta=0.25):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim])
        self.levels, self.k, self.beta = levels, k, beta
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(k, latent) * 0.01) for _ in range(levels)])
        self.initialised = False
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def norm(self, x):
        return (x - self.mu) / self.sd

    @torch.no_grad()
    def kmeans_init(self, x):
        r = self.encoder(self.norm(x))
        for l in range(self.levels):
            c, a = kmeans(r, self.k, iters=10, seed=l)
            self.codebooks[l].data.copy_(c)
            r = r - c[a]
        self.initialised = True

    def quantize(self, z):
        r, zq, codes, closs = z, torch.zeros_like(z), [], 0.0
        for cb in self.codebooks:
            d = (r.pow(2).sum(1, keepdim=True) - 2 * r @ cb.T + cb.pow(2).sum(1)[None])
            a = d.argmin(1)
            e = cb[a]
            closs = closs + F.mse_loss(e, r.detach()) + self.beta * F.mse_loss(r, e.detach())
            codes.append(a)
            zq = zq + e
            r = r - e.detach()
        return zq, torch.stack(codes, 1), closs

    def forward(self, x):
        x = self.norm(x)
        z = self.encoder(x)
        zq, codes, closs = self.quantize(z)
        zq_st = z + (zq - z).detach()                # straight-through
        xh = self.decoder(zq_st)
        rec = F.mse_loss(xh, x)                      # in the (possibly standardised) input space
        return xh, codes, rec, closs

    @torch.no_grad()
    def encode(self, x, batch=65536):
        out = []
        for s in range(0, x.shape[0], batch):
            out.append(self.quantize(self.encoder(self.norm(x[s:s + batch])))[1])
        return torch.cat(out)


def train_rqvae(x: np.ndarray, steps: int = 3000, batch: int = 1024, lr: float = 0.4, k: int = 256, levels: int = 3,
                seed: int = 0, device: str = "cpu", dead_code_reset: bool = False, log_every: int = 500, log=print,
                optimizer: str = "adagrad", standardize: bool = False):
    """standardize: scale inputs to zero mean, unit variance per dimension (unit-norm sentence embeddings have a
    per-dimension variance near 1/768, which makes the reconstruction term tiny next to the commitment terms)."""
    torch.manual_seed(seed)
    X = torch.as_tensor(x, dtype=torch.float32, device=device)

    m = RQVAE(in_dim=X.shape[1], levels=levels, k=k).to(device)
    if standardize:
        m.mu.copy_(X.mean(0))
        m.sd.copy_(X.std(0).clamp(min=1e-6))
    g = torch.Generator(device="cpu").manual_seed(seed)
    first = X[torch.randperm(X.shape[0], generator=g)[:max(batch, 4 * k)].to(device)]
    m.kmeans_init(first)
    opt = (torch.optim.Adagrad(m.parameters(), lr=lr) if optimizer == "adagrad"
           else torch.optim.AdamW(m.parameters(), lr=lr))
    hist = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, X.shape[0], (batch,), generator=g).to(device)
        xb = X[idx]
        _, codes, rec, closs = m(xb)
        loss = rec + closs
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dead_code_reset and step % 200 == 0:
            with torch.no_grad():
                z = m.encoder(m.norm(xb))
                r = z
                for l, cb in enumerate(m.codebooks):
                    used = torch.bincount(codes[:, l], minlength=k) > 0
                    dead = (~used).nonzero().flatten()
                    if len(dead):
                        cb.data[dead] = r[torch.randint(0, r.shape[0], (len(dead),), generator=g).to(device)]
                    r = r - cb[codes[:, l]]
        if step % log_every == 0 or step == steps:
            use = [int(torch.unique(codes[:, l]).numel()) for l in range(levels)]
            hist.append({"step": step, "rec": float(rec), "commit": float(closs), "batch_codes_used": use})
            log(f"rqvae step {step} rec {float(rec):.5f} commit {float(closs):.5f} batch codes used {use}")
    m.eval()
    return m, hist
