"""Plain Lloyd k-means in torch (CPU or GPU), chunked so 1.2M x 8192 fits on an L4."""
from __future__ import annotations

import torch


def _assign(x: torch.Tensor, c: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    c2 = (c * c).sum(1)
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        d = c2[None, :] - 2.0 * xb @ c.T          # ||x||^2 is constant per row
        out[s:s + chunk] = d.argmin(1)
    return out


def kmeans(x: torch.Tensor, k: int, iters: int = 20, seed: int = 0, init: str = "kmeans++",
           sample_init: int = 200_000):
    """Returns (centroids [k, d], assignment [n]). Empty clusters are re-seeded from the farthest points."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    if k >= n:
        c = torch.cat([x, x[torch.randint(0, n, (k - n,), generator=g)].to(x.device)])[:k].clone()
        return c, torch.arange(n, device=x.device)
    if init == "kmeans++":
        c = _kmeanspp(x, k, g, sample_init)
    else:
        c = x[torch.randperm(n, generator=g)[:k].to(x.device)].clone()
    a = None
    for _ in range(iters):
        a = _assign(x, c)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).to(x.dtype)
        empty = cnt == 0
        c = torch.where(empty[:, None], c, sums / cnt.clamp(min=1)[:, None])
        if empty.any():
            d = ((x - c[a]) ** 2).sum(1)
            far = d.topk(int(empty.sum())).indices
            c[empty] = x[far]
    a = _assign(x, c)
    return c, a


def _kmeanspp(x, k, g, sample_init):
    """k-means++ on a random subsample (k-means|| would be faster; this is enough at our sizes)."""
    n = x.shape[0]
    idx = torch.randperm(n, generator=g)[:min(n, max(sample_init, 4 * k))].to(x.device)
    xs = x[idx]
    m = xs.shape[0]
    first = int(torch.randint(0, m, (1,), generator=g))
    c = [xs[first]]
    d = ((xs - xs[first]) ** 2).sum(1)
    # batched k-means++: draw several centres per round from the D^2 distribution (greedy enough for init)
    per = max(1, k // 64)
    while len(c) < k:
        p = (d / d.sum()).cpu()
        draw = torch.multinomial(p, min(per, k - len(c)), replacement=False, generator=g).to(x.device)
        new = xs[draw]
        c.extend(list(new))
        dn = torch.cdist(xs, new).pow(2).min(1).values
        d = torch.minimum(d, dn)
    return torch.stack(c[:k]).clone()
