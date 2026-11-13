"""Fast tests for the researcher's tokenisers (CPU, seconds). Development only: no score here."""
import numpy as np
import torch

from genret.kmeans import kmeans
from genret.rqkmeans import RQKMeans
from genret.rqvae import train_rqvae
from genret.sids import SidTrie, collision_stats, dedup_suffix, sid_str


def test_dedup_suffix_makes_sids_unique_and_is_deterministic():
    codes = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3]])
    ids = ["c", "a", "x", "b"]
    out = dedup_suffix(codes, ids)
    assert out.shape == (4, 4)
    assert len({tuple(r) for r in out}) == 4
    assert [int(out[i, 3]) for i in range(4)] == [2, 0, 0, 1]      # a, b, c in id order
    assert (dedup_suffix(codes, ids) == out).all()


def test_collision_stats_match_the_referee_definitions():
    codes = np.array([[1, 2], [1, 2], [3, 4], [5, 6]])
    s = collision_stats(codes)["full"]
    assert s["n_sids"] == 3 and s["max_products_per_sid"] == 2
    assert s["product_collision_rate"] == 0.5 and abs(s["sid_collision_share"] - 1 / 3) < 1e-9


def test_trie_allows_only_real_prefixes():
    codes = np.array([[1, 2, 0], [1, 3, 0], [4, 2, 0]])
    t = SidTrie(codes, ["a", "b", "c"])
    assert list(t.allowed(())) == [1, 4]
    assert list(t.allowed((1,))) == [2, 3]
    assert list(t.allowed((4, 3))) == []
    assert t.items((1, 3, 0)) == ["b"] and (4, 2, 0) in t and (4, 3, 0) not in t
    assert sid_str((1, 3, 0)) == "1-3-0"


def _blobs(n=600, d=16, k=6, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 5
    return (centres[rng.integers(0, k, n)] + rng.normal(size=(n, d)) * 0.3).astype(np.float32)


def test_kmeans_recovers_blobs():
    x = torch.as_tensor(_blobs())
    c, a = kmeans(x, 6, iters=15)
    assert len(torch.unique(a)) == 6
    inertia = ((x - c[a]) ** 2).sum(1).mean()
    assert inertia < 0.3 ** 2 * 16 * 1.5


def test_rqkmeans_residual_levels_reduce_error():
    x = _blobs(n=800, d=16, k=8)
    q = RQKMeans(levels=3, k=16, iters=10).fit(x)
    codes = q.encode(x)
    assert codes.shape == (800, 3)
    errs = []
    for l in range(1, 4):
        rec = sum(q.codebooks[j][torch.as_tensor(codes[:, j])] for j in range(l)).numpy()
        errs.append(((x - rec) ** 2).sum(1).mean())
    assert errs[0] > errs[1] > errs[2]


def test_rqvae_trains_and_encodes():
    x = _blobs(n=1000, d=32, k=8)
    m, hist = train_rqvae(x, steps=60, batch=128, k=16, levels=3, log_every=30, log=lambda *_: None)
    codes = m.encode(torch.as_tensor(x)).numpy()
    assert codes.shape == (1000, 3) and codes.max() < 16
    assert hist[-1]["rec"] < hist[0]["rec"] * 1.5


def test_constrained_beam_returns_only_real_ids_in_score_order():
    from genret.beam import NEG, beam_search
    codes = np.array([[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 2, 1]])
    trie = SidTrie(codes, ["a", "b", "c", "d"])
    torch.manual_seed(0)
    table = torch.randn(3, 4).log_softmax(1)          # a "model" that ignores the prefix

    def step(state, prefixes, l):
        return table[l].expand(prefixes.shape[0], -1), state

    out, sc = beam_search(step, None, batch=2, depth=3, ks=[4, 4, 4], width=8, trie=trie)
    real = {tuple(r) for r in codes}
    got = [tuple(r.tolist()) for r, s in zip(out[0], sc[0]) if s > NEG / 2]
    assert set(got) == real and len(got) == 4
    assert (sc[0][:-1] >= sc[0][1:]).all()
    free, _ = beam_search(step, None, batch=1, depth=3, ks=[4, 4, 4], width=8, unconstrained=True)
    assert any(tuple(r.tolist()) not in real for r in free[0])
