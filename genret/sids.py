"""Semantic-id tables: TIGER's disambiguation suffix, collision statistics, the sid map file, and a prefix trie for
constrained decoding.

A sid is its codes joined by "-" (the referee's run-file contract, referee/runfile.py on steward/referee).
"""
from __future__ import annotations

import gzip
from collections import Counter, defaultdict

import numpy as np


def sid_str(codes) -> str:
    return "-".join(str(int(c)) for c in codes)


def dedup_suffix(codes: np.ndarray, ids) -> np.ndarray:
    """TIGER: append a code 0, 1, 2, ... to items that share all their semantic codes (0 when unique), in the
    order of the product id, so the result is deterministic. Returns codes with one more column."""
    order = np.lexsort([np.asarray(ids, dtype=object).astype(str)] + [codes[:, j] for j in range(codes.shape[1] - 1, -1, -1)])
    suffix = np.zeros(len(ids), dtype=np.int64)
    prev, run = None, 0
    for i in order:
        key = codes[i].tobytes()
        run = run + 1 if key == prev else 0
        suffix[i] = run
        prev = key
    return np.concatenate([codes, suffix[:, None]], 1)


def collision_stats(codes: np.ndarray, levels=None) -> dict:
    """The referee's definitions (referee/score.py sid_map_stats), for our own development numbers."""
    def one(c):
        keys = [r.tobytes() for r in c]
        groups = Counter(keys)
        shared = sum(v for v in groups.values() if v > 1)
        return {"n_sids": len(groups), "product_collision_rate": shared / len(keys),
                "sid_collision_share": sum(1 for v in groups.values() if v > 1) / len(groups),
                "max_products_per_sid": max(groups.values()),
                "unique_product_share": sum(1 for v in groups.values() if v == 1) / len(keys)}
    c = np.ascontiguousarray(codes)
    out = {"full": one(c)}
    if levels and levels < codes.shape[1]:
        out["semantic_prefix"] = one(np.ascontiguousarray(c[:, :levels]))
    out["per_level_codes_used"] = [int(len(np.unique(c[:, l]))) for l in range(c.shape[1])]
    for l in range(1, c.shape[1]):
        out.setdefault("prefix_distinct", []).append(int(len({r.tobytes() for r in np.ascontiguousarray(c[:, :l])})))
    return out


def write_sid_map(path, ids, codes) -> None:
    lines = ["product_id\tsid"] + [f"{p}\t{sid_str(c)}" for p, c in zip(ids, codes)]
    data = ("\n".join(lines) + "\n").encode()
    if str(path).endswith(".gz"):
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(data)
    else:
        open(path, "wb").write(data)


class SidTrie:
    """Prefix -> allowed next codes, over the sids that exist, for constrained beam search.

    `allowed(prefix)` takes a tuple of codes already generated and returns the codes that keep it a prefix of some
    real sid (an empty array at a complete sid). `items(sid)` gives the products of a complete sid.
    """

    def __init__(self, codes: np.ndarray, ids):
        self.depth = codes.shape[1]
        nxt = defaultdict(set)
        self.by_sid = defaultdict(list)
        for p, c in zip(ids, codes):
            t = tuple(int(v) for v in c)
            for l in range(self.depth):
                nxt[t[:l]].add(t[l])
            self.by_sid[t].append(p)
        self._next = {k: np.array(sorted(v), dtype=np.int64) for k, v in nxt.items()}

    def allowed(self, prefix) -> np.ndarray:
        return self._next.get(tuple(int(v) for v in prefix), np.empty(0, dtype=np.int64))

    def items(self, sid) -> list:
        return self.by_sid.get(tuple(int(v) for v in sid), [])

    def __contains__(self, sid) -> bool:
        return tuple(int(v) for v in sid) in self.by_sid
