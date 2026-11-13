"""The scorer: a run file against the test judgments, with the search floor's metrics at the pin.

Quality (s3-search `referee/metrics.py`, imported at the pinned commit, not copied):
- nDCG@10 with the ESCI gains E 1.0, S 0.1, C 0.01, I 0.0; an unjudged product counts 0; the ideal DCG is over the
  query's judged products; averaged over queries with a positive ideal.
- Exact recall@k (k 10, 100, 1000), averaged over queries with at least one Exact product.
- A judged query with no line in the run scores 0 (an empty list), so leaving a query out never raises a mean.

Generative-retrieval checks (this floor's own):
- **Invalid share**: the share of entries in the scored lists that are not a product of the catalogue: `null`
  (undecodable), a sid not in the sid map, a product id not in the catalogue, a sid with a product the map does not
  give it (inconsistent), or a sid shared by several products with no product chosen (ambiguous). An invalid entry
  keeps its rank with gain 0: the slot was spent.
- **Duplicates**: a product already listed above in the same list keeps its rank with gain 0, so repeating a good
  product cannot add gain. Reported apart from invalid entries.
- **Collision rate** (from the sid map, over the catalogue): the share of products whose sid is shared with another
  product, on the full sid and on its semantic prefix (`sid_levels`); and the share of valid generated entries that
  landed on a shared sid.
- **Short lists**: the share of scored queries with fewer than k valid, distinct results, for k 10, 100 and 1000.
- **Coverage**: the share of the catalogue the sid map gives a sid. A product with no sid cannot be generated; a map
  that leaves products out is refused as a score of record (it would shrink the search space).

Owner: evaluation-steward. Makers do not edit this folder (P6).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from referee import upstream
from referee.runfile import Run, RunError

KS = (10, 100, 1000)
INVALID_KINDS = ("undecodable", "unknown_sid", "unknown_product", "inconsistent", "ambiguous")


def _prefix(sid: str, levels):
    return sid if not levels else "-".join(sid.split("-")[:levels])


def sid_map_stats(sid_map: dict, catalogue_ids: set, sid_levels=None) -> dict:
    """Coverage and collisions of a sid map over the catalogue."""
    covered = [p for p in sid_map if p in catalogue_ids]
    out = {"n_catalogue": len(catalogue_ids), "n_mapped": len(covered),
           "coverage": (len(covered) / len(catalogue_ids)) if catalogue_ids else None,
           "n_foreign": len(sid_map) - len(covered), "sid_levels": sid_levels}

    def collisions(key):
        groups = Counter(key(sid_map[p]) for p in covered)
        shared = sum(c for c in groups.values() if c > 1)
        return {"n_sids": len(groups),
                "product_collision_rate": (shared / len(covered)) if covered else None,
                "sid_collision_share": (sum(1 for c in groups.values() if c > 1) / len(groups)) if groups else None,
                "max_products_per_sid": max(groups.values()) if groups else 0}

    out["full_sid"] = collisions(lambda s: s)
    if sid_levels:
        out["semantic_prefix"] = collisions(lambda s: _prefix(s, sid_levels))
    lengths = Counter(len(sid_map[p].split("-")) for p in covered)
    out["codes_per_sid"] = {str(k): v for k, v in sorted(lengths.items())}
    width = max(lengths) if lengths else 0
    per_level = [set() for _ in range(width)]
    for p in covered:
        for i, c in enumerate(sid_map[p].split("-")):
            per_level[i].add(c)
    out["distinct_codes_per_level"] = [len(s) for s in per_level]
    return out


class Resolved:
    """A run resolved against the catalogue and the sid map: what each slot holds, and the per-query counts."""

    def __init__(self, run: Run, catalogue_ids: set, sid_map=None, expected_qids=None):
        if expected_qids is not None:
            extra = set(run.ranked) - set(expected_qids)
            if extra:
                raise RunError(f"{len(extra)} qids are not test queries of this dataset (for example "
                               f"{sorted(extra)[:3]}); a run answers the split's test queries only")
        by_sid: dict = {}
        if sid_map is not None:
            for p, s in sid_map.items():
                by_sid.setdefault(s, []).append(p)
        shared = {s for s, ps in by_sid.items() if sum(1 for p in ps if p in catalogue_ids) > 1}
        self.lists, self.counts = {}, {}
        for q, entries in run.ranked.items():
            out, seen = [], set()
            c = Counter()
            for i, e in enumerate(entries):
                c["entries"] += 1
                pid, kind, hit_shared = None, None, False
                if e is None:
                    kind = "undecodable"
                elif isinstance(e, str):
                    pid = e
                else:
                    if sid_map is None:
                        raise RunError("the run gives sids but no sid map was given (--sid-map)")
                    s = e["sid"]
                    if s not in by_sid:
                        kind = "unknown_sid"
                    elif e.get("pid") is not None:
                        pid = e["pid"]
                        if sid_map.get(pid) != s:
                            kind, pid = "inconsistent", None
                    else:
                        cands = [p for p in by_sid[s] if p in catalogue_ids]
                        if len(cands) == 1:
                            pid = cands[0]
                        elif len(cands) > 1:
                            kind = "ambiguous"
                        else:
                            kind = "unknown_product"
                    hit_shared = s in shared
                if kind is None and pid not in catalogue_ids:
                    kind, pid = "unknown_product", None
                if kind is None and hit_shared:
                    c["on_shared_sid"] += 1
                if kind is not None:
                    c[kind] += 1
                    out.append(f"\x00{kind}#{i}")      # an unjudged placeholder: keeps the rank, gain 0
                elif pid in seen:
                    c["duplicate"] += 1
                    out.append(f"\x00duplicate#{i}")
                else:
                    seen.add(pid)
                    c["valid"] += 1
                    out.append(pid)
            self.lists[q], self.counts[q] = out, c

    def per_query_ndcg(self, judgments: dict) -> dict:
        """{qid: nDCG@10} over the queries in `judgments` with a positive ideal (the ones the mean is over)."""
        per = upstream.metrics().score_run(self.lists, judgments, k_ndcg=10, recall_ks=())["per_query"]
        return {q: m["ndcg@10"] for q, m in per.items() if m["ndcg@10"] is not None}

    def summary(self, judgments: dict) -> dict:
        """Quality and the generative checks over the queries in `judgments`."""
        m = upstream.metrics()
        quality = m.score_run(self.lists, judgments, k_ndcg=10, recall_ks=KS)["mean"]
        tot = Counter()
        valid_per_q = []
        for q in judgments:
            c = self.counts.get(q, Counter())
            tot.update(c)
            valid_per_q.append(c["valid"])
        n_entries = tot["entries"]
        invalid = sum(tot[k] for k in INVALID_KINDS)
        n_valid_or_dup = tot["valid"] + tot["duplicate"]
        v = np.asarray(valid_per_q)
        return {
            "n_queries": len(judgments),
            "n_queries_missing": sum(1 for q in judgments if q not in self.lists),
            "quality": quality,
            "genret": {
                "n_entries": n_entries,
                "invalid_share": (invalid / n_entries) if n_entries else None,
                "invalid_by_kind": {k: tot[k] for k in INVALID_KINDS},
                "duplicate_share": (tot["duplicate"] / n_entries) if n_entries else None,
                "shared_sid_hit_share": (tot["on_shared_sid"] / n_valid_or_dup) if n_valid_or_dup else None,
                "short_list_share": {f"<{k}": (float(np.mean(v < k)) if len(v) else None) for k in KS},
                "mean_valid_per_query": float(v.mean()) if len(v) else None,
            },
        }


def score(run: Run, judgments: dict, catalogue_ids: set, sid_map=None, expected_qids=None, subsets=None) -> dict:
    """Scores one run on `judgments` and on each subset ({name: set of qids}) of them."""
    res = Resolved(run, catalogue_ids, sid_map, expected_qids)
    out = res.summary(judgments)
    out["run_k"] = run.header.get("k")
    if sid_map is not None:
        out["sid_map"] = sid_map_stats(sid_map, catalogue_ids, run.header.get("sid_levels"))
    if subsets:
        out["subsets"] = {name: res.summary({q: j for q, j in judgments.items() if q in ids})
                          for name, ids in subsets.items()}
    return out


def paired(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Run A minus run B in nDCG@10, query by query ({qid: nDCG@10}, the same queries): the mean difference and a 95%
    bootstrap interval over queries (seeded, so the same two runs always give the same interval)."""
    qs = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError("paired: the two runs were scored on different queries")
    d = np.array([a[q] - b[q] for q in qs], dtype=np.float64)
    if not len(d):
        return {"n_queries": 0, "mean_diff": None, "ci95": None}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"n_queries": len(d), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}
