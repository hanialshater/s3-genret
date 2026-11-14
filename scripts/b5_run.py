"""GR2 arm B5, the control: the referee's bm25f-tuned candidates (top 250) re-ranked by the same kind of ranker as
B1 (MiniLM cross-encoder fine-tuned on the fit judgments, genret.ranker, same seed and settings). No generation.

    python scripts/b5_run.py --dataset esci-us --candidates bm25f-tuned.top250.jsonl.gz --out /content/out/gr2-b5

The served order is 0.7 x ranker + 0.3 x the first stage's score, each min-max normalised per query, as B1. The
baseline run file carries ranks, not BM25 scores, so the first stage's score is minus the rank. Development only until
the founder's `referee full`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from genret import data, ranker as rk, text  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esci-us")
    ap.add_argument("--candidates", required=True, help="a run file of product ids (the bm25f-tuned baseline)")
    ap.add_argument("--depth", type=int, default=250)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--w-ranker", type=float, default=0.7)
    ap.add_argument("--ranker-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="development until the held-out draw")
    a = ap.parse_args(argv)
    from referee import runfile
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731
    ds = data.load(a.dataset)
    cat, fit = ds.catalogue, ds.fit
    ids = cat["product_id"].astype(str).tolist()
    ptext = {p: t for p, t in zip(ids, text.catalogue_texts(cat))}
    base = runfile.read_run(a.candidates)
    qtext = {int(q): t for q, t in zip(ds.test_queries["query_id"], ds.test_queries["query"])}
    log(f"candidates: {len(base.ranked)} queries from {a.candidates} ({base.header.get('name')})")
    tr, tu = rk.fit_examples(fit, ptext, seed=a.seed)
    log(f"ranker: {len(tr)} training pairs, {len(tu)} tuning pairs (fit queries)")
    m = rk.train_ranker(tr, epochs=a.ranker_epochs, device=a.device)
    ranked, t1 = {}, time.time()
    for q, lst in base.ranked.items():
        c = [p for p in lst if isinstance(p, str) and p in ptext][:a.depth]
        if not c or q not in qtext:
            ranked[q] = []
            continue
        rs = m.predict([(qtext[q], ptext[p]) for p in c], batch_size=256, show_progress_bar=False)
        sc = rk.combine(rs, [-i for i in range(len(c))], a.w_ranker)
        ranked[q] = [c[i] for i in np.argsort(-sc, kind="stable")[:a.k]]
    rerank_ms = round(1000 * (time.time() - t1) / max(1, len(ranked)), 2)
    header = {"name": "gr2-b5-us", "dataset": a.dataset, "split": "test", "k": a.k, "trained_on": ["catalogue", "fit"],
              "system": f"B5: bm25f-tuned top {a.depth} + MiniLM ranker (fit judgments), w={a.w_ranker}; {a.label}"}
    runfile.write_run(out / "run-b5-bm25-ranker.jsonl.gz", header, ranked)
    (out / "b5.json").write_text(json.dumps({"args": vars(a), "rerank_per_query_ms": rerank_ms,
                                             "seconds_total": round(time.time() - t0, 1),
                                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    log(f"B5 written; re-ranking {rerank_ms} ms/query")


if __name__ == "__main__":
    main()
