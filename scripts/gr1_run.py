"""GR1: train the TIGER-style seq2seq retriever on a sid map, decode the test queries, write the referee's run file.

    python scripts/gr1_run.py --dataset esci-sample --sids results/dev/minilm-rqvae-256 --steps 50 \
        --max-queries 20 --out results/dev/gr1-smoke

Arms (lab/predictions/genret-GR1.md): A1 default; A2 --no-indexing; A3 --random-ids; A4 --unconstrained;
A5 a RQ-KMeans sid map. The scores are the referee's; this script writes only the run file and its own log.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from genret import data, gr1  # noqa: E402
from genret.sids import SidTrie, write_sid_map  # noqa: E402


def read_codes(sid_dir: Path, ids):
    rows = gzip.open(sid_dir / "sid_map.tsv.gz", "rt").read().splitlines()[1:]
    m = {p: list(map(int, s.split("-"))) for p, s in (r.split("\t") for r in rows)}
    return np.array([m[p] for p in ids], dtype=np.int64)


def _commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


def random_ids(codes: np.ndarray, seed: int) -> np.ndarray:
    """A3: the same shape and the same per-level code ranges, assigned at random, then re-separated."""
    rng = np.random.default_rng(seed)
    sem = np.stack([rng.integers(0, codes[:, l].max() + 1, len(codes)) for l in range(codes.shape[1] - 1)], 1)
    return sem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esci-sample")
    ap.add_argument("--sids", required=True, help="a tokenise output directory (sid_map.tsv.gz, tokenise.json)")
    ap.add_argument("--model", default="google-t5/t5-small")
    ap.add_argument("--steps", type=int, default=17000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--no-indexing", action="store_true")
    ap.add_argument("--index-epochs", type=int, default=2, help="epochs that include the indexing task")
    ap.add_argument("--random-ids", action="store_true")
    ap.add_argument("--unconstrained", action="store_true")
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--decode-batch", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=0, help="development: decode only the first n test queries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="gr1")
    ap.add_argument("--label", default="development only, never a claim")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731

    ds = data.load(a.dataset)
    ids = ds.catalogue["product_id"].astype(str).tolist()
    tokinfo = json.loads((Path(a.sids) / "tokenise.json").read_text())
    levels = tokinfo["sid_levels"]
    codes = read_codes(Path(a.sids), ids)
    if a.random_ids:
        from genret.sids import dedup_suffix
        codes = dedup_suffix(random_ids(codes, a.seed), ids)
        write_sid_map(out / "sid_map.tsv.gz", ids, codes)
    ks = [int(codes[:, l].max()) + 1 for l in range(codes.shape[1])]
    ks[:levels] = [max(ks[l], tokinfo["k"]) for l in range(levels)]
    trie = SidTrie(codes, ids)
    log(f"{a.dataset}: {len(ids)} products, sid shape {codes.shape}, ks {ks}")

    tok, model, vocab = gr1.make_model(a.model, ks)
    fit = ds.fit
    hist = gr1.train(model, tok, vocab,
                     lambda e: gr1.build_examples(fit, ds.catalogue, codes, ids, e,
                                                  indexing=(not a.no_indexing) and e < a.index_epochs, seed=a.seed),
                     steps=a.steps, batch=a.batch, lr=a.lr, warmup=a.warmup, device=a.device, log=log)
    t_train = time.time() - t0
    tq = ds.test_queries
    if a.max_queries:
        tq = tq.head(a.max_queries)
    res = gr1.retrieve(model, tok, vocab, tq["query"].tolist(), trie, width=a.width, batch=a.decode_batch,
                       device=a.device, unconstrained=a.unconstrained)
    t_dec = time.time() - t0 - t_train
    ranked = {int(q): gr1.to_ranked(r, trie, a.k) for q, r in zip(tq["query_id"], res)}
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from referee import runfile                                         # the steward's writer and format
    header = {"name": a.name, "dataset": a.dataset, "split": "test", "k": a.k, "sid_levels": levels,
              "trained_on": ["catalogue", "fit"],
              "system": f"GR1 {a.model} on {a.sids}; indexing={not a.no_indexing}; random_ids={a.random_ids}; "
                        f"constrained={not a.unconstrained}; beam {a.width}; commit {_commit()}; {a.label}"}
    runfile.write_run(out / "run.jsonl.gz", header, ranked)
    invalid = sum(e is None for r in ranked.values() for e in r[:10]) / max(1, sum(len(r[:10]) for r in ranked.values()))
    report = {"label": a.label, "args": vars(a), "ks": ks, "train_history": hist, "n_queries": len(ranked),
              "own_top10_undecodable_share": invalid,
              "seconds": {"train": round(t_train, 1), "decode": round(t_dec, 1),
                          "decode_per_query_ms": round(1000 * t_dec / max(1, len(ranked)), 1)},
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "gr1.json").write_text(json.dumps(report, indent=1) + "\n")
    log(json.dumps({k: report[k] for k in ("n_queries", "own_top10_undecodable_share", "seconds")}))
    if a.device.startswith("cuda"):
        torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
