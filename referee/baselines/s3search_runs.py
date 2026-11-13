"""The search floor's baselines on a dataset's test queries, run by s3-search's own code at the pinned commit.

Run by `python -m referee baselines` in a separate process whose working directory is the pinned s3-search checkout,
so `referee` here is s3-search's package and `engine` its engine. Do not run it by hand.

For each system it builds s3-search's engine index on the dataset's catalogue, runs every test query through
s3-search's own benchmark (`referee.bench.run`, which also scores the run with s3-search's metrics on the test
judgments), and writes the ranked product ids as a Floor 3 run file. The Floor 3 referee then scores the same file
itself; the two scores must agree (the shared-loader drift check of the floor's decision record).

Systems (the search floor's, unchanged):
- `bm25f`: BM25F over the five text fields with weight 1, k1 1.2, b 0.75 (s3-search's default request and mapping);
- `bm25f-tuned`: the prototype's tuned BM25F (`referee/requests/bm25f-tuned.json` and `bm25f-tuned.bm25.json`), the
  search floor's lexical bar (0.304 nDCG@10 on ESCI-US in the prototype);
- `dense-tfidf-svd-128`: exact dense search with the engine's encoder, TF-IDF reduced to 128 dimensions by SVD,
  fitted on the catalogue, over title | brand | color (the encoder s3-search's judges use), fitted with numpy.

Owner: evaluation-steward. Makers do not edit this folder (P6).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from referee import bench, loader  # s3-search's, from the working directory

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("genret_runfile", HERE.parent / "runfile.py")
runfile = importlib.util.module_from_spec(_spec)
sys.modules["genret_runfile"] = runfile
_spec.loader.exec_module(runfile)

ENC = {"embedding": {"model": "tfidf-svd-128", "inputs": ["product_title", "product_brand", "product_color"],
                     "fit": "numpy"}}
SYSTEMS = ("bm25f", "bm25f-tuned", "dense-tfidf-svd-128")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def system(name: str, ds, k: int):
    """(mapping, template, what) for one system: exactly what s3-search runs for it."""
    mapping = copy.deepcopy(ds.mapping)
    what = {"system": name}
    if name == "bm25f":
        template = bench.DEFAULT_TEMPLATE
    elif name == "bm25f-tuned":
        req, par = Path("referee/requests/bm25f-tuned.json"), Path("referee/requests/bm25f-tuned.bm25.json")
        template = json.loads(req.read_text())
        mapping["_meta"]["engine"]["bm25"] = json.loads(par.read_text())
        what["files"] = {str(req): _sha(req), str(par): _sha(par)}
    elif name == "dense-tfidf-svd-128":
        mapping["properties"]["embedding"] = {"type": "knn_vector", "dimension": 128}
        mapping["_meta"]["engine"]["embeddings"] = copy.deepcopy(ENC)
        template = {"query": {"neural": {"embedding": {"query_text": "{{query}}", "k": k}}}}
    else:
        raise ValueError(name)
    what["request_template"] = template
    what["mapping_meta_engine"] = mapping["_meta"]["engine"]
    return mapping, template, what


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("esci-sample", "esci-us"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import engine.api as api
    ds = loader.load_dataset(a.dataset)
    judg = loader._load_dataset_test_judgments(a.dataset)       # the scorer's read, inside the referee
    queries = ds.test_queries
    meta = {}
    for name in a.systems.split(","):
        mapping, template, what = system(name, ds, a.k)
        sds = dataclasses.replace(ds, mapping=mapping)
        runs: dict = {}
        t = time.perf_counter()
        res = bench.run(sds, queries, judg, api, device=a.device, template=template, k=a.k, batch=64,
                        latency_queries=20, split="test", engine="engine", runs_out=runs,
                        progress=lambda m, n=name: print(f"[{n}] {m}", file=sys.stderr, flush=True))
        path = out_dir / f"{name}.run.jsonl.gz"
        runfile.write_run(path, {"name": name, "dataset": a.dataset, "split": "test", "k": a.k,
                                 "system": f"s3-search engine, {name}, device {a.device}"},
                          {int(q): [str(p) for p in ids] for q, ids in runs.items()})
        meta[name] = {**what, "run_file": path.name, "run_sha256": _sha(path), "n_queries": len(runs),
                      "s3search_quality": res["quality"], "s3search_commit": res["commit"], "device": a.device,
                      "build_s": res["build_s"], "wall_s": time.perf_counter() - t,
                      "latency_ms_p50_batch1": res["latency_ms"]["p50"], "host": res["host"]}
        print(f"[{name}] nDCG@10 {res['quality']['ndcg@10']:.4f} (s3-search's own score) -> {path}",
              file=sys.stderr, flush=True)
    meta_path = out_dir / "s3search_meta.json"
    merged = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    merged.update(meta)
    meta_path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
