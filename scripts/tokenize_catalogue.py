"""Give every catalogue product a semantic id, and write the sid map the referee reads, plus statistics.

    python scripts/tokenize_catalogue.py --dataset esci-sample --encoder sentence-transformers/all-MiniLM-L6-v2 \
        --scheme rqvae --k 256 --levels 3 --out results/dev/tok-rqvae

Schemes:
- rqvae    (GR1, TIGER): RQ-VAE, `--levels` codes of `--k`, then TIGER's disambiguation code (sid_levels = levels);
- rqkmeans (GR2, SID-LLM): residual k-means, `--levels` codes of `--k`, no disambiguation code (`--dedup` adds
  TIGER's, for GR1's arm A5: RQ-KMeans against RQ-VAE under the same model).

Every number this prints on esci-sample, or with a development encoder, is development only, never a claim.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from genret import data, embed, text  # noqa: E402
from genret.rqkmeans import RQKMeans  # noqa: E402
from genret.rqvae import train_rqvae  # noqa: E402
from genret.sids import collision_stats, dedup_suffix, write_sid_map  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esci-sample")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--scheme", choices=("rqvae", "rqkmeans"), default="rqvae")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000, help="RQ-VAE steps")
    ap.add_argument("--iters", type=int, default=20, help="k-means iterations per level")
    ap.add_argument("--dead-code-reset", action="store_true")
    ap.add_argument("--optimizer", choices=("adagrad", "adamw"), default="adagrad")
    ap.add_argument("--lr", type=float, default=0.4, help="RQ-VAE lr (TIGER: Adagrad 0.4)")
    ap.add_argument("--standardize", action="store_true", help="RQ-VAE: per-dimension standardised inputs")
    ap.add_argument("--nondeterministic-kmeans", action="store_true",
                    help="the pre-2026-11-14 index_add_ update (reproduces nothing; for comparison only)")
    ap.add_argument("--dedup", action="store_true", help="rqkmeans: append TIGER's 4th code (GR1 arm A5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128, help="encoder batch")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", default="results/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="development only, never a claim")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.nondeterministic_kmeans:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS (embeddings, k-means)
        torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731

    ds = data.load(a.dataset)
    cat = ds.catalogue
    ids = cat["product_id"].astype(str).tolist()
    texts = text.catalogue_texts(cat)
    log(f"{a.dataset}: {len(ids)} products")
    emb = embed.cached_embeddings(texts, ids, a.encoder, a.cache, batch_size=a.batch_size, device=a.device)
    t_emb = time.time() - t0
    log(f"embeddings {emb.shape} ({a.encoder})")

    torch.manual_seed(a.seed)
    hist = []
    if a.scheme == "rqvae":
        model, hist = train_rqvae(emb, steps=a.steps, k=a.k, levels=a.levels, seed=a.seed, device=a.device,
                                  dead_code_reset=a.dead_code_reset, log=log,
                                  optimizer=a.optimizer, lr=a.lr, standardize=a.standardize)
        codes = model.encode(torch.as_tensor(emb, device=a.device)).cpu().numpy()
        with torch.no_grad():
            X = torch.as_tensor(emb, device=a.device)
            rec = float(sum(torch.nn.functional.mse_loss(model(X[s:s + 65536])[0], model.norm(X[s:s + 65536]),
                                                         reduction="sum") for s in range(0, len(X), 65536))
                        / X.numel())
        torch.save(model.state_dict(), out / "rqvae.pt")
        full = dedup_suffix(codes, ids)
    else:
        q = RQKMeans(levels=a.levels, k=a.k, iters=a.iters, seed=a.seed, device=a.device,
                     deterministic=not a.nondeterministic_kmeans).fit(emb)
        codes = q.encode(emb)
        rec = float(((emb - q.reconstruct(codes)) ** 2).mean())
        torch.save(q.state_dict(), out / "rqkmeans.pt")
        full = dedup_suffix(codes, ids) if a.dedup else codes
    t_tok = time.time() - t0 - t_emb
    stats = collision_stats(full, levels=a.levels)
    write_sid_map(out / "sid_map.tsv.gz", ids, full)
    import hashlib
    sid_sha = hashlib.sha256((out / "sid_map.tsv.gz").read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=Path(__file__).resolve().parent).strip()
    except Exception:
        commit = None
    report = {"label": a.label, "dataset": a.dataset, "encoder": a.encoder, "scheme": a.scheme, "k": a.k,
              "levels": a.levels, "sid_levels": a.levels, "sid_length": int(full.shape[1]),
              "max_suffix": int(full[:, -1].max()) if full.shape[1] > a.levels else None,
              "steps": a.steps if a.scheme == "rqvae" else None,
              "rqvae": {"optimizer": a.optimizer, "lr": a.lr, "standardize": a.standardize,
                        "dead_code_reset": a.dead_code_reset} if a.scheme == "rqvae" else None, "iters": a.iters if a.scheme == "rqkmeans" else None,
              "seed": a.seed, "device": a.device, "deterministic_kmeans": not a.nondeterministic_kmeans, "n_products": len(ids), "emb_dim": int(emb.shape[1]),
              "reconstruction_mse_per_dim": rec, "collisions": stats, "sid_map_sha256": sid_sha, "rqvae_history": hist,
              "seconds": {"embed": round(t_emb, 1), "tokenise": round(t_tok, 1)}, "commit": commit,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "tokenise.json").write_text(json.dumps(report, indent=1) + "\n")
    log(json.dumps({k: report[k] for k in ("scheme", "k", "levels", "max_suffix", "reconstruction_mse_per_dim")}))
    log(json.dumps(stats))


if __name__ == "__main__":
    main()
