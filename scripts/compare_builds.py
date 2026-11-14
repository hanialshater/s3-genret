"""Compare two tokenisation builds byte for byte: the sid maps and the cached embeddings they came from.

    python scripts/compare_builds.py OUT_A CACHE_A OUT_B CACHE_B [--require-identical]
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    oa, ca, ob, cb = sys.argv[1:5]
    ea, eb = sorted(glob.glob(f"{ca}/emb-*.npy"))[-1], sorted(glob.glob(f"{cb}/emb-*.npy"))[-1]
    xa, xb = np.load(ea), np.load(eb)
    ma, mb = Path(oa) / "sid_map.tsv.gz", Path(ob) / "sid_map.tsv.gz"
    import gzip
    la, lb = gzip.open(ma, "rt").read().splitlines(), gzip.open(mb, "rt").read().splitlines()
    rep = {"embeddings_identical": bool(np.array_equal(xa, xb)), "embeddings_max_abs_diff": float(np.abs(xa - xb).max()),
           "sid_map_sha256": [sha(ma), sha(mb)], "sid_maps_identical": sha(ma) == sha(mb),
           "rows_agreeing": sum(x == y for x, y in zip(la[1:], lb[1:])) / max(1, len(la) - 1)}
    print(json.dumps(rep), flush=True)
    Path(ob).joinpath("compare.json").write_text(json.dumps(rep, indent=1) + "\n")
    if "--require-identical" in sys.argv and not rep["sid_maps_identical"]:
        sys.exit("sid maps differ")


if __name__ == "__main__":
    main()
