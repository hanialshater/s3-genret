"""The search floor's referee, at the pinned commit: the only way this referee reads ESCI.

Both floors must score the same queries and products with the same gains, so this referee does not load ESCI itself.
It runs `s3-search`'s own `referee/loader.py` and `referee/metrics.py` from a checkout of `s3-search` at the commit in
`referee/pin.json`, and refuses to run if the checkout is at another commit, has local changes, or a pinned file's
sha256 differs.

- Where the checkout is: `$GENRET_S3_SEARCH`, else `<this repo>/.cache/s3-search` (`python -m referee setup` clones it
  from GitHub; s3-search is public).
- Where the data are: `$GENRET_DATA`, else `<this repo>/data`. The raw ESCI files go in `esci/` and the sample in
  `esci-sample/`; they are passed to the pinned loader as `S3_SEARCH_DATA` and `S3_SEARCH_SAMPLE`.

The two modules are loaded under their own names (`s3search_loader`, `s3search_metrics`), because both repositories
call their package `referee`. The search floor's baselines run in a separate process inside the checkout
(`referee/baselines/s3search_runs.py`), so they import `s3-search`'s package as `s3-search` itself does.

Owner: evaluation-steward. Makers do not edit this folder (P6).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIN = json.loads((HERE / "pin.json").read_text())


class PinError(RuntimeError):
    """The s3-search checkout is not the pinned one."""


def checkout() -> Path:
    return Path(os.environ.get("GENRET_S3_SEARCH") or (REPO / ".cache" / "s3-search")).resolve()


def data_root() -> Path:
    return Path(os.environ.get("GENRET_DATA") or (REPO / "data")).resolve()


def data_env() -> dict:
    root = data_root()
    return {"S3_SEARCH_DATA": str(root / "esci"), "S3_SEARCH_SAMPLE": str(root / "esci-sample")}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=str(cwd or checkout()), capture_output=True, text=True)


def verify(root: Path = None) -> dict:
    """Raises PinError unless `root` is s3-search at the pinned commit with the pinned files unchanged."""
    root = Path(root or checkout())
    if not (root / "referee" / "loader.py").exists():
        raise PinError(f"no s3-search checkout at {root}; run `python -m referee setup`")
    head = _git("rev-parse", "HEAD", cwd=root).stdout.strip()
    if head != PIN["commit"]:
        raise PinError(f"s3-search at {root} is at {head or '?'}, the pin is {PIN['commit']}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=root).stdout.strip()
    if dirty:
        raise PinError(f"s3-search at {root} has local changes:\n{dirty}")
    for rel, want in PIN["files"].items():
        got = sha256_file(root / rel)
        if got != want:
            raise PinError(f"s3-search {rel}: sha256 {got}, pinned {want}")
    return {"repo": PIN["repo"], "commit": head, "checkout": str(root), "files_verified": sorted(PIN["files"])}


def setup(fetch_data: bool = True, build_sample: bool = True) -> dict:
    """Clone s3-search at the pin (if absent), fetch ESCI and build esci-sample through the pinned loader."""
    root = checkout()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "--quiet", PIN["repo"], str(root)])
        if r.returncode:
            raise PinError(f"git clone {PIN['repo']} failed")
    if _git("rev-parse", "HEAD", cwd=root).stdout.strip() != PIN["commit"]:
        _git("fetch", "--quiet", "origin", PIN["commit"], cwd=root)
        r = _git("checkout", "--quiet", PIN["commit"], cwd=root)
        if r.returncode:
            raise PinError(f"cannot check out {PIN['commit']} in {root}: {r.stderr.strip()}")
    out = {"pin": verify(root)}
    ld = loader()
    if fetch_data:
        ld.fetch(ld.DEFAULT_DIR)
    if build_sample:
        if not ld._sample_ready():
            ld.build_esci_sample()
        out["sample_manifest_ok"] = sample_manifest_ok()
    return out


_MODULES: dict = {}


def _load(name: str, rel: str):
    if name in _MODULES:
        return _MODULES[name]
    verify()
    for k, v in data_env().items():
        os.environ[k] = v  # the pinned loader reads these at import
    spec = importlib.util.spec_from_file_location(name, checkout() / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[name] = mod
    return mod


def loader():
    """s3-search's `referee/loader.py` at the pin."""
    return _load("s3search_loader", "referee/loader.py")


def metrics():
    """s3-search's `referee/metrics.py` at the pin (the gains and the nDCG and recall definitions)."""
    return _load("s3search_metrics", "referee/metrics.py")


def sample_manifest_ok() -> bool:
    """The local esci-sample has the pinned manifest's counts and content hashes (everyone develops on the same one)."""
    import pandas as pd
    ld = loader()
    d = ld.SAMPLE_DIR
    prod = pd.read_parquet(d / ld.PRODUCTS)
    ex = pd.read_parquet(d / ld.EXAMPLES)
    return ld.sample_manifest(prod, ex) == json.loads(Path(ld.SAMPLE_MANIFEST).read_text())
