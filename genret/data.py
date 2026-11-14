"""ESCI for the researcher's code, read only through the referee's loader.

Order of preference:
1. `referee.data.load` in this repository (the evaluation steward's, which wraps s3-search's loader at its pin);
2. until that lands on this branch, s3-search's own `referee/loader.py` from a checkout named by `S3_SEARCH_ROOT`
   (default `~/labs/s3-search`), the same loader the steward pins.

Only `catalogue`, `fit` (train-split judgments) and `test_queries` (texts, no judgments) are used. The test judgments
are never read here.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _floor3_referee():
    if not (REPO / "referee" / "data.py").exists():
        return None
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from referee import data  # noqa: E402  (the steward's)
    return data


def _s3search_loader():
    root = Path(os.environ.get("S3_SEARCH_ROOT", Path.home() / "labs" / "s3-search")).expanduser()
    os.environ.setdefault("S3_SEARCH_DATA", str(root / "data" / "esci"))
    os.environ.setdefault("S3_SEARCH_SAMPLE", str(root / "data" / "esci-sample"))
    spec = importlib.util.spec_from_file_location("s3search_referee_loader", root / "referee" / "loader.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


def load(name: str = "esci-sample"):
    """A Dataset with .catalogue, .fit, .test_queries (no test judgments)."""
    ref = _floor3_referee()
    if ref is not None:
        return ref.load(name)
    return _s3search_loader().load_dataset(name)
