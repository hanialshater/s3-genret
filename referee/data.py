"""What members of Floor 3 may read of ESCI: the search floor's loader at the pin, nothing else.

    from referee import data
    ds = data.load("esci-sample")      # or "esci-us"
    ds.catalogue      # product_id + the five text columns: the products to give semantic ids to (all of them)
    ds.fit            # query_id, query, product_id, esci_label: the train split, to train and tune on
    ds.test_queries   # query_id, query: the queries a run answers, with no judgments

The test judgments are read only by the scorer (`referee/score.py`, through `_test_judgments`). Loading them, or the
founder's held-out draw, any other way breaks the floor's rules. ESCI is public, so this is a rule, not a secret.

Owner: evaluation-steward. Makers do not edit this folder (P6).
"""
from __future__ import annotations

from referee import upstream

DATASETS = ("esci-sample", "esci-us")


def load(name: str = "esci-sample"):
    """The pinned loader's `Dataset` for esci-sample or esci-us (toy is the search floor's, not served here)."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; one of {DATASETS}")
    return upstream.loader().load_dataset(name)


def available(name: str) -> bool:
    return upstream.loader().available(name)


def _test_judgments(name: str) -> dict:
    """{query_id: {product_id: label}} for the dataset's test split. The scorer only."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; one of {DATASETS}")
    return upstream.loader()._load_dataset_test_judgments(name)
