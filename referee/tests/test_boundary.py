"""The pin and the data boundary."""
from __future__ import annotations

import json
import subprocess

import pytest

from referee import data, upstream


def test_the_pin_verifies():
    v = upstream.verify()
    assert v["commit"] == upstream.PIN["commit"]


def test_a_checkout_at_another_commit_is_refused(tmp_path):
    root = tmp_path / "s3"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(upstream.checkout()), str(root)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "HEAD~1"], cwd=root, check=True)
    with pytest.raises(upstream.PinError):
        upstream.verify(root)


def test_a_changed_pinned_file_is_refused(tmp_path):
    root = tmp_path / "s3"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(upstream.checkout()), str(root)], check=True)
    subprocess.run(["git", "checkout", "--quiet", upstream.PIN["commit"]], cwd=root, check=True)
    (root / "referee" / "metrics.py").write_text("GAIN = {'E': 1.0, 'S': 1.0, 'C': 1.0, 'I': 1.0}\n")
    with pytest.raises(upstream.PinError):
        upstream.verify(root)


def test_toy_is_not_served():
    with pytest.raises(ValueError):
        data.load("toy")


@pytest.mark.skipif(not (upstream.data_root() / "esci-sample").exists(), reason="esci-sample not built here")
def test_members_see_no_test_judgments():
    ds = data.load("esci-sample")
    assert list(ds.test_queries.columns) == ["query_id", "query"]
    assert set(ds.fit["query_id"]).isdisjoint(set(ds.test_queries["query_id"]))
    assert "esci_label" not in ds.test_queries.columns


def test_dev_baselines_are_recorded_at_the_pin():
    rec = json.loads((upstream.HERE / "baselines" / "esci-sample" / "baselines.json").read_text())
    assert rec["s3search"]["commit"] == upstream.PIN["commit"]
    assert set(rec["systems"]) == {"bm25f", "bm25f-tuned", "dense-tfidf-svd-128"}
    for name, m in rec["systems"].items():
        assert m["floors_agree"], name
        assert upstream.sha256_file(upstream.HERE / "baselines" / "esci-sample" / m["run_file"]) == m["run_sha256"]
