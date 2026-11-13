"""The full mode's bookkeeping on a hand-made dataset: baselines checked, the held-out subset scored, and the
score of record refused whenever a condition is missing."""
from __future__ import annotations

import json
from argparse import Namespace

import pandas as pd

from referee import cli, runfile, score, upstream
from referee.tests.test_score import CAT, JUDG, SIDS


def _baselines(bdir):
    bdir.mkdir()
    ranked = {1: ["p2", "p1"], 2: ["p6", "p5"]}
    runfile.write_run(bdir / "bm25f.run.jsonl.gz", {"name": "bm25f", "k": 10}, ranked)
    q = score.score(runfile.read_run(bdir / "bm25f.run.jsonl.gz"), JUDG, CAT)["quality"]
    rec = {"s3search": {"commit": upstream.PIN["commit"]},
           "systems": {"bm25f": {"run_file": "bm25f.run.jsonl.gz", "system": "bm25f",
                                 "run_sha256": upstream.sha256_file(bdir / "bm25f.run.jsonl.gz"),
                                 "genret_score": {"quality": q}, "s3search_quality": q}}}
    (bdir / "baselines.json").write_text(json.dumps(rec))


def _args(tmp_path, **kw):
    base = dict(run=str(tmp_path / "gr.jsonl"), sid_map=str(tmp_path / "m.tsv"),
                baselines_dir=str(tmp_path / "b"), k=10, device="cpu", queries=None,
                out=str(tmp_path / "results.json"))
    base.update(kw)
    return Namespace(**base)


def _setup(tmp_path, monkeypatch, sids=SIDS):
    monkeypatch.setattr(cli, "_scorer_inputs", lambda name: (JUDG, CAT, {1, 2}))
    _baselines(tmp_path / "b")
    runfile.write_run(tmp_path / "gr.jsonl", {"name": "gr", "k": 10, "sid_levels": 2,
                                              "trained_on": ["catalogue", "fit"]},
                      {1: [{"sid": "1-1-0"}, None], 2: [{"sid": "3-3-0", "pid": "p5"}]})
    (tmp_path / "m.tsv").write_text("product_id\tsid\n" + "".join(f"{p}\t{s}\n" for p, s in sids.items()))


def test_full_scores_held_out_and_baselines(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    pd.DataFrame({"query_id": [2]}).to_parquet(tmp_path / "ho.parquet")
    monkeypatch.setenv("GENRET_HELD_OUT", str(tmp_path / "ho.parquet"))
    monkeypatch.setattr(cli, "tree_clean", lambda: True)
    assert cli.cmd_full(_args(tmp_path)) == 0
    out = json.loads((tmp_path / "results.json").read_text())
    assert out["result"]["subsets"]["held_out"]["n_queries"] == 1
    assert out["result"]["subsets"]["held_out"]["quality"]["ndcg@10"] == 1.0
    assert "bm25f" in out["baselines"] and "held_out" in out["vs_baselines"]["bm25f"]
    assert out["score_of_record"] is True and out["caveats"] == []


def test_full_without_held_out_or_full_coverage_is_not_of_record(tmp_path, monkeypatch):
    part = {p: s for p, s in SIDS.items() if p != "p4"}
    _setup(tmp_path, monkeypatch, sids=part)
    monkeypatch.delenv("GENRET_HELD_OUT", raising=False)
    monkeypatch.setattr(cli, "tree_clean", lambda: True)
    assert cli.cmd_full(_args(tmp_path)) == 0
    out = json.loads((tmp_path / "results.json").read_text())
    assert out["score_of_record"] is False
    assert any("held-out" in c for c in out["caveats"]) and any("covers" in c for c in out["caveats"])


def test_full_refuses_a_tampered_baseline(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    runfile.write_run(tmp_path / "b" / "bm25f.run.jsonl.gz", {"name": "bm25f", "k": 10}, {1: ["p1"], 2: ["p5"]})
    assert cli.cmd_full(_args(tmp_path)) == 1


def test_development_mode_never_writes_results_json(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert cli.cmd_full(_args(tmp_path, queries=1)) == 2


def test_full_refuses_training_on_test_queries_as_of_record(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    runfile.write_run(tmp_path / "gr.jsonl", {"name": "gr", "k": 10, "trained_on": ["catalogue", "test_queries"]},
                      {1: ["p1"]})
    pd.DataFrame({"query_id": [2]}).to_parquet(tmp_path / "ho.parquet")
    monkeypatch.setenv("GENRET_HELD_OUT", str(tmp_path / "ho.parquet"))
    monkeypatch.setattr(cli, "tree_clean", lambda: True)
    assert cli.cmd_full(_args(tmp_path)) == 0
    out = json.loads((tmp_path / "results.json").read_text())
    assert out["score_of_record"] is False and any("trained_on" in c for c in out["caveats"])
