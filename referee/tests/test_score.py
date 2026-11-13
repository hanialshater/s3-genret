"""The scorer on hand-made runs, with planted errors it must catch. No ESCI data needed (the pinned metrics are)."""
from __future__ import annotations

import gzip
import json
import math

import pytest

from referee import runfile, score, upstream
from referee.runfile import Run, RunError

CAT = {"p1", "p2", "p3", "p4", "p5", "p6"}
JUDG = {1: {"p1": "E", "p2": "S", "p3": "C", "p4": "I"}, 2: {"p5": "E"}}
SIDS = {"p1": "1-1-0", "p2": "1-2-0", "p3": "2-1-0", "p4": "2-1-1", "p5": "3-3-0", "p6": "3-3-0"}


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def run(ranked, k=10, **h):
    return Run(header={"genret_run": 1, "k": k, **h}, ranked=ranked)


def test_gains_are_the_search_floors():
    assert upstream.metrics().GAIN == {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def test_ndcg_matches_hand_computation():
    res = score.score(run({1: ["p2", "p1", "p6"], 2: ["p5"]}), JUDG, CAT)
    ideal = dcg([1.0, 0.1, 0.01, 0.0])
    want = (dcg([0.1, 1.0, 0.0]) / ideal + 1.0) / 2
    assert res["quality"]["ndcg@10"] == pytest.approx(want, abs=1e-12)
    assert res["quality"]["exact_recall@10"] == 1.0
    assert res["genret"]["invalid_share"] == 0.0


def test_planted_duplicate_cannot_add_gain():
    once = score.score(run({1: ["p1", "p6"], 2: []}), JUDG, CAT)
    twice = score.score(run({1: ["p1", "p1"], 2: []}), JUDG, CAT)
    assert twice["quality"]["ndcg@10"] == once["quality"]["ndcg@10"]
    assert twice["genret"]["duplicate_share"] == 0.5


def test_planted_invalid_ids_keep_their_rank_with_gain_zero():
    # An invalid id at rank 1 pushes the Exact product to rank 2: it must score as rank 2, not be dropped.
    res = score.score(run({1: [None, "p1"], 2: [{"sid": "9-9-9"}, "nope"]}), JUDG, CAT, SIDS)
    ideal = dcg([1.0, 0.1, 0.01])
    assert res["quality"]["ndcg@10"] == pytest.approx((dcg([0.0, 1.0]) / ideal + 0.0) / 2, abs=1e-12)
    g = res["genret"]
    assert g["invalid_share"] == 0.75
    assert g["invalid_by_kind"]["undecodable"] == 1
    assert g["invalid_by_kind"]["unknown_sid"] == 1
    assert g["invalid_by_kind"]["unknown_product"] == 1


def test_sids_resolve_through_the_map_and_collisions_need_a_choice():
    res = score.score(run({1: [{"sid": "1-1-0"}], 2: [{"sid": "3-3-0"}]}), JUDG, CAT, SIDS)
    g = res["genret"]
    assert g["invalid_by_kind"]["ambiguous"] == 1          # 3-3-0 is p5 and p6: no product chosen
    assert res["quality"]["ndcg@10"] == pytest.approx((1.0 / dcg([1.0, 0.1, 0.01, 0.0]) + 0.0) / 2)
    chose = score.score(run({1: [], 2: [{"sid": "3-3-0", "pid": "p5"}]}), JUDG, CAT, SIDS)
    assert chose["genret"]["invalid_share"] == 0.0
    assert chose["genret"]["shared_sid_hit_share"] == 1.0
    wrong = score.score(run({1: [], 2: [{"sid": "1-1-0", "pid": "p5"}]}), JUDG, CAT, SIDS)
    assert wrong["genret"]["invalid_by_kind"]["inconsistent"] == 1


def test_collision_rates_on_full_sid_and_semantic_prefix():
    st = score.sid_map_stats(SIDS, CAT, sid_levels=2)
    assert st["coverage"] == 1.0
    assert st["full_sid"]["product_collision_rate"] == pytest.approx(2 / 6)       # p5, p6
    assert st["semantic_prefix"]["product_collision_rate"] == pytest.approx(4 / 6)  # p3, p4 and p5, p6
    assert st["distinct_codes_per_level"] == [3, 3, 2]
    partial = score.sid_map_stats({"p1": "1", "zz": "2"}, CAT)
    assert partial["coverage"] == pytest.approx(1 / 6) and partial["n_foreign"] == 1


def test_short_lists_and_missing_queries():
    res = score.score(run({1: ["p1", None]}), JUDG, CAT)
    assert res["n_queries_missing"] == 1                      # query 2 has no line: it scores 0
    assert res["quality"]["ndcg@10"] == pytest.approx((1.0 / dcg([1.0, 0.1, 0.01, 0.0])) / 2)
    assert res["genret"]["short_list_share"]["<10"] == 1.0


def test_planted_queries_outside_the_split_are_refused():
    with pytest.raises(RunError):
        score.score(run({1: ["p1"], 99: ["p5"]}), JUDG, CAT, expected_qids={1, 2})


def test_sids_without_a_map_are_refused():
    with pytest.raises(RunError):
        score.score(run({1: [{"sid": "1-1-0"}]}), JUDG, CAT)


def test_paired_difference_and_interval():
    p = score.paired({1: 0.5, 2: 0.5}, {1: 0.25, 2: 0.25})
    assert p["mean_diff"] == 0.25 and p["ci95"] == [0.25, 0.25] and p["wins"] == 2
    with pytest.raises(ValueError):
        score.paired({1: 0.5}, {2: 0.5})


def test_run_file_round_trip_and_refusals(tmp_path):
    path = tmp_path / "r.jsonl.gz"
    runfile.write_run(path, {"name": "t", "k": 3}, {2: ["p5"], 1: ["p1", {"sid": "1-2-0"}, None]})
    first = upstream.sha256_file(path)
    runfile.write_run(path, {"name": "t", "k": 3}, {2: ["p5"], 1: ["p1", {"sid": "1-2-0"}, None]})
    assert upstream.sha256_file(path) == first                # gzip with mtime 0: same run, same checksum
    r = runfile.read_run(path)
    assert r.header["k"] == 3 and r.ranked[1][1] == {"sid": "1-2-0"}

    def bad(lines):
        p = tmp_path / "bad.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        with pytest.raises(RunError):
            runfile.read_run(p)

    bad([{"k": 3}, {"qid": 1, "ranked": []}])                           # no genret_run
    bad([{"genret_run": 1, "k": 2000}])                                 # k over 1000
    bad([{"genret_run": 1, "k": 1}, {"qid": 1, "ranked": ["a", "b"]}])  # longer than k
    bad([{"genret_run": 1, "k": 2}, {"qid": 1, "ranked": []}, {"qid": 1, "ranked": []}])  # qid twice
    bad([{"genret_run": 1, "k": 2}, {"qid": 1, "ranked": [7]}])         # not an entry
    bad([{"genret_run": 1, "k": 2}, {"qid": 1, "ranked": [], "score": [1]}])  # extra key


def test_sid_map_files(tmp_path):
    p = tmp_path / "m.tsv.gz"
    p.write_bytes(gzip.compress(b"product_id\tsid\np1\t1-2\np2\t3-4\n"))
    assert runfile.read_sid_map(p) == {"p1": "1-2", "p2": "3-4"}
    d = tmp_path / "d.tsv"
    d.write_text("product_id\tsid\np1\t1\np1\t2\n")
    with pytest.raises(RunError):
        runfile.read_sid_map(d)
