"""The referee's commands (`python -m referee <command>`); see referee/README.md.

Owner: evaluation-steward. Makers do not edit this folder (P6).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from referee import data, runfile, score, upstream

HERE = Path(__file__).resolve().parent
DEV_BASELINES = HERE / "baselines" / "esci-sample"
# What a model may be trained or tuned on (README, "Training data"): the catalogue and the fit split. Never the test
# queries' texts, labelled or not: the founder's held-out draw is a subset of them.
ALLOWED_TRAINING = {"catalogue", "fit"}
DEV_LABEL = "DEVELOPMENT (CPU, esci-sample): not a score of record; compare with esci-sample numbers only"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _rev(cwd: Path):
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True)
    return r.stdout.strip() or None


def tree_clean():
    """True when this checkout's tracked files equal its commit."""
    try:
        r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=str(HERE.parent), capture_output=True)
    except OSError:
        return None
    return {0: True, 1: False}.get(r.returncode)


def _held_out(env: str):
    path = os.environ.get(env)
    if not path or not Path(path).exists():
        return None
    import pandas as pd
    return set(int(q) for q in pd.read_parquet(path, columns=["query_id"])["query_id"])


def _close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


# ---- baselines ------------------------------------------------------------------------------------------------------

def run_baselines(dataset: str, out_dir: Path, k: int, device: str, systems=None) -> dict:
    """The search floor's runs on `dataset`'s test queries (s3-search's code, its own process), then scored here too.
    Writes `baselines.json` beside the run files and returns it."""
    pin = upstream.verify()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **upstream.data_env(), "PYTHONPATH": str(upstream.checkout())}
    (out_dir / "s3search_meta.json").unlink(missing_ok=True)
    # One process per system: each index is built in a fresh process (on macOS a fork after Accelerate has run, as the
    # engine's parallel build does, can crash), and no system's state can reach the next.
    for name in (systems or "bm25f,bm25f-tuned,dense-tfidf-svd-128").split(","):
        cmd = [sys.executable, str(HERE / "baselines" / "s3search_runs.py"), f"--dataset={dataset}",
               f"--out-dir={out_dir}", f"--k={k}", f"--device={device}", f"--systems={name}"]
        r = subprocess.run(cmd, cwd=str(upstream.checkout()), env=env)
        if r.returncode:
            raise SystemExit(f"referee baselines: s3-search's {name} run failed (exit {r.returncode})")
    meta = json.loads((out_dir / "s3search_meta.json").read_text())
    judg, cat_ids, tq = _scorer_inputs(dataset)
    record = {"kind": "genret-baselines", "dataset": dataset, "split": "test", "k": k, "created": _now(),
              "label": (f"DEVELOPMENT (esci-sample, device {device}): not a score of record; compare with esci-sample "
                        f"numbers only") if dataset == "esci-sample" else
                       (f"esci-us baselines, device {device}: development until scored inside the founder's "
                        f"`referee full` (its results.json is the score of record)"),
              "device": device,
              "s3search": pin, "sample_manifest_sha256": upstream.PIN["files"]["datasets/esci-sample.manifest.json"],
              "systems": {}}
    for name, m in meta.items():
        res = score.score(runfile.read_run(out_dir / m["run_file"]), judg, cat_ids, expected_qids=tq)
        agree = all(_close(res["quality"][key], m["s3search_quality"].get(key))
                    for key in m["s3search_quality"] if key in res["quality"])
        record["systems"][name] = {**m, "genret_score": {"quality": res["quality"], "genret": res["genret"]},
                                   "floors_agree": agree}
        if not agree:
            raise SystemExit(f"referee baselines: {name}: s3-search scores {m['s3search_quality']} and this referee "
                             f"{res['quality']}: the shared loader or metrics drifted")
    (out_dir / "baselines.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def _scorer_inputs(dataset: str):
    ds = data.load(dataset)
    judg = data._test_judgments(dataset)          # the scorer's read
    cat_ids = set(ds.catalogue["product_id"].astype(str))
    tq = set(int(q) for q in ds.test_queries["query_id"])
    return judg, cat_ids, tq


def _load_baselines(bdir: Path, judg, cat_ids, tq, check=True):
    """{name: Resolved} for the baseline runs in `bdir`, after checking each file's sha256 and that this referee
    still gives the score recorded (and s3-search's own) for it."""
    rec = json.loads((bdir / "baselines.json").read_text())
    if rec["s3search"]["commit"] != upstream.PIN["commit"]:
        raise SystemExit(f"{bdir}/baselines.json was made at s3-search {rec['s3search']['commit']}, the pin is "
                         f"{upstream.PIN['commit']}: regenerate the baselines")
    out, problems = {}, []
    for name, m in rec["systems"].items():
        path = bdir / m["run_file"]
        if upstream.sha256_file(path) != m["run_sha256"]:
            problems.append(f"{name}: {path.name} sha256 differs from baselines.json")
            continue
        run = runfile.read_run(path)
        res = score.Resolved(run, cat_ids, None, tq)
        if check:
            q = res.summary(judg)["quality"]
            for key, want in m["genret_score"]["quality"].items():
                if not _close(q.get(key), want):
                    problems.append(f"{name}: {key} {q.get(key)} now, {want} recorded")
            for key, want in m["s3search_quality"].items():
                if key in q and not _close(q[key], want):
                    problems.append(f"{name}: {key} {q[key]} here, s3-search's own score {want}")
        out[name] = res
    return rec, out, problems


# ---- commands -------------------------------------------------------------------------------------------------------

def cmd_setup(a) -> int:
    out = upstream.setup(fetch_data=not a.no_data, build_sample=not a.no_data)
    print(json.dumps(out, indent=1))
    return 0 if out.get("sample_manifest_ok", True) else 1


def cmd_fast(a) -> int:
    """Under a minute on CPU: the pin, the unit tests (planted errors included), and on esci-sample the baselines'
    drift check; with --run, a development score of that run."""
    t0 = time.perf_counter()
    lines, ok = [], True
    try:
        pin = upstream.verify()
        lines.append(f"pin: s3-search {pin['commit'][:12]} verified ({len(pin['files_verified'])} files)")
    except upstream.PinError as e:
        print(f"referee fast: FAILED: {e}", file=sys.stderr)
        return 1
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE / "tests")],
                       cwd=str(HERE.parent), capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    lines.append(f"tests: {tail}")
    if r.returncode:
        ok = False
        print(r.stdout[-4000:], file=sys.stderr)
    if data.available("esci-sample") and upstream.loader()._sample_ready():
        if not upstream.sample_manifest_ok():
            ok = False
            lines.append("esci-sample: the local build does NOT match the pinned manifest")
        else:
            judg, cat_ids, tq = _scorer_inputs("esci-sample")
            rec, base, problems = _load_baselines(DEV_BASELINES, judg, cat_ids, tq)
            if problems:
                ok = False
                lines += [f"drift: {p}" for p in problems]
            else:
                lines.append(f"drift: {len(base)} baseline runs rescored, equal to baselines.json and to s3-search's "
                             f"own scores")
            for name, m in rec["systems"].items():
                q = m["genret_score"]["quality"]
                lines.append(f"  {name}: nDCG@10 {q['ndcg@10']:.4f}, Exact recall@100 {q['exact_recall@100']:.4f}"
                             f"  ({DEV_LABEL.split(':')[0]})")
            if a.run:
                res = _score_dev(a, judg, cat_ids, tq, base)
                lines.append(f"run {a.run}: nDCG@10 {res['quality']['ndcg@10']:.4f}, invalid share "
                             f"{res['genret']['invalid_share']}, written to {a.out}")
    else:
        lines.append("esci-sample: not built here (python -m referee setup); drift check skipped")
        if a.run:
            ok = False
            lines.append("--run needs esci-sample")
    dt = time.perf_counter() - t0
    for line in lines:
        print(line)
    print(f"referee fast: {'ok' if ok else 'FAILED'} in {dt:.1f} s")
    return 0 if ok else 1


def _score_dev(a, judg, cat_ids, tq, base) -> dict:
    run = runfile.read_run(a.run)
    if run.header.get("dataset") not in (None, "esci-sample"):
        raise SystemExit(f"{a.run}: header dataset {run.header.get('dataset')!r}; development scoring is esci-sample")
    sid_map = runfile.read_sid_map(a.sid_map) if a.sid_map else None
    res = score.score(run, judg, cat_ids, sid_map, tq)
    mine = score.Resolved(run, cat_ids, sid_map, tq).per_query_ndcg(judg)
    res["vs_baselines"] = {n: score.paired(mine, r.per_query_ndcg(judg)) for n, r in base.items()}
    res.update({"kind": "genret-score", "label": DEV_LABEL, "dataset": "esci-sample", "split": "test",
                "created": _now(), "score_of_record": False, "s3search": upstream.verify(),
                "run": {"path": str(a.run), "sha256": upstream.sha256_file(a.run), "header": run.header},
                "sid_map_file": ({"path": str(a.sid_map), "sha256": upstream.sha256_file(a.sid_map)}
                                 if a.sid_map else None),
                "genret_commit": _rev(HERE.parent), "tree_clean": tree_clean()})
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return res


def cmd_score(a) -> int:
    judg, cat_ids, tq = _scorer_inputs("esci-sample")
    _, base, problems = _load_baselines(DEV_BASELINES, judg, cat_ids, tq)
    if problems:
        print("referee score: the baselines drifted:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    res = _score_dev(a, judg, cat_ids, tq, base)
    q, g = res["quality"], res["genret"]
    print(f"{a.run} on esci-sample test ({res['n_queries']} queries; {DEV_LABEL})")
    print(f"  nDCG@10 {q['ndcg@10']:.4f} | Exact recall@10/100/1000 {q['exact_recall@10']:.4f} / "
          f"{q['exact_recall@100']:.4f} / {q['exact_recall@1000']:.4f}")
    print(f"  invalid share {g['invalid_share']} {g['invalid_by_kind']} | duplicates {g['duplicate_share']} | "
          f"short lists {g['short_list_share']}")
    if "sid_map" in res:
        sm = res["sid_map"]
        print(f"  sid map: coverage {sm['coverage']}, product collision rate {sm['full_sid']['product_collision_rate']}"
              + (f" (semantic prefix {sm['semantic_prefix']['product_collision_rate']})" if "semantic_prefix" in sm
                 else ""))
    for n, p in res["vs_baselines"].items():
        print(f"  vs {n}: nDCG@10 {p['mean_diff']:+.4f}, 95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]")
    print(f"  -> {a.out}")
    return 0


def cmd_baselines(a) -> int:
    out_dir = Path(a.out_dir or (DEV_BASELINES if a.dataset == "esci-sample" else "baselines-esci-us"))
    rec = run_baselines(a.dataset, out_dir, a.k, a.device, a.systems)
    for name, m in rec["systems"].items():
        q = m["genret_score"]["quality"]
        print(f"{name}: nDCG@10 {q['ndcg@10']:.4f}, Exact recall@{a.k} {q.get(f'exact_recall@{a.k}')}, "
              f"floors agree {m['floors_agree']}, run sha256 {m['run_sha256'][:12]}")
    print(f"-> {out_dir / 'baselines.json'} ({rec['label']})")
    return 0


def cmd_full(a) -> int:
    """The score of record: esci-us, every test query, the founder's held-out draw, beside the baselines."""
    t0 = time.perf_counter()

    def say(msg):
        print(f"[referee full {time.perf_counter() - t0:7.1f} s] {msg}", flush=True)

    dev = a.queries is not None
    out_path = Path(a.out or ("results-dev.json" if dev else "results.json"))
    if dev and out_path.name == "results.json":
        print("referee full: development mode never writes results.json", file=sys.stderr)
        return 2
    pin = upstream.verify()
    say(f"pin: s3-search {pin['commit'][:12]} verified")
    say("data: loading esci-us through the pinned loader")
    full_judg, cat_ids, tq = _scorer_inputs("esci-us")
    judg = full_judg
    if dev:
        keep = set(sorted(tq)[: a.queries])
        judg = {q: j for q, j in judg.items() if q in keep}
    say(f"data: {len(cat_ids):,} products, {len(tq):,} test queries, {len(judg):,} scored")
    subsets = {}
    for name, env in (("held_out", "GENRET_HELD_OUT"), ("search_held_out", "S3_SEARCH_HELD_OUT")):
        ids = _held_out(env)
        if ids is not None:
            subsets[name] = ids & set(judg)
    bdir = Path(a.baselines_dir)
    if not (bdir / "baselines.json").exists():
        say(f"baselines: none in {bdir}; running s3-search's on esci-us (k {a.k}, device {a.device})")
        run_baselines("esci-us", bdir, a.k, a.device)
    rec, base, problems = _load_baselines(bdir, full_judg, cat_ids, tq)
    if problems:
        print("referee full: the baselines do not check:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    say(f"baselines: {', '.join(base)} checked")
    run = runfile.read_run(a.run)
    sid_map = runfile.read_sid_map(a.sid_map) if a.sid_map else None
    res = score.score(run, judg, cat_ids, sid_map, tq, subsets)
    resolved = score.Resolved(run, cat_ids, sid_map, tq)
    out = {"kind": "genret-full", "created": _now(), "dataset": "esci-us", "split": "test", "s3search": pin,
           "genret_commit": _rev(HERE.parent), "tree_clean": tree_clean(),
           "run": {"path": str(a.run), "sha256": upstream.sha256_file(a.run), "header": run.header},
           "sid_map_file": ({"path": str(a.sid_map), "sha256": upstream.sha256_file(a.sid_map)} if a.sid_map
                            else None),
           "host": {"python": platform.python_version(), "machine": platform.machine()},
           "result": res, "baselines": {}, "vs_baselines": {}}
    for name, r in base.items():
        b = r.summary(judg)
        b["subsets"] = {s: r.summary({q: j for q, j in judg.items() if q in ids}) for s, ids in subsets.items()}
        out["baselines"][name] = {"score": b, "run_sha256": rec["systems"][name]["run_sha256"],
                                  "system": rec["systems"][name]["system"]}
        cmp = {"all": score.paired(resolved.per_query_ndcg(judg), r.per_query_ndcg(judg))}
        for s, ids in subsets.items():
            sub = {q: j for q, j in judg.items() if q in ids}
            cmp[s] = score.paired(resolved.per_query_ndcg(sub), r.per_query_ndcg(sub))
        out["vs_baselines"][name] = cmp
    caveats = []
    if "held_out" not in subsets:
        caveats.append("no held-out draw scored (GENRET_HELD_OUT unset or missing): never a score of record")
    if sid_map is None:
        caveats.append("no sid map: coverage and collisions not checked: never a score of record")
    elif res["sid_map"]["coverage"] != 1.0:
        caveats.append(f"the sid map covers {res['sid_map']['coverage']:.6f} of the catalogue, not all of it: "
                       f"never a score of record")
    trained = run.header.get("trained_on")
    if not isinstance(trained, list) or not trained or not set(trained) <= ALLOWED_TRAINING:
        caveats.append(f"the run header's trained_on is {trained!r}, not a non-empty list within "
                       f"{sorted(ALLOWED_TRAINING)}: never a score of record")
    if out["tree_clean"] is not True:
        caveats.append("this checkout's tracked files differ from its commit: never a score of record")
    if dev:
        caveats.append(f"DEVELOPMENT MODE: the first {a.queries} test queries only")
    out["caveats"] = caveats
    out["score_of_record"] = not caveats
    out["label"] = "score of record" if out["score_of_record"] else "not a score of record (see caveats)"
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    q = res["quality"]
    say(f"done: nDCG@10 {q['ndcg@10']:.4f}, held-out {res.get('subsets', {}).get('held_out', {}).get('quality', {}).get('ndcg@10')}; "
        f"invalid share {res['genret']['invalid_share']}; {out['label']} -> {out_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m referee", description="Floor 3's referee (referee/README.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="clone s3-search at the pin, fetch ESCI, build and check esci-sample")
    s.add_argument("--no-data", action="store_true", help="the checkout only (CI)")
    s.set_defaults(fn=cmd_setup)
    s = sub.add_parser("fast", help="CPU, under a minute: pin, tests, the baselines' drift check")
    s.add_argument("--run")
    s.add_argument("--sid-map")
    s.add_argument("--out", default="results-dev.json")
    s.set_defaults(fn=cmd_fast)
    s = sub.add_parser("score", help="development score of a run on esci-sample's test queries")
    s.add_argument("--run", required=True)
    s.add_argument("--sid-map")
    s.add_argument("--out", default="results-dev.json")
    s.set_defaults(fn=cmd_score)
    s = sub.add_parser("baselines", help="s3-search's BM25F, tuned BM25F and dense runs on a dataset's test queries")
    s.add_argument("--dataset", default="esci-sample", choices=data.DATASETS)
    s.add_argument("--k", type=int, default=100)
    s.add_argument("--device", default="cpu")
    s.add_argument("--systems")
    s.add_argument("--out-dir")
    s.set_defaults(fn=cmd_baselines)
    s = sub.add_parser("full", help="the score of record on esci-us (Colab, run by the founder)")
    s.add_argument("--run", required=True)
    s.add_argument("--sid-map")
    s.add_argument("--baselines-dir", default="baselines-esci-us")
    s.add_argument("--k", type=int, default=1000)
    s.add_argument("--device", default="cuda")
    s.add_argument("--queries", type=int, default=None, help="DEVELOPMENT MODE: the first N test queries")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_full)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except (runfile.RunError, upstream.PinError) as e:
        print(f"referee {a.cmd}: refused: {e}", file=sys.stderr)
        return 2
