"""Build a self-contained Colab job that runs commands on a clean copy of this repository at a commit.

The repository is private, so the job carries `git archive <commit>` as a base64 tarball (no credential on the VM).
On the VM it unpacks to /content/s3-genret, runs `python -m referee setup` (s3-search at the pinned commit, ESCI,
esci-sample), installs the pinned extras, runs each `--cmd` in order from the repository root, and tars `--collect`
into /content/job-out.tar.gz.

    python scripts/build_colab_job.py --commit HEAD --out results/colab/gr1-dev-job.py \
        --cmd "python scripts/gr1_run.py --dataset esci-sample --sids /content/tok --steps 2000 --out /content/out/gr1" \
        --collect /content/out
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIP = ["sentence-transformers==3.0.1", "sentencepiece", "pyarrow", "pandas", "snowballstemmer", "orjson", "pytest"]

JOB = r'''#!/usr/bin/env python3
import base64, io, json, os, shlex, subprocess, sys, tarfile, time
T0 = time.time()
SPEC = json.loads(SPEC_JSON)
def sh(cmd, **kw):
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)
code = "/content/s3-genret"
os.makedirs(code, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(code)
sh([sys.executable, "-m", "pip", "install", "--quiet"] + SPEC["pip"])
env = dict(os.environ, PYTHONPATH=code, GENRET_DATA="/content/data", TF_CPP_MIN_LOG_LEVEL="3", USE_TF="0")
sh([sys.executable, "-m", "referee", "setup"], cwd=code, env=env)
sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
for c in SPEC["cmds"]:
    sh(c.replace("python ", sys.executable + " ", 1) if c.startswith("python ") else c, cwd=code, env=env)
os.makedirs("/content/job-meta", exist_ok=True)
json.dump({"commit": SPEC["commit"], "cmds": SPEC["cmds"], "seconds": round(time.time() - T0, 1),
           "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open("/content/job-meta/job.json", "w"))
sh(["tar", "czf", "/content/job-out.tar.gz", "-C", "/content", "job-meta"] +
   [os.path.relpath(p, "/content") for p in SPEC["collect"] if os.path.exists(p)])
print("done", round(time.time() - T0), "s", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmd", action="append", required=True)
    ap.add_argument("--collect", action="append", default=[])
    a = ap.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", a.commit], cwd=ROOT, text=True).strip()
    tar = subprocess.check_output(["git", "archive", "--format=tar.gz", sha], cwd=ROOT)
    spec = {"commit": sha, "cmds": a.cmd, "collect": a.collect, "pip": PIP}
    body = JOB.replace("SPEC = json.loads(SPEC_JSON)", f"SPEC = json.loads({json.dumps(json.dumps(spec))})")
    body = body.replace("T0 = time.time()", f"T0 = time.time()\nPAYLOAD = \"{base64.b64encode(tar).decode()}\"", 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(body)
    print(f"wrote {a.out} (commit {sha[:12]}, {len(tar)} bytes packed)")


if __name__ == "__main__":
    main()
