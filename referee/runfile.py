"""Run files and semantic-ID maps: the two files a system hands the referee.

Run file (`.jsonl` or `.jsonl.gz`). The first line is a header, then one line per query:

    {"genret_run": 1, "name": "gr1-dev-3", "dataset": "esci-sample", "split": "test", "k": 100,
     "system": "free text: what produced it", "sid_levels": 3, "trained_on": ["catalogue", "fit"]}
    {"qid": 12, "ranked": ["B07XYZ", {"sid": "17-401-9-0"}, {"sid": "3-3-3-0", "pid": "B01ABC"}, null]}

- `ranked` is the result list in rank order, at most `k` entries (and k at most 1000). An entry is
  - a product id (a string): what a retrieval system returns, or a generative system that decoded its own ids;
  - `{"sid": s}`: a generated semantic id, which the referee looks up in the sid map;
  - `{"sid": s, "pid": p}`: a generated semantic id and the product the system chose for it (needed when the id is
    shared by several products: a collision);
  - `null`: the system generated something it could not decode to any id.
- `sid_levels` (optional): how many leading codes of a sid are semantic, the rest being a disambiguation suffix
  (TIGER appends one). The collision rate is reported on the full sid and on the semantic prefix.
- `trained_on`: what the model was trained or tuned on, within ["catalogue", "fit"]. A run that names anything else
  (the test queries' texts included) is never a score of record (README, "Training data").

Sid map (`.parquet` with columns `product_id`, `sid`, or a two-column `.tsv` / `.tsv.gz` with that header): the
semantic id of every product in the dataset's catalogue. A sid is its codes joined by "-" (for example "17-401-9").

Owner: evaluation-steward. Makers do not edit this folder (P6).
"""
from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

K_MAX = 1000


class RunError(ValueError):
    """The file does not follow the format; it is refused, not scored."""


@dataclass
class Run:
    header: dict
    ranked: dict = field(default_factory=dict)   # {qid: [entry, ...]}


def _open_text(path):
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return io.StringIO(raw.decode("utf-8"))


def _check_entry(e, where):
    if e is None or isinstance(e, str):
        return
    if isinstance(e, dict) and set(e) <= {"sid", "pid"} and isinstance(e.get("sid"), str) \
            and (e.get("pid") is None or isinstance(e.get("pid"), str)):
        return
    raise RunError(f"{where}: an entry is a product id, {{'sid': ...}}, {{'sid': ..., 'pid': ...}} or null, not {e!r}")


def read_run(path) -> Run:
    f = _open_text(path)
    first = f.readline()
    try:
        header = json.loads(first)
    except json.JSONDecodeError as e:
        raise RunError(f"{path}: line 1 is not JSON ({e})")
    if not isinstance(header, dict) or header.get("genret_run") != 1:
        raise RunError(f"{path}: line 1 must be the header, with \"genret_run\": 1")
    k = header.get("k")
    if not isinstance(k, int) or not 1 <= k <= K_MAX:
        raise RunError(f"{path}: header k must be an integer in 1..{K_MAX}, not {k!r}")
    run = Run(header=header)
    for n, line in enumerate(f, start=2):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise RunError(f"{path}:{n}: not JSON ({e})")
        if not isinstance(row, dict) or set(row) != {"qid", "ranked"}:
            raise RunError(f"{path}:{n}: a query line has exactly the keys qid and ranked")
        q, ranked = row["qid"], row["ranked"]
        if not isinstance(q, int) or isinstance(q, bool):
            raise RunError(f"{path}:{n}: qid must be an integer")
        if q in run.ranked:
            raise RunError(f"{path}:{n}: qid {q} appears twice")
        if not isinstance(ranked, list):
            raise RunError(f"{path}:{n}: ranked must be a list")
        if len(ranked) > k:
            raise RunError(f"{path}:{n}: qid {q} has {len(ranked)} entries, more than the header's k {k}")
        for e in ranked:
            _check_entry(e, f"{path}:{n}")
        run.ranked[q] = ranked
    return run


def write_run(path, header: dict, ranked: dict) -> None:
    """Writes a run file; `.gz` paths are gzipped with a zero mtime, so the same run has the same sha256."""
    lines = [json.dumps({"genret_run": 1, **header}, sort_keys=True)]
    lines += [json.dumps({"qid": int(q), "ranked": ranked[q]}) for q in sorted(ranked)]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = Path(path)
    if path.suffix == ".gz":
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
            g.write(data)
        data = buf.getvalue()
    path.write_bytes(data)


def read_sid_map(path) -> dict:
    """{product_id: sid}. Refuses a product listed twice or an empty sid."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        frame = pd.read_parquet(path, columns=["product_id", "sid"])
        pids, sids = frame["product_id"].astype(str).tolist(), frame["sid"].astype(str).tolist()
    else:
        f = _open_text(path)
        head = f.readline().rstrip("\n").split("\t")
        if head != ["product_id", "sid"]:
            raise RunError(f"{path}: the header line must be 'product_id<TAB>sid'")
        pids, sids = [], []
        for n, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise RunError(f"{path}:{n}: two tab-separated columns")
            pids.append(parts[0])
            sids.append(parts[1])
    out: dict = {}
    for p, s in zip(pids, sids):
        if not s:
            raise RunError(f"{path}: product {p} has an empty sid")
        if p in out:
            raise RunError(f"{path}: product {p} is listed twice")
        out[p] = s
    return out
