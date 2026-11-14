"""GR2: SID-LLM-style generative retrieval on a sid map of RQ-KMeans ids. Writes the referee's run files:
B2 (MTP, static resolution), B1 (MTP + ranker) and, with --ar, B3 (autoregressive constrained beam, static).
B4 is this script with --no-cojudged. Design: lab/predictions/genret-GR2.md.

    python scripts/gr2_run.py --dataset esci-sample --sids <tokenise dir of e5 rqkmeans> --emb-cache /content/cache \
        --stage1-steps 200 --stage2-steps 1000 --mtp-steps 300 --out /content/out/gr2-dev

Every number from esci-sample is development only, never a claim.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from genret import data, gr2, ranker as rk, text  # noqa: E402
from genret.mtp import MTPHeads  # noqa: E402
from genret.sids import SidTrie  # noqa: E402


def read_codes(sid_dir: Path, ids):
    rows = gzip.open(sid_dir / "sid_map.tsv.gz", "rt").read().splitlines()[1:]
    m = {p: list(map(int, s.split("-"))) for p, s in (r.split("\t") for r in rows)}
    return np.array([m[p] for p in ids], dtype=np.int64)


def static_order(sid_dir: Path, emb_cache, ids, codes, log):
    """A query-free order inside each shared id: distance of the product's embedding to the id's reconstruction
    (closest first), standing in for the paper's static quality score. Falls back to product-id order."""
    st = sid_dir / "rqkmeans.pt"
    files = sorted(glob.glob(str(Path(emb_cache) / "emb-intfloat_e5-base-v2-*.npy"))) if emb_cache else []
    if not st.exists() or not files:
        log("static order: product id (no embeddings or codebooks found)")
        return {p: i for i, p in enumerate(sorted(ids))}
    emb = np.load(files[-1])
    if emb.shape[0] != len(ids):
        log(f"static order: product id (embedding cache has {emb.shape[0]} rows, not {len(ids)})")
        return {p: i for i, p in enumerate(sorted(ids))}
    cb = torch.load(st)["codebooks"]
    rec = sum(cb[l][torch.as_tensor(codes[:, l])] for l in range(len(cb))).numpy()
    d = ((emb - rec) ** 2).sum(1)
    log("static order: distance to the id's reconstruction")
    return {p: float(x) for p, x in zip(ids, d)}


def _commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parent).strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esci-sample")
    ap.add_argument("--sids", required=True)
    ap.add_argument("--emb-cache", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--stage1-steps", type=int, default=2000)
    ap.add_argument("--stage2-steps", type=int, default=0, help="0: one pass over the stage-2 mix")
    ap.add_argument("--mtp-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=2e-4)
    ap.add_argument("--catalogue-epochs", type=float, default=2.0, help="attributes -> id, per stage-2 pass")
    ap.add_argument("--title-frac", type=float, default=0.33, help="id -> title on this share of products")
    ap.add_argument("--query-epochs", type=int, default=6)
    ap.add_argument("--cojudged-epochs", type=int, default=4)
    ap.add_argument("--no-cojudged", action="store_true", help="arm B4")
    ap.add_argument("--width", type=int, default=250)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--cap", type=int, default=300, help="products scored by the ranker per query")
    ap.add_argument("--ar", action="store_true", help="also arm B3: autoregressive constrained beam")
    ap.add_argument("--ar-width", type=int, default=100)
    ap.add_argument("--ranker-epochs", type=int, default=1)
    ap.add_argument("--w-ranker", type=float, default=0.7)
    ap.add_argument("--no-ranker", action="store_true")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--fit-check", type=int, default=200)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="gr2")
    ap.add_argument("--label", default="development only, never a claim")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731
    report = {"label": a.label, "args": vars(a), "commit": _commit()}

    ds = data.load(a.dataset)
    cat, fit = ds.catalogue, ds.fit
    ids = cat["product_id"].astype(str).tolist()
    codes = read_codes(Path(a.sids), ids)
    ks = [int(json.loads((Path(a.sids) / "tokenise.json").read_text())["k"])] * codes.shape[1]
    trie = SidTrie(codes, ids)
    static = static_order(Path(a.sids), a.emb_cache, ids, codes, log)
    log(f"{a.dataset}: {len(ids)} products, {len(trie.by_sid)} distinct ids, ks {ks}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    base = lm.get_input_embeddings().weight.shape[0]
    vocab = gr2.add_sid_vocab(lm, base, ks)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    enc_batch = lambda ts: tok(ts, add_special_tokens=False)["input_ids"]  # noqa: E731
    ex = gr2.Examples(enc, vocab, cat, codes, ids, fit, eos=eos, seed=a.seed, encode_batch=enc_batch)
    log("catalogue tokenised")
    rng = random.Random(a.seed)
    n = len(ids)

    # stage 1: alignment (new rows only)
    s1 = [ex.attr_to_sid(i) for i in rng.sample(range(n), min(n, a.stage1_steps * a.batch // 2))]
    s1 += [ex.sid_to_title(i) for i in rng.sample(range(n), min(n, a.stage1_steps * a.batch // 2))]
    log(f"stage 1: {len(s1)} examples")
    report["stage1"] = gr2.train_lm(lm, s1, a.stage1_steps, a.batch, a.lr1, a.device, pad, warmup=100,
                                    only_new_rows_from=base, log=log, max_len=a.max_len, decay_frac=0.5)
    del s1

    # stage 2: joint pre-training
    s2 = []
    for e in range(int(np.ceil(a.catalogue_epochs))):
        idx = list(range(n)) if e + 1 <= a.catalogue_epochs else rng.sample(range(n), int(n * (a.catalogue_epochs - e)))
        s2 += [ex.attr_to_sid(i) for i in idx]
    s2 += [ex.sid_to_title(i) for i in rng.sample(range(n), int(n * a.title_frac))]
    for e in range(a.query_epochs):
        s2 += ex.query_pairs(e)
    if not a.no_cojudged:
        for e in range(a.cojudged_epochs):
            s2 += ex.cojudged(e)
    ntok = sum(min(len(x), a.max_len) for x, _ in s2)
    steps2 = a.stage2_steps or max(1, len(s2) // a.batch)
    log(f"stage 2: {len(s2)} examples, about {ntok / 1e6:.1f}M tokens; {steps2} steps, {steps2 * a.batch / len(s2):.2f} passes")
    report["stage2"] = {"examples": len(s2), "tokens": ntok,
                        "steps": steps2, "history": gr2.train_lm(lm, s2, steps2, a.batch, a.lr2, a.device, pad, warmup=300,
                                                log=log, max_len=a.max_len, grad_ckpt=a.grad_ckpt)}
    del s2

    # MTP heads
    pos = {p: i for i, p in enumerate(ids)}

    def pairs(e):
        r = random.Random(1000 + e)
        out_ = [(q, codes[pos[str(p)]]) for q, p, l in zip(fit["query"], fit["product_id"], fit["esci_label"])
                if str(p) in pos and (l == "E" or (l == "S" and r.random() < gr2.S_RATE))]
        r.shuffle(out_)
        return out_

    heads = MTPHeads(lm.config.hidden_size, ks)
    report["mtp"] = gr2.train_mtp(lm, heads, enc, vocab, pairs, a.mtp_steps, a.batch, a.device, pad, log=log)

    # fit-query check (training queries: is it learnt?)
    if a.fit_check:
        fq = fit.drop_duplicates("query_id").head(a.fit_check)
        exact = fit[fit["esci_label"] == "E"].groupby("query_id")["product_id"].apply(lambda x: set(map(str, x)))
        fres = gr2.decode_mtp(lm, heads, enc, vocab, fq["query"].tolist(), trie, 50, a.device, pad)
        hits = [len({p for p, _, _ in gr2.expand(r, trie, static, 10)} & exact.get(q, set())) > 0
                for q, r in zip(fq["query_id"], fres)]
        report["fit_check"] = {"n": len(hits), "share_with_an_exact_in_top10": sum(hits) / max(1, len(hits)),
                               "note": "fit queries the model was trained on: a check that training is learnt"}
        log(f"fit check: {report['fit_check']}")

    tq = ds.test_queries.head(a.max_queries) if a.max_queries else ds.test_queries
    queries, qids = tq["query"].tolist(), [int(q) for q in tq["query_id"]]
    from referee import runfile

    def header(name, system):
        return {"name": name, "dataset": a.dataset, "split": "test", "k": a.k, "sid_levels": len(ks),
                "trained_on": ["catalogue", "fit"],
                "system": f"GR2 {a.model}; {system}; cojudged={not a.no_cojudged}; commit {report['commit']}; {a.label}"}

    t1 = time.time()
    mres = gr2.decode_mtp(lm, heads, enc, vocab, queries, trie, a.width, a.device, pad)
    report["seconds_mtp_decode_per_query_ms"] = round(1000 * (time.time() - t1) / max(1, len(queries)), 2)
    cands = {q: gr2.expand(r, trie, static, a.cap) for q, r in zip(qids, mres)}
    runfile.write_run(out / "run-b2-mtp.jsonl.gz", header(a.name + "-b2", f"MTP beam walk W={a.width}, static resolution"),
                      {q: gr2.ranked_entries(c, a.k) for q, c in cands.items()})
    log(f"B2 written; MTP decode {report['seconds_mtp_decode_per_query_ms']} ms/query")

    if a.ar:
        t1 = time.time()
        ares = gr2.decode_ar(lm, enc, vocab, queries, trie, a.ar_width, a.device, pad)
        report["seconds_ar_decode_per_query_ms"] = round(1000 * (time.time() - t1) / max(1, len(queries)), 2)
        runfile.write_run(out / "run-b3-ar.jsonl.gz", header(a.name + "-b3", f"AR constrained beam {a.ar_width}, static"),
                          {q: gr2.ranked_entries(gr2.expand(r, trie, static, a.k), a.k) for q, r in zip(qids, ares)})
        log(f"B3 written; AR decode {report['seconds_ar_decode_per_query_ms']} ms/query")

    lm.to("cpu")
    del lm
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()

    if not a.no_ranker:
        ptext = {p: t for p, t in zip(ids, text.catalogue_texts(cat))}
        tr, tu = rk.fit_examples(fit, ptext, seed=a.seed)
        log(f"ranker: {len(tr)} training pairs, {len(tu)} tuning pairs (fit queries)")
        m = rk.train_ranker(tr, epochs=a.ranker_epochs, device=a.device)
        ranked = {}
        t1 = time.time()
        for q, qt in zip(qids, queries):
            c = cands[q]
            if not c:
                ranked[q] = []
                continue
            rs = m.predict([(qt, ptext[p]) for p, _, _ in c], batch_size=256, show_progress_bar=False)
            sc = rk.combine(rs, [s for _, _, s in c], a.w_ranker)
            order = np.argsort(-sc, kind="stable")
            ranked[q] = gr2.ranked_entries([c[i] for i in order], a.k)
        report["seconds_rerank_per_query_ms"] = round(1000 * (time.time() - t1) / max(1, len(queries)), 2)
        runfile.write_run(out / "run-b1-mtp-ranker.jsonl.gz",
                          header(a.name + "-b1", f"MTP W={a.width} + MiniLM ranker over {a.cap} products, w={a.w_ranker}"),
                          ranked)
        log("B1 written")
    report["seconds_total"] = round(time.time() - t0, 1)
    report["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "gr2.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    log("done")


if __name__ == "__main__":
    main()
