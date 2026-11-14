# Floor 3's referee: generative retrieval on ESCI-US

owner: evaluation-steward · written 2026-11-13 (G0) · **makers do not edit this folder (P6)**. A maker who thinks a
check is wrong raises it with the steward on `docs/BOARD.md` and does not edit it.

It judges generative retrieval (GR1 TIGER-style, GR2 SID-LLM-style) on ESCI-US **like for like with the search
floor**: the same queries, the same products, the same gains, and the search floor's own BM25F and dense runs as the
bar.

## One data path, pinned

- ESCI is read only through **s3-search's referee loader and metrics, at the commit in `pin.json`**
  (`767397600d89`, s3-search `main` on 2026-11-13). `upstream.py` loads them from a checkout of s3-search at that
  commit and refuses to run if the checkout is at another commit, has local changes, or a pinned file's sha256 differs.
  Nothing here copies the loader or the metrics.
- Members read data with `from referee import data; ds = data.load("esci-sample" | "esci-us")`: the catalogue, the
  fit split (train-split judgments) and the test queries' texts, with no judgments. The test judgments are read only
  by the scorer. The search floor's `toy` is not served here.
- `esci-sample` is the search floor's sample (2,000 train and 500 test queries, seed 20261109, 48,639 products),
  rebuilt through the pinned loader and checked against the pinned manifest. Its catalogue holds only judged products,
  so its numbers are much higher than ESCI-US's: compare esci-sample numbers with each other only.

## What it measures

The quality numbers are s3-search's `referee/metrics.py`, called at the pin:

- **nDCG@10** on the graded labels with the search floor's gains: E 1.0, S 0.1, C 0.01, I 0.0. An unjudged product
  counts 0, and the ideal DCG is taken over the query's judged products. The mean is over queries with a positive
  ideal.
- **Exact recall@k** for k 10, 100 and 1000 (the search floor's recall), over queries with an Exact product. A list
  shorter than k is not padded: a run of k 100 has recall@1000 equal to its recall@100.
- A judged query with no line in the run scores 0.

On top of that, it reports what is specific to generative retrieval (`score.py`):

| Check | What it is |
|---|---|
| `invalid_share` | Share of the entries in the scored lists that are not a catalogue product. The kinds are: `undecodable` (null), `unknown_sid`, `unknown_product`, `inconsistent` (a sid with a product the map does not give it) and `ambiguous` (a shared sid with no product chosen). **An invalid entry keeps its rank with gain 0**: the slot was spent |
| `duplicate_share` | A product repeated lower in the same list keeps its rank with gain 0, so a repeat cannot add gain |
| `sid_map.full_sid.product_collision_rate` | Share of catalogue products whose sid is shared with another product. Also given on the semantic prefix (`sid_levels` leading codes, without TIGER's disambiguation suffix), with the share of sids that collide and the largest group |
| `shared_sid_hit_share` | Share of the valid generated entries that landed on a shared sid |
| `short_list_share` | Share of queries with fewer than k valid, distinct results, for k 10, 100 and 1000 |
| `sid_map.coverage` | Share of the catalogue the sid map gives a sid. Anything under 1 is never a score of record, because a map that leaves products out shrinks the search space |
| `vs_baselines` | Per-query paired nDCG@10 difference against each baseline: the mean, a 95% bootstrap interval (seed 0, 2,000 resamples), and wins, losses and ties |

## The two files a system hands in

The format is in `runfile.py`. A **run file** is JSONL (it may be gzipped). Its first line is a header, then there is
one line per test query:

```json
{"genret_run": 1, "name": "gr1-dev-3", "dataset": "esci-sample", "split": "test", "k": 100, "sid_levels": 3,
 "trained_on": ["catalogue", "fit"], "system": "what produced it, with its commit"}
{"qid": 12, "ranked": ["B07XYZ", {"sid": "17-401-9-0"}, {"sid": "3-3-3-0", "pid": "B01ABC"}, null]}
```

An entry is one of four things:
- a product id;
- a generated sid, which the referee resolves through the map;
- a sid with the product the system chose for it (needed on a collision);
- `null`, for output the system could not decode.

`k` is at most 1000. A qid that is not one of the dataset's test queries is refused.

A **sid map** gives every catalogue product its sid (codes joined by `-`). It is a `.parquet` file with columns
`product_id` and `sid`, or a `.tsv` / `.tsv.gz` file with that header.

## Training data

A model may be trained and tuned on **the catalogue and the fit split only**, and its run's header says so
(`trained_on`). The test queries' texts are served so that a run can answer them. They are never training data,
labelled or not, because the founder's held-out draw is a subset of them. Training on them could fit the held-out
queries. A run whose `trained_on` names anything else is never a score of record.

## Baselines: the search floor's runs on the same queries

`python -m referee baselines --dataset=…` runs **s3-search's own engine and benchmark at the pin**
(`baselines/s3search_runs.py`) in a separate process inside the pinned checkout, one process per system:

| System | What it is |
|---|---|
| `bm25f` | BM25F over the five text fields with weight 1, k1 1.2, b 0.75 (s3-search's default request) |
| `bm25f-tuned` | The prototype's tuned BM25F (`referee/requests/bm25f-tuned.json` and `.bm25.json` in s3-search): the search floor's lexical bar, 0.304 nDCG@10 on ESCI-US in the prototype |
| `dense-tfidf-svd-128` | Exact dense search with the engine's encoder: TF-IDF reduced to 128 dimensions by SVD over title, brand and color, fitted on the catalogue with numpy |

s3-search scores each run with its own harness. This referee then scores the same file itself, and the two must agree
to 1e-12, or the command fails. `baselines.json` records the s3-search commit and file hashes, each run file's
sha256, both scores, the request and mapping, and the host.

**Development baselines** (esci-sample, 500 test queries, k 100, CPU, s3-search `767397600d89`, made
2026-11-13T23:21:08Z). They are in `baselines/esci-sample/` and are **not scores of record**:

| System | nDCG@10 | Exact recall@10 | Exact recall@100 |
|---|---|---|---|
| bm25f | 0.4753 | 0.4362 | 0.7788 |
| bm25f-tuned | 0.5144 | 0.4766 | 0.7982 |
| dense-tfidf-svd-128 | 0.1732 | 0.1417 | 0.3624 |

The dense bar is weak: it is the engine's v1 encoder, a small CPU model, and not a learned dense retriever. A GR claim
that beats only the dense bar has not beaten the search floor. The bar is `bm25f-tuned`.

**ESCI-US baselines** (all 8,956 test queries, 1,215,854 products, k 1000, Colab L4, s3-search `767397600d89`,
made 2026-11-13T23:56:26Z by the steward). They are **development**: a score of record is the founder's `referee full`
with the held-out draw. The records are in `baselines/esci-us/`: `baselines.json`, `s3search_meta.json`, and
s3-search's own `referee full` of bm25f-tuned at the pin. The run files (146 MB) are not in git. They rebuild
deterministically by `python -m referee baselines --dataset=esci-us --k=1000 --device=cuda`, and their sha256 are in
`baselines.json`.

| System | nDCG@10 | Exact recall@10 | Exact recall@100 | Exact recall@1000 |
|---|---|---|---|---|
| bm25f | 0.2487 | 0.2311 | 0.4834 | 0.7128 |
| bm25f-tuned | **0.3039** | 0.2802 | 0.5338 | 0.7416 |
| dense-tfidf-svd-128 | 0.0323 | 0.0276 | 0.0754 | 0.2018 |

- bm25f-tuned equals s3-search's own bench score of the same run exactly, and s3-search's own `referee full` at the
  pin (0.3039127, contract skipped, so not of record there either). It matches the prototype's 0.304.
- On ESCI-US the TF-IDF+SVD dense bar is close to useless (0.032). The bar for GR1 and GR2 is bm25f-tuned.
- On esci-sample, the cuda runs give the same metrics as the CPU runs, but their files differ (other sha256), most
  likely in the order of tied products. Compare runs made on the same device.

## Commands

| Command | Where | What |
|---|---|---|
| `python -m referee setup [--no-data]` | once | Clone s3-search at the pin into `.cache/s3-search` (or `$GENRET_S3_SEARCH`), fetch ESCI (1.1 GB) into `data/esci` (or `$GENRET_DATA/esci`), build esci-sample and check it against the pinned manifest |
| `python -m referee fast [--run=r.jsonl.gz --sid-map=m.tsv]` | CPU, every pull request, **under a minute** | The pin, the unit tests (planted errors the scorer must catch), and on esci-sample the drift check: the three committed baseline runs are rescored and must equal `baselines.json` and s3-search's own scores. With `--run`, a development score |
| `python -m referee score --run=… [--sid-map=…] [--out=results-dev.json]` | CPU | A development score on esci-sample's 500 test queries, with the paired comparison against each baseline. It is labelled DEVELOPMENT |
| `python -m referee baselines --dataset=esci-sample\|esci-us [--k=100] [--device=cpu]` | CPU (sample), Colab (esci-us) | The search floor's runs, as above. Only the steward regenerates the committed sample baselines |
| `python -m referee full --run=… --sid-map=… [--baselines-dir=…]` | the founder's Colab | **The score of record**: every ESCI-US test query, the founder's held-out draw (`$GENRET_HELD_OUT`), and the search floor's draw too when `$S3_SEARCH_HELD_OUT` is set, beside the baselines on the same queries, to `results.json` |

`full` writes `score_of_record: true` only when all of these hold:
- the held-out draw was scored;
- a sid map was given and covers the whole catalogue;
- `trained_on` is within the catalogue and the fit split;
- this checkout is clean;
- the baselines' files and scores check.

Otherwise every missing condition is listed in `caveats`. `--queries=N` is development mode and never writes
`results.json`.

Measured on 2026-11-13: `referee fast` takes 7 to 9 s on the steward's Mac (8 cores; it used about one core). That
includes the 23 tests and the drift check of the three baselines. Without esci-sample (CI), the drift check is
skipped and it takes about 7 s. Building the three sample baselines takes about
90 s.

## The held-out queries

The founder draws them and keeps them outside every repository: 2,000 of the 8,956 ESCI-US test queries, with a
private seed, saved to his Drive. The procedure and the manifest are in the lab's `lab/held-out/genret-esci.md`.
`full` reads the draw from `$GENRET_HELD_OUT` in the founder's Colab and scores the run and every baseline on it.
Members never see which queries it holds. Members never look for it.

## Moving the pin

Only the steward moves the pin, by pull request, with the reason in the commit. The steps:
1. Update `pin.json` (the commit and the file hashes).
2. Regenerate `baselines/esci-sample/` at the new commit.
3. State any change in the baseline numbers in the pull request.

A run scored at one pin is compared only with baselines made at the same pin.
