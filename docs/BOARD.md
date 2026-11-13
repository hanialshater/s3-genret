# The board

Floor 3's work items, by milestone. A role claims an item by putting its name and the date (UTC) in the Claimed
column, in a commit, before it touches the area (P4). It releases an item by clearing that column. The predictions
are in the lab: `company/milestones.json` and `lab/predictions/`.

Status values: open · claimed · in review · done.

A maker who thinks a referee check is wrong adds a row under **Referee questions**. It does not edit `referee/`.

## G0 · The floor set up (evaluation-steward)

| Id | Item | Status | Claimed |
|---|---|---|---|
| G0-1 | `referee/`: ESCI through s3-search's loader and metrics pinned to a commit; the run-file and sid-map formats; nDCG@10 (ESCI gains) and Exact recall@k; invalid share, duplicates, collision rate, short lists, coverage; paired comparison with the baselines | in review | evaluation-steward, 2026-11-13 |
| G0-2 | The baselines: s3-search's `bm25f`, `bm25f-tuned` and `dense-tfidf-svd-128` on the same test queries, with the commit and checksums, scored by both floors' scorers (development: esci-sample) | in review | evaluation-steward, 2026-11-13 |
| G0-3 | `referee fast` (CPU, under 1 minute), and `referee full` with `colab_referee.ipynb` (the score of record; not yet run in Colab) | in review | evaluation-steward, 2026-11-13 |
| G0-4 | The held-out procedure and the manifest stub (the lab's `lab/held-out/genret-esci.md`); the founder draws the queries | in review | evaluation-steward, 2026-11-13 |
| G0-5 | Bootstrap: `CLAUDE.md`, this board, `.claude/agents/` (researcher, evaluation-steward, reproducer), CI | in review | evaluation-steward, 2026-11-13 |
| G0-6 | The founder's draw of the held-out queries (`genret-esci-v1`) | open | (founder) |
| G0-7 | The first Colab run: the ESCI-US baselines at k 1000 and prediction G0-steward-2 | open | (founder) |

## GR1 · TIGER-style (researcher)

| Id | Item | Status | Claimed |
|---|---|---|---|
| GR1-1 | Predictions in the lab's `lab/predictions/genret-GR1.md`, including what the stand-ins are on CPU | in review | researcher, 2026-11-13 |
| GR1-2 | Product-content embeddings and an RQ-VAE tokeniser; a sid map of the whole catalogue | claimed | researcher, 2026-11-13 |
| GR1-3 | A seq2seq query-to-sid model trained on the fit split; constrained beam search over the sids that exist | claimed | researcher, 2026-11-13 |
| GR1-4 | A development score on esci-sample (`python -m referee score`), and the command that writes the run file and sid map from a clean checkout | claimed | researcher, 2026-11-13 |
| GR1-5 | The Colab score of record (founder), then the reproduction (reproducer) | open | |

## GR2 · SID-LLM-style (researcher)

| Id | Item | Status | Claimed |
|---|---|---|---|
| GR2-1 | Predictions in the lab's `lab/predictions/genret-GR2.md`, naming the ESCI signal that stands in for behavioural data | in review | researcher, 2026-11-13 |
| GR2-2 | 3 × 8,192 semantic ids; joint catalogue pre-training; multi-token prediction and a lightweight ranker | claimed | researcher, 2026-11-13 |

## Referee questions

| Id | Raised by | Question | Steward's answer |
|---|---|---|---|
