# S3 GenRet: rules for every session

This is the repository of **Floor 3 · Generative retrieval** in Shakabuku Labs (System 3 city). It is a research
floor. It asks whether a model that *generates* a product's identifier from the query can match or beat the search
floor's BM25F and dense retrieval on ESCI-US. The founder decided it on 2026-11-13 (the lab's
`company/decisions/2026-11-13-genret-floor.md`).

The floor's records are in `hanialshater/shakabuku-labs`, checked out beside this repository:
- the milestones G0, GR1 and GR2, in `company/milestones.json`;
- the asks, in `company/asks.json`;
- the predictions, in `lab/predictions/`;
- the reproductions, in `lab/reproductions/`;
- the held-out manifest, in `lab/held-out/`;
- the lessons, in `company/knowledge/entries.jsonl`.

The lab's `CLAUDE.md` applies here too. **The disk is the memory.**

## The crew

The roles are in `.claude/agents/` and run as subagents. The session the founder talks to coordinates and keeps the
records. It does not build (P11).

| Role | Owns | Never |
|---|---|---|
| **researcher** (the maker) | the semantic-ID tokenisers, the generative models, the experiments; everything outside `referee/` and the bootstrap files | edits `referee/`, sees the held-out queries, checks its own claim |
| **evaluation-steward** | `referee/` (the contract, the pinned data path, the baselines), the held-out procedure, this file, `.github/` | builds what it judges |
| **reproducer** | reruns each headline claim from a clean checkout on the held-out queries, before reading the researcher's account | reproduces its own work |

The founder holds the held-out queries and runs the Colab score of record.

## Start of session

1. Read the floor's milestones and asks in the lab repository. If you were given a role, read its file in
   `.claude/agents/`.
2. Claim an item in `docs/BOARD.md`, in a commit, before you touch its area (P4).
3. Run `python -m referee fast` before you change anything, and record the result. Run `python -m referee setup` once
   first. It clones s3-search at the pinned commit, fetches ESCI and builds esci-sample.

## Hard rules

- **Clean room.** Public data and the founder's own work only. Nothing from any employer or client: no code, data,
  documents or numbers. Public papers (TIGER, the Shopify SID paper) are read and cited, not copied.
- **Data access.** ESCI is read only through `referee.data`, which is the search floor's loader at the commit in
  `referee/pin.json`. Never load the test judgments, and never look for the founder's held-out draw.
- **Training data.** Models are trained and tuned on the catalogue and the fit split only. The test queries' texts
  are never training data, labelled or not, because the held-out draw is a subset of them. Every run file's header
  says what it was trained on (`trained_on`).
- **The referee.** Only the evaluation steward changes `referee/` (P6). A maker who thinks a check is wrong raises it
  on the board and does not edit it. CI refuses a change to `referee/` or this file from a branch not named
  `steward/…`.
- **Land past the referee.** A change lands on `main` only by pull request with `python -m referee fast` green. A
  claim of a quality number needs the Colab score of record (`referee/colab_referee.ipynb`), which the founder runs.
  CPU and esci-sample numbers are labelled development, always.
- **Predictions before runs.** The researcher commits each claim's predictions, with wrong-if and tolerance, in the
  lab's `lab/predictions/genret-GR1.md` (or `-GR2.md`) before its run. A prediction is never edited after its run
  starts.
- **No headline before it reproduces.** No README, paper, deck or post states a headline the reproducer has not
  reproduced.
- **No spending without approval.** Spending needs a budget line the founder has approved. The cost target is $25 per
  experiment, pre-registered, run and reproduced. Work over $50 pauses for the founder. GPU work runs in the founder's
  Colab (an L4 by default, see the lab's `colab-gpu-run` skill). No paid API or model service.
- **Development on CPU.** Use small models and esci-sample. `python -m referee fast` stays under 1 minute.
- **Nothing large in git.** Data, checkpoints and run outputs stay out (`.gitignore`). A claim's run file is rebuilt
  from its commit by its command.
- **Public.** This repository is public (the founder, 2026-11-13). It holds only code and results. Predictions,
  reproductions, transcripts and the held-out manifest stay in the private lab repository. Nothing here states a
  headline before it reproduces.
- **No secrets and no personal data.**
- **Times** are UTC, from `date -u`.

## Before you finish

- Run `python -m referee fast` and record the result in the pull request.
- Write the result against its predictions, wrong ones included, in the lab's `company/milestones.json`.
- Update `docs/BOARD.md`, and add a lesson to the lab's `company/knowledge/entries.jsonl` if the next session should
  know it.
