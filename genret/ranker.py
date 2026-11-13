"""GR2's lightweight ranker: a small public cross-encoder fine-tuned on the fit judgments with ESCI's graded gains,
standing in for SID-LLM's judge-distilled 350M ranker. It scores every product of every candidate id, so it also
resolves ids shared by several products (the paper resolves those by a static quality score, which ESCI lacks).
"""
from __future__ import annotations

import numpy as np

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def fit_examples(fit, product_text: dict, holdout_frac: float = 0.1, seed: int = 0):
    """(train, tune) InputExamples from the fit split; the tune slice is fit queries, never test queries."""
    from sentence_transformers import InputExample
    rng = np.random.default_rng(seed)
    qids = np.array(sorted(fit["query_id"].unique()))
    tune = set(rng.choice(qids, size=max(1, int(len(qids) * holdout_frac)), replace=False).tolist())
    tr, tu = [], []
    for q, qt, p, l in zip(fit["query_id"], fit["query"], fit["product_id"], fit["esci_label"]):
        ex = InputExample(texts=[qt, product_text.get(str(p), "")], label=GAIN[l])
        (tu if q in tune else tr).append(ex)
    return tr, tu


def train_ranker(train, epochs: int = 2, batch: int = 64, base: str = BASE, device=None):
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader
    m = CrossEncoder(base, num_labels=1, max_length=192, device=device)
    m.fit(train_dataloader=DataLoader(train, shuffle=True, batch_size=batch), epochs=epochs, warmup_steps=500,
          show_progress_bar=False)
    return m


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min() if len(x) else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def combine(ranker_scores, gen_scores, w_ranker: float = 0.7):
    """The served order: w x ranker + (1 - w) x generator score, each min-max normalised per query."""
    return w_ranker * minmax(ranker_scores) + (1 - w_ranker) * minmax(gen_scores)
