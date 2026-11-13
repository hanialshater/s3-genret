"""GR1 model plumbing on a tiny random T5 (no download): training lowers the loss, decoding returns real ids only."""
import numpy as np
import pandas as pd
import torch

from genret import gr1
from genret.sids import SidTrie, dedup_suffix


class _Enc(dict):
    __getattr__ = dict.__getitem__

    def to(self, d):
        return _Enc({k: v.to(d) for k, v in self.items()})


class ByteTok:
    def __call__(self, texts, max_length=64, truncation=True, padding=True, return_tensors="pt"):
        ids = [[min(b, 250) + 2 for b in t.encode()[:max_length]] + [1] for t in texts]
        n = max(map(len, ids))
        x = torch.tensor([r + [0] * (n - len(r)) for r in ids])
        return _Enc(input_ids=x, attention_mask=(x != 0).long())


def _tiny(ks):
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=256, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=2,
                   decoder_start_token_id=0, pad_token_id=0)
    m = T5ForConditionalGeneration(cfg)
    v = gr1.SidVocab(ks, 256)
    m.resize_token_embeddings(256 + v.size + 1)
    return m, v


def test_gr1_trains_and_decodes_only_real_ids():
    torch.manual_seed(0)
    ids = [f"p{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    codes = dedup_suffix(rng.integers(0, 4, (12, 3)), ids)
    ks = [4, 4, 4, int(codes[:, 3].max()) + 1]
    cat = pd.DataFrame({"product_id": ids, "product_title": [f"thing {i}" for i in range(12)],
                        "product_brand": ["b"] * 12})
    fit = pd.DataFrame({"query": [f"want {i}" for i in range(12)], "product_id": ids, "esci_label": ["E"] * 12})
    m, v = _tiny(ks)
    hist = gr1.train(m, ByteTok(), v, lambda e: gr1.build_examples(fit, cat, codes, ids, e), steps=40, batch=8,
                     lr=3e-3, warmup=5, log_every=20, log=lambda *_: None)
    assert hist[-1]["loss"] < hist[0]["loss"] + 1.0
    trie = SidTrie(codes, ids)
    res = gr1.retrieve(m, ByteTok(), v, ["want 1", "want 2", "want 3"], trie, width=6, batch=2)
    assert len(res) == 3
    for r in res:
        assert 1 <= len(r) <= 6 and all(c in trie for c, _ in r)
        ranked = gr1.to_ranked(r, trie)
        assert all(isinstance(e, str) and e in ids for e in ranked)
    free = gr1.retrieve(m, ByteTok(), v, ["want 1"], trie, width=6, batch=1, unconstrained=True)
    assert len(free[0]) == 6
