"""GR2 plumbing on a tiny random Qwen2 (no download): stages train, MTP and AR decoding return real ids only."""
import numpy as np
import pandas as pd
import torch

from genret import gr2
from genret.mtp import MTPHeads
from genret.sids import SidTrie


def _enc(t):
    return [min(b, 250) + 2 for b in t.encode()][:40]


def _setup():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    lm = Qwen2ForCausalLM(cfg)
    ks = [5, 5, 5]
    v = gr2.add_sid_vocab(lm, 256, ks)
    ids = [f"p{i}" for i in range(10)]
    codes = np.random.default_rng(0).integers(0, 5, (10, 3))
    cat = pd.DataFrame({"product_id": ids, "product_title": [f"thing {i}" for i in range(10)],
                        "product_brand": ["b"] * 10, "product_color": [""] * 10})
    fit = pd.DataFrame({"query_id": [i // 3 for i in range(10)], "query": [f"want {i}" for i in range(10)],
                        "product_id": ids, "esci_label": ["E"] * 10})
    return lm, v, ks, ids, codes, cat, fit


def test_stages_and_decoding():
    lm, v, ks, ids, codes, cat, fit = _setup()
    ex = gr2.Examples(_enc, v, cat, codes, ids, fit, eos=1)
    stage1 = [ex.attr_to_sid(i) for i in range(10)] + [ex.sid_to_title(i) for i in range(10)]
    before = lm.get_input_embeddings().weight.detach().clone()
    h1 = gr2.train_lm(lm, stage1, steps=10, batch=4, lr=1e-2, device="cpu", pad=0, warmup=2,
                      only_new_rows_from=256, log=lambda *_: None, log_every=5)
    after = lm.get_input_embeddings().weight.detach()
    assert torch.equal(before[:256], after[:256]) and not torch.equal(before[256:], after[256:])
    stage2 = stage1 + ex.query_pairs(0) + ex.cojudged(0)
    h2 = gr2.train_lm(lm, stage2, steps=10, batch=4, lr=1e-3, device="cpu", pad=0, warmup=2,
                      log=lambda *_: None, log_every=5)
    assert len(h1) and len(h2)
    heads = MTPHeads(32, ks)
    pairs = lambda e: [(q, codes[ids.index(p)]) for q, p in zip(fit["query"], fit["product_id"])]  # noqa: E731
    gr2.train_mtp(lm, heads, _enc, v, pairs, steps=5, batch=4, device="cpu", pad=0, log=lambda *_: None)
    trie = SidTrie(codes, ids)
    for res in (gr2.decode_mtp(lm, heads, _enc, v, ["want 1", "want 2"], trie, width=6, device="cpu", pad=0),
                gr2.decode_ar(lm, _enc, v, ["want 1", "want 2"], trie, width=6, device="cpu", pad=0, batch=2)):
        assert len(res) == 2
        for r in res:
            assert r and all(c in trie for c, _ in r)
            cands = gr2.expand(r, trie, {p: i for i, p in enumerate(ids)}, cap=20)
            ent = gr2.ranked_entries(cands, 5)
            assert all(e["pid"] in ids for e in ent)
