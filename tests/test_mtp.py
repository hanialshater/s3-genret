import numpy as np
import torch

from genret.beam import NEG
from genret.mtp import MTPHeads, last_hidden, mtp_decode
from genret.sids import SidTrie


def test_mtp_heads_learn_and_decode_real_ids():
    torch.manual_seed(0)
    codes = np.array([[0, 1, 2], [1, 1, 0], [2, 0, 1], [3, 2, 2]])
    trie = SidTrie(codes, list("abcd"))
    heads = MTPHeads(16, [4, 3, 3])
    h = torch.randn(4, 16)
    opt = torch.optim.Adam(heads.parameters(), lr=0.05)
    y = torch.as_tensor(codes)
    for _ in range(100):
        opt.zero_grad()
        loss = heads.loss(h, y)
        loss.backward()
        opt.step()
    assert float(loss) < 0.1
    out, sc = mtp_decode(heads, h, trie, width=3)
    for b in range(4):
        assert tuple(out[b, 0].tolist()) == tuple(codes[b])        # the learnt id comes first
        assert all(tuple(r.tolist()) in trie for r, s in zip(out[b], sc[b]) if s > NEG / 2)


def test_last_hidden_takes_the_last_real_token():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1)
    lm = Qwen2ForCausalLM(cfg).eval()
    ids = torch.tensor([[5, 6, 7, 0], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    h = last_hidden(lm, ids, mask)
    full = lm(input_ids=ids[1:, :2], output_hidden_states=True).hidden_states[-1][0, -1]
    assert torch.allclose(h[1], full, atol=1e-5)
