"""Content embeddings of the catalogue with a public pretrained sentence encoder.

GR1 (TIGER): sentence-transformers/sentence-t5-base (768-d), the encoder TIGER used (Sentence-T5).
GR2 (SID-LLM): intfloat/e5-base-v2 (768-d), a public retrieval-trained encoder standing in for Shopify's
in-house product-search embedding. Passages get e5's "passage: " prefix.
CPU development uses sentence-transformers/all-MiniLM-L6-v2 (384-d) for speed; never for a claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PREFIX = {"intfloat/e5-base-v2": "passage: ", "intfloat/e5-small-v2": "passage: "}


def embed_texts(texts, model_name: str, batch_size: int = 128, max_length: int = 128, device=None,
                normalize: bool = True) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    if str(model.device).startswith("cuda"):
        model.half()                       # fp16 on GPU: about 2x faster, cosine-level differences only
    pre = PREFIX.get(model_name, "")
    if pre:
        texts = [pre + t for t in texts]
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
                       normalize_embeddings=normalize)
    return emb.astype(np.float32)


def cached_embeddings(texts, ids, model_name: str, cache_dir, **kw) -> np.ndarray:
    """Embeds once per (model, texts) and caches as .npy, keyed by a hash of the ids and texts."""
    h = hashlib.sha256(model_name.encode())
    for i, t in zip(ids, texts):
        h.update(str(i).encode() + b"\x1f" + t.encode() + b"\x1e")
    key = h.hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"emb-{model_name.replace('/', '_')}-{key}.npy"
    if path.exists():
        return np.load(path)
    emb = embed_texts(texts, model_name, **kw)
    np.save(path, emb)
    (path.with_suffix(".json")).write_text(json.dumps({"model": model_name, "n": len(ids), "dim": int(emb.shape[1]),
                                                       "key": key}) + "\n")
    return emb
