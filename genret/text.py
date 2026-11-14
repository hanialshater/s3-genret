"""The text of a product that its content embedding is computed from.

ESCI has title, brand, colour, bullet points and description (no price or category, which TIGER used). We use
title | brand | colour | the start of the bullet points, truncated by the encoder's own max length.
"""
from __future__ import annotations

import pandas as pd

BULLET_CHARS = 300


def _clean(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return " ".join(str(s).split())


def product_text(row) -> str:
    parts = [_clean(row.get("product_title"))]
    for field, label in (("product_brand", "brand"), ("product_color", "color")):
        v = _clean(row.get(field))
        if v:
            parts.append(f"{label}: {v}")
    b = _clean(row.get("product_bullet_point"))[:BULLET_CHARS]
    if b:
        parts.append(b)
    return " | ".join(p for p in parts if p)


def catalogue_texts(catalogue: pd.DataFrame) -> list:
    return [product_text(r) for r in catalogue.to_dict("records")]
