"""Lightweight unit tests that don't require downloading any HF models.

These run in CI. The heavier `test_smoke.py` (which actually loads
sentence-transformers + a causal LM) is intended for local use.
"""

from __future__ import annotations

import numpy as np


def test_catalog_dot_product_topk():
    """Catalog is independent of the embedder; we can feed it fake vectors."""
    from app.catalog import Catalog, Item

    items = [Item(id=f"i{i}", title=f"t{i}", tags=[], text="x") for i in range(5)]
    # 5 items, 4-d embeddings, orthogonal so we know the answer
    emb = np.eye(5, 4, dtype=np.float32)  # rows 0-3 = e0..e3, row 4 = zeros
    cat = Catalog(items, emb)

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    top = cat.candidates(q, 3)
    # The closest item is i0 (matches q exactly).
    assert top[0][0].id == "i0"
    assert top[0][1] > 0.99
    assert len(top) == 3


def test_catalog_normalizes_at_construction():
    """Catalog should L2-normalize its embedding matrix once at __init__."""
    from app.catalog import Catalog, Item

    items = [Item(id="a", title="a", tags=[], text="")]
    emb = np.array([[3.0, 4.0]], dtype=np.float32)  # norm=5
    cat = Catalog(items, emb)
    assert abs(np.linalg.norm(cat.emb[0]) - 1.0) < 1e-5


def test_prompt_template_contains_fields():
    """Read the scorer source as text and check the template includes all
    placeholders + a yes/no instruction. Avoids importing torch in CI."""
    from pathlib import Path

    src = Path("app/scorer.py").read_text()
    for token in ("{profile}", "{title}", "{tags}", "{desc}", "yes", "no"):
        assert token in src, f"missing {token!r} in scorer.py prompt template"
