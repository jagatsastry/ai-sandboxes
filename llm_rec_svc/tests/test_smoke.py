"""Offline smoke tests: catalog + scorer work without network at request time
(assuming the HF models have already been downloaded once)."""
from __future__ import annotations

import json
import os
from pathlib import Path


def test_catalog_loads_and_ranks(tmp_path):
    from sentence_transformers import SentenceTransformer
    from app.catalog import Catalog

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cat = Catalog.from_json("data/items.json", emb)
    assert len(cat) > 0
    q = emb.encode(["distributed systems and ranking"], convert_to_numpy=True)[0]
    cands = cat.candidates(q, 5)
    assert len(cands) == 5
    titles = [it.title for it, _ in cands]
    # at least one obviously-relevant item should land in top-5
    assert any(k in " ".join(titles).lower()
               for k in ["distributed", "ranking", "search", "data-intensive"])


def test_scorer_batches():
    from app.scorer import LLMScorer

    s = LLMScorer("sshleifer/tiny-gpt2")
    rows = [
        ("backend engineer", "Designing Data-Intensive Applications",
         ["systems"], "distributed systems book"),
        ("backend engineer", "The Manga Guide to Databases",
         ["beginner"], "intro to databases"),
    ]
    scores, t = s.score_batch(rows)
    assert len(scores) == 2
    assert t["gpu_ms"] >= 0.0
