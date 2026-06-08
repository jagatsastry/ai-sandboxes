"""Item catalog + dense candidate generator.

We embed each item once at startup with a small sentence-transformer.
Candidate generation = cosine top-N over a numpy matrix. For a real system
this would be FAISS / ScaNN / a vector DB, but for a sandbox numpy is
plenty fast at <10k items and keeps the dependency footprint small.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Item:
    id: str
    title: str
    tags: list[str]
    text: str

    def as_doc(self) -> str:
        return f"{self.title}. Tags: {', '.join(self.tags)}. {self.text}"


class Catalog:
    def __init__(self, items: Sequence[Item], embeddings: np.ndarray) -> None:
        assert len(items) == embeddings.shape[0]
        self.items = list(items)
        # L2-normalize once so cosine = dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self.emb = (embeddings / norms).astype(np.float32)
        self._by_id = {it.id: i for i, it in enumerate(self.items)}

    @classmethod
    def from_json(cls, path: str | Path, embedder) -> Catalog:
        raw = json.loads(Path(path).read_text())
        items = [Item(**r) for r in raw]
        docs = [it.as_doc() for it in items]
        emb = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        return cls(items, emb)

    def candidates(self, query_vec: np.ndarray, n: int) -> list[tuple[Item, float]]:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        scores = self.emb @ q.astype(np.float32)
        n = min(n, len(self.items))
        # argpartition for top-n, then sort just those
        idx = np.argpartition(-scores, n - 1)[:n]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.items[i], float(scores[i])) for i in idx]

    def get(self, item_id: str) -> Item:
        return self.items[self._by_id[item_id]]

    def __len__(self) -> int:
        return len(self.items)
