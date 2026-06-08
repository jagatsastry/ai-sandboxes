"""FastAPI service: two-stage rec with LLM reranker.

Pipeline per request:
    1. embed user profile string                 -> query vector
    2. candidate gen (cosine top-N over catalog) -> N items
    3. LLM rerank (batched, GPU if available)    -> reordered top-K
    4. final blend = w * llm + (1-w) * retrieval

Concurrency: when BATCH_ENABLE=1 (the default), an in-process batcher
coalesces concurrent /recommend calls into a single forward pass on the
LLM, flattening the latency-vs-concurrency curve and lifting throughput
without adding model replicas. Set BATCH_ENABLE=0 to fall back to the
legacy one-request-at-a-time path (the bench uses this to A/B the two
regimes).

All stages timed; timings returned in the response and rolled up in /metrics.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .batcher import BatchingScorer
from .catalog import Catalog
from .scorer import LLMScorer


# ---------------- config ----------------
DATA_PATH = Path(os.getenv("ITEMS_PATH", "data/items.json"))
EMBEDDER_NAME = os.getenv("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")
LLM_NAME = os.getenv("LLM", "sshleifer/tiny-gpt2")
DEFAULT_CANDIDATES = int(os.getenv("N_CANDIDATES", "20"))
DEFAULT_TOPK = int(os.getenv("TOPK", "5"))
LLM_WEIGHT = float(os.getenv("LLM_WEIGHT", "0.7"))
# Batching knobs. BATCH_ENABLE=0 forces legacy per-request scoring.
BATCH_ENABLE = os.getenv("BATCH_ENABLE", "1") not in ("0", "false", "False")
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "64"))
BATCH_MAX_WAIT_MS = float(os.getenv("BATCH_MAX_WAIT_MS", "5.0"))


# ---------------- API models ----------------
class RecRequest(BaseModel):
    profile: str = Field(..., description="Free-text user profile / context")
    top_k: int = Field(DEFAULT_TOPK, ge=1, le=50)
    n_candidates: int = Field(DEFAULT_CANDIDATES, ge=1, le=500)
    explain: bool = False


class RecItem(BaseModel):
    id: str
    title: str
    tags: List[str]
    final_score: float
    llm_score: float
    retrieval_score: float


class Timings(BaseModel):
    embed_ms: float
    candidate_ms: float
    tokenize_ms: float
    gpu_ms: float
    total_ms: float
    batch_size: int = 0
    queue_ms: float = 0.0


class RecResponse(BaseModel):
    items: List[RecItem]
    timings: Timings
    device: str
    n_candidates: int


# ---------------- global state ----------------
class State:
    embedder: Optional[SentenceTransformer] = None
    catalog: Optional[Catalog] = None
    scorer: Optional[LLMScorer] = None
    batcher: Optional[BatchingScorer] = None
    metrics_lock = Lock()
    metrics = {
        "requests": 0,
        "sum_total_ms": 0.0,
        "sum_gpu_ms": 0.0,
        "sum_embed_ms": 0.0,
        "sum_candidate_ms": 0.0,
        "sum_queue_ms": 0.0,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[boot] loading embedder: {EMBEDDER_NAME}")
    State.embedder = SentenceTransformer(EMBEDDER_NAME)
    print(f"[boot] loading catalog: {DATA_PATH}")
    State.catalog = Catalog.from_json(DATA_PATH, State.embedder)
    print(f"[boot] loading LLM: {LLM_NAME}")
    State.scorer = LLMScorer(LLM_NAME)
    print(f"[boot] device: {State.scorer.device}  items: {len(State.catalog)}")
    print("[boot] warmup...")
    State.scorer.warmup()
    if BATCH_ENABLE:
        print(f"[boot] starting batcher: max_size={BATCH_MAX_SIZE} "
              f"max_wait_ms={BATCH_MAX_WAIT_MS}")
        State.batcher = BatchingScorer(
            score_fn=State.scorer.score_batch,
            max_batch_size=BATCH_MAX_SIZE,
            max_wait_ms=BATCH_MAX_WAIT_MS,
        )
        await State.batcher.start()
    print("[boot] ready")
    yield
    if State.batcher is not None:
        await State.batcher.stop()
    print("[shutdown]")


app = FastAPI(title="llm-rec-svc", version="0.1.0", lifespan=lifespan)


# ---------------- routes ----------------
@app.get("/healthz")
def healthz():
    ready = State.scorer is not None and State.catalog is not None
    return {
        "ready": ready,
        "device": State.scorer.device if State.scorer else None,
        "items": len(State.catalog) if State.catalog else 0,
        "cuda_available": torch.cuda.is_available(),
        "batch_enabled": State.batcher is not None,
    }


@app.get("/metrics")
def metrics():
    with State.metrics_lock:
        m = dict(State.metrics)
    n = max(m["requests"], 1)
    out = {
        "requests": m["requests"],
        "avg_total_ms": m["sum_total_ms"] / n,
        "avg_gpu_ms": m["sum_gpu_ms"] / n,
        "avg_embed_ms": m["sum_embed_ms"] / n,
        "avg_candidate_ms": m["sum_candidate_ms"] / n,
        "avg_queue_ms": m["sum_queue_ms"] / n,
        "batch_enabled": State.batcher is not None,
    }
    if State.batcher is not None:
        out["batcher"] = State.batcher.stats.snapshot()
    return out


@app.post("/recommend", response_model=RecResponse)
async def recommend(req: RecRequest):
    assert State.embedder and State.catalog and State.scorer
    t0 = time.perf_counter()

    # 1) embed profile (CPU; run on a thread so we don't hog the event
    # loop and prevent other coroutines from reaching the batcher).
    def _embed():
        return State.embedder.encode(
            [req.profile], convert_to_numpy=True, show_progress_bar=False,
        )[0]
    q = await asyncio.to_thread(_embed)
    t1 = time.perf_counter()

    # 2) candidate generation (numpy dot+argpartition; offload so a slow
    # catalog scan can't starve the event loop or block the batcher drain).
    cands = await asyncio.to_thread(State.catalog.candidates, q, req.n_candidates)
    t2 = time.perf_counter()

    # 3) LLM rerank.
    # Path A (batched): submit rows to the cross-request batcher; many
    #                   concurrent /recommend calls coalesce into ONE
    #                   tokenize+forward pass.
    # Path B (legacy):  run score_batch on a thread so we don't block the
    #                   event loop; no cross-request batching.
    rows = [(req.profile, it.title, it.tags, it.text) for it, _ in cands]
    if State.batcher is not None:
        llm_scores, ttimes = await State.batcher.score(rows)
    else:
        llm_scores, ttimes = await asyncio.to_thread(
            State.scorer.score_batch, rows
        )
        ttimes = {**ttimes, "batch_size": len(rows), "queue_ms": 0.0}

    # 4) blend & sort
    # normalize llm scores to [0,1] within the batch (stable, scale-free blend)
    if llm_scores:
        lo, hi = min(llm_scores), max(llm_scores)
        rng = (hi - lo) or 1.0
        llm_norm = [(s - lo) / rng for s in llm_scores]
    else:
        llm_norm = []

    scored = []
    for (it, ret), raw_llm, n_llm in zip(cands, llm_scores, llm_norm):
        # retrieval score is cosine in [-1,1]; map to [0,1]
        ret_n = (ret + 1.0) / 2.0
        final = LLM_WEIGHT * n_llm + (1.0 - LLM_WEIGHT) * ret_n
        scored.append((it, raw_llm, ret, final))

    scored.sort(key=lambda x: -x[3])
    top = scored[: req.top_k]
    t3 = time.perf_counter()

    timings = Timings(
        embed_ms=(t1 - t0) * 1000.0,
        candidate_ms=(t2 - t1) * 1000.0,
        tokenize_ms=ttimes["tokenize_ms"],
        gpu_ms=ttimes["gpu_ms"],
        total_ms=(t3 - t0) * 1000.0,
        batch_size=int(ttimes.get("batch_size", len(rows))),
        queue_ms=float(ttimes.get("queue_ms", 0.0)),
    )

    with State.metrics_lock:
        State.metrics["requests"] += 1
        State.metrics["sum_total_ms"] += timings.total_ms
        State.metrics["sum_gpu_ms"] += timings.gpu_ms
        State.metrics["sum_embed_ms"] += timings.embed_ms
        State.metrics["sum_candidate_ms"] += timings.candidate_ms
        State.metrics["sum_queue_ms"] += timings.queue_ms

    return RecResponse(
        items=[
            RecItem(
                id=it.id, title=it.title, tags=it.tags,
                final_score=final, llm_score=llm, retrieval_score=ret,
            )
            for (it, llm, ret, final) in top
        ],
        timings=timings,
        device=State.scorer.device,
        n_candidates=len(cands),
    )
