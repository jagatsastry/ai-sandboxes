"""Tests for the cross-request micro-batching layer.

These don't load any HF model -- we stub `score_fn` so the test is
deterministic and fast.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence

import pytest
from app.batcher import BatchingScorer


class _FakeScorer:
    """Counts calls and the largest batch ever passed to it."""

    def __init__(self, per_row_ms: float = 5.0):
        self.calls = 0
        self.max_batch = 0
        self.batch_sizes: list[int] = []
        self.per_row_ms = per_row_ms
        self._lock = threading.Lock()

    def __call__(self, rows: Sequence[tuple[str, str, list, str]]):
        with self._lock:
            self.calls += 1
            self.batch_sizes.append(len(rows))
            self.max_batch = max(self.max_batch, len(rows))
        # Fixed per-call cost regardless of batch size (mimics the
        # dominant tokenizer+framework dispatch overhead the real path has).
        time.sleep(self.per_row_ms / 1000.0)
        scores = [float(len(r[1])) for r in rows]  # deterministic
        return scores, {"tokenize_ms": 1.0, "gpu_ms": 2.0}


def _row(i: int):
    return (f"profile-{i}", f"title-{i}", [], f"desc-{i}")


@pytest.mark.asyncio
async def test_single_caller_returns_correct_slice():
    fake = _FakeScorer()
    bs = BatchingScorer(fake, max_batch_size=8, max_wait_ms=2.0)
    await bs.start()
    try:
        scores, t = await bs.score([_row(1), _row(2)])
        assert len(scores) == 2
        assert t["batch_size"] == 2
        assert "queue_ms" in t
        assert fake.calls == 1
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_concurrent_calls_coalesce_into_one_batch():
    """Fire 8 concurrent score() calls; they should land in <= 2 batches
    (ideally 1) given a generous wait window."""
    fake = _FakeScorer(per_row_ms=10.0)
    bs = BatchingScorer(fake, max_batch_size=64, max_wait_ms=20.0)
    await bs.start()
    try:
        coros = [bs.score([_row(i), _row(i + 100)]) for i in range(8)]
        results = await asyncio.gather(*coros)
        # All callers got their own 2-row slice
        for scores, _t in results:
            assert len(scores) == 2
        # The whole point: dramatically fewer model calls than callers.
        assert fake.calls <= 3
        assert fake.max_batch >= 4  # at least half coalesced
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_respects_max_batch_size():
    fake = _FakeScorer(per_row_ms=2.0)
    bs = BatchingScorer(fake, max_batch_size=4, max_wait_ms=50.0)
    await bs.start()
    try:
        # 6 concurrent callers, 1 row each = 6 rows. Cap is 4 -> must split.
        coros = [bs.score([_row(i)]) for i in range(6)]
        results = await asyncio.gather(*coros)
        assert all(len(s) == 1 for s, _ in results)
        assert fake.max_batch <= 4
        assert fake.calls >= 2
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_does_not_split_a_caller_across_batches():
    """If a caller's rows would overflow the current batch, the caller is
    deferred to the next batch -- never sliced across forward passes."""
    fake = _FakeScorer(per_row_ms=2.0)
    bs = BatchingScorer(fake, max_batch_size=4, max_wait_ms=50.0)
    await bs.start()
    try:
        # Caller A submits 3 rows first; caller B then submits 3 rows.
        # Cap=4 means B can't piggyback; B must wait its own batch.
        a = asyncio.create_task(bs.score([_row(i) for i in range(3)]))
        await asyncio.sleep(0.001)
        b = asyncio.create_task(bs.score([_row(i + 10) for i in range(3)]))
        a_scores, _ = await a
        b_scores, _ = await b
        assert len(a_scores) == 3
        assert len(b_scores) == 3
        # Two separate forward passes (since 3+3 > 4)
        assert fake.calls == 2
        assert all(s <= 4 for s in fake.batch_sizes)
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_exception_propagates_to_all_callers_in_batch():
    class Boom:
        def __call__(self, rows):
            raise RuntimeError("model exploded")

    bs = BatchingScorer(Boom(), max_batch_size=8, max_wait_ms=5.0)
    await bs.start()
    try:
        coros = [bs.score([_row(i)]) for i in range(4)]
        results = await asyncio.gather(*coros, return_exceptions=True)
        assert len(results) == 4
        assert all(isinstance(r, RuntimeError) for r in results)
        assert all("model exploded" in str(r) for r in results)
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_empty_rows_short_circuits():
    fake = _FakeScorer()
    bs = BatchingScorer(fake, max_batch_size=8, max_wait_ms=5.0)
    await bs.start()
    try:
        scores, t = await bs.score([])
        assert scores == []
        assert t["batch_size"] == 0
        # Crucially: did NOT call the underlying model.
        assert fake.calls == 0
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_stats_snapshot_is_well_formed():
    fake = _FakeScorer()
    bs = BatchingScorer(fake, max_batch_size=8, max_wait_ms=10.0)
    await bs.start()
    try:
        await asyncio.gather(*[bs.score([_row(i)]) for i in range(5)])
        snap = bs.stats.snapshot()
        assert snap["requests"] == 5
        assert snap["batches"] >= 1
        assert snap["rows_total"] == 5
        assert snap["rows_per_batch_max"] >= 1
        assert snap["avg_callers_per_batch"] >= 1.0
    finally:
        await bs.stop()


def test_constructor_validates():
    with pytest.raises(ValueError):
        BatchingScorer(lambda r: ([], {}), max_batch_size=0)
    with pytest.raises(ValueError):
        BatchingScorer(lambda r: ([], {}), max_wait_ms=-1.0)


@pytest.mark.asyncio
async def test_scorer_length_mismatch_propagates_to_callers():
    """M4 (from step c review): if score_fn returns fewer scores than rows,
    callers must see a ValueError, not silently mis-ranked results. With
    `zip(..., strict=True)` in main.py the bug would now surface; here we
    pin the same expectation at the batcher boundary as well.
    """

    def bad_score_fn(rows):
        # Intentionally return one fewer score than rows.
        scores = [1.0] * (len(rows) - 1)
        return scores, {"tokenize_ms": 0.0, "gpu_ms": 0.0}

    bs = BatchingScorer(bad_score_fn, max_batch_size=8, max_wait_ms=2.0)
    await bs.start()
    try:
        # 2 rows in, 1 score back -> dispatch slicing must raise rather than
        # quietly return an empty / short list.
        with pytest.raises((ValueError, IndexError, AssertionError)):
            await bs.score([_row(1), _row(2)])
    finally:
        await bs.stop()


@pytest.mark.asyncio
async def test_cancellation_does_not_break_sibling_callers():
    """L5: if one caller is cancelled mid-flight, siblings in the same batch
    must still get their results (or at least not see InvalidStateError leak).

    Regression for the adversary review HIGH-1 finding: when set_result/
    set_exception fires after a future is cancelled, we must swallow
    InvalidStateError and keep delivering to the remaining callers.
    """
    fake = _FakeScorer(per_row_ms=20.0)  # slow enough for cancel-mid-flight
    bs = BatchingScorer(fake, max_batch_size=8, max_wait_ms=5.0)
    await bs.start()
    try:
        # Launch 4 concurrent callers; cancel the first one promptly.
        tasks = [asyncio.create_task(bs.score([_row(i)])) for i in range(4)]
        await asyncio.sleep(0.001)  # let them enqueue
        tasks[0].cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # First should be cancelled; siblings should each get 1 score.
        assert isinstance(results[0], asyncio.CancelledError)
        for r in results[1:]:
            assert not isinstance(r, BaseException), f"sibling failed: {r!r}"
            scores, _ = r
            assert len(scores) == 1
    finally:
        await bs.stop()
