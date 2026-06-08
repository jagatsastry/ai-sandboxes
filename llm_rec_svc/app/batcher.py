"""Dynamic micro-batching for the LLM reranker.

WHY: With one model + one CPU/GPU worker, requests serialize. At c=8 on
a single sync worker we saw ~9x p50 latency and ~0.84x throughput. The
fix is to batch across requests instead of within a single request:
collect concurrent candidate rows into one tokenize+forward call.

DESIGN:
- Each request enqueues its rows + an asyncio.Future on a shared queue
  and awaits the future.
- A single background coroutine (`_run`) drains the queue, builds a
  super-batch up to `max_batch_size` or until `max_wait_ms` elapses,
  calls the synchronous `score_fn` once in a thread (so we don't block
  the event loop), and resolves each waiter with its slice of the
  result.
- One worker thread per model is correct here: HF models do their own
  internal parallelism and adding model-level concurrency on CPU just
  thrashes the cache. The batcher amortizes the per-call overhead
  (tokenizer dispatch, framework dispatch, autograd disable) across N
  concurrent callers.

CONFIG:
- max_batch_size: hard cap on rows per forward pass.
- max_wait_ms:    upper bound on the queueing time the FIRST row in a
                  batch can incur before the batcher gives up waiting
                  for more rows and flushes. 0 means "no extra wait,
                  but still drain anything already in the queue".

OBSERVABILITY:
- BatchStats counts requests, batches, total rows, max batch size,
  rolling-mean batch size, rolling-mean callers/batch. Surfaced at
  /metrics. All reads/writes are guarded by a threading.Lock since
  `record()` runs from `asyncio.to_thread` worker threads and
  `snapshot()` is called from FastAPI sync route handlers (also on
  worker threads).

THREAD-SAFETY:
- The wrapped sync `score_fn` is called from a single background
  coroutine via `asyncio.to_thread`. It is NEVER invoked concurrently.
- BatchStats has its own lock; safe to read from any thread.
- asyncio primitives (Queue, Future) are loop-bound -- the batcher's
  start/stop/score must all run on the same event loop. We capture
  `asyncio.get_running_loop()` in `start()` and assert it stays the
  same in subsequent calls.

CANCELLATION:
- If a caller cancels their `score()` future (e.g. client disconnect),
  `_dispatch` will detect the cancelled future via try/except
  InvalidStateError and skip delivery for that caller. The cancelled
  caller's rows are still scored (and discarded) because we already
  committed to running them in this batch -- changing that would
  require splitting the batch mid-flight.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Sequence, Tuple


# Row shape: (profile, title, tags, desc) -- same as scorer.score_batch.
Row = Tuple[str, str, Sequence[str], str]


# Typed sentinel so we never confuse a real (empty-rows) caller with the
# stop signal. Empty real rows are short-circuited by `score()` and never
# reach the queue, but this is defence-in-depth.
class _StopSentinel:
    __slots__ = ()


_SENTINEL = _StopSentinel()


@dataclass
class _Pending:
    rows: List[Row]
    future: "asyncio.Future[Tuple[List[float], dict]]"
    enqueued_at: float


@dataclass
class BatchStats:
    """Thread-safe counters surfaced at /metrics.

    All mutating methods take `_lock`; `snapshot()` also takes it so
    readers never see a torn list during append/evict.
    """
    requests: int = 0           # incoming caller-level requests
    batches: int = 0            # forward passes actually executed
    rows_total: int = 0         # total scored rows (incl. cancelled callers)
    rows_max_batch: int = 0     # largest single batch we ran
    last_batch_sizes: Deque[int] = field(default_factory=lambda: deque(maxlen=64))
    last_callers_per_batch: Deque[int] = field(default_factory=lambda: deque(maxlen=64))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, batch_rows: int, n_callers: int) -> None:
        with self._lock:
            self.requests += n_callers
            self.batches += 1
            self.rows_total += batch_rows
            if batch_rows > self.rows_max_batch:
                self.rows_max_batch = batch_rows
            self.last_batch_sizes.append(batch_rows)
            self.last_callers_per_batch.append(n_callers)

    def note_request(self, n: int = 1) -> None:
        """Count a caller that took the empty-rows fast path."""
        with self._lock:
            self.requests += n

    def snapshot(self) -> dict:
        with self._lock:
            recent_rows = list(self.last_batch_sizes)
            recent_callers = list(self.last_callers_per_batch)
            cum_requests = self.requests
            cum_batches = self.batches
            cum_rows = self.rows_total
            cum_max = self.rows_max_batch
        avg_callers_recent = (sum(recent_callers) / len(recent_callers)) if recent_callers else 0.0
        return {
            "requests": cum_requests,
            "batches": cum_batches,
            "rows_total": cum_rows,
            "rows_per_batch_max": cum_max,
            "rows_per_batch_avg": (sum(recent_rows) / len(recent_rows))
                                   if recent_rows else 0.0,
            # kept for backward compat with existing tests + /metrics consumers
            "avg_callers_per_batch": avg_callers_recent,
            "callers_per_batch_avg_recent": avg_callers_recent,
            "callers_per_batch_avg_cum": (cum_requests / cum_batches)
                                           if cum_batches else 0.0,
        }


class BatchingScorer:
    """Async wrapper around a sync `score_fn(rows) -> (scores, timings)`."""

    def __init__(
        self,
        score_fn: Callable[[Sequence[Row]], Tuple[List[float], dict]],
        max_batch_size: int = 64,
        max_wait_ms: float = 5.0,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        self.score_fn = score_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # asyncio primitives are loop-bound; create them in start() so we
        # don't bind to whatever loop happens to be current at construction.
        self._queue: "asyncio.Queue | None" = None
        self._task: "asyncio.Task | None" = None
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._stopped = False
        self.stats = BatchStats()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._stopped = False
        self._task = asyncio.create_task(self._run(), name="batcher")

    async def stop(self, drain_timeout_s: float = 30.0) -> None:
        """Signal the drainer to exit. Waits up to `drain_timeout_s` for
        any in-flight forward pass to complete before force-cancelling.

        Default 30 s is conservative enough for slow CPU models (e.g.
        Qwen2.5-0.5B at K=20 ~ 6 s). Override for tighter shutdowns.
        """
        if self._task is None or self._queue is None:
            return
        self._stopped = True
        # Push a typed sentinel so the queue.get unblocks.
        sentinel_pending = _Pending(
            rows=[],
            future=self._loop.create_future(),  # type: ignore[union-attr]
            enqueued_at=time.monotonic(),
        )
        sentinel_pending.future.set_result(([], {}))  # never read
        # We use a tuple (sentinel_pending, _SENTINEL) to mark this entry
        # unambiguously. _run checks `is _SENTINEL`.
        await self._queue.put(_SENTINEL)
        try:
            try:
                await asyncio.wait_for(self._task, timeout=drain_timeout_s)
            except asyncio.TimeoutError:
                self._task.cancel()
        finally:
            self._task = None
            self._queue = None
            self._loop = None

    async def score(self, rows: Sequence[Row]) -> Tuple[List[float], dict]:
        """Submit a list of rows; await the (scores, timings) for THIS caller."""
        if not rows:
            self.stats.note_request(1)
            return [], {"tokenize_ms": 0.0, "gpu_ms": 0.0,
                        "batch_size": 0, "queue_ms": 0.0}
        if self._queue is None or self._loop is None:
            raise RuntimeError("BatchingScorer not started")
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            raise RuntimeError(
                "BatchingScorer.score() called from a different event loop "
                "than start() ran on"
            )
        fut: asyncio.Future = loop.create_future()
        pending = _Pending(
            rows=list(rows), future=fut, enqueued_at=time.monotonic(),
        )
        await self._queue.put(pending)
        return await fut

    async def _run(self) -> None:
        """Drain the queue, build batches, dispatch, return slices to callers."""
        assert self._queue is not None
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return
            if first is _SENTINEL:
                return

            head: _Pending = first  # type: ignore[assignment]
            batch_pending: List[_Pending] = [head]
            batch_rows: List[Row] = list(head.rows)

            # Phase 1: immediately drain anything already sitting in the
            # queue without sleeping. This is what makes max_wait_ms=0
            # still coalesce concurrent arrivals.
            self._drain_ready(batch_pending, batch_rows)

            # Phase 2: wait up to max_wait_ms for late arrivals, but only
            # if we still have room.
            if (len(batch_rows) < self.max_batch_size
                    and self.max_wait_ms > 0):
                deadline = head.enqueued_at + (self.max_wait_ms / 1000.0)
                while len(batch_rows) < self.max_batch_size:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(
                            self._queue.get(), timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        break
                    if nxt is _SENTINEL:
                        await self._dispatch(batch_pending, batch_rows)
                        return
                    remaining = self.max_batch_size - len(batch_rows)
                    np = nxt  # type: _Pending  # noqa: F841 - clarity
                    if len(nxt.rows) <= remaining:
                        batch_pending.append(nxt)
                        batch_rows.extend(nxt.rows)
                    else:
                        # Caller would overflow this batch; defer to the
                        # next batch as its own head, resetting its
                        # enqueue clock so its max_wait_ms is fresh.
                        nxt.enqueued_at = time.monotonic()
                        await self._queue.put(nxt)
                        break

            await self._dispatch(batch_pending, batch_rows)

    def _drain_ready(
        self,
        batch_pending: List[_Pending],
        batch_rows: List[Row],
    ) -> None:
        """Pull anything that's already enqueued, no waiting.

        Stops at max_batch_size or when a caller wouldn't fit (the
        offender is re-queued for the next batch with a fresh clock).
        """
        assert self._queue is not None
        while len(batch_rows) < self.max_batch_size:
            try:
                nxt = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if nxt is _SENTINEL:
                # Put it back so the main loop sees it and stops cleanly.
                # We can't process the stop here because we still owe a
                # dispatch.
                self._queue.put_nowait(_SENTINEL)
                return
            remaining = self.max_batch_size - len(batch_rows)
            if len(nxt.rows) <= remaining:
                batch_pending.append(nxt)
                batch_rows.extend(nxt.rows)
            else:
                nxt.enqueued_at = time.monotonic()
                self._queue.put_nowait(nxt)
                return

    async def _dispatch(
        self,
        pendings: List[_Pending],
        rows: List[Row],
    ) -> None:
        """Run the model on `rows` and resolve each pending future."""
        if not rows:
            return
        try:
            scores, timings = await asyncio.to_thread(self.score_fn, rows)
        except BaseException as exc:  # noqa: BLE001 - propagate to every caller
            for p in pendings:
                # Swallow InvalidStateError if a caller cancelled between
                # the check above and set_exception below (TOCTOU).
                try:
                    p.future.set_exception(exc)
                except asyncio.InvalidStateError:
                    pass
            # Do NOT record stats for a failed batch: callers that got an
            # exception didn't really "use" the model.
            return

        # Slice scores back to each caller in arrival order.
        cursor = 0
        delivered_callers = 0
        for p in pendings:
            n = len(p.rows)
            sl = scores[cursor:cursor + n]
            cursor += n
            caller_timings = {
                **timings,
                "batch_size": len(rows),
                "queue_ms": (time.monotonic() - p.enqueued_at) * 1000.0,
            }
            try:
                p.future.set_result((sl, caller_timings))
                delivered_callers += 1
            except asyncio.InvalidStateError:
                # Caller cancelled between enqueue and result; their rows
                # were still scored, the work just gets discarded.
                pass

        # Stats reflect ACTUAL forward passes and the callers that were
        # delivered. Cancelled callers don't count toward
        # callers_per_batch but the batch itself does count.
        self.stats.record(batch_rows=len(rows), n_callers=delivered_callers)
