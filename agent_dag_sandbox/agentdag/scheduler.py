"""Thread-pool scheduler with queues, retries, and tracing.

How it works:

    1. validate the DAG and compute initial ready set (no deps).
    2. spawn N worker threads; each pops from a shared `queue.Queue`.
    3. on success, store result on blackboard under the node's name, then
       check every dependent: if all its deps are done, enqueue it.
    4. on failure, consult RetryPolicy; either re-enqueue (with backoff) or
       mark failed and propagate cancellation.

We intentionally use threads (not asyncio) because the agents in this sandbox
are CPU-light Python code that may also do blocking I/O when wired to real
APIs -- a thread pool is the simplest model that scales to both.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from .blackboard import Blackboard
from .dag import DAG, NodeContext
from .tracer import Tracer


class NodeFailed(Exception):
    def __init__(self, node: str, original: BaseException):
        super().__init__(f"node {node!r} failed: {original!r}")
        self.node = node
        self.original = original


@dataclass
class _WorkItem:
    node_name: str
    attempt: int  # 1-based


class Scheduler:
    def __init__(
        self,
        dag: DAG,
        blackboard: Blackboard | None = None,
        tracer: Tracer | None = None,
        workers: int = 4,
    ) -> None:
        self.dag = dag
        self.bb = blackboard or Blackboard()
        self.tracer = tracer or Tracer()
        self.workers = max(1, workers)

        self._q: queue.Queue[_WorkItem | None] = queue.Queue()
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._ever_enqueued: set[str] = set()
        self._done_lock = threading.Lock()
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()
        self._cancel = threading.Event()
        self._error: NodeFailed | None = None

    # ---- public ----------------------------------------------------------
    def run(self) -> dict[str, object]:
        self.dag.validate()

        # initial ready set
        for n in self.dag.nodes.values():
            if not n.deps:
                self._ever_enqueued.add(n.name)
                self._enqueue(n.name, attempt=1)

        threads = [
            threading.Thread(target=self._worker_loop, name=f"w{i}", daemon=True)
            for i in range(self.workers)
        ]
        for t in threads:
            t.start()

        # Wait until queue drained AND nothing in flight (or cancelled).
        # We poll: cheap and simple, and lets us bail on cancel quickly.
        while True:
            if self._cancel.is_set() and self._q.empty():
                # let in-flight finish
                with self._in_flight_lock:
                    if self._in_flight == 0:
                        break
            else:
                with self._in_flight_lock:
                    idle = self._in_flight == 0
                if idle and self._q.empty():
                    break
            time.sleep(0.005)

        # poison pills
        for _ in threads:
            self._q.put(None)
        for t in threads:
            t.join(timeout=2.0)

        if self._error is not None:
            raise self._error

        # return view of final blackboard
        _, snap = self.bb.snapshot()
        return snap

    # ---- internals -------------------------------------------------------
    def _enqueue(self, name: str, attempt: int) -> None:
        self.tracer.emit("enqueued", name, attempt=attempt)
        self._q.put(_WorkItem(node_name=name, attempt=attempt))

    def _worker_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            with self._in_flight_lock:
                self._in_flight += 1
            try:
                self._run_one(item)
            finally:
                with self._in_flight_lock:
                    self._in_flight -= 1
                self._q.task_done()

    def _run_one(self, item: _WorkItem) -> None:
        if self._cancel.is_set():
            return
        node = self.dag.nodes[item.node_name]
        ctx = NodeContext(name=node.name, attempt=item.attempt, bb=self.bb, tracer=self.tracer)
        self.tracer.emit("started", node.name, attempt=item.attempt)
        try:
            result = node.fn(ctx)
        except BaseException as exc:  # noqa: BLE001 - we re-raise via NodeFailed
            if node.retry.should_retry(item.attempt, exc):
                self.tracer.emit(
                    "retry",
                    node.name,
                    attempt=item.attempt,
                    error=repr(exc),
                )
                if node.retry.backoff_ms > 0:
                    time.sleep(node.retry.backoff_ms / 1000.0)
                self._enqueue(node.name, attempt=item.attempt + 1)
                return
            self.tracer.emit("failed", node.name, attempt=item.attempt, error=repr(exc))
            with self._done_lock:
                self._failed.add(node.name)
            self._error = NodeFailed(node.name, exc)
            self._cancel.set()
            return

        # success: write result and unlock dependents
        self.bb.put(node.name, result)
        self.tracer.emit("finished", node.name, attempt=item.attempt)
        with self._done_lock:
            self._done.add(node.name)
            newly_ready = []
            for cand in self.dag.nodes.values():
                if cand.name in self._ever_enqueued:
                    continue
                if all(d in self._done for d in cand.deps):
                    self._ever_enqueued.add(cand.name)
                    newly_ready.append(cand.name)
        for nm in newly_ready:
            self._enqueue(nm, attempt=1)
