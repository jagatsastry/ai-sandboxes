"""Structured tracing for DAG runs.

Each event is a small dict written to an in-memory list and optionally appended
to a JSONL file. The analyzer turns that stream into a per-node table that
shows where wall-clock time went: queue wait, execution, retries.

Event kinds:
    enqueued   -> node is ready, placed on the work queue
    started    -> a worker pulled it off the queue
    finished   -> node returned successfully
    retry      -> node raised but will be retried (attempt N -> N+1)
    failed     -> node exhausted retries and gave up
    log        -> free-form message emitted by an agent

Every event carries: ts (monotonic seconds), wall (wall-clock epoch),
node, attempt, kind, and an optional payload dict.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    ts: float  # monotonic; for durations
    wall: float  # epoch seconds; for humans
    kind: str  # enqueued|started|finished|retry|failed|log
    node: str
    attempt: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tracer:
    def __init__(self, path: str | Path | None = None) -> None:
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._path = Path(path) if path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # truncate
            self._path.write_text("")

    def emit(self, kind: str, node: str, attempt: int = 0, **payload: Any) -> None:
        ev = TraceEvent(
            ts=time.monotonic(),
            wall=time.time(),
            kind=kind,
            node=node,
            attempt=attempt,
            payload=payload,
        )
        with self._lock:
            self._events.append(ev)
            if self._path:
                with self._path.open("a") as f:
                    f.write(json.dumps(ev.to_dict()) + "\n")

    @property
    def events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    # ---- analysis ---------------------------------------------------------
    def summarize(self) -> dict[str, Any]:
        """Per-node timings + a rollup.

        Returns dict like:
            {
              "nodes": {
                 "planner": {"attempts":1, "queue_ms": 0.4, "run_ms": 12.0,
                             "retries":0, "status":"finished"},
                 ...
              },
              "total_ms": ...,
              "wall_start": ..., "wall_end": ...,
              "critical_path_ms": ...   # longest finished path of run_ms
            }
        """
        evs = self.events
        if not evs:
            return {"nodes": {}, "total_ms": 0.0}

        # Per-node, per-attempt bookkeeping
        node_state: dict[str, dict[str, Any]] = {}
        for ev in evs:
            st = node_state.setdefault(
                ev.node,
                {
                    "attempts": 0,
                    "queue_ms": 0.0,
                    "run_ms": 0.0,
                    "retries": 0,
                    "status": "pending",
                    "_enqueued_ts": None,
                    "_started_ts": None,
                },
            )
            if ev.kind == "enqueued":
                st["_enqueued_ts"] = ev.ts
            elif ev.kind == "started":
                st["attempts"] = max(st["attempts"], ev.attempt)
                st["_started_ts"] = ev.ts
                if st["_enqueued_ts"] is not None:
                    st["queue_ms"] += (ev.ts - st["_enqueued_ts"]) * 1000.0
                    st["_enqueued_ts"] = None
            elif ev.kind == "finished":
                if st["_started_ts"] is not None:
                    st["run_ms"] += (ev.ts - st["_started_ts"]) * 1000.0
                    st["_started_ts"] = None
                st["status"] = "finished"
            elif ev.kind == "retry":
                if st["_started_ts"] is not None:
                    st["run_ms"] += (ev.ts - st["_started_ts"]) * 1000.0
                    st["_started_ts"] = None
                st["retries"] += 1
                st["status"] = "retrying"
            elif ev.kind == "failed":
                if st["_started_ts"] is not None:
                    st["run_ms"] += (ev.ts - st["_started_ts"]) * 1000.0
                    st["_started_ts"] = None
                st["status"] = "failed"

        for st in node_state.values():
            st.pop("_enqueued_ts", None)
            st.pop("_started_ts", None)

        wall_start = min(e.wall for e in evs)
        wall_end = max(e.wall for e in evs)
        total_ms = (wall_end - wall_start) * 1000.0
        critical_path_ms = sum(st["run_ms"] for st in node_state.values())

        return {
            "nodes": node_state,
            "total_ms": total_ms,
            "wall_start": wall_start,
            "wall_end": wall_end,
            "sum_run_ms": critical_path_ms,
        }

    def render_table(self) -> str:
        s = self.summarize()
        nodes = s["nodes"]
        if not nodes:
            return "(no events)"
        rows = [
            ("node", "status", "attempts", "retries", "queue_ms", "run_ms"),
        ]
        for name, st in nodes.items():
            rows.append(
                (
                    name,
                    st["status"],
                    str(st["attempts"]),
                    str(st["retries"]),
                    f"{st['queue_ms']:.1f}",
                    f"{st['run_ms']:.1f}",
                )
            )
        widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        lines = []
        for i, r in enumerate(rows):
            lines.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
            if i == 0:
                lines.append("  ".join("-" * w for w in widths))
        lines.append("")
        lines.append(f"total wall: {s['total_ms']:.1f} ms")
        lines.append(f"sum of run_ms (work done): {s['sum_run_ms']:.1f} ms")
        return "\n".join(lines)
