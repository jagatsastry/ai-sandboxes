"""Thread-safe shared memory ("blackboard") used by all agents.

Design notes:
    - Single global dict guarded by an RLock.
    - Every write bumps a monotonic version counter; readers can grab a snapshot
      (a shallow copy) so they reason about a consistent view of the world even
      while other agents are writing.
    - Namespaced keys are just dotted strings ("plan.steps", "code.v1"); we
      don't enforce schema -- agents agree on conventions.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, Optional, Tuple


class Blackboard:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._version: int = 0

    # ---- basic kv ---------------------------------------------------------
    def put(self, key: str, value: Any) -> int:
        with self._lock:
            self._data[key] = value
            self._version += 1
            return self._version

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._version += 1

    # ---- append helpers (common pattern: list of trials, list of tests) ---
    def append(self, key: str, value: Any) -> int:
        with self._lock:
            lst = self._data.setdefault(key, [])
            if not isinstance(lst, list):
                raise TypeError(f"blackboard key {key!r} is not a list")
            lst.append(value)
            self._version += 1
            return self._version

    # ---- snapshot ---------------------------------------------------------
    def snapshot(self) -> Tuple[int, Dict[str, Any]]:
        """Return (version, shallow_copy). Mutating the copy is safe."""
        with self._lock:
            return self._version, dict(self._data)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def items(self) -> Iterator[Tuple[str, Any]]:
        with self._lock:
            return iter(list(self._data.items()))

    def __repr__(self) -> str:  # pragma: no cover - debug only
        with self._lock:
            return f"Blackboard(v={self._version}, keys={list(self._data)})"
