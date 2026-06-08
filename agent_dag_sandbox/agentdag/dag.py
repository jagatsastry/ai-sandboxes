"""DAG model: Nodes wrap callables; edges express data dependencies.

A Node is `(name, fn, deps, retry)`. The function signature is:

    fn(ctx: NodeContext) -> Any

where NodeContext gives the agent access to the blackboard, tracer, its own
name, the current attempt number, and a `log()` helper.

The result of fn is stored on the blackboard under the node's name by the
scheduler, so downstream nodes can read it via `ctx.bb.get("<upstream>")`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .blackboard import Blackboard
from .tracer import Tracer


@dataclass
class RetryPolicy:
    max_attempts: int = 1  # 1 = no retries
    backoff_ms: int = 0  # fixed backoff between attempts
    # retry only if the raised exception's type name is in this set;
    # empty = retry on any Exception.
    retry_on: set[str] = field(default_factory=set)

    def should_retry(self, attempt: int, exc: BaseException) -> bool:
        if attempt >= self.max_attempts:
            return False
        if not self.retry_on:
            return isinstance(exc, Exception)
        return type(exc).__name__ in self.retry_on


@dataclass
class NodeContext:
    name: str
    attempt: int
    bb: Blackboard
    tracer: Tracer

    def log(self, msg: str, **fields: Any) -> None:
        self.tracer.emit("log", self.name, self.attempt, message=msg, **fields)


NodeFn = Callable[[NodeContext], Any]


@dataclass
class Node:
    name: str
    fn: NodeFn
    deps: list[str] = field(default_factory=list)
    retry: RetryPolicy = field(default_factory=RetryPolicy)


class DAG:
    def __init__(self, name: str = "dag") -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}

    def add(
        self,
        name: str,
        fn: NodeFn,
        deps: Iterable[str] | None = None,
        retry: RetryPolicy | None = None,
    ) -> Node:
        if name in self._nodes:
            raise ValueError(f"duplicate node: {name}")
        node = Node(
            name=name,
            fn=fn,
            deps=list(deps or []),
            retry=retry or RetryPolicy(),
        )
        self._nodes[name] = node
        return node

    @property
    def nodes(self) -> dict[str, Node]:
        return self._nodes

    def validate(self) -> None:
        # All deps must exist
        for n in self._nodes.values():
            for d in n.deps:
                if d not in self._nodes:
                    raise ValueError(f"node {n.name!r} depends on unknown {d!r}")
        # No cycles
        self.topological_order()

    def topological_order(self) -> list[str]:
        indeg: dict[str, int] = {n: 0 for n in self._nodes}
        for n in self._nodes.values():
            for _d in n.deps:
                indeg[n.name] += 1
        ready = [n for n, k in indeg.items() if k == 0]
        order: list[str] = []
        while ready:
            ready.sort()  # deterministic
            cur = ready.pop(0)
            order.append(cur)
            for n in self._nodes.values():
                if cur in n.deps:
                    indeg[n.name] -= 1
                    if indeg[n.name] == 0:
                        ready.append(n.name)
        if len(order) != len(self._nodes):
            cyc = [n for n, k in indeg.items() if k > 0]
            raise ValueError(f"cycle detected involving: {cyc}")
        return order
