"""agentdag: a local sandbox for DAGs of coding agents.

Primitives:
    - Blackboard: thread-safe shared memory with versioning
    - Tracer: structured event log + analyzer
    - Node / DAG: declarative graph of agent steps
    - Scheduler: thread-pool executor with queues + retries
    - agents.*: pluggable agent roles (planner, coder, adversary, council, ...)
"""

from .blackboard import Blackboard
from .dag import DAG, Node, RetryPolicy
from .scheduler import NodeFailed, Scheduler
from .tracer import TraceEvent, Tracer

__all__ = [
    "Blackboard",
    "Tracer",
    "TraceEvent",
    "DAG",
    "Node",
    "RetryPolicy",
    "Scheduler",
    "NodeFailed",
]
