"""Built-in agent roles. Each agent is a callable taking NodeContext.

All agents in the sandbox are MOCK LLMs -- deterministic local Python that
follows the same input/output contract a real LLM-backed agent would. Swap
the body for an OpenAI/Anthropic call and the DAG keeps working.
"""
from .llm import LLMClient, EchoLLM
from .roles import (
    Planner,
    Breakdown,
    CaseWriter,
    Coder,
    Adversary,
    Verifier,
    Council,
)

# Back-compat alias (deprecated): older name shadowed pytest's Test* heuristic.
TestWriter = CaseWriter

__all__ = [
    "LLMClient",
    "EchoLLM",
    "Planner",
    "Breakdown",
    "CaseWriter",
    "TestWriter",
    "Coder",
    "Adversary",
    "Verifier",
    "Council",
]
