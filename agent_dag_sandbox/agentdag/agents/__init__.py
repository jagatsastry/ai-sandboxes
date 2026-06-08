"""Built-in agent roles. Each agent is a callable taking NodeContext.

All agents in the sandbox are MOCK LLMs -- deterministic local Python that
follows the same input/output contract a real LLM-backed agent would. Swap
the body for an OpenAI/Anthropic call and the DAG keeps working.
"""
from .llm import LLMClient, EchoLLM
from .roles import (
    Planner,
    Breakdown,
    TestWriter,
    Coder,
    Adversary,
    Verifier,
    Council,
)

__all__ = [
    "LLMClient",
    "EchoLLM",
    "Planner",
    "Breakdown",
    "TestWriter",
    "Coder",
    "Adversary",
    "Verifier",
    "Council",
]
