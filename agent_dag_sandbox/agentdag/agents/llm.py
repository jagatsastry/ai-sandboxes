"""LLM client interface + a deterministic offline mock.

The whole sandbox depends ONLY on this tiny interface. Want real LLMs?
Subclass LLMClient and route .complete() to OpenAI / Anthropic / vLLM /
whatever. Nothing else in the codebase needs to change.
"""
from __future__ import annotations

import time
from typing import Protocol


class LLMClient(Protocol):
    name: str
    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str: ...


class EchoLLM:
    """A fake 'LLM' that returns canned responses based on prompt keywords.

    It's not trying to be smart -- it's trying to be deterministic and fast
    so the sandbox runs offline. Each instance can carry a 'persona' string
    to make council members disagree in predictable ways.
    """
    def __init__(self, name: str = "echo", persona: str = "default",
                 latency_ms: int = 5):
        self.name = name
        self.persona = persona
        self.latency_ms = latency_ms

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> str:
        # simulate work so traces look realistic
        time.sleep(self.latency_ms / 1000.0)
        # The 'agents' in roles.py do all the real logic locally and only
        # use this as a stand-in for a network LLM call. So we just echo
        # back a marker the caller can ignore.
        return f"[{self.name}:{self.persona}] ok"
