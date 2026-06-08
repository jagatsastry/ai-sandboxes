"""Tests for the LLM-backed coder and the OpenAI-compatible HTTP client."""
from __future__ import annotations

import warnings

import pytest

from agentdag import Blackboard, DAG, Scheduler
from agentdag.agents import Coder, LLMCoder
from agentdag.agents.llm import LLMClient


class _StubLLM:
    """Lets tests dictate exactly what the LLM 'returns'."""
    name = "stub"
    def __init__(self, response: str, raise_exc: Exception | None = None):
        self.response = response
        self.raise_exc = raise_exc
        self.calls = []

    def complete(self, prompt, *, system="", temperature=0.0):
        self.calls.append({"prompt": prompt, "system": system})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


def _run(coder, **bb_kvs):
    bb = Blackboard()
    bb.put("task_spec", {"goal": "implement solve(s)"})
    for k, v in bb_kvs.items():
        bb.put(k, v)
    dag = DAG()
    dag.add("coder", coder)
    # Silence the fallback RuntimeWarning during tests; assertions
    # individually opt back in where they care about it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = Scheduler(dag, blackboard=bb).run()
    return out["coder"]


# -------- happy paths -------------------------------------------------------

def test_llm_coder_happy_path_with_fenced_code():
    stub = _StubLLM(
        "Here is the solution:\n\n```python\n"
        "def solve(s):\n    return s == s[::-1]\n"
        "```\n"
    )
    code = _run(LLMCoder(stub))
    assert code["entrypoint"] == "solve"
    assert "def solve" in code["source"]
    assert "s[::-1]" in code["source"]
    assert code["fallback_used"] is False
    assert len(stub.calls) == 1


def test_llm_coder_accepts_py_language_tag():
    stub = _StubLLM("```py\ndef solve(s):\n    return True\n```")
    code = _run(LLMCoder(stub))
    assert "def solve" in code["source"]
    assert code["fallback_used"] is False


def test_llm_coder_accepts_python_uppercase_tag():
    stub = _StubLLM("```Python\ndef solve(s):\n    return True\n```")
    code = _run(LLMCoder(stub))
    assert "def solve" in code["source"]
    assert code["fallback_used"] is False


def test_llm_coder_accepts_bare_fence():
    stub = _StubLLM("```\ndef solve(s):\n    return True\n```")
    code = _run(LLMCoder(stub))
    assert "def solve" in code["source"]
    assert code["fallback_used"] is False


def test_llm_coder_picks_last_code_block():
    # Scratch / example block first, real answer second.
    stub = _StubLLM(
        "First, an example:\n"
        "```python\ndef solve(s):\n    return False  # scratch\n```\n"
        "And the final answer:\n"
        "```python\ndef solve(s):\n    return True  # final\n```\n"
    )
    code = _run(LLMCoder(stub))
    assert "final" in code["source"]
    assert "scratch" not in code["source"]


# -------- hard fallback paths ----------------------------------------------

def test_llm_coder_falls_back_on_syntax_error():
    stub = _StubLLM("```python\ndef solve(s):\n    if True:\n```")  # incomplete
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert "depth" in code["source"]  # the GOOD fallback
    assert code["fallback_used"] is True
    assert "syntax-error" in code["fallback_reason"]


def test_llm_coder_falls_back_when_entrypoint_missing():
    stub = _StubLLM("```python\ndef other_name(s):\n    return True\n```")
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert "def solve" in code["source"]
    assert code["fallback_used"] is True


def test_llm_coder_falls_back_on_exception():
    stub = _StubLLM("", raise_exc=RuntimeError("network down"))
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert "def solve" in code["source"]
    assert code["fallback_used"] is True
    assert "llm-call-failed" in code["fallback_reason"]


def test_llm_coder_falls_back_on_no_code_block():
    # Raw prose, no fence at all -- must NOT exec it as code.
    stub = _StubLLM("def solve(s):\n    return True\n")
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert code["fallback_used"] is True
    assert "no-code-block" in code["fallback_reason"]
    # And the fallback source is what we ended up with:
    assert "depth" in code["source"]


def test_llm_coder_falls_back_on_empty_response():
    stub = _StubLLM("")
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert code["fallback_used"] is True
    assert code["fallback_reason"] == "empty-llm-response"


def test_llm_coder_falls_back_on_empty_code_block():
    stub = _StubLLM("```python\n   \n```")
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert code["fallback_used"] is True


def test_llm_coder_falls_back_when_entrypoint_is_not_callable():
    # `solve = 42` makes the name exist but uncallable. Probe-exec catches it.
    stub = _StubLLM("```python\nsolve = 42\n```")
    code = _run(LLMCoder(stub, fallback=Coder(buggy=False)))
    assert code["fallback_used"] is True
    assert "not-callable" in code["fallback_reason"]


def test_llm_coder_warns_on_fallback():
    stub = _StubLLM("no fence here, just prose")
    bb = Blackboard()
    bb.put("task_spec", {"goal": "implement solve(s)"})
    dag = DAG()
    dag.add("coder", LLMCoder(stub, fallback=Coder(buggy=False)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Scheduler(dag, blackboard=bb).run()
    # Exactly one RuntimeWarning from LLMCoder fallback.
    rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(rw) == 1
    assert "LLMCoder fallback engaged" in str(rw[0].message)


# -------- prompt construction ---------------------------------------------

def test_llm_coder_passes_adversary_failures_in_prompt():
    stub = _StubLLM("```python\ndef solve(s):\n    return False\n```")
    _run(LLMCoder(stub),
         adversary_failures=[{"input": ")(", "expected": False, "got": True}])
    assert len(stub.calls) == 1
    p = stub.calls[0]["prompt"]
    assert "<adversarial_failures>" in p
    assert "</adversarial_failures>" in p
    assert ")(" in p


def test_llm_coder_uses_not_run_default_for_missing_got():
    stub = _StubLLM("```python\ndef solve(s):\n    return True\n```")
    _run(LLMCoder(stub),
         adversary_failures=[{"input": "x", "expected": False}])  # no 'got'
    p = stub.calls[0]["prompt"]
    assert "<not-run>" in p


def test_llm_coder_shortens_long_adversarial_input():
    long_input = "(" * 1000
    stub = _StubLLM("```python\ndef solve(s):\n    return True\n```")
    _run(LLMCoder(stub),
         adversary_failures=[{"input": long_input, "expected": False, "got": True}])
    p = stub.calls[0]["prompt"]
    # The full 1000-char string should NOT appear verbatim; the _short marker should.
    assert long_input not in p
    assert "1000 chars" in p
