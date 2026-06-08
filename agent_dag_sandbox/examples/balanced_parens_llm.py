"""Same balanced-parens TDD workflow, but with a real LLM as the coder.

Picks an LLM client based on env vars:
    LLM_PROVIDER=openai    -> uses OPENAI_API_KEY + LLM_MODEL (default gpt-4o-mini)
    LLM_PROVIDER=ollama    -> uses http://localhost:11434/v1 + LLM_MODEL (default qwen2.5:0.5b)
    LLM_PROVIDER=mock      -> EchoLLM (default) so the example runs offline

If the LLM call fails or returns unparseable code, the LLMCoder falls back to
the deterministic `Coder` -- so this script always finishes successfully,
whether or not you have an API key. Look at the trace's `log` events to see
which path was taken.

Run:
    python -m examples.balanced_parens_llm
    LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... python -m examples.balanced_parens_llm
    LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:0.5b python -m examples.balanced_parens_llm
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentdag import DAG, Blackboard, RetryPolicy, Scheduler, Tracer
from agentdag.agents import (
    Adversary,
    Breakdown,
    CaseWriter,
    Coder,
    Council,
    EchoLLM,
    LLMCoder,
    OpenAIChatLLM,
    Planner,
    Verifier,
)
from agentdag.agents.roles import union_tests


def pick_llm():
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    model = os.environ.get("LLM_MODEL", "")
    if provider == "openai":
        return OpenAIChatLLM(model=model or "gpt-4o-mini", base_url="https://api.openai.com/v1")
    if provider == "ollama":
        return OpenAIChatLLM(
            model=model or "qwen2.5:0.5b", base_url="http://localhost:11434/v1", api_key="ollama"
        )
    if provider == "vllm":
        return OpenAIChatLLM(
            model=model or "Qwen/Qwen2.5-0.5B-Instruct",
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
            api_key="not-needed",
        )
    return EchoLLM("mock-llm")


def build_dag(llm):
    dag = DAG("balanced_parens_llm")

    dag.add("planner", Planner())
    dag.add("breakdown", Breakdown(), deps=["planner"])

    dag.add(
        "test_council",
        Council(
            members=[
                CaseWriter(focus="general"),
                CaseWriter(focus="edge"),
                CaseWriter(focus="mixed"),
            ],
            aggregator=union_tests,
        ),
        deps=["breakdown"],
        retry=RetryPolicy(max_attempts=2, backoff_ms=10),
    )

    # Real (or mock) LLM coder, with deterministic fallback for resilience.
    dag.add(
        "coder",
        LLMCoder(llm, entrypoint="solve", fallback=Coder(buggy=True)),
        deps=["breakdown"],
        retry=RetryPolicy(max_attempts=2, backoff_ms=100),
    )

    dag.add("adversary", Adversary(), deps=["coder"])

    # Patching coder: same LLM, sees adversary_failures on the blackboard.
    dag.add(
        "patched_coder",
        LLMCoder(llm, entrypoint="solve", fallback=Coder(buggy=False)),
        deps=["adversary"],
        retry=RetryPolicy(max_attempts=2, backoff_ms=100),
    )

    dag.add("verifier", Verifier(), deps=["patched_coder", "test_council"])
    return dag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Live-stream every DAG event to stderr (also AGENTDAG_VERBOSE=1).",
    )
    args = parser.parse_args()

    llm = pick_llm()
    print(f"[setup] using LLM: {llm.name}")

    bb = Blackboard()
    bb.put(
        "task_spec",
        {
            "goal": "Implement is_balanced_parens(s: str) -> bool",
            "signature": "solve(s: str) -> bool",
            "notes": "Return True iff every '(' has a matching ')' in correct order.",
        },
    )

    trace_path = Path(__file__).resolve().parent.parent / "traces" / "balanced_parens_llm.jsonl"
    tracer = Tracer(trace_path, verbose=args.verbose or None)

    dag = build_dag(llm)
    sched = Scheduler(dag, blackboard=bb, tracer=tracer, workers=4)
    final = sched.run()

    print("\n=== FINAL VERIFIER REPORT ===")
    print(json.dumps(final["verifier"], indent=2))

    print("\n=== ADVERSARY FOUND ===")
    print(json.dumps(final["adversary"], indent=2, default=str))

    print("\n=== CODER OUTPUT (final) ===")
    print(final["patched_coder"]["source"])

    print("\n=== TRACE ===")
    print(tracer.render_table())
    print(f"\nfull JSONL: {trace_path}")


if __name__ == "__main__":
    main()
