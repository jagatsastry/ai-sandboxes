"""End-to-end sample: solve `is_balanced_parens` with a DAG of agents.

DAG shape:

    planner -> breakdown -> test_council ----\\
                         \\                    +-> verifier -> report
                          -> coder -> adversary -> patched_coder -/

Roles:
    planner        - drafts a plan
    breakdown      - splits plan into subtasks
    test_council   - 3 TestWriters (general, edge, mixed) merged via union
    coder          - writes initial implementation (intentionally BUGGY
                     so we can show the adversary catching it and the
                     patcher fixing it -- this is the whole point of TDD
                     + adversary in the sandbox)
    adversary      - runs red-team cases, publishes failures
    patched_coder  - re-runs coder; because adversary_failures is set,
                     coder returns the GOOD implementation
    verifier       - runs union(tests) + adversarial cases

Run:  python -m examples.balanced_parens
"""
from __future__ import annotations

import json
from pathlib import Path

from agentdag import DAG, Blackboard, RetryPolicy, Scheduler, Tracer
from agentdag.agents import (
    Adversary,
    Breakdown,
    Coder,
    Council,
    Planner,
    CaseWriter,
    Verifier,
)
from agentdag.agents.roles import union_tests


def build_dag() -> DAG:
    dag = DAG("balanced_parens_tdd")

    dag.add("planner", Planner())
    dag.add("breakdown", Breakdown(), deps=["planner"])

    # COUNCIL: 3 test writers with different focuses, merged via union.
    test_council = Council(
        members=[
            CaseWriter(focus="general"),
            CaseWriter(focus="edge"),
            CaseWriter(focus="mixed"),
        ],
        aggregator=union_tests,
        name="test_council",
    )
    dag.add(
        "test_council",
        test_council,
        deps=["breakdown"],
        retry=RetryPolicy(max_attempts=2, backoff_ms=10),
    )

    # Initial coder is intentionally BUGGY so the adversary has something to find.
    dag.add("coder", Coder(buggy=True), deps=["breakdown"])

    # Adversary depends on coder; runs red-team cases.
    dag.add(
        "adversary",
        Adversary(),
        deps=["coder"],
        retry=RetryPolicy(max_attempts=2),
    )

    # Patching coder: same Coder class, runs after adversary; because
    # adversary published 'adversary_failures' to the blackboard, the
    # coder will emit the GOOD implementation this time.
    dag.add("patched_coder", Coder(buggy=False), deps=["adversary"])

    # Verifier needs the final code AND the council's test suite.
    dag.add("verifier", Verifier(), deps=["patched_coder", "test_council"])

    return dag


def main() -> None:
    bb = Blackboard()
    bb.put("task_spec", {
        "goal": "Implement is_balanced_parens(s: str) -> bool",
        "signature": "solve(s: str) -> bool",
        "notes": "Return True iff every '(' has a matching ')' in correct order.",
    })

    trace_path = Path(__file__).resolve().parent.parent / "traces" / "balanced_parens.jsonl"
    tracer = Tracer(trace_path)

    dag = build_dag()
    sched = Scheduler(dag, blackboard=bb, tracer=tracer, workers=4)
    final = sched.run()

    print("\n=== FINAL VERIFIER REPORT ===")
    print(json.dumps(final["verifier"], indent=2))

    print("\n=== ADVERSARY FOUND ===")
    print(json.dumps(final["adversary"], indent=2, default=str))

    print("\n=== TEST COUNCIL MERGED SUITE ===")
    print(f"{len(final['test_council'])} unique cases after union of 3 writers")

    print("\n=== TRACE (where time went) ===")
    print(tracer.render_table())

    print(f"\nfull JSONL trace: {trace_path}")


if __name__ == "__main__":
    main()
