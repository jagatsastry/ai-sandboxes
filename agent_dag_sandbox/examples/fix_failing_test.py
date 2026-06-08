"""End-to-end sample: the *fix-failing-test* workflow.

Where ``balanced_parens.py`` shows agents building code from scratch
with TDD, this example shows the dual workflow: a maintainer-shaped
loop where the agents are handed a broken project + a known-failing
test and must localize the bug, propose a patch, apply it, and
verify the full suite still passes.

DAG shape:

    test_runner --> bug_localizer --> patch_proposer --> patch_applier --> regression_guard --> report

Run:  python -m examples.fix_failing_test

The "buggy" project below is a deliberately classic textbook bug:
``fizzbuzz(15)`` returns ``"Fizz"`` because the ``%3`` branch fires
before the ``%15`` check. The test catches it, the localizer scores
``fizzbuzz`` as the top suspect, the patch proposer pulls the correct
implementation from its fixer-book, the applier swaps it into a copy
of the tree, and the regression guard re-runs the suite to confirm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentdag import DAG, Blackboard, RetryPolicy, Scheduler, Tracer
from agentdag.agents import (
    BugLocalizer,
    PatchApplier,
    PatchProposer,
    RegressionGuard,
    TestRunner,
)

# ---------------------------------------------------------------------------
# The "project under repair" lives entirely in this dict. We do this so
# the example is hermetic + side-effect-free: nothing on the developer's
# filesystem is read or written outside a tempdir created by TestRunner.
# ---------------------------------------------------------------------------

BUGGY_PROJECT: dict[str, str] = {
    "fizzbuzz.py": (
        '"""Classic fizzbuzz, with a classic bug."""\n'
        "\n"
        "def fizzbuzz(n: int) -> str:\n"
        "    # BUG: %3 branch is checked before %15, so multiples of 15\n"
        "    # incorrectly return 'Fizz' instead of 'FizzBuzz'.\n"
        "    if n % 3 == 0:\n"
        "        return 'Fizz'\n"
        "    if n % 5 == 0:\n"
        "        return 'Buzz'\n"
        "    if n % 15 == 0:\n"
        "        return 'FizzBuzz'\n"
        "    return str(n)\n"
    ),
    "tests/test_fizzbuzz.py": (
        "from fizzbuzz import fizzbuzz\n"
        "\n"
        "def test_fizzbuzz_one():\n"
        "    assert fizzbuzz(1) == '1'\n"
        "\n"
        "def test_fizzbuzz_three():\n"
        "    assert fizzbuzz(3) == 'Fizz'\n"
        "\n"
        "def test_fizzbuzz_five():\n"
        "    assert fizzbuzz(5) == 'Buzz'\n"
        "\n"
        "def test_fizzbuzz_fifteen():\n"
        "    # This is the one that catches the bug.\n"
        "    assert fizzbuzz(15) == 'FizzBuzz'\n"
    ),
}


def build_dag() -> DAG:
    dag = DAG("fix_failing_test")

    # 1) Run the suite to discover what's broken.
    dag.add(
        "test_runner",
        TestRunner(),
        retry=RetryPolicy(max_attempts=1),  # the run itself shouldn't be flaky
    )

    # 2) Localize: which symbol most likely contains the bug?
    dag.add("bug_localizer", BugLocalizer(), deps=["test_runner"])

    # 3) Propose patches for the top suspect(s).
    dag.add("patch_proposer", PatchProposer(), deps=["bug_localizer"])

    # 4) Apply the first patch into a copy of the tree.
    dag.add("patch_applier", PatchApplier(), deps=["patch_proposer"])

    # 5) Regression-guard: full suite must pass on the patched tree.
    dag.add(
        "regression_guard",
        RegressionGuard(),
        deps=["patch_applier"],
        retry=RetryPolicy(max_attempts=1),
    )

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

    bb = Blackboard()
    bb.put("project_tree", BUGGY_PROJECT)

    trace_path = Path(__file__).resolve().parent.parent / "traces" / "fix_failing_test.jsonl"
    tracer = Tracer(trace_path, verbose=args.verbose or None)

    dag = build_dag()
    sched = Scheduler(dag, blackboard=bb, tracer=tracer, workers=2)
    final = sched.run()

    print("\n=== INITIAL TEST RUN (before patch) ===")
    initial = final["test_runner"]
    print(f"  returncode: {initial['returncode']}")
    print(f"  failing_cases: {initial['failing_cases']}")

    print("\n=== BUG LOCALIZER (top 3 suspects) ===")
    for s in final["bug_localizer"][:3]:
        print(f"  [{s['score']:.2f}] {s['file']}::{s['symbol']}  ({s['why']})")

    print("\n=== PATCH PROPOSED & APPLIED ===")
    print(json.dumps(final["patch_applier"], indent=2))

    print("\n=== REGRESSION GUARD ===")
    print(json.dumps(final["regression_guard"], indent=2))

    print("\n=== TRACE (where time went) ===")
    print(tracer.render_table())

    print(f"\nfull JSONL trace: {trace_path}")

    # Exit nonzero if the regression guard didn't actually go green,
    # so CI catches regressions in the example itself.
    if not final["regression_guard"]["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
