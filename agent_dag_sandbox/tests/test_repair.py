"""Unit tests for the fix-failing-test agent roles + the example DAG.

These deliberately don't talk to an LLM so they're fast and CI-stable.
"""

from __future__ import annotations

from agentdag import DAG, Blackboard, Scheduler, Tracer
from agentdag.agents import (
    BugLocalizer,
    PatchApplier,
    PatchProposer,
    RegressionGuard,
    TestRunner,
)
from agentdag.agents.repair import (
    _extract_short_traceback,
    _match_in_tree,
    _replace_top_level_def,
    _run_pytest_on_tree,
)

# ---------------------------------------------------------------------------
# tiny shared fixture: the same buggy fizzbuzz the example uses
# ---------------------------------------------------------------------------


def _buggy_project() -> dict[str, str]:
    return {
        "fizzbuzz.py": (
            "def fizzbuzz(n: int) -> str:\n"
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
            "def test_fizzbuzz_fifteen():\n"
            "    assert fizzbuzz(15) == 'FizzBuzz'\n"
        ),
    }


# ---------------------------------------------------------------------------
# _run_pytest_on_tree: structural sanity
# ---------------------------------------------------------------------------


def test_pytest_on_passing_tree_returns_zero():
    good = {
        "lib.py": "def add(a, b):\n    return a + b\n",
        "tests/test_lib.py": "from lib import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    }
    rep = _run_pytest_on_tree(good, target=None)
    assert rep.returncode == 0
    assert rep.failing_cases == []


def test_pytest_on_buggy_tree_returns_nonzero_with_named_failure():
    rep = _run_pytest_on_tree(_buggy_project(), target=None)
    assert rep.returncode != 0
    assert any("test_fizzbuzz_fifteen" in c for c in rep.failing_cases)
    # Adversary HIGH-1: short_traceback must actually contain something
    # informative — the original state machine always returned "".
    assert rep.short_traceback, "short_traceback should not be empty"
    assert "FAILURES" in rep.short_traceback or "AssertionError" in rep.short_traceback


def test_extract_short_traceback_handles_real_pytest_output():
    # Synthetic pytest -q --tb=short output containing the canonical banners.
    out = (
        "...F\n"
        "=================================== FAILURES ===================================\n"
        "___ test_x ___\n"
        "tests/test_x.py:14: in test_x\n"
        "    assert wrong()\n"
        "E   AssertionError\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_x.py::test_x\n"
    )
    block = _extract_short_traceback(out)
    assert "FAILURES" in block
    assert "AssertionError" in block
    # Must stop at the summary banner (we don't want to keep the FAILED line).
    assert "FAILED tests/test_x.py::test_x" not in block


# ---------------------------------------------------------------------------
# BugLocalizer: ranks the buggy symbol first
# ---------------------------------------------------------------------------


def test_localizer_ranks_imported_symbol():
    tree = _buggy_project()
    rep = _run_pytest_on_tree(tree, target=None)
    # Drive the inner ranker directly; the agent wrapper just unpacks bb.
    suspects = BugLocalizer._localize(rep.to_dict(), tree)
    assert suspects, "localizer must produce at least one suspect"
    top = suspects[0]
    assert top.file == "fizzbuzz.py"
    assert top.symbol == "fizzbuzz"
    # Name-match boost should have pushed it well above the 0.55 base.
    assert top.score >= 0.6


def test_localizer_falls_back_when_no_traceback():
    tree = {"foo.py": "def foo():\n    return 1\n"}
    fake_report = {"short_traceback": "", "stdout_tail": "", "failing_cases": []}
    suspects = BugLocalizer._localize(fake_report, tree)
    # Fallback should still surface `foo` so downstream isn't stuck.
    assert any(s.symbol == "foo" for s in suspects)


# ---------------------------------------------------------------------------
# _replace_top_level_def: indentation-aware replacement
# ---------------------------------------------------------------------------


def test_replace_top_level_def_handles_return_annotation():
    src = (
        '"""mod."""\n'
        "\n"
        "def buggy(n: int) -> str:\n"
        "    return 'wrong'\n"
        "\n"
        "def other(x):\n"
        "    return x\n"
    )
    new = _replace_top_level_def(src, "buggy", "def buggy(n):\n    return 'right'\n")
    assert new is not None
    assert "return 'right'" in new
    assert "return 'wrong'" not in new
    # The OTHER def must not have been clobbered.
    assert "def other(x):" in new


def test_replace_top_level_def_missing_symbol_returns_none():
    src = "def a():\n    return 1\n"
    assert _replace_top_level_def(src, "nope", "def nope(): pass\n") is None


def test_replace_top_level_def_preserves_blank_lines_in_body():
    src = "def foo():\n    a = 1\n\n    b = 2\n    return a + b\n\ndef bar():\n    return 0\n"
    new = _replace_top_level_def(src, "foo", "def foo():\n    return 99\n")
    assert new is not None
    assert "def bar():" in new
    assert "return 99" in new


def test_replace_top_level_def_handles_multiline_signature():
    # Adversary HIGH-2: the old regex-based replacer corrupted files when
    # the signature spanned multiple lines.
    src = "def foo(\n    x: int,\n    y: int,\n) -> int:\n    return x + y\n\ndef bar():\n    return 0\n"
    new = _replace_top_level_def(src, "foo", "def foo(x, y):\n    return x * y\n")
    assert new is not None
    # No dangling signature remnants.
    assert "-> int:" not in new
    assert "return x + y" not in new
    assert "return x * y" in new
    # Sibling def must still parse and be present.
    assert "def bar():" in new
    # The replaced file as a whole must still be valid Python.
    import ast as _ast

    _ast.parse(new)


def test_replace_top_level_def_preserves_blank_line_separator():
    # Adversary MEDIUM-3: blank line between defs must survive.
    src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    new = _replace_top_level_def(src, "foo", "def foo():\n    return 99\n")
    assert new is not None
    assert "return 99\n\ndef bar():" in new


def test_replace_top_level_def_handles_decorated_function():
    src = "from functools import cache\n" "\n" "@cache\n" "def slow(x):\n" "    return x * 2\n"
    new = _replace_top_level_def(src, "slow", "def slow(x):\n    return x + 1\n")
    assert new is not None
    # Decorator must be replaced along with the function.
    assert "@cache" not in new
    assert "return x + 1" in new


def test_replace_top_level_def_handles_no_trailing_newline():
    # Adversary MEDIUM-7: file ends with a def and no trailing newline.
    src = "def foo():\n    return 1"  # no trailing \n
    new = _replace_top_level_def(src, "foo", "def foo():\n    return 2\n")
    assert new is not None
    assert "return 2" in new
    assert "return 1" not in new


# ---------------------------------------------------------------------------
# _match_in_tree: tolerant path matching
# ---------------------------------------------------------------------------


def test_match_in_tree_by_basename():
    tree = {"pkg/mod.py": "...", "other.py": "..."}
    assert _match_in_tree("/tmp/x/pkg/mod.py", tree) == "pkg/mod.py"
    assert _match_in_tree("/tmp/x/other.py", tree) == "other.py"
    assert _match_in_tree("/tmp/x/nope.py", tree) is None


# ---------------------------------------------------------------------------
# Full DAG end-to-end
# ---------------------------------------------------------------------------


def _build_repair_dag() -> DAG:
    dag = DAG("repair")
    dag.add("test_runner", TestRunner())
    dag.add("bug_localizer", BugLocalizer(), deps=["test_runner"])
    dag.add("patch_proposer", PatchProposer(), deps=["bug_localizer"])
    dag.add("patch_applier", PatchApplier(), deps=["patch_proposer"])
    dag.add("regression_guard", RegressionGuard(), deps=["patch_applier"])
    return dag


def test_repair_dag_fixes_fizzbuzz_end_to_end(tmp_path):
    bb = Blackboard()
    bb.put("project_tree", _buggy_project())
    tracer = Tracer(tmp_path / "trace.jsonl")
    dag = _build_repair_dag()
    final = Scheduler(dag, blackboard=bb, tracer=tracer, workers=2).run()

    # Initial run must have reported the failure.
    assert final["test_runner"]["returncode"] != 0
    assert final["test_runner"]["failing_cases"]

    # A patch must have been applied successfully.
    assert final["patch_applier"]["applied"] is True
    assert final["patch_applier"]["symbol"] == "fizzbuzz"

    # The regression guard must be green after the patch.
    assert final["regression_guard"]["ok"] is True
    assert final["regression_guard"]["returncode"] == 0

    # Adversary LOW-12: the patched tree itself must contain the corrected
    # branch logic (n % 15 first). This is a stronger guarantee than just
    # "the test suite happened to pass."
    patched_fizzbuzz = bb.get("patched_tree")["fizzbuzz.py"]
    assert "n % 15 == 0" in patched_fizzbuzz
    # And the buggy comment must be gone.
    assert "%3 branch is checked before %15" not in patched_fizzbuzz


def test_repair_dag_is_safe_when_no_fix_available(tmp_path):
    """When the fixer-book has no entry for the buggy symbol, the applier
    should report applied=False and the guard should reflect failure
    (not crash).
    """
    tree = {
        "unknown.py": "def mystery_bug():\n    raise ZeroDivisionError('boom')\n",
        "tests/test_unknown.py": (
            "from unknown import mystery_bug\n" "def test_mystery():\n    mystery_bug()\n"
        ),
    }
    bb = Blackboard()
    bb.put("project_tree", tree)
    tracer = Tracer(tmp_path / "trace.jsonl")
    final = Scheduler(_build_repair_dag(), blackboard=bb, tracer=tracer, workers=2).run()
    # No entry for `mystery_bug` in the fixer-book.
    assert final["patch_applier"]["applied"] is False
    # Regression guard still ran, and (correctly) reports a failure.
    assert final["regression_guard"]["ok"] is False
