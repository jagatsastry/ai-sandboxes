"""Built-in agent roles for the sample TDD coding workflow.

Each role is a class with __call__(ctx) so it plugs straight into DAG.add().
All roles are deterministic Python -- think of them as "what a perfect LLM
would emit", coded inline. Swap any of them with real LLM calls and the
DAG stays valid.

Roles:
    Planner    - turns a task spec into a high-level plan
    Breakdown  - turns the plan into concrete subtasks
    TestWriter - writes a list of (input, expected) test cases (TDD)
    Coder      - writes Python code for the task
    Adversary  - tries to find edge cases the tests missed
    Verifier   - actually runs the code against all tests, returns report
    Council    - runs N agents in parallel, votes/aggregates their output
"""
from __future__ import annotations

import concurrent.futures
import re
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..dag import NodeContext
from .llm import EchoLLM, LLMClient


# ---------------- Planner -------------------------------------------------
class Planner:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or EchoLLM("planner")

    def __call__(self, ctx: NodeContext) -> Dict[str, Any]:
        spec = ctx.bb.get("task_spec")
        if not spec:
            raise ValueError("blackboard missing 'task_spec'")
        self.llm.complete(f"plan: {spec}")  # simulate model latency
        plan = {
            "goal": spec["goal"],
            "approach": [
                "understand the function signature and constraints",
                "write tests covering happy path + edge cases (TDD)",
                "implement the function",
                "run adversary to find missed cases",
                "patch implementation if needed",
                "final verification",
            ],
        }
        ctx.log("plan drafted", steps=len(plan["approach"]))
        return plan


# ---------------- Breakdown ----------------------------------------------
class Breakdown:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or EchoLLM("breakdown")

    def __call__(self, ctx: NodeContext) -> List[Dict[str, str]]:
        plan = ctx.bb.get("planner")
        self.llm.complete(f"breakdown: {plan}")
        subtasks = [
            {"id": f"st{i+1}", "do": step}
            for i, step in enumerate(plan["approach"])
        ]
        ctx.log("subtasks created", n=len(subtasks))
        return subtasks


# ---------------- TestWriter ---------------------------------------------
class TestWriter:
    """Emits a list of (input, expected) test cases for the target function.

    The 'persona' affects which kinds of cases it focuses on so council
    members produce complementary test suites.
    """
    def __init__(self, llm: LLMClient | None = None, focus: str = "general"):
        self.llm = llm or EchoLLM("tester", persona=focus)
        self.focus = focus

    def __call__(self, ctx: NodeContext) -> List[Tuple[Any, Any]]:
        spec = ctx.bb.get("task_spec")
        self.llm.complete(f"tests[{self.focus}]: {spec}")
        # For the sample task: is_balanced_parens(s) -> bool
        common = [
            ("", True),
            ("()", True),
            ("(", False),
            (")", False),
        ]
        if self.focus == "edge":
            extra = [
                ("(" * 1000 + ")" * 1000, True),
                (")(", False),
                ("(()", False),
                ("())", False),
            ]
        elif self.focus == "mixed":
            extra = [
                ("()()", True),
                ("(())", True),
                ("(()(()))", True),
                ("(()))(()", False),
            ]
        else:  # general
            extra = [
                ("(())", True),
                ("()()()", True),
                ("(()", False),
            ]
        cases = common + extra
        ctx.log("tests written", n=len(cases), focus=self.focus)
        return cases


# ---------------- Coder ---------------------------------------------------
class Coder:
    """Returns Python source as a string + the function name to call."""

    GOOD = textwrap.dedent("""
        def solve(s):
            depth = 0
            for ch in s:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth < 0:
                        return False
            return depth == 0
    """).strip()

    # An intentionally subtly-wrong version used when patch_mode=False is
    # exercised in tests; default Coder returns the correct one.
    BUGGY = textwrap.dedent("""
        def solve(s):
            # bug: doesn't catch ')(' style imbalance
            return s.count('(') == s.count(')')
    """).strip()

    def __init__(self, llm: LLMClient | None = None, buggy: bool = False):
        self.llm = llm or EchoLLM("coder")
        self.buggy = buggy

    def __call__(self, ctx: NodeContext) -> Dict[str, str]:
        self.llm.complete("write code")
        # If a previous adversary added a "patch_needed" hint, always
        # produce the good version even if we started buggy.
        if ctx.bb.get("adversary_failures"):
            source = self.GOOD
            ctx.log("patched after adversary")
        else:
            source = self.BUGGY if self.buggy else self.GOOD
        return {"source": source, "entrypoint": "solve"}


# ---------------- Adversary ----------------------------------------------
class Adversary:
    """Generates extra adversarial test cases and runs them against the
    candidate code. Reports any failures so the coder can patch."""

    ADVERSARIAL = [
        (")(", False),
        ("(" * 50 + ")" * 49, False),
        ("(" * 49 + ")" * 50, False),
        ("()" * 500, True),
    ]

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or EchoLLM("adversary", persona="redteam")

    def __call__(self, ctx: NodeContext) -> Dict[str, Any]:
        code = ctx.bb.get("coder")
        self.llm.complete("adversary probe")
        failures = _run_cases(code["source"], code["entrypoint"], self.ADVERSARIAL)
        ctx.log("adversary ran", failures=len(failures))
        if failures:
            # publish a flag the patching coder will read
            ctx.bb.put("adversary_failures", failures)
        return {"cases_tried": len(self.ADVERSARIAL), "failures": failures}


# ---------------- Verifier ------------------------------------------------
class Verifier:
    """Runs the (post-patch) code against the council's merged test suite
    AND the adversarial set. Returns a structured pass/fail report."""

    def __call__(self, ctx: NodeContext) -> Dict[str, Any]:
        code = ctx.bb.get("patched_coder") or ctx.bb.get("coder")
        suite = ctx.bb.get("test_council") or []
        all_cases = list(suite) + list(Adversary.ADVERSARIAL)
        failures = _run_cases(code["source"], code["entrypoint"], all_cases)
        report = {
            "total": len(all_cases),
            "passed": len(all_cases) - len(failures),
            "failed": len(failures),
            "failures": failures[:5],  # cap to keep blackboard tidy
            "ok": len(failures) == 0,
        }
        ctx.log("verified", **{k: v for k, v in report.items() if k != "failures"})
        return report


# ---------------- Council -------------------------------------------------
@dataclass
class CouncilResult:
    members: List[Any]
    aggregate: Any


class Council:
    """Runs N agents in parallel (thread pool) and aggregates their outputs.

    aggregator: callable(list_of_outputs) -> aggregate
    The default aggregator for test suites = set-union (dedup); for code
    pick the majority by exact-match.
    """
    def __init__(
        self,
        members: Sequence[Callable[[NodeContext], Any]],
        aggregator: Callable[[List[Any]], Any] | None = None,
        name: str = "council",
    ):
        if not members:
            raise ValueError("council needs at least one member")
        self.members = list(members)
        self.aggregator = aggregator or self._default_aggregate
        self.name = name

    @staticmethod
    def _default_aggregate(outs: List[Any]) -> Any:
        # majority vote by repr; ties broken by first
        from collections import Counter
        c = Counter(repr(o) for o in outs)
        winner_repr, _ = c.most_common(1)[0]
        for o in outs:
            if repr(o) == winner_repr:
                return o
        return outs[0]

    def __call__(self, ctx: NodeContext) -> Any:
        ctx.log("council convening", members=len(self.members))
        # Each member sees the SAME NodeContext (same blackboard). They run
        # in parallel threads. Their outputs are passed to the aggregator;
        # the aggregate becomes the council node's result on the blackboard.
        outs: List[Any] = [None] * len(self.members)

        def _run(i_member):
            i, member = i_member
            return i, member(ctx)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.members)
        ) as ex:
            for i, out in ex.map(_run, list(enumerate(self.members))):
                outs[i] = out

        agg = self.aggregator(outs)
        ctx.log("council aggregated", n_outputs=len(outs))
        return agg


# ---------------- helpers -------------------------------------------------
def _run_cases(
    source: str, entrypoint: str, cases: Sequence[Tuple[Any, Any]]
) -> List[Dict[str, Any]]:
    """Exec `source` in a fresh namespace and run each case."""
    ns: Dict[str, Any] = {}
    try:
        exec(compile(source, "<agent-code>", "exec"), ns)
    except Exception as e:
        return [{"input": None, "expected": None, "error": f"compile: {e!r}"}]
    fn = ns.get(entrypoint)
    if not callable(fn):
        return [{"input": None, "expected": None, "error": f"no entrypoint {entrypoint!r}"}]
    fails = []
    for inp, expected in cases:
        try:
            got = fn(inp)
        except Exception as e:
            fails.append({"input": _short(inp), "expected": expected,
                          "got": f"<exc {e!r}>"})
            continue
        if got != expected:
            fails.append({"input": _short(inp), "expected": expected, "got": got})
    return fails


def _short(x: Any, n: int = 40) -> Any:
    if isinstance(x, str) and len(x) > n:
        return x[:n] + f"...<{len(x)} chars>"
    return x


# ---------------- aggregators -------------------------------------------
def union_tests(outs: List[List[Tuple[Any, Any]]]) -> List[Tuple[Any, Any]]:
    """Aggregator for TestWriter council: dedup by (input, expected)."""
    seen = set()
    merged: List[Tuple[Any, Any]] = []
    for suite in outs:
        for case in suite:
            key = (repr(case[0]), repr(case[1]))
            if key not in seen:
                seen.add(key)
                merged.append(case)
    return merged
