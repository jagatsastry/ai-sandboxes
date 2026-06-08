"""Agents for the *fix-failing-test* workflow.

Where ``roles.py`` covers "build code from scratch with TDD", this module
covers the opposite-but-equally-realistic loop: you have an existing
codebase plus a known-failing test, and the agents must diagnose and
patch the bug.

DAG roles in this module:

    TestRunner       - runs pytest in a sandboxed working dir, publishes
                       a structured failure report (stdout/stderr/return
                       code + parsed failing case names).
    BugLocalizer     - reads the failure report and the project source,
                       proposes ranked "suspicious symbols" + file:line
                       candidates. Pure-function and deterministic so
                       the example can run in CI; mirrors what an LLM
                       would emit.
    PatchProposer    - given the suspect symbols, emits one or more
                       candidate patches (each a small Python source
                       string replacing a target function).
    PatchApplier     - writes a chosen patch into a *copy* of the
                       project tree on the blackboard and re-runs the
                       tests. Never touches the original tree.
    RegressionGuard  - re-runs the FULL test suite (not just the
                       originally-failing case) against the patched
                       tree to make sure the fix didn't break anything
                       else. This is the "verifier" of this workflow.

All roles are MOCK in the same sense as roles.py: deterministic local
Python that follows the same contract a real LLM agent would. Swap
``BugLocalizer.localize()`` or ``PatchProposer.propose()`` for an LLM
call and the DAG keeps working unchanged.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dag import NodeContext

# ---------------------------------------------------------------------------
# data shapes (deliberately plain dicts on the blackboard; dataclasses here
# are just for the in-process API)
# ---------------------------------------------------------------------------


@dataclass
class FailureReport:
    """Parsed pytest output. ``failing_cases`` lists pytest node-ids."""

    returncode: int
    stdout: str
    stderr: str
    failing_cases: list[str] = field(default_factory=list)
    short_traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout_tail": self.stdout[-4000:],
            "stderr_tail": self.stderr[-2000:],
            "failing_cases": list(self.failing_cases),
            "short_traceback": self.short_traceback,
        }


@dataclass
class Suspect:
    """A symbol the localizer thinks contains the bug."""

    file: str
    symbol: str
    score: float  # in [0, 1]; higher = more suspicious
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "symbol": self.symbol, "score": self.score, "why": self.why}


@dataclass
class PatchCandidate:
    """A drop-in replacement for a target function."""

    file: str
    symbol: str
    new_source: str  # full function source, ``def foo(...): ...``
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "symbol": self.symbol, "rationale": self.rationale}


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------

_PYTEST_FAIL_RE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)


class TestRunner:
    """Invokes pytest on a project tree and structures the result.

    The project tree is read from ``ctx.bb.get("project_tree")`` as a
    mapping of relative-path -> file-text. We materialize it into a
    temp dir per call so multiple test runs in the same DAG (e.g.
    pre-patch, post-patch) don't stomp on each other.
    """

    # Tell pytest not to try collecting this as a test class (the name
    # starts with `Test`, which trips pytest's collection heuristic).
    __test__ = False

    def __init__(self, target: str | None = None) -> None:
        # If target is None, run the whole suite. Used by RegressionGuard.
        self.target = target

    def __call__(self, ctx: NodeContext) -> dict[str, Any]:
        tree = ctx.bb.get("project_tree")
        if not isinstance(tree, dict):
            raise ValueError("blackboard.project_tree missing or not a mapping")

        # If a previous patcher published a patched tree, prefer it.
        # This lets us reuse TestRunner pre- and post-patch.
        patched = ctx.bb.get("patched_tree")
        if patched is not None and isinstance(patched, dict):
            tree = patched

        report = _run_pytest_on_tree(tree, target=self.target)
        ctx.log(
            "tests ran",
            returncode=report.returncode,
            n_failures=len(report.failing_cases),
        )
        return report.to_dict()


def _run_pytest_on_tree(tree: dict[str, str], target: str | None) -> FailureReport:
    """Materialize ``tree`` into a temp dir and run pytest."""
    with tempfile.TemporaryDirectory(prefix="agentdag_repair_") as td:
        root = Path(td)
        for rel, text in tree.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

        # Ensure the project root is importable by tests
        env_path_setup = "import sys; sys.path.insert(0, '.')\n"
        conftest = root / "conftest.py"
        if not conftest.exists():
            conftest.write_text(env_path_setup)

        argv = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--no-header"]
        if target:
            argv.append(target)
        # Build a clean env so the outer monorepo's PYTHONPATH/cwd doesn't
        # leak into the sandboxed pytest subprocess and mask real import
        # failures in the project-under-test.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k
            in {
                "PATH",
                "HOME",
                "LANG",
                "LC_ALL",
                "TMPDIR",
                "TEMP",
                "TMP",
                "SYSTEMROOT",
            }
        }
        try:
            proc = subprocess.run(
                argv,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=clean_env,
            )
        except subprocess.TimeoutExpired as exc:
            return FailureReport(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\n[runner] pytest timed out",
                failing_cases=[],
                short_traceback="timeout",
            )

        failing = _PYTEST_FAIL_RE.findall(proc.stdout) + _PYTEST_FAIL_RE.findall(proc.stderr)
        short_tb = _extract_short_traceback(proc.stdout)
        return FailureReport(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            failing_cases=failing,
            short_traceback=short_tb,
        )


def _extract_short_traceback(out: str) -> str:
    """Pull the last ``FAILED``/``ERROR`` block out of pytest -q output.

    We capture everything between the first ``=== FAILURES ===`` (or
    ``=== ERRORS ===``) banner and the ``=== short test summary info ===``
    banner that follows it. If neither banner is present we return the
    last 2000 chars as a best-effort tail.
    """
    lines = out.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and ("= FAILURES =" in line or "= ERRORS =" in line):
            start = i
        elif start is not None and "= short test summary info =" in line:
            end = i
            break
    if start is None:
        return out[-2000:]
    block = "\n".join(lines[start : end if end is not None else len(lines)])
    return block[-2000:]


# ---------------------------------------------------------------------------
# BugLocalizer
# ---------------------------------------------------------------------------


class BugLocalizer:
    """Reads the failure report and ranks candidate buggy symbols.

    Pure-function heuristic so this example is deterministic in CI:
      1. Scan the traceback for `File "X", line N, in symbol`.
      2. Boost any symbol whose name appears in the failing test's
         identifier (e.g. ``test_fizzbuzz_three`` -> boost ``fizzbuzz``).
      3. As a fallback, return every top-level function in every .py
         file the tests import, lightly ranked.
    """

    _TB_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
    _SHORT_TB_FRAME_RE = re.compile(r"^([\w./\\-]+\.py):(\d+):\s+in\s+(\w+)", re.MULTILINE)
    # Terminated at end-of-line so we don't gobble the next def block.
    _IMPORT_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import\s+([^\n#]+)", re.MULTILINE)

    def __call__(self, ctx: NodeContext) -> list[dict[str, Any]]:
        report = ctx.bb.get("test_runner")
        tree = ctx.bb.get("project_tree")
        if not report or not tree:
            raise ValueError("BugLocalizer needs test_runner + project_tree on blackboard")
        suspects = self._localize(report, tree)
        ctx.log("localized", n=len(suspects), top=[s.symbol for s in suspects[:3]])
        return [s.to_dict() for s in suspects]

    @classmethod
    def _localize(cls, report: dict[str, Any], tree: dict[str, str]) -> list[Suspect]:
        traceback = report.get("short_traceback", "") + "\n" + report.get("stdout_tail", "")
        failing_cases: list[str] = report.get("failing_cases", [])

        seen: dict[tuple[str, str], Suspect] = {}
        # Pattern 1: classic --tb=long frames `File "X", line N, in symbol`
        for m in cls._TB_FRAME_RE.finditer(traceback):
            f, _line, sym = m.group(1), m.group(2), m.group(3)
            if "pytest" in f or "/test_" in f or f.endswith("/conftest.py"):
                continue
            rel = _match_in_tree(f, tree)
            if rel is None:
                continue
            key = (rel, sym)
            if key in seen:
                seen[key].score = min(1.0, seen[key].score + 0.2)
            else:
                seen[key] = Suspect(
                    file=rel, symbol=sym, score=0.6, why="appeared in failing traceback"
                )

        # Pattern 2: --tb=short frames `tests/test_x.py:14: in test_y`.
        # Symbol there is the *test* function, not the buggy code; but
        # we can pivot via the imports in that test file.
        for m in cls._SHORT_TB_FRAME_RE.finditer(traceback):
            test_file, _line, _test_sym = m.group(1), m.group(2), m.group(3)
            rel = _match_in_tree(test_file, tree)
            if rel is None or not rel.startswith("tests/"):
                continue
            test_src = tree.get(rel, "")
            for imp_match in cls._IMPORT_RE.finditer(test_src):
                module = imp_match.group(1)
                names = [n.strip() for n in imp_match.group(2).split(",") if n.strip()]
                # Resolve module -> file in the tree (`fizzbuzz` -> `fizzbuzz.py`)
                candidates = [
                    rel2
                    for rel2 in tree
                    if rel2 == f"{module}.py" or rel2.endswith(f"/{module}.py")
                ]
                for cand in candidates:
                    for name in names:
                        if name == "*":
                            continue
                        key = (cand, name)
                        if key in seen:
                            seen[key].score = min(1.0, seen[key].score + 0.15)
                        else:
                            seen[key] = Suspect(
                                file=cand,
                                symbol=name,
                                score=0.55,
                                why=f"imported by failing test {rel}",
                            )

        # Name-match boost: if a failing test is `test_fizzbuzz_basic` and
        # there's a function named `fizzbuzz`, boost it hard.
        for case in failing_cases:
            tail = case.rsplit("::", 1)[-1].lower()
            # Split on `_` so `test_str_format` -> {"test","str","format"}
            # and we don't falsely boost a 2-letter symbol like `is`.
            tail_words = set(re.split(r"[_\W]+", tail))
            for (_rel, sym), s in seen.items():
                if sym.lower() in tail_words:
                    s.score = min(1.0, s.score + 0.3)
                    s.why += "; name appears in failing test id"

        # Fallback: if traceback parsing produced nothing, list every
        # top-level def in the source tree with a low base score. Skip
        # ANY path that contains a `tests/` segment, not just root-level
        # tests/ — nested layouts like `pkg/tests/test_x.py` are common.
        if not seen:
            for rel, text in tree.items():
                if not rel.endswith(".py"):
                    continue
                norm = rel.replace("\\", "/")
                if "tests/" in norm + "/" or "/test_" in "/" + norm:
                    continue
                for m in re.finditer(r"^def\s+(\w+)\s*\(", text, re.MULTILINE):
                    sym = m.group(1)
                    seen[(rel, sym)] = Suspect(
                        file=rel, symbol=sym, score=0.2, why="fallback: no traceback frames"
                    )

        return sorted(seen.values(), key=lambda s: s.score, reverse=True)


def _match_in_tree(path: str, tree: dict[str, str]) -> str | None:
    """Map an absolute tempdir path back to its key in ``tree``."""
    for rel in tree:
        if path.endswith("/" + rel) or path.endswith("\\" + rel) or path == rel:
            return rel
    base = path.rsplit("/", 1)[-1]
    for rel in tree:
        if rel.endswith("/" + base) or rel == base:
            return rel
    return None


# ---------------------------------------------------------------------------
# PatchProposer
# ---------------------------------------------------------------------------


class PatchProposer:
    """Proposes candidate replacement implementations for the top suspect.

    For the demo we hard-code a small "fixer book" keyed by symbol name
    so the example runs without an LLM. Real deployments would swap the
    body of ``propose()`` for an LLM call that takes the failing test +
    the current symbol source and emits replacement source.
    """

    def __init__(self, fixer_book: dict[str, str] | None = None) -> None:
        self.fixer_book = fixer_book or _DEFAULT_FIXER_BOOK

    def __call__(self, ctx: NodeContext) -> list[dict[str, Any]]:
        suspects = ctx.bb.get("bug_localizer", [])
        tree = ctx.bb.get("project_tree", {})
        if not suspects:
            raise ValueError("PatchProposer found no suspects on blackboard")
        patches = self._propose(suspects, tree)
        ctx.log("proposed", n=len(patches), symbols=[p.symbol for p in patches])
        return [p.to_dict() | {"new_source": p.new_source} for p in patches]

    def _propose(
        self, suspects: list[dict[str, Any]], tree: dict[str, str]
    ) -> list[PatchCandidate]:
        out: list[PatchCandidate] = []
        for s in suspects[:3]:  # top-3 only
            sym = s["symbol"]
            new_src = self.fixer_book.get(sym)
            if new_src is None:
                continue
            out.append(
                PatchCandidate(
                    file=s["file"],
                    symbol=sym,
                    new_source=textwrap.dedent(new_src).strip() + "\n",
                    rationale=f"fixer-book entry for {sym}",
                )
            )
        return out


_DEFAULT_FIXER_BOOK: dict[str, str] = {
    "fizzbuzz": """
        def fizzbuzz(n: int) -> str:
            if n % 15 == 0:
                return "FizzBuzz"
            if n % 3 == 0:
                return "Fizz"
            if n % 5 == 0:
                return "Buzz"
            return str(n)
    """,
    "is_palindrome": """
        def is_palindrome(s: str) -> bool:
            s = ''.join(ch.lower() for ch in s if ch.isalnum())
            return s == s[::-1]
    """,
}


# ---------------------------------------------------------------------------
# PatchApplier
# ---------------------------------------------------------------------------


class PatchApplier:
    """Applies the first proposed patch into a *copy* of the tree and
    publishes ``patched_tree`` on the blackboard.

    We deliberately do NOT mutate ``project_tree``; downstream nodes can
    still read the original.
    """

    def __call__(self, ctx: NodeContext) -> dict[str, Any]:
        patches = ctx.bb.get("patch_proposer", [])
        tree = ctx.bb.get("project_tree", {})
        if not patches:
            return {"applied": False, "reason": "no patches proposed"}
        chosen = patches[0]
        new_tree = dict(tree)  # shallow copy of mapping (str -> str)
        target_text = new_tree.get(chosen["file"])
        if target_text is None:
            return {"applied": False, "reason": f"target {chosen['file']} missing"}
        replaced = _replace_top_level_def(target_text, chosen["symbol"], chosen["new_source"])
        if replaced is None:
            return {"applied": False, "reason": f"symbol {chosen['symbol']} not found"}
        new_tree[chosen["file"]] = replaced
        ctx.bb.put("patched_tree", new_tree)
        ctx.log("patched", file=chosen["file"], symbol=chosen["symbol"])
        return {
            "applied": True,
            "file": chosen["file"],
            "symbol": chosen["symbol"],
            "rationale": chosen["rationale"],
        }


def _replace_top_level_def(source: str, symbol: str, new_def_source: str) -> str | None:
    """Replace the first top-level ``def <symbol>(...)`` block.

    Uses ``ast`` to find the *exact* byte span of the function (including
    decorators, multi-line signatures, nested helpers, one-liners, and a
    final-line def with no trailing newline). Falls back to ``None`` if
    the source doesn't parse or the symbol isn't a top-level function.

    We deliberately preserve the line separator BEFORE the def (e.g. the
    blank line that visually separates it from the previous def) and the
    line separator AFTER the def. ``new_def_source`` should already end
    in a single ``\\n``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    src_lines = source.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != symbol:
            continue
        # If there are decorators, the def block starts at the first one.
        start_line = (
            node.decorator_list[0].lineno if node.decorator_list else node.lineno
        ) - 1  # ast linenos are 1-based
        # end_lineno is inclusive 1-based; convert to exclusive 0-based.
        end_line = node.end_lineno or node.lineno  # py3.8+
        # Slice by lines so we keep all the original line separators outside
        # the def block intact.
        prefix = "".join(src_lines[:start_line])
        suffix = "".join(src_lines[end_line:])
        # Make sure the replacement ends with exactly one newline so the
        # next chunk doesn't accidentally glue to ours.
        if not new_def_source.endswith("\n"):
            new_def_source += "\n"
        return prefix + new_def_source + suffix
    return None


# ---------------------------------------------------------------------------
# RegressionGuard
# ---------------------------------------------------------------------------


class RegressionGuard:
    """Runs the FULL test suite against the patched tree.

    Distinct from a second TestRunner instance only in intent + the
    structured verdict it emits.
    """

    def __call__(self, ctx: NodeContext) -> dict[str, Any]:
        tree = ctx.bb.get("patched_tree") or ctx.bb.get("project_tree")
        if not tree:
            raise ValueError("RegressionGuard found no tree on blackboard")
        report = _run_pytest_on_tree(tree, target=None)
        ok = report.returncode == 0 and not report.failing_cases
        verdict = {
            "ok": ok,
            "returncode": report.returncode,
            "n_failures": len(report.failing_cases),
            "failing_cases": report.failing_cases,
            "short_traceback": report.short_traceback,
        }
        ctx.log("regression", **verdict)
        return verdict


# Re-export the temp-tree helper so tests can drive it directly.
__all__ = [
    "TestRunner",
    "BugLocalizer",
    "PatchProposer",
    "PatchApplier",
    "RegressionGuard",
    "FailureReport",
    "Suspect",
    "PatchCandidate",
]


# A tiny helper used by tests; lets us delete an extracted tree.
def _rmtree_quiet(p: Path) -> None:  # pragma: no cover - test plumbing only
    shutil.rmtree(p, ignore_errors=True)
