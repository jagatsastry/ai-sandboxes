"""Tests for the Tracer verbose-mode live streaming."""

from __future__ import annotations

import io

from agentdag import DAG, Scheduler, Tracer


def _build_simple_dag() -> DAG:
    dag = DAG()

    def step_a(ctx):
        return {"x": 1}

    def step_b(ctx):
        return {"x": ctx.bb.get("a")["x"] + 1}

    dag.add("a", step_a)
    dag.add("b", step_b, deps=["a"])
    return dag


def test_verbose_default_is_quiet():
    """Verbose must be off by default so existing scripts/CI stay quiet."""
    buf = io.StringIO()
    tr = Tracer(stream=buf)
    assert tr.verbose is False
    tr.emit("log", "n1", message="hi")
    assert buf.getvalue() == ""


def test_verbose_streams_each_event():
    """When verbose=True, every emit() writes one line to the stream."""
    buf = io.StringIO()
    tr = Tracer(verbose=True, stream=buf)
    sched = Scheduler(_build_simple_dag(), tracer=tr, workers=2)
    out = sched.run()
    assert out["b"]["x"] == 2

    lines = buf.getvalue().splitlines()
    # we expect at least: enqueued/started/finished for both nodes (6 events)
    assert len(lines) >= 6
    # canonical event kinds appear
    text = "\n".join(lines)
    for kind in ("enqueued", "started", "finished"):
        assert kind in text
    # the relative-timestamp prefix and node columns are present
    assert all(line.startswith("[t+") for line in lines)
    assert "node=" not in lines[0]  # we use bare label, not key=value


def test_verbose_truncates_large_payloads():
    """A payload bigger than 80 chars must be truncated with an ellipsis."""
    buf = io.StringIO()
    tr = Tracer(verbose=True, stream=buf)
    big = "x" * 500
    tr.emit("log", "n1", blob=big)
    line = buf.getvalue().strip()
    assert "..." in line
    # truncated to ~80 char preview + ellipsis: full 500-char blob must not appear
    assert big not in line


def test_verbose_env_var_enables(monkeypatch):
    """AGENTDAG_VERBOSE=1 flips verbose on without an explicit flag."""
    monkeypatch.setenv("AGENTDAG_VERBOSE", "1")
    buf = io.StringIO()
    tr = Tracer(stream=buf)
    assert tr.verbose is True
    tr.emit("started", "n1", attempt=1)
    assert "started" in buf.getvalue()


def test_verbose_env_var_off_when_unset(monkeypatch):
    monkeypatch.delenv("AGENTDAG_VERBOSE", raising=False)
    buf = io.StringIO()
    tr = Tracer(stream=buf)
    assert tr.verbose is False


def test_verbose_explicit_overrides_env(monkeypatch):
    """An explicit verbose=False must win over AGENTDAG_VERBOSE=1."""
    monkeypatch.setenv("AGENTDAG_VERBOSE", "1")
    buf = io.StringIO()
    tr = Tracer(verbose=False, stream=buf)
    assert tr.verbose is False
    tr.emit("log", "n1", message="hi")
    assert buf.getvalue() == ""


def test_verbose_does_not_break_persistence(tmp_path):
    """Verbose mode must still write the JSONL trace file."""
    path = tmp_path / "trace.jsonl"
    buf = io.StringIO()
    tr = Tracer(path, verbose=True, stream=buf)
    tr.emit("log", "n1", message="hello")
    # both the stream and the file got the event
    assert "n1" in buf.getvalue()
    assert path.exists()
    assert "hello" in path.read_text()
