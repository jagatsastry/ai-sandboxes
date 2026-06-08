import pytest
from agentdag import DAG, NodeFailed, RetryPolicy, Scheduler


def test_linear_dag_runs_in_order():
    dag = DAG()
    order = []

    def make(name):
        def fn(ctx):
            order.append(name)
            return name

        return fn

    dag.add("a", make("a"))
    dag.add("b", make("b"), deps=["a"])
    dag.add("c", make("c"), deps=["b"])

    sched = Scheduler(dag, workers=2)
    out = sched.run()
    assert order == ["a", "b", "c"]
    assert out["c"] == "c"


def test_parallel_branches_both_complete():
    dag = DAG()
    dag.add("root", lambda ctx: "r")
    dag.add("l", lambda ctx: "L", deps=["root"])
    dag.add("r", lambda ctx: "R", deps=["root"])
    dag.add("join", lambda ctx: ctx.bb.get("l") + ctx.bb.get("r"), deps=["l", "r"])
    out = Scheduler(dag, workers=4).run()
    assert out["join"] == "LR"


def test_retry_eventually_succeeds():
    dag = DAG()
    state = {"n": 0}

    def flaky(ctx):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("not yet")
        return "ok"

    dag.add("f", flaky, retry=RetryPolicy(max_attempts=3, backoff_ms=1))
    out = Scheduler(dag, workers=2).run()
    assert out["f"] == "ok"
    assert state["n"] == 3


def test_retry_exhausted_raises():
    dag = DAG()

    def always_bad(ctx):
        raise RuntimeError("nope")

    dag.add("f", always_bad, retry=RetryPolicy(max_attempts=2))
    with pytest.raises(NodeFailed):
        Scheduler(dag, workers=2).run()


def test_cycle_detected():
    dag = DAG()
    dag.add("a", lambda ctx: 1, deps=["b"])
    dag.add("b", lambda ctx: 2, deps=["a"])
    with pytest.raises(ValueError):
        dag.validate()
