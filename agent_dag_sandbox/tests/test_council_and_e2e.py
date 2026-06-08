from agentdag import DAG, Blackboard, Scheduler, Tracer
from agentdag.agents import Council, TestWriter
from agentdag.agents.roles import union_tests


def test_test_council_union_dedups():
    bb = Blackboard()
    bb.put("task_spec", {"goal": "demo"})
    dag = DAG()
    dag.add(
        "test_council",
        Council(
            members=[TestWriter(focus="general"),
                     TestWriter(focus="edge"),
                     TestWriter(focus="mixed")],
            aggregator=union_tests,
        ),
    )
    out = Scheduler(dag, blackboard=bb, workers=3).run()
    cases = out["test_council"]
    assert len(cases) > 5
    # no duplicates
    keys = [(repr(i), repr(o)) for i, o in cases]
    assert len(keys) == len(set(keys))


def test_majority_council():
    bb = Blackboard()
    dag = DAG()
    dag.add(
        "vote",
        Council(
            members=[lambda ctx: "A",
                     lambda ctx: "A",
                     lambda ctx: "B"],
        ),
    )
    out = Scheduler(dag, blackboard=bb).run()
    assert out["vote"] == "A"


def test_end_to_end_balanced_parens():
    from examples.balanced_parens import build_dag

    bb = Blackboard()
    bb.put("task_spec", {"goal": "is_balanced_parens"})
    tracer = Tracer()
    out = Scheduler(build_dag(), blackboard=bb, tracer=tracer, workers=4).run()
    rep = out["verifier"]
    assert rep["ok"], rep
    assert rep["failed"] == 0
    # adversary must have caught the buggy first coder
    assert len(out["adversary"]["failures"]) > 0
    # tracer must contain a finished event for every node
    finished = {e.node for e in tracer.events if e.kind == "finished"}
    assert {"planner", "breakdown", "test_council", "coder",
            "adversary", "patched_coder", "verifier"}.issubset(finished)
