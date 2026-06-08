from agentdag import Blackboard


def test_put_get_version():
    bb = Blackboard()
    assert bb.version == 0
    bb.put("a", 1)
    assert bb.get("a") == 1
    assert bb.version == 1
    bb.put("a", 2)
    assert bb.version == 2


def test_snapshot_is_isolated():
    bb = Blackboard()
    bb.put("xs", [1, 2])
    v, snap = bb.snapshot()
    snap["xs"].append(999)         # mutating snapshot must not affect bb
    assert bb.get("xs") == [1, 2]


def test_append_typecheck():
    bb = Blackboard()
    bb.append("xs", 1)
    bb.append("xs", 2)
    assert bb.get("xs") == [1, 2]
    bb.put("y", "string")
    try:
        bb.append("y", 3)
    except TypeError:
        return
    raise AssertionError("expected TypeError")
