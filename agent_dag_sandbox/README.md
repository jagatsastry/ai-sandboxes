# agent_dag_sandbox

A tiny local framework for testing multi-agent coding architectures. Lets you
wire LLM-style agent roles into a DAG with queues, retries, shared memory, and
structured traces — so you can experiment with patterns (council voting, TDD,
adversaries, patching loops) without paying for real LLM calls.

All built-in agents are **deterministic Python mocks** that follow the same
interface a real LLM-backed agent would. Replace the body of any agent with an
OpenAI/Anthropic call and the rest of the framework keeps working.

## What's in it

| Primitive | File | What it does |
|---|---|---|
| `Blackboard` | `agentdag/blackboard.py` | Thread-safe shared memory with versioning + isolated snapshots |
| `Tracer` | `agentdag/tracer.py` | Structured event log (`enqueued`/`started`/`finished`/`retry`/`failed`/`log`) → JSONL + analyzer |
| `DAG` / `Node` / `RetryPolicy` | `agentdag/dag.py` | Declarative graph; per-node retry policy with exception filter |
| `Scheduler` | `agentdag/scheduler.py` | Thread-pool executor; queue-driven, cancels DAG on hard failure |
| Agent roles | `agentdag/agents/roles.py` | Planner, Breakdown, CaseWriter, Coder, Adversary, Verifier, Council |
| Trace CLI | `agentdag/analyze.py` | `python -m agentdag.analyze trace.jsonl` |

## Sample workflow (TDD + adversary + council)

`examples/balanced_parens.py` solves `is_balanced_parens(s)` end-to-end:

```
planner ─► breakdown ─┬─► test_council (3 CaseWriters → union) ─┐
                      │                                          ├─► verifier
                      └─► coder (BUGGY) ─► adversary ─► patched_coder ─┘
```

- `test_council` runs **3 CaseWriters in parallel** (focus=general/edge/mixed)
  and merges their suites via union.
- `coder` intentionally emits a **buggy** first implementation
  (`s.count('(') == s.count(')')`, fails `")("`).
- `adversary` runs red-team cases, detects the failure, writes
  `adversary_failures` to the blackboard.
- `patched_coder` reads that flag and produces the correct implementation.
- `verifier` runs the full merged test suite + adversarial cases.

### What you get

```
=== FINAL VERIFIER REPORT ===
{ "total": 17, "passed": 17, "failed": 0, "ok": true }

=== ADVERSARY FOUND ===
{ "cases_tried": 4, "failures": [{ "input": ")(", "expected": false, "got": true }] }

=== TRACE (where time went) ===
node           status    attempts  retries  queue_ms  run_ms
planner        finished  1         0        0.2       5.3
breakdown      finished  1         0        0.0       5.2
test_council   finished  1         0        0.1       6.1
coder          finished  1         0        0.1       5.2
adversary      finished  1         0        0.0       5.3
patched_coder  finished  1         0        0.0       5.1
verifier       finished  1         0        0.0       0.3
total wall: 27.1 ms
```

The trace shows the full sequence ran in ~27 ms wall (each agent has a 5 ms
simulated "LLM latency" baked in). `queue_ms` is time spent waiting for a
worker; `run_ms` is real work. Plug in a real LLM and these numbers become
informative.

## Run it

```bash
cd agent_dag_sandbox

# install only needs python>=3.10 + pytest (no deps for the framework itself)
pip install pytest

# all-in-one: tests + example + trace analysis
bash scripts/run_e2e.sh

# or step by step:
python -m pytest -q
python -m examples.balanced_parens
python -m agentdag.analyze traces/balanced_parens.jsonl
```

## Adding your own workflow

```python
from agentdag import DAG, Scheduler, Blackboard, Tracer, RetryPolicy

dag = DAG("my_workflow")

def my_agent(ctx):
    spec = ctx.bb.get("task_spec")
    ctx.log("thinking", spec=spec)
    return {"answer": 42}

dag.add("my_agent", my_agent,
        retry=RetryPolicy(max_attempts=3, backoff_ms=50))
dag.add("downstream", lambda ctx: ctx.bb.get("my_agent")["answer"] * 2,
        deps=["my_agent"])

bb = Blackboard(); bb.put("task_spec", {"goal": "demo"})
out = Scheduler(dag, blackboard=bb, tracer=Tracer("traces/demo.jsonl")).run()
print(out["downstream"])  # 84
```

### Swapping in a real LLM

Implement the `LLMClient` protocol (`agentdag/agents/llm.py`) — a single
`.complete(prompt, system, temperature)` method — and inject it into any
agent role's constructor. The mock `EchoLLM` is the reference implementation.

## Patterns the sandbox supports out of the box

- **Council voting** — N agents in parallel, pluggable aggregator
  (majority vote by default, `union_tests` for set-merge, BYO for anything else).
- **TDD** — TestCaseWriter (renamed `CaseWriter`) emits cases *before* the
  coder runs; verifier executes them.
- **Adversary / red-team** — generates extra hard cases, runs them, publishes
  failures so a downstream coder can patch.
- **Retry with exception filter** — `RetryPolicy(max_attempts=3, retry_on={"TimeoutError"})`.
- **Fan-out / fan-in** — natural with the DAG; the sample uses both.

## Tests

```bash
python -m pytest -q
```

11 tests covering: blackboard semantics, DAG cycle detection + topo order,
scheduler retries / failure propagation / parallel branches, council
aggregation, and a full end-to-end run of the sample workflow.
