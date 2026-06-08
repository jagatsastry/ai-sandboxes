# ai-sandboxes

Small, self-contained sandboxes for experimenting with AI infrastructure
patterns. Two independent projects:

| Project | What it is | Best for |
|---|---|---|
| [`llm_rec_svc/`](./llm_rec_svc) | Two-stage GPU-aware recommendation service (FastAPI) with cross-request batching | Studying inference-serving tradeoffs: latency vs. throughput, batching, concurrency curves |
| [`agent_dag_sandbox/`](./agent_dag_sandbox) | Local DAG of coding agents with a shared blackboard, tracer, and pluggable LLM | Studying multi-agent orchestration patterns: planner/coder/verifier councils, repair loops, traces |

Status: both projects working, tests green, CI lint + tests on every push.
Tests: **42** (agent sandbox) + **15** (rec service). Lint: ruff + black across
30 files.

---

## Quick start

```bash
git clone https://github.com/jagatsastry/ai-sandboxes.git
cd ai-sandboxes

# Either project is independent:
cd llm_rec_svc        && pip install -r requirements.txt
# or
cd agent_dag_sandbox  && pip install -r requirements.txt
```

Each project has its own README with deeper notes. This top-level doc is a
**use-case index** — pick the scenario closest to what you want to try and
follow the commands.

---

## `llm_rec_svc/` — use cases

A two-stage recommender: dense retrieval (MiniLM) → LLM pointwise reranker
(batched, fp16 on CUDA, autodetects CPU). FastAPI service with per-stage
latency in every response, `/healthz`, `/metrics`, and a benchmark harness.

### 1. Just see it work (smoke test)

```bash
cd llm_rec_svc
pip install -r requirements.txt
uvicorn app.main:app --port 8000 &
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/recommend \
  -H 'content-type: application/json' \
  -d '{"profile":"backend engineer into distributed systems","top_k":3}' | jq
```

Every response includes per-stage latency (`embed_ms`, `candidate_ms`,
`gpu_ms`, `tokenize_ms`) so you can see where time goes.

### 2. Swap the reranker model

The reranker is selectable by env var. fp16 + CUDA is autodetected.

```bash
# Default tiny GPT-2 (CPU-friendly)
uvicorn app.main:app

# Half-billion-param Qwen reranker
LLM_MODEL=Qwen/Qwen2.5-0.5B uvicorn app.main:app

# Anything else from Hugging Face that supports causal LM
LLM_MODEL=gpt2-medium uvicorn app.main:app
```

### 3. Benchmark latency at one concurrency level

```bash
python scripts/bench.py --n 50 --cands 10 --topk 5
# -> p50 / p95 / p99 latency, plus per-stage breakdown
```

### 4. Study the concurrency curve (the main learning use case)

This is what shows you the latency-vs-throughput tradeoff of a single model
worker.

```bash
python scripts/bench_concurrency.py --levels 1,2,4,8,16 --reqs 30
```

Output: per-level p50/p95/p99 + throughput table. With **sync** uvicorn +
one model worker on CPU, throughput is flat and p50 rises linearly — the
canonical "model is the bottleneck" signature.

### 5. Study cross-request batching

The async batcher is on by default in the current code. Compare with/without:

```bash
# A/B run: batched vs. serial
./scripts/bench_ab_batched.sh
```

In the local CPU benchmark this gave **+29% throughput** and **−41% p95** at
c=8 — and the win compounds on GPU because batched matmul is closer to free.
See [`llm_rec_svc/BENCHMARKS.md`](./llm_rec_svc/BENCHMARKS.md) for raw
numbers.

### 6. Drive a full sweep (CSV + plot-ready)

```bash
python scripts/bench_sweep.py        # sweeps concurrency × candidate count
```

Useful for picking an operating point before deploying.

---

## `agent_dag_sandbox/` — use cases

A tiny framework for multi-agent architectures: declarative **DAG**,
thread-pool **scheduler** with retries, shared **blackboard** memory,
structured **tracer**, and pluggable agent roles wired into a **council**
voting pattern.

### 1. Run the canonical "build from scratch" demo (TDD council)

The flagship workflow: planner → breakdown → test-writer → coder council
(N coders + adversary + verifier voting) on a small coding problem.

```bash
cd agent_dag_sandbox
pip install -r requirements.txt
python -m examples.balanced_parens
# Prints final code + a per-stage trace showing where time went.
```

### 2. Same demo, but with a *real* LLM as the coder

The coder role accepts any OpenAI-compatible client. If the LLM call fails
or returns unparseable code, the workflow falls back to a deterministic
coder — so this script always finishes whether or not you have an API key.

```bash
# Offline / deterministic (default)
python -m examples.balanced_parens_llm

# Hosted (OpenAI)
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
  python -m examples.balanced_parens_llm

# Local (Ollama on :11434)
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:0.5b \
  python -m examples.balanced_parens_llm
```

Inspect the trace's `log` events to see which path was taken on each step
and how the council voted.

### 3. Run the "fix a failing test" workflow (maintainer-shaped repair loop)

Dual to the TDD demo: agents are handed a broken project + a known-failing
test and must localize the bug, propose a patch, apply it, and verify the
full suite still passes.

```bash
python -m examples.fix_failing_test
```

DAG shape:

```
test_runner → bug_localizer → patch_proposer → patch_applier → regression_guard → report
```

The fixture is a classic buggy `fizzbuzz` (the `%3` branch fires before
`%15`). Watch the trace to see localization scores, patch candidates, and
the post-patch verdict.

### 4. Analyze a trace from any run

Every example writes a JSONL trace to `traces/`. You can post-mortem it:

```bash
python -m agentdag.analyze traces/balanced_parens.jsonl
python -m agentdag.analyze traces/fix_failing_test.jsonl
```

Output: per-agent wall time, retries, total span, critical path.

### 5. Wire your own agent / DAG

The framework is the point — examples are just demos. Minimal recipe:

```python
from agentdag.dag import DAG
from agentdag.scheduler import Scheduler
from agentdag.blackboard import Blackboard
from agentdag.tracer import Tracer

def my_agent(bb, **inputs):
    bb["my_output"] = inputs["x"] * 2
    return {"ok": True}

dag = DAG()
dag.add_node("doubler", my_agent, inputs={"x": 21})

bb, tracer = Blackboard(), Tracer()
Scheduler(dag, bb, tracer).run()
print(bb["my_output"])      # -> 42
print(tracer.to_jsonl())    # structured trace
```

See `agentdag/agents/repair.py` for a fully worked five-agent example
(dataclass-typed inputs/outputs, retries, AST-based code rewriting, clean
subprocess env for `pytest`).

---

## Repo layout

```
ai-sandboxes/
├── llm_rec_svc/                two-stage recommender + benchmarks
│   ├── app/                    FastAPI service, batcher, scorer, catalog
│   ├── scripts/                bench.py, bench_concurrency.py, bench_sweep.py
│   ├── tests/                  15 tests (incl. batcher + length-mismatch regression)
│   └── BENCHMARKS.md
├── agent_dag_sandbox/          DAG / blackboard / scheduler / agents
│   ├── agentdag/               framework
│   │   ├── agents/             planner, coder, verifier, repair, …
│   │   ├── dag.py, scheduler.py, blackboard.py, tracer.py, analyze.py
│   ├── examples/               balanced_parens, balanced_parens_llm, fix_failing_test
│   └── tests/                  42 tests
├── .github/workflows/ci.yml    lint (ruff + black) + tests on every push
├── pyproject.toml              shared ruff/black config
└── Makefile                    `make lint`, `make test`, `make fmt`
```

## Dev loop

```bash
make fmt         # black .
make lint        # ruff check . && black --check .
make test        # pytest in both projects
```

Pre-commit hooks (optional):

```bash
pip install pre-commit && pre-commit install
```

## License

MIT
