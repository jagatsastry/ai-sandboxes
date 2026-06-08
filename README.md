# ai-sandboxes

Small, self-contained sandboxes for experimenting with AI infrastructure
patterns. Two independent projects:

| Project | What it is | Best for |
|---|---|---|
| [`llm_rec_svc/`](./llm_rec_svc) | Two-stage GPU-aware recommendation service (FastAPI) with cross-request batching | Studying inference-serving tradeoffs: latency vs. throughput, batching, concurrency curves |
| [`agent_dag_sandbox/`](./agent_dag_sandbox) | Local DAG of coding agents with a shared blackboard, tracer, and pluggable LLM | Studying multi-agent orchestration patterns: planner/coder/verifier councils, repair loops, traces |

Status: both projects working, tests green, CI lint + tests on every push.
Tests: **49** (agent sandbox) + **15** (rec service). Lint: ruff + black across
31 files.

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
follow the commands. Every example below shows the actual output you'll see.

> **Verbose mode (new):** every example here accepts `--verbose` / `-v` (or
> `AGENTDAG_VERBOSE=1` / `VERBOSE=1` env vars) for live, per-event logging.
> See [Verbose mode](#verbose-mode) below.

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
  -d '{"profile":"Senior backend engineer into distributed systems and ranking","top_k":3,"n_candidates":10}'
```

**Output** (CPU, default tiny-gpt2 reranker):

```json
{"ready": true, "device": "cpu", "items": 25, "cuda_available": false, "batch_enabled": true}
```

```json
{
  "items": [
    {"id": "i001", "title": "Designing Data-Intensive Applications", "final_score": 0.93, "retrieval_score": 0.54, ...},
    {"id": "i013", "title": "Building LLM Powered Applications",      "final_score": 0.90, "retrieval_score": 0.34, ...},
    {"id": "i024", "title": "The Phoenix Project",                    "final_score": 0.84, "retrieval_score": 0.31, ...}
  ],
  "timings": {
    "embed_ms": 17.4, "candidate_ms": 0.3, "tokenize_ms": 2.3,
    "gpu_ms": 110.6, "queue_ms": 128.1, "batch_size": 10, "total_ms": 145.9
  },
  "device": "cpu", "n_candidates": 10
}
```

Every response includes per-stage latency so you can see where time goes.

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

Boot logs (with `VERBOSE=1`):

```
[boot] loading embedder: sentence-transformers/all-MiniLM-L6-v2
[boot] loading catalog: data/items.json
[boot] loading LLM: sshleifer/tiny-gpt2
[boot] device: cpu  items: 25
[boot] warmup...
[boot] starting batcher: max_size=64 max_wait_ms=5.0
[boot] ready
```

### 3. Benchmark latency at one concurrency level

```bash
python scripts/bench.py -n 10 --cands 10 --topk 5
```

**Output:**

```
n=10  topk=5  cands=10
throughput: 14.5 req/s
  total  p50=  66.9ms  p95=  77.1ms  avg=  67.5ms
  gpu    p50=  41.0ms  p95=  48.2ms  avg=  41.3ms
  embed  p50=  10.7ms  p95=  13.8ms  avg=  11.1ms
  cand   p50=   0.2ms  p95=   0.2ms  avg=   0.2ms
```

Per-stage breakdown shows GPU/CPU reranking dominates (~40 ms), with embedding
~10 ms and candidate gen sub-ms.

Add `-v` for per-request traces:

```
[  1/10] client=  76.6ms server=  75.1ms (embed= 12.9 cand=  0.2 tok=  1.6 gpu= 46.0 queue= 61.9 batch=10)
[  2/10] client=  71.1ms server=  69.6ms (embed= 12.0 cand=  0.2 tok=  1.5 gpu= 42.7 queue= 57.4 batch=10)
...
```

### 4. Study the concurrency curve (the main learning use case)

This is what shows you the latency-vs-throughput tradeoff of a single model
worker.

```bash
python scripts/bench_concurrency.py --levels 1,2,4,8 --reqs 8
```

**Output (single-worker CPU baseline — the canonical "model is the bottleneck" signature):**

```
conc  reqs   thru rps  ||   wall p50     p95     p99  ||   srv p50     p95  ||   gpu p50   embed p50
----------------------------------------------------------------------------------------------------
   1     8      13.41  ||       67.3    80.0    80.0  ||      65.8    75.8  ||      40.4        10.4
   2     8      14.87  ||      127.9   153.7   153.7  ||     126.1   151.3  ||      55.7        28.7
   4     8       9.66  ||      471.9   558.5   558.5  ||     470.1   552.0  ||     161.1        33.9
   8     8      10.14  ||      753.5   782.6   782.6  ||     746.2   780.4  ||     471.9        59.8

From c=1 to c=8 (8x concurrency):
  p50 wall latency multiplied by 11.19x
  throughput multiplied by      0.76x
  -> classic single-worker serialization. Concurrency does NOT add throughput;
     it only adds queue latency. Wins require: more workers, async batching, or GPU.
```

Latency rose 11x while throughput went *backwards* — exactly the shape you'd
expect from a single model worker on CPU.

Add `-v` to see individual requests as they complete, including how the
batcher coalesces them:

```
[c= 4 1/4] wall= 101.1ms srv=  99.2ms (gpu= 50.9 tok=  2.6 queue= 65.8 batch=10)
[c= 4 2/4] wall= 282.0ms srv= 279.2ms (gpu=151.8 tok=  4.5 queue=216.9 batch=30)  <-- 3 reqs coalesced
[c= 4 3/4] wall= 264.6ms srv= 261.7ms (gpu=151.8 tok=  4.5 queue=235.3 batch=30)
[c= 4 4/4] wall= 267.4ms srv= 265.1ms (gpu=151.8 tok=  4.5 queue=235.6 batch=30)
```

### 5. Study cross-request batching

The async batcher is on by default. Compare with/without:

```bash
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

The flagship workflow: planner → breakdown → test-writer council → coder
(intentionally buggy) → adversary → patched coder → verifier.

```bash
cd agent_dag_sandbox
pip install -r requirements.txt
python -m examples.balanced_parens
```

**Output (tail):**

```
=== FINAL VERIFIER REPORT ===
{ "total": 17, "passed": 17, "failed": 0, "ok": true }

=== ADVERSARY FOUND ===
{
  "cases_tried": 4,
  "failures": [{"input": ")(", "expected": false, "got": true}]
}

=== TEST COUNCIL MERGED SUITE ===
13 unique cases after union of 3 writers

=== TRACE (where time went) ===
node           status    attempts  retries  queue_ms  run_ms
-------------  --------  --------  -------  --------  ------
planner        finished  1         0        0.2       5.3
breakdown      finished  1         0        0.0       5.2
test_council   finished  1         0        0.0       6.1
coder          finished  1         0        0.1       5.2
adversary      finished  1         0        0.0       5.3
patched_coder  finished  1         0        0.0       5.1
verifier       finished  1         0        0.0       0.3

total wall: 27.0 ms
sum of run_ms (work done): 32.4 ms
```

The adversary caught the buggy initial coder on `")("` , the patched coder
fixed it, and the verifier confirmed all 17 union-merged tests pass.

### 2. Same demo, but with a *real* LLM as the coder

Accepts any OpenAI-compatible client; falls back to a deterministic coder if
the LLM call fails or returns unparseable code.

```bash
# Offline / deterministic (default — uses EchoLLM, falls back to deterministic Coder)
python -m examples.balanced_parens_llm

# Hosted (OpenAI)
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... \
  python -m examples.balanced_parens_llm

# Local (Ollama on :11434)
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:0.5b \
  python -m examples.balanced_parens_llm
```

**Output (mock provider — default):**

```
[setup] using LLM: mock-llm

=== FINAL VERIFIER REPORT ===
{ "total": 17, "passed": 17, "failed": 0, "ok": true }

=== CODER OUTPUT (final) ===
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
```

Run with `-v` to see the LLM fallback path being taken (since mock-llm doesn't
return real code):

```
[t+   16.3ms] log  coder          attempt=1  message='llm fallback engaged' reason='no-code-block'
[t+   32.0ms] log  patched_coder  attempt=1  message='llm fallback engaged' reason='no-code-block'
```

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

**Output:**

```
=== INITIAL TEST RUN (before patch) ===
  returncode: 1
  failing_cases: ['tests/test_fizzbuzz.py::test_fizzbuzz_fifteen']

=== BUG LOCALIZER (top 3 suspects) ===
  [1.00] fizzbuzz.py::fizzbuzz  (imported by failing test tests/test_fizzbuzz.py; name appears in failing test id)

=== PATCH PROPOSED & APPLIED ===
{ "applied": true, "file": "fizzbuzz.py", "symbol": "fizzbuzz",
  "rationale": "fixer-book entry for fizzbuzz" }

=== REGRESSION GUARD ===
{ "ok": true, "returncode": 0, "n_failures": 0, "failing_cases": [],
  "short_traceback": "....    [100%]\n4 passed in 0.01s\n" }

=== TRACE (where time went) ===
node              status    attempts  retries  queue_ms  run_ms
----------------  --------  --------  -------  --------  ------
test_runner       finished  1         0        0.2       1474.7
bug_localizer     finished  1         0        0.1       0.3
patch_proposer    finished  1         0        0.0       0.1
patch_applier     finished  1         0        0.0       0.2
regression_guard  finished  1         0        0.0       1448.5

total wall: 2924.4 ms
sum of run_ms (work done): 2923.8 ms
```

The trace makes it obvious: the two pytest invocations dominate (~1.5 s each
on a cold VM), while every other agent finishes in sub-millisecond time.

### 4. Analyze a trace from any run

Every example writes a JSONL trace to `traces/`. You can post-mortem it:

```bash
python -m agentdag.analyze traces/fix_failing_test.jsonl
```

**Output:**

```
node              status    attempts  retries  queue_ms  run_ms
----------------  --------  --------  -------  --------  ------
test_runner       finished  1         0        0.6       1478.4
bug_localizer     finished  1         0        0.1       0.3
patch_proposer    finished  1         0        0.1       0.1
patch_applier     finished  1         0        0.0       0.2
regression_guard  finished  1         0        0.0       1452.1

total wall: 2932.2 ms
sum of run_ms (work done): 2931.2 ms

slowest node: test_runner  (1478.4 ms)
```

### 5. Wire your own agent / DAG

The framework is the point — examples are just demos. Minimal recipe:

```python
from agentdag import DAG, Blackboard, Scheduler, Tracer

def my_agent(ctx):
    return {"doubled": ctx.bb.get("x") * 2}

dag = DAG()
dag.add("seed", lambda ctx: 21)
dag.add("doubler", my_agent, deps=["seed"])

bb, tracer = Blackboard(), Tracer()
bb.put("x", 21)
out = Scheduler(dag, blackboard=bb, tracer=tracer).run()
print(out["doubler"])      # -> {'doubled': 42}
print(tracer.render_table())
```

See `agentdag/agents/repair.py` for a fully worked five-agent example
(dataclass-typed inputs/outputs, retries, AST-based code rewriting, clean
subprocess env for `pytest`).

---

## Verbose mode

Both projects ship a verbose-logging mode for digging into runs without
post-hoc trace analysis.

### Agent sandbox: `-v` / `--verbose` / `AGENTDAG_VERBOSE=1`

`Tracer(verbose=True)` (and the env var) live-stream every DAG event to
stderr as it happens, with a `[t+<ms>]` prefix relative to run start. Useful
when you want to see parallelism unfold in real time.

```bash
python -m examples.balanced_parens --verbose
# or
AGENTDAG_VERBOSE=1 python -m examples.balanced_parens
```

**Output (excerpt — shows parallel branches and the council coalescing):**

```
[t+    0.0ms] enqueued planner            attempt=1
[t+    0.2ms] started  planner            attempt=1
[t+    5.5ms] log      planner            attempt=1  message='plan drafted' steps=6
[t+    5.5ms] finished planner            attempt=1
[t+   10.8ms] finished breakdown          attempt=1
[t+   10.9ms] enqueued test_council       attempt=1
[t+   10.9ms] enqueued coder              attempt=1     <-- enqueued together
[t+   10.9ms] started  test_council       attempt=1
[t+   11.0ms] started  coder              attempt=1     <-- both running on different workers
[t+   16.6ms] log      test_council       attempt=1  message='tests written' n=7 focus='general'
[t+   16.7ms] log      test_council       attempt=1  message='tests written' n=8 focus='edge'
[t+   16.8ms] log      test_council       attempt=1  message='tests written' n=8 focus='mixed'
[t+   17.0ms] log      test_council       attempt=1  message='council aggregated' n_outputs=3
[t+   21.5ms] log      adversary          attempt=1  message='adversary ran' failures=1
[t+   27.2ms] log      verifier           attempt=1  message='verified' total=17 passed=17 failed=0 ok=True
```

Long payloads are truncated to keep lines scannable. The JSONL trace file
still has full event payloads.

### Rec service: `VERBOSE=1`

Set on the uvicorn process; logs a one-liner per request with all stage
timings:

```bash
VERBOSE=1 uvicorn app.main:app --port 8000
```

**Output:**

```
[boot] loading embedder: sentence-transformers/all-MiniLM-L6-v2
[boot] loading catalog: data/items.json
[boot] loading LLM: sshleifer/tiny-gpt2
[boot] device: cpu  items: 25
[boot] starting batcher: max_size=64 max_wait_ms=5.0
[boot] ready
[req] cands=10 top_k=3 embed=17.4ms cand=0.3ms tok=2.3ms gpu=110.6ms queue=128.1ms batch=10 total=145.9ms  top1='Designing Data-Intensive Applications'
```

### Benchmarks: `-v` / `--verbose`

`bench.py` and `bench_concurrency.py` both accept `-v` to print every
individual request with full stage breakdown — see Use case 3 and 4 above
for the exact output.

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
│   └── tests/                  49 tests (incl. 7 verbose-tracer tests)
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
