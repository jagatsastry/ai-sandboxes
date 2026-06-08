# ai-sandboxes

Small, self-contained sandboxes for experimenting with AI infrastructure patterns.
Each subproject is independently runnable.

## Projects

### [`llm_rec_svc/`](./llm_rec_svc) — GPU LLM recommendation service
Two-stage low-latency recommender: dense retrieval (MiniLM) → LLM pointwise
reranker (batched, fp16 on CUDA, autodetects CPU). FastAPI service with
per-stage latency in every response, `/healthz`, `/metrics`, and a benchmark
harness. Swap the reranker model via env var.

- **Status:** working, benchmarked (CPU + Qwen2.5-0.5B numbers in the README).
- **Run:** `cd llm_rec_svc && pip install -r requirements.txt && uvicorn app.main:app`

### [`agent_dag_sandbox/`](./agent_dag_sandbox) — local DAG of coding agents
A tiny framework for testing multi-agent architectures locally: declarative
DAG, thread-pool scheduler with retries, shared blackboard memory, structured
tracing, and pluggable agent roles (planner / breakdown / TDD test-writer /
coder / adversary / verifier) wired into a **council** voting pattern. Includes
an end-to-end sample workflow that solves a small coding problem and prints a
trace showing where time went.

- **Status:** core + sample workflow + tests in place.
- **Run:** `cd agent_dag_sandbox && python -m examples.balanced_parens`

## License

MIT
