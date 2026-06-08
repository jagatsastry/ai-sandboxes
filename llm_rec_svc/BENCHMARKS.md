# Benchmark results

All numbers from this repo's sandbox env: **2 vCPU, 8 GB RAM, CPU-only**
(`cuda_available: false`). Real GPU numbers will be 5–20× faster.

## 1. Single-request latency by K (number of candidates reranked)

### tiny-gpt2 (~5M params) — sanity baseline

| cands | top_k | total p50 | total p95 | gpu p50 | gpu p95 | embed p50 | tok p50 | cand p50 |
|------:|------:|----------:|----------:|--------:|--------:|----------:|--------:|---------:|
| 5  | 5  |  35.6 ms |  42.3 ms |  20.3 ms |  26.6 ms | 10.0 ms | 1.0 ms | 0.09 ms |
| 10 | 5  |  62.3 ms |  73.4 ms |  41.8 ms |  53.0 ms | 10.3 ms | 1.5 ms | 0.10 ms |
| 20 | 5  | 138.9 ms | 158.6 ms | 106.9 ms | 122.9 ms | 12.4 ms | 2.7 ms | 0.10 ms |
| 25 | 5  | 148.9 ms | 186.4 ms | 114.8 ms | 145.6 ms | 10.6 ms | 3.0 ms | 0.10 ms |

Throughput: **8.1 req/s** sequential at cands=20.

### Qwen2.5-0.5B-Instruct (~500M params) — real model

| cands | top_k | total p50 | total p95 | gpu p50 | gpu p95 |
|------:|------:|----------:|----------:|--------:|--------:|
| 5  | 5 | 1,529 ms | 1,555 ms | 1,503 ms | 1,527 ms |
| 10 | 5 | 2,879 ms | 2,955 ms | 2,841 ms | 2,915 ms |
| 20 | 5 | 6,283 ms | 6,499 ms | 6,222 ms | 6,437 ms |

Per-candidate cost on CPU: **~310 ms**. GPU stage = **99% of latency**.

### Ranking quality (same query)

> "backend engineer prepping for AI infra system design interviews"

| Rank | tiny-gpt2 (random-init) | Qwen2.5-0.5B (real) |
|---|---|---|
| 1 | Elements of Programming Interviews | System Design Interview Vol 2 ✓ |
| 2 | Designing Data-Intensive Applications | Building LLM Powered Applications ✓ |
| 3 | **The Phoenix Project** ✗ (devops fiction) | Designing Data-Intensive Applications ✓ |
| 4 | Building LLM Powered Applications | System Design Interview Vol 1 ✓ |
| 5 | Database Internals | Search Engines: IR in Practice ✓ |

A tiny LLM gives you the architecture but not the quality; a real one earns
the two-stage design's keep.

## 2. Concurrency-vs-latency (tiny-gpt2, cands=20, K=5, 24 reqs per level)

```
conc  reqs   thru rps  ||   wall p50     p95     p99
   1    24       7.53          123.0   146.8   163.8
   2    24       8.14          235.9   327.5   346.6
   4    24       7.65          493.9   653.1   716.2
   8    24       6.33         1103.7  1714.7  1741.9
```

**From c=1 to c=8 (8× concurrency):**
- p50 wall latency: **+9.0×** (123 ms → 1,104 ms)
- throughput: **0.84×** — drops slightly under contention

### Interpretation

This is the **classic signature of a single serialized model worker**.
Requests pile up in front of one CPU-bound model forward pass. Adding
in-flight requests does *not* add capacity; it only adds queue depth, which
shows up as proportional latency growth.

The gpu_ms field in the response confirms this: at c=1, gpu_ms ≈ 92 ms; at
c=8, gpu_ms ≈ 912 ms — almost exactly 8× — because each request waits its
turn through the GIL + PyTorch matmul.

### What would change this

| Lever | Effect |
|---|---|
| **More uvicorn workers** (`--workers N`) | Linear throughput up to #cores, no batching benefit |
| **GPU** | 5–20× cheaper per forward pass, headroom for >1 in flight |
| **Continuous batching (vLLM / TGI)** | Concurrent requests get *merged* into one GPU forward → throughput becomes ~constant in concurrency, latency only mildly rises. This is the prod answer. |
| **Smaller K** | Linear win — halving K halves per-request cost, doubles throughput |
| **Quantization (int8 / fp8)** | 2–4× per-forward speedup, small quality cost |

## 3. Where the time goes (Qwen, K=20)

```
total 6,283 ms = embed       14 ms (0.2%)
              + candidate     0.1 ms (0.0%)
              + tokenize      3 ms (0.05%)
              + gpu        6,222 ms (99.0%)
              + framework   ~44 ms (0.7%)
```

Optimization budget = the LLM forward pass. Everything else is noise.

## How to reproduce

```bash
# fast (tiny-gpt2)
bash llm_rec_svc/scripts/run_e2e.sh

# real model
LLM=Qwen/Qwen2.5-0.5B-Instruct bash llm_rec_svc/scripts/run_e2e.sh

# just the concurrency benchmark (service must already be up)
python llm_rec_svc/scripts/bench_concurrency.py --cands 20 --reqs 24 --levels 1,2,4,8
```
