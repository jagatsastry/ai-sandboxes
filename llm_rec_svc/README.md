# llm-rec-svc

A very small two-stage recommendation service: cheap dense retrieval +
LLM-based pointwise reranking on GPU (auto-falls-back to CPU).

## Architecture

```
                 +-------------------+
 profile text -> | sentence-transf.  | -> 384-d query vec
                 +-------------------+
                          |
                          v
                 +-------------------+
                 |  cosine top-N     |  (numpy matmul over item embeddings)
                 +-------------------+
                          |
                          v  (N candidates)
                 +-------------------+
                 |  LLM scorer       |  causal LM, single batched forward pass,
                 |  fp16 on CUDA     |  score = log p(yes) - log p(no)
                 +-------------------+
                          |
                          v
                 blend(llm, retrieval) -> top-K
```

## Why these choices

- **Two stages.** Pure-LLM ranking over the whole catalog blows latency
  budgets fast. Retrieval narrows to ~20 items so the LLM only does one
  small batched forward pass per request.
- **Pointwise yes/no scoring** instead of generation: no decoding loop, one
  forward pass, deterministic, easy to batch.
- **fp16 + autocast on CUDA**, fp32 on CPU. `torch.inference_mode()` everywhere.
- **Tokenizer pads to longest-in-batch** so we don't waste GPU on padding.
- **Warmup at boot** so the first user request doesn't pay cold-start cost.
- **Per-stage timings returned in every response** so you can see exactly
  where time goes (embed / candidate / tokenize / gpu / total).

## Install & run

```bash
cd llm_rec_svc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# defaults: tiny-gpt2 LLM, MiniLM embedder, items.json catalog
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First boot downloads the HF models (~100 MB). Subsequent boots are instant.

### Try it

```bash
curl -s localhost:8000/healthz | jq

curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{
  "profile": "Senior backend engineer prepping for AI infra system design interviews; cares about ranking, search, distributed systems.",
  "top_k": 5,
  "n_candidates": 20
}' | jq
```

You get back:

```json
{
  "items": [{"id":"i008","title":"System Design Interview Vol 2", ...}, ...],
  "timings": {
    "embed_ms":    5.2,
    "candidate_ms":0.4,
    "tokenize_ms": 1.1,
    "gpu_ms":      8.3,
    "total_ms":   15.7
  },
  "device": "cuda",
  "n_candidates": 20
}
```

### Benchmark

```bash
python scripts/bench.py -n 100 --topk 5 --cands 20
```

## Swapping the model

Set env vars before launch:

```bash
LLM=Qwen/Qwen2.5-0.5B-Instruct \
EMBEDDER=sentence-transformers/all-MiniLM-L6-v2 \
N_CANDIDATES=30 TOPK=10 LLM_WEIGHT=0.7 \
uvicorn app.main:app --port 8000
```

Any HF causal LM works. Bigger model = better ranking, higher GPU time.
For real low-latency production, swap the HF loop for vLLM/TGI and keep
this scoring contract.

## Files

- `app/catalog.py` — item store + cosine candidate gen
- `app/scorer.py`  — batched LLM yes/no scorer
- `app/main.py`    — FastAPI service, warmup, metrics
- `scripts/bench.py` — latency percentile harness
- `tests/test_smoke.py` — offline sanity tests
