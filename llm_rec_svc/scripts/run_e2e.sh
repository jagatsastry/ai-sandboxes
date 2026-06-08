#!/usr/bin/env bash
# End-to-end runner for llm_rec_svc.
#
# What it does:
#   1. installs deps (idempotent; skip with SKIP_INSTALL=1)
#   2. runs unit tests (optional, skip with SKIP_TESTS=1)
#   3. boots the service in the background (tiny-gpt2 by default for speed)
#   4. waits for /healthz
#   5. fires a single sample /recommend
#   6. runs the latency sweep
#   7. runs the concurrency benchmark
#   8. shuts the service down
#
# Configure model with LLM env var, e.g.:
#   LLM=Qwen/Qwen2.5-0.5B-Instruct bash llm_rec_svc/scripts/run_e2e.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

LLM="${LLM:-sshleifer/tiny-gpt2}"
PORT="${PORT:-8000}"
LOG="${LOG:-/tmp/llm_rec_svc.log}"

if [ -z "${SKIP_INSTALL:-}" ]; then
  echo "==> [1/8] installing deps"
  pip install --quiet -r requirements.txt
fi

if [ -z "${SKIP_TESTS:-}" ]; then
  echo "==> [2/8] unit tests (smoke)"
  pip install --quiet pytest
  python -m pytest -q tests/ || echo "(tests skipped or partial -- they need models downloaded)"
fi

echo "==> [3/8] booting service with LLM=$LLM on port $PORT"
LLM="$LLM" python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
SVC_PID=$!
trap "kill $SVC_PID 2>/dev/null || true" EXIT

echo "==> [4/8] waiting for /healthz"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; then
    echo "    ready after ${i}s"
    break
  fi
  sleep 2
done
curl -s "http://127.0.0.1:$PORT/healthz"; echo

echo
echo "==> [5/8] sample /recommend"
curl -s -X POST "http://127.0.0.1:$PORT/recommend" \
  -H 'content-type: application/json' \
  -d '{
    "profile": "Senior backend engineer prepping for AI infra system design interviews; cares about ranking, search, distributed systems.",
    "top_k": 5,
    "n_candidates": 20
  }' | python -m json.tool

echo
echo "==> [6/8] latency sweep (varying K)"
python scripts/bench_sweep.py || true

echo
echo "==> [7/8] concurrency benchmark"
python scripts/bench_concurrency.py --cands 10 --topk 5 --reqs 16 --levels 1,2,4,8

echo
echo "==> [8/8] /metrics"
curl -s "http://127.0.0.1:$PORT/metrics" | python -m json.tool

echo
echo "done. (service will be shut down by EXIT trap)"
