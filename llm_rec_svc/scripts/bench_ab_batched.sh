#!/usr/bin/env bash
# A/B benchmark: legacy (per-request) vs batched (cross-request) scoring.
#
# Boots the service twice with the same model + data, once with
# BATCH_ENABLE=0 and once with BATCH_ENABLE=1, runs the concurrency bench
# against each, and prints the results side-by-side at the end.
#
# WARNING: this script kills any process listening on PORT (default 8000)
# before each boot.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT=${PORT:-8000}
CANDS=${CANDS:-20}
TOPK=${TOPK:-5}
REQS=${REQS:-24}
LEVELS=${LEVELS:-1,2,4,8}
WAIT=${WAIT:-25}
BATCH_SIZE=${BATCH_SIZE:-160}

stop_server() {
  ps -ef | grep "[u]vicorn app.main:app --host 127.0.0.1 --port ${PORT}" \
    | awk '{print $2}' | xargs -r kill 2>/dev/null || true
  sleep 2
}

start_server() {
  local mode="$1"
  echo
  echo "================================================================"
  echo " >>> booting server in $mode mode"
  echo "================================================================"
  stop_server
  if [[ "$mode" == "legacy" ]]; then
    BATCH_ENABLE=0 nohup uvicorn app.main:app \
      --host 127.0.0.1 --port "${PORT}" --log-level warning \
      > /tmp/uvicorn_ab.log 2>&1 &
  else
    BATCH_ENABLE=1 BATCH_MAX_WAIT_MS="${WAIT}" BATCH_MAX_SIZE="${BATCH_SIZE}" \
      nohup uvicorn app.main:app \
      --host 127.0.0.1 --port "${PORT}" --log-level warning \
      > /tmp/uvicorn_ab.log 2>&1 &
  fi

  echo " ... waiting for /healthz"
  for i in $(seq 1 30); do
    sleep 2
    out=$(curl -s -m 2 "http://127.0.0.1:${PORT}/healthz" 2>/dev/null || true)
    if [[ -n "$out" ]]; then
      echo " ... ready: $out"
      return 0
    fi
  done
  echo " !! server did not come up; tail /tmp/uvicorn_ab.log:"
  tail -20 /tmp/uvicorn_ab.log
  return 1
}

run_bench() {
  local label="$1"
  echo
  echo "----------------------------------------------------------------"
  echo " >>> bench: $label"
  echo "----------------------------------------------------------------"
  python scripts/bench_concurrency.py \
    --url "http://127.0.0.1:${PORT}/recommend" \
    --cands "${CANDS}" --topk "${TOPK}" \
    --reqs "${REQS}" --levels "${LEVELS}"
  echo
  echo "Server metrics after $label:"
  curl -s "http://127.0.0.1:${PORT}/metrics" | python -m json.tool || true
}

trap stop_server EXIT

start_server legacy
run_bench "LEGACY (BATCH_ENABLE=0)"

start_server batched
run_bench "BATCHED (max_wait_ms=${WAIT} max_size=${BATCH_SIZE})"
