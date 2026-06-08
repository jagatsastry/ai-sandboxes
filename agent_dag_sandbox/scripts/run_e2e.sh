#!/usr/bin/env bash
# End-to-end runner for the agent_dag_sandbox.
#
# What it does:
#   1. runs pytest (unit + e2e tests)
#   2. runs the balanced-parens example workflow
#   3. analyzes the resulting trace
#
# Run from anywhere:
#   bash agent_dag_sandbox/scripts/run_e2e.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "==> [1/3] running unit + e2e tests"
python -m pytest -q

echo
echo "==> [2/3] running sample workflow: balanced_parens"
python -m examples.balanced_parens

echo
echo "==> [3/3] analyzing the trace"
python -m agentdag.analyze traces/balanced_parens.jsonl

echo
echo "done."
