"""CLI: pretty-print a saved trace.

Usage:
    python -m agentdag.analyze traces/balanced_parens.jsonl
    python -m agentdag.analyze traces/run.jsonl --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tracer import TraceEvent, Tracer


def load_trace(path: Path) -> Tracer:
    """Rehydrate a Tracer from a JSONL file. We bypass __init__'s truncation
    by constructing manually."""
    t = Tracer.__new__(Tracer)
    import threading

    t._events = []
    t._lock = threading.Lock()
    t._path = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t._events.append(TraceEvent(**d))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze an agentdag JSONL trace.")
    ap.add_argument("path", type=Path, help="path to .jsonl trace file")
    ap.add_argument("--json", action="store_true", help="emit JSON summary")
    args = ap.parse_args()

    t = load_trace(args.path)
    if args.json:
        print(json.dumps(t.summarize(), indent=2))
    else:
        print(t.render_table())
        s = t.summarize()
        nodes = s["nodes"]
        if nodes:
            slow = max(nodes.items(), key=lambda kv: kv[1]["run_ms"])
            retried = [n for n, v in nodes.items() if v["retries"] > 0]
            failed = [n for n, v in nodes.items() if v["status"] == "failed"]
            print()
            print(f"slowest node: {slow[0]}  ({slow[1]['run_ms']:.1f} ms)")
            if retried:
                print(f"retried: {', '.join(retried)}")
            if failed:
                print(f"FAILED: {', '.join(failed)}")


if __name__ == "__main__":
    main()
