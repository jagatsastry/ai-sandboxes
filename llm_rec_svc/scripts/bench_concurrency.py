"""Concurrency-vs-latency benchmark.

Drives the /recommend endpoint at increasing concurrency levels with a thread
pool and measures:
    - per-request total latency (p50, p95, p99)
    - aggregate throughput (req/s)
    - server-side per-stage latency (gpu, embed, tokenize, candidate)

This reveals where the service moves from latency-bound (low concurrency, all
serial) into a regime where requests start queueing on a single model worker.
For sync uvicorn + one model on CPU, throughput will be roughly flat and p50
will rise linearly -- that's the signature you're looking for.

Usage:
    python scripts/bench_concurrency.py
    python scripts/bench_concurrency.py --url http://127.0.0.1:8000/recommend \\
                                        --cands 10 --topk 5 \\
                                        --levels 1,2,4,8 --reqs 20

Output: a table per concurrency level + a final summary.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

PROFILES = [
    "Senior backend engineer prepping for AI infra system design interviews; cares about ranking, search, distributed systems.",
    "ML engineer wanting hands-on with LLMs and production deployment.",
    "Engineering manager interested in leadership and org design.",
    "Junior dev curious about databases and SQL fundamentals.",
    "SRE focused on reliability of large-scale streaming systems.",
]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def one_call(client: httpx.Client, url: str, profile: str, cands: int, topk: int) -> dict:
    t0 = time.perf_counter()
    r = client.post(url, json={"profile": profile, "top_k": topk, "n_candidates": cands})
    r.raise_for_status()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    j = r.json()
    j["timings"]["wall_ms"] = wall_ms  # client-measured (incl. network + framework)
    return j["timings"]


def run_level(url: str, concurrency: int, reqs: int, cands: int, topk: int) -> dict:
    """Fire `reqs` requests with `concurrency` workers in flight at once."""
    lats = {"wall": [], "total": [], "gpu": [], "embed": [], "tokenize": [], "candidate": []}

    # one client per worker thread (httpx.Client isn't fully thread-safe across
    # concurrent requests on the same underlying connection pool config; safer
    # to give each worker its own).
    def worker(i: int) -> dict:
        with httpx.Client(timeout=300.0) as c:
            return one_call(c, url, PROFILES[i % len(PROFILES)], cands, topk)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker, i) for i in range(reqs)]
        for f in as_completed(futs):
            t = f.result()
            lats["wall"].append(t["wall_ms"])
            lats["total"].append(t["total_ms"])
            lats["gpu"].append(t["gpu_ms"])
            lats["embed"].append(t["embed_ms"])
            lats["tokenize"].append(t["tokenize_ms"])
            lats["candidate"].append(t["candidate_ms"])
    elapsed = time.perf_counter() - t0

    return {
        "concurrency": concurrency,
        "requests": reqs,
        "elapsed_s": elapsed,
        "throughput_rps": reqs / elapsed,
        "lats": lats,
    }


def fmt_table(rows: list[dict]) -> str:
    header = (
        f"{'conc':>4}  {'reqs':>4}  "
        f"{'thru rps':>9}  ||  "
        f"{'wall p50':>9} {'p95':>7} {'p99':>7}  ||  "
        f"{'srv p50':>8} {'p95':>7}  ||  "
        f"{'gpu p50':>8}  {'embed p50':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        L = r["lats"]
        lines.append(
            f"{r['concurrency']:>4}  {r['requests']:>4}  "
            f"{r['throughput_rps']:>9.2f}  ||  "
            f"{pct(L['wall'], 50):>9.1f} {pct(L['wall'], 95):>7.1f} {pct(L['wall'], 99):>7.1f}  ||  "
            f"{pct(L['total'], 50):>8.1f} {pct(L['total'], 95):>7.1f}  ||  "
            f"{pct(L['gpu'], 50):>8.1f}  {pct(L['embed'], 50):>10.1f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/recommend")
    ap.add_argument("--cands", type=int, default=10)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--reqs", type=int, default=16, help="requests per concurrency level (after warmup)"
    )
    ap.add_argument("--levels", default="1,2,4,8", help="comma-separated concurrency levels")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    # quick warmup so first-request cold cost doesn't bias level 1
    print(f"warming up @ {args.url}")
    with httpx.Client(timeout=300.0) as c:
        one_call(c, args.url, PROFILES[0], args.cands, args.topk)

    rows = []
    for level in levels:
        print(
            f"\n>> concurrency={level}  reqs={args.reqs}  " f"cands={args.cands}  topk={args.topk}"
        )
        r = run_level(args.url, level, args.reqs, args.cands, args.topk)
        rows.append(r)
        L = r["lats"]
        print(
            f"   thru={r['throughput_rps']:.2f} rps   "
            f"wall p50={pct(L['wall'],50):.0f}ms p95={pct(L['wall'],95):.0f}ms   "
            f"srv p50={pct(L['total'],50):.0f}ms"
        )

    print()
    print("=" * 80)
    print("SUMMARY (wall = client-measured incl. queueing; srv = server-internal)")
    print(fmt_table(rows))

    # quick interpretation
    if len(rows) >= 2:
        base = rows[0]
        last = rows[-1]
        ratio_lat = pct(last["lats"]["wall"], 50) / pct(base["lats"]["wall"], 50)
        ratio_thru = last["throughput_rps"] / base["throughput_rps"]
        ratio_conc = last["concurrency"] / base["concurrency"]
        print()
        print(
            f"From c={base['concurrency']} to c={last['concurrency']} "
            f"({ratio_conc:.0f}x concurrency):"
        )
        print(f"  p50 wall latency multiplied by {ratio_lat:.2f}x")
        print(f"  throughput multiplied by      {ratio_thru:.2f}x")
        if ratio_thru < 1.3 and ratio_lat > 1.5:
            print(
                "  -> classic single-worker serialization. Concurrency does "
                "NOT add throughput; it only adds queue latency. "
                "Wins require: more workers, async batching, or GPU."
            )


if __name__ == "__main__":
    main()
