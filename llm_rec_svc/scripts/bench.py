"""Tiny benchmark: hit /recommend N times, print latency percentiles."""

from __future__ import annotations

import argparse
import statistics
import time

import httpx

PROFILES = [
    "Senior backend engineer prepping for AI infra system design interviews; cares about ranking, search, distributed systems.",
    "ML engineer wanting hands-on with LLMs and production deployment.",
    "Engineering manager interested in leadership and org design.",
    "Junior dev curious about databases and SQL fundamentals.",
    "SRE focused on reliability of large-scale streaming systems.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/recommend")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cands", type=int, default=20)
    args = ap.parse_args()

    lats, gpu, emb, cand = [], [], [], []
    with httpx.Client(timeout=60.0) as c:
        # warm
        c.post(
            args.url, json={"profile": PROFILES[0], "top_k": args.topk, "n_candidates": args.cands}
        )
        t0 = time.perf_counter()
        for i in range(args.n):
            r = c.post(
                args.url,
                json={
                    "profile": PROFILES[i % len(PROFILES)],
                    "top_k": args.topk,
                    "n_candidates": args.cands,
                },
            )
            r.raise_for_status()
            j = r.json()
            lats.append(j["timings"]["total_ms"])
            gpu.append(j["timings"]["gpu_ms"])
            emb.append(j["timings"]["embed_ms"])
            cand.append(j["timings"]["candidate_ms"])
        elapsed = time.perf_counter() - t0

    def pct(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p / 100))]

    print(f"n={args.n}  topk={args.topk}  cands={args.cands}")
    print(f"throughput: {args.n / elapsed:.1f} req/s")
    for name, xs in [("total", lats), ("gpu", gpu), ("embed", emb), ("cand", cand)]:
        print(
            f"  {name:5s}  p50={pct(xs,50):6.1f}ms  "
            f"p95={pct(xs,95):6.1f}ms  "
            f"avg={statistics.mean(xs):6.1f}ms"
        )


if __name__ == "__main__":
    main()
