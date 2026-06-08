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
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every per-request latency with its server-side stage breakdown.",
    )
    args = ap.parse_args()

    lats, gpu, emb, cand = [], [], [], []
    with httpx.Client(timeout=60.0) as c:
        # warm
        c.post(
            args.url, json={"profile": PROFILES[0], "top_k": args.topk, "n_candidates": args.cands}
        )
        t0 = time.perf_counter()
        for i in range(args.n):
            t_req = time.perf_counter()
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
            client_ms = (time.perf_counter() - t_req) * 1000.0
            t = j["timings"]
            lats.append(t["total_ms"])
            gpu.append(t["gpu_ms"])
            emb.append(t["embed_ms"])
            cand.append(t["candidate_ms"])
            if args.verbose:
                print(
                    f"  [{i + 1:3d}/{args.n}] client={client_ms:6.1f}ms "
                    f"server={t['total_ms']:6.1f}ms "
                    f"(embed={t['embed_ms']:5.1f} cand={t['candidate_ms']:5.1f} "
                    f"tok={t['tokenize_ms']:5.1f} gpu={t['gpu_ms']:5.1f} "
                    f"queue={t.get('queue_ms', 0):5.1f} batch={t.get('batch_size', 0):2d})"
                )
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
