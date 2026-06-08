"""Latency sweep: vary K (candidates rerank by LLM) and N (candidates from retrieval).

Prints a table of p50/p95/avg per stage for each config so you can see how
each stage scales.
"""

from __future__ import annotations

import time

import httpx

URL = "http://127.0.0.1:8000/recommend"

PROFILES = [
    "Senior backend engineer prepping for AI infra system design interviews; cares about ranking, search, distributed systems.",
    "ML engineer wanting hands-on with LLMs and production deployment.",
    "Engineering manager interested in leadership and org design.",
    "Junior dev curious about databases and SQL fundamentals.",
    "SRE focused on reliability of large-scale streaming systems.",
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100))]


def bench(n_calls: int, n_candidates: int, top_k: int) -> dict:
    lats = {"total": [], "embed": [], "candidate": [], "tokenize": [], "gpu": []}
    with httpx.Client(timeout=60.0) as c:
        # warm
        c.post(URL, json={"profile": PROFILES[0], "top_k": top_k, "n_candidates": n_candidates})
        for i in range(n_calls):
            r = c.post(
                URL,
                json={
                    "profile": PROFILES[i % len(PROFILES)],
                    "top_k": top_k,
                    "n_candidates": n_candidates,
                },
            )
            r.raise_for_status()
            t = r.json()["timings"]
            lats["total"].append(t["total_ms"])
            lats["embed"].append(t["embed_ms"])
            lats["candidate"].append(t["candidate_ms"])
            lats["tokenize"].append(t["tokenize_ms"])
            lats["gpu"].append(t["gpu_ms"])
    return lats


def main():
    configs = [
        # (n_candidates, top_k)
        (5, 5),
        (10, 5),
        (20, 5),
        (25, 5),  # full catalog
        (25, 10),
    ]
    n_calls = 30

    print(
        f"{'cands':>5} {'topk':>4}  ||"
        f"  {'total p50':>9} {'p95':>6} ||"
        f"  {'gpu p50':>7} {'p95':>6} ||"
        f"  {'embed p50':>9} ||"
        f"  {'tok p50':>7} ||"
        f"  {'cand p50':>8}"
    )
    print("-" * 100)
    for n_c, k in configs:
        L = bench(n_calls, n_c, k)
        print(
            f"{n_c:>5} {k:>4}  ||"
            f"  {pct(L['total'],50):>9.1f} {pct(L['total'],95):>6.1f} ||"
            f"  {pct(L['gpu'],50):>7.1f} {pct(L['gpu'],95):>6.1f} ||"
            f"  {pct(L['embed'],50):>9.1f} ||"
            f"  {pct(L['tokenize'],50):>7.1f} ||"
            f"  {pct(L['candidate'],50):>8.2f}"
        )

    # throughput at the default config
    print()
    n_thru = 100
    t0 = time.perf_counter()
    bench(n_thru, 20, 5)
    elapsed = time.perf_counter() - t0
    print(
        f"throughput @ cands=20 topk=5, sequential: "
        f"{n_thru/elapsed:.1f} req/s ({n_thru} calls in {elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
