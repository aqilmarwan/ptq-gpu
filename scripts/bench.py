#!/usr/bin/env python3
"""Latency benchmark for /generate: p50/p95/p99 per variant.

The project's point is the latency tradeoff across precisions, so this fires many
generations at each variant and reports percentiles of the end-to-end (wall) time
and the server's own denoise/total breakdown -- warm, after a discarded cold run.

Usage:
    python3 scripts/bench.py <base-url> [--variants fp16-base,int8-base] \
        [-n 30] [-c 1] [--steps 20] [--width 1024] [--height 1024]

Examples:
    python3 scripts/bench.py https://api.example.com
    python3 scripts/bench.py http://localhost:8000 --variants fp16-base,int8-base -n 50 -c 4

Stdlib only. Concurrency (-c) drives load so p95/p99 reflect queueing, not just
a quiet single-stream best case.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def _generate(base: str, variant: str, steps: int, w: int, h: int, seed: int) -> dict:
    """One generation. Returns wall-clock ms + the server's reported metrics."""
    body = json.dumps({
        "prompt": "a red fox in snow, sharp focus, natural light",
        "variantId": variant, "steps": steps, "seed": seed, "width": w, "height": h,
    }).encode()
    req = urllib.request.Request(base + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    metrics = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            evt = json.loads(data)
            if evt.get("type") == "done":
                metrics = evt["result"]["metrics"]
            elif evt.get("type") == "error":
                raise RuntimeError(evt["message"])
    return {"wall_ms": (time.perf_counter() - t0) * 1000.0, "metrics": metrics}


def _pct(xs: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy-style)."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (q / 100.0) * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _bench_variant(base: str, variant: str, n: int, conc: int, steps: int, w: int, h: int) -> dict:
    # One warmup (loads the engine into VRAM) -- reported separately as cold.
    cold = _generate(base, variant, steps, w, h, seed=0)

    walls: list[float] = []
    denoise: list[float] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(_generate, base, variant, steps, w, h, i + 1) for i in range(n)]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
                errors += 1
                print(f"  {variant}: request failed: {exc}", file=sys.stderr)
                continue
            walls.append(r["wall_ms"])
            if r["metrics"]:
                denoise.append(r["metrics"]["latencyMs"]["denoise"])

    return {
        "variant": variant, "n": len(walls), "errors": errors,
        "cold_ms": cold["wall_ms"],
        "wall": {q: _pct(walls, q) for q in (50, 95, 99)},
        "denoise": {q: _pct(denoise, q) for q in (50, 95, 99)},
        "vram": (cold["metrics"] or {}).get("vramPeakGB", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--variants", default="fp16-base", help="comma-separated variant ids")
    ap.add_argument("-n", type=int, default=30, help="requests per variant (after warmup)")
    ap.add_argument("-c", "--concurrency", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"benchmarking {base}  n={args.n}  concurrency={args.concurrency}  steps={args.steps}\n")

    rows = [_bench_variant(base, v, args.n, args.concurrency, args.steps, args.width, args.height)
            for v in variants]

    hdr = f"{'variant':<12} {'n':>4} {'cold':>7} {'wall p50':>9} {'p95':>7} {'p99':>7} {'denoise p50':>12} {'p95':>7} {'vram':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<12} {r['n']:>4} {r['cold_ms']:>7.0f} "
              f"{r['wall'][50]:>9.0f} {r['wall'][95]:>7.0f} {r['wall'][99]:>7.0f} "
              f"{r['denoise'][50]:>12.0f} {r['denoise'][95]:>7.0f} {r['vram']:>6.1f}")
    print("\n(all latencies in ms; cold = first request incl. engine load)")


if __name__ == "__main__":
    main()
