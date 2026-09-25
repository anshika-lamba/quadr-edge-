#!/usr/bin/env python3
"""
src/test_runtime.py
QADR Phase 1 — Edge SoC Runtime Verification
Target: ARM64 Android (Termux) + ONNX Runtime Mobile
"""

from __future__ import annotations
import argparse
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024.0 * 1024.0)
    return usage.ru_maxrss / 1024.0

def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]

def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    s = sorted(latencies_ms)
    return {
        "n": float(len(s)),
        "mean": float(statistics.fmean(s)),
        "std": float(statistics.stdev(s)) if len(s) > 1 else 0.0,
        "p50": float(percentile(s, 50)),
        "p95": float(percentile(s, 95)),
        "p99": float(percentile(s, 99)),
        "min": float(s[0]),
        "max": float(s[-1]),
    }

def make_session(
    model_path: Path,
    intra_op_threads: int = 2,
    graph_opt: str = "basic",
    prefer_nnapi: bool = False,
) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if graph_opt == "all":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    elif graph_opt == "extended":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

    available = ort.get_available_providers()
    providers: List[str] = []
    if prefer_nnapi and "NnapiExecutionProvider" in available:
        providers.append("NnapiExecutionProvider")
    providers.append("CPUExecutionProvider")

    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)

def profile_model(
    model_path: Path,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup: int = 20,
    iterations: int = 100,
    intra_op_threads: int = 2,
    graph_opt: str = "basic",
    prefer_nnapi: bool = False,
) -> Dict[str, object]:
    rss_before = rss_mb()
    t_load0 = time.perf_counter()
    session = make_session(
        model_path,
        intra_op_threads=intra_op_threads,
        graph_opt=graph_opt,
        prefer_nnapi=prefer_nnapi,
    )
    t_load1 = time.perf_counter()
    rss_after_load = rss_mb()

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal(input_shape, dtype=np.float32)
    feeds = {input_name: dummy}

    # Cold start
    t0 = time.perf_counter()
    _ = session.run(output_names, feeds)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warmup
    for _ in range(warmup):
        _ = session.run(output_names, feeds)

    # Steady-state loop
    latencies: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(output_names, feeds)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    rss_peak = rss_mb()
    stats = summarize_latencies(latencies)

    return {
        "model": str(model_path),
        "providers_available": ort.get_available_providers(),
        "providers_used": session.get_providers(),
        "session_load_ms": (t_load1 - t_load0) * 1000.0,
        "cold_start_ms": cold_ms,
        "intra_op_num_threads": intra_op_threads,
        "graph_opt": graph_opt,
        "latency_ms": stats,
        "rss_mb": {
            "before_session": rss_before,
            "after_session_load": rss_after_load,
            "after_steady_state": rss_peak,
        },
        "memory_pass_200mb": rss_peak < 200.0,
        "latency_pass_15ms_p95": stats["p95"] < 15.0,
    }

def print_report(result: Dict[str, object]) -> None:
    lat = result["latency_ms"]
    rss = result["rss_mb"]
    print("=" * 60)
    print(f"Model            : {result['model']}")
    print(f"Providers used   : {result['providers_used']}")
    print(f"Graph opt        : {result['graph_opt']} | threads={result['intra_op_num_threads']}")
    print("-" * 60)
    print(f"Session load     : {result['session_load_ms']:.2f} ms")
    print(f"Cold-start       : {result['cold_start_ms']:.2f} ms")
    print(
        f"Steady-state     : mean={lat['mean']:.3f}  std={lat['std']:.3f}  "
        f"P50={lat['p50']:.3f}  P95={lat['p95']:.3f}  P99={lat['p99']:.3f}  "
        f"min={lat['min']:.3f}  max={lat['max']:.3f}  (N={int(lat['n'])})"
    )
    print(
        f"RSS (MiB)        : before={rss['before_session']:.2f}  "
        f"after_load={rss['after_session_load']:.2f}  "
        f"peak={rss['after_steady_state']:.2f}"
    )
    mem_status = "PASS" if result["memory_pass_200mb"] else "FAIL"
    lat_status = "PASS" if result["latency_pass_15ms_p95"] else "FAIL"
    print(f"[STATUS] Memory < 200 MB        : {mem_status}")
    print(f"[STATUS] P95 latency < 15 ms    : {lat_status}")
    print("=" * 60)

def profile_numpy_proxy(iterations: int = 200) -> None:
    print("=" * 60)
    print("NumPy QADR-proxy microbench (Pre-op Normalization Only)")
    print("=" * 60)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(20):
        _ = np.clip(x * 1.05 + 0.02, 0.0, 1.0)

    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = np.clip(x * 1.05 + 0.02, 0.0, 1.0)
        samples.append((time.perf_counter() - t0) * 1000.0)

    stats = summarize_latencies(samples)
    print(
        f"QADR-proxy latency ms: mean={stats['mean']:.4f}  "
        f"P95={stats['p95']:.4f}  max={stats['max']:.4f}"
    )
    print(f"Peak RSS MiB         : {rss_mb():.2f}")
    print("=" * 60)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", type=Path, default=Path("models/mobilenetv4_small_fp32.onnx"))
    parser.add_argument("--int8", type=Path, default=Path("models/mobilenetv4_small_int8.onnx"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--graph-opt", choices=["basic", "extended", "all"], default="basic")
    parser.add_argument("--nnapi", action="store_true")
    parser.add_argument("--numpy-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("QADR Phase 1 — Edge SoC Runtime Verification")
    print(f"Platform : {platform.platform()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"ORT      : {ort.__version__}")
    print("=" * 60)

    if args.numpy_only:
        profile_numpy_proxy()
        return 0

    models = [m for m in [args.fp32, args.int8] if m.exists()]
    if not models:
        print("[ERROR] No models found under models/. Pass --numpy-only or acquire ONNX models.")
        return 1

    failures = 0
    for model in models:
        try:
            result = profile_model(
                model,
                warmup=args.warmup,
                iterations=args.iterations,
                intra_op_threads=args.threads,
                graph_opt=args.graph_opt,
                prefer_nnapi=args.nnapi,
            )
            print_report(result)
            if not result["memory_pass_200mb"] or not result["latency_pass_15ms_p95"]:
                failures += 1
        except Exception as exc:
            print(f"[ERROR] Failed profiling {model}: {exc}")
            failures += 1

    profile_numpy_proxy()
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
