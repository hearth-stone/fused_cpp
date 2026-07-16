#!/usr/bin/env python3
"""Benchmark configurable packed-B windows in the production async SVE expert."""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch

from fused_cpp.moe import fused_moe_bf16_tiled_async
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


BF16_BYTES = 2


@dataclass(frozen=True)
class WindowResult:
    requested_bytes: int
    w13_ranges: int
    w13_max_bytes: int
    w2_ranges: int
    w2_max_bytes: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    aggregate_tflops: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=24)
    parser.add_argument("--routes", type=int, default=2040)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads-per-expert", type=int, default=4)
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--window-mib", default="0,2,1")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def window_shape(k: int, n: int, n_tile: int, requested_bytes: int, legacy_ranges: int) -> tuple[int, int]:
    total_tiles = n // n_tile
    if requested_bytes > 0:
        bytes_per_tile = k * n_tile * BF16_BYTES
        max_tiles = max(1, requested_bytes // bytes_per_tile)
        ranges = (total_tiles + max_tiles - 1) // max_tiles
    else:
        ranges = legacy_ranges
    max_range_tiles = (total_tiles + ranges - 1) // ranges
    return ranges, max_range_tiles * k * n_tile * BF16_BYTES


def bf16_normal(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)


def main() -> None:
    args = parse_args()
    windows = [round(float(value) * (1 << 20)) for value in args.window_mib.split(",")]
    if min(args.experts, args.routes, args.hidden, args.intermediate, args.threads_per_expert, args.runs) <= 0:
        raise ValueError("shape, thread, expert, and run arguments must be positive")
    if min(windows) < 0:
        raise ValueError("--window-mib values must be non-negative")

    total_threads = args.experts * args.threads_per_expert
    os.environ.setdefault("FUSED_CPP_MOE_SVE", "1")
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ.setdefault("FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER", "1")
    os.environ.setdefault("FUSED_CPP_MOE_W2_BF16_ROUTE", "0")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)

    w13 = bf16_normal((args.experts, 2 * args.intermediate, args.hidden), generator)
    w2 = bf16_normal((args.experts, args.hidden, args.intermediate), generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE backend")

    tokens = args.experts * args.routes
    hidden = bf16_normal((tokens, args.hidden), generator)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32)
    topk_ids = torch.arange(args.experts, dtype=torch.int32).repeat_interleave(args.routes).reshape(tokens, 1)
    expert_ids = torch.arange(args.experts, dtype=torch.int32)
    core_begins = torch.arange(args.experts, dtype=torch.int32) * args.threads_per_expert
    task_threads = torch.full((args.experts,), args.threads_per_expert, dtype=torch.int32)
    dep_offsets = torch.zeros(args.experts + 1, dtype=torch.int32)
    deps = torch.empty(0, dtype=torch.int32)
    cpu_ids = torch.arange(args.cpu_start, args.cpu_start + total_threads, dtype=torch.int32)

    def invoke(window_bytes: int) -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            core_begins,
            task_threads,
            dep_offsets,
            deps,
            thread_cpu_ids=cpu_ids,
            num_threads=total_threads,
            skip_weighted=True,
            w13_split=True,
            weight_window_bytes=window_bytes,
        )

    reference = invoke(windows[0])
    for window_bytes in windows[1:]:
        torch.testing.assert_close(invoke(window_bytes).float(), reference.float(), atol=0, rtol=0)

    useful_flops = 6 * args.experts * args.routes * args.hidden * args.intermediate
    results: list[WindowResult] = []
    for window_bytes in windows:
        for _ in range(args.warmup):
            invoke(window_bytes)
        samples = []
        for _ in range(args.runs):
            begin = time.perf_counter_ns()
            invoke(window_bytes)
            samples.append((time.perf_counter_ns() - begin) / 1.0e6)
        median_ms = statistics.median(samples)
        w13_ranges, w13_max_bytes = window_shape(
            args.hidden,
            2 * args.intermediate,
            packed.backend_n_tile,
            window_bytes,
            legacy_ranges=2,
        )
        w2_ranges, w2_max_bytes = window_shape(
            args.intermediate,
            args.hidden,
            packed.backend_n_tile,
            window_bytes,
            legacy_ranges=1,
        )
        results.append(
            WindowResult(
                requested_bytes=window_bytes,
                w13_ranges=w13_ranges,
                w13_max_bytes=w13_max_bytes,
                w2_ranges=w2_ranges,
                w2_max_bytes=w2_max_bytes,
                median_ms=median_ms,
                p10_ms=quantile(samples, 0.10),
                p90_ms=quantile(samples, 0.90),
                aggregate_tflops=useful_flops / (median_ms * 1.0e9),
            )
        )

    print(
        f"experts={args.experts} routes={args.routes} H={args.hidden} F={args.intermediate} "
        f"threads_per_expert={args.threads_per_expert} total_threads={total_threads}"
    )
    print("window_MiB  W13_ranges/max_MiB  W2_ranges/max_MiB  p50_ms  p10_ms  p90_ms  TFLOP/s")
    for result in results:
        print(
            f"{result.requested_bytes / (1 << 20):10.3f}  "
            f"{result.w13_ranges:4d}/{result.w13_max_bytes / (1 << 20):7.3f}  "
            f"{result.w2_ranges:4d}/{result.w2_max_bytes / (1 << 20):7.3f}  "
            f"{result.median_ms:7.3f}  {result.p10_ms:7.3f}  {result.p90_ms:7.3f}  "
            f"{result.aggregate_tflops:7.3f}"
        )


if __name__ == "__main__":
    main()
