#!/usr/bin/env python3
"""Benchmark explicit W8A8 DeepSeek V4 attention projections."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch

from fused_cpp import bf16_linear
from fused_cpp.deepseek_v4_w8a8 import (
    _HAS_DEEPSEEK_V4_W8A8,
    deepseek_v4_w8a8_linear,
    deepseek_v4_w8a8_linear_pair,
    deepseek_v4_wo_b_w8a8,
    prepare_deepseek_v4_w8a8_linear_weight,
)


def _measure(function: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]], warmup: int, runs: int) -> float:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).item()
    return {
        "relative_l2": torch.linalg.vector_norm(delta).item() / max(denominator, 1e-30),
        "max_abs": delta.abs().max().item(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--q-k", type=int, default=1024)
    parser.add_argument("--q-n", type=int, default=8192)
    parser.add_argument("--wo-k", type=int, default=2048)
    parser.add_argument("--wo-n", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not _HAS_DEEPSEEK_V4_W8A8:
        raise RuntimeError("DeepSeek V4 W8A8 requires a Linux AArch64 SVE+i8mm build")
    if args.m <= 0 or min(args.q_k, args.q_n, args.wo_k, args.wo_n, args.threads, args.runs) <= 0:
        raise ValueError("all shapes, threads, and runs must be positive")

    torch.manual_seed(20260821)
    torch.set_num_threads(args.threads)
    q_input = torch.empty((args.m, args.q_k), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    first_q_weight = torch.empty((args.q_n, args.q_k), dtype=torch.bfloat16).uniform_(-0.02, 0.02)
    second_q_weight = torch.empty_like(first_q_weight).uniform_(-0.02, 0.02)
    wo_input = torch.empty((args.m, args.wo_k), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    wo_weight = torch.empty((args.wo_n, args.wo_k), dtype=torch.bfloat16).uniform_(-0.02, 0.02)

    first_q_bf16 = bf16_linear.prepare(first_q_weight)
    second_q_bf16 = bf16_linear.prepare(second_q_weight)
    wo_bf16 = bf16_linear.prepare(wo_weight)
    first_q_w8a8 = prepare_deepseek_v4_w8a8_linear_weight(first_q_weight)
    second_q_w8a8 = prepare_deepseek_v4_w8a8_linear_weight(second_q_weight)
    wo_w8a8 = prepare_deepseek_v4_w8a8_linear_weight(wo_weight)

    first_q_reference = bf16_linear.linear(q_input, first_q_bf16, nthreads=args.threads)
    second_q_reference = bf16_linear.linear(q_input, second_q_bf16, nthreads=args.threads)
    wo_reference = bf16_linear.linear(wo_input, wo_bf16, nthreads=args.threads)
    first_q_actual, second_q_actual = deepseek_v4_w8a8_linear_pair(
        q_input, first_q_w8a8, second_q_w8a8, num_threads=args.threads
    )
    wo_actual = deepseek_v4_wo_b_w8a8(wo_input, wo_w8a8, num_threads=args.threads)

    timings = {
        "first_q_bf16_ms": _measure(
            lambda: bf16_linear.linear(q_input, first_q_bf16, nthreads=args.threads), args.warmup, args.runs
        ),
        "first_q_w8a8_ms": _measure(
            lambda: deepseek_v4_w8a8_linear(q_input, first_q_w8a8, num_threads=args.threads),
            args.warmup,
            args.runs,
        ),
        "two_q_bf16_ms": _measure(
            lambda: (
                bf16_linear.linear(q_input, first_q_bf16, nthreads=args.threads),
                bf16_linear.linear(q_input, second_q_bf16, nthreads=args.threads),
            ),
            args.warmup,
            args.runs,
        ),
        "two_q_w8a8_pair_ms": _measure(
            lambda: deepseek_v4_w8a8_linear_pair(
                q_input, first_q_w8a8, second_q_w8a8, num_threads=args.threads
            ),
            args.warmup,
            args.runs,
        ),
        "wo_bf16_ms": _measure(
            lambda: bf16_linear.linear(wo_input, wo_bf16, nthreads=args.threads), args.warmup, args.runs
        ),
        "wo_w8a8_ms": _measure(
            lambda: deepseek_v4_wo_b_w8a8(wo_input, wo_w8a8, num_threads=args.threads),
            args.warmup,
            args.runs,
        ),
    }
    print(
        json.dumps(
            {
                "shape": vars(args),
                "timings": timings,
                "ratios": {
                    "first_q_bf16_over_w8a8": timings["first_q_bf16_ms"] / timings["first_q_w8a8_ms"],
                    "two_q_bf16_over_w8a8_pair": timings["two_q_bf16_ms"] / timings["two_q_w8a8_pair_ms"],
                    "wo_bf16_over_w8a8": timings["wo_bf16_ms"] / timings["wo_w8a8_ms"],
                },
                "error": {
                    "first_q": _error(first_q_actual, first_q_reference),
                    "second_q": _error(second_q_actual, second_q_reference),
                    "wo": _error(wo_actual, wo_reference),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
