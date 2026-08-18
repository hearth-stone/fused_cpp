#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark the standalone shared MLP against the existing E=1 MoE path."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_shared_mlp_bf16_tiled_weights
from fused_cpp.moe import shared_mlp_bf16_tiled


def _warmup(fn, warmup: int) -> None:
    for _ in range(warmup):
        fn()


def _measure_pair(baseline, candidate, runs: int) -> tuple[list[float], list[float]]:
    samples = {"baseline": [], "candidate": []}
    for _ in range(runs):
        order = (
            ("baseline", "candidate")
            if len(samples["baseline"]) % 2 == 0
            else ("candidate", "baseline")
        )
        for name in order:
            begin = time.perf_counter_ns()
            (baseline if name == "baseline" else candidate)()
            samples[name].append((time.perf_counter_ns() - begin) / 1e6)
    return samples["baseline"], samples["candidate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(20260816)
    hidden = torch.empty((args.rows, args.hidden), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    w13 = torch.empty((2 * args.intermediate, args.hidden), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    w2 = torch.empty((args.hidden, args.intermediate), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    weights = prepare_shared_mlp_bf16_tiled_weights(w13, w2)
    topk_weights = torch.ones((args.rows, 1), dtype=torch.float32)
    topk_ids = torch.zeros((args.rows, 1), dtype=torch.int32)

    def baseline() -> torch.Tensor:
        return fused_moe_bf16_tiled(
            hidden,
            weights,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            skip_weighted=True,
        )

    def candidate() -> torch.Tensor:
        return shared_mlp_bf16_tiled(hidden, weights, num_threads=args.threads)
    baseline_out = baseline()
    candidate_out = candidate()
    torch.testing.assert_close(candidate_out, baseline_out, atol=0, rtol=0)

    _warmup(baseline, args.warmup)
    _warmup(candidate, args.warmup)
    baseline_ms, candidate_ms = _measure_pair(baseline, candidate, args.runs)
    baseline_median = statistics.median(baseline_ms)
    candidate_median = statistics.median(candidate_ms)
    print(
        json.dumps(
            {
                "shape": {"M": args.rows, "H": args.hidden, "F": args.intermediate},
                "dtype": "bfloat16",
                "threads": args.threads,
                "warmup": args.warmup,
                "runs": args.runs,
                "backend": weights.backend_name,
                "n_tile": weights.backend_n_tile,
                "baseline": "fused_moe_bf16_tiled E=1 top_k=1 skip_weighted=True",
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "speedup": baseline_median / candidate_median,
                "baseline_samples_ms": baseline_ms,
                "candidate_samples_ms": candidate_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
