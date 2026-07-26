# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


_ENVIRONMENT = "FUSED_CPP_MOE_X86_PERSISTENT_INPUT"
_MODES = ("transient", "persistent")


def _set_mode(mode: str) -> None:
    os.environ[_ENVIRONMENT] = "1" if mode == "persistent" else "0"


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate transient and persistent x86 MoE gathered-input scratch.")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("x86_avx512_bf16", "x86_amx_bf16"), default="x86_amx_bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=31)
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.experts, args.threads, args.runs) <= 0:
        parser.error("shape dimensions, experts, threads, and runs must be positive")
    if args.experts < 2:
        parser.error("--experts must be at least 2 so the aligned single-expert input bypass stays disabled")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    torch.manual_seed(20260726)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16).normal_(std=0.01)
    w13 = torch.empty(
        (args.experts, 2 * args.intermediate, args.hidden),
        dtype=torch.bfloat16,
    ).normal_(std=0.01)
    w2 = torch.empty(
        (args.experts, args.hidden, args.intermediate),
        dtype=torch.bfloat16,
    ).normal_(std=0.01)
    topk_ids = (torch.arange(args.tokens, dtype=torch.int32) % args.experts).view(-1, 1)
    topk_weights = torch.ones((args.tokens, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend=args.backend)
    output = torch.empty_like(inputs)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            skip_weighted=True,
            out=output,
        )

    previous = os.environ.get(_ENVIRONMENT)
    previous_intermediate = os.environ.get("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE")
    os.environ["FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE"] = "1"
    try:
        results: dict[str, torch.Tensor] = {}
        for mode in _MODES:
            _set_mode(mode)
            results[mode] = run().clone()
        reference = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids, skip_weighted=True)
        torch.testing.assert_close(results["persistent"].float(), results["transient"].float(), atol=0, rtol=0)
        max_abs = float((results["persistent"].float() - reference.float()).abs().max())

        for iteration in range(args.warmup):
            order = _MODES if iteration % 2 == 0 else tuple(reversed(_MODES))
            for mode in order:
                _set_mode(mode)
                run()

        samples: dict[str, list[float]] = {mode: [] for mode in _MODES}
        for iteration in range(args.runs):
            order = _MODES if iteration % 2 == 0 else tuple(reversed(_MODES))
            for mode in order:
                _set_mode(mode)
                start = time.perf_counter_ns()
                run()
                samples[mode].append((time.perf_counter_ns() - start) / 1e6)
    finally:
        if previous is None:
            os.environ.pop(_ENVIRONMENT, None)
        else:
            os.environ[_ENVIRONMENT] = previous
        if previous_intermediate is None:
            os.environ.pop("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE", None)
        else:
            os.environ["FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE"] = previous_intermediate

    flops = float(args.tokens * 6 * args.hidden * args.intermediate)
    metrics = {
        mode: {
            "median_ms": statistics.median(values),
            "p90_ms": _percentile(values, 0.90),
            "best_ms": min(values),
            "median_gflops": flops / statistics.median(values) / 1e6,
        }
        for mode, values in samples.items()
    }
    print(
        json.dumps(
            {
                "backend": packed.backend_name,
                "shape": {
                    "tokens": args.tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts": args.experts,
                    "top_k": 1,
                    "routing": "balanced",
                },
                "threads": args.threads,
                "warmup": args.warmup,
                "runs": args.runs,
                "max_abs_vs_torch": max_abs,
                "results": metrics,
                "persistent_speedup": metrics["transient"]["median_ms"] / metrics["persistent"]["median_ms"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
