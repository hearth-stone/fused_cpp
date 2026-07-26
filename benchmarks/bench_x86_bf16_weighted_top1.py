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


_ENVIRONMENT = "FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT"
_MODES = ("workspace", "direct")


def _set_mode(mode: str) -> None:
    os.environ[_ENVIRONMENT] = "1" if mode == "direct" else "0"


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _routing(tokens: int, experts: int, mode: str) -> torch.Tensor:
    if mode == "hot":
        return torch.zeros((tokens, 1), dtype=torch.int32)
    return (torch.arange(tokens, dtype=torch.int32) % experts).view(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate weighted top-1 route-workspace and direct-BF16 epilogues.")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--routing", choices=("hot", "balanced"), default="hot")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("x86_avx512_bf16", "x86_amx_bf16"), default="x86_amx_bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=31)
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.experts, args.threads, args.runs) <= 0:
        parser.error("shape dimensions, experts, threads, and runs must be positive")
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
    topk_ids = _routing(args.tokens, args.experts, args.routing)
    topk_weights = torch.sigmoid(torch.randn((args.tokens, 1), dtype=torch.float32))
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend=args.backend)
    output = torch.empty_like(inputs)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            out=output,
        )

    previous = os.environ.get(_ENVIRONMENT)
    try:
        results: dict[str, torch.Tensor] = {}
        for mode in _MODES:
            _set_mode(mode)
            results[mode] = run().clone()
        reference = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)
        torch.testing.assert_close(results["direct"].float(), results["workspace"].float(), atol=0, rtol=0)
        max_abs = float((results["direct"].float() - reference.float()).abs().max())

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
                    "routing": args.routing,
                },
                "threads": args.threads,
                "warmup": args.warmup,
                "runs": args.runs,
                "max_abs_vs_torch": max_abs,
                "eliminated_route_workspace_bytes": args.tokens * args.hidden * 4,
                "results": metrics,
                "direct_speedup": metrics["workspace"]["median_ms"] / metrics["direct"]["median_ms"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
