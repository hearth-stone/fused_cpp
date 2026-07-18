# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


def _measure(function: Callable[[], torch.Tensor], warmup: int, runs: int) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples), min(samples)


def _routing(tokens: int, experts: int, top_k: int, mode: str) -> torch.Tensor:
    if mode == "hot":
        return torch.zeros((tokens, top_k), dtype=torch.int32)
    return torch.tensor(
        [[(token * top_k + slot) % experts for slot in range(top_k)] for token in range(tokens)],
        dtype=torch.int32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AVX-512 BF16 fused expert against PyTorch/oneDNN.")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--routing", choices=("balanced", "hot"), default="balanced")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--baseline-runs", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(20260718)
    # The custom executor requests its own explicit parallel team. Keep the
    # default PyTorch pool at one thread while measuring it so an earlier
    # oneDNN worker cannot occupy the second pinned core during block time.
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
    topk_ids = _routing(args.tokens, args.experts, args.top_k, args.routing)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k)), dim=-1)

    pack_start = time.perf_counter_ns()
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    pack_ms = (time.perf_counter_ns() - pack_start) / 1e6

    def custom() -> torch.Tensor:
        return fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
        )

    def torch_onednn() -> torch.Tensor:
        return fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    custom_output = custom()
    torch.set_num_threads(args.threads)
    reference = torch_onednn()
    max_abs = float((custom_output.float() - reference.float()).abs().max())
    torch.set_num_threads(1)
    custom_median_ms, custom_best_ms = _measure(custom, args.warmup, args.runs)
    torch.set_num_threads(args.threads)
    baseline_median_ms, baseline_best_ms = _measure(torch_onednn, 2, args.baseline_runs)
    routes = args.tokens * args.top_k
    flops = float(routes * 6 * args.hidden * args.intermediate)

    print(
        json.dumps(
            {
                "shape": {
                    "tokens": args.tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts": args.experts,
                    "top_k": args.top_k,
                    "routes": routes,
                    "routing": args.routing,
                },
                "threads": args.threads,
                "torch_baseline_threads": args.threads,
                "backend": packed.backend_name,
                "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA"),
                "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
                "torch_mkldnn_available": torch.backends.mkldnn.is_available(),
                "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
                "prepack_ms": pack_ms,
                "max_abs_vs_torch": max_abs,
                "custom": {
                    "median_ms": custom_median_ms,
                    "best_ms": custom_best_ms,
                    "median_gflops": flops / custom_median_ms / 1e6,
                    "best_gflops": flops / custom_best_ms / 1e6,
                },
                "torch_onednn_staged": {
                    "median_ms": baseline_median_ms,
                    "best_ms": baseline_best_ms,
                    "median_gflops": flops / baseline_median_ms / 1e6,
                    "best_gflops": flops / baseline_best_ms / 1e6,
                },
                "speedup_vs_torch_median": baseline_median_ms / custom_median_ms,
                "warmup": args.warmup,
                "runs": args.runs,
                "baseline_runs": args.baseline_runs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
