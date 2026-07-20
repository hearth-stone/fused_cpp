# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


_W13_ENVIRONMENT = "FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS"
_W2_ENVIRONMENT = "FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS"


@dataclass(frozen=True)
class CacheConfig:
    w13_blocks: int | None
    w2_blocks: int | None

    @property
    def label(self) -> str:
        w13 = "auto" if self.w13_blocks is None else str(self.w13_blocks)
        w2 = "auto" if self.w2_blocks is None else str(self.w2_blocks)
        return f"w13={w13},w2={w2}"


def _parse_block_count(field: str, item: str) -> int | None:
    if field == "auto":
        return None
    try:
        value = int(field)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"cache config must contain 'auto' or integers, got {item!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError(f"cache block counts must be non-negative, got {item!r}")
    return value


def _parse_configs(raw: str) -> tuple[CacheConfig, ...]:
    configs: list[CacheConfig] = []
    for item in raw.split(","):
        fields = item.split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(f"cache config must be W13:W2, got {item!r}")
        w13_blocks, w2_blocks = (_parse_block_count(field, item) for field in fields)
        config = CacheConfig(w13_blocks, w2_blocks)
        if config not in configs:
            configs.append(config)
    if not configs:
        raise argparse.ArgumentTypeError("at least one cache config is required")
    if CacheConfig(0, 0) not in configs:
        raise argparse.ArgumentTypeError("cache configs must include the 0:0 baseline")
    return tuple(configs)


def _routing(tokens: int, experts: int, top_k: int, mode: str) -> torch.Tensor:
    routes = tokens * top_k
    if mode == "hot":
        flat = [0] * routes
    elif mode == "skewed":
        hot_routes = (routes * 3) // 4
        flat = [0] * hot_routes
        if experts == 1:
            flat.extend([0] * (routes - hot_routes))
        else:
            flat.extend(1 + index % (experts - 1) for index in range(routes - hot_routes))
    else:
        flat = [index % experts for index in range(routes)]
    return torch.tensor(flat, dtype=torch.int32).view(tokens, top_k)


def _set_cache_config(config: CacheConfig) -> None:
    if config.w13_blocks is None:
        os.environ.pop(_W13_ENVIRONMENT, None)
    else:
        os.environ[_W13_ENVIRONMENT] = str(config.w13_blocks)
    if config.w2_blocks is None:
        os.environ.pop(_W2_ENVIRONMENT, None)
    else:
        os.environ[_W2_ENVIRONMENT] = str(config.w2_blocks)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(samples: list[float], flops: float, baseline_ms: float) -> dict[str, float]:
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "best_ms": min(samples),
        "median_gflops": flops / median_ms / 1e6,
        "speedup_vs_0_0": baseline_ms / median_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan x86 fused-expert W13/W2 cache-block windows.")
    parser.add_argument("--backend", choices=("x86_avx512_bf16", "x86_amx_bf16"), required=True)
    parser.add_argument("--amx-pattern", choices=("auto", "m1n2", "m2n2", "m1n4"), default="auto")
    parser.add_argument("--configs", type=_parse_configs, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--routing", choices=("balanced", "hot", "skewed"), default="hot")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--skip-weighted", action="store_true")
    args = parser.parse_args()

    if args.runs < 10:
        parser.error("--runs must be at least 10")
    if args.skip_weighted and args.top_k != 1:
        parser.error("--skip-weighted requires --top-k 1")

    torch.manual_seed(20260719)
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
    if args.skip_weighted:
        topk_weights = torch.ones((args.tokens, args.top_k), dtype=torch.float32)
    else:
        topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k)), dim=-1)
    route_histogram = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts).tolist()

    if args.backend == "x86_amx_bf16":
        os.environ["FUSED_CPP_MOE_AMX_PATTERN"] = args.amx_pattern

    pack_start = time.perf_counter_ns()
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend=args.backend)
    pack_ms = (time.perf_counter_ns() - pack_start) / 1e6

    def custom() -> torch.Tensor:
        return fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            skip_weighted=args.skip_weighted,
        )

    torch.set_num_threads(args.threads)
    reference = fused_moe_naive(
        inputs,
        w13,
        w2,
        topk_weights,
        topk_ids,
        skip_weighted=args.skip_weighted,
    )
    torch.set_num_threads(1)

    max_abs: dict[str, float] = {}
    samples = {config.label: [] for config in args.configs}
    for config in args.configs:
        _set_cache_config(config)
        output = custom()
        max_abs[config.label] = float((output.float() - reference.float()).abs().max())
        for _ in range(args.warmup):
            custom()

    for iteration in range(args.runs):
        offset = iteration % len(args.configs)
        order = args.configs[offset:] + args.configs[:offset]
        for config in order:
            _set_cache_config(config)
            start = time.perf_counter_ns()
            custom()
            samples[config.label].append((time.perf_counter_ns() - start) / 1e6)

    routes = args.tokens * args.top_k
    flops = float(routes * 6 * args.hidden * args.intermediate)
    baseline_ms = statistics.median(samples[CacheConfig(0, 0).label])
    results = {}
    for config in args.configs:
        result = _summary(samples[config.label], flops, baseline_ms)
        result["max_abs_vs_torch"] = max_abs[config.label]
        results[config.label] = result

    k_granularity = 32 if args.backend == "x86_amx_bf16" else 2
    w13_k_pad = ((args.hidden + k_granularity - 1) // k_granularity) * k_granularity
    w2_k_pad = ((args.intermediate + k_granularity - 1) // k_granularity) * k_granularity
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
                    "routes_per_expert": route_histogram,
                    "routing": args.routing,
                },
                "backend": packed.backend_name,
                "amx_pattern": args.amx_pattern if args.backend == "x86_amx_bf16" else None,
                "threads": args.threads,
                "skip_weighted": args.skip_weighted,
                "warmup_per_config": args.warmup,
                "runs_per_config": args.runs,
                "prepack_ms": pack_ms,
                "w13_bytes_per_block": w13_k_pad * 32 * 2,
                "w2_bytes_per_block": w2_k_pad * 32 * 2,
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
