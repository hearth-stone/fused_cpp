"""Rotate x86 BF16 fused-MoE policy variants on identical inputs and weights."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from fused_cpp import _moe_C
from fused_cpp.moe import PreparedBF16TiledFusedMoEWeights
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


_BACKENDS = {
    "auto": "auto",
    "avx512": "x86_avx512_bf16",
    "amx": "x86_amx_bf16",
}


@dataclass
class _Variant:
    label: str
    weights: PreparedBF16TiledFusedMoEWeights
    output: torch.Tensor
    samples_ms: list[float]


def _parse_variants(raw: str) -> tuple[str, ...]:
    variants: list[str] = []
    for value in raw.split(","):
        if value not in _BACKENDS:
            raise argparse.ArgumentTypeError(
                f"unknown policy variant {value!r}; expected one of {tuple(_BACKENDS)}",
            )
        if value not in variants:
            variants.append(value)
    if not variants:
        raise argparse.ArgumentTypeError("at least one policy variant is required")
    return tuple(variants)


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


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(samples: list[float], flops: float) -> dict[str, float]:
    median_ms = statistics.median(samples)
    return {
        "median_ms": median_ms,
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "best_ms": min(samples),
        "median_gflops": flops / median_ms / 1.0e6,
    }


def _timed_call(function: Callable[[], torch.Tensor]) -> float:
    start = time.perf_counter_ns()
    function()
    return (time.perf_counter_ns() - start) / 1.0e6


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate automatic, AVX-512, and AMX fused-MoE paths for x86 policy calibration.",
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--routing", choices=("balanced", "hot", "skewed"), default="hot")
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--variants", type=_parse_variants, default=("auto", "avx512", "amx"))
    parser.add_argument("--skip-weighted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=31)
    args = parser.parse_args()

    if args.skip_weighted and args.top_k != 1:
        parser.error("--skip-weighted requires --top-k 1")

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
    topk_ids = _routing(args.tokens, args.experts, args.top_k, args.routing)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k)), dim=-1)
    route_histogram = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts).tolist()
    policy = _moe_C.fused_moe_test_x86_policy(
        args.hidden,
        args.intermediate,
        route_histogram,
        args.threads,
    )

    variants: list[_Variant] = []
    prepack_ms: dict[str, float] = {}
    for label in args.variants:
        start = time.perf_counter_ns()
        packed = prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend=_BACKENDS[label],
        )
        prepack_ms[label] = (time.perf_counter_ns() - start) / 1.0e6
        variants.append(
            _Variant(
                label=label,
                weights=packed,
                output=torch.empty_like(inputs),
                samples_ms=[],
            ),
        )

    def run(variant: _Variant) -> torch.Tensor:
        return fused_moe_bf16_tiled(
            inputs,
            variant.weights,
            topk_weights,
            topk_ids,
            num_threads=args.threads,
            skip_weighted=args.skip_weighted,
            out=variant.output,
        )

    with torch.inference_mode():
        torch.set_num_threads(args.threads)
        reference = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)
        torch.set_num_threads(1)
        max_abs_vs_torch: dict[str, float] = {}
        for variant in variants:
            actual = run(variant)
            torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.15)
            max_abs_vs_torch[variant.label] = float((actual.float() - reference.float()).abs().max())
            for _ in range(args.warmup):
                run(variant)

        for iteration in range(args.runs):
            offset = iteration % len(variants)
            for index in range(len(variants)):
                variant = variants[(offset + index) % len(variants)]
                variant.samples_ms.append(_timed_call(lambda current=variant: run(current)))

    total_routes = args.tokens * args.top_k
    flops = float(total_routes * 6 * args.hidden * args.intermediate)
    results = {
        variant.label: {
            **_summary(variant.samples_ms, flops),
            "backend_name": variant.weights.backend_name,
            "backend_id": variant.weights.gemm_backend,
            "max_abs_vs_torch": max_abs_vs_torch[variant.label],
        }
        for variant in variants
    }
    best_median_ms = min(result["median_ms"] for result in results.values())
    for result in results.values():
        result["regret_vs_best_pct"] = (result["median_ms"] / best_median_ms - 1.0) * 100.0

    print(
        json.dumps(
            {
                "shape": {
                    "tokens": args.tokens,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts": args.experts,
                    "top_k": args.top_k,
                    "routes": total_routes,
                    "routes_per_expert": route_histogram,
                    "routing": args.routing,
                },
                "threads": args.threads,
                "automatic_policy": policy,
                "skip_weighted": args.skip_weighted,
                "warmup_per_variant": args.warmup,
                "runs_per_variant": args.runs,
                "prepack_ms": prepack_ms,
                "variants": results,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
