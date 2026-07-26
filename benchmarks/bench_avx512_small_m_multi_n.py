"""Rotate AVX-512 small-M exact-M and multi-N JIT kernels on identical tensors."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


_ENVIRONMENT = "FUSED_CPP_MOE_AVX512_SMALL_M_MULTI_N"
_VALID_VARIANTS = ("baseline", "w13", "w2", "multi_n", "auto")


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value) for value in raw.split(",")))
    if not values or any(value < 1 or value > 4 for value in values):
        raise argparse.ArgumentTypeError("routes must be a comma-separated subset of 1,2,3,4")
    return values


def _parse_variants(raw: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(raw.split(",")))
    if not values or any(value not in _VALID_VARIANTS for value in values):
        raise argparse.ArgumentTypeError(f"variants must be selected from {_VALID_VARIANTS}")
    if "baseline" not in values:
        raise argparse.ArgumentTypeError("variants must include baseline")
    return values


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


@dataclass
class _Variant:
    name: str
    output: torch.Tensor
    samples_ms: list[float]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=_parse_csv_ints, default=(1, 2, 3, 4))
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--variants", type=_parse_variants, default=_VALID_VARIANTS)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--runs", type=int, default=31)
    args = parser.parse_args()

    torch.manual_seed(20260726)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    max_routes = max(args.routes)
    inputs = torch.empty((max_routes, args.hidden), dtype=torch.bfloat16).normal_(std=0.01)
    w13 = torch.empty((1, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16).normal_(std=0.01)
    w2 = torch.empty((1, args.hidden, args.intermediate), dtype=torch.bfloat16).normal_(std=0.01)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_avx512_bf16",
    )

    results: dict[str, object] = {}
    with torch.inference_mode():
        for routes in args.routes:
            case_inputs = inputs[:routes]
            topk_ids = torch.zeros((routes, 1), dtype=torch.int32)
            topk_weights = torch.ones((routes, 1), dtype=torch.float32)
            variants = [
                _Variant(name=name, output=torch.empty_like(case_inputs), samples_ms=[])
                for name in args.variants
            ]

            torch.set_num_threads(args.threads)
            reference = fused_moe_naive(case_inputs, w13, w2, topk_weights, topk_ids, skip_weighted=True)
            torch.set_num_threads(1)

            def run(variant: _Variant) -> torch.Tensor:
                return fused_moe_bf16_tiled(
                    case_inputs,
                    packed,
                    topk_weights,
                    topk_ids,
                    num_threads=args.threads,
                    skip_weighted=True,
                    out=variant.output,
                )

            max_abs: dict[str, float] = {}
            for variant in variants:
                os.environ[_ENVIRONMENT] = variant.name
                actual = run(variant)
                torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.15)
                max_abs[variant.name] = float((actual.float() - reference.float()).abs().max())
                for _ in range(args.warmup):
                    run(variant)

            for iteration in range(args.runs):
                offset = iteration % len(variants)
                for index in range(len(variants)):
                    variant = variants[(offset + index) % len(variants)]
                    os.environ[_ENVIRONMENT] = variant.name
                    variant.samples_ms.append(_timed_call(lambda current=variant: run(current)))

            flops = float(routes * 6 * args.hidden * args.intermediate)
            case_result = {
                variant.name: {
                    **_summary(variant.samples_ms, flops),
                    "max_abs_vs_torch": max_abs[variant.name],
                }
                for variant in variants
            }
            baseline_ms = case_result["baseline"]["median_ms"]
            for value in case_result.values():
                value["speedup_vs_baseline"] = baseline_ms / value["median_ms"]
            results[str(routes)] = case_result

    os.environ.pop(_ENVIRONMENT, None)
    print(
        json.dumps(
            {
                "shape": {
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "routes": args.routes,
                    "experts": 1,
                    "top_k": 1,
                },
                "threads": args.threads,
                "warmup_per_variant": args.warmup,
                "runs_per_variant": args.runs,
                "variants": args.variants,
                "results_by_routes": results,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
