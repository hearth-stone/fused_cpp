# SPDX-License-Identifier: Apache-2.0
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


_BACKENDS = ("x86_amx_bf16", "x86_amx_bf16_n64")
_PATTERNS = ("m1n2", "m2n2", "m1n4")


def _parse_csv(raw: str, allowed: tuple[str, ...], name: str) -> tuple[str, ...]:
    values: list[str] = []
    for value in raw.split(","):
        if value not in allowed:
            raise argparse.ArgumentTypeError(f"unknown {name} {value!r}; expected one of {allowed}")
        if value not in values:
            values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"at least one {name} is required")
    return tuple(values)


def _parse_routes(raw: str) -> tuple[int, ...]:
    routes: list[int] = []
    for field in raw.split(","):
        try:
            value = int(field)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"routes must be comma-separated integers, got {raw!r}") from error
        if value <= 0:
            raise argparse.ArgumentTypeError("route counts must be positive")
        if value not in routes:
            routes.append(value)
    if not routes:
        raise argparse.ArgumentTypeError("at least one route count is required")
    return tuple(routes)


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
        "best_ms": min(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median_gflops": flops / median_ms / 1e6,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the AMX N32 and N64/K32-streaming packed-B layouts in one process.",
    )
    parser.add_argument("--routes", type=_parse_routes, default=(16, 32, 64, 128, 512, 2048))
    parser.add_argument("--patterns", type=lambda raw: _parse_csv(raw, _PATTERNS, "pattern"), default=_PATTERNS)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=21)
    args = parser.parse_args()
    if args.runs < 10:
        parser.error("--runs must be at least 10")

    os.environ["FUSED_CPP_MOE_AMX_SILU_EPILOGUE"] = "resident"
    os.environ["FUSED_CPP_MOE_AMX_W2_EPILOGUE"] = "baseline"
    os.environ["FUSED_CPP_MOE_AMX_TILE_STATE"] = "per_call"
    os.environ.pop("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS", None)
    os.environ.pop("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS", None)

    generator = torch.Generator(device="cpu").manual_seed(20260726)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    max_routes = max(args.routes)
    inputs = torch.empty((max_routes, args.hidden), dtype=torch.bfloat16).normal_(std=0.01, generator=generator)
    w13 = torch.empty((1, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16).normal_(
        std=0.01,
        generator=generator,
    )
    w2 = torch.empty((1, args.hidden, args.intermediate), dtype=torch.bfloat16).normal_(
        std=0.01,
        generator=generator,
    )

    packed = {}
    prepack_ms = {}
    for backend in _BACKENDS:
        start = time.perf_counter_ns()
        packed[backend] = prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend=backend,
        )
        prepack_ms[backend] = (time.perf_counter_ns() - start) / 1e6
        if packed[backend].backend_name != backend:
            raise RuntimeError(f"requested {backend}, got {packed[backend].backend_name}")

    results = []
    for routes in args.routes:
        case_inputs = inputs[:routes]
        topk_ids = torch.zeros((routes, 1), dtype=torch.int32)
        topk_weights = torch.ones((routes, 1), dtype=torch.float32)
        torch.set_num_threads(args.threads)
        reference = fused_moe_naive(case_inputs, w13, w2, topk_weights, topk_ids)
        torch.set_num_threads(1)
        flops = float(routes * 6 * args.hidden * args.intermediate)

        for pattern in args.patterns:
            os.environ["FUSED_CPP_MOE_AMX_PATTERN"] = pattern

            def run(backend: str) -> torch.Tensor:
                return fused_moe_bf16_tiled(
                    case_inputs,
                    packed[backend],
                    topk_weights,
                    topk_ids,
                    num_threads=args.threads,
                )

            outputs = {backend: run(backend) for backend in _BACKENDS}
            for backend in _BACKENDS:
                max_abs = float((outputs[backend].float() - reference.float()).abs().max())
                if max_abs > 0.07:
                    raise RuntimeError(
                        f"{backend}/{pattern}/M{routes} failed correctness: max_abs_vs_torch={max_abs}",
                    )
                for _ in range(args.warmup):
                    run(backend)

            samples = {backend: [] for backend in _BACKENDS}
            for iteration in range(args.runs):
                order = _BACKENDS if iteration % 2 == 0 else tuple(reversed(_BACKENDS))
                for backend in order:
                    start = time.perf_counter_ns()
                    run(backend)
                    samples[backend].append((time.perf_counter_ns() - start) / 1e6)

            summaries = {backend: _summary(samples[backend], flops) for backend in _BACKENDS}
            n32_ms = summaries["x86_amx_bf16"]["median_ms"]
            n64_ms = summaries["x86_amx_bf16_n64"]["median_ms"]
            results.append(
                {
                    "routes": routes,
                    "pattern": pattern,
                    "n64_speedup_vs_n32": n32_ms / n64_ms,
                    "layouts": {
                        backend: {
                            **summaries[backend],
                            "max_abs_vs_torch": float(
                                (outputs[backend].float() - reference.float()).abs().max(),
                            ),
                            "max_abs_vs_n32": float(
                                (outputs[backend].float() - outputs["x86_amx_bf16"].float()).abs().max(),
                            ),
                        }
                        for backend in _BACKENDS
                    },
                },
            )

    print(
        json.dumps(
            {
                "shape": {
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts": 1,
                    "top_k": 1,
                },
                "threads": args.threads,
                "warmup_per_layout": args.warmup,
                "runs_per_layout": args.runs,
                "prepack_ms": prepack_ms,
                "results": results,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
