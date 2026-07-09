#!/usr/bin/env python3
"""Profile T_expert(routes, threads) for the scheduled BF16 tiled MoE kernel.

Each measurement runs a consecutive window of distinct experts and reports the
per-expert average. This models the normal MoE case where expert weights are
streamed from memory/LLC instead of repeatedly reusing one hot expert.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import List, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fused_cpp.moe import (  # noqa: E402
    _HAS_BF16_TILED_FUSED_MOE,
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


DEFAULT_ROUTE_BUCKETS = "1,2,4,8,16,32,64,128,256,512,1024,2048"
DEFAULT_THREAD_BUCKETS = "1,2,3,4,5,6,7,8"
DEFAULT_NUM_PROFILE_EXPERTS = 64
DEFAULT_MEASUREMENT_EXPERTS = 8


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"bucket values must be positive: {value}")
        values.append(value)
    if not values:
        raise ValueError("bucket list must not be empty")
    return values


def clamp_measurement_experts(num_experts: int, requested: int) -> int:
    if requested <= 0:
        raise ValueError("--measurement-experts must be positive")
    if num_experts < 2:
        raise ValueError(
            "--num-experts must be at least 2; single-expert hot-cache "
            "profiling is intentionally unsupported"
        )
    return min(num_experts, requested)


def percentile_ns(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return int(round(ordered[lo] * (1.0 - weight) + ordered[hi] * weight))


def bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def measure_call(run, *, warmup: int, runs: int) -> List[int]:
    for _ in range(warmup):
        out = run()
        _ = float(out.flatten()[0])
    gc.collect()

    times_ns: List[int] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        out = run()
        _ = float(out.flatten()[0])
        times_ns.append(time.perf_counter_ns() - t0)
    return times_ns


def entry_from_times(routes: int, threads: int, times_ns: Sequence[int]) -> dict:
    return {
        "routes": routes,
        "threads": threads,
        "median_ns": int(statistics.median(times_ns)),
        "p10_ns": percentile_ns(times_ns, 0.10),
        "p90_ns": percentile_ns(times_ns, 0.90),
        "p99_ns": percentile_ns(times_ns, 0.99),
        "min_ns": min(times_ns),
        "mean_ns": int(statistics.mean(times_ns)),
        "stddev_ns": int(statistics.pstdev(times_ns)) if len(times_ns) > 1 else 0,
        "num_iters": len(times_ns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a cost_model/profile_schema.md JSON table by measuring "
            "streaming distinct-expert scheduled BF16 MoE calls."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_PROFILE_EXPERTS)
    parser.add_argument(
        "--measurement-experts",
        type=int,
        default=DEFAULT_MEASUREMENT_EXPERTS,
        help=(
            "Number of consecutive distinct experts executed per timed call. "
            "The recorded T_expert is full_call_time / measurement_experts."
        ),
    )
    parser.add_argument("--route-buckets", default=DEFAULT_ROUTE_BUCKETS)
    parser.add_argument("--thread-buckets", default=DEFAULT_THREAD_BUCKETS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--activation", choices=["silu", "swigluoai"], default="silu")
    parser.add_argument(
        "--with-bias",
        action="store_true",
        help="Include w13/w2 bias in the profiled expert path.",
    )
    parser.add_argument(
        "--weighted-merge",
        action="store_true",
        help=(
            "Keep the weighted top-k merge enabled. By default top_k=1 uses "
            "skip_weighted=True so the table focuses on expert execution."
        ),
    )
    parser.add_argument(
        "--fuse-silu",
        action="store_true",
        help="Prepare weights with the fused SiLU epilogue layout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError("BF16 tiled fused MoE backend is unavailable")
    if args.hidden_size <= 0 or args.ffn_hidden_size <= 0:
        raise ValueError("hidden and FFN sizes must be positive")
    measurement_experts = clamp_measurement_experts(
        args.num_experts, args.measurement_experts
    )
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    route_buckets = parse_int_list(args.route_buckets)
    thread_buckets = parse_int_list(args.thread_buckets)

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)

    w13_weight = bf16_normal(
        (args.num_experts, 2 * args.ffn_hidden_size, args.hidden_size),
        generator=generator,
        std=args.std,
    )
    w2_weight = bf16_normal(
        (args.num_experts, args.hidden_size, args.ffn_hidden_size),
        generator=generator,
        std=args.std,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13_weight, w2_weight, fuse_silu=args.fuse_silu
    )

    w13_bias = None
    w2_bias = None
    if args.with_bias:
        w13_bias = torch.randn(
            args.num_experts,
            2 * args.ffn_hidden_size,
            generator=generator,
            dtype=torch.float32,
        ) * args.std
        w2_bias = torch.randn(
            args.num_experts,
            args.hidden_size,
            generator=generator,
            dtype=torch.float32,
        ) * args.std

    entries = []
    for routes in route_buckets:
        hidden_states = bf16_normal(
            (routes * measurement_experts, args.hidden_size),
            generator=generator,
            std=args.std,
        )
        topk_ids = torch.empty((routes * measurement_experts, 1), dtype=torch.int32)
        for local_expert in range(measurement_experts):
            begin = local_expert * routes
            end = begin + routes
            topk_ids[begin:end, 0] = local_expert
        topk_weights = torch.ones(
            (routes * measurement_experts, 1), dtype=torch.float32
        )

        for threads in thread_buckets:
            wave_offsets = torch.arange(
                measurement_experts + 1, dtype=torch.int32
            )
            team_expert_ids = torch.arange(measurement_experts, dtype=torch.int32)
            team_threads = torch.full(
                (measurement_experts,), threads, dtype=torch.int32
            )
            thread_cpu_ids = torch.arange(threads, dtype=torch.int32)

            def run() -> torch.Tensor:
                return fused_moe_bf16_tiled_scheduled(
                    hidden_states,
                    packed,
                    topk_weights,
                    topk_ids,
                    wave_offsets,
                    team_expert_ids,
                    team_threads,
                    thread_cpu_ids=thread_cpu_ids,
                    w13_bias=w13_bias,
                    w2_bias=w2_bias,
                    num_threads=threads,
                    activation=args.activation,
                    global_num_experts=args.num_experts,
                    skip_weighted=not args.weighted_merge,
                )

            full_times_ns = measure_call(run, warmup=args.warmup, runs=args.runs)
            times_ns = [
                max(1, int(round(value / measurement_experts)))
                for value in full_times_ns
            ]
            entry = entry_from_times(routes, threads, times_ns)
            entry["measurement_experts"] = measurement_experts
            entry["full_call_median_ns"] = int(statistics.median(full_times_ns))
            entries.append(entry)
            print(
                "profiled "
                f"routes={routes:<5} threads={threads:<3} "
                f"per_expert_median={entry['median_ns']} ns "
                f"full_median={entry['full_call_median_ns']} ns "
                f"p90={entry['p90_ns']} ns"
            )

    payload = {
        "schema_version": 1,
        "target": {
            "machine": platform.machine(),
            "cpu": platform.processor() or platform.machine(),
            "num_cores": os.cpu_count(),
            "os": platform.platform(),
        },
        "expert_shape": {
            "dtype": "bf16",
            "hidden_size": args.hidden_size,
            "intermediate_size": args.ffn_hidden_size,
            "kernel": "fused_cpp_moe_bf16_tiled_scheduled_auto_mn_split",
            "activation": args.activation,
            "num_experts": args.num_experts,
            "measurement_experts": measurement_experts,
            "expert_selection": "consecutive_window",
            "weight_reuse": "streaming_distinct_experts",
            "top_k": 1,
            "skip_weighted": not args.weighted_merge,
            "with_bias": args.with_bias,
            "fuse_silu": args.fuse_silu,
        },
        "route_buckets": route_buckets,
        "thread_buckets": thread_buckets,
        "metric": "median_ns",
        "entries": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
