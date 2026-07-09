#!/usr/bin/env python3
"""Profile async MoE isolated cost and homogeneous contention derates.

Measurements use consecutive windows of distinct experts. This avoids the old
single-expert hot-cache profile and better matches MoE forwards where each
expert weight block is streamed when that expert is visited.
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

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async,
    prepare_fused_moe_bf16_tiled_weights,
)


DEFAULT_ISOLATED_ROUTES = "1,2,4,8,16,32,64,128,256,512,1024,2048"
DEFAULT_CONTENTION_ROUTES = "1,4,16,64,256,1024,2048"
DEFAULT_THREADS = "1,2,4,8,16,32"
DEFAULT_NUM_PROFILE_EXPERTS = 64
DEFAULT_MEASUREMENT_EXPERTS = 8
DEFAULT_SHAPES = (
    "32;"
    "16x2;"
    "16,8,8;"
    "16,8,4,4;"
    "16,4,4,4,4;"
    "8x4;"
    "8,8,8,4,4;"
    "8,8,4,4,4,4;"
    "8,4,4,4,4,4,4;"
    "4x8;"
    "2x16;"
    "1x32"
)


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def parse_shapes(text: str) -> list[list[int]]:
    shapes: list[list[int]] = []
    for raw_shape in text.split(";"):
        raw_shape = raw_shape.strip()
        if not raw_shape:
            continue
        shape: list[int] = []
        for raw_item in raw_shape.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if "x" in item:
                left, right = item.split("x", 1)
                value = int(left)
                count = int(right)
                shape.extend([value] * count)
            else:
                shape.append(int(item))
        if not shape:
            raise ValueError(f"contention shape must not be empty: {raw_shape!r}")
        if any(v <= 0 for v in shape):
            raise ValueError(f"invalid contention shape: {raw_shape!r}")
        shapes.append(shape)
    if not shapes:
        raise ValueError("at least one contention shape is required")
    return shapes


def percentile_ns(values: list[int], p: float) -> int:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    w = rank - lo
    return int(round(ordered[lo] * (1.0 - w) + ordered[hi] * w))


def summarize_times(values: list[int]) -> dict:
    return {
        "median_ns": int(statistics.median(values)),
        "p10_ns": percentile_ns(values, 0.10),
        "p90_ns": percentile_ns(values, 0.90),
        "min_ns": min(values),
        "mean_ns": int(statistics.mean(values)),
        "num_iters": len(values),
    }


def clamp_measurement_experts(num_experts: int, requested: int) -> int:
    if requested <= 0:
        raise ValueError("--measurement-experts must be positive")
    if num_experts < 2:
        raise ValueError(
            "--num-experts must be at least 2; single-expert hot-cache "
            "profiling is intentionally unsupported"
        )
    return min(num_experts, requested)


def bf16(shape: tuple[int, ...], generator: torch.Generator, std: float) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, std, generator=generator)


def measure(run, *, warmup: int, runs: int) -> list[int]:
    for _ in range(warmup):
        out = run()
        _ = float(out.flatten()[0])
    gc.collect()

    times: list[int] = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        out = run()
        _ = float(out.flatten()[0])
        times.append(time.perf_counter_ns() - t0)
    return times


def make_async_run(
    *,
    packed,
    hidden_size: int,
    routes: int,
    shape: list[int],
    measurement_experts: int,
    num_profile_experts: int,
    generator: torch.Generator,
    std: float,
):
    num_lanes = len(shape)
    num_groups = max(1, math.ceil(measurement_experts / num_lanes))
    num_tasks = num_groups * num_lanes
    total_tokens = num_tasks * routes
    x = bf16((total_tokens, hidden_size), generator, std)
    topk_ids = torch.empty((total_tokens, 1), dtype=torch.int32)
    task_expert_values: list[int] = []
    for task_id in range(num_tasks):
        expert_id = task_id % num_profile_experts
        task_expert_values.append(expert_id)
        begin = task_id * routes
        topk_ids[begin : begin + routes, 0] = expert_id
    topk_weights = torch.ones((total_tokens, 1), dtype=torch.float32)

    begins: list[int] = []
    core = 0
    for threads in shape:
        begins.append(core)
        core += threads

    task_expert_ids = torch.tensor(task_expert_values, dtype=torch.int32)
    task_core_begins = torch.tensor(
        [begins[task_id % num_lanes] for task_id in range(num_tasks)],
        dtype=torch.int32,
    )
    task_threads = torch.tensor(
        [shape[task_id % num_lanes] for task_id in range(num_tasks)],
        dtype=torch.int32,
    )
    dep_offsets = [0]
    deps: list[int] = []
    for task_id in range(num_tasks):
        lane = task_id % num_lanes
        prev = task_id - num_lanes
        if prev >= 0:
            deps.append(prev)
        dep_offsets.append(len(deps))
    task_dep_offsets = torch.tensor(dep_offsets, dtype=torch.int32)
    task_deps = torch.tensor(deps, dtype=torch.int32)
    thread_cpu_ids = torch.arange(core, dtype=torch.int32)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            x,
            packed,
            topk_weights,
            topk_ids,
            task_expert_ids,
            task_core_begins,
            task_threads,
            task_dep_offsets,
            task_deps,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=core,
            activation="silu",
            global_num_experts=num_profile_experts,
            skip_weighted=True,
        )

    return run, num_groups, num_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_PROFILE_EXPERTS)
    parser.add_argument(
        "--measurement-experts",
        type=int,
        default=DEFAULT_MEASUREMENT_EXPERTS,
        help=(
            "Number of consecutive distinct experts represented in each timed "
            "call. Contention shapes round this up to complete shape groups."
        ),
    )
    parser.add_argument(
        "--route-buckets",
        default=None,
        help=(
            "Legacy alias: when set, uses the same route buckets for isolated "
            "and contention measurements."
        ),
    )
    parser.add_argument("--isolated-route-buckets", default=DEFAULT_ISOLATED_ROUTES)
    parser.add_argument("--contention-route-buckets", default=DEFAULT_CONTENTION_ROUTES)
    parser.add_argument("--thread-buckets", default=DEFAULT_THREADS)
    parser.add_argument("--contention-shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.route_buckets:
        isolated_routes = parse_int_list(args.route_buckets)
        contention_routes = list(isolated_routes)
    else:
        isolated_routes = parse_int_list(args.isolated_route_buckets)
        contention_routes = parse_int_list(args.contention_route_buckets)
    thread_buckets = parse_int_list(args.thread_buckets)
    shapes = parse_shapes(args.contention_shapes)
    measurement_experts = clamp_measurement_experts(
        args.num_experts, args.measurement_experts
    )
    missing_contention_routes = sorted(set(contention_routes) - set(isolated_routes))
    if missing_contention_routes:
        raise ValueError(
            "contention routes must also be present in isolated routes: "
            f"{missing_contention_routes}"
        )

    shape_threads = sorted({t for shape in shapes for t in shape})
    missing = [t for t in shape_threads if t not in thread_buckets]
    if missing:
        raise ValueError(f"contention shapes use thread counts missing from table: {missing}")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    w13 = bf16(
        (args.num_experts, 2 * args.ffn_hidden_size, args.hidden_size),
        generator,
        args.std,
    )
    w2 = bf16(
        (args.num_experts, args.hidden_size, args.ffn_hidden_size),
        generator,
        args.std,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    isolated: list[dict] = []
    iso_lookup: dict[tuple[int, int], int] = {}
    for routes in isolated_routes:
        for threads in thread_buckets:
            run, num_groups, num_tasks = make_async_run(
                packed=packed,
                hidden_size=args.hidden_size,
                routes=routes,
                shape=[threads],
                measurement_experts=measurement_experts,
                num_profile_experts=args.num_experts,
                generator=generator,
                std=args.std,
            )
            full_times = measure(run, warmup=args.warmup, runs=args.runs)
            times = [max(1, int(round(value / num_groups))) for value in full_times]
            summary = summarize_times(times)
            entry = {"routes": routes, "threads": threads, **summary}
            entry["measurement_experts"] = measurement_experts
            entry["measurement_tasks"] = num_tasks
            entry["full_call_median_ns"] = int(statistics.median(full_times))
            isolated.append(entry)
            iso_lookup[(routes, threads)] = int(summary["median_ns"])
            print(
                f"isolated routes={routes:<5} threads={threads:<3} "
                f"per_expert_median={summary['median_ns'] / 1e6:8.3f} ms "
                f"full_median={entry['full_call_median_ns'] / 1e6:8.3f} ms"
            )

    entries: list[dict] = []
    for shape in shapes:
        for routes in contention_routes:
            run, num_groups, num_tasks = make_async_run(
                packed=packed,
                hidden_size=args.hidden_size,
                routes=routes,
                shape=shape,
                measurement_experts=measurement_experts,
                num_profile_experts=args.num_experts,
                generator=generator,
                std=args.std,
            )
            full_times = measure(run, warmup=args.warmup, runs=args.runs)
            group_times = [
                max(1, int(round(value / num_groups))) for value in full_times
            ]
            summary = summarize_times(group_times)
            iso_max = max(iso_lookup[(routes, threads)] for threads in shape)
            derate = float(summary["median_ns"]) / float(iso_max)
            entry = {
                "shape": shape,
                "distinct_experts": len(shape),
                "routes": routes,
                "makespan_ns": int(summary["median_ns"]),
                "full_call_median_ns": int(statistics.median(full_times)),
                "measurement_experts": measurement_experts,
                "measurement_groups": num_groups,
                "measurement_tasks": num_tasks,
                "iso_max_ns": int(iso_max),
                "derate": derate,
                "p10_ns": int(summary["p10_ns"]),
                "p90_ns": int(summary["p90_ns"]),
                "num_iters": int(summary["num_iters"]),
            }
            entries.append(entry)
            print(
                f"contention shape={shape} routes={routes:<5} "
                f"per_group_median={summary['median_ns'] / 1e6:8.3f} ms "
                f"full_median={entry['full_call_median_ns'] / 1e6:8.3f} ms "
                f"derate={derate:6.3f}"
            )

    payload = {
        "schema_version": 1,
        "kind": "contention_derate",
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
            "kernel": "fused_moe_bf16_tiled_async",
            "activation": "silu",
            "num_experts": args.num_experts,
            "measurement_experts": measurement_experts,
            "expert_selection": "consecutive_window",
            "weight_reuse": "streaming_distinct_experts",
            "top_k": 1,
            "skip_weighted": True,
        },
        "measurement": {
            "path": "fused_moe_bf16_tiled_async",
            "pinning": "interval (thread_cpu_ids, disjoint per task)",
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND", ""),
            "runs": args.runs,
            "warmup": args.warmup,
            "derate": "per_group_makespan / max_i streaming_T_iso(R, threads_i)",
        },
        "routes": contention_routes,
        "isolated_routes": isolated_routes,
        "contention_routes": contention_routes,
        "thread_buckets": thread_buckets,
        "contention_shapes": shapes,
        "isolated": isolated,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
