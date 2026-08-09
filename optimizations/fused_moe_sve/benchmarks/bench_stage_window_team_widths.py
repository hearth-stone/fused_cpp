#!/usr/bin/env python3
"""Sweep expert team widths with the production per-task stage windows.

The isolated sweep measures streaming distinct-expert latency at every requested
team width.  The homogeneous sweep fills one NUMA rank with equal-width teams,
so its aggregate TFLOP/s directly answers which width is best when enough
experts are available.  Both paths use strict Plan V2 tasks and apply the
current profile-bound W13/W2 window policy to every measured task.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from fused_cpp.moe.plan import upgrade_legacy_async_plan  # noqa: E402
from stage_window_policy import AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4  # noqa: E402
from weight_window import fused_moe_task_weight_windows  # noqa: E402


DEFAULT_ROUTES = (
    "1-12,13,24,28,36,40,48,49,56,72,84,95,96,108,120,132,143,"
    "144,160,176,192,200,215,216,224,240,256,287,288,320,384,480,"
    "512,575,576,768,1024,1536,2040"
)
DEFAULT_ISOLATED_WIDTHS = "1-32"
DEFAULT_HOMOGENEOUS_WIDTHS = "1,2,3,4,6,8,12,16,24,32"


def parse_int_ranges(text: str, *, minimum: int = 1) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < minimum or last < first:
                raise ValueError(f"invalid integer range with minimum {minimum}: {item!r}")
            values.extend(range(first, last + 1))
        else:
            value = int(item)
            if value < minimum:
                raise ValueError(f"values must be at least {minimum}: {value}")
            values.append(value)
    if not values:
        raise ValueError("integer list must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"integer list contains duplicates: {text!r}")
    return values


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(
    *,
    routes: int,
    team_width: int,
    lanes: int,
    experts: int,
    cpu_ids: list[int],
) -> tuple[AsyncMoEPlanV2, dict[str, int]]:
    if experts % lanes:
        raise ValueError(f"experts={experts} must be divisible by lanes={lanes}")
    active_cores = lanes * team_width
    if active_cores > len(cpu_ids):
        raise ValueError(f"plan needs {active_cores} CPUs, got {len(cpu_ids)}")

    task_experts: list[int] = []
    task_core_begins: list[int] = []
    task_threads: list[int] = []
    dep_offsets = [0]
    dependencies: list[int] = []
    for lane in range(lanes):
        previous: int | None = None
        for expert in range(lane, experts, lanes):
            task_id = len(task_experts)
            task_experts.append(expert)
            task_core_begins.append(lane * team_width)
            task_threads.append(team_width)
            if previous is not None:
                dependencies.append(previous)
            dep_offsets.append(len(dependencies))
            previous = task_id

    w13_target, w2_target = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4.select(routes, team_width)
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": active_cores,
            "thread_cpu_ids": cpu_ids[:active_cores],
            "task_expert_ids": task_experts,
            "task_core_begins": task_core_begins,
            "task_threads": task_threads,
            "task_dep_offsets": dep_offsets,
            "task_deps": dependencies,
        }
    )
    bridge["early_merge"] = False
    bridge["task_w13_window_bytes"] = [w13_target] * experts
    bridge["task_w2_window_bytes"] = [w2_target] * experts

    w13_geometry, w2_geometry = fused_moe_task_weight_windows(
        hidden_size=4096,
        intermediate_size=512,
        n_tile=8,
        inherited_target_bytes=0,
        w13_target_bytes=w13_target,
        w2_target_bytes=w2_target,
        w13_fallback_ranges=2,
    )
    geometry = {
        "w13_target_bytes": w13_target,
        "w2_target_bytes": w2_target,
        "w13_ranges": w13_geometry.ranges,
        "w2_ranges": w2_geometry.ranges,
        "w13_range_bytes": w13_geometry.max_range_bytes,
        "w2_range_bytes": w2_geometry.max_range_bytes,
        "w13_worker_bytes": w13_geometry.bytes_per_worker(team_width),
        "w2_worker_bytes": w2_geometry.bytes_per_worker(team_width),
    }
    return AsyncMoEPlanV2.from_dict(bridge), geometry


def measure(run: Callable[[], torch.Tensor], *, warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        output = run()
        _ = float(output.flatten()[0])
    samples_ms: list[float] = []
    for _ in range(runs):
        begin = time.perf_counter_ns()
        output = run()
        _ = float(output.flatten()[0])
        samples_ms.append((time.perf_counter_ns() - begin) / 1e6)
    return samples_ms


def benchmark_point(
    *,
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed,
    output: torch.Tensor,
    routes: int,
    width: int,
    lanes: int,
    experts: int,
    cpu_ids: list[int],
    warmup: int,
    runs: int,
    store_samples: bool,
) -> dict:
    plan, geometry = build_plan(
        routes=routes,
        team_width=width,
        lanes=lanes,
        experts=experts,
        cpu_ids=cpu_ids,
    )

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            plan,
            activation="silu",
            global_num_experts=256,
            skip_weighted=True,
            w13_split=True,
            weight_window_bytes=0,
            out=output,
        )

    full_samples = measure(run, warmup=warmup, runs=runs)
    groups = experts // lanes
    group_samples = [sample / groups for sample in full_samples]
    median_ms = statistics.median(group_samples)
    flops_per_expert = 6 * routes * 4096 * 512
    entry = {
        "routes": routes,
        "threads": width,
        "lanes": lanes,
        "active_cores": lanes * width,
        "measurement_experts": experts,
        "groups": groups,
        "median_ms": median_ms,
        "p10_ms": percentile(group_samples, 0.10),
        "p90_ms": percentile(group_samples, 0.90),
        "min_ms": min(group_samples),
        "full_call_median_ms": statistics.median(full_samples),
        "aggregate_tflops": lanes * flops_per_expert / median_ms / 1e9,
        "per_expert_tflops": flops_per_expert / median_ms / 1e9,
        **geometry,
    }
    if store_samples:
        entry["full_call_samples_ms"] = full_samples
        entry["group_samples_ms"] = group_samples
    return entry


def tensors_for_routes(
    *,
    routes: int,
    experts: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.empty((experts * routes, 4096), dtype=torch.bfloat16).normal_(
        0.0,
        0.01,
        generator=generator,
    )
    topk_ids = torch.empty((experts * routes, 1), dtype=torch.int32)
    for expert in range(experts):
        topk_ids[expert * routes : (expert + 1) * routes, 0] = expert
    topk_weights = torch.ones((experts * routes, 1), dtype=torch.float32)
    return hidden, topk_ids, topk_weights, torch.empty_like(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep production stage-window team widths on one NUMA rank.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", default=DEFAULT_ROUTES)
    parser.add_argument("--isolated-widths", default=DEFAULT_ISOLATED_WIDTHS)
    parser.add_argument("--homogeneous-widths", default=DEFAULT_HOMOGENEOUS_WIDTHS)
    parser.add_argument("--cpu-ids", default="0-95")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--isolated-experts", type=int, default=8)
    parser.add_argument("--homogeneous-experts", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--store-samples", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routes_grid = parse_int_ranges(args.routes)
    isolated_widths = parse_int_ranges(args.isolated_widths)
    homogeneous_widths = parse_int_ranges(args.homogeneous_widths)
    cpu_ids = parse_int_ranges(args.cpu_ids, minimum=0)
    if len(cpu_ids) != 96:
        raise ValueError(f"this calibrated sweep requires exactly 96 NUMA-local CPUs, got {len(cpu_ids)}")
    if max(isolated_widths + homogeneous_widths) > len(cpu_ids):
        raise ValueError("team width exceeds the available CPU set")
    invalid_homogeneous = [width for width in homogeneous_widths if len(cpu_ids) % width]
    if invalid_homogeneous:
        raise ValueError(f"homogeneous widths must divide 96 exactly: {invalid_homogeneous}")
    if args.num_experts < max(args.isolated_experts, args.homogeneous_experts):
        raise ValueError("--num-experts must cover every measured expert")
    if args.isolated_experts <= 0 or args.homogeneous_experts <= 0:
        raise ValueError("measurement expert counts must be positive")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty((args.num_experts, 1024, 4096), dtype=torch.bfloat16).normal_(
        0.0,
        0.01,
        generator=generator,
    )
    w2 = torch.empty((args.num_experts, 4096, 512), dtype=torch.bfloat16).normal_(
        0.0,
        0.01,
        generator=generator,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    del w13, w2
    if int(packed.gemm_backend) != 1:
        raise RuntimeError(f"SVE backend is required, got backend id {packed.gemm_backend}")

    entries: list[dict] = []
    point_order: list[dict[str, object]] = []
    randomizer = random.Random(args.seed)
    with torch.inference_mode():
        for routes in routes_grid:
            hidden, topk_ids, topk_weights, output = tensors_for_routes(
                routes=routes,
                experts=args.homogeneous_experts,
                generator=generator,
            )

            isolated_order = list(isolated_widths)
            homogeneous_order = list(homogeneous_widths)
            randomizer.shuffle(isolated_order)
            randomizer.shuffle(homogeneous_order)
            point_order.append(
                {
                    "routes": routes,
                    "isolated_widths": isolated_order,
                    "homogeneous_widths": homogeneous_order,
                }
            )

            isolated_rows = args.isolated_experts * routes
            for width in isolated_order:
                entry = benchmark_point(
                    hidden=hidden[:isolated_rows],
                    topk_ids=topk_ids[:isolated_rows],
                    topk_weights=topk_weights[:isolated_rows],
                    packed=packed,
                    output=output[:isolated_rows],
                    routes=routes,
                    width=width,
                    lanes=1,
                    experts=args.isolated_experts,
                    cpu_ids=cpu_ids,
                    warmup=args.warmup,
                    runs=args.runs,
                    store_samples=args.store_samples,
                )
                entry["mode"] = "isolated"
                entries.append(entry)
                print(
                    f"isolated   M={routes:<4} T={width:<2} "
                    f"time={entry['median_ms']:>8.3f} ms "
                    f"rate={entry['per_expert_tflops']:>7.3f} TFLOP/s "
                    f"window={entry['w13_worker_bytes'] // 1024:>4}/"
                    f"{entry['w2_worker_bytes'] // 1024:<4} KiB/T",
                    flush=True,
                )

            for width in homogeneous_order:
                lanes = len(cpu_ids) // width
                if args.homogeneous_experts % lanes:
                    raise ValueError(
                        f"--homogeneous-experts={args.homogeneous_experts} must be divisible by lanes={lanes} "
                        f"for width={width}"
                    )
                entry = benchmark_point(
                    hidden=hidden,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    packed=packed,
                    output=output,
                    routes=routes,
                    width=width,
                    lanes=lanes,
                    experts=args.homogeneous_experts,
                    cpu_ids=cpu_ids,
                    warmup=args.warmup,
                    runs=args.runs,
                    store_samples=args.store_samples,
                )
                entry["mode"] = "homogeneous"
                entries.append(entry)
                print(
                    f"homogeneous M={routes:<4} T={width:<2} lanes={lanes:<2} "
                    f"wave={entry['median_ms']:>8.3f} ms "
                    f"rate={entry['aggregate_tflops']:>7.3f} TFLOP/s "
                    f"window={entry['w13_worker_bytes'] // 1024:>4}/"
                    f"{entry['w2_worker_bytes'] // 1024:<4} KiB/T",
                    flush=True,
                )

    for mode in ("isolated", "homogeneous"):
        for routes in routes_grid:
            selected = [entry for entry in entries if entry["mode"] == mode and entry["routes"] == routes]
            if mode == "isolated":
                best = min(selected, key=lambda entry: entry["median_ms"])
                for entry in selected:
                    entry["regret"] = entry["median_ms"] / best["median_ms"] - 1.0
            else:
                best = max(selected, key=lambda entry: entry["aggregate_tflops"])
                for entry in selected:
                    entry["regret"] = best["aggregate_tflops"] / entry["aggregate_tflops"] - 1.0
            for entry in selected:
                entry["best_threads"] = best["threads"]

    native_module = importlib.import_module("fused_cpp._moe_C")
    extension = Path(native_module.__file__)
    payload = {
        "schema_version": 1,
        "kind": "stage_window_team_width_sweep",
        "target": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cpu_ids": cpu_ids,
            "numa_threads": len(cpu_ids),
        },
        "kernel": {
            "backend": "sve",
            "sve_implementation": "jit",
            "m_tail_policy": "xbyak_exact_m",
            "w13_split": True,
            "w13_split_chunks": 2,
            "stage_window_policy": AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4.name,
            "backend_n_tile": int(packed.backend_n_tile),
            "extension": str(extension),
            "extension_sha256": file_sha256(extension),
            "page_policy": os.environ.get("FUSED_CPP_PAGES", ""),
            "page_size_mb": os.environ.get("FUSED_CPP_PAGE_SIZE_MB", ""),
            "hugetlbfs_path": os.environ.get("FUSED_CPP_HUGETLBFS_PATH", ""),
        },
        "shape": {
            "hidden_size": 4096,
            "intermediate_size": 512,
            "global_experts": 256,
            "local_experts": 256,
            "activation": "silu",
            "dtype": "bf16",
        },
        "measurement": {
            "routes": routes_grid,
            "isolated_widths": isolated_widths,
            "homogeneous_widths": homogeneous_widths,
            "isolated_experts": args.isolated_experts,
            "homogeneous_experts": args.homogeneous_experts,
            "warmup": args.warmup,
            "runs": args.runs,
            "seed": args.seed,
            "point_order": point_order,
            "weight_reuse": "streaming_distinct_experts",
        },
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
