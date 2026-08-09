#!/usr/bin/env python3
"""Hold out analytical stage-window decisions against real fused-MoE runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from analytic_model import AnalyticMachineCalibration, AnalyticMoeCostModel  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from fused_cpp.moe.plan import upgrade_legacy_async_plan  # noqa: E402


def parse_int_ranges(text: str, *, minimum: int = 1) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            begin_text, end_text = item.split("-", 1)
            begin, end = int(begin_text), int(end_text)
            if begin < minimum or end < begin:
                raise argparse.ArgumentTypeError(f"invalid range {item!r}")
            values.extend(range(begin, end + 1))
        else:
            value = int(item)
            if value < minimum:
                raise argparse.ArgumentTypeError(f"value must be >= {minimum}: {value}")
            values.append(value)
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer list must be non-empty and unique")
    return values


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_targets(model: AnalyticMoeCostModel, pair: tuple[int, int]) -> tuple[int, int]:
    return (
        model._w13_geometry.max_range_bytes if pair[0] < 0 else pair[0],
        model._w2_geometry.max_range_bytes if pair[1] < 0 else pair[1],
    )


def candidate_pairs(policy, model: AnalyticMoeCostModel, routes: int, threads: int) -> list[dict]:
    selected = policy.select(routes, threads)
    selected_resolved = resolved_targets(model, selected)
    explanation = policy.explain(routes, threads)
    stage_rows = explanation["candidates"]
    candidates: dict[tuple[int, int], set[str]] = {}

    def add(pair: tuple[int, int], label: str) -> None:
        key = resolved_targets(model, pair)
        candidates.setdefault(key, set()).add(label)

    add(selected, "analytic")
    add((-1, -1), "inherited")
    w13_rows = stage_rows["w13"]["rows"]
    w2_rows = stage_rows["w2"]["rows"]
    for row in w13_rows:
        add((int(row["range_bytes"]), selected_resolved[1]), f"w13:{int(row['worker_bytes'])}")
    for row in w2_rows:
        add((selected_resolved[0], int(row["range_bytes"])), f"w2:{int(row['worker_bytes'])}")

    def neighbors(rows: list[dict], selected_range: int) -> list[int]:
        ordered = sorted({int(row["range_bytes"]) for row in rows})
        index = ordered.index(selected_range)
        return ordered[max(0, index - 1) : min(len(ordered), index + 2)]

    for w13 in neighbors(w13_rows, selected_resolved[0]):
        for w2 in neighbors(w2_rows, selected_resolved[1]):
            add((w13, w2), "local_cross")

    result: list[dict] = []
    for pair, labels in candidates.items():
        w13_score = model.score_stage_window("w13", routes, threads, pair[0])
        w2_score = model.score_stage_window("w2", routes, threads, pair[1])
        result.append(
            {
                "w13_target_bytes": pair[0],
                "w2_target_bytes": pair[1],
                "labels": sorted(labels),
                "analytic_objective_ns": w13_score.objective_ns + w2_score.objective_ns,
                "analytic_w13_objective_ns": w13_score.objective_ns,
                "analytic_w2_objective_ns": w2_score.objective_ns,
                "w13_worker_bytes": w13_score.worker_bytes,
                "w2_worker_bytes": w2_score.worker_bytes,
                "w13_ranges": w13_score.ranges,
                "w2_ranges": w2_score.ranges,
            }
        )
    return sorted(result, key=lambda item: (item["analytic_objective_ns"], item["w13_target_bytes"], item["w2_target_bytes"]))


def build_plan(
    *,
    experts: int,
    threads: int,
    cpu_ids: list[int],
    w13_target: int,
    w2_target: int,
    start_lane: int,
) -> AsyncMoEPlanV2:
    if len(cpu_ids) % threads:
        raise ValueError(f"team width {threads} must divide {len(cpu_ids)} CPUs")
    lanes = len(cpu_ids) // threads
    if experts % lanes:
        raise ValueError(f"experts={experts} must be divisible by lanes={lanes}")
    task_experts: list[int] = []
    task_core_begins: list[int] = []
    task_threads: list[int] = []
    dependencies: list[int] = []
    dep_offsets = [0]
    for lane in range(lanes):
        previous: int | None = None
        physical_lane = (lane + start_lane) % lanes
        for wave in range(experts // lanes):
            expert = (physical_lane + wave * lanes) % experts
            task_id = len(task_experts)
            task_experts.append(expert)
            task_core_begins.append(lane * threads)
            task_threads.append(threads)
            if previous is not None:
                dependencies.append(previous)
            dep_offsets.append(len(dependencies))
            previous = task_id
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": len(cpu_ids),
            "thread_cpu_ids": cpu_ids,
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
    return AsyncMoEPlanV2.from_dict(bridge)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", default="0-95")
    parser.add_argument("--routes", default="28,72,120,216,320,768,2040")
    parser.add_argument("--widths", default="4,8")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--measurement-experts", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--store-samples", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    cpu_ids = parse_int_ranges(args.cpu_ids, minimum=0)
    routes_grid = parse_int_ranges(args.routes)
    widths = parse_int_ranges(args.widths)
    if any(len(cpu_ids) % width for width in widths):
        raise ValueError("every width must divide the selected CPU count")
    if any(args.measurement_experts % (len(cpu_ids) // width) for width in widths):
        raise ValueError("measurement experts must divide into whole waves at every width")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs positive")

    calibration = AnalyticMachineCalibration.from_path(args.calibration)
    if calibration.cores_per_rank != len(cpu_ids):
        raise ValueError(
            f"calibration has {calibration.cores_per_rank} cores, selected CPU set has {len(cpu_ids)}"
        )
    model = AnalyticMoeCostModel(
        calibration,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        global_experts=args.measurement_experts,
        local_experts=args.measurement_experts,
        mode="standalone",
        supported_widths=widths,
    )
    policy = model.default_task_stage_window_policy(num_cores=len(cpu_ids), cpu_ids=cpu_ids)
    if policy is None:
        raise RuntimeError("analytical model did not generate a stage-window policy")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty(
        (args.measurement_experts, 2 * args.intermediate_size, args.hidden_size),
        dtype=torch.bfloat16,
    ).normal_(0.0, 0.01, generator=generator)
    w2 = torch.empty(
        (args.measurement_experts, args.hidden_size, args.intermediate_size),
        dtype=torch.bfloat16,
    ).normal_(0.0, 0.01, generator=generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    del w13, w2
    if int(packed.backend_n_tile) != model.policy.backend_n_tile:
        raise RuntimeError(
            "analytical calibration/runtime N-tile mismatch: "
            f"calibration={model.policy.backend_n_tile}, runtime={packed.backend_n_tile}"
        )

    max_rows = args.measurement_experts * max(routes_grid)
    hidden = torch.empty((max_rows, args.hidden_size), dtype=torch.bfloat16).normal_(
        0.0,
        0.01,
        generator=generator,
    )
    topk_ids = torch.empty((max_rows, 1), dtype=torch.int32)
    topk_weights = torch.ones((max_rows, 1), dtype=torch.float32)
    output = torch.empty_like(hidden)
    entries: list[dict] = []
    order_log: list[dict] = []
    randomizer = random.Random(args.seed)
    for routes in routes_grid:
        rows = args.measurement_experts * routes
        for expert in range(args.measurement_experts):
            topk_ids[expert * routes : (expert + 1) * routes, 0] = expert
        for threads in widths:
            points = candidate_pairs(policy, model, routes, threads)
            randomizer.shuffle(points)
            lanes = len(cpu_ids) // threads
            jobs: list[dict] = []
            for point in points:
                plans = [
                    build_plan(
                        experts=args.measurement_experts,
                        threads=threads,
                        cpu_ids=cpu_ids,
                        w13_target=point["w13_target_bytes"],
                        w2_target=point["w2_target_bytes"],
                        start_lane=start,
                    )
                    for start in range(min(lanes, args.warmup + args.runs))
                ]
                jobs.append({"point": point, "plans": plans, "samples_ms": []})

            round_orders: list[list[int]] = []
            for round_index in range(args.warmup + args.runs):
                round_order = list(range(len(jobs)))
                randomizer.shuffle(round_order)
                round_orders.append(round_order)
                for job_index in round_order:
                    job = jobs[job_index]
                    plans = job["plans"]
                    plan = plans[round_index % len(plans)]
                    begin = time.perf_counter_ns()
                    result = fused_moe_bf16_tiled_async_plan(
                        hidden[:rows],
                        packed,
                        topk_weights[:rows],
                        topk_ids[:rows],
                        plan,
                        activation="silu",
                        global_num_experts=args.measurement_experts,
                        skip_weighted=True,
                        w13_split=True,
                        weight_window_bytes=0,
                        out=output[:rows],
                    )
                    _ = float(result.flatten()[0])
                    if round_index >= args.warmup:
                        job["samples_ms"].append((time.perf_counter_ns() - begin) / 1e6)

            order_log.append(
                {
                    "routes": routes,
                    "threads": threads,
                    "pairs": [[point["w13_target_bytes"], point["w2_target_bytes"]] for point in points],
                    "round_orders": round_orders,
                }
            )
            for job in jobs:
                point = job["point"]
                samples = job["samples_ms"]
                median_ms = statistics.median(samples)
                flops = 6 * routes * args.hidden_size * args.intermediate_size * args.measurement_experts
                entry = {
                    "routes": routes,
                    "threads": threads,
                    "lanes": lanes,
                    "waves": args.measurement_experts // lanes,
                    "median_ms": median_ms,
                    "p10_ms": percentile(samples, 0.10),
                    "p90_ms": percentile(samples, 0.90),
                    "aggregate_tflops": flops / median_ms / 1e9,
                    **point,
                }
                if args.store_samples:
                    entry["samples_ms"] = samples
                entries.append(entry)
                marker = "*" if "analytic" in point["labels"] else " "
                print(
                    f"{marker} M={routes:<4} T={threads:<2} "
                    f"omega={point['w13_worker_bytes'] // 1024:>4}/"
                    f"{point['w2_worker_bytes'] // 1024:<4} KiB "
                    f"R={point['w13_ranges']:>3}/{point['w2_ranges']:<3} "
                    f"time={median_ms:>8.3f} ms rate={entry['aggregate_tflops']:>7.3f} TF/s",
                    flush=True,
                )

    groups: list[dict] = []
    for routes in routes_grid:
        for threads in widths:
            rows = [row for row in entries if row["routes"] == routes and row["threads"] == threads]
            oracle = min(rows, key=lambda row: row["median_ms"])
            selected = next(row for row in rows if "analytic" in row["labels"])
            inherited = next(row for row in rows if "inherited" in row["labels"])
            groups.append(
                {
                    "routes": routes,
                    "threads": threads,
                    "analytic_ms": selected["median_ms"],
                    "oracle_ms": oracle["median_ms"],
                    "inherited_ms": inherited["median_ms"],
                    "regret": selected["median_ms"] / oracle["median_ms"] - 1.0,
                    "gain_vs_inherited": inherited["median_ms"] / selected["median_ms"] - 1.0,
                    "analytic_pair": [selected["w13_worker_bytes"], selected["w2_worker_bytes"]],
                    "oracle_pair": [oracle["w13_worker_bytes"], oracle["w2_worker_bytes"]],
                    "oracle_labels": oracle["labels"],
                }
            )

    payload = {
        "schema_version": 1,
        "kind": "analytic_stage_window_holdout",
        "config": {
            "calibration": str(args.calibration),
            "calibration_sha256": sha256(args.calibration),
            "machine_id": calibration.machine_id,
            "cpu_ids": cpu_ids,
            "routes": routes_grid,
            "widths": widths,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "measurement_experts": args.measurement_experts,
            "warmup": args.warmup,
            "runs": args.runs,
            "sampling": "shuffled_group_round_robin",
            "policy": policy.name,
            "pages": os.environ.get("FUSED_CPP_PAGES"),
            "page_size_mb": os.environ.get("FUSED_CPP_PAGE_SIZE_MB"),
        },
        "groups": groups,
        "entries": entries,
        "point_order": order_log,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
