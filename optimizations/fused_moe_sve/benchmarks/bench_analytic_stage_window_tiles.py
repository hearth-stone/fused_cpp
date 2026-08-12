#!/usr/bin/env python3
"""Hold out the analytical tile-window selector against real fused-MoE runs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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
from stage_window_policy import default_stage_window_policy  # noqa: E402


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


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = 0.5 * (begin + end - 1)
        for index in order[begin:end]:
            ranks[index] = rank
        begin = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return math.nan
    ranks_a, ranks_b = _ranks(values_a), _ranks(values_b)
    mean_a, mean_b = statistics.mean(ranks_a), statistics.mean(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denominator = math.sqrt(
        sum((a - mean_a) ** 2 for a in ranks_a)
        * sum((b - mean_b) ** 2 for b in ranks_b)
    )
    return numerator / denominator if denominator else math.nan


def _abi_tiles(score) -> int:
    return 0 if score.is_full_stripe else score.window_tiles


def candidate_pairs(
    policy,
    model: AnalyticMoeCostModel,
    routes: int,
    threads: int,
    *,
    full_cartesian: bool,
) -> list[dict]:
    selected = policy.select(routes, threads)
    w13_scores = policy.stage_scores("w13", routes, threads)
    w2_scores = policy.stage_scores("w2", routes, threads)
    w13_tiles = [_abi_tiles(score) for score in w13_scores]
    w2_tiles = [_abi_tiles(score) for score in w2_scores]
    candidates: dict[tuple[int, int], set[str]] = {}

    def add(pair: tuple[int, int], label: str) -> None:
        candidates.setdefault((int(pair[0]), int(pair[1])), set()).add(label)

    add(selected, "analytic")
    add((0, 0), "full_stripe")
    empirical = default_stage_window_policy(
        hidden_size=model.hidden_size,
        intermediate_size=model.intermediate_size,
        backend_n_tile=model.policy.backend_n_tile,
    )
    if empirical is not None:
        add(empirical.select(routes, threads), "empirical_v5")

    if full_cartesian:
        for pair in itertools.product(w13_tiles, w2_tiles):
            add(pair, "cartesian")
    else:
        for window in w13_tiles:
            add((window, selected[1]), "w13_axis")
        for window in w2_tiles:
            add((selected[0], window), "w2_axis")

        def local(stage: str, values: list[int], selected_value: int) -> list[int]:
            resolved = sorted(
                values,
                key=lambda value: model.score_stage_window(
                    stage,
                    routes,
                    threads,
                    value,
                ).window_tiles,
            )
            index = resolved.index(selected_value)
            return resolved[max(0, index - 1) : min(len(resolved), index + 2)]

        for pair in itertools.product(
            local("w13", w13_tiles, selected[0]),
            local("w2", w2_tiles, selected[1]),
        ):
            add(pair, "local_cross")

    result: list[dict] = []
    for (w13, w2), labels in candidates.items():
        w13_evaluation = policy.stage_evaluation("w13", routes, threads, w13)
        w2_evaluation = policy.stage_evaluation("w2", routes, threads, w2)
        w13_score = w13_evaluation.score
        w2_score = w2_evaluation.score
        result.append(
            {
                "w13_window_tiles": w13,
                "w2_window_tiles": w2,
                "w13_resolved_tiles": w13_score.window_tiles,
                "w2_resolved_tiles": w2_score.window_tiles,
                "w13_worker_bytes": w13_score.owner_window_bytes,
                "w2_worker_bytes": w2_score.owner_window_bytes,
                "w13_windows": w13_score.windows,
                "w2_windows": w2_score.windows,
                "analytic_objective_ns": (
                    w13_evaluation.policy_objective_ns + w2_evaluation.policy_objective_ns
                ),
                "analytic_w13_ns": w13_evaluation.policy_objective_ns,
                "analytic_w2_ns": w2_evaluation.policy_objective_ns,
                "analytic_w13_task_local_ns": w13_score.objective_ns,
                "analytic_w2_task_local_ns": w2_score.objective_ns,
                "analytic_w13_shared_b_llc_spill_ns": w13_evaluation.shared_b_llc_spill_ns,
                "analytic_w2_shared_b_llc_spill_ns": w2_evaluation.shared_b_llc_spill_ns,
                "labels": sorted(labels),
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["analytic_objective_ns"],
            item["w13_resolved_tiles"],
            item["w2_resolved_tiles"],
        ),
    )


def build_plan(
    *,
    experts: int,
    threads: int,
    cpu_ids: list[int],
    w13_window_tiles: int,
    w2_window_tiles: int,
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
    bridge["task_w13_window_tiles"] = [w13_window_tiles] * experts
    bridge["task_w2_window_tiles"] = [w2_window_tiles] * experts
    return AsyncMoEPlanV2.from_dict(bridge)


def interaction_summary(rows: list[dict], selected: dict) -> dict | None:
    by_pair = {
        (row["w13_resolved_tiles"], row["w2_resolved_tiles"]): row
        for row in rows
    }
    w13_values = sorted({pair[0] for pair in by_pair})
    w2_values = sorted({pair[1] for pair in by_pair})
    if len(by_pair) != len(w13_values) * len(w2_values):
        return None
    anchor = (selected["w13_resolved_tiles"], selected["w2_resolved_tiles"])
    anchor_ms = by_pair[anchor]["median_ms"]
    sample_counts = {len(row.get("samples_ms", ())) for row in rows}
    paired = len(sample_counts) == 1 and next(iter(sample_counts), 0) > 0
    residuals = []
    for w13, w2 in itertools.product(w13_values, w2_values):
        if paired:
            residual_samples = [
                value - w13_axis - w2_axis + anchor_value
                for value, w13_axis, w2_axis, anchor_value in zip(
                    by_pair[(w13, w2)]["samples_ms"],
                    by_pair[(w13, anchor[1])]["samples_ms"],
                    by_pair[(anchor[0], w2)]["samples_ms"],
                    by_pair[anchor]["samples_ms"],
                )
            ]
            residual = statistics.median(residual_samples)
            row = {
                "w13_tiles": w13,
                "w2_tiles": w2,
                "residual_ms": residual,
                "p10_residual_ms": percentile(residual_samples, 0.10),
                "p90_residual_ms": percentile(residual_samples, 0.90),
            }
        else:
            residual = (
                by_pair[(w13, w2)]["median_ms"]
                - by_pair[(w13, anchor[1])]["median_ms"]
                - by_pair[(anchor[0], w2)]["median_ms"]
                + anchor_ms
            )
            row = {"w13_tiles": w13, "w2_tiles": w2, "residual_ms": residual}
        residuals.append(row)
    worst = max(residuals, key=lambda row: abs(row["residual_ms"]))
    return {
        "anchor": list(anchor),
        "anchor_ms": anchor_ms,
        "method": "paired_round_residual" if paired else "independent_median_residual",
        "max_absolute_residual_ms": abs(worst["residual_ms"]),
        "max_absolute_residual_fraction": abs(worst["residual_ms"]) / anchor_ms,
        "worst": worst,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", default="0-95")
    parser.add_argument("--routes", default="12,72,120,216,320,768,2040")
    parser.add_argument("--widths", default="1,2,4,8,16,32")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--measurement-experts", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--full-cartesian", action="store_true")
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
        raise ValueError("measurement experts must form whole waves at every width")
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
        supported_widths=widths,
    )
    policy = model.shadow_stage_window_policy()

    torch.set_num_threads(1)
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
        0.0, 0.01, generator=generator
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
            points = candidate_pairs(
                policy,
                model,
                routes,
                threads,
                full_cartesian=args.full_cartesian,
            )
            randomizer.shuffle(points)
            lanes = len(cpu_ids) // threads
            jobs: list[dict] = []
            for point in points:
                plans = [
                    build_plan(
                        experts=args.measurement_experts,
                        threads=threads,
                        cpu_ids=cpu_ids,
                        w13_window_tiles=point["w13_window_tiles"],
                        w2_window_tiles=point["w2_window_tiles"],
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
                    plan = job["plans"][round_index % len(job["plans"])]
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
                        out=output[:rows],
                    )
                    _ = int(result.view(torch.int16).flatten()[0])
                    if round_index >= args.warmup:
                        job["samples_ms"].append((time.perf_counter_ns() - begin) / 1e6)

            order_log.append(
                {
                    "routes": routes,
                    "threads": threads,
                    "pairs": [
                        [point["w13_window_tiles"], point["w2_window_tiles"]]
                        for point in points
                    ],
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
                    f"w={point['w13_resolved_tiles']:>3}/{point['w2_resolved_tiles']:<3} "
                    f"R={point['w13_windows']:>3}/{point['w2_windows']:<3} "
                    f"time={median_ms:>8.3f} ms rate={entry['aggregate_tflops']:>7.3f} TF/s",
                    flush=True,
                )

    groups: list[dict] = []
    for routes in routes_grid:
        for threads in widths:
            rows = [row for row in entries if row["routes"] == routes and row["threads"] == threads]
            oracle = min(rows, key=lambda row: row["median_ms"])
            selected = next(row for row in rows if "analytic" in row["labels"])
            empirical_rows = [row for row in rows if "empirical_v5" in row["labels"]]
            full = next(row for row in rows if "full_stripe" in row["labels"])
            groups.append(
                {
                    "routes": routes,
                    "threads": threads,
                    "analytic_ms": selected["median_ms"],
                    "oracle_ms": oracle["median_ms"],
                    "full_stripe_ms": full["median_ms"],
                    "empirical_v5_ms": empirical_rows[0]["median_ms"] if empirical_rows else None,
                    "analytic_regret": selected["median_ms"] / oracle["median_ms"] - 1.0,
                    "empirical_v5_regret": (
                        empirical_rows[0]["median_ms"] / oracle["median_ms"] - 1.0
                        if empirical_rows
                        else None
                    ),
                    "gain_vs_full_stripe": full["median_ms"] / selected["median_ms"] - 1.0,
                    "analytic_pair": [
                        selected["w13_resolved_tiles"],
                        selected["w2_resolved_tiles"],
                    ],
                    "oracle_pair": [oracle["w13_resolved_tiles"], oracle["w2_resolved_tiles"]],
                    "oracle_labels": oracle["labels"],
                    "rank_spearman": spearman(
                        [row["analytic_objective_ns"] for row in rows],
                        [row["median_ms"] for row in rows],
                    ),
                    "interaction": interaction_summary(rows, selected),
                }
            )

    payload = {
        "schema_version": 1,
        "kind": "analytic_stage_window_tiles_holdout",
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
            "candidate_mode": "full_cartesian" if args.full_cartesian else "axes_plus_local_cross",
            "policy": policy.name,
            "pages": os.environ.get("FUSED_CPP_PAGES"),
            "page_size_mb": os.environ.get("FUSED_CPP_PAGE_SIZE_MB"),
            "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH"),
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
