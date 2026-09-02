#!/usr/bin/env python3
"""Calibrate the 2x1T-concurrent to 1x2T-serial MoE lane transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR)]

from analytic_model import (  # noqa: E402
    AnalyticMoeCostModel,
    NarrowTeamContentionCorrection,
)
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)


TARGET_ROUTE_CASES = {
    "balanced": ((35, 13, 8, 3, 1), (27, 12, 6, 2, 1)),
    "head_heavy": ((68, 22, 10, 3, 1), (13, 8, 5, 2, 1)),
    "tail_dense": ((16, 10, 6, 3, 1), (15, 9, 5, 2, 1)),
}
WIDE_BACKGROUND_ROUTES = ((1800, 600),) * 4
NARROW_BACKGROUND_ROUTES = ((68, 35, 13),) * 14
MODES = (
    "pair_isolated",
    "merge_isolated",
    "pair_background",
    "merge_background",
)
TARGET_CORE_BEGIN = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_narrow_lane_merge"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-calibration", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": ordered[round((len(ordered) - 1) * 0.10)],
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "samples_ms": values,
    }


def _target_specs(
    model: AnalyticMoeCostModel,
    target_routes: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    merged: bool,
) -> list[tuple[int, int, int, int]]:
    lane_tasks = [
        [(expert, routes) for expert, routes in enumerate(target_routes[0])],
        [(expert + len(target_routes[0]), routes) for expert, routes in enumerate(target_routes[1])],
    ]
    if not merged:
        return [
            (expert, routes, TARGET_CORE_BEGIN + lane, 1)
            for lane, tasks in enumerate(lane_tasks)
            for expert, routes in tasks
        ]
    tasks = sorted(
        (*lane_tasks[0], *lane_tasks[1]),
        key=lambda item: (-model.T_iso(item[1], 2), item[0]),
    )
    return [(expert, routes, TARGET_CORE_BEGIN, 2) for expert, routes in tasks]


def _background_specs(first_expert: int) -> list[tuple[int, int, int, int]]:
    tasks = []
    expert = first_expert
    for lane, routes_for_lane in enumerate(WIDE_BACKGROUND_ROUTES):
        for routes in routes_for_lane:
            tasks.append((expert, routes, 16 * lane, 16))
            expert += 1
    for lane, routes_for_lane in enumerate(NARROW_BACKGROUND_ROUTES):
        for routes in routes_for_lane:
            tasks.append((expert, routes, TARGET_CORE_BEGIN + 2 + lane, 1))
            expert += 1
    return tasks


def _build_bridge(
    model: AnalyticMoeCostModel,
    target_routes: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    thread_cpu_ids: tuple[int, ...],
    mode: str,
) -> tuple[dict[str, object], dict[int, int]]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    merged = mode.startswith("merge")
    isolated = mode.endswith("isolated")
    target = _target_specs(model, target_routes, merged=merged)
    tasks = target + _background_specs(len(target))
    target_tails = []
    dependencies: list[list[int]] = []
    lane_prior: dict[tuple[int, int], int] = {}
    for task_id, (_, _, core_begin, width) in enumerate(tasks):
        lane = (core_begin, width)
        if lane in lane_prior:
            deps = [lane_prior[lane]]
        elif task_id >= len(target) and isolated:
            deps = list(target_tails)
        else:
            deps = []
        dependencies.append(deps)
        lane_prior[lane] = task_id
        if task_id < len(target):
            target_tails = [
                previous
                for previous in target_tails
                if tasks[previous][2:4] != (core_begin, width)
            ]
            target_tails.append(task_id)

    flat_dependencies = [dependency for values in dependencies for dependency in values]
    dependency_offsets = [0]
    for values in dependencies:
        dependency_offsets.append(dependency_offsets[-1] + len(values))
    policy = model.shadow_stage_window_policy()
    windows = [policy.select(routes, width) for _, routes, _, width in tasks]
    widths = [width for _, _, _, width in tasks]
    num_tasks = len(tasks)
    bridge = {
        "plan_version": 2,
        "execution_mode": "strict",
        "num_threads": len(thread_cpu_ids),
        "thread_cpu_ids": list(thread_cpu_ids),
        "task_expert_ids": [expert for expert, _, _, _ in tasks],
        "task_core_begins": [core_begin for _, _, core_begin, _ in tasks],
        "task_threads": widths,
        "task_dep_offsets": dependency_offsets,
        "task_deps": flat_dependencies,
        "task_preferred_threads": widths,
        "task_min_threads": widths,
        "task_max_threads": widths,
        "task_allowed_thread_offsets": list(range(num_tasks + 1)),
        "task_allowed_threads": widths,
        "task_placement_modes": [0] * num_tasks,
        "task_numa_nodes": [-1] * num_tasks,
        "task_stage_ids": [0] * num_tasks,
        "task_resize_points": [0] * num_tasks,
        "task_range_granularities": [0] * num_tasks,
        "task_w13_window_tiles": [window[0] for window in windows],
        "task_w2_window_tiles": [window[1] for window in windows],
        "early_merge": False,
    }
    return bridge, {expert: routes for expert, routes, _, _ in tasks}


def _placed_tasks(
    bridge: dict[str, object],
    routes_by_expert: dict[int, int],
) -> list[tuple[int, int, tuple[int, ...], list[int]]]:
    cpu_ids = tuple(int(value) for value in bridge["thread_cpu_ids"])
    experts = [int(value) for value in bridge["task_expert_ids"]]
    begins = [int(value) for value in bridge["task_core_begins"]]
    widths = [int(value) for value in bridge["task_threads"]]
    offsets = [int(value) for value in bridge["task_dep_offsets"]]
    dependencies = [int(value) for value in bridge["task_deps"]]
    return [
        (
            routes_by_expert[expert],
            width,
            cpu_ids[core_begin : core_begin + width],
            dependencies[offsets[index] : offsets[index + 1]],
        )
        for index, (expert, core_begin, width) in enumerate(zip(experts, begins, widths, strict=True))
    ]


def _parse_target_calls(path: Path, target_tasks: int) -> list[dict[int, tuple[float, float]]]:
    calls = []
    current: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
        if words[0] == "MOE_CALL":
            current = {}
        elif words[0] == "PHASE":
            task = int(fields["group"])
            if not 0 <= task < target_tasks:
                continue
            start_ms = float(fields["start_ms"])
            end_ms = float(fields["end_ms"])
            previous = current.get(task)
            current[task] = (
                start_ms if previous is None else min(previous[0], start_ms),
                end_ms if previous is None else max(previous[1], end_ms),
            )
        elif words[0] == "MOE_CALL_END":
            if len(current) != target_tasks:
                raise RuntimeError(
                    f"trace call has {len(current)} target tasks, expected {target_tasks}"
                )
            calls.append(dict(current))
    if not calls:
        raise RuntimeError(f"no complete target calls in {path}")
    return calls


def _target_spans(calls: list[dict[int, tuple[float, float]]]) -> list[float]:
    return [
        max(end for _, end in call.values()) - min(start for start, _ in call.values())
        for call in calls
    ]


def _model_with_narrow_calibration(
    model: AnalyticMoeCostModel,
    *,
    width2_expert_fixed_ns: float,
    width2_route_ns: float,
    width1_correction: float = 1.0,
    width2_correction: float = 1.0,
) -> AnalyticMoeCostModel:
    overheads = model.calibration.overheads
    calibration = replace(
        model.calibration,
        overheads=replace(
            overheads,
            by_width=(
                *(
                    point
                    for point in overheads.by_width
                    if point[0] != 2
                ),
                (2, float(width2_expert_fixed_ns), float(width2_route_ns)),
            ),
        ),
        narrow_team_contention_correction=NarrowTeamContentionCorrection(
            full_cohort_correction=(
                (1, float(width1_correction)),
                (2, float(width2_correction)),
            ),
        ),
    )
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=model.hidden_size,
        intermediate_size=model.intermediate_size,
        global_experts=model.policy.global_experts,
        local_experts=model.local_experts,
        mode=model.policy.mode,
        degree=model.policy.degree,
        concurrent_ranks=model.policy.concurrent_ranks,
        down_output_element_bytes=model.down_output_element_bytes,
    )


def _model_target_span_ms(
    model: AnalyticMoeCostModel,
    bridge: dict[str, object],
    routes_by_expert: dict[int, int],
    target_tasks: int,
) -> float:
    explanation = model.explain_dag_placed(_placed_tasks(bridge, routes_by_expert))
    return (
        max(float(value) for value in explanation["task_finish_ns"][:target_tasks])
        - model.call_setup_ns
    ) / 1.0e6


def _fit_width2_overhead(
    model: AnalyticMoeCostModel,
    cases: dict[str, dict[str, object]],
) -> tuple[float, float]:
    residuals = []
    for case in cases.values():
        bridge = case["bridges"]["merge_isolated"]
        routes_by_expert = case["routes_by_expert"]["merge_isolated"]
        experts = [int(value) for value in bridge["task_expert_ids"]]
        calls = case["trace_calls"]["merge_isolated"]
        for task, expert in enumerate(experts[: int(case["target_tasks"])]):
            measured_ms = statistics.median(call[task][1] - call[task][0] for call in calls)
            routes = routes_by_expert[expert]
            residuals.append((float(routes), measured_ms * 1.0e6 - model.T_iso(routes, 2)))
    count = len(residuals)
    sum_x = sum(routes for routes, _ in residuals)
    sum_y = sum(residual for _, residual in residuals)
    sum_xx = sum(routes * routes for routes, _ in residuals)
    sum_xy = sum(routes * residual for routes, residual in residuals)
    denominator = count * sum_xx - sum_x * sum_x
    route_ns = max((count * sum_xy - sum_x * sum_y) / denominator, 0.0)
    expert_fixed_ns = max((sum_y - route_ns * sum_x) / count, 0.0)
    return expert_fixed_ns, route_ns


def _fit_narrow_corrections(
    model: AnalyticMoeCostModel,
    cases: dict[str, dict[str, object]],
) -> tuple[float, float]:
    overheads = model.calibration.overheads
    width2 = next(point for point in overheads.by_width if point[0] == 2)

    def fit(*, mode: str, width1: float | None = None) -> float:
        def residual(correction: float) -> float:
            candidate = _model_with_narrow_calibration(
                model,
                width2_expert_fixed_ns=width2[1],
                width2_route_ns=width2[2],
                width1_correction=correction if width1 is None else width1,
                width2_correction=1.0 if width1 is None else correction,
            )
            errors = []
            for case in cases.values():
                predicted = _model_target_span_ms(
                    candidate,
                    case["bridges"][mode],
                    case["routes_by_expert"][mode],
                    int(case["target_tasks"]),
                )
                measured = float(case["measured_target_span_ms"][mode])
                errors.append(math.log(predicted / measured))
            return statistics.median(errors)

        lower, upper = 0.05, 2.0
        if residual(lower) >= 0.0:
            return lower
        if residual(upper) <= 0.0:
            return upper
        for _ in range(48):
            midpoint = (lower + upper) / 2.0
            if residual(midpoint) < 0.0:
                lower = midpoint
            else:
                upper = midpoint
        return (lower + upper) / 2.0

    width1 = fit(mode="pair_background")
    width2_correction = fit(mode="merge_background", width1=width1)
    return width1, width2_correction


def _write_calibration(
    source: Path,
    output: Path,
    *,
    width2_expert_fixed_ns: float,
    width2_route_ns: float,
    width1_correction: float,
    width2_correction: float,
    args: argparse.Namespace,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    overheads = payload.setdefault("overheads", {})
    by_width = [point for point in overheads.get("by_width", []) if int(point["threads"]) != 2]
    overheads["by_width"] = [
        *by_width,
        {
            "threads": 2,
            "expert_fixed_ns": width2_expert_fixed_ns,
            "route_ns": width2_route_ns,
        },
    ]
    payload.setdefault("planner", {})["narrow_team_contention_correction"] = {
        "full_cohort_correction": [
            {"threads": 1, "correction": width1_correction},
            {"threads": 2, "correction": width2_correction},
        ]
    }
    payload.setdefault("provenance", {})["narrow_lane_merge_transition"] = {
        "benchmark": "optimizations/fused_moe_sve/benchmarks/bench_narrow_lane_merge_transition.py",
        "machine": "Arm-codex-internal NUMA3 CPUs 240-319",
        "fit_target": "background/isolated target-span dilation with fixed 2-core ownership",
        "fit_statistic": "zero median log span error through complete placed event simulation",
        "full_cohort_correction": {
            "1": width1_correction,
            "2": width2_correction,
        },
        "width2_expert_overhead": {
            "expert_fixed_ns": width2_expert_fixed_ns,
            "route_ns": width2_route_ns,
            "fit_observation": "per-task phase span in merge_isolated only",
        },
        "trace_runs": args.runs,
        "weight_copies": args.weight_copies,
        "early_merge": False,
        "holdout_policy": "isolated pair, merged 2T plans, and captured planner traces excluded from fit",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80:
        raise ValueError("the initial narrow-lane transition design requires exactly 80 threads")
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    expert_count = 10 + sum(map(len, WIDE_BACKGROUND_ROUTES)) + sum(map(len, NARROW_BACKGROUND_ROUTES))
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=expert_count,
        local_experts=expert_count,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty((expert_count, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((expert_count, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = [
        prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend="arm_sve_bf16",
        )
        for _ in range(args.weight_copies)
    ]
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"

    case_results: dict[str, dict[str, object]] = {}
    for case_index, (case_name, target_routes) in enumerate(TARGET_ROUTE_CASES.items()):
        built = {
            mode: _build_bridge(model, target_routes, thread_cpu_ids=cpu_ids, mode=mode)
            for mode in MODES
        }
        bridges = {mode: value[0] for mode, value in built.items()}
        routes_by_expert = {mode: value[1] for mode, value in built.items()}
        plans = {mode: AsyncMoEPlanV2.from_dict(bridge) for mode, bridge in bridges.items()}
        route_map = routes_by_expert[MODES[0]]
        topk_ids = torch.repeat_interleave(
            torch.arange(expert_count, dtype=torch.int32),
            torch.tensor([route_map[expert] for expert in range(expert_count)], dtype=torch.int64),
        ).reshape(-1, 1)
        hidden = torch.empty((topk_ids.shape[0], args.hidden), dtype=torch.bfloat16)
        hidden.normal_(mean=0.0, std=0.01, generator=generator)
        topk_weights = torch.ones((topk_ids.shape[0], 1), dtype=torch.float32)
        outputs = {mode: torch.empty_like(hidden) for mode in MODES}

        def run(mode: str, copy_index: int) -> torch.Tensor:
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed[copy_index],
                topk_weights,
                topk_ids,
                plans[mode],
                global_num_experts=expert_count,
                out=outputs[mode],
            )

        reference = run(MODES[0], 0).clone()
        for mode in MODES:
            torch.testing.assert_close(run(mode, 0).float(), reference.float(), atol=0, rtol=0)
        order = random.Random(args.seed ^ (case_index << 8) ^ 0xA5A5)
        for round_index in range(args.warmup):
            names = list(MODES)
            order.shuffle(names)
            for position, mode in enumerate(names):
                run(mode, (round_index + position) % len(packed))

        trace_paths = {
            mode: args.trace_dir / case_name / f"{mode}.log"
            for mode in MODES
        }
        for trace_path in trace_paths.values():
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.unlink(missing_ok=True)
        os.environ["FUSED_CPP_MOE_TRACE"] = "1"
        trace_order = random.Random(args.seed ^ (case_index << 8) ^ 0x5A5A)
        for run_index in range(args.runs):
            names = list(MODES)
            trace_order.shuffle(names)
            for position, mode in enumerate(names):
                os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[mode])
                run(mode, (run_index + position) % len(packed))
        os.environ["FUSED_CPP_MOE_TRACE"] = "0"
        target_tasks = sum(map(len, target_routes))
        trace_calls = {
            mode: _parse_target_calls(path, target_tasks)
            for mode, path in trace_paths.items()
        }
        spans = {mode: _target_spans(calls) for mode, calls in trace_calls.items()}
        case_results[case_name] = {
            "target_routes": [list(values) for values in target_routes],
            "target_tasks": target_tasks,
            "bridges": bridges,
            "routes_by_expert": routes_by_expert,
            "trace_calls": trace_calls,
            "target_span": {mode: _stats(values) for mode, values in spans.items()},
            "measured_target_span_ms": {
                mode: statistics.median(values)
                for mode, values in spans.items()
            },
        }

    width2_expert_fixed_ns, width2_route_ns = _fit_width2_overhead(model, case_results)
    overhead_model = _model_with_narrow_calibration(
        model,
        width2_expert_fixed_ns=width2_expert_fixed_ns,
        width2_route_ns=width2_route_ns,
    )
    width1_correction, width2_correction = _fit_narrow_corrections(
        overhead_model,
        case_results,
    )
    fitted_model = _model_with_narrow_calibration(
        model,
        width2_expert_fixed_ns=width2_expert_fixed_ns,
        width2_route_ns=width2_route_ns,
        width1_correction=width1_correction,
        width2_correction=width2_correction,
    )
    serializable_cases = {}
    for case_name, case in case_results.items():
        base_predictions = {
            mode: _model_target_span_ms(
                model,
                case["bridges"][mode],
                case["routes_by_expert"][mode],
                int(case["target_tasks"]),
            )
            for mode in MODES
        }
        fitted_predictions = {
            mode: _model_target_span_ms(
                fitted_model,
                case["bridges"][mode],
                case["routes_by_expert"][mode],
                int(case["target_tasks"]),
            )
            for mode in MODES
        }
        overhead_predictions = {
            mode: _model_target_span_ms(
                overhead_model,
                case["bridges"][mode],
                case["routes_by_expert"][mode],
                int(case["target_tasks"]),
            )
            for mode in MODES
        }
        measured = case["measured_target_span_ms"]
        serializable_cases[case_name] = {
            "target_routes": case["target_routes"],
            "target_span": case["target_span"],
            "base_model_target_span_ms": base_predictions,
            "width2_overhead_model_target_span_ms": overhead_predictions,
            "fitted_model_target_span_ms": fitted_predictions,
            "measured_pair_over_merge": {
                "isolated": measured["pair_isolated"] / measured["merge_isolated"],
                "background": measured["pair_background"] / measured["merge_background"],
            },
            "base_model_pair_over_merge": {
                "isolated": base_predictions["pair_isolated"] / base_predictions["merge_isolated"],
                "background": base_predictions["pair_background"] / base_predictions["merge_background"],
            },
            "width2_overhead_model_pair_over_merge": {
                "isolated": overhead_predictions["pair_isolated"] / overhead_predictions["merge_isolated"],
                "background": overhead_predictions["pair_background"] / overhead_predictions["merge_background"],
            },
            "fitted_model_pair_over_merge": {
                "isolated": fitted_predictions["pair_isolated"] / fitted_predictions["merge_isolated"],
                "background": fitted_predictions["pair_background"] / fitted_predictions["merge_background"],
            },
        }

    result = {
        "kind": "narrow_lane_merge_transition",
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": expert_count,
            "threads": args.threads,
            "target_cores": [TARGET_CORE_BEGIN, TARGET_CORE_BEGIN + 1],
            "wide_background": [list(values) for values in WIDE_BACKGROUND_ROUTES],
            "narrow_background": [list(values) for values in NARROW_BACKGROUND_ROUTES],
        },
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "early_merge": False,
            "fit_cases": list(TARGET_ROUTE_CASES),
            "fit_observation": "merge_isolated per-task spans plus background/isolated dilation ratios",
            "holdouts": ["pair_isolated absolute span", "captured planner traces"],
        },
        "fitted_width2_overhead": {
            "expert_fixed_ns": width2_expert_fixed_ns,
            "route_ns": width2_route_ns,
        },
        "fitted_narrow_team_correction": {
            "1": width1_correction,
            "2": width2_correction,
        },
        "cases": serializable_cases,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_calibration is not None:
        _write_calibration(
            args.analytic_calibration,
            args.output_calibration,
            width2_expert_fixed_ns=width2_expert_fixed_ns,
            width2_route_ns=width2_route_ns,
            width1_correction=width1_correction,
            width2_correction=width2_correction,
            args=args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
