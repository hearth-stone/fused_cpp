#!/usr/bin/env python3
"""Calibrate 1T victim slowdown under concurrent 16T MoE peer gangs."""

from __future__ import annotations

import argparse
import json
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

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)


VICTIM_ROUTES = (35, 27, 22, 13, 10, 8, 6, 6, 5, 3, 2, 2, 2, 1, 1)
WIDE_PEER_ROUTES = ((1800, 600),) * 4
SMALL_PEER_ROUTES = ((68, 35, 13),) * 15
PEER_MODES = ("all_delayed", "wide_only", "all_concurrent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--victim-core", type=int, default=77)
    parser.add_argument("--peer-width", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_mixed_width_pressure"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _task_specs(
    peer_width: int,
    victim_core: int = 77,
) -> list[tuple[int, int, int, int]]:
    tasks = [(expert, routes, victim_core, 1) for expert, routes in enumerate(VICTIM_ROUTES)]
    expert = len(tasks)
    for lane, lane_routes in enumerate(WIDE_PEER_ROUTES):
        for routes in lane_routes:
            tasks.append((expert, routes, lane * peer_width, peer_width))
            expert += 1
    small_peer_cores = [core for core in range(64, 80) if core != victim_core]
    if len(small_peer_cores) != len(SMALL_PEER_ROUTES):
        raise ValueError("small peer routes must cover every non-victim 1T lane")
    for core, lane_routes in zip(small_peer_cores, SMALL_PEER_ROUTES, strict=True):
        for routes in lane_routes:
            tasks.append((expert, routes, core, 1))
            expert += 1
    return tasks


def _build_bridge(
    model: AnalyticMoeCostModel,
    *,
    thread_cpu_ids: tuple[int, ...],
    peer_width: int,
    victim_core: int,
    peer_mode: str,
) -> dict[str, object]:
    if peer_mode not in PEER_MODES:
        raise ValueError(f"peer_mode must be one of {PEER_MODES}, got {peer_mode!r}")
    tasks = _task_specs(peer_width, victim_core)
    victim_count = len(VICTIM_ROUTES)
    dependencies: list[list[int]] = []
    prior_victim = None
    peer_prior: dict[tuple[int, int], int] = {}
    for task_id, (_, _, core_begin, width) in enumerate(tasks):
        if width == 1 and core_begin == victim_core:
            deps = [] if prior_victim is None else [prior_victim]
            prior_victim = task_id
        else:
            lane = (core_begin, width)
            if lane in peer_prior:
                deps = [peer_prior[lane]]
            else:
                delay_root = peer_mode == "all_delayed" or (peer_mode == "wide_only" and width == 1)
                deps = [victim_count - 1] if delay_root else []
            peer_prior[lane] = task_id
        dependencies.append(deps)
    flat_dependencies = [value for task_dependencies in dependencies for value in task_dependencies]
    dependency_offsets = [0]
    for task_dependencies in dependencies:
        dependency_offsets.append(dependency_offsets[-1] + len(task_dependencies))
    policy = model.shadow_stage_window_policy()
    windows = [policy.select(routes, width) for _, routes, _, width in tasks]
    widths = [width for _, _, _, width in tasks]
    num_tasks = len(tasks)
    return {
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


def _placed_tasks(
    bridge: dict[str, object],
    routes: tuple[int, ...],
) -> list[tuple[int, int, tuple[int, ...], list[int]]]:
    cpu_ids = tuple(int(value) for value in bridge["thread_cpu_ids"])
    begins = [int(value) for value in bridge["task_core_begins"]]
    widths = [int(value) for value in bridge["task_threads"]]
    offsets = [int(value) for value in bridge["task_dep_offsets"]]
    flat_dependencies = [int(value) for value in bridge["task_deps"]]
    return [
        (
            route_count,
            width,
            cpu_ids[core_begin : core_begin + width],
            flat_dependencies[offsets[index] : offsets[index + 1]],
        )
        for index, (route_count, core_begin, width) in enumerate(zip(routes, begins, widths, strict=True))
    ]


def _parse_victim_calls(path: Path) -> list[dict[int, tuple[float, float]]]:
    calls: list[dict[int, tuple[float, float]]] = []
    current: dict[int, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
        if words[0] == "MOE_CALL":
            current = {}
        elif words[0] == "PHASE" and int(fields["tid"]) == 77:
            task = int(fields["group"])
            if not 0 <= task < len(VICTIM_ROUTES):
                continue
            start = float(fields["start_ms"])
            end = float(fields["end_ms"])
            previous = current.get(task)
            current[task] = (
                start if previous is None else min(previous[0], start),
                end if previous is None else max(previous[1], end),
            )
        elif words[0] == "MOE_CALL_END":
            if len(current) != len(VICTIM_ROUTES):
                raise RuntimeError(f"trace call has {len(current)} victim tasks, expected {len(VICTIM_ROUTES)}")
            calls.append(dict(current))
    if not calls:
        raise RuntimeError(f"no complete victim trace calls in {path}")
    return calls


def _parse_victim_spans(path: Path) -> list[float]:
    return [
        max(end for _, end in call.values()) - min(start for start, _ in call.values())
        for call in _parse_victim_calls(path)
    ]


def _fit_width_overhead(
    model: AnalyticMoeCostModel,
    calls: list[dict[int, tuple[float, float]]],
) -> tuple[float, float]:
    residuals = []
    for task, routes in enumerate(VICTIM_ROUTES):
        measured_ms = statistics.median(end - start for start, end in (call[task] for call in calls))
        residuals.append((float(routes), measured_ms * 1.0e6 - model.T_iso(routes, 1)))
    count = len(residuals)
    sum_x = sum(routes for routes, _ in residuals)
    sum_y = sum(residual for _, residual in residuals)
    sum_xx = sum(routes * routes for routes, _ in residuals)
    sum_xy = sum(routes * residual for routes, residual in residuals)
    denominator = count * sum_xx - sum_x * sum_x
    route_ns = max((count * sum_xy - sum_x * sum_y) / denominator, 0.0)
    expert_fixed_ns = max((sum_y - route_ns * sum_x) / count, 0.0)
    return expert_fixed_ns, route_ns


def _model_victim_span_ms(
    model: AnalyticMoeCostModel,
    bridge: dict[str, object],
    routes: tuple[int, ...],
) -> float:
    explanation = model.explain_dag_placed(_placed_tasks(bridge, routes))
    return (float(explanation["task_finish_ns"][len(VICTIM_ROUTES) - 1]) - model.call_setup_ns) / 1.0e6


def _model_with_1t_overhead(
    model: AnalyticMoeCostModel,
    expert_fixed_ns: float,
    route_ns: float,
) -> AnalyticMoeCostModel:
    calibration = replace(
        model.calibration,
        overheads=replace(
            model.calibration.overheads,
            by_width=(
                (1, expert_fixed_ns, route_ns),
                *(
                    point
                    for point in model.calibration.overheads.by_width
                    if point[0] != 1
                ),
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


def _stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p10_ms": ordered[round((len(ordered) - 1) * 0.10)],
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "samples_ms": values,
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads != 80 or args.victim_core != 77 or args.peer_width != 16:
        raise ValueError("the initial calibration design is fixed to 80T, victim core 77, and 16T peers")
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=(
            len(VICTIM_ROUTES)
            + sum(len(routes) for routes in WIDE_PEER_ROUTES)
            + sum(len(routes) for routes in SMALL_PEER_ROUTES)
        ),
        local_experts=(
            len(VICTIM_ROUTES)
            + sum(len(routes) for routes in WIDE_PEER_ROUTES)
            + sum(len(routes) for routes in SMALL_PEER_ROUTES)
        ),
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    bridges = {
        peer_mode: _build_bridge(
            model,
            thread_cpu_ids=cpu_ids,
            peer_width=args.peer_width,
            victim_core=args.victim_core,
            peer_mode=peer_mode,
        )
        for peer_mode in PEER_MODES
    }
    plans = {name: AsyncMoEPlanV2.from_dict(bridge) for name, bridge in bridges.items()}
    task_routes = (
        VICTIM_ROUTES
        + tuple(value for lane in WIDE_PEER_ROUTES for value in lane)
        + tuple(value for lane in SMALL_PEER_ROUTES for value in lane)
    )
    topk_ids = torch.repeat_interleave(
        torch.arange(len(task_routes), dtype=torch.int32),
        torch.tensor(task_routes, dtype=torch.int64),
    ).reshape(-1, 1)
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((topk_ids.shape[0], args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.ones((topk_ids.shape[0], 1), dtype=torch.float32)
    w13 = torch.empty(
        (len(task_routes), 2 * args.intermediate, args.hidden),
        dtype=torch.bfloat16,
    )
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty(
        (len(task_routes), args.hidden, args.intermediate),
        dtype=torch.bfloat16,
    )
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
    outputs = {name: torch.empty_like(hidden) for name in plans}

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"

    def run(name: str, copy_index: int) -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[copy_index],
            topk_weights,
            topk_ids,
            plans[name],
            global_num_experts=len(task_routes),
            out=outputs[name],
        )

    reference = run("all_delayed", 0).clone()
    for name in plans:
        torch.testing.assert_close(run(name, 0).float(), reference.float(), atol=0, rtol=0)
    names = list(plans)
    order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        order.shuffle(names)
        for position, name in enumerate(names):
            run(name, (round_index + position) % len(packed))

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = {name: args.trace_dir / f"{name}.log" for name in names}
    for trace_path in trace_paths.values():
        trace_path.unlink(missing_ok=True)
    os.environ["FUSED_CPP_MOE_TRACE"] = "1"
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        names = list(plans)
        trace_order.shuffle(names)
        for position, name in enumerate(names):
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[name])
            run(name, (run_index + position) % len(packed))
    trace_calls = {name: _parse_victim_calls(trace_path) for name, trace_path in trace_paths.items()}
    traces = {
        name: [max(end for _, end in call.values()) - min(start for start, _ in call.values()) for call in calls]
        for name, calls in trace_calls.items()
    }
    fitted_expert_fixed_ns, fitted_route_ns = _fit_width_overhead(
        model,
        trace_calls["all_delayed"],
    )
    fitted_model = _model_with_1t_overhead(
        model,
        fitted_expert_fixed_ns,
        fitted_route_ns,
    )
    measured_medians = {name: statistics.median(values) for name, values in traces.items()}
    base_model_spans = {name: _model_victim_span_ms(model, bridge, task_routes) for name, bridge in bridges.items()}
    fitted_model_spans = {
        name: _model_victim_span_ms(fitted_model, bridge, task_routes) for name, bridge in bridges.items()
    }
    result = {
        "kind": "mixed_width_victim_pressure",
        "identity": {
            "calibration": str(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
        },
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": len(task_routes),
            "tokens": int(topk_ids.shape[0]),
            "top_k": 1,
            "threads": args.threads,
            "victim_threads": 1,
            "victim_core": args.victim_core,
            "wide_peer_threads": args.peer_width,
            "wide_peer_teams": len(WIDE_PEER_ROUTES),
            "wide_peer_routes": [list(values) for values in WIDE_PEER_ROUTES],
            "small_peer_threads": 1,
            "small_peer_teams": len(SMALL_PEER_ROUTES),
            "small_peer_routes": [list(values) for values in SMALL_PEER_ROUTES],
            "victim_routes": list(VICTIM_ROUTES),
        },
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "early_merge": False,
            "control": "same tasks and placement; only 1T peer roots depend on victim tail",
        },
        "victim_span": {name: _stats(values) for name, values in traces.items()},
        "measured_ratios": {
            "wide_only_over_all_delayed": (measured_medians["wide_only"] / measured_medians["all_delayed"]),
            "all_concurrent_over_wide_only": (measured_medians["all_concurrent"] / measured_medians["wide_only"]),
            "all_concurrent_over_all_delayed": (measured_medians["all_concurrent"] / measured_medians["all_delayed"]),
        },
        "base_model_victim_span_ms": base_model_spans,
        "fitted_1t_overhead": {
            "expert_fixed_ns": fitted_expert_fixed_ns,
            "route_ns": fitted_route_ns,
        },
        "fitted_model_victim_span_ms": fitted_model_spans,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
