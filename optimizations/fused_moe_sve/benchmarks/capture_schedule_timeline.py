#!/usr/bin/env python3
"""Capture one production MoE plan as predicted and per-core actual timelines."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from bench_vllm_staged_schedule import (  # noqa: E402
    make_elastic_w2_plan,
    make_static_tail_repartition_plan,
    materialize_topk_ids,
)
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from phase_model import ContentionCostModel  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402
from schedule_timeline_metrics import (  # noqa: E402
    DEFAULT_GFLOPS_COLOR_MAX,
    enrich_actual_timeline,
)
from workload_catalog import default_offline_workloads  # noqa: E402


DEFAULT_PROFILE = (
    REPO_ROOT
    / "cpu_moe_schedule_optimization"
    / "cost_model"
    / "profiles"
    / "contention_async_amazon_c5_192c_numa0_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "optimizations"
    / "fused_moe_sve"
    / "results"
    / "amazon_192c_active8_schedule_timeline.json"
)
DEFAULT_TRACE = DEFAULT_OUTPUT.with_suffix(".trace")


def parse_args() -> argparse.Namespace:
    workloads = default_offline_workloads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--preset", choices=sorted(workloads), default="moe256-active-set-8")
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--route-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--gflops-color-max",
        type=float,
        default=DEFAULT_GFLOPS_COLOR_MAX,
        help="fixed per-core GEMM GFLOP/s represented by the darkest timeline color",
    )
    parser.add_argument(
        "--early-merge",
        choices=("auto", "off", "on"),
        default="auto",
        help="override Plan V2 early merge without changing the planned task graph",
    )
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--elastic-w2-transitions",
        default="",
        help="comma-separated selected:preferred W2 widths, for example 16:32",
    )
    parser.add_argument(
        "--elastic-task-ids",
        default="",
        help="comma-separated task ids eligible for W2 resize",
    )
    parser.add_argument(
        "--elastic-w2-core-begins",
        default="",
        help="comma-separated task_id:logical_core_begin W2 targets",
    )
    parser.add_argument("--elastic-timeout-us", type=float, default=0.0)
    parser.add_argument(
        "--static-tail-width",
        type=int,
        help="capture a 6x16T-to-2xWT static tail plan instead of the strict baseline",
    )
    parser.add_argument(
        "--strict-tail-steal",
        action="store_true",
        help="enable equal-width strict tail task stealing for the captured run",
    )
    parser.add_argument(
        "--compare-strict-tail-steal",
        action="store_true",
        help="interleave strict and strict-tail-steal timing on the same packed weights",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write the predicted timeline without allocating weights or running the native kernel",
    )
    return parser.parse_args()


def override_early_merge(plan: AsyncMoEPlanV2, policy: str) -> AsyncMoEPlanV2:
    if policy == "auto":
        return plan
    return replace(plan, early_merge=policy == "on")


def parse_int_mapping(value: str, *, description: str) -> dict[int, int]:
    if not value:
        return {}
    result: dict[int, int] = {}
    for item in value.split(","):
        try:
            key_text, value_text = item.split(":", maxsplit=1)
            key, parsed_value = int(key_text), int(value_text)
        except ValueError as error:
            raise ValueError(f"invalid {description} {item!r}; expected integer:integer") from error
        if key < 0 or parsed_value < 0:
            raise ValueError(f"{description} values must be non-negative, got {item!r}")
        result[key] = parsed_value
    return result


def parse_task_ids(value: str) -> list[int] | None:
    if not value:
        return None
    task_ids = [int(item) for item in value.split(",")]
    if any(task < 0 for task in task_ids):
        raise ValueError("--elastic-task-ids values must be non-negative")
    return task_ids


def task_dependencies(bridge: dict[str, Any], task: int) -> list[int]:
    offsets = bridge["task_dep_offsets"]
    dependencies = bridge["task_deps"]
    return [int(value) for value in dependencies[offsets[task] : offsets[task + 1]]]


def materialized_plan_tasks(plan: AsyncMoEPlanV2, histogram: list[int] | tuple[int, ...]) -> list[dict[str, Any]]:
    offsets = plan.task_dep_offsets.tolist()
    dependencies = plan.task_deps.tolist()
    experts = plan.task_expert_ids.tolist()
    core_begins = plan.task_core_begins.tolist()
    threads = plan.task_threads.tolist()
    placement_modes = plan.task_placement_modes.tolist()
    range_granularities = plan.task_range_granularities.tolist()
    w13_windows = plan.task_w13_window_bytes.tolist()
    w2_windows = plan.task_w2_window_bytes.tolist()
    covered_rows = [0] * len(histogram)
    tasks: list[dict[str, Any]] = []
    for task, expert in enumerate(experts):
        total_routes = int(histogram[expert])
        granularity = int(range_granularities[task])
        if granularity == 0:
            if covered_rows[expert] != 0:
                raise ValueError(f"expert {expert} mixes full-range and sliced timeline tasks")
            route_begin = 0
            routes = total_routes
        else:
            route_begin = covered_rows[expert]
            routes = min(granularity, total_routes - route_begin)
            if routes <= 0:
                raise ValueError(f"expert {expert} has a route slice beyond its histogram")
        covered_rows[expert] += routes
        tasks.append(
            {
                "task": task,
                "expert": int(expert),
                "route_begin": route_begin,
                "route_end": route_begin + routes,
                "routes": routes,
                "range_granularity": granularity,
                "core_begin": int(core_begins[task]),
                "threads": int(threads[task]),
                "placement_mode": int(placement_modes[task]),
                "dependencies": [
                    int(value) for value in dependencies[offsets[task] : offsets[task + 1]]
                ],
                "w13_window_bytes": int(w13_windows[task]),
                "w2_window_bytes": int(w2_windows[task]),
            }
        )
    for expert, total_routes in enumerate(histogram):
        if int(total_routes) > 0 and covered_rows[expert] != int(total_routes):
            raise ValueError(
                f"timeline tasks cover {covered_rows[expert]} of {int(total_routes)} routes for expert {expert}"
            )
    return tasks


def phase_descriptions(model: ContentionCostModel, routes: int, threads: int) -> list[dict[str, Any]]:
    raw_phases = model._task_phases(routes, threads)
    w13_ranges, _, w2_ranges, _ = model._task_stage_geometry(routes, threads)
    descriptions: list[dict[str, Any]] = []
    raw_index = 0
    if raw_phases and raw_phases[0][1] == 0:
        duration, workset = raw_phases[0]
        descriptions.append(
            {
                "duration_ns": float(duration),
                "workset_bytes": int(workset),
                "stage": "task_overhead",
                "range": 0,
            }
        )
        raw_index = 1
    for range_index in range(max(int(w13_ranges), 1)):
        duration, workset = raw_phases[raw_index]
        descriptions.append(
            {
                "duration_ns": float(duration),
                "workset_bytes": int(workset),
                "stage": "w13",
                "range": range_index,
            }
        )
        raw_index += 1
    for range_index in range(max(int(w2_ranges), 1)):
        duration, workset = raw_phases[raw_index]
        descriptions.append(
            {
                "duration_ns": float(duration),
                "workset_bytes": int(workset),
                "stage": "w2",
                "range": range_index,
            }
        )
        raw_index += 1
    if raw_index != len(raw_phases):
        raise RuntimeError("cost-model phase layout changed; update the timeline decoder")
    return descriptions


def append_segment(segments: list[dict[str, Any]], segment: dict[str, Any]) -> None:
    if (
        segments
        and segments[-1]["task"] == segment["task"]
        and segments[-1]["stage"] == segment["stage"]
        and segments[-1]["range"] == segment["range"]
        and abs(segments[-1]["end_ms"] - segment["start_ms"]) < 1.0e-9
        and abs(segments[-1]["slowdown"] - segment["slowdown"]) < 1.0e-9
    ):
        segments[-1]["end_ms"] = segment["end_ms"]
        return
    segments.append(segment)


def predict_timeline(
    model: ContentionCostModel,
    task_rows_threads_deps: list[tuple[int, int, list[int]]],
) -> tuple[float, list[list[dict[str, Any]]]]:
    count = len(task_rows_threads_deps)
    routes = [int(value) for value, _, _ in task_rows_threads_deps]
    threads = [int(value) for _, value, _ in task_rows_threads_deps]
    phases = [phase_descriptions(model, routes[index], threads[index]) for index in range(count)]
    phase_index = [0] * count
    remaining = [task_phases[0]["duration_ns"] for task_phases in phases]
    dependency_count, successors, started = model._dag_state(task_rows_threads_deps)
    finished = [False] * count
    segments: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    wall = float(model.call_setup_ns)
    guard = 0
    max_events = sum(len(task_phases) for task_phases in phases) + count + 2

    while not all(finished):
        guard += 1
        if guard > 2 * max_events:
            raise RuntimeError("timeline simulation did not converge")
        active = [index for index in range(count) if started[index] and not finished[index]]
        if not active:
            raise ValueError("DAG deadlock while producing timeline")
        worksets = [0] * count
        for index in active:
            worksets[index] = int(phases[index][phase_index[index]]["workset_bytes"])
        slowdown = float(model._working_set_derate(active, routes, threads, worksets))
        effective = {index: (1.0 if worksets[index] == 0 else slowdown) for index in active}
        elapsed = min(remaining[index] * effective[index] for index in active)
        next_wall = wall + elapsed
        for index in active:
            phase = phases[index][phase_index[index]]
            append_segment(
                segments[index],
                {
                    "task": index,
                    "stage": phase["stage"],
                    "range": int(phase["range"]),
                    "start_ms": wall / 1.0e6,
                    "end_ms": next_wall / 1.0e6,
                    "workset_bytes": int(phase["workset_bytes"]),
                    "slowdown": effective[index],
                },
            )
            remaining[index] -= elapsed / effective[index]
        wall = next_wall
        completed_phases = [index for index in active if remaining[index] <= 1.0e-6]
        for index in completed_phases:
            phase_index[index] += 1
            if phase_index[index] < len(phases[index]):
                remaining[index] = float(phases[index][phase_index[index]]["duration_ns"])
                continue
            finished[index] = True
            for successor in successors[index]:
                dependency_count[successor] -= 1
                if dependency_count[successor] == 0:
                    started[successor] = True

    expected = float(model.dag_makespan(task_rows_threads_deps))
    if abs(wall - expected) > max(1.0, expected * 1.0e-9):
        raise RuntimeError(f"timeline/model makespan mismatch: {wall} vs {expected}")
    return wall / 1.0e6, segments


def parse_fields(line: str) -> tuple[str, dict[str, str]]:
    words = line.strip().split()
    return words[0], dict(word.split("=", maxsplit=1) for word in words[1:] if "=" in word)


def parse_last_trace_call(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    calls: list[tuple[dict[str, str], list[dict[str, Any]], bool]] = []
    current_header: dict[str, str] | None = None
    current_phases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        kind, fields = parse_fields(line)
        if kind == "MOE_CALL":
            current_header = fields
            current_phases = []
        elif kind == "PHASE" and current_header is not None:
            current_phases.append(
                {
                    "seq": int(fields["seq"]),
                    "tid": int(fields["tid"]),
                    "cpu": int(fields["cpu"]),
                    "task": int(fields["group"]),
                    "local_tid": int(fields["local_tid"]),
                    "expert": int(fields["expert"]),
                    "rows": int(fields["rows"]),
                    "stage": fields["stage"],
                    "start_ms": float(fields["start_ms"]),
                    "end_ms": float(fields["end_ms"]),
                }
            )
        elif kind == "MOE_CALL_END" and current_header is not None:
            calls.append((current_header, current_phases, True))
            current_header = None
            current_phases = []
    if not calls:
        raise RuntimeError(f"no complete MOE_CALL in {path}")
    header, phases, _ = calls[-1]
    return header, phases


def compact_actual_segments(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for phase in sorted(phases, key=lambda item: (item["start_ms"], item["end_ms"], item["seq"])):
        reduced = {key: value for key, value in phase.items() if key != "seq"}
        if (
            compacted
            and reduced["stage"] == compacted[-1]["stage"]
            and reduced["task"] == compacted[-1]["task"]
            and reduced["expert"] == compacted[-1]["expert"]
            and reduced["start_ms"] - compacted[-1]["end_ms"] <= 0.005
        ):
            compacted[-1]["end_ms"] = max(compacted[-1]["end_ms"], reduced["end_ms"])
            compacted[-1]["rows"] += reduced["rows"]
            continue
        compacted.append(reduced)
    return compacted


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], ContentionCostModel, Any]:
    workloads = default_offline_workloads()
    workload = workloads[args.preset]
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if not args.profile.is_file():
        raise ValueError(f"profile does not exist: {args.profile}")
    cpu_ids = list(range(args.threads)) if args.plan_only else sorted(os.sched_getaffinity(0))[: args.threads]
    if len(cpu_ids) != args.threads:
        raise ValueError(f"affinity exposes {len(cpu_ids)} CPUs, expected {args.threads}")

    model = ContentionCostModel(args.profile)
    planner = PlannedMoE(model, args.threads, cpu_ids=cpu_ids)
    spec = planner.plan_spec_for(workload.experts)
    bridge = spec["bridge"]
    plan = override_early_merge(AsyncMoEPlanV2.from_dict(bridge), args.early_merge)
    bound_model = planner.interval_planners[0].model
    tasks = materialized_plan_tasks(plan, workload.histogram)
    predicted_cores: dict[str, list[dict[str, Any]]] = {str(core): [] for core in range(args.threads)}
    if spec["execution_mode"] == "strict":
        model_tasks = [
            (int(task["routes"]), int(task["threads"]), list(task["dependencies"]))
            for task in tasks
        ]
        predicted_ms, task_segments = predict_timeline(bound_model, model_tasks)
        planner_makespan_ms = predicted_ms
        uniform_routes = {
            int(routes)
            for routes in workload.histogram
            if int(routes) > 0
        }
        tail_width = spec["tail_repartition_width"]
        route_slices = int(spec["tail_repartition_route_slices"])
        if (
            tail_width is not None
            and len(uniform_routes) == 1
            and bound_model.can_use_bounded_tail_repartition_anchor(
                next(iter(uniform_routes)),
                spec["shape"],
                int(tail_width),
                route_slices,
            )
        ):
            planner_makespan_ns, _ = bound_model.profiled_bounded_tail_repartition(
                next(iter(uniform_routes)),
                spec["shape"],
                int(tail_width),
                route_slices,
            )
            planner_makespan_ms = float(planner_makespan_ns) / 1.0e6
        for task, segments in zip(tasks, task_segments, strict=True):
            for core in range(task["core_begin"], task["core_begin"] + task["threads"]):
                predicted_cores[str(core)].extend(
                    {
                        **segment,
                        "expert": task["expert"],
                        "route_begin": task["route_begin"],
                        "route_end": task["route_end"],
                        "routes": task["routes"],
                    }
                    for segment in segments
                )
        predicted_scope = "phase-model decomposition; planner score may use an exact full-call anchor"
        predicted_per_core = True
    else:
        interval_planner = planner.interval_planners[0]
        lanes = interval_planner._lanes(tuple(spec["shape"]))
        raw_tasks = interval_planner._build_tasks(
            workload.experts,
            lanes,
            interval_planner._assign(workload.experts, lanes),
        )
        simulation_tasks, _, _ = interval_planner._tail_pool_simulation(
            raw_tasks,
            pool_threads=int(spec["tail_pool_threads"]),
            max_pooled_routes=int(spec["tail_pool_max_routes"]),
        )
        predicted_ms = float(bound_model.dag_makespan(simulation_tasks)) / 1.0e6
        planner_makespan_ms = predicted_ms
        predicted_scope = "aggregate tail-pool simulation; dynamic per-core assignment is runtime-only"
        predicted_per_core = False

    payload: dict[str, Any] = {
        "schema_version": 2,
        "case": {
            "machine": "AmazonC5192Cores",
            "numa_node": 0,
            "cpu_ids": cpu_ids,
            "preset": workload.name,
            "tokens": workload.tokens,
            "top_k": workload.top_k,
            "experts": workload.num_experts,
            "active_experts": workload.observed_active_experts,
            "hidden_size": int(model.policy.hidden_size),
            "intermediate_size": int(model.policy.intermediate_size),
            "backend_n_tile": int(model.policy.backend_n_tile),
            "route_dtype": args.route_dtype,
            "profile": str(args.profile),
            "capture": {
                "seed": args.seed,
                "warmup": args.warmup,
                "runs": args.runs,
                "gflops_color_max": args.gflops_color_max,
            },
        },
        "plan": {
            "execution_mode": spec["execution_mode"],
            "shape": list(spec["shape"]),
            "task_stage_window_policy": spec["task_stage_window_policy"],
            "w13_split": bool(spec["w13_split"]),
            "weight_window_bytes": int(spec["weight_window_bytes"]),
            "tail_repartition_width": spec["tail_repartition_width"],
            "tail_repartition_tasks": int(spec["tail_repartition_tasks"]),
            "tail_repartition_route_slices": int(spec["tail_repartition_route_slices"]),
            "tail_pool_threads": spec["tail_pool_threads"],
            "tail_pool_max_routes": spec["tail_pool_max_routes"],
            "tail_pool_tasks": int(spec["tail_pool_tasks"]),
            "early_merge": plan.early_merge,
            "strict_tail_steal": bool(args.strict_tail_steal),
            "tasks": tasks,
        },
        "predicted": {
            "scope": predicted_scope,
            "per_core_available": predicted_per_core,
            "makespan_ms": predicted_ms,
            "planner_makespan_ms": planner_makespan_ms,
            "host_segments": (
                [
                    {
                        "stage": "call_setup",
                        "start_ms": 0.0,
                        "end_ms": float(bound_model.call_setup_ns) / 1.0e6,
                    }
                ]
                if bound_model.call_setup_ns > 0
                else []
            ),
            "cores": predicted_cores,
        },
        "actual": None,
    }
    return payload, bound_model, workload


@torch.inference_mode()
def capture_actual(
    args: argparse.Namespace,
    payload: dict[str, Any],
    workload: Any,
) -> dict[str, Any]:
    plan_payload = PlannedMoE(
        ContentionCostModel(args.profile),
        args.threads,
        cpu_ids=payload["case"]["cpu_ids"],
    ).plan_spec_for(workload.experts)["bridge"]
    plan = AsyncMoEPlanV2.from_dict(plan_payload)
    transitions = parse_int_mapping(
        args.elastic_w2_transitions,
        description="elastic W2 transition",
    )
    if transitions and args.static_tail_width is not None:
        raise ValueError("--elastic-w2-transitions and --static-tail-width are mutually exclusive")
    if transitions:
        task_ids = parse_task_ids(args.elastic_task_ids)
        target_core_begins = parse_int_mapping(
            args.elastic_w2_core_begins,
            description="elastic W2 target",
        )
        plan = make_elastic_w2_plan(
            plan,
            torch.tensor(workload.histogram, dtype=torch.int64),
            profile=args.profile,
            transitions=transitions,
            timeout_ns=round(args.elastic_timeout_us * 1.0e3),
            static_stage_windows=False,
            resizable_task_ids=task_ids,
            task_preferred_core_begins=target_core_begins or None,
        )
        payload["plan"]["execution_mode"] = plan.execution_mode
        payload["plan"]["elastic_w2"] = {
            "transitions": {str(key): value for key, value in transitions.items()},
            "task_ids": task_ids,
            "target_core_begins": {
                str(key): value for key, value in target_core_begins.items()
            },
            "timeout_us": args.elastic_timeout_us,
        }
        preferred_threads = plan.task_preferred_threads.tolist()
        preferred_core_begins = plan.task_preferred_core_begins.tolist()
        resize_points = plan.task_resize_points.tolist()
        for task in payload["plan"]["tasks"]:
            task_id = task["task"]
            task["preferred_threads"] = int(preferred_threads[task_id])
            task["preferred_core_begin"] = int(preferred_core_begins[task_id])
            task["resize_point"] = int(resize_points[task_id])
    elif args.static_tail_width is not None:
        plan = make_static_tail_repartition_plan(
            plan,
            torch.tensor(workload.histogram, dtype=torch.int64),
            profile=args.profile,
            tail_width=args.static_tail_width,
            static_stage_windows=payload["plan"]["task_stage_window_policy"] is not None,
        )
        payload["plan"]["tasks"] = materialized_plan_tasks(plan, workload.histogram)
        payload["plan"]["static_tail_width"] = args.static_tail_width
        payload["plan"]["shape"] = [16, 16, 16, 16, 16, 16, args.static_tail_width, args.static_tail_width]
    plan = override_early_merge(plan, args.early_merge)
    payload["plan"]["early_merge"] = plan.early_merge
    bridge = payload["plan"]["tasks"]
    policy = ContentionCostModel(args.profile).policy
    assert policy is not None

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((workload.tokens, policy.hidden_size), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty(
        (workload.num_experts, 2 * policy.intermediate_size, policy.hidden_size),
        dtype=torch.bfloat16,
    )
    w2 = torch.empty(
        (workload.num_experts, policy.hidden_size, policy.intermediate_size),
        dtype=torch.bfloat16,
    )
    topk_ids = materialize_topk_ids(
        workload.histogram,
        tokens=workload.tokens,
        top_k=workload.top_k,
        seed=args.seed,
    )
    topk_weights = torch.softmax(
        torch.randn((workload.tokens, workload.top_k), generator=generator),
        dim=-1,
    )

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if args.route_dtype == "bf16" else "0"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS"] = "0"
    os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"
    os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(args.trace_file)
    os.environ["FUSED_CPP_MOE_TRACE"] = "0"
    os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "1" if args.strict_tail_steal else "0"

    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="arm_sve_bf16",
    )
    del w13, w2
    output = torch.empty_like(hidden)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            plan,
            global_num_experts=workload.num_experts,
            w13_split=bool(payload["plan"]["w13_split"]),
            weight_window_bytes=int(payload["plan"]["weight_window_bytes"]),
            out=output,
        )

    comparison: dict[str, Any] | None = None
    if args.compare_strict_tail_steal:
        comparison_samples = {"strict": [], "strict_tail_steal": []}
        for iteration in range(args.warmup):
            for enabled in ((False, True) if iteration % 2 == 0 else (True, False)):
                os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "1" if enabled else "0"
                run()
        for iteration in range(args.runs):
            for enabled in ((False, True) if iteration % 2 == 0 else (True, False)):
                os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "1" if enabled else "0"
                begin = time.perf_counter_ns()
                run()
                name = "strict_tail_steal" if enabled else "strict"
                comparison_samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
        comparison = {
            name: {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
            }
            for name, samples in comparison_samples.items()
        }
        strict_median = comparison["strict"]["median_ms"]
        steal_median = comparison["strict_tail_steal"]["median_ms"]
        comparison["speedup_pct"] = 100.0 * (strict_median / steal_median - 1.0)
        selected_name = "strict_tail_steal" if args.strict_tail_steal else "strict"
        untraced_samples_ms = comparison_samples[selected_name]
    else:
        for _ in range(args.warmup):
            run()
        untraced_samples_ms = []
        for _ in range(args.runs):
            begin = time.perf_counter_ns()
            run()
            untraced_samples_ms.append((time.perf_counter_ns() - begin) / 1.0e6)

    args.trace_file.parent.mkdir(parents=True, exist_ok=True)
    args.trace_file.unlink(missing_ok=True)
    os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "1" if args.strict_tail_steal else "0"
    os.environ["FUSED_CPP_MOE_TRACE"] = "1"
    run()
    os.environ["FUSED_CPP_MOE_TRACE"] = "0"
    header, phases = parse_last_trace_call(args.trace_file)

    core_phases: dict[str, list[dict[str, Any]]] = {}
    host_phases: list[dict[str, Any]] = []
    expected_cpus = payload["case"]["cpu_ids"]
    for phase in phases:
        if phase["tid"] < 0:
            host_phases.append(phase)
            continue
        if phase["tid"] >= args.threads:
            raise RuntimeError(f"trace has out-of-range logical tid {phase['tid']}")
        if phase["cpu"] != expected_cpus[phase["tid"]]:
            raise RuntimeError(
                f"logical tid {phase['tid']} ran on CPU {phase['cpu']}, expected {expected_cpus[phase['tid']]}"
            )
        core_phases.setdefault(str(phase["tid"]), []).append(phase)
    for core in range(args.threads):
        core_phases[str(core)] = compact_actual_segments(core_phases.get(str(core), []))

    task_by_id = {task["task"]: task for task in bridge}
    for segments in core_phases.values():
        for segment in segments:
            task = task_by_id.get(segment["task"])
            if task is not None:
                segment["planned_core_begin"] = task["core_begin"]
                segment["planned_threads"] = task["threads"]

    traced_e2e_ms = float(header["e2e_ms"])
    scheduled = [phase for phase in host_phases if phase["stage"] == "scheduled_compute"]
    return {
        "untraced_samples_ms": untraced_samples_ms,
        "untraced_median_ms": statistics.median(untraced_samples_ms),
        "traced_e2e_ms": traced_e2e_ms,
        "trace_overhead_pct": 100.0 * (traced_e2e_ms / statistics.median(untraced_samples_ms) - 1.0),
        "scheduled_compute_ms": (
            scheduled[-1]["end_ms"] - scheduled[-1]["start_ms"] if scheduled else None
        ),
        "host_segments": compact_actual_segments(host_phases),
        "cores": core_phases,
        "trace_file": str(args.trace_file),
        "early_merge": plan.early_merge,
        "strict_tail_steal": args.strict_tail_steal,
        "strict_tail_steal_comparison": comparison,
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("--warmup must be non-negative and --runs must be positive")
    if args.gflops_color_max <= 0:
        raise ValueError("--gflops-color-max must be positive")
    payload, _, workload = build_plan(args)
    if not args.plan_only:
        payload["actual"] = capture_actual(args, payload, workload)
        enrich_actual_timeline(payload, gflops_color_max=args.gflops_color_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    actual = payload["actual"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shape": payload["plan"]["shape"],
                "tasks": len(payload["plan"]["tasks"]),
                "predicted_ms": payload["predicted"]["makespan_ms"],
                "untraced_median_ms": actual["untraced_median_ms"] if actual is not None else None,
                "traced_e2e_ms": actual["traced_e2e_ms"] if actual is not None else None,
                "scheduled_compute_ms": actual["scheduled_compute_ms"] if actual is not None else None,
                "internal_idle_core_ms": (
                    actual["idle_metrics"]["internal_idle_core_ms"] if actual is not None else None
                ),
                "tail_idle_core_ms": (actual["idle_metrics"]["tail_idle_core_ms"] if actual is not None else None),
                "strict_tail_steal_comparison": (
                    actual["strict_tail_steal_comparison"] if actual is not None else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
