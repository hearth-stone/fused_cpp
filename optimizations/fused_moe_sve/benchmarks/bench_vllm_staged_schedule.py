#!/usr/bin/env python3
"""Compare production, independently planned stages, and vLLM-style MoE schedules."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    decode_async_moe_elastic_stats,
    fused_moe_bf16_tiled_async,
    fused_moe_bf16_tiled_async_plan,
    fused_moe_bf16_tiled_planned_staged,
    fused_moe_bf16_tiled_vllm_staged,
    make_async_moe_elastic_stats,
    prepare_fused_moe_bf16_tiled_weights,
)
from stage_window_policy import (  # noqa: E402
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1,
)
from workload_catalog import default_offline_workloads  # noqa: E402


PAPER_WORKLOADS = default_offline_workloads()
AUTO_VARIANT = "production_auto"
STATIC_STAGE_WINDOW_VARIANT = "production_auto_stage_windows"
DYNAMIC_POOL_VARIANT = "dynamic_16t_to_4t_pool"
STATIC_SPLIT_VARIANT = "static_16t_to_4x4t"
VLLM_VARIANT = "vllm_staged"
MATCHED_STAGED_VARIANT = "planned_staged_matched"
FINE_STAGED_VARIANT = "planned_staged_independent"
ELASTIC_VARIANT_PREFIX = "elastic_w2"
STATIC_TAIL_VARIANT_PREFIX = "static_tail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--team-threads", type=int, default=0)
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--distribution", choices=("hot-topk", "round-robin"))
    routing.add_argument("--preset", choices=sorted(PAPER_WORKLOADS))
    parser.add_argument(
        "--production-profile",
        type=Path,
        help="schema-v2 contention profile; enables the production PlannedMoE schedule",
    )
    parser.add_argument(
        "--production-ready-token-merge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable production ready-token merge (default: enabled with --production-profile)",
    )
    parser.add_argument(
        "--ready-token-drain-batches",
        help=(
            "comma-separated same-job ready-token batch sizes; adds an interleaved "
            "legacy-final comparator and requires a production auto plan"
        ),
    )
    parser.add_argument(
        "--static-16-to-4",
        action="store_true",
        help="benchmark a static DAG that releases each 16T long team as four 4T short-team lanes",
    )
    parser.add_argument(
        "--dynamic-short-pool",
        action="store_true",
        help="add a forced-4T tail-pool comparator alongside the default planner-selected variant",
    )
    parser.add_argument(
        "--static-stage-windows",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "compare the default per-task W13/W2 window policy with an explicitly "
            "disabled production-auto plan (default: enabled with --production-profile)"
        ),
    )
    parser.add_argument(
        "--static-long-route-threshold",
        type=int,
        default=12,
        help="experts above this route count use the 16T head of the static split DAG",
    )
    parser.add_argument(
        "--elastic-w2-transitions",
        help="comma-separated planner W2 width transitions, for example 8:16,2:8",
    )
    parser.add_argument(
        "--elastic-timeout-us",
        type=float,
        action="append",
        help="elastic cohort timeout in microseconds; repeat to compare values (default: 0)",
    )
    parser.add_argument(
        "--elastic-task-ids",
        help="comma-separated task ids eligible for W2 resizing; default: every task matching a transition",
    )
    parser.add_argument(
        "--elastic-w2-core-begins",
        help="comma-separated task_id:logical_core_begin targets; allows disjoint W2 migration",
    )
    parser.add_argument(
        "--static-tail-widths",
        help=(
            "comma-separated widths for a 6x16T-to-2xWT static tail experiment; "
            "requires an eight-expert strict production plan"
        ),
    )
    parser.add_argument(
        "--static-tail-m-split",
        action="store_true",
        help=(
            "compare grouped and interleaved 6x16T-to-4x24T tails by splitting "
            "each of the two terminal experts into two equal route ranges"
        ),
    )
    parser.add_argument("--route-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def parse_width_transitions(value: str) -> dict[int, int]:
    transitions: dict[int, int] = {}
    for item in value.split(","):
        try:
            selected_text, preferred_text = item.split(":", maxsplit=1)
            selected = int(selected_text)
            preferred = int(preferred_text)
        except ValueError as error:
            raise ValueError(f"invalid elastic W2 transition {item!r}; expected selected:preferred") from error
        if selected <= 0 or preferred <= selected or preferred % selected != 0:
            raise ValueError(
                f"elastic W2 transition must expand to a larger multiple, got {selected}:{preferred}"
            )
        transitions[selected] = preferred
    if not transitions:
        raise ValueError("at least one elastic W2 transition is required")
    return transitions


def parse_integer_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_task_core_begins(value: str) -> dict[int, int]:
    targets: dict[int, int] = {}
    for item in value.split(","):
        try:
            task_text, core_text = item.split(":", maxsplit=1)
            task = int(task_text)
            core_begin = int(core_text)
        except ValueError as error:
            raise ValueError(
                f"invalid elastic W2 target {item!r}; expected task_id:logical_core_begin"
            ) from error
        if task < 0 or core_begin < 0:
            raise ValueError(f"elastic W2 target values must be non-negative, got {item!r}")
        targets[task] = core_begin
    if not targets:
        raise ValueError("at least one elastic W2 target is required")
    return targets


def cpu_numa_node(cpu: int) -> int:
    nodes = sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    if len(nodes) != 1:
        raise RuntimeError(f"cannot determine a unique NUMA node for CPU {cpu}")
    return int(nodes[0].name.removeprefix("node"))


def make_elastic_w2_plan(
    base_plan: AsyncMoEPlanV2,
    route_counts: torch.Tensor,
    *,
    profile: Path,
    transitions: dict[int, int],
    timeout_ns: int,
    static_stage_windows: bool,
    resizable_task_ids: list[int] | None = None,
    task_preferred_core_begins: dict[int, int] | None = None,
) -> AsyncMoEPlanV2:
    from interval_planner import IntervalPlanner
    from phase_model import ContentionCostModel

    dependency_offsets = base_plan.task_dep_offsets.tolist()
    dependencies = base_plan.task_deps.tolist()
    tasks = []
    for task, expert in enumerate(base_plan.task_expert_ids.tolist()):
        begin = dependency_offsets[task]
        end = dependency_offsets[task + 1]
        tasks.append(
            (
                int(expert),
                int(route_counts[expert]),
                int(base_plan.task_core_begins[task]),
                int(base_plan.task_threads[task]),
                [int(dep) for dep in dependencies[begin:end]],
            )
        )
    cpu_ids = [int(cpu) for cpu in base_plan.thread_cpu_ids.tolist()]
    numa_nodes = {cpu_numa_node(cpu) for cpu in cpu_ids}
    if len(numa_nodes) != 1:
        raise ValueError(f"elastic W2 benchmark cannot regroup across NUMA nodes: {sorted(numa_nodes)}")
    planner = IntervalPlanner(
        ContentionCostModel(profile),
        base_plan.num_threads,
        cpu_ids=cpu_ids,
        task_stage_window_policy=(
            AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1 if static_stage_windows else None
        ),
    )
    bridge = planner.to_elastic_w2_bridge(
        tasks,
        width_transitions=transitions,
        numa_node=next(iter(numa_nodes)),
        resize_timeout_ns=timeout_ns,
        resizable_task_ids=resizable_task_ids,
        task_preferred_core_begins=task_preferred_core_begins,
    )
    if not any(int(point) != 0 for point in bridge["task_resize_points"]):
        raise ValueError(
            f"selected strict plan has no task matching elastic transitions {sorted(transitions.items())}"
        )
    return AsyncMoEPlanV2.from_dict(bridge)


def make_static_tail_repartition_plan(
    base_plan: AsyncMoEPlanV2,
    route_counts: torch.Tensor,
    *,
    profile: Path,
    tail_width: int,
    static_stage_windows: bool,
) -> AsyncMoEPlanV2:
    from interval_planner import IntervalPlanner
    from phase_model import ContentionCostModel

    if base_plan.execution_mode != "strict":
        raise ValueError("static tail repartition requires a strict base plan")
    if tail_width <= 0 or base_plan.num_threads % tail_width != 0:
        raise ValueError(
            f"tail width must be a positive divisor of num_threads={base_plan.num_threads}, got {tail_width}"
        )

    dependency_offsets = base_plan.task_dep_offsets.tolist()
    dependencies = base_plan.task_deps.tolist()
    base_tasks: list[tuple[int, int, int, int, list[int]]] = []
    for task, expert in enumerate(base_plan.task_expert_ids.tolist()):
        begin = dependency_offsets[task]
        end = dependency_offsets[task + 1]
        base_tasks.append(
            (
                int(expert),
                int(route_counts[expert]),
                int(base_plan.task_core_begins[task]),
                int(base_plan.task_threads[task]),
                [int(dep) for dep in dependencies[begin:end]],
            )
        )

    roots = sorted((task for task in base_tasks if not task[4]), key=lambda task: task[2])
    tails = sorted((task for task in base_tasks if task[4]), key=lambda task: task[2])
    if len(roots) != 6 or len(tails) != 2:
        raise ValueError(
            "static tail repartition requires exactly six first-wave and two tail tasks, "
            f"got {len(roots)} and {len(tails)}"
        )
    if any(task[3] != 16 for task in roots):
        raise ValueError("static tail repartition requires six 16-thread first-wave tasks")
    if 2 * tail_width > base_plan.num_threads:
        raise ValueError(f"two {tail_width}-thread tail tasks exceed {base_plan.num_threads} threads")

    half_threads = base_plan.num_threads // len(tails)
    tail_core_begins = [index * half_threads for index in range(len(tails))]
    if tail_core_begins[-1] + tail_width > base_plan.num_threads:
        raise ValueError(f"cannot place two {tail_width}-thread tail tasks")

    tasks: list[tuple[int, int, int, int, list[int]]] = [
        (expert, routes, core_begin, threads, [])
        for expert, routes, core_begin, threads, _ in roots
    ]
    for tail, core_begin in zip(tails, tail_core_begins, strict=True):
        core_end = core_begin + tail_width
        blockers = [
            root_id
            for root_id, (_, _, root_begin, root_threads, _) in enumerate(roots)
            if root_begin < core_end and core_begin < root_begin + root_threads
        ]
        if not blockers:
            raise ValueError(f"tail interval [{core_begin}, {core_end}) has no first-wave blockers")
        tasks.append((tail[0], tail[1], core_begin, tail_width, blockers))

    cpu_ids = [int(cpu) for cpu in base_plan.thread_cpu_ids.tolist()]
    planner = IntervalPlanner(
        ContentionCostModel(profile),
        base_plan.num_threads,
        cpu_ids=cpu_ids,
        task_stage_window_policy=(
            AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1 if static_stage_windows else None
        ),
    )
    return AsyncMoEPlanV2.from_dict(planner.to_async_bridge(tasks))


def make_static_tail_m_split_plan(
    base_plan: AsyncMoEPlanV2,
    route_counts: torch.Tensor,
    *,
    profile: Path,
    layout: str,
    static_stage_windows: bool,
) -> AsyncMoEPlanV2:
    from interval_planner import IntervalPlanner
    from phase_model import ContentionCostModel

    if layout not in {"grouped", "interleaved"}:
        raise ValueError(f"unsupported static tail M-split layout: {layout}")
    if base_plan.execution_mode != "strict" or base_plan.num_threads != 96:
        raise ValueError("static tail M-split requires a strict 96-thread base plan")

    dependency_offsets = base_plan.task_dep_offsets.tolist()
    dependencies = base_plan.task_deps.tolist()
    base_tasks: list[tuple[int, int, int, int, list[int]]] = []
    for task, expert in enumerate(base_plan.task_expert_ids.tolist()):
        begin = dependency_offsets[task]
        end = dependency_offsets[task + 1]
        base_tasks.append(
            (
                int(expert),
                int(route_counts[expert]),
                int(base_plan.task_core_begins[task]),
                int(base_plan.task_threads[task]),
                [int(dep) for dep in dependencies[begin:end]],
            )
        )

    roots = sorted((task for task in base_tasks if not task[4]), key=lambda task: task[2])
    tails = sorted((task for task in base_tasks if task[4]), key=lambda task: task[2])
    if len(roots) != 6 or len(tails) != 2 or any(task[3] != 16 for task in roots):
        raise ValueError("static tail M-split requires a 6x16T head and exactly two terminal experts")
    if any(tail[1] % 2 != 0 or tail[1] // 2 < 120 for tail in tails):
        raise ValueError("each tail expert must split into two equal route ranges of at least 120 rows")

    tail_width = base_plan.num_threads // 4
    starts = [index * tail_width for index in range(4)]
    if layout == "grouped":
        assignments = ((tails[0], 0), (tails[0], 1), (tails[1], 0), (tails[1], 1))
    else:
        assignments = ((tails[0], 0), (tails[1], 0), (tails[0], 1), (tails[1], 1))

    tasks: list[tuple[int, int, int, int, list[int]]] = [
        (expert, routes, core_begin, threads, [])
        for expert, routes, core_begin, threads, _ in roots
    ]
    for core_begin, (tail, _slice) in zip(starts, assignments, strict=True):
        core_end = core_begin + tail_width
        blockers = [
            root_id
            for root_id, (_, _, root_begin, root_threads, _) in enumerate(roots)
            if root_begin < core_end and core_begin < root_begin + root_threads
        ]
        if not blockers:
            raise ValueError(f"tail interval [{core_begin}, {core_end}) has no first-wave blockers")
        tasks.append((tail[0], tail[1] // 2, core_begin, tail_width, blockers))

    cpu_ids = [int(cpu) for cpu in base_plan.thread_cpu_ids.tolist()]
    planner = IntervalPlanner(
        ContentionCostModel(profile),
        base_plan.num_threads,
        cpu_ids=cpu_ids,
        task_stage_window_policy=(
            AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1 if static_stage_windows else None
        ),
    )
    return AsyncMoEPlanV2.from_dict(planner.to_async_bridge(tasks))


def materialize_topk_ids(
    histogram: list[int] | tuple[int, ...],
    *,
    tokens: int,
    top_k: int,
    seed: int,
) -> torch.Tensor:
    """Build exact per-expert degrees with no duplicate expert inside a token."""
    counts = [int(value) for value in histogram]
    if tokens <= 0 or top_k <= 0:
        raise ValueError("tokens and top-k must be positive")
    if any(value < 0 or value > tokens for value in counts):
        raise ValueError("each route count must be in [0, tokens]")
    if sum(counts) != tokens * top_k:
        raise ValueError("route counts must sum to tokens * top-k")

    remaining = [(-count, expert) for expert, count in enumerate(counts) if count]
    heapq.heapify(remaining)
    rows: list[list[int]] = []
    for _ in range(tokens):
        if len(remaining) < top_k:
            raise ValueError("route histogram cannot form distinct TopK rows")
        selected = [heapq.heappop(remaining) for _ in range(top_k)]
        rows.append([expert for _, expert in selected])
        for negative_count, expert in selected:
            if negative_count < -1:
                heapq.heappush(remaining, (negative_count + 1, expert))
    if remaining:
        raise RuntimeError("route materialization left unassigned expert degrees")

    generator = random.Random(seed)
    generator.shuffle(rows)
    for row in rows:
        generator.shuffle(row)
    return torch.tensor(rows, dtype=torch.int32)


def make_topk_ids(args: argparse.Namespace) -> torch.Tensor:
    if args.preset is not None:
        workload = PAPER_WORKLOADS[args.preset]
        expected = (workload.tokens, workload.top_k, workload.num_experts)
        actual = (args.tokens, args.top_k, args.experts)
        if actual != expected:
            raise ValueError(f"preset {args.preset!r} requires tokens/top-k/experts={expected}, got {actual}")
        return materialize_topk_ids(
            workload.histogram,
            tokens=args.tokens,
            top_k=args.top_k,
            seed=args.seed,
        )
    distribution = args.distribution or "hot-topk"
    if distribution == "hot-topk":
        return torch.arange(args.top_k, dtype=torch.int32).repeat(args.tokens, 1)
    return torch.tensor(
        [[(token * args.top_k + slot) % args.experts for slot in range(args.top_k)] for token in range(args.tokens)],
        dtype=torch.int32,
    )


def make_fixed_team_schedule(
    active_experts: torch.Tensor,
    *,
    threads: int,
    requested_team_threads: int,
) -> tuple[torch.Tensor, ...]:
    active_count = int(active_experts.numel())
    if active_count == 0:
        raise ValueError("at least one expert must be active")
    team_threads = requested_team_threads or max(1, threads // min(active_count, threads))
    if team_threads <= 0 or team_threads > threads:
        raise ValueError(f"team-threads must be in [1, {threads}], got {team_threads}")
    slots = threads // team_threads
    if slots <= 0:
        raise ValueError("team-threads leaves no runnable team slot")

    core_begins: list[int] = []
    team_widths: list[int] = []
    dep_offsets = [0]
    deps: list[int] = []
    previous_by_slot = [-1] * slots
    for task_id in range(active_count):
        slot = task_id % slots
        core_begins.append(slot * team_threads)
        team_widths.append(team_threads)
        previous = previous_by_slot[slot]
        if previous >= 0:
            deps.append(previous)
        dep_offsets.append(len(deps))
        previous_by_slot[slot] = task_id

    return (
        active_experts.to(torch.int32),
        torch.tensor(core_begins, dtype=torch.int32),
        torch.tensor(team_widths, dtype=torch.int32),
        torch.tensor(dep_offsets, dtype=torch.int32),
        torch.tensor(deps, dtype=torch.int32),
    )


def make_static_16t_to_4x4t_schedule(
    route_counts: torch.Tensor,
    *,
    threads: int,
    long_route_threshold: int,
    iso_time_ns: Callable[[int, int], float],
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Build a fixed DAG whose 16T long heads release four 4T short lanes."""
    long_team_threads = 16
    short_team_threads = 4
    if route_counts.dim() != 1:
        raise ValueError(f"route_counts must be one-dimensional, got shape={tuple(route_counts.shape)}")
    if threads <= 0 or threads % long_team_threads != 0:
        raise ValueError(f"threads must be a positive multiple of {long_team_threads}, got {threads}")
    if long_route_threshold < 0:
        raise ValueError(f"static long-route threshold must be non-negative, got {long_route_threshold}")

    jobs = [(expert, int(routes)) for expert, routes in enumerate(route_counts.tolist()) if routes > 0]
    long_jobs = [(expert, routes) for expert, routes in jobs if routes > long_route_threshold]
    short_jobs = [(expert, routes) for expert, routes in jobs if routes <= long_route_threshold]
    superteam_count = threads // long_team_threads
    if not long_jobs or not short_jobs:
        raise ValueError(
            "static 16T -> 4x4T schedule requires both long and short active experts: "
            f"long={len(long_jobs)} short={len(short_jobs)} threshold={long_route_threshold}"
        )
    if len(long_jobs) > superteam_count:
        raise ValueError(
            "static 16T -> 4x4T schedule supports at most one long head per 16T superteam: "
            f"long={len(long_jobs)} superteams={superteam_count}"
        )

    iso_costs: dict[tuple[int, int], float] = {}

    def isolated_cost(routes: int, team_threads: int) -> float:
        key = (routes, team_threads)
        cost = iso_costs.get(key)
        if cost is None:
            cost = float(iso_time_ns(routes, team_threads))
            if not math.isfinite(cost) or cost <= 0.0:
                raise ValueError(
                    f"isolated time must be finite and positive for routes={routes}, threads={team_threads}, got {cost}"
                )
            iso_costs[key] = cost
        return cost

    long_jobs.sort(key=lambda job: (-isolated_cost(job[1], long_team_threads), -job[1], job[0]))
    short_jobs.sort(key=lambda job: (-isolated_cost(job[1], short_team_threads), -job[1], job[0]))

    short_lanes_per_superteam = long_team_threads // short_team_threads
    short_lane_count = threads // short_team_threads
    parent_by_superteam: list[tuple[int, int] | None] = [None] * superteam_count
    lane_load_ns = [0.0] * short_lane_count
    lane_release_ns = [0.0] * short_lane_count
    lane_jobs: list[list[int]] = [[] for _ in range(short_lane_count)]
    for superteam, job in enumerate(long_jobs):
        parent_by_superteam[superteam] = job
        release_ns = isolated_cost(job[1], long_team_threads)
        lane_begin = superteam * short_lanes_per_superteam
        for lane in range(lane_begin, lane_begin + short_lanes_per_superteam):
            lane_load_ns[lane] = release_ns
            lane_release_ns[lane] = release_ns

    for expert, routes in short_jobs:
        lane = min(range(short_lane_count), key=lambda candidate: (lane_load_ns[candidate], candidate))
        lane_jobs[lane].append(expert)
        lane_load_ns[lane] += isolated_cost(routes, short_team_threads)

    task_experts: list[int] = []
    task_core_begins: list[int] = []
    task_threads: list[int] = []
    task_dep_offsets = [0]
    task_deps: list[int] = []

    def append_task(expert: int, core_begin: int, team_threads: int, dependency: int | None) -> int:
        task_id = len(task_experts)
        task_experts.append(expert)
        task_core_begins.append(core_begin)
        task_threads.append(team_threads)
        if dependency is not None:
            task_deps.append(dependency)
        task_dep_offsets.append(len(task_deps))
        return task_id

    parent_task_by_superteam: list[int | None] = [None] * superteam_count
    for superteam, parent in enumerate(parent_by_superteam):
        if parent is None:
            continue
        parent_task_by_superteam[superteam] = append_task(
            parent[0],
            superteam * long_team_threads,
            long_team_threads,
            None,
        )

    for lane, experts in enumerate(lane_jobs):
        superteam = lane // short_lanes_per_superteam
        previous = parent_task_by_superteam[superteam]
        for expert in experts:
            previous = append_task(
                expert,
                lane * short_team_threads,
                short_team_threads,
                previous,
            )

    schedule = (
        torch.tensor(task_experts, dtype=torch.int32),
        torch.tensor(task_core_begins, dtype=torch.int32),
        torch.tensor(task_threads, dtype=torch.int32),
        torch.tensor(task_dep_offsets, dtype=torch.int32),
        torch.tensor(task_deps, dtype=torch.int32),
    )
    metadata = {
        "variant": STATIC_SPLIT_VARIANT,
        "long_route_threshold": long_route_threshold,
        "long_team_threads": long_team_threads,
        "short_team_threads": short_team_threads,
        "long_tasks": len(long_jobs),
        "short_tasks": len(short_jobs),
        "long_experts": [expert for expert, _ in long_jobs],
        "short_lane_task_counts": [len(experts) for experts in lane_jobs],
        "short_lane_release_ms": [value / 1.0e6 for value in lane_release_ns],
        "short_lane_finish_ms": [value / 1.0e6 for value in lane_load_ns],
        "predicted_isolated_makespan_ms": max(lane_load_ns) / 1.0e6,
    }
    return schedule, metadata


def make_production_schedule(
    route_counts: torch.Tensor,
    *,
    profile: Path,
    threads: int,
    cpu_ids: list[int],
    hidden: int,
    intermediate: int,
    experts: int,
    tail_pool_threads: int | None = None,
    tail_pool_max_routes: int = 12,
    static_stage_windows: bool = False,
) -> tuple[
    tuple[torch.Tensor, ...],
    AsyncMoEPlanV2,
    AsyncMoEPlanV2,
    AsyncMoEPlanV2 | None,
    AsyncMoEPlanV2 | None,
    tuple[AsyncMoEPlanV2, AsyncMoEPlanV2],
    tuple[AsyncMoEPlanV2, AsyncMoEPlanV2],
    dict[str, object],
    Callable[[int, int], float],
]:
    from phase_model import ContentionCostModel
    from interval_planner import PlannedTwoStagePlanner
    from planned_moe import PlannedMoE

    model = ContentionCostModel(profile)
    policy = model.policy
    if policy is None:
        raise ValueError(f"production profile must use schema v2: {profile}")
    expected = (hidden, intermediate, experts, threads)
    actual = (
        policy.hidden_size,
        policy.intermediate_size,
        policy.local_experts,
        policy.cores_per_rank,
    )
    if actual != expected:
        raise ValueError(
            "production profile hidden/intermediate/local-experts/cores mismatch: "
            f"expected {expected}, got {actual}"
        )

    native_module = importlib.import_module("fused_cpp._moe_C")
    runtime_extension_path = Path(native_module.__file__)
    runtime_extension_sha256 = hashlib.sha256(runtime_extension_path.read_bytes()).hexdigest()
    counts = [(expert, int(routes)) for expert, routes in enumerate(route_counts.tolist()) if routes > 0]
    # Keep the historical no-stage-window plan as the controlled comparator.
    # PlannedMoE itself enables the matching static policy by default.
    planner = PlannedMoE(
        model,
        threads,
        cpu_ids=cpu_ids,
        use_default_stage_window_policy=False,
    )
    begin = time.perf_counter_ns()
    planner.plan_spec_for(counts)
    cold_plan_ns = time.perf_counter_ns() - begin
    cold_auto_metadata = dict(planner.last)
    begin = time.perf_counter_ns()
    auto_spec = planner.plan_spec_for(counts)
    warm_plan_ns = time.perf_counter_ns() - begin
    strict_spec = planner.plan_spec_for(counts, dynamic_tail_pool=False)
    strict_plan = AsyncMoEPlanV2.from_dict(strict_spec["bridge"])
    auto_plan = AsyncMoEPlanV2.from_dict(auto_spec["bridge"])
    static_stage_window_plan = None
    static_stage_window_metadata = None
    if static_stage_windows:
        static_planner = PlannedMoE(
            model,
            threads,
            cpu_ids=cpu_ids,
        )
        static_spec = static_planner.plan_spec_for(counts)
        static_bridge = static_spec["bridge"]
        static_stage_window_plan = AsyncMoEPlanV2.from_dict(static_bridge)
        range_pairs = list(
            zip(
                static_bridge["task_w13_ranges"],
                static_bridge["task_w2_ranges"],
            )
        )
        baseline_ranges = (
            model.measurement_geometry.w13_ranges,
            model.measurement_geometry.w2_ranges,
        )
        static_stage_window_metadata = {
            "name": AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1.name,
            "shape": list(static_spec["shape"]),
            "execution_mode": static_spec["execution_mode"],
            "tail_pool_threads": static_spec["tail_pool_threads"],
            "tail_pool_max_routes": static_spec["tail_pool_max_routes"],
            "tail_repartition_width": static_spec["tail_repartition_width"],
            "tail_repartition_tasks": static_spec["tail_repartition_tasks"],
            "tail_repartition_route_slices": static_spec["tail_repartition_route_slices"],
            "overridden_tasks": sum(pair != baseline_ranges for pair in range_pairs),
            "range_pairs": sorted({f"{w13}:{w2}" for w13, w2 in range_pairs}),
        }
    tail_pool_plan = None
    if tail_pool_threads is not None:
        tail_pool_spec = planner.plan_spec_for(
            counts,
            tail_pool_threads=tail_pool_threads,
            tail_pool_max_routes=tail_pool_max_routes,
        )
        tail_pool_plan = AsyncMoEPlanV2.from_dict(tail_pool_spec["bridge"])
    matched_staged_plan = static_stage_window_plan or auto_plan
    matched_staged_source = (
        STATIC_STAGE_WINDOW_VARIANT if static_stage_window_plan is not None else AUTO_VARIANT
    )
    if matched_staged_plan.task_expert_ids.numel() != len(counts):
        # The benchmark-only two-stage runtime still requires one task per
        # expert. Keep that comparator valid when production selects M slices.
        matched_staged_plan = strict_plan
        matched_staged_source = "production_strict"
    matched_staged_plans = (matched_staged_plan, matched_staged_plan)
    stage_window_policy = (
        AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1 if static_stage_windows else None
    )
    staged_planner = PlannedTwoStagePlanner(
        model,
        threads,
        cpu_ids=cpu_ids,
        task_stage_window_policy=stage_window_policy,
    )
    begin = time.perf_counter_ns()
    staged_spec = staged_planner.plan(counts)
    staged_plan_ns = time.perf_counter_ns() - begin
    fine_staged_plans = (
        AsyncMoEPlanV2.from_dict(staged_spec["w13"]["bridge"]),
        AsyncMoEPlanV2.from_dict(staged_spec["w2"]["bridge"]),
    )
    schedule = strict_plan.legacy_schedule()
    return (
        schedule,
        strict_plan,
        auto_plan,
        static_stage_window_plan,
        tail_pool_plan,
        matched_staged_plans,
        fine_staged_plans,
        {
            "profile": str(profile),
            "profile_extension_sha256": policy.extension_sha256,
            "runtime_extension_sha256": runtime_extension_sha256,
            "extension_hash_match": runtime_extension_sha256 == policy.extension_sha256,
            "shape": list(auto_spec["shape"]),
            "strict_shape": list(strict_spec["shape"]),
            "execution_mode": auto_spec["execution_mode"],
            "tail_pool_threads": auto_spec["tail_pool_threads"],
            "tail_pool_max_routes": auto_spec["tail_pool_max_routes"],
            "tail_pool_tasks": auto_spec["tail_pool_tasks"],
            "tail_repartition_width": auto_spec["tail_repartition_width"],
            "tail_repartition_tasks": auto_spec["tail_repartition_tasks"],
            "tail_repartition_route_slices": auto_spec["tail_repartition_route_slices"],
            "tail_repartition_candidates": cold_auto_metadata["tail_repartition_candidates"],
            "task_range_pairs": sorted(
                {
                    f"{int(w13)}:{int(w2)}"
                    for w13, w2 in zip(
                        auto_spec["bridge"]["task_w13_ranges"],
                        auto_spec["bridge"]["task_w2_ranges"],
                        strict=True,
                    )
                }
            ),
            "calibration_ranges": [
                model.measurement_geometry.w13_ranges,
                model.measurement_geometry.w2_ranges,
            ],
            "cold_plan_ms": cold_plan_ns / 1.0e6,
            "warm_plan_ms": warm_plan_ns / 1.0e6,
            "static_stage_windows": static_stage_window_metadata,
            "planned_staged": {
                "matched_source": matched_staged_source,
                "plan_ms": staged_plan_ns / 1.0e6,
                "predicted_ms": staged_spec["makespan_ns"] / 1.0e6,
                "w13_shape": list(staged_spec["w13"]["shape"]),
                "w13_execution_mode": staged_spec["w13"]["execution_mode"],
                "w13_tail_pool_threads": staged_spec["w13"]["tail_pool_threads"],
                "w2_shape": list(staged_spec["w2"]["shape"]),
                "w2_execution_mode": staged_spec["w2"]["execution_mode"],
                "w2_tail_pool_threads": staged_spec["w2"]["tail_pool_threads"],
            },
        },
        model.T_iso,
    )


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    positive = (
        args.tokens,
        args.hidden,
        args.intermediate,
        args.experts,
        args.top_k,
        args.threads,
        args.runs,
    )
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("shapes, threads, and runs must be positive; warmup must be non-negative")
    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed experts")
    if args.hidden % 8 != 0 or args.intermediate % 8 != 0:
        raise ValueError("hidden and intermediate must be multiples of 8")
    if args.production_profile is not None and not args.production_profile.is_file():
        raise ValueError(f"production profile does not exist: {args.production_profile}")
    elastic_transitions = (
        parse_width_transitions(args.elastic_w2_transitions)
        if args.elastic_w2_transitions is not None
        else None
    )
    elastic_timeouts_us = args.elastic_timeout_us or [0.0]
    elastic_task_ids = (
        parse_integer_list(args.elastic_task_ids)
        if args.elastic_task_ids is not None
        else None
    )
    elastic_w2_core_begins = (
        parse_task_core_begins(args.elastic_w2_core_begins)
        if args.elastic_w2_core_begins is not None
        else None
    )
    static_tail_widths = (
        parse_integer_list(args.static_tail_widths)
        if args.static_tail_widths is not None
        else []
    )
    if any(timeout < 0 for timeout in elastic_timeouts_us):
        raise ValueError("--elastic-timeout-us must be non-negative")
    if elastic_transitions is not None and args.production_profile is None:
        raise ValueError("--elastic-w2-transitions requires --production-profile")
    if (elastic_task_ids is not None or elastic_w2_core_begins is not None) and elastic_transitions is None:
        raise ValueError("--elastic-task-ids/--elastic-w2-core-begins require --elastic-w2-transitions")
    if static_tail_widths and args.production_profile is None:
        raise ValueError("--static-tail-widths requires --production-profile")
    if args.static_tail_m_split and args.production_profile is None:
        raise ValueError("--static-tail-m-split requires --production-profile")
    if args.static_16_to_4 and args.production_profile is None:
        raise ValueError("--static-16-to-4 requires --production-profile for isolated-time lane assignment")
    if args.dynamic_short_pool and args.production_profile is None:
        raise ValueError("--dynamic-short-pool requires --production-profile")
    if args.static_stage_windows is None:
        args.static_stage_windows = args.production_profile is not None
    elif args.static_stage_windows and args.production_profile is None:
        raise ValueError("--static-stage-windows requires --production-profile")
    if args.dynamic_short_pool and (args.static_long_route_threshold <= 0 or args.threads % 4 != 0):
        raise ValueError("--dynamic-short-pool requires a positive route threshold and threads divisible by 4")

    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    thread_cpu_ids = torch.tensor(cpu_ids, dtype=torch.int32)
    torch.set_num_threads(1)

    topk_ids = make_topk_ids(args)
    route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    active_experts = torch.nonzero(route_counts, as_tuple=False).flatten()
    planner_metadata: dict[str, object] | None = None
    iso_time_ns: Callable[[int, int], float] | None = None
    production_plan: AsyncMoEPlanV2 | None = None
    auto_plan: AsyncMoEPlanV2 | None = None
    static_stage_window_plan: AsyncMoEPlanV2 | None = None
    tail_pool_plan: AsyncMoEPlanV2 | None = None
    matched_staged_plans: tuple[AsyncMoEPlanV2, AsyncMoEPlanV2] | None = None
    fine_staged_plans: tuple[AsyncMoEPlanV2, AsyncMoEPlanV2] | None = None
    elastic_plans: dict[str, AsyncMoEPlanV2] = {}
    static_tail_plans: dict[str, AsyncMoEPlanV2] = {}
    if args.production_profile is None:
        baseline_variant = "fixed_team_async"
        schedule = make_fixed_team_schedule(
            active_experts,
            threads=args.threads,
            requested_team_threads=args.team_threads,
        )
        team_threads = int(schedule[2][0])
        w13_ranges, w2_ranges = 2, 1
    else:
        baseline_variant = "production_strict"
        (
            schedule,
            production_plan,
            auto_plan,
            static_stage_window_plan,
            tail_pool_plan,
            matched_staged_plans,
            fine_staged_plans,
            planner_metadata,
            iso_time_ns,
        ) = make_production_schedule(
            route_counts,
            profile=args.production_profile,
            threads=args.threads,
            cpu_ids=cpu_ids,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
            tail_pool_threads=4 if args.dynamic_short_pool else None,
            tail_pool_max_routes=args.static_long_route_threshold,
            static_stage_windows=args.static_stage_windows,
        )
        team_threads = None
        calibration_ranges = planner_metadata["calibration_ranges"]
        w13_ranges, w2_ranges = map(int, calibration_ranges)
        if elastic_transitions is not None:
            assert production_plan is not None
            for timeout_us in elastic_timeouts_us:
                timeout_ns = round(timeout_us * 1000.0)
                variant = (
                    f"{ELASTIC_VARIANT_PREFIX}_"
                    f"{'_'.join(f'{selected}to{preferred}' for selected, preferred in sorted(elastic_transitions.items()))}"
                    f"_{timeout_ns}ns"
                )
                elastic_plans[variant] = make_elastic_w2_plan(
                    production_plan,
                    route_counts,
                    profile=args.production_profile,
                    transitions=elastic_transitions,
                    timeout_ns=timeout_ns,
                    static_stage_windows=args.static_stage_windows,
                    resizable_task_ids=elastic_task_ids,
                    task_preferred_core_begins=elastic_w2_core_begins,
                )
        if static_tail_widths:
            assert production_plan is not None
            for tail_width in static_tail_widths:
                variant = f"{STATIC_TAIL_VARIANT_PREFIX}_2x{tail_width}t"
                static_tail_plans[variant] = make_static_tail_repartition_plan(
                    production_plan,
                    route_counts,
                    profile=args.production_profile,
                    tail_width=tail_width,
                    static_stage_windows=args.static_stage_windows,
                )
        if args.static_tail_m_split:
            assert production_plan is not None
            for layout in ("grouped", "interleaved"):
                variant = f"{STATIC_TAIL_VARIANT_PREFIX}_m2_{layout}_4x24t"
                static_tail_plans[variant] = make_static_tail_m_split_plan(
                    production_plan,
                    route_counts,
                    profile=args.production_profile,
                    layout=layout,
                    static_stage_windows=args.static_stage_windows,
                )
    static_schedule: tuple[torch.Tensor, ...] | None = None
    static_metadata: dict[str, object] | None = None
    ready_token_policy_variants: dict[str, tuple[bool, int, bool]] = {}
    if args.ready_token_drain_batches:
        if auto_plan is None:
            raise ValueError("--ready-token-drain-batches requires --production-profile")
        drain_batches = list(dict.fromkeys(parse_integer_list(args.ready_token_drain_batches)))
        if any(batch < 1 or batch > 64 for batch in drain_batches):
            raise ValueError("ready-token drain batch sizes must be in [1, 64]")
        ready_token_policy_variants[f"{AUTO_VARIANT}_ready_legacy"] = (False, 1, False)
        for batch in drain_batches:
            ready_token_policy_variants[f"{AUTO_VARIANT}_ready_drain_b{batch}"] = (
                True,
                batch,
                batch > 1,
            )
    variant_names = [baseline_variant]
    if auto_plan is not None:
        variant_names.append(AUTO_VARIANT)
        variant_names.extend(ready_token_policy_variants)
    if static_stage_window_plan is not None:
        variant_names.append(STATIC_STAGE_WINDOW_VARIANT)
    if args.static_16_to_4:
        assert iso_time_ns is not None
        static_schedule, static_metadata = make_static_16t_to_4x4t_schedule(
            route_counts,
            threads=args.threads,
            long_route_threshold=args.static_long_route_threshold,
            iso_time_ns=iso_time_ns,
        )
        variant_names.append(STATIC_SPLIT_VARIANT)
    if args.dynamic_short_pool:
        variant_names.append(DYNAMIC_POOL_VARIANT)
    variant_names.extend(static_tail_plans)
    variant_names.extend(elastic_plans)
    if matched_staged_plans is not None:
        variant_names.extend((MATCHED_STAGED_VARIANT, FINE_STAGED_VARIANT))
    variant_names.append(VLLM_VARIANT)
    variants = tuple(variant_names)
    ready_token_merge = args.production_ready_token_merge
    if ready_token_merge is None:
        ready_token_merge = args.production_profile is not None
    if elastic_plans:
        ready_token_merge = False
    if ready_token_policy_variants and not ready_token_merge:
        raise ValueError(
            "--ready-token-drain-batches requires production ready-token merge"
        )

    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty((args.experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((args.experts, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k), generator=generator), dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "1" if ready_token_merge else "0"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if args.route_dtype == "bf16" else "0"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_MAX_ROWS"] = str(args.static_long_route_threshold)
    os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"
    default_ready_token_policy = (
        os.environ.get("FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN", "1") != "0",
        int(os.environ.get("FUSED_CPP_MOE_ASYNC_READY_TOKEN_BATCH", "2")),
        os.environ.get("FUSED_CPP_MOE_ASYNC_READY_TOKEN_PREFETCH", "1") != "0",
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 fused MoE backend")
    del w13, w2

    outputs = {name: torch.empty_like(hidden) for name in variants}
    elastic_stats = {name: make_async_moe_elastic_stats() for name in elastic_plans}

    def run(name: str) -> torch.Tensor:
        drain, batch, prefetch = ready_token_policy_variants.get(
            name, default_ready_token_policy
        )
        os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN"] = "1" if drain else "0"
        os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_BATCH"] = str(batch)
        os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_PREFETCH"] = "1" if prefetch else "0"
        if name in static_tail_plans:
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                static_tail_plans[name],
                global_num_experts=args.experts,
                out=outputs[name],
            )
        if name in elastic_plans:
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                elastic_plans[name],
                global_num_experts=args.experts,
                out=outputs[name],
                elastic_stats_out=elastic_stats[name],
            )
        if name in {MATCHED_STAGED_VARIANT, FINE_STAGED_VARIANT}:
            selected = matched_staged_plans if name == MATCHED_STAGED_VARIANT else fine_staged_plans
            assert selected is not None
            return fused_moe_bf16_tiled_planned_staged(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                selected[0],
                selected[1],
                global_num_experts=args.experts,
                out=outputs[name],
            )
        if name == DYNAMIC_POOL_VARIANT:
            assert tail_pool_plan is not None
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                tail_pool_plan,
                global_num_experts=args.experts,
                out=outputs[name],
            )
        if name == AUTO_VARIANT or name in ready_token_policy_variants:
            assert auto_plan is not None
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                auto_plan,
                global_num_experts=args.experts,
                out=outputs[name],
            )
        if name == STATIC_STAGE_WINDOW_VARIANT:
            assert static_stage_window_plan is not None
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                static_stage_window_plan,
                global_num_experts=args.experts,
                out=outputs[name],
            )
        if name == baseline_variant and production_plan is not None:
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                production_plan,
                global_num_experts=args.experts,
                out=outputs[name],
            )
        if name != VLLM_VARIANT:
            selected_schedule = schedule
            if name == STATIC_SPLIT_VARIANT:
                assert static_schedule is not None
                selected_schedule = static_schedule
            return fused_moe_bf16_tiled_async(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                *selected_schedule,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=args.threads,
                global_num_experts=args.experts,
                w13_ranges=w13_ranges,
                w2_ranges=w2_ranges,
                out=outputs[name],
            )
        return fused_moe_bf16_tiled_vllm_staged(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=args.threads,
            global_num_experts=args.experts,
            out=outputs[name],
        )

    reference = run(baseline_variant).clone()
    for name in variants:
        if name == baseline_variant:
            continue
        candidate = run(name).clone()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for _ in range(args.warmup):
        order = list(variants)
        warmup_order.shuffle(order)
        for name in order:
            run(name)

    samples = {name: [] for name in variants}
    elastic_stats_samples: dict[str, list[dict[str, int]]] = {name: [] for name in elastic_plans}
    timed_order = random.Random(args.seed ^ 0x5A5A)
    sink = 0
    for _ in range(args.runs):
        order = list(variants)
        timed_order.shuffle(order)
        for name in order:
            begin = time.perf_counter_ns()
            result = run(name)
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            if name in elastic_stats:
                elastic_stats_samples[name].append(decode_async_moe_elastic_stats(elastic_stats[name]))
            sink ^= int(result.view(torch.int16)[0, 0])

    if args.stage_timing:
        os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "1"
        for name in variants:
            run(name)
        os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"

    total_flops = 6 * args.tokens * args.top_k * args.hidden * args.intermediate
    baseline_ms = statistics.median(samples[baseline_variant])
    auto_ms = statistics.median(samples[AUTO_VARIANT]) if AUTO_VARIANT in samples else baseline_ms
    records: list[dict[str, object]] = []
    for name in variants:
        median_ms = statistics.median(samples[name])
        record: dict[str, object] = {
            "variant": name,
            "median_ms": median_ms,
            "p10_ms": percentile(samples[name], 0.10),
            "p90_ms": percentile(samples[name], 0.90),
            "aggregate_tflops": total_flops / median_ms / 1.0e9,
            "gain_pct": 100.0 * (baseline_ms / median_ms - 1.0),
            "gain_vs_auto_pct": 100.0 * (auto_ms / median_ms - 1.0),
            "samples_ms": samples[name],
        }
        if name in ready_token_policy_variants:
            drain, batch, prefetch = ready_token_policy_variants[name]
            record["ready_token_policy"] = {
                "same_job_drain": drain,
                "batch": batch,
                "prefetch": prefetch,
            }
        if name in elastic_stats_samples:
            stat_samples = elastic_stats_samples[name]
            totals = {
                field: sum(sample[field] for sample in stat_samples)
                for field in stat_samples[0]
            }
            eligible = totals["eligible_tasks"]
            totals["natural_opportunity_pct"] = (
                100.0 * totals["natural_opportunities"] / eligible if eligible else 0.0
            )
            totals["preferred_assignment_pct"] = (
                100.0 * totals["preferred_assignments"] / eligible if eligible else 0.0
            )
            record["elastic_stats"] = totals
            record["elastic_stats_samples"] = stat_samples
        records.append(record)

    result = {
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "top_k": args.top_k,
            "threads": args.threads,
            "distribution": args.preset or args.distribution or "hot-topk",
            "active_experts": int(active_experts.numel()),
            "team_threads": team_threads,
            "route_dtype": args.route_dtype,
            "route_counts": route_counts.tolist(),
        },
        "method": {
            "baseline_variant": baseline_variant,
            "w13_ranges": w13_ranges,
            "w2_ranges": w2_ranges,
            "async_ready_token_merge": ready_token_merge,
            "ready_token_policy_sweep": {
                name: {
                    "same_job_drain": policy[0],
                    "batch": policy[1],
                    "prefetch": policy[2],
                }
                for name, policy in ready_token_policy_variants.items()
            }
            if ready_token_policy_variants
            else None,
            "direct_route_store": True,
            "route_merge_unroll": 1,
            "planner": planner_metadata,
            "static_16_to_4": static_metadata,
            "dynamic_short_pool": {
                "plan_version": 2,
                "execution_mode": "tail_pool",
                "pool_threads": 4,
                "max_rows": args.static_long_route_threshold,
            }
            if args.dynamic_short_pool
            else None,
            "elastic_w2": {
                "transitions": elastic_transitions,
                "timeouts_us": elastic_timeouts_us,
                "task_ids": elastic_task_ids,
                "task_preferred_core_begins": elastic_w2_core_begins,
                "cross_numa": False,
                "ready_token_merge_forced_off": bool(elastic_plans),
            }
            if elastic_plans
            else None,
            "static_tail_repartition": {
                name: {
                    "task_expert_ids": plan.task_expert_ids.tolist(),
                    "task_core_begins": plan.task_core_begins.tolist(),
                    "task_threads": plan.task_threads.tolist(),
                    "task_dep_offsets": plan.task_dep_offsets.tolist(),
                    "task_deps": plan.task_deps.tolist(),
                    "task_range_granularities": plan.task_range_granularities.tolist(),
                }
                for name, plan in static_tail_plans.items()
            }
            if static_tail_plans
            else None,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "records": records,
        "sink": sink,
    }

    if planner_metadata is not None:
        if not planner_metadata["extension_hash_match"]:
            print(
                "WARNING: runtime extension hash differs from the calibration profile: "
                f"{planner_metadata['runtime_extension_sha256']} != "
                f"{planner_metadata['profile_extension_sha256']}"
            )
        pool_description = "none"
        if planner_metadata["tail_pool_threads"] is not None:
            pool_description = f"{planner_metadata['tail_pool_threads']}T/M<={planner_metadata['tail_pool_max_routes']}"
        print(
            f"production shape={tuple(planner_metadata['shape'])} "
            f"mode={planner_metadata['execution_mode']} "
            f"pool={pool_description} "
            f"plan_ms(cold/warm)={planner_metadata['cold_plan_ms']:.3f}/{planner_metadata['warm_plan_ms']:.3f}"
        )
    print("variant                  median_ms   TFLOP/s   gain_pct     p10_ms     p90_ms")
    for record in records:
        print(
            f"{record['variant']:<24} {record['median_ms']:>9.3f} "
            f"{record['aggregate_tflops']:>9.3f} {record['gain_pct']:>10.2f} "
            f"{record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f}"
        )
    if static_stage_window_plan is not None:
        static_record = next(record for record in records if record["variant"] == STATIC_STAGE_WINDOW_VARIANT)
        print(f"static-stage-window gain vs production_auto: {static_record['gain_vs_auto_pct']:.2f}%")
    for record in records:
        if "elastic_stats" not in record:
            continue
        stats = record["elastic_stats"]
        assert isinstance(stats, dict)
        print(
            f"{record['variant']} opportunities: natural={stats['natural_opportunity_pct']:.2f}% "
            f"preferred={stats['preferred_assignment_pct']:.2f}% "
            f"timeout_fallbacks={stats['timeout_fallbacks']} "
            f"max_wait_us={stats['max_wait_ns'] / 1000.0:.3f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
