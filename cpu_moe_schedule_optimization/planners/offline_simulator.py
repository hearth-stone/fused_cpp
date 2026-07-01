#!/usr/bin/env python3
"""Offline simulator for CPU MoE expert scheduling plans.

This is a research-only tool. It estimates T_plan + T_execute from a routing
histogram and an expert cost model, without touching the fused_cpp runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class PlanKind(str, Enum):
    FIXED_GLOBAL_THREADS = "FIXED_GLOBAL_THREADS"
    SORTED_TOKEN_BALANCED_1T = "SORTED_TOKEN_BALANCED_1T"
    UNIFORM_WAVES = "UNIFORM_WAVES"
    GREEDY_MARGINAL_GAIN = "GREEDY_MARGINAL_GAIN"
    ENUMERATE_CORE_GROUPS = "ENUMERATE_CORE_GROUPS"


PLANNER_ALIASES = {
    "fixed": PlanKind.FIXED_GLOBAL_THREADS,
    "fixed_global_threads": PlanKind.FIXED_GLOBAL_THREADS,
    "balanced": PlanKind.SORTED_TOKEN_BALANCED_1T,
    "lpt": PlanKind.SORTED_TOKEN_BALANCED_1T,
    "lpt_balanced": PlanKind.SORTED_TOKEN_BALANCED_1T,
    "token_balanced": PlanKind.SORTED_TOKEN_BALANCED_1T,
    "sorted_token_balanced": PlanKind.SORTED_TOKEN_BALANCED_1T,
    "sorted_token_balanced_1t": PlanKind.SORTED_TOKEN_BALANCED_1T,
    "uniform": PlanKind.UNIFORM_WAVES,
    "uniform_waves": PlanKind.UNIFORM_WAVES,
    "greedy": PlanKind.GREEDY_MARGINAL_GAIN,
    "greedy_marginal_gain": PlanKind.GREEDY_MARGINAL_GAIN,
    "groups": PlanKind.ENUMERATE_CORE_GROUPS,
    "core_groups": PlanKind.ENUMERATE_CORE_GROUPS,
    "enumerate": PlanKind.ENUMERATE_CORE_GROUPS,
    "enumerate_core_groups": PlanKind.ENUMERATE_CORE_GROUPS,
}

DEFAULT_PLANNER_COSTS_NS = {
    PlanKind.FIXED_GLOBAL_THREADS: 1_000,
    PlanKind.SORTED_TOKEN_BALANCED_1T: 2_000,
    PlanKind.UNIFORM_WAVES: 5_000,
    PlanKind.GREEDY_MARGINAL_GAIN: 20_000,
    PlanKind.ENUMERATE_CORE_GROUPS: 30_000,
}

COMPLEXITY_COST_COEFFICIENTS_NS = {
    "fixed_base": 500,
    "balanced_base": 700,
    "uniform_base": 800,
    "greedy_base": 1_000,
    "groups_base": 1_200,
    "cost_lookup": 2,
    "linear_scan": 5,
    "wave_pack": 4,
    "gain_scan": 3,
    "sort_compare": 6,
    "budget_eval": 100,
}

MAX_CORE_GROUP_SHAPES = 512


@dataclass(frozen=True)
class ExpertWork:
    expert_id: int
    routes: int
    route_bucket: Optional[int] = None

    def to_dict(self) -> Dict[str, int]:
        data = {"expert_id": self.expert_id, "routes": self.routes}
        if self.route_bucket is not None:
            data["route_bucket"] = self.route_bucket
        return data


@dataclass(frozen=True)
class Team:
    expert_id: int
    routes: int
    threads: int
    estimated_time_ns: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "expert_id": self.expert_id,
            "routes": self.routes,
            "threads": self.threads,
            "estimated_time_ns": self.estimated_time_ns,
        }


@dataclass(frozen=True)
class Wave:
    wave_id: int
    teams: List[Team]
    estimated_wave_time_ns: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "wave_id": self.wave_id,
            "teams": [team.to_dict() for team in self.teams],
            "estimated_wave_time_ns": self.estimated_wave_time_ns,
        }


@dataclass(frozen=True)
class Plan:
    kind: PlanKind
    num_cores: int
    active_experts: List[ExpertWork]
    waves: List[Wave]
    measured_plan_cost_ns: int
    estimated_plan_cost_ns: int
    exactness_scope: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def estimated_execute_cost_ns(self) -> int:
        return sum(wave.estimated_wave_time_ns for wave in self.waves)

    @property
    def estimated_total_cost_ns(self) -> int:
        return self.estimated_plan_cost_ns + self.estimated_execute_cost_ns

    def to_scheduled_bridge(self) -> Dict[str, object]:
        """Return tensors-as-lists accepted by fused_moe_bf16_tiled_scheduled."""
        wave_offsets = [0]
        team_expert_ids: List[int] = []
        team_threads: List[int] = []
        for wave in self.waves:
            for team in wave.teams:
                team_expert_ids.append(team.expert_id)
                team_threads.append(team.threads)
            wave_offsets.append(len(team_expert_ids))
        return {
            "num_threads": self.num_cores,
            "thread_cpu_ids": list(range(self.num_cores)),
            "wave_offsets": wave_offsets,
            "team_expert_ids": team_expert_ids,
            "team_threads": team_threads,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind.value,
            "num_cores": self.num_cores,
            "active_experts": [work.to_dict() for work in self.active_experts],
            "waves": [wave.to_dict() for wave in self.waves],
            "measured_plan_cost_ns": self.measured_plan_cost_ns,
            "estimated_plan_cost_ns": self.estimated_plan_cost_ns,
            "estimated_execute_cost_ns": self.estimated_execute_cost_ns,
            "estimated_total_cost_ns": self.estimated_total_cost_ns,
            "exactness_scope": self.exactness_scope,
            "scheduled_bridge": self.to_scheduled_bridge(),
            "metadata": self.metadata,
        }


class ExpertCostModel:
    """Lookup wrapper for T_expert(routes, threads)."""

    def __init__(
        self,
        lookup: Callable[[int, int], int],
        source: str,
        metric: str = "median_ns",
    ) -> None:
        self._lookup = lookup
        self.source = source
        self.metric = metric

    def estimate_ns(self, routes: int, threads: int) -> int:
        if routes <= 0:
            raise ValueError("routes must be positive")
        if threads <= 0:
            raise ValueError("threads must be positive")
        return self._lookup(routes, threads)

    @classmethod
    def synthetic(cls) -> "ExpertCostModel":
        def lookup(routes: int, threads: int) -> int:
            useful_threads = max(1, min(threads, routes))
            serial_ns = 18_000 + routes * 1_850
            speedup = 1.0 + 0.82 * ((useful_threads - 1) ** 0.72)
            thread_overhead_ns = 900 * threads + 35 * threads * threads
            return int(serial_ns / speedup + thread_overhead_ns)

        return cls(lookup=lookup, source="synthetic", metric="median_ns")

    @classmethod
    def from_json(cls, path: Path) -> "ExpertCostModel":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        metric = payload.get("metric", "median_ns")
        entries = payload.get("entries", [])
        table: Dict[Tuple[int, int], int] = {}

        for entry in entries:
            routes = int(entry["routes"])
            threads = int(entry["threads"])
            table[(routes, threads)] = int(entry[metric])

        if not table:
            raise ValueError(f"cost table has no entries: {path}")

        route_buckets = sorted({routes for routes, _ in table})
        thread_buckets = sorted({threads for _, threads in table})

        def nearest(value: int, buckets: Sequence[int]) -> int:
            return min(buckets, key=lambda bucket: (abs(bucket - value), bucket))

        def lookup(routes: int, threads: int) -> int:
            key = (routes, threads)
            if key in table:
                return table[key]
            route_bucket = nearest(routes, route_buckets)
            thread_bucket = nearest(threads, thread_buckets)
            return table[(route_bucket, thread_bucket)]

        return cls(lookup=lookup, source=str(path), metric=metric)


@dataclass(frozen=True)
class PlannerCostModel:
    """Cost model used for scoring planner overhead.

    The simulator keeps Python wall time as diagnostics but scores plans with
    this model by default. That avoids confusing Python prototype cost with a
    future native scheduler cost.
    """

    source: str
    static_costs_ns: Dict[PlanKind, int]
    override_costs_ns: Dict[PlanKind, int] = field(default_factory=dict)
    coefficients_ns: Dict[str, int] = field(
        default_factory=lambda: dict(COMPLEXITY_COST_COEFFICIENTS_NS)
    )
    profile_costs_ns: Dict[PlanKind, List[Tuple[int, int, int]]] = field(
        default_factory=dict
    )
    profile_metric: str = "total_native_median_ns"
    profile_source: Optional[str] = None

    def estimate_ns(
        self,
        kind: PlanKind,
        measured_plan_cost_ns: int,
        active_count: int,
        num_cores: int,
        extra_wave_window: int = 8,
        core_group_shape_count: Optional[int] = None,
    ) -> int:
        if self.source == "measured":
            return measured_plan_cost_ns
        if kind in self.override_costs_ns:
            return self.override_costs_ns[kind]
        if self.source in {"native_table", "profile"}:
            return self.profile_estimate_ns(kind, active_count, num_cores)
        if self.source == "model":
            return self.static_costs_ns[kind]
        if self.source == "complexity":
            return self.complexity_estimate_ns(
                kind=kind,
                active_count=active_count,
                num_cores=num_cores,
                extra_wave_window=extra_wave_window,
                core_group_shape_count=core_group_shape_count,
            )
        raise ValueError(f"unknown planner cost source: {self.source}")

    def profile_estimate_ns(
        self,
        kind: PlanKind,
        active_count: int,
        num_cores: int,
    ) -> int:
        candidates = self.profile_costs_ns.get(kind)
        if not candidates:
            return self.static_costs_ns[kind]
        active_count = max(0, active_count)
        num_cores = max(1, num_cores)
        active, cores, cost_ns = min(
            candidates,
            key=lambda item: (
                abs(item[1] - num_cores),
                abs(item[0] - active_count),
                item[1],
                item[0],
            ),
        )
        del active, cores
        return cost_ns

    def complexity_estimate_ns(
        self,
        kind: PlanKind,
        active_count: int,
        num_cores: int,
        extra_wave_window: int,
        core_group_shape_count: Optional[int],
    ) -> int:
        features = self.complexity_features(
            kind,
            active_count,
            num_cores,
            extra_wave_window,
            core_group_shape_count,
        )
        coeffs = self.coefficients_ns
        return int(
            features["base"]
            + features["cost_lookup_ops"] * coeffs["cost_lookup"]
            + features["linear_scan_ops"] * coeffs["linear_scan"]
            + features["wave_pack_ops"] * coeffs["wave_pack"]
            + features["gain_scan_ops"] * coeffs["gain_scan"]
            + features["sort_compare_ops"] * coeffs["sort_compare"]
            + features["budget_eval_ops"] * coeffs["budget_eval"]
        )

    def complexity_features(
        self,
        kind: PlanKind,
        active_count: int,
        num_cores: int,
        extra_wave_window: int,
        core_group_shape_count: Optional[int] = None,
    ) -> Dict[str, int]:
        active_count = max(0, active_count)
        num_cores = max(1, num_cores)
        log_active = int(math.ceil(math.log2(max(2, active_count))))

        if kind == PlanKind.FIXED_GLOBAL_THREADS:
            return {
                "base": self.coefficients_ns["fixed_base"],
                "cost_lookup_ops": active_count,
                "linear_scan_ops": active_count,
                "wave_pack_ops": active_count,
                "gain_scan_ops": 0,
                "sort_compare_ops": 0,
                "budget_eval_ops": 1,
            }

        if kind == PlanKind.SORTED_TOKEN_BALANCED_1T:
            return {
                "base": self.coefficients_ns["balanced_base"],
                "cost_lookup_ops": active_count,
                "linear_scan_ops": active_count * num_cores,
                "wave_pack_ops": active_count,
                "gain_scan_ops": 0,
                "sort_compare_ops": active_count * log_active,
                "budget_eval_ops": 1,
            }

        if kind == PlanKind.UNIFORM_WAVES:
            enum_count = num_cores
            enum_work = active_count * enum_count
            return {
                "base": self.coefficients_ns["uniform_base"],
                "cost_lookup_ops": enum_work,
                "linear_scan_ops": enum_work,
                "wave_pack_ops": enum_work,
                "gain_scan_ops": 0,
                "sort_compare_ops": 0,
                "budget_eval_ops": enum_count,
            }

        if kind == PlanKind.GREEDY_MARGINAL_GAIN:
            if active_count == 0:
                min_waves = 0
                max_waves = 0
            else:
                min_waves = math.ceil(active_count / num_cores)
                max_waves = min(active_count, min_waves + extra_wave_window)
            wave_budgets = list(range(min_waves, max_waves + 1))
            budget_count = len(wave_budgets)
            remaining_slots = [
                max(0, wave_budget * num_cores - active_count)
                for wave_budget in wave_budgets
            ]
            gain_scan_ops = active_count * sum(remaining_slots)
            final_team_ops = active_count * budget_count
            return {
                "base": self.coefficients_ns["greedy_base"],
                "cost_lookup_ops": 2 * gain_scan_ops + final_team_ops,
                "linear_scan_ops": final_team_ops,
                "wave_pack_ops": final_team_ops,
                "gain_scan_ops": gain_scan_ops,
                "sort_compare_ops": active_count * log_active * budget_count,
                "budget_eval_ops": budget_count,
            }

        if kind == PlanKind.ENUMERATE_CORE_GROUPS:
            shape_count = (
                core_group_shape_count
                if core_group_shape_count is not None
                else count_core_group_shapes(num_cores)
            )
            enum_work = active_count * shape_count
            return {
                "base": self.coefficients_ns["groups_base"],
                "cost_lookup_ops": enum_work,
                "linear_scan_ops": enum_work,
                "wave_pack_ops": enum_work,
                "gain_scan_ops": 0,
                "sort_compare_ops": active_count * log_active + shape_count * log_active,
                "budget_eval_ops": shape_count,
            }

        raise ValueError(f"unknown plan kind: {kind}")

    def metadata_for(
        self,
        kind: PlanKind,
        active_count: int,
        num_cores: int,
        extra_wave_window: int = 8,
        core_group_shape_count: Optional[int] = None,
    ) -> Dict[str, object]:
        if self.source == "measured":
            return {"planner_cost_source": "measured_python_wall_time"}
        if kind in self.override_costs_ns:
            return {
                "planner_cost_source": self.source,
                "planner_cost_override_ns": self.override_costs_ns[kind],
            }
        if self.source in {"native_table", "profile"}:
            return {
                "planner_cost_source": "native_table",
                "planner_cost_profile": self.profile_source,
                "planner_cost_profile_metric": self.profile_metric,
                "planner_cost_profile_ns": self.profile_estimate_ns(
                    kind,
                    active_count,
                    num_cores,
                ),
            }
        if self.source == "complexity":
            return {
                "planner_cost_source": "complexity",
                "planner_cost_profile": self.profile_source,
                "planner_cost_complexity_features": self.complexity_features(
                    kind=kind,
                    active_count=active_count,
                    num_cores=num_cores,
                    extra_wave_window=extra_wave_window,
                    core_group_shape_count=core_group_shape_count,
                ),
                "planner_cost_coefficients_ns": self.coefficients_ns,
            }
        return {
            "planner_cost_source": self.source,
            "planner_cost_model_ns": self.static_costs_ns[kind],
        }


def active_experts_from_routes(routes: Sequence[int]) -> List[ExpertWork]:
    active = [
        ExpertWork(expert_id=expert_id, routes=int(route_count))
        for expert_id, route_count in enumerate(routes)
        if route_count > 0
    ]
    return sorted(active, key=lambda work: (-work.routes, work.expert_id))


def make_team(work: ExpertWork, threads: int, cost_model: ExpertCostModel) -> Team:
    return Team(
        expert_id=work.expert_id,
        routes=work.routes,
        threads=threads,
        estimated_time_ns=cost_model.estimate_ns(work.routes, threads),
    )


def enumerate_core_group_shapes(
    num_cores: int,
    max_shapes: int = MAX_CORE_GROUP_SHAPES,
) -> List[Tuple[int, ...]]:
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")
    if max_shapes <= 0:
        raise ValueError("max_shapes must be positive")

    shapes: List[Tuple[int, ...]] = []

    def visit(remaining: int, max_next: int, current: List[int]) -> None:
        if remaining == 0:
            shapes.append(tuple(current))
            return
        for value in range(min(max_next, remaining), 0, -1):
            current.append(value)
            visit(remaining - value, value, current)
            current.pop()

    visit(num_cores, num_cores, [])
    shapes = sorted(shapes, key=lambda shape: (len(shape), tuple(-item for item in shape)))
    if len(shapes) <= max_shapes:
        return shapes

    keep: List[Tuple[int, ...]] = []
    seen = set()

    def add(shape: Tuple[int, ...]) -> None:
        if shape in seen:
            return
        seen.add(shape)
        keep.append(shape)

    # Always preserve common uniform groupings and single-expert full-core mode.
    add((num_cores,))
    for group_size in range(num_cores, 0, -1):
        if num_cores % group_size == 0:
            add(tuple([group_size] * (num_cores // group_size)))

    for shape in shapes:
        add(shape)
        if len(keep) >= max_shapes:
            break

    return keep


def count_core_group_shapes(num_cores: int) -> int:
    return len(enumerate_core_group_shapes(num_cores))


def build_waves_from_ordered_teams(
    ordered_teams: Sequence[Team],
    num_cores: int,
) -> List[Wave]:
    waves: List[Wave] = []
    current: List[Team] = []
    used_threads = 0

    for team in ordered_teams:
        if team.threads > num_cores:
            raise ValueError(
                f"team uses {team.threads} threads, exceeds core budget {num_cores}"
            )
        if current and used_threads + team.threads > num_cores:
            waves.append(make_wave(len(waves), current))
            current = []
            used_threads = 0
        current.append(team)
        used_threads += team.threads

    if current:
        waves.append(make_wave(len(waves), current))

    return waves


def build_waves_for_core_group_shape(
    active: Sequence[ExpertWork],
    group_shape: Sequence[int],
    cost_model: ExpertCostModel,
) -> List[Wave]:
    if not group_shape:
        raise ValueError("core group shape must not be empty")
    if any(threads <= 0 for threads in group_shape):
        raise ValueError(f"invalid core group shape: {group_shape}")

    ordered = sorted(active, key=lambda work: (-work.routes, work.expert_id))
    waves: List[Wave] = []
    slots = list(group_shape)
    for wave_start in range(0, len(ordered), len(slots)):
        chunk = ordered[wave_start : wave_start + len(slots)]
        teams = [
            make_team(work, threads, cost_model)
            for work, threads in zip(chunk, slots)
        ]
        waves.append(make_wave(len(waves), teams))
    return waves


def make_wave(wave_id: int, teams: Sequence[Team]) -> Wave:
    if not teams:
        raise ValueError("empty waves are invalid")
    return Wave(
        wave_id=wave_id,
        teams=list(teams),
        estimated_wave_time_ns=max(team.estimated_time_ns for team in teams),
    )


def build_waves_from_sorted_token_balanced_queues(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
) -> Tuple[List[Wave], List[int], List[List[int]]]:
    if num_cores <= 0:
        raise ValueError("num_cores must be positive")

    queues: List[List[Team]] = [[] for _ in range(num_cores)]
    route_loads = [0 for _ in range(num_cores)]
    ordered = sorted(active, key=lambda work: (-work.routes, work.expert_id))

    for work in ordered:
        core = min(range(num_cores), key=lambda idx: (route_loads[idx], idx))
        queues[core].append(make_team(work, 1, cost_model))
        route_loads[core] += work.routes

    waves: List[Wave] = []
    max_queue_depth = max((len(queue) for queue in queues), default=0)
    for depth in range(max_queue_depth):
        teams = [
            queue[depth]
            for queue in queues
            if depth < len(queue)
        ]
        if teams:
            waves.append(make_wave(len(waves), teams))

    expert_queues = [
        [team.expert_id for team in queue]
        for queue in queues
    ]
    return waves, route_loads, expert_queues


def plan_fixed_global_threads(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model: PlannerCostModel,
) -> Plan:
    start_ns = time.perf_counter_ns()
    teams = [make_team(work, 1, cost_model) for work in active]
    waves = build_waves_from_ordered_teams(teams, num_cores)
    measured_plan_cost_ns = time.perf_counter_ns() - start_ns
    estimated_plan_cost_ns = planner_cost_model.estimate_ns(
        PlanKind.FIXED_GLOBAL_THREADS,
        measured_plan_cost_ns,
        active_count=len(active),
        num_cores=num_cores,
    )
    metadata = {
        "planner": "offline_simulator.py",
        "cost_model": cost_model.source,
        "cost_metric": cost_model.metric,
        "model_note": "one-thread expert waves used as fixed-thread baseline",
    }
    metadata.update(
        planner_cost_model.metadata_for(
            PlanKind.FIXED_GLOBAL_THREADS,
            active_count=len(active),
            num_cores=num_cores,
        )
    )
    return Plan(
        kind=PlanKind.FIXED_GLOBAL_THREADS,
        num_cores=num_cores,
        active_experts=list(active),
        waves=waves,
        measured_plan_cost_ns=measured_plan_cost_ns,
        estimated_plan_cost_ns=estimated_plan_cost_ns,
        exactness_scope="baseline",
        metadata=metadata,
    )


def plan_sorted_token_balanced_1t(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model: PlannerCostModel,
) -> Plan:
    start_ns = time.perf_counter_ns()
    waves, core_route_loads, core_expert_queues = (
        build_waves_from_sorted_token_balanced_queues(
            active,
            num_cores,
            cost_model,
        )
    )
    measured_plan_cost_ns = time.perf_counter_ns() - start_ns
    estimated_plan_cost_ns = planner_cost_model.estimate_ns(
        PlanKind.SORTED_TOKEN_BALANCED_1T,
        measured_plan_cost_ns,
        active_count=len(active),
        num_cores=num_cores,
    )
    metadata = {
        "planner": "offline_simulator.py",
        "cost_model": cost_model.source,
        "cost_metric": cost_model.metric,
        "model_note": (
            "sorted-routes LPT baseline: one-thread experts assigned to "
            "the lightest logical core queue by routed-token count"
        ),
        "assignment": "largest_routes_to_lightest_core_queue",
        "core_route_loads": core_route_loads,
        "core_expert_queues": core_expert_queues,
    }
    metadata.update(
        planner_cost_model.metadata_for(
            PlanKind.SORTED_TOKEN_BALANCED_1T,
            active_count=len(active),
            num_cores=num_cores,
        )
    )
    return Plan(
        kind=PlanKind.SORTED_TOKEN_BALANCED_1T,
        num_cores=num_cores,
        active_experts=list(active),
        waves=waves,
        measured_plan_cost_ns=measured_plan_cost_ns,
        estimated_plan_cost_ns=estimated_plan_cost_ns,
        exactness_scope="heuristic_baseline",
        metadata=metadata,
    )


def plan_uniform_waves(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model: PlannerCostModel,
) -> Plan:
    start_ns = time.perf_counter_ns()
    best: Optional[Tuple[int, List[Wave]]] = None

    for threads_per_expert in range(1, num_cores + 1):
        teams = [
            make_team(work, threads_per_expert, cost_model)
            for work in active
            if threads_per_expert <= num_cores
        ]
        waves = build_waves_from_ordered_teams(teams, num_cores)
        execute_ns = sum(wave.estimated_wave_time_ns for wave in waves)
        if best is None or execute_ns < best[0]:
            best = (execute_ns, waves)

    if best is None:
        raise ValueError("unable to build uniform plan")

    measured_plan_cost_ns = time.perf_counter_ns() - start_ns
    estimated_plan_cost_ns = planner_cost_model.estimate_ns(
        PlanKind.UNIFORM_WAVES,
        measured_plan_cost_ns,
        active_count=len(active),
        num_cores=num_cores,
    )
    metadata = {
        "planner": "offline_simulator.py",
        "cost_model": cost_model.source,
        "cost_metric": cost_model.metric,
        "enumerated_threads_per_expert": list(range(1, num_cores + 1)),
    }
    metadata.update(
        planner_cost_model.metadata_for(
            PlanKind.UNIFORM_WAVES,
            active_count=len(active),
            num_cores=num_cores,
        )
    )
    return Plan(
        kind=PlanKind.UNIFORM_WAVES,
        num_cores=num_cores,
        active_experts=list(active),
        waves=best[1],
        measured_plan_cost_ns=measured_plan_cost_ns,
        estimated_plan_cost_ns=estimated_plan_cost_ns,
        exactness_scope="exact_within_enumerated_space",
        metadata=metadata,
    )


def plan_enumerate_core_groups(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model: PlannerCostModel,
) -> Plan:
    start_ns = time.perf_counter_ns()
    shapes = enumerate_core_group_shapes(num_cores)
    best: Optional[Tuple[int, Tuple[int, ...], List[Wave]]] = None

    for shape in shapes:
        waves = build_waves_for_core_group_shape(active, shape, cost_model)
        execute_ns = sum(wave.estimated_wave_time_ns for wave in waves)
        if best is None or execute_ns < best[0]:
            best = (execute_ns, shape, waves)

    if best is None:
        raise ValueError("unable to build core group enumeration plan")

    measured_plan_cost_ns = time.perf_counter_ns() - start_ns
    estimated_plan_cost_ns = planner_cost_model.estimate_ns(
        PlanKind.ENUMERATE_CORE_GROUPS,
        measured_plan_cost_ns,
        active_count=len(active),
        num_cores=num_cores,
        core_group_shape_count=len(shapes),
    )
    metadata = {
        "planner": "offline_simulator.py",
        "cost_model": cost_model.source,
        "cost_metric": cost_model.metric,
        "enumerated_core_group_shapes": [list(shape) for shape in shapes],
        "selected_core_group_shape": list(best[1]),
        "core_group_shape_count": len(shapes),
        "max_core_group_shapes": MAX_CORE_GROUP_SHAPES,
        "assignment": "largest_experts_fill_descending_group_slots",
    }
    metadata.update(
        planner_cost_model.metadata_for(
            PlanKind.ENUMERATE_CORE_GROUPS,
            active_count=len(active),
            num_cores=num_cores,
            core_group_shape_count=len(shapes),
        )
    )
    return Plan(
        kind=PlanKind.ENUMERATE_CORE_GROUPS,
        num_cores=num_cores,
        active_experts=list(active),
        waves=best[2],
        measured_plan_cost_ns=measured_plan_cost_ns,
        estimated_plan_cost_ns=estimated_plan_cost_ns,
        exactness_scope="heuristic",
        metadata=metadata,
    )


def plan_greedy_marginal_gain(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model: PlannerCostModel,
    extra_wave_window: int = 8,
) -> Plan:
    start_ns = time.perf_counter_ns()

    if not active:
        raise ValueError("active expert list must not be empty")

    min_waves = math.ceil(len(active) / num_cores)
    max_waves = min(len(active), min_waves + extra_wave_window)
    best_waves: Optional[List[Wave]] = None
    best_execute_ns: Optional[int] = None
    best_wave_budget: Optional[int] = None

    for wave_budget in range(min_waves, max_waves + 1):
        thread_counts = allocate_threads_by_marginal_gain(
            active=active,
            num_cores=num_cores,
            wave_budget=wave_budget,
            cost_model=cost_model,
        )
        teams = [
            make_team(work, thread_counts[work.expert_id], cost_model)
            for work in active
        ]
        teams.sort(key=lambda team: (-team.estimated_time_ns, -team.threads, team.expert_id))
        waves = first_fit_decreasing(teams, num_cores)
        execute_ns = sum(wave.estimated_wave_time_ns for wave in waves)
        if best_execute_ns is None or execute_ns < best_execute_ns:
            best_execute_ns = execute_ns
            best_waves = waves
            best_wave_budget = wave_budget

    if best_waves is None or best_wave_budget is None:
        raise ValueError("unable to build greedy plan")

    measured_plan_cost_ns = time.perf_counter_ns() - start_ns
    estimated_plan_cost_ns = planner_cost_model.estimate_ns(
        PlanKind.GREEDY_MARGINAL_GAIN,
        measured_plan_cost_ns,
        active_count=len(active),
        num_cores=num_cores,
        extra_wave_window=extra_wave_window,
    )
    metadata = {
        "planner": "offline_simulator.py",
        "cost_model": cost_model.source,
        "cost_metric": cost_model.metric,
        "min_waves": min_waves,
        "selected_wave_budget": best_wave_budget,
        "extra_wave_window": extra_wave_window,
    }
    metadata.update(
        planner_cost_model.metadata_for(
            PlanKind.GREEDY_MARGINAL_GAIN,
            active_count=len(active),
            num_cores=num_cores,
            extra_wave_window=extra_wave_window,
        )
    )
    return Plan(
        kind=PlanKind.GREEDY_MARGINAL_GAIN,
        num_cores=num_cores,
        active_experts=list(active),
        waves=best_waves,
        measured_plan_cost_ns=measured_plan_cost_ns,
        estimated_plan_cost_ns=estimated_plan_cost_ns,
        exactness_scope="heuristic",
        metadata=metadata,
    )


def allocate_threads_by_marginal_gain(
    active: Sequence[ExpertWork],
    num_cores: int,
    wave_budget: int,
    cost_model: ExpertCostModel,
) -> Dict[int, int]:
    total_thread_slots = wave_budget * num_cores
    if total_thread_slots < len(active):
        raise ValueError("wave budget cannot fit all active experts")

    thread_counts = {work.expert_id: 1 for work in active}
    remaining_slots = total_thread_slots - len(active)

    while remaining_slots > 0:
        best_work: Optional[ExpertWork] = None
        best_gain = 0

        for work in active:
            current_threads = thread_counts[work.expert_id]
            if current_threads >= num_cores:
                continue
            current_ns = cost_model.estimate_ns(work.routes, current_threads)
            next_ns = cost_model.estimate_ns(work.routes, current_threads + 1)
            gain = current_ns - next_ns
            if gain > best_gain:
                best_gain = gain
                best_work = work

        if best_work is None or best_gain <= 0:
            break

        thread_counts[best_work.expert_id] += 1
        remaining_slots -= 1

    return thread_counts


def first_fit_decreasing(teams: Sequence[Team], num_cores: int) -> List[Wave]:
    bins: List[Tuple[int, List[Team]]] = []

    for team in teams:
        placed = False
        for bin_idx, (used_threads, bin_teams) in enumerate(bins):
            if used_threads + team.threads <= num_cores:
                bin_teams.append(team)
                bins[bin_idx] = (used_threads + team.threads, bin_teams)
                placed = True
                break
        if not placed:
            bins.append((team.threads, [team]))

    return [make_wave(wave_id, bin_teams) for wave_id, (_, bin_teams) in enumerate(bins)]


def workload_stats(routes: Sequence[int]) -> Dict[str, float]:
    active_counts = [count for count in routes if count > 0]
    total = sum(active_counts)
    num_experts = len(routes)
    if not active_counts or total == 0:
        return {
            "active_experts": 0,
            "active_fraction": 0.0,
            "total_routes": 0,
            "routes_max": 0,
            "routes_mean_all": 0.0,
            "routes_mean_active": 0.0,
            "routes_std_active": 0.0,
            "entropy": 0.0,
            "gini": 0.0,
            "maxvio": 0.0,
        }

    mean = total / len(active_counts)
    mean_all = total / num_experts
    variance = sum((count - mean) ** 2 for count in active_counts) / len(active_counts)
    entropy = -sum((count / total) * math.log(count / total) for count in active_counts)
    if abs(entropy) < 1e-12:
        entropy = 0.0
    sorted_counts = sorted(active_counts)
    weighted_sum = sum((idx + 1) * count for idx, count in enumerate(sorted_counts))
    gini = (2 * weighted_sum) / (len(sorted_counts) * total)
    gini -= (len(sorted_counts) + 1) / len(sorted_counts)
    if abs(gini) < 1e-12:
        gini = 0.0

    return {
        "active_experts": len(active_counts),
        "active_fraction": len(active_counts) / num_experts,
        "total_routes": total,
        "routes_max": max(active_counts),
        "routes_mean_all": mean_all,
        "routes_mean_active": mean,
        "routes_std_active": math.sqrt(variance),
        "entropy": entropy,
        "gini": gini,
        "maxvio": (max(active_counts) - mean_all) / mean_all,
    }


def generate_routes(
    distribution: str,
    num_experts: int,
    tokens: int,
    top_k: int,
    seed: int,
    zipf_alpha: float,
    active_experts: int,
    hot_experts: int,
    hot_fraction: float,
    heavy_experts: int,
    heavy_fraction: float,
    dirichlet_alpha: float,
    lognormal_sigma: float,
    cluster_experts: int,
    background_fraction: float,
) -> List[int]:
    total_routes = tokens * top_k
    if distribution == "uniform":
        return integer_allocation([1.0] * num_experts, total_routes)
    if distribution == "one_hot":
        routes = [0] * num_experts
        routes[0] = total_routes
        return routes
    if distribution == "zipf":
        weights = [1.0 / ((rank + 1) ** zipf_alpha) for rank in range(num_experts)]
        return integer_allocation(weights, total_routes)
    if distribution == "active_subset":
        active_count = clamp_count(active_experts, num_experts)
        weights = [1.0 if expert_id < active_count else 0.0 for expert_id in range(num_experts)]
        return integer_allocation(weights, total_routes)
    if distribution == "hotspot":
        hot_count = clamp_count(hot_experts, num_experts)
        hot_share = clamp_fraction(hot_fraction)
        cold_count = num_experts - hot_count
        cold_share = 1.0 - hot_share
        weights = []
        for expert_id in range(num_experts):
            if expert_id < hot_count:
                weights.append(hot_share / hot_count)
            elif cold_count > 0:
                weights.append(cold_share / cold_count)
            else:
                weights.append(0.0)
        return integer_allocation(weights, total_routes)
    if distribution == "heavy_light":
        heavy_count = clamp_count(heavy_experts, num_experts)
        heavy_share = clamp_fraction(heavy_fraction)
        light_count = num_experts - heavy_count
        weights = []
        for expert_id in range(num_experts):
            if expert_id < heavy_count:
                rank = expert_id + 1
                weights.append(heavy_share / rank)
            elif light_count > 0:
                weights.append((1.0 - heavy_share) / light_count)
            else:
                weights.append(0.0)
        return integer_allocation(weights, total_routes)
    if distribution == "dirichlet":
        if dirichlet_alpha <= 0.0:
            raise ValueError("--dirichlet-alpha must be positive")
        rng = random.Random(seed)
        weights = [rng.gammavariate(dirichlet_alpha, 1.0) for _ in range(num_experts)]
        return integer_allocation(weights, total_routes)
    if distribution == "lognormal":
        if lognormal_sigma <= 0.0:
            raise ValueError("--lognormal-sigma must be positive")
        rng = random.Random(seed)
        weights = [rng.lognormvariate(0.0, lognormal_sigma) for _ in range(num_experts)]
        return integer_allocation(weights, total_routes)
    if distribution == "domain_cluster":
        cluster_count = clamp_count(cluster_experts, num_experts)
        background_share = clamp_fraction(background_fraction)
        rng = random.Random(seed)
        cluster_start = rng.randrange(num_experts)
        cluster_ids = {
            (cluster_start + offset) % num_experts for offset in range(cluster_count)
        }
        weights = []
        for expert_id in range(num_experts):
            if expert_id in cluster_ids:
                rank = ((expert_id - cluster_start) % num_experts) + 1
                weights.append((1.0 - background_share) / rank)
            elif num_experts > cluster_count:
                weights.append(background_share / (num_experts - cluster_count))
            else:
                weights.append(0.0)
        return integer_allocation(weights, total_routes)
    if distribution == "random_balanced":
        rng = random.Random(seed)
        routes = [0] * num_experts
        for _ in range(total_routes):
            routes[rng.randrange(num_experts)] += 1
        return routes
    raise ValueError(f"unknown distribution: {distribution}")


def clamp_count(value: int, upper_bound: int) -> int:
    return max(1, min(value, upper_bound))


def clamp_fraction(value: float) -> float:
    return max(0.0, min(value, 1.0))


def integer_allocation(weights: Sequence[float], total: int) -> List[int]:
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("weights must contain at least one positive value")
    raw = [(weight / weight_sum) * total for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda idx: (raw[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def select_best_plan(plans: Iterable[Plan]) -> Plan:
    return min(
        plans,
        key=lambda plan: (
            plan.estimated_total_cost_ns,
            plan.estimated_execute_cost_ns,
            plan.kind.value,
        ),
    )


def select_auto_planners(
    routes: Sequence[int],
    num_cores: int,
) -> Tuple[List[str], str, Dict[str, float]]:
    stats = workload_stats(routes)
    active = int(stats["active_experts"])
    gini = float(stats["gini"])
    maxvio = float(stats["maxvio"])
    active_fraction = float(stats["active_fraction"])

    if active <= 0:
        raise ValueError("auto selector requires at least one active expert")

    if active <= 1:
        names = ["fixed", "balanced", "uniform", "greedy", "groups"]
        reason = "single_active_expert"
    elif active <= num_cores:
        names = ["fixed", "balanced", "uniform", "groups"]
        reason = "active_experts_fit_in_cores"
    elif gini < 0.15 and maxvio < 2.0:
        names = ["balanced"]
        reason = "balanced_load"
    elif gini < 0.65 and maxvio < 16.0:
        # AWS bridge measurements show dense moderate-skew workloads favor
        # simple one-thread-per-expert schedules over extra grouped waves.
        names = ["fixed", "balanced", "uniform"]
        reason = "moderate_skew_dense_fixed_preferred"
    elif active_fraction < 0.35 and gini < 0.70 and maxvio < 32.0:
        names = ["fixed", "balanced", "uniform", "groups"]
        reason = "sparse_active_moderate_skew"
    else:
        names = ["fixed", "balanced", "uniform", "greedy", "groups"]
        reason = "high_skew"

    selector_stats = {
        "active_experts": active,
        "active_fraction": active_fraction,
        "gini": gini,
        "maxvio": maxvio,
    }
    return names, reason, selector_stats


def normalize_planner_kind(name: str) -> PlanKind:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in PLANNER_ALIASES:
        return PLANNER_ALIASES[normalized]
    for kind in PlanKind:
        if normalized == kind.value.lower():
            return kind
    raise ValueError(f"unknown planner kind: {name}")


def parse_duration_ns(value: str) -> int:
    text = value.strip().lower()
    multipliers = {
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
    }
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            number = float(text[: -len(suffix)])
            if number < 0.0:
                raise ValueError("duration must be non-negative")
            return int(number * multiplier)
    number = int(text)
    if number < 0:
        raise ValueError("duration must be non-negative")
    return number


def load_native_planner_cost_profile(
    path: Path,
) -> Tuple[
    Dict[PlanKind, List[Tuple[int, int, int]]],
    Dict[PlanKind, int],
    Dict[str, int],
    str,
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = str(payload.get("metric", "total_native_median_ns"))
    entries = payload.get("entries", [])
    if not entries and isinstance(payload.get("table"), dict):
        expanded_entries = []
        table = payload["table"]
        for cores_text, by_planner in table.items():
            cores = int(cores_text)
            for planner_name, by_active in by_planner.items():
                for active_text, cost_ns in by_active.items():
                    expanded_entries.append(
                        {
                            "planner": planner_name,
                            "cores": cores,
                            "active_experts": int(active_text),
                            metric: int(cost_ns),
                        }
                    )
        entries = expanded_entries
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"planner cost profile has no entries: {path}")

    profile_costs: Dict[PlanKind, List[Tuple[int, int, int]]] = {}
    by_kind_values: Dict[PlanKind, List[int]] = {}
    for entry in entries:
        kind = normalize_planner_kind(str(entry["planner"]))
        active_count = int(entry.get("active_experts", entry.get("num_active", 0)))
        cores = int(entry["cores"])
        cost_ns = int(entry[metric])
        profile_costs.setdefault(kind, []).append((active_count, cores, cost_ns))
        by_kind_values.setdefault(kind, []).append(cost_ns)

    static_costs = dict(DEFAULT_PLANNER_COSTS_NS)
    for kind, values in by_kind_values.items():
        ordered = sorted(values)
        static_costs[kind] = ordered[len(ordered) // 2]

    coefficients = dict(COMPLEXITY_COST_COEFFICIENTS_NS)
    raw_coefficients = payload.get("calibrated_coefficients_ns", {})
    if isinstance(raw_coefficients, dict):
        for key, value in raw_coefficients.items():
            if key in coefficients:
                coefficients[key] = int(value)

    return profile_costs, static_costs, coefficients, metric


def build_planner_cost_model(
    source: str,
    overrides: Optional[Sequence[str]],
    profile_path: Optional[Path] = None,
) -> PlannerCostModel:
    if source == "measured":
        return PlannerCostModel(
            source="measured",
            static_costs_ns=dict(DEFAULT_PLANNER_COSTS_NS),
        )

    if source in {"native_table", "profile"}:
        if profile_path is None:
            raise ValueError(
                "--planner-cost-profile is required for "
                f"source={source}"
            )
        profile_costs, static_costs, coefficients, metric = (
            load_native_planner_cost_profile(profile_path)
        )
        return PlannerCostModel(
            source="native_table",
            static_costs_ns=static_costs,
            profile_costs_ns=profile_costs,
            profile_metric=metric,
            profile_source=str(profile_path),
            coefficients_ns=coefficients,
        )

    override_costs_ns: Dict[PlanKind, int] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                "--planner-cost must use KEY=VALUE, for example fixed=1000 or greedy=20us"
            )
        key, value = item.split("=", 1)
        kind = normalize_planner_kind(key)
        override_costs_ns[kind] = parse_duration_ns(value)

    return PlannerCostModel(
        source=source,
        static_costs_ns=dict(DEFAULT_PLANNER_COSTS_NS),
        override_costs_ns=override_costs_ns,
        coefficients_ns=dict(COMPLEXITY_COST_COEFFICIENTS_NS),
    )


def build_plans(
    active: Sequence[ExpertWork],
    num_cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model: PlannerCostModel,
    planner_names: Sequence[str],
) -> List[Plan]:
    planners = {
        "fixed": plan_fixed_global_threads,
        "balanced": plan_sorted_token_balanced_1t,
        "uniform": plan_uniform_waves,
        "greedy": plan_greedy_marginal_gain,
        "groups": plan_enumerate_core_groups,
    }
    names = list(planners) if "all" in planner_names else list(planner_names)
    unknown = [name for name in names if name not in planners]
    if unknown:
        raise ValueError(f"unknown planner(s): {', '.join(unknown)}")
    return [
        planners[name](active, num_cores, cost_model, planner_cost_model)
        for name in names
    ]


def format_ns(ns: int) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns} ns"


def planner_cost_summary(planner_cost_model: PlannerCostModel) -> str:
    if planner_cost_model.source == "complexity":
        override_text = ""
        if planner_cost_model.override_costs_ns:
            overrides = ", ".join(
                f"{kind.value}={format_ns(cost_ns)}"
                for kind, cost_ns in sorted(
                    planner_cost_model.override_costs_ns.items(),
                    key=lambda item: item[0].value,
                )
            )
            override_text = f" overrides=[{overrides}]"
        return (
            "source=complexity features=A,C,lookups,scans,sorts,waves"
            + override_text
        )

    if planner_cost_model.source == "measured":
        return "source=measured_python_wall_time"

    if planner_cost_model.source in {"native_table", "profile"}:
        return (
            "source=native_table "
            f"metric={planner_cost_model.profile_metric} "
            f"profile={planner_cost_model.profile_source}"
        )

    def configured_cost(kind: PlanKind) -> int:
        return planner_cost_model.override_costs_ns.get(
            kind,
            planner_cost_model.static_costs_ns[kind],
        )

    return (
        "source=model "
        f"fixed={format_ns(configured_cost(PlanKind.FIXED_GLOBAL_THREADS))} "
        "balanced="
        f"{format_ns(configured_cost(PlanKind.SORTED_TOKEN_BALANCED_1T))} "
        f"uniform={format_ns(configured_cost(PlanKind.UNIFORM_WAVES))} "
        f"greedy={format_ns(configured_cost(PlanKind.GREEDY_MARGINAL_GAIN))} "
        f"groups={format_ns(configured_cost(PlanKind.ENUMERATE_CORE_GROUPS))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline CPU MoE schedule simulator.",
    )
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument(
        "--distribution",
        choices=[
            "uniform",
            "random_balanced",
            "active_subset",
            "hotspot",
            "heavy_light",
            "zipf",
            "dirichlet",
            "lognormal",
            "domain_cluster",
            "one_hot",
        ],
        default="zipf",
    )
    parser.add_argument("--zipf-alpha", type=float, default=1.15)
    parser.add_argument(
        "--active-experts",
        type=int,
        default=32,
        help="Active expert count for active_subset.",
    )
    parser.add_argument(
        "--hot-experts",
        type=int,
        default=4,
        help="Hot expert count for hotspot.",
    )
    parser.add_argument(
        "--hot-fraction",
        type=float,
        default=0.70,
        help="Route fraction assigned to hot experts for hotspot.",
    )
    parser.add_argument(
        "--heavy-experts",
        type=int,
        default=8,
        help="Heavy expert count for heavy_light.",
    )
    parser.add_argument(
        "--heavy-fraction",
        type=float,
        default=0.80,
        help="Route fraction assigned to heavy experts for heavy_light.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.25,
        help="Dirichlet concentration. Values below 1.0 are skewed.",
    )
    parser.add_argument(
        "--lognormal-sigma",
        type=float,
        default=1.50,
        help="Log-normal sigma. Larger values produce stronger heavy tails.",
    )
    parser.add_argument(
        "--cluster-experts",
        type=int,
        default=32,
        help="Expert count in the hot cluster for domain_cluster.",
    )
    parser.add_argument(
        "--background-fraction",
        type=float,
        default=0.10,
        help="Route fraction outside the hot cluster for domain_cluster.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planner",
        action="append",
        choices=[
            "all",
            "auto",
            "fixed",
            "balanced",
            "uniform",
            "greedy",
            "groups",
        ],
        default=None,
        help="Planner to run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--cost-table",
        type=Path,
        default=None,
        help="Optional JSON cost table following cost_model/profile_schema.md.",
    )
    parser.add_argument(
        "--plan-cost-source",
        choices=["complexity", "model", "measured", "native_table", "profile"],
        default="complexity",
        help=(
            "Use complexity-based cost, static model cost, Python measured "
            "wall time, or a native C++ planner timing table for scoring. "
            "'profile' is kept as an alias for native_table."
        ),
    )
    parser.add_argument(
        "--planner-cost-profile",
        type=Path,
        default=None,
        help=(
            "Native planner cost profile generated by "
            "benchmarks/profile_native_planner_cost.py or a compact table "
            "generated by cost_model/build_lightweight_planner_cost.py. Used "
            "only with --plan-cost-source native_table/profile."
        ),
    )
    parser.add_argument(
        "--planner-cost",
        action="append",
        default=None,
        help=(
            "Override planner cost in KEY=VALUE form. VALUE is ns by default and "
            "also accepts us/ms suffixes, for example fixed=1000 or greedy=20us."
        ),
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print full JSON result instead of a compact summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.cores <= 0:
        raise ValueError("--cores must be positive")

    cost_model = (
        ExpertCostModel.from_json(args.cost_table)
        if args.cost_table is not None
        else ExpertCostModel.synthetic()
    )
    planner_cost_model = build_planner_cost_model(
        source=args.plan_cost_source,
        overrides=args.planner_cost,
        profile_path=args.planner_cost_profile,
    )
    routes = generate_routes(
        distribution=args.distribution,
        num_experts=args.num_experts,
        tokens=args.tokens,
        top_k=args.top_k,
        seed=args.seed,
        zipf_alpha=args.zipf_alpha,
        active_experts=args.active_experts,
        hot_experts=args.hot_experts,
        hot_fraction=args.hot_fraction,
        heavy_experts=args.heavy_experts,
        heavy_fraction=args.heavy_fraction,
        dirichlet_alpha=args.dirichlet_alpha,
        lognormal_sigma=args.lognormal_sigma,
        cluster_experts=args.cluster_experts,
        background_fraction=args.background_fraction,
    )
    active = active_experts_from_routes(routes)
    planner_names = args.planner or ["all"]
    auto_selector = None
    if "auto" in planner_names:
        if len(planner_names) > 1:
            raise ValueError("--planner auto cannot be combined with other --planner values")
        enabled_planners, reason, selector_stats = select_auto_planners(routes, args.cores)
        planner_names = enabled_planners
        auto_selector = {
            "enabled_planners": enabled_planners,
            "reason": reason,
            "stats": selector_stats,
        }
    plans = build_plans(
        active,
        args.cores,
        cost_model,
        planner_cost_model,
        planner_names,
    )
    best = select_best_plan(plans)

    result = {
        "input": {
            "num_experts": args.num_experts,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "num_cores": args.cores,
            "distribution": args.distribution,
            "zipf_alpha": args.zipf_alpha,
            "active_experts_arg": args.active_experts,
            "hot_experts": args.hot_experts,
            "hot_fraction": args.hot_fraction,
            "heavy_experts": args.heavy_experts,
            "heavy_fraction": args.heavy_fraction,
            "dirichlet_alpha": args.dirichlet_alpha,
            "lognormal_sigma": args.lognormal_sigma,
            "cluster_experts": args.cluster_experts,
            "background_fraction": args.background_fraction,
            "plan_cost_source": args.plan_cost_source,
            "planner_cost_profile": (
                str(args.planner_cost_profile)
                if args.planner_cost_profile is not None
                else None
            ),
            "planner_cost_model": {
                "source": planner_cost_model.source,
                "static_costs_ns": {
                    kind.value: planner_cost_model.static_costs_ns[kind]
                    for kind in PlanKind
                },
                "override_costs_ns": {
                    kind.value: cost_ns
                    for kind, cost_ns in planner_cost_model.override_costs_ns.items()
                },
                "coefficients_ns": planner_cost_model.coefficients_ns,
            },
            "seed": args.seed,
        },
        "workload_stats": workload_stats(routes),
        "auto_selector": auto_selector,
        "best_plan_kind": best.kind.value,
        "candidates": [plan.to_dict() for plan in plans],
    }

    if args.dump_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    stats = result["workload_stats"]
    print(
        "workload: "
        f"active={stats['active_experts']} "
        f"active_frac={stats['active_fraction']:.3f} "
        f"total_routes={stats['total_routes']} "
        f"max={stats['routes_max']} "
        f"maxvio={stats['maxvio']:.2f} "
        f"mean={stats['routes_mean_active']:.2f} "
        f"std={stats['routes_std_active']:.2f} "
        f"entropy={stats['entropy']:.3f} "
        f"gini={stats['gini']:.3f}"
    )
    print(f"cost_model: {cost_model.source} metric={cost_model.metric}")
    print(f"planner_cost: {planner_cost_summary(planner_cost_model)}")
    if auto_selector is not None:
        print(
            "auto_selector: "
            f"reason={auto_selector['reason']} "
            f"enabled={','.join(auto_selector['enabled_planners'])}"
        )
    print("candidates:")
    for plan in sorted(plans, key=lambda item: item.estimated_total_cost_ns):
        print(
            "  "
            f"{plan.kind.value:<24} "
            f"waves={len(plan.waves):<3} "
            f"plan={format_ns(plan.estimated_plan_cost_ns):>10} "
            f"py_plan={format_ns(plan.measured_plan_cost_ns):>10} "
            f"execute={format_ns(plan.estimated_execute_cost_ns):>10} "
            f"total={format_ns(plan.estimated_total_cost_ns):>10}"
        )
    print(f"best: {best.kind.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
