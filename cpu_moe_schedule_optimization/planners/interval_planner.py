"""Static interval-DAG planner with exact-profile and policy-aware search."""

from __future__ import annotations

import importlib
import math
import os
import sys
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402
from profile_catalog import ProfileCompatibilityError  # noqa: E402
from stage_window_policy import TaskStageWindowPolicy  # noqa: E402


_ASYNC_PLAN_VERSION = 2
_ASYNC_EXECUTION_STRICT = "strict"
_ASYNC_EXECUTION_TAIL_POOL = "tail_pool"
_ASYNC_PLACEMENT_FIXED = 0
_ASYNC_PLACEMENT_TAIL_POOL = 1
_ASYNC_STAGE_EXPERT = 0
_ASYNC_RESIZE_NONE = 0
_ASYNC_FULL_EXPERT_RANGE = 0
_AUTO_TAIL_POOL_WIDTHS = frozenset((1, 2, 4))
_AUTO_TAIL_POOL_THRESHOLDS = (1, 2, 4, 8, 12)
_AUTO_TAIL_POOL_MIN_HEAD_SHAPES = 2


class PlannerPolicy(Protocol):
    w13_split: bool
    w13_split_chunks: int
    weight_window_bytes: int
    w13_window_ranges: int
    w2_window_ranges: int
    intermediate_size: int
    mode: str
    degree: int
    llc_bytes_per_rank: int

    def key_without_kernel_policy(self) -> tuple[object, ...]: ...

    def kernel_policy_key(self) -> tuple[object, ...]: ...


class PlannerCostModel(Protocol):
    schema_version: int
    supported_shapes: Sequence[Sequence[int]]
    supported_widths: Sequence[int]
    profile_path: Path
    policy: PlannerPolicy | None
    max_stage_bytes: int
    has_full_workload_anchors: bool
    local_experts: int
    profile_runs: int

    def T_iso(self, routes: int, threads: int) -> float: ...

    def dag_makespan(self, tasks) -> float: ...

    def supports_shape(self, shape) -> bool: ...

    def relative_uncertainty(self, routes: int, shape) -> float: ...

    def relative_full_call_uncertainty(self, routes: int, shape) -> float: ...

    def profiled_full_call_time(self, routes: int, shape) -> float: ...

    def window_bytes_per_worker(self, threads: int) -> int: ...


def _partitions(n: int, parts: Sequence[int]) -> List[Tuple[int, ...]]:
    parts = sorted(parts, reverse=True)
    output: List[Tuple[int, ...]] = []

    def recurse(remaining: int, max_part: int, current: List[int]) -> None:
        if remaining == 0:
            output.append(tuple(current))
            return
        for part in parts:
            if part <= max_part and part <= remaining:
                recurse(remaining - part, part, current + [part])

    recurse(n, max(parts), [])
    return output


def _default_widths(num_cores: int) -> tuple[int, ...]:
    widths: list[int] = []
    value = 1
    while value <= num_cores:
        widths.append(value)
        value *= 2
    return tuple(widths)


def _model_candidate_shapes(
    model: PlannerCostModel,
    num_cores: int,
) -> tuple[tuple[int, ...], ...]:
    candidate_shapes = getattr(model, "candidate_shapes", None)
    if callable(candidate_shapes):
        return tuple(tuple(int(value) for value in shape) for shape in candidate_shapes(num_cores))
    if model.schema_version >= 2:
        return tuple(tuple(int(value) for value in shape) for shape in model.supported_shapes)
    return ()


class IntervalPlanner:
    def __init__(
        self,
        model: PlannerCostModel,
        num_cores: int = 8,
        widths: Sequence[int] | None = None,
        *,
        cpu_ids: Sequence[int] | None = None,
        shapes: Sequence[Sequence[int]] | None = None,
        native_cold_planner: bool | None = None,
        planner_threads: int | None = None,
        task_stage_window_policy: TaskStageWindowPolicy | None = None,
    ):
        self.model = model
        self.num_cores = int(num_cores)
        model_widths = getattr(model, "supported_widths", None)
        self.widths = tuple(widths or model_widths or _default_widths(self.num_cores))
        self.cpu_ids = tuple(int(cpu) for cpu in (cpu_ids if cpu_ids is not None else range(num_cores)))
        if len(self.cpu_ids) != self.num_cores or len(set(self.cpu_ids)) != num_cores:
            raise ValueError("cpu_ids must contain num_cores unique physical CPUs")

        if shapes is not None:
            candidates = [tuple(int(value) for value in shape) for shape in shapes]
        elif model.schema_version >= 2:
            candidates = list(_model_candidate_shapes(model, self.num_cores))
        else:
            candidates = _partitions(self.num_cores, self.widths)
        self.shapes = tuple(
            shape
            for shape in candidates
            if sum(shape) == self.num_cores and all(width in self.widths for width in shape)
        )
        if not self.shapes:
            raise ProfileCompatibilityError(
                f"no supported shape covers {self.num_cores} cores with widths {self.widths}"
            )
        self.task_stage_window_policy = task_stage_window_policy
        self._native_planner = self._create_native_planner(
            native_cold_planner=native_cold_planner,
            planner_threads=planner_threads,
        )

    def _create_native_planner(
        self,
        *,
        native_cold_planner: bool | None,
        planner_threads: int | None,
    ):
        configured = os.environ.get("FUSED_CPP_MOE_NATIVE_COLD_PLANNER")
        explicitly_requested = native_cold_planner is True
        if native_cold_planner is None:
            if configured is None:
                native_cold_planner = True
            else:
                native_cold_planner = configured.strip().lower() not in {
                    "",
                    "0",
                    "false",
                    "off",
                    "no",
                }
                explicitly_requested = native_cold_planner
        if not native_cold_planner:
            return None
        exporter = getattr(self.model, "native_interval_planner_payload", None)
        if not callable(exporter):
            if explicitly_requested:
                raise RuntimeError(
                    "native cold planner was requested but the cost model "
                    "does not support native calibration export"
                )
            return None
        if planner_threads is None:
            planner_threads = int(os.environ.get("FUSED_CPP_MOE_PLANNER_THREADS", "0"))
        if planner_threads < 0:
            raise ValueError(f"planner_threads must be non-negative, got {planner_threads}")
        try:
            extension = importlib.import_module("fused_cpp._C")
            native_type = extension.NativeIntervalPlanner
        except (ImportError, AttributeError):
            if explicitly_requested:
                raise RuntimeError(
                    "native cold planner was requested but fused_cpp._C.NativeIntervalPlanner is unavailable"
                )
            return None
        return native_type(
            self.num_cores,
            list(self.widths),
            [list(shape) for shape in self.shapes],
            exporter(),
            planner_threads,
        )

    def _lanes(self, shape: Tuple[int, ...]):
        begins, core = [], 0
        for width in shape:
            begins.append(core)
            core += width
        return list(zip(begins, shape))

    def _assign(self, experts, lanes):
        lane_count = len(lanes)
        load = [0.0] * lane_count
        lane_experts: List[List[int]] = [[] for _ in range(lane_count)]
        order = sorted(range(len(experts)), key=lambda index: -experts[index][1])
        for index in order:
            _, routes = experts[index]
            lane = min(
                range(lane_count),
                key=lambda candidate: load[candidate] + self.model.T_iso(routes, lanes[candidate][1]),
            )
            lane_experts[lane].append(index)
            load[lane] += self.model.T_iso(routes, lanes[lane][1])
        return lane_experts

    def _build_tasks(self, experts, lanes, lane_experts):
        tasks = []
        for lane, expert_indices in enumerate(lane_experts):
            core_begin, width = lanes[lane]
            previous = None
            for index in expert_indices:
                expert_id, routes = experts[index]
                dependencies = [previous] if previous is not None else []
                tasks.append((expert_id, routes, core_begin, width, dependencies))
                previous = len(tasks) - 1
        return tasks

    def _score(self, tasks) -> float:
        return self.model.dag_makespan([(routes, threads, deps) for _, routes, _, threads, deps in tasks])

    def _uncertainty(
        self,
        experts,
        shape,
        makespan: float,
        *,
        use_full_workload_anchor: bool = True,
    ) -> float:
        if self.model.schema_version < 2:
            return 0.0
        if use_full_workload_anchor and self._uses_full_workload_anchor(experts):
            relative = self.model.relative_full_call_uncertainty(experts[0][1], shape)
            return makespan * relative / math.sqrt(self.model.profile_runs)
        relative = max(self.model.relative_uncertainty(routes, shape) for _, routes in experts)
        waves = max(1, math.ceil(len(experts) / len(shape)))
        return makespan * relative / math.sqrt(waves * self.model.profile_runs)

    def active_working_set_bytes(self, shape, tasks=None) -> int:
        if tasks is None:
            active_lanes = len(shape)
        else:
            active_lanes = len({(core, threads) for _, _, core, threads, _ in tasks})
        return active_lanes * self.model.max_stage_bytes

    def window_bytes_per_worker(self, shape, tasks=None) -> tuple[int, ...]:
        if tasks is None:
            active_widths = tuple(int(width) for width in shape)
        else:
            active_widths = tuple(
                threads
                for _, threads in sorted(
                    {(core, threads) for _, _, core, threads, _ in tasks},
                )
            )
        return tuple(self.model.window_bytes_per_worker(width) for width in active_widths)

    def _uses_full_workload_anchor(self, experts) -> bool:
        return (
            self.model.has_full_workload_anchors
            and len(experts) == self.model.local_experts
            and bool(experts)
            and all(routes == experts[0][1] for _, routes in experts)
        )

    def score_shape(self, experts, shape):
        signature = tuple(int(value) for value in shape)
        if self.model.schema_version >= 2 and not self.model.supports_shape(signature):
            raise ProfileCompatibilityError(f"shape {signature} is not supported by {self.model.profile_path.name}")
        lanes = self._lanes(signature)
        tasks = self._build_tasks(experts, lanes, self._assign(experts, lanes))
        if self._uses_full_workload_anchor(experts):
            return self.model.profiled_full_call_time(experts[0][1], signature), tasks
        return self._score(tasks), tasks

    def _candidate(self, experts, shape) -> dict:
        makespan, tasks = self.score_shape(experts, shape)
        uncertainty = self._uncertainty(experts, shape, makespan)
        resource_groups = len({(core, threads) for _, _, core, threads, _ in tasks})
        return {
            "shape": tuple(shape),
            "execution_mode": _ASYNC_EXECUTION_STRICT,
            "tail_pool_threads": None,
            "tail_pool_max_routes": None,
            "tail_pool_tasks": 0,
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": tasks,
            "active_working_set_bytes": self.active_working_set_bytes(shape, tasks),
            "window_bytes_per_worker": self.window_bytes_per_worker(shape, tasks),
            "resource_groups": resource_groups,
        }

    @staticmethod
    def _select(candidates: list[dict]) -> dict:
        candidates.sort(key=lambda candidate: candidate["makespan_ns"])
        fastest = candidates[0]
        fastest_lower = fastest["makespan_ns"] - fastest["uncertainty_ns"]
        fastest_upper = fastest["pessimistic_ns"]
        overlapping = [
            candidate
            for candidate in candidates
            if candidate["makespan_ns"] - candidate["uncertainty_ns"] <= fastest_upper
            and candidate["pessimistic_ns"] >= fastest_lower
        ]
        return min(
            overlapping,
            key=lambda candidate: (
                candidate["active_working_set_bytes"],
                candidate.get("resource_groups", len(candidate["shape"])),
                candidate.get("execution_mode", _ASYNC_EXECUTION_STRICT) != _ASYNC_EXECUTION_STRICT,
                candidate["pessimistic_ns"],
                candidate["makespan_ns"],
            ),
        )

    @staticmethod
    def _peak_active_tasks(intervals: Sequence[tuple[float, float]]) -> int:
        events = [(begin, 1) for begin, _ in intervals]
        events.extend((end, -1) for _, end in intervals)
        active = 0
        peak = 0
        for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
            active += delta
            peak = max(peak, active)
        return peak

    def _tail_pool_layout(
        self,
        tasks,
        *,
        pool_threads: int,
        max_pooled_routes: int,
    ) -> tuple[list[bool], list[list[int]]]:
        pool_threads = int(pool_threads)
        max_pooled_routes = int(max_pooled_routes)
        if pool_threads <= 0 or pool_threads > self.num_cores:
            raise ValueError(f"pool_threads must be in [1, {self.num_cores}], got {pool_threads}")
        if self.num_cores % pool_threads != 0:
            raise ValueError(
                f"pool_threads must divide num_cores: pool_threads={pool_threads}, num_cores={self.num_cores}"
            )
        if max_pooled_routes <= 0:
            raise ValueError(f"max_pooled_routes must be positive, got {max_pooled_routes}")

        pooled = [routes <= max_pooled_routes for _, routes, _, _, _ in tasks]
        if not any(pooled):
            raise ValueError(f"tail_pool found no task with routes <= {max_pooled_routes}")

        resolved_dependencies: list[list[int]] = []
        for task_id, (_, _, core_begin, threads, dependencies) in enumerate(tasks):
            if pooled[task_id]:
                resolved_dependencies.append([])
                continue
            if core_begin % pool_threads != 0 or threads % pool_threads != 0:
                raise ValueError(
                    "fixed task intervals must align to pool_threads: "
                    f"task={task_id}, core_begin={core_begin}, "
                    f"threads={threads}, pool_threads={pool_threads}"
                )
            frontier = list(dependencies)
            fixed_dependencies: set[int] = set()
            while frontier:
                dependency = frontier.pop()
                if dependency < 0 or dependency >= task_id:
                    raise ValueError(
                        f"task dependencies must refer to earlier task ids: task={task_id}, dependency={dependency}"
                    )
                if pooled[dependency]:
                    frontier.extend(tasks[dependency][4])
                else:
                    fixed_dependencies.add(dependency)
            resolved_dependencies.append(sorted(fixed_dependencies))
        return pooled, resolved_dependencies

    def _tail_pool_simulation(
        self,
        tasks,
        *,
        pool_threads: int,
        max_pooled_routes: int,
    ) -> tuple[list[tuple[int, int, list[int]]], int, tuple[int, ...]]:
        """Lower online whole-expert claiming to a deterministic list-scheduled DAG."""
        pooled, resolved_dependencies = self._tail_pool_layout(
            tasks,
            pool_threads=pool_threads,
            max_pooled_routes=max_pooled_routes,
        )
        fixed_task_ids = [task_id for task_id, is_pooled in enumerate(pooled) if not is_pooled]
        fixed_sim_ids: dict[int, int] = {}
        isolated_finish: dict[int, float] = {}
        simulation_tasks: list[tuple[int, int, list[int]]] = []
        intervals: list[tuple[float, float]] = []

        for task_id in fixed_task_ids:
            _, routes, _, threads, _ = tasks[task_id]
            dependencies = resolved_dependencies[task_id]
            start = max((isolated_finish[dependency] for dependency in dependencies), default=0.0)
            finish = start + self.model.T_iso(routes, threads)
            sim_dependencies = [fixed_sim_ids[dependency] for dependency in dependencies]
            fixed_sim_ids[task_id] = len(simulation_tasks)
            isolated_finish[task_id] = finish
            simulation_tasks.append((routes, threads, sim_dependencies))
            intervals.append((start, finish))

        group_count = self.num_cores // pool_threads
        group_blockers: list[list[int]] = [[] for _ in range(group_count)]
        for task_id in fixed_task_ids:
            _, _, core_begin, threads, _ = tasks[task_id]
            first_group = core_begin // pool_threads
            for group in range(first_group, first_group + threads // pool_threads):
                group_blockers[group].append(task_id)

        availability = [
            (
                max((isolated_finish[task_id] for task_id in blockers), default=0.0),
                group,
            )
            for group, blockers in enumerate(group_blockers)
        ]
        heapify(availability)
        previous_pool_task: list[int | None] = [None] * group_count
        pooled_task_ids = sorted(
            (task_id for task_id, is_pooled in enumerate(pooled) if is_pooled),
            key=lambda task_id: (-tasks[task_id][1], tasks[task_id][0]),
        )
        for task_id in pooled_task_ids:
            available_ns, group = heappop(availability)
            _, routes, _, _, _ = tasks[task_id]
            previous = previous_pool_task[group]
            if previous is None:
                dependencies = [fixed_sim_ids[blocker] for blocker in group_blockers[group]]
            else:
                dependencies = [previous]
            sim_id = len(simulation_tasks)
            finish = available_ns + self.model.T_iso(routes, pool_threads)
            simulation_tasks.append((routes, pool_threads, dependencies))
            intervals.append((available_ns, finish))
            previous_pool_task[group] = sim_id
            heappush(availability, (finish, group))

        peak_active = self._peak_active_tasks(intervals)
        fixed_widths = tuple(
            threads
            for _, threads in sorted(
                {(core_begin, threads) for task_id, (_, _, core_begin, threads, _) in enumerate(tasks) if not pooled[task_id]}
            )
        )
        active_pool_groups = min(len(pooled_task_ids), group_count)
        active_widths = fixed_widths + (pool_threads,) * active_pool_groups
        return simulation_tasks, peak_active, active_widths

    def _tail_pool_candidate(
        self,
        experts,
        strict_candidate: dict,
        *,
        pool_threads: int,
        max_pooled_routes: int,
    ) -> dict:
        simulation_tasks, peak_active, active_widths = self._tail_pool_simulation(
            strict_candidate["tasks"],
            pool_threads=pool_threads,
            max_pooled_routes=max_pooled_routes,
        )
        makespan = self.model.dag_makespan(simulation_tasks)
        shape = strict_candidate["shape"]
        uncertainty = self._uncertainty(
            experts,
            shape,
            makespan,
            use_full_workload_anchor=False,
        )
        pooled_tasks = sum(routes <= max_pooled_routes for _, routes in experts)
        return {
            "shape": shape,
            "execution_mode": _ASYNC_EXECUTION_TAIL_POOL,
            "tail_pool_threads": pool_threads,
            "tail_pool_max_routes": max_pooled_routes,
            "tail_pool_tasks": pooled_tasks,
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": strict_candidate["tasks"],
            "active_working_set_bytes": peak_active * self.model.max_stage_bytes,
            "window_bytes_per_worker": tuple(
                self.model.window_bytes_per_worker(width) for width in active_widths
            ),
            "resource_groups": peak_active,
        }

    def _tail_pool_candidates(
        self,
        experts,
        strict_candidates,
        *,
        max_pooled_routes: int,
        forced_pool_threads: int | None,
    ) -> list[dict]:
        if max_pooled_routes <= 0:
            raise ValueError(f"tail_pool_max_routes must be positive, got {max_pooled_routes}")
        if forced_pool_threads is not None:
            thresholds = [max_pooled_routes]
        else:
            thresholds = []
            seen_pooled_sets: set[tuple[int, ...]] = set()
            threshold_points = sorted(
                {
                    threshold
                    for threshold in (*_AUTO_TAIL_POOL_THRESHOLDS, max_pooled_routes)
                    if threshold <= max_pooled_routes
                }
            )
            for threshold in threshold_points:
                pooled_set = tuple(expert for expert, routes in experts if routes <= threshold)
                if not pooled_set or pooled_set in seen_pooled_sets:
                    continue
                seen_pooled_sets.add(pooled_set)
                thresholds.append(threshold)
        if not thresholds:
            return []
        widths = (
            [forced_pool_threads]
            if forced_pool_threads is not None
            else [width for width in self.widths if width in _AUTO_TAIL_POOL_WIDTHS]
        )
        candidates: list[dict] = []
        for strict_candidate in strict_candidates:
            for threshold in thresholds:
                for pool_threads in widths:
                    if pool_threads is None:
                        continue
                    try:
                        candidates.append(
                            self._tail_pool_candidate(
                                experts,
                                strict_candidate,
                                pool_threads=int(pool_threads),
                                max_pooled_routes=threshold,
                            )
                        )
                    except (KeyError, ValueError):
                        if forced_pool_threads is not None:
                            continue
        return candidates

    @staticmethod
    def _tail_pool_head_candidates(strict_candidates: Sequence[dict]) -> list[dict]:
        """Keep statistically competitive heads plus the two fastest shapes."""
        ordered = sorted(strict_candidates, key=lambda candidate: candidate["makespan_ns"])
        fastest = ordered[0]
        fastest_lower = fastest["makespan_ns"] - fastest["uncertainty_ns"]
        fastest_upper = fastest["pessimistic_ns"]
        selected_shapes = {
            candidate["shape"]
            for candidate in ordered
            if candidate["makespan_ns"] - candidate["uncertainty_ns"] <= fastest_upper
            and candidate["pessimistic_ns"] >= fastest_lower
        }
        selected_shapes.update(candidate["shape"] for candidate in ordered[:_AUTO_TAIL_POOL_MIN_HEAD_SHAPES])
        return [candidate for candidate in ordered if candidate["shape"] in selected_shapes]

    def _finalize_plan(
        self,
        selected: dict,
        candidates: Sequence[dict],
        *,
        planner_backend: str,
        planner_workers: int,
        strict_candidates: int,
        dynamic_candidates: int,
    ) -> Dict[str, object]:
        policy = None
        if self.model.policy is not None:
            policy = {
                "profile": str(self.model.profile_path),
                "w13_split": self.model.policy.w13_split,
                "w13_split_chunks": self.model.policy.w13_split_chunks,
                "weight_window_bytes": self.model.policy.weight_window_bytes,
                "w13_window_ranges": self.model.policy.w13_window_ranges,
                "w2_window_ranges": self.model.policy.w2_window_ranges,
                "intermediate_size": self.model.policy.intermediate_size,
                "mode": self.model.policy.mode,
                "degree": self.model.policy.degree,
            }
        bridge = (
            self.to_tail_pool_bridge(
                selected["tasks"],
                pool_threads=selected["tail_pool_threads"],
                max_pooled_routes=selected["tail_pool_max_routes"],
            )
            if selected["execution_mode"] == _ASYNC_EXECUTION_TAIL_POOL
            else self.to_async_bridge(selected["tasks"])
        )
        return {
            "plan_version": _ASYNC_PLAN_VERSION,
            "shape": tuple(selected["shape"]),
            "execution_mode": selected["execution_mode"],
            "tail_pool_threads": selected["tail_pool_threads"],
            "tail_pool_max_routes": selected["tail_pool_max_routes"],
            "tail_pool_tasks": selected["tail_pool_tasks"],
            "makespan_ns": selected["makespan_ns"],
            "uncertainty_ns": selected["uncertainty_ns"],
            "active_working_set_bytes": selected["active_working_set_bytes"],
            "resource_groups": selected["resource_groups"],
            "w13_split": (self.model.policy.w13_split if self.model.policy is not None else None),
            "weight_window_bytes": (self.model.policy.weight_window_bytes if self.model.policy is not None else None),
            "w13_window_ranges": (self.model.policy.w13_window_ranges if self.model.policy is not None else None),
            "w2_window_ranges": (self.model.policy.w2_window_ranges if self.model.policy is not None else None),
            "window_bytes_per_worker": tuple(selected["window_bytes_per_worker"]),
            "task_stage_window_policy": (
                self.task_stage_window_policy.name if self.task_stage_window_policy is not None else None
            ),
            "policy": policy,
            "tasks": selected["tasks"],
            "bridge": bridge,
            "planner_backend": planner_backend,
            "planner_workers": planner_workers,
            "strict_candidates": strict_candidates,
            "dynamic_candidates": dynamic_candidates,
            "ranking": [
                {
                    "shape": tuple(candidate["shape"]),
                    "execution_mode": candidate["execution_mode"],
                    "tail_pool_threads": candidate["tail_pool_threads"],
                    "tail_pool_max_routes": candidate["tail_pool_max_routes"],
                    "tail_pool_tasks": candidate["tail_pool_tasks"],
                    "makespan_ms": round(candidate["makespan_ns"] / 1e6, 6),
                    "pessimistic_ms": round(candidate["pessimistic_ns"] / 1e6, 6),
                    "active_working_set_bytes": candidate["active_working_set_bytes"],
                    "window_bytes_per_worker": tuple(candidate["window_bytes_per_worker"]),
                    "resource_groups": candidate["resource_groups"],
                }
                for candidate in sorted(candidates, key=lambda candidate: candidate["makespan_ns"])
            ],
        }

    def plan(
        self,
        experts: List[Tuple[int, int]],
        *,
        dynamic_tail_pool: bool = True,
        tail_pool_max_routes: int = 12,
        forced_tail_pool_threads: int | None = None,
    ) -> Dict[str, object]:
        experts = [(expert, routes) for expert, routes in experts if routes > 0]
        if not experts:
            raise ValueError("at least one active expert is required")
        if self._native_planner is not None:
            native = self._native_planner.plan(
                [expert for expert, _ in experts],
                [routes for _, routes in experts],
                dynamic_tail_pool,
                tail_pool_max_routes,
                forced_tail_pool_threads,
            )
            return self._finalize_plan(
                native["selected"],
                native["candidates"],
                planner_backend="cpp",
                planner_workers=int(native["configured_workers"]),
                strict_candidates=int(native["strict_candidates"]),
                dynamic_candidates=int(native["dynamic_candidates"]),
            )
        strict_candidates = [self._candidate(experts, shape) for shape in self.shapes]
        tail_pool_candidates: list[dict] = []
        if dynamic_tail_pool or forced_tail_pool_threads is not None:
            tail_pool_heads = (
                strict_candidates
                if forced_tail_pool_threads is not None
                else self._tail_pool_head_candidates(strict_candidates)
            )
            tail_pool_candidates = self._tail_pool_candidates(
                experts,
                tail_pool_heads,
                max_pooled_routes=tail_pool_max_routes,
                forced_pool_threads=forced_tail_pool_threads,
            )
        if forced_tail_pool_threads is not None:
            if not tail_pool_candidates:
                raise ValueError(
                    "no valid forced tail-pool candidate: "
                    f"pool_threads={forced_tail_pool_threads}, max_routes={tail_pool_max_routes}"
                )
            candidates = tail_pool_candidates
        else:
            candidates = list(strict_candidates)
            if dynamic_tail_pool:
                candidates.extend(tail_pool_candidates)
        selected = self._select(candidates)
        return self._finalize_plan(
            selected,
            candidates,
            planner_backend="python",
            planner_workers=1,
            strict_candidates=len(strict_candidates),
            dynamic_candidates=len(tail_pool_candidates),
        )

    def _task_stage_windows(self, tasks, task_threads: Sequence[int]) -> tuple[list[int], list[int]]:
        if self.task_stage_window_policy is None:
            inherited = [-1] * len(tasks)
            return inherited, list(inherited)
        w13_windows: list[int] = []
        w2_windows: list[int] = []
        for values, threads in zip(tasks, task_threads):
            _, routes, _, _, _ = values
            w13_bytes, w2_bytes = self.task_stage_window_policy.select(int(routes), int(threads))
            if min(w13_bytes, w2_bytes) < -1:
                raise ValueError("task stage-window policy must return -1 or non-negative byte counts")
            w13_windows.append(int(w13_bytes))
            w2_windows.append(int(w2_bytes))
        return w13_windows, w2_windows

    def to_async_bridge(self, tasks) -> Dict[str, object]:
        dependency_offsets, flat_dependencies = [0], []
        for _, _, _, _, dependencies in tasks:
            flat_dependencies.extend(dependencies)
            dependency_offsets.append(len(flat_dependencies))
        task_threads = [threads for _, _, _, threads, _ in tasks]
        task_w13_window_bytes, task_w2_window_bytes = self._task_stage_windows(tasks, task_threads)
        num_tasks = len(tasks)
        return {
            "plan_version": _ASYNC_PLAN_VERSION,
            "execution_mode": _ASYNC_EXECUTION_STRICT,
            "num_threads": self.num_cores,
            "thread_cpu_ids": list(self.cpu_ids),
            "task_expert_ids": [expert for expert, _, _, _, _ in tasks],
            "task_core_begins": [core for _, _, core, _, _ in tasks],
            "task_threads": task_threads,
            "task_dep_offsets": dependency_offsets,
            "task_deps": flat_dependencies,
            "task_preferred_threads": list(task_threads),
            "task_min_threads": list(task_threads),
            "task_max_threads": list(task_threads),
            "task_allowed_thread_offsets": list(range(num_tasks + 1)),
            "task_allowed_threads": list(task_threads),
            "task_placement_modes": [_ASYNC_PLACEMENT_FIXED] * num_tasks,
            "task_numa_nodes": [-1] * num_tasks,
            "task_stage_ids": [_ASYNC_STAGE_EXPERT] * num_tasks,
            "task_resize_points": [_ASYNC_RESIZE_NONE] * num_tasks,
            "task_range_granularities": [_ASYNC_FULL_EXPERT_RANGE] * num_tasks,
            "task_w13_window_bytes": task_w13_window_bytes,
            "task_w2_window_bytes": task_w2_window_bytes,
        }

    def to_tail_pool_bridge(
        self,
        tasks,
        *,
        pool_threads: int,
        max_pooled_routes: int = 12,
    ) -> Dict[str, object]:
        """Build the selected whole-expert tail-pool Plan V2 bridge."""
        pooled, resolved_dependencies = self._tail_pool_layout(
            tasks,
            pool_threads=pool_threads,
            max_pooled_routes=max_pooled_routes,
        )

        dependency_offsets = [0]
        flat_dependencies: list[int] = []
        for dependencies in resolved_dependencies:
            flat_dependencies.extend(dependencies)
            dependency_offsets.append(len(flat_dependencies))

        task_threads = [pool_threads if pooled[task] else int(values[3]) for task, values in enumerate(tasks)]
        task_w13_window_bytes, task_w2_window_bytes = self._task_stage_windows(tasks, task_threads)
        num_tasks = len(tasks)
        return {
            "plan_version": _ASYNC_PLAN_VERSION,
            "execution_mode": _ASYNC_EXECUTION_TAIL_POOL,
            "num_threads": self.num_cores,
            "thread_cpu_ids": list(self.cpu_ids),
            "task_expert_ids": [expert for expert, _, _, _, _ in tasks],
            "task_core_begins": [-1 if pooled[task] else int(values[2]) for task, values in enumerate(tasks)],
            "task_threads": task_threads,
            "task_dep_offsets": dependency_offsets,
            "task_deps": flat_dependencies,
            "task_preferred_threads": list(task_threads),
            "task_min_threads": list(task_threads),
            "task_max_threads": list(task_threads),
            "task_allowed_thread_offsets": list(range(num_tasks + 1)),
            "task_allowed_threads": list(task_threads),
            "task_placement_modes": [
                _ASYNC_PLACEMENT_TAIL_POOL if is_pooled else _ASYNC_PLACEMENT_FIXED for is_pooled in pooled
            ],
            "task_numa_nodes": [-1] * num_tasks,
            "task_stage_ids": [_ASYNC_STAGE_EXPERT] * num_tasks,
            "task_resize_points": [_ASYNC_RESIZE_NONE] * num_tasks,
            "task_range_granularities": [_ASYNC_FULL_EXPERT_RANGE] * num_tasks,
            "task_w13_window_bytes": task_w13_window_bytes,
            "task_w2_window_bytes": task_w2_window_bytes,
        }


class PolicyAwarePlanner:
    """Search packed-B window policy and an exact profiled core shape together."""

    def __init__(
        self,
        models: Sequence[PlannerCostModel],
        num_cores: int,
        *,
        cpu_ids: Sequence[int] | None = None,
        working_set_target_fraction: float = 2.0 / 3.0,
        task_stage_window_policy: TaskStageWindowPolicy | None = None,
        task_stage_window_policies: Sequence[TaskStageWindowPolicy | None] | None = None,
    ):
        if not models:
            raise ValueError("at least one policy model is required")
        if task_stage_window_policy is not None and task_stage_window_policies is not None:
            raise ValueError("provide either one task stage-window policy or per-model policies, not both")
        if task_stage_window_policies is None:
            task_stage_window_policies = (task_stage_window_policy,) * len(models)
        else:
            task_stage_window_policies = tuple(task_stage_window_policies)
            if len(task_stage_window_policies) != len(models):
                raise ValueError("task_stage_window_policies must match the model count")
        policies = [model.policy for model in models]
        if any(policy is None for policy in policies):
            raise ProfileCompatibilityError("joint policy search requires schema-v2 profiles")
        base_key = policies[0].key_without_kernel_policy()
        if any(policy.key_without_kernel_policy() != base_key for policy in policies[1:]):
            raise ProfileCompatibilityError("joint planner profiles differ outside the packed-B window policy")
        variant_keys = [policy.kernel_policy_key() for policy in policies]
        if len(variant_keys) != len(set(variant_keys)):
            raise ProfileCompatibilityError("joint planner received duplicate packed-B window policies")
        self.models = tuple(models)
        self.num_cores = int(num_cores)
        self.cpu_ids = cpu_ids
        self.working_set_target_fraction = float(working_set_target_fraction)
        self.planners = tuple(
            IntervalPlanner(
                model,
                num_cores,
                cpu_ids=cpu_ids,
                shapes=self._pruned_shapes(model),
                task_stage_window_policy=stage_window_policy,
            )
            for model, stage_window_policy in zip(self.models, task_stage_window_policies)
        )

    def _pruned_shapes(self, model: PlannerCostModel):
        policy = model.policy
        assert policy is not None
        target = policy.llc_bytes_per_rank * self.working_set_target_fraction
        shapes = [shape for shape in _model_candidate_shapes(model, self.num_cores) if sum(shape) == self.num_cores]
        by_distance = sorted(
            shapes,
            key=lambda shape: abs(len(shape) * model.max_stage_bytes - target),
        )
        keep = {shape for shape in shapes if len(shape) == 1 or len(shape) * model.max_stage_bytes <= target * 1.25}
        keep.update(by_distance[:2])
        lane_counts = {len(shape) for shape in keep}
        for shape in shapes:
            if len(shape) in lane_counts:
                keep.add(shape)
        return tuple(sorted(keep, key=lambda shape: (len(shape), shape), reverse=False))

    def plan(
        self,
        experts: List[Tuple[int, int]],
        *,
        dynamic_tail_pool: bool = True,
        tail_pool_max_routes: int = 12,
        forced_tail_pool_threads: int | None = None,
    ) -> Dict[str, object]:
        policy_results = [
            planner.plan(
                experts,
                dynamic_tail_pool=dynamic_tail_pool,
                tail_pool_max_routes=tail_pool_max_routes,
                forced_tail_pool_threads=forced_tail_pool_threads,
            )
            for planner in self.planners
        ]
        candidates = []
        for result in policy_results:
            candidates.append(
                {
                    "result": result,
                    "makespan_ns": result["makespan_ns"],
                    "uncertainty_ns": result["uncertainty_ns"],
                    "pessimistic_ns": result["makespan_ns"] + result["uncertainty_ns"],
                    "active_working_set_bytes": result["active_working_set_bytes"],
                    "shape": result["shape"],
                    "resource_groups": result["resource_groups"],
                    "execution_mode": result["execution_mode"],
                }
            )
        selected = IntervalPlanner._select(candidates)["result"]
        selected["policy_ranking"] = [
            {
                "w13_split": result["w13_split"],
                "weight_window_bytes": result["weight_window_bytes"],
                "w13_window_ranges": result["w13_window_ranges"],
                "w2_window_ranges": result["w2_window_ranges"],
                "shape": result["shape"],
                "execution_mode": result["execution_mode"],
                "tail_pool_threads": result["tail_pool_threads"],
                "tail_pool_max_routes": result["tail_pool_max_routes"],
                "tail_pool_tasks": result["tail_pool_tasks"],
                "makespan_ms": result["makespan_ns"] / 1e6,
                "uncertainty_ms": result["uncertainty_ns"] / 1e6,
                "active_working_set_bytes": result["active_working_set_bytes"],
                "window_bytes_per_worker": result["window_bytes_per_worker"],
            }
            for result in sorted(policy_results, key=lambda result: result["makespan_ns"])
        ]
        return selected


if __name__ == "__main__":
    model = ContentionCostModel(sys.argv[1])
    planner = IntervalPlanner(model, num_cores=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    result = planner.plan([(index, 192) for index in range(8)])
    print(result)
