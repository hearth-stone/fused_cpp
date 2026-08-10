"""Static interval-DAG planner with exact-domain calibration."""

from __future__ import annotations

import importlib
import math
import os
import sys
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Dict, List, Mapping, Protocol, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402
from profile_catalog import ProfileCompatibilityError  # noqa: E402
from stage_window_policy import TaskStageWindowPolicy  # noqa: E402


_ASYNC_PLAN_VERSION = 2
_ASYNC_EXECUTION_STRICT = "strict"
_ASYNC_EXECUTION_TAIL_POOL = "tail_pool"
_ASYNC_EXECUTION_ELASTIC = "elastic"
_ASYNC_PLACEMENT_FIXED = 0
_ASYNC_PLACEMENT_TAIL_POOL = 1
_ASYNC_STAGE_EXPERT = 0
_ASYNC_RESIZE_NONE = 0
_ASYNC_RESIZE_BEFORE_W2 = 1
_ASYNC_FULL_EXPERT_RANGE = 0
_AUTO_TAIL_POOL_WIDTHS = frozenset((1, 2, 4))
_AUTO_TAIL_POOL_THRESHOLDS = (1, 2, 4, 8, 12)
_AUTO_TAIL_POOL_MIN_HEAD_SHAPES = 2
_BOUNDED_TAIL_TASKS = 2
_BOUNDED_TAIL_DEFAULT_CORES = 96
_EARLY_MERGE_EQUAL_FINISH_REL_TOL = 1e-6
_EARLY_MERGE_EQUAL_FINISH_ABS_NS = 1.0


class PlannerPolicy(Protocol):
    intermediate_size: int
    mode: str
    degree: int
    llc_bytes_per_rank: int

    def identity_key(self) -> tuple[object, ...]: ...


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

    def dag_task_finish_times(self, tasks) -> Sequence[float]: ...

    def stage_T_iso(self, stage: str, routes: int, threads: int) -> float: ...

    def stage_dag_makespan(self, stage: str, tasks) -> float: ...

    def task_stage_bytes(self, stage: str, routes: int, threads: int) -> int: ...

    def task_stage_ranges(self, routes: int, threads: int) -> tuple[int, int]: ...

    def stage_window_bytes_per_worker(self, stage: str, threads: int, routes: int | None = None) -> int: ...

    def supports_shape(self, shape) -> bool: ...

    def relative_uncertainty(self, routes: int, shape) -> float: ...

    def relative_full_call_uncertainty(self, routes: int, shape) -> float: ...

    def profiled_full_call_time(self, routes: int, shape) -> float: ...

    def window_bytes_per_worker(self, threads: int, routes: int | None = None) -> int: ...

    def can_use_bounded_tail_repartition_anchor(
        self,
        routes: int,
        root_shape,
        tail_width: int,
        route_slices: int = 1,
    ) -> bool: ...

    def profiled_bounded_tail_repartition(
        self,
        routes: int,
        root_shape,
        tail_width: int,
        route_slices: int = 1,
    ) -> tuple[float, float]: ...


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


def _default_tail_repartition_widths(num_cores: int) -> tuple[int, ...]:
    """Widths calibrated by the NUMA-local 96-core static tail experiment."""
    if num_cores != _BOUNDED_TAIL_DEFAULT_CORES:
        return ()
    return (num_cores // 4, num_cores // 3, num_cores // 2)


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
        tail_repartition_widths: Sequence[int] | None = None,
        stage: str | None = None,
    ):
        if stage not in {None, "w13", "w2"}:
            raise ValueError(f"stage must be None, 'w13', or 'w2', got {stage!r}")
        self.stage = stage
        self.task_stage_window_policy = task_stage_window_policy
        if task_stage_window_policy is None:
            self.model = model
        else:
            binder = getattr(model, "with_task_stage_window_policy", None)
            if not callable(binder):
                raise ProfileCompatibilityError(
                    "task stage-window policy requires a cost model that can bind execution windows"
                )
            self.model = binder(task_stage_window_policy)
        self.num_cores = int(num_cores)
        model_widths = getattr(self.model, "supported_widths", None)
        self.widths = tuple(widths or model_widths or _default_widths(self.num_cores))
        configured_tail_widths = (
            _default_tail_repartition_widths(self.num_cores)
            if tail_repartition_widths is None
            else tuple(int(width) for width in tail_repartition_widths)
        )
        if any(
            width <= 0
            or width > self.num_cores // _BOUNDED_TAIL_TASKS
            or self.num_cores % width != 0
            for width in configured_tail_widths
        ):
            raise ValueError(
                "tail_repartition_widths must be positive divisors no wider than "
                f"num_cores/{_BOUNDED_TAIL_TASKS}"
            )
        self.tail_repartition_widths = tuple(sorted(set(configured_tail_widths)))
        self.cpu_ids = tuple(int(cpu) for cpu in (cpu_ids if cpu_ids is not None else range(num_cores)))
        if len(self.cpu_ids) != self.num_cores or len(set(self.cpu_ids)) != num_cores:
            raise ValueError("cpu_ids must contain num_cores unique physical CPUs")

        if shapes is not None:
            candidates = [tuple(int(value) for value in shape) for shape in shapes]
        elif self.model.schema_version >= 2:
            candidates = list(_model_candidate_shapes(self.model, self.num_cores))
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
        if self.stage is None:
            self._native_planner = self._create_native_planner(
                native_cold_planner=native_cold_planner,
                planner_threads=planner_threads,
            )
        else:
            if native_cold_planner is True:
                raise ValueError("native cold planner does not yet support stage-specific scoring")
            self._native_planner = None

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
                    "native cold planner was requested but the cost model does not support native calibration export"
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
            list(self.tail_repartition_widths),
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
                key=lambda candidate: load[candidate] + self._task_time(routes, lanes[candidate][1]),
            )
            lane_experts[lane].append(index)
            load[lane] += self._task_time(routes, lanes[lane][1])
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
        return self._dag_makespan([(routes, threads, deps) for _, routes, _, threads, deps in tasks])

    def _task_time(self, routes: int, threads: int) -> float:
        if self.stage is None:
            return self.model.T_iso(routes, threads)
        return self.model.stage_T_iso(self.stage, routes, threads)

    def _dag_makespan(self, tasks) -> float:
        if self.stage is None:
            return self.model.dag_makespan(tasks)
        return self.model.stage_dag_makespan(self.stage, tasks)

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
        if use_full_workload_anchor and self._uses_full_workload_anchor(experts, shape):
            relative = self.model.relative_full_call_uncertainty(experts[0][1], shape)
            return makespan * relative / math.sqrt(self.model.profile_runs)
        relative = max(self.model.relative_uncertainty(routes, shape) for _, routes in experts)
        waves = max(1, math.ceil(len(experts) / len(shape)))
        return makespan * relative / math.sqrt(waves * self.model.profile_runs)

    def _task_max_stage_bytes(self, routes: int, threads: int) -> int:
        if self.stage is not None:
            resolver = getattr(self.model, "task_stage_bytes", None)
            if callable(resolver):
                return int(resolver(self.stage, routes, threads))
        resolver = getattr(self.model, "task_max_stage_bytes", None)
        if callable(resolver):
            return int(resolver(routes, threads))
        return int(self.model.max_stage_bytes)

    def _task_window_bytes_per_worker(self, routes: int, threads: int) -> int:
        if self.stage is not None:
            resolver = getattr(self.model, "stage_window_bytes_per_worker", None)
            if callable(resolver):
                return int(resolver(self.stage, threads, routes))
        resolver = self.model.window_bytes_per_worker
        if self.task_stage_window_policy is not None:
            return int(resolver(threads, routes))
        return int(resolver(threads))

    def active_working_set_bytes(self, shape, tasks=None) -> int:
        if tasks is None:
            active_lanes = len(shape)
            if self.stage is None:
                return active_lanes * self.model.max_stage_bytes
            stage_bytes = self._task_max_stage_bytes(1, max(int(width) for width in shape))
            return active_lanes * stage_bytes
        lane_bytes: dict[tuple[int, int], int] = {}
        for _, routes, core, threads, _ in tasks:
            resource = (core, threads)
            lane_bytes[resource] = max(
                lane_bytes.get(resource, 0),
                self._task_max_stage_bytes(routes, threads),
            )
        return sum(lane_bytes.values())

    def window_bytes_per_worker(self, shape, tasks=None) -> tuple[int, ...]:
        if tasks is None:
            if self.stage is not None:
                return tuple(
                    self.model.stage_window_bytes_per_worker(self.stage, int(width)) for width in shape
                )
            return tuple(self.model.window_bytes_per_worker(int(width)) for width in shape)
        lane_windows: dict[tuple[int, int], int] = {}
        for _, routes, core, threads, _ in tasks:
            resource = (core, threads)
            lane_windows[resource] = max(
                lane_windows.get(resource, 0),
                self._task_window_bytes_per_worker(routes, threads),
            )
        return tuple(lane_windows[resource] for resource in sorted(lane_windows))

    def _uses_full_workload_anchor(self, experts, shape) -> bool:
        if self.stage is not None:
            return False
        eligible = (
            self.model.has_full_workload_anchors
            and len(experts) == self.model.local_experts
            and bool(experts)
            and all(routes == experts[0][1] for _, routes in experts)
        )
        if not eligible:
            return False
        resolver = getattr(self.model, "can_use_full_workload_anchor", None)
        return not callable(resolver) or bool(resolver(experts[0][1], shape))

    def score_shape(self, experts, shape):
        signature = tuple(int(value) for value in shape)
        if self.model.schema_version >= 2 and not self.model.supports_shape(signature):
            raise ProfileCompatibilityError(f"shape {signature} is not supported by {self.model.profile_path.name}")
        lanes = self._lanes(signature)
        tasks = self._build_tasks(experts, lanes, self._assign(experts, lanes))
        if self._uses_full_workload_anchor(experts, signature):
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
            "tail_repartition_width": None,
            "tail_repartition_tasks": 0,
            "tail_repartition_route_slices": 1,
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

    def _bounded_tail_repartition_tasks(
        self,
        tasks,
        tail_width: int,
        route_slices: int = 1,
    ):
        """Rewrite two terminal experts onto wider fixed route-slice intervals."""
        tail_width = int(tail_width)
        route_slices = int(route_slices)
        physical_tail_tasks = _BOUNDED_TAIL_TASKS * route_slices
        if (
            self.stage is not None
            or tail_width <= 0
            or route_slices <= 0
            or self.num_cores % _BOUNDED_TAIL_TASKS != 0
            or self.num_cores % tail_width != 0
            or 2 * tail_width > self.num_cores
            or (route_slices > 1 and physical_tail_tasks * tail_width != self.num_cores)
        ):
            raise ValueError("bounded tail width is not feasible for this planner domain")

        successors = [0] * len(tasks)
        for task_id, (_, _, _, _, dependencies) in enumerate(tasks):
            for dependency in dependencies:
                if dependency < 0 or dependency >= task_id:
                    raise ValueError(
                        f"task dependencies must refer to earlier task ids: task={task_id}, dependency={dependency}"
                    )
                successors[dependency] += 1

        roots = [(task_id, task) for task_id, task in enumerate(tasks) if not task[4]]
        tails = [(task_id, task) for task_id, task in enumerate(tasks) if task[4]]
        if (
            len(tails) != _BOUNDED_TAIL_TASKS
            or len(roots) + len(tails) != len(tasks)
            or any(successors[task_id] != 0 for task_id, _ in tails)
            or any(tail_width <= int(task[3]) for _, task in tails)
            or any(int(task[1]) % route_slices != 0 for _, task in tails)
        ):
            raise ValueError("bounded tail repartition requires exactly two terminal second-wave tasks")

        ordered_roots = sorted(roots, key=lambda item: (int(item[1][2]), item[0]))
        ordered_tails = sorted(tails, key=lambda item: (int(item[1][2]), item[0]))
        rewritten = [
            (int(expert), int(routes), int(core_begin), int(threads), [])
            for _, (expert, routes, core_begin, threads, _) in ordered_roots
        ]
        half = self.num_cores // _BOUNDED_TAIL_TASKS
        for tail_index, (_, tail) in enumerate(ordered_tails):
            for route_slice in range(route_slices):
                core_begin = (
                    tail_index * half
                    if route_slices == 1
                    else (tail_index * route_slices + route_slice) * tail_width
                )
                core_end = core_begin + tail_width
                if core_end > self.num_cores:
                    raise ValueError("bounded tail intervals exceed the planner domain")
                blockers = [
                    root_id
                    for root_id, (_, (_, _, root_begin, root_threads, _)) in enumerate(ordered_roots)
                    if int(root_begin) < core_end and core_begin < int(root_begin) + int(root_threads)
                ]
                if not blockers:
                    raise ValueError(f"tail interval [{core_begin}, {core_end}) has no first-wave blockers")
                rewritten.append(
                    (
                        int(tail[0]),
                        int(tail[1]) // route_slices,
                        core_begin,
                        tail_width,
                        blockers,
                    )
                )
        return rewritten

    def _bounded_tail_repartition_candidate(
        self,
        experts,
        strict_candidate: dict,
        *,
        tail_width: int,
        route_slices: int = 1,
    ) -> dict:
        tasks = self._bounded_tail_repartition_tasks(
            strict_candidate["tasks"],
            tail_width,
            route_slices,
        )
        shape = strict_candidate["shape"]
        uniform_routes = {int(routes) for _, routes in experts}
        anchor_supported = getattr(self.model, "can_use_bounded_tail_repartition_anchor", None)
        anchor_lookup = getattr(self.model, "profiled_bounded_tail_repartition", None)
        anchor = None
        if (
            len(uniform_routes) == 1
            and callable(anchor_supported)
            and callable(anchor_lookup)
            and anchor_supported(
                next(iter(uniform_routes)),
                shape,
                tail_width,
                route_slices,
            )
        ):
            anchor = anchor_lookup(
                next(iter(uniform_routes)),
                shape,
                tail_width,
                route_slices,
            )
        if route_slices > 1 and anchor is None:
            raise ValueError("route-sliced bounded-tail candidates require an exact layout anchor")
        if anchor is None:
            makespan = self._score(tasks)
            uncertainty = self._uncertainty(
                experts,
                shape,
                makespan,
                use_full_workload_anchor=False,
            )
        else:
            makespan, uncertainty = anchor
        physical_tail_tasks = _BOUNDED_TAIL_TASKS * route_slices
        root_tasks = tasks[:-physical_tail_tasks]
        tail_tasks = tasks[-physical_tail_tasks:]
        root_bytes = [self._task_max_stage_bytes(routes, threads) for _, routes, _, threads, _ in root_tasks]
        tail_bytes = [self._task_max_stage_bytes(routes, threads) for _, routes, _, threads, _ in tail_tasks]
        root_windows = [
            self._task_window_bytes_per_worker(routes, threads) for _, routes, _, threads, _ in root_tasks
        ]
        tail_windows = [
            self._task_window_bytes_per_worker(routes, threads) for _, routes, _, threads, _ in tail_tasks
        ]
        root_working_set = sum(root_bytes)
        tail_working_set = sum(tail_bytes)
        active_windows = root_windows if root_working_set >= tail_working_set else tail_windows
        return {
            "shape": shape,
            "execution_mode": _ASYNC_EXECUTION_STRICT,
            "tail_pool_threads": None,
            "tail_pool_max_routes": None,
            "tail_pool_tasks": 0,
            "tail_repartition_width": int(tail_width),
            "tail_repartition_tasks": _BOUNDED_TAIL_TASKS,
            "tail_repartition_route_slices": int(route_slices),
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": tasks,
            "active_working_set_bytes": max(root_working_set, tail_working_set),
            "window_bytes_per_worker": tuple(active_windows),
            "resource_groups": max(len(root_tasks), len(tail_tasks)),
        }

    def _bounded_tail_repartition_candidates(self, experts, strict_candidates) -> list[dict]:
        candidates: list[dict] = []
        if self.stage is not None:
            return candidates
        for strict_candidate in strict_candidates:
            for tail_width in self.tail_repartition_widths:
                route_slice_options = (1, 2) if 4 * tail_width == self.num_cores else (1,)
                for route_slices in route_slice_options:
                    try:
                        candidates.append(
                            self._bounded_tail_repartition_candidate(
                                experts,
                                strict_candidate,
                                tail_width=tail_width,
                                route_slices=route_slices,
                            )
                        )
                    except (KeyError, ValueError):
                        continue
        return candidates

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
            finish = start + self._task_time(routes, threads)
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
            finish = available_ns + self._task_time(routes, pool_threads)
            simulation_tasks.append((routes, pool_threads, dependencies))
            intervals.append((available_ns, finish))
            previous_pool_task[group] = sim_id
            heappush(availability, (finish, group))

        peak_active = self._peak_active_tasks(intervals)
        fixed_windows: dict[tuple[int, int], int] = {}
        for task_id in fixed_task_ids:
            _, routes, core_begin, threads, _ = tasks[task_id]
            resource = (core_begin, threads)
            fixed_windows[resource] = max(
                fixed_windows.get(resource, 0),
                self._task_window_bytes_per_worker(routes, threads),
            )
        active_pool_groups = min(len(pooled_task_ids), group_count)
        pooled_window = max(
            (self._task_window_bytes_per_worker(tasks[task_id][1], pool_threads) for task_id in pooled_task_ids),
            default=0,
        )
        active_windows = tuple(fixed_windows[resource] for resource in sorted(fixed_windows))
        active_windows += (pooled_window,) * active_pool_groups
        return simulation_tasks, peak_active, active_windows

    def _tail_pool_candidate(
        self,
        experts,
        strict_candidate: dict,
        *,
        pool_threads: int,
        max_pooled_routes: int,
    ) -> dict:
        simulation_tasks, peak_active, active_windows = self._tail_pool_simulation(
            strict_candidate["tasks"],
            pool_threads=pool_threads,
            max_pooled_routes=max_pooled_routes,
        )
        makespan = self._dag_makespan(simulation_tasks)
        shape = strict_candidate["shape"]
        uncertainty = self._uncertainty(
            experts,
            shape,
            makespan,
            use_full_workload_anchor=False,
        )
        pooled_tasks = sum(routes <= max_pooled_routes for _, routes in experts)
        task_working_sets = sorted(
            (self._task_max_stage_bytes(routes, threads) for routes, threads, _ in simulation_tasks),
            reverse=True,
        )
        return {
            "shape": shape,
            "execution_mode": _ASYNC_EXECUTION_TAIL_POOL,
            "tail_pool_threads": pool_threads,
            "tail_pool_max_routes": max_pooled_routes,
            "tail_pool_tasks": pooled_tasks,
            "tail_repartition_width": None,
            "tail_repartition_tasks": 0,
            "tail_repartition_route_slices": 1,
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": strict_candidate["tasks"],
            "active_working_set_bytes": sum(task_working_sets[:peak_active]),
            "window_bytes_per_worker": active_windows,
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
        tail_repartition_candidates: int,
    ) -> Dict[str, object]:
        policy = None
        if self.model.policy is not None:
            policy = {
                "profile": str(self.model.profile_path),
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
            "stage": self.stage,
            "shape": tuple(selected["shape"]),
            "execution_mode": selected["execution_mode"],
            "tail_pool_threads": selected["tail_pool_threads"],
            "tail_pool_max_routes": selected["tail_pool_max_routes"],
            "tail_pool_tasks": selected["tail_pool_tasks"],
            "tail_repartition_width": selected["tail_repartition_width"],
            "tail_repartition_tasks": selected["tail_repartition_tasks"],
            "tail_repartition_route_slices": selected["tail_repartition_route_slices"],
            "makespan_ns": selected["makespan_ns"],
            "uncertainty_ns": selected["uncertainty_ns"],
            "active_working_set_bytes": selected["active_working_set_bytes"],
            "resource_groups": selected["resource_groups"],
            "window_bytes_per_worker": tuple(selected["window_bytes_per_worker"]),
            "task_stage_window_policy": (
                self.task_stage_window_policy.name if self.task_stage_window_policy is not None else None
            ),
            "early_merge": bridge["early_merge"],
            "policy": policy,
            "tasks": selected["tasks"],
            "bridge": bridge,
            "planner_backend": planner_backend,
            "planner_workers": planner_workers,
            "strict_candidates": strict_candidates,
            "dynamic_candidates": dynamic_candidates,
            "tail_repartition_candidates": tail_repartition_candidates,
            "ranking": [
                {
                    "shape": tuple(candidate["shape"]),
                    "execution_mode": candidate["execution_mode"],
                    "tail_pool_threads": candidate["tail_pool_threads"],
                    "tail_pool_max_routes": candidate["tail_pool_max_routes"],
                    "tail_pool_tasks": candidate["tail_pool_tasks"],
                    "tail_repartition_width": candidate["tail_repartition_width"],
                    "tail_repartition_tasks": candidate["tail_repartition_tasks"],
                    "tail_repartition_route_slices": candidate["tail_repartition_route_slices"],
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
        bounded_tail_repartition: bool | None = None,
    ) -> Dict[str, object]:
        experts = [(expert, routes) for expert, routes in experts if routes > 0]
        if not experts:
            raise ValueError("at least one active expert is required")
        if bounded_tail_repartition is None:
            bounded_tail_repartition = dynamic_tail_pool and forced_tail_pool_threads is None
        if self._native_planner is not None:
            native = self._native_planner.plan(
                [expert for expert, _ in experts],
                [routes for _, routes in experts],
                dynamic_tail_pool,
                tail_pool_max_routes,
                forced_tail_pool_threads,
                bounded_tail_repartition,
            )
            return self._finalize_plan(
                native["selected"],
                native["candidates"],
                planner_backend="cpp",
                planner_workers=int(native["configured_workers"]),
                strict_candidates=int(native["strict_candidates"]),
                dynamic_candidates=int(native["dynamic_candidates"]),
                tail_repartition_candidates=int(native["tail_repartition_candidates"]),
            )
        strict_candidates = [self._candidate(experts, shape) for shape in self.shapes]
        tail_repartition_candidates: list[dict] = []
        if bounded_tail_repartition and forced_tail_pool_threads is None:
            tail_repartition_candidates = self._bounded_tail_repartition_candidates(
                experts,
                self._tail_pool_head_candidates(strict_candidates),
            )
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
            if bounded_tail_repartition:
                candidates.extend(tail_repartition_candidates)
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
            tail_repartition_candidates=len(tail_repartition_candidates),
        )

    def _task_stage_windows(
        self,
        tasks,
        w13_threads: Sequence[int],
        w2_threads: Sequence[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        if self.task_stage_window_policy is None:
            inherited = [-1] * len(tasks)
            return inherited, list(inherited)
        if w2_threads is None:
            w2_threads = w13_threads
        w13_windows: list[int] = []
        w2_windows: list[int] = []
        for values, w13_width, w2_width in zip(tasks, w13_threads, w2_threads, strict=True):
            _, routes, _, _, _ = values
            w13_bytes, _ = self.task_stage_window_policy.select(int(routes), int(w13_width))
            _, w2_bytes = self.task_stage_window_policy.select(int(routes), int(w2_width))
            if min(w13_bytes, w2_bytes) < -1:
                raise ValueError("task stage-window policy must return -1 or non-negative byte counts")
            w13_windows.append(int(w13_bytes))
            w2_windows.append(int(w2_bytes))
        return w13_windows, w2_windows

    def _task_stage_ranges(
        self,
        tasks,
        w13_threads: Sequence[int],
        w2_threads: Sequence[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        if w2_threads is None:
            w2_threads = w13_threads
        w13_ranges: list[int] = []
        w2_ranges: list[int] = []
        for values, w13_width, w2_width in zip(tasks, w13_threads, w2_threads, strict=True):
            _, routes, _, _, _ = values
            resolved_w13, _ = self.model.task_stage_ranges(int(routes), int(w13_width))
            _, resolved_w2 = self.model.task_stage_ranges(int(routes), int(w2_width))
            if min(resolved_w13, resolved_w2) <= 0:
                raise ValueError("cost model must resolve positive per-task stage ranges")
            w13_ranges.append(int(resolved_w13))
            w2_ranges.append(int(resolved_w2))
        return w13_ranges, w2_ranges

    def _early_merge_policy(self, tasks) -> bool | None:
        """Disable early merge only when the model predicts no overlap window."""
        if self.stage is not None:
            return None
        finish_time_fn = getattr(self.model, "dag_task_finish_times", None)
        if not callable(finish_time_fn):
            return None
        model_tasks = [
            (int(routes), int(threads), list(dependencies))
            for _, routes, _, threads, dependencies in tasks
        ]
        try:
            task_finish_times = tuple(float(value) for value in finish_time_fn(model_tasks))
        except (KeyError, ValueError):
            return None
        if len(task_finish_times) != len(tasks) or not task_finish_times:
            return None
        expert_finish_times: dict[int, float] = {}
        for values, finish_time in zip(tasks, task_finish_times, strict=True):
            expert = int(values[0])
            expert_finish_times[expert] = max(
                finish_time,
                expert_finish_times.get(expert, -math.inf),
            )
        earliest = min(expert_finish_times.values())
        latest = max(expert_finish_times.values())
        if math.isclose(
            earliest,
            latest,
            rel_tol=_EARLY_MERGE_EQUAL_FINISH_REL_TOL,
            abs_tol=_EARLY_MERGE_EQUAL_FINISH_ABS_NS,
        ):
            return False
        # The current objective does not model merge/computation contention,
        # so a predicted gap is insufficient evidence to force early merge on.
        return None

    def to_async_bridge(self, tasks) -> Dict[str, object]:
        dependency_offsets, flat_dependencies = [0], []
        for _, _, _, _, dependencies in tasks:
            flat_dependencies.extend(dependencies)
            dependency_offsets.append(len(flat_dependencies))
        task_threads = [threads for _, _, _, threads, _ in tasks]
        task_w13_ranges, task_w2_ranges = self._task_stage_ranges(tasks, task_threads)
        num_tasks = len(tasks)
        expert_task_counts: dict[int, int] = {}
        expert_slice_rows: dict[int, int] = {}
        for expert, routes, _, _, _ in tasks:
            expert = int(expert)
            routes = int(routes)
            expert_task_counts[expert] = expert_task_counts.get(expert, 0) + 1
            previous_rows = expert_slice_rows.setdefault(expert, routes)
            if previous_rows != routes:
                raise ValueError(
                    f"route-sliced expert {expert} requires equal-sized task ranges"
                )
        task_range_granularities = [
            int(routes) if expert_task_counts[int(expert)] > 1 else _ASYNC_FULL_EXPERT_RANGE
            for expert, routes, _, _, _ in tasks
        ]
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
            "task_range_granularities": task_range_granularities,
            "task_w13_ranges": task_w13_ranges,
            "task_w2_ranges": task_w2_ranges,
            "task_release_ns": [0] * num_tasks,
            "task_resize_timeout_ns": [0] * num_tasks,
            "task_preferred_core_begins": [-1] * num_tasks,
            "early_merge": self._early_merge_policy(tasks),
        }

    def to_elastic_w2_bridge(
        self,
        tasks,
        *,
        width_transitions: Mapping[int, int],
        numa_node: int,
        resize_timeout_ns: int = 0,
        resizable_task_ids: Sequence[int] | None = None,
        task_preferred_core_begins: Mapping[int, int] | None = None,
    ) -> Dict[str, object]:
        """Build a fixed-W13 plan with planner-selected local W2 cohorts.

        ``task_preferred_core_begins`` is keyed by task id. An explicit target
        may be disjoint from the selected W13 interval, in which case the
        runtime migrates W2 after acquiring the complete destination team.
        """
        if numa_node < 0:
            raise ValueError(f"numa_node must be non-negative, got {numa_node}")
        if resize_timeout_ns < 0:
            raise ValueError(f"resize_timeout_ns must be non-negative, got {resize_timeout_ns}")
        transitions = {int(width): int(preferred) for width, preferred in width_transitions.items()}
        for width, preferred in transitions.items():
            if width <= 0 or preferred <= width or preferred % width != 0:
                raise ValueError(
                    "elastic W2 transitions must map a positive width to a larger multiple: "
                    f"{width}->{preferred}"
                )

        bridge = self.to_async_bridge(tasks)
        selected = [int(width) for width in bridge["task_threads"]]
        num_tasks = len(selected)
        eligible_tasks = (
            set(range(num_tasks))
            if resizable_task_ids is None
            else {int(task) for task in resizable_task_ids}
        )
        if any(task < 0 or task >= num_tasks for task in eligible_tasks):
            raise ValueError("resizable_task_ids contains an out-of-range task id")
        explicit_core_begins = {
            int(task): int(core_begin)
            for task, core_begin in (task_preferred_core_begins or {}).items()
        }
        if any(task < 0 or task >= num_tasks for task in explicit_core_begins):
            raise ValueError("task_preferred_core_begins contains an out-of-range task id")
        if any(task not in eligible_tasks for task in explicit_core_begins):
            raise ValueError("task_preferred_core_begins may only target resizable tasks")
        preferred = [
            transitions.get(width, width) if task in eligible_tasks else width
            for task, width in enumerate(selected)
        ]
        if any(preferred[task] == selected[task] for task in explicit_core_begins):
            raise ValueError(
                "task_preferred_core_begins may only target tasks whose selected width has an elastic transition"
            )
        allowed_offsets = [0]
        allowed_widths: list[int] = []
        min_widths: list[int] = []
        max_widths: list[int] = []
        resize_points: list[int] = []
        task_numa_nodes: list[int] = []
        timeouts: list[int] = []
        preferred_core_begins: list[int] = []
        for task, (core_begin, width, target) in enumerate(
            zip(bridge["task_core_begins"], selected, preferred, strict=True)
        ):
            widths = [width] if target == width else [width, target]
            allowed_widths.extend(widths)
            allowed_offsets.append(len(allowed_widths))
            min_widths.append(widths[0])
            max_widths.append(widths[-1])
            if target == width:
                resize_points.append(_ASYNC_RESIZE_NONE)
                task_numa_nodes.append(-1)
                timeouts.append(0)
                preferred_core_begins.append(-1)
                continue
            cohort_begin = explicit_core_begins.get(
                task,
                int(core_begin) // target * target,
            )
            if cohort_begin < 0 or cohort_begin + target > self.num_cores:
                raise ValueError(
                    f"task {task} preferred cohort [{cohort_begin}, {cohort_begin + target}) "
                    f"exceeds num_cores={self.num_cores}"
                )
            if cohort_begin % target != 0:
                raise ValueError(
                    f"task {task} preferred cohort must align to width {target}: "
                    f"core_begin={cohort_begin}"
                )
            source_end = int(core_begin) + width
            target_end = cohort_begin + target
            contains_source = cohort_begin <= int(core_begin) and source_end <= target_end
            disjoint_source = source_end <= cohort_begin or target_end <= int(core_begin)
            if not contains_source and not disjoint_source:
                raise ValueError(
                    f"task {task} preferred cohort must contain or be disjoint from "
                    f"the selected interval [{core_begin}, {source_end})"
                )
            resize_points.append(_ASYNC_RESIZE_BEFORE_W2)
            task_numa_nodes.append(int(numa_node))
            timeouts.append(int(resize_timeout_ns))
            preferred_core_begins.append(int(cohort_begin))

        w13_ranges, w2_ranges = self._task_stage_ranges(tasks, selected, preferred)
        bridge.update(
            {
                "execution_mode": _ASYNC_EXECUTION_ELASTIC,
                "task_preferred_threads": preferred,
                "task_min_threads": min_widths,
                "task_max_threads": max_widths,
                "task_allowed_thread_offsets": allowed_offsets,
                "task_allowed_threads": allowed_widths,
                "task_numa_nodes": task_numa_nodes,
                "task_resize_points": resize_points,
                "task_w13_ranges": w13_ranges,
                "task_w2_ranges": w2_ranges,
                "task_resize_timeout_ns": timeouts,
                "task_preferred_core_begins": preferred_core_begins,
            }
        )
        return bridge

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
        task_w13_ranges, task_w2_ranges = self._task_stage_ranges(tasks, task_threads)
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
            "task_w13_ranges": task_w13_ranges,
            "task_w2_ranges": task_w2_ranges,
            "task_release_ns": [0] * num_tasks,
            "task_resize_timeout_ns": [0] * num_tasks,
            "task_preferred_core_begins": [-1] * num_tasks,
            "early_merge": None,
        }


class PlannedTwoStagePlanner:
    """Independently search W13 and W2 expert-team plans.

    The returned plans retain the Plan V2 bridge format, but their task
    lifetimes are stage-local and a global W13-to-W2 barrier is explicit.
    """

    def __init__(
        self,
        model: PlannerCostModel,
        num_cores: int,
        widths: Sequence[int] | None = None,
        *,
        cpu_ids: Sequence[int] | None = None,
        shapes: Sequence[Sequence[int]] | None = None,
        task_stage_window_policy: TaskStageWindowPolicy | None = None,
    ):
        common = {
            "widths": widths,
            "cpu_ids": cpu_ids,
            "shapes": shapes,
            "native_cold_planner": False,
            "task_stage_window_policy": task_stage_window_policy,
        }
        self.model = model
        self.w13_planner = IntervalPlanner(model, num_cores, stage="w13", **common)
        self.w2_planner = IntervalPlanner(model, num_cores, stage="w2", **common)

    def plan(
        self,
        experts: List[Tuple[int, int]],
        *,
        dynamic_tail_pool: bool = True,
        tail_pool_max_routes: int = 12,
        forced_tail_pool_threads: int | None = None,
        bounded_tail_repartition: bool | None = None,
    ) -> Dict[str, object]:
        options = {
            "dynamic_tail_pool": dynamic_tail_pool,
            "tail_pool_max_routes": tail_pool_max_routes,
            "forced_tail_pool_threads": forced_tail_pool_threads,
            "bounded_tail_repartition": bounded_tail_repartition,
        }
        w13 = self.w13_planner.plan(experts, **options)
        w2 = self.w2_planner.plan(experts, **options)
        call_setup_ns = float(getattr(self.model, "call_setup_ns", 0.0))
        stage_barrier_ns = 0.0
        return {
            "plan_version": _ASYNC_PLAN_VERSION,
            "execution_mode": "planned_staged",
            "w13": w13,
            "w2": w2,
            "call_setup_ns": call_setup_ns,
            "stage_barrier_ns": stage_barrier_ns,
            "makespan_ns": call_setup_ns + w13["makespan_ns"] + stage_barrier_ns + w2["makespan_ns"],
            "uncertainty_ns": w13["uncertainty_ns"] + w2["uncertainty_ns"],
        }


if __name__ == "__main__":
    model = ContentionCostModel(sys.argv[1])
    planner = IntervalPlanner(model, num_cores=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    result = planner.plan([(index, 192) for index in range(8)])
    print(result)
