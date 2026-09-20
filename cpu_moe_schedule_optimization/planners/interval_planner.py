"""Static interval-DAG planner with exact-domain calibration."""

from __future__ import annotations

import importlib
import math
import os
import sys
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Tuple

try:
    from ..cost_model.phase_model import ContentionCostModel
    from ..cost_model.profile_catalog import ProfileCompatibilityError
    from .stage_window_policy import FULL_STRIPE as _FULL_STRIPE_WINDOW
    from .stage_window_policy import default_stage_window_policy
except ImportError:  # pragma: no cover - direct script and legacy path import
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
    from phase_model import ContentionCostModel  # noqa: E402
    from profile_catalog import ProfileCompatibilityError  # noqa: E402
    from stage_window_policy import FULL_STRIPE as _FULL_STRIPE_WINDOW  # noqa: E402
    from stage_window_policy import default_stage_window_policy  # noqa: E402


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
_BOUNDED_TAIL_TASKS = 2
_BOUNDED_TAIL_DEFAULT_CORES = 96
_TEMPORAL_ASSIGNMENT_REL_TOL = 1e-10
_TEMPORAL_ASSIGNMENT_ABS_NS = 1e-6
_ASSIGNMENT_ORDER_LPT = "lpt"
_ASSIGNMENT_ORDER_REVERSE_ODD = "reverse_odd"
_ASSIGNMENT_ORDER_REVERSE_EVEN = "reverse_even"
# Sentinel so a resolved-to-None stage-window policy is only looked up once.
_UNSET = object()


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

    def stage_bytes_per_worker(self, stage: str, threads: int, routes: int | None = None) -> int: ...

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
        tail_repartition_widths: Sequence[int] | None = None,
        stage: str | None = None,
        stage_window_policy=_UNSET,
    ):
        if stage not in {None, "w13", "w2"}:
            raise ValueError(f"stage must be None, 'w13', or 'w2', got {stage!r}")
        self.stage = stage
        self.model = model
        # _UNSET resolves the calibrated policy for the model's shape; None forces
        # the full stripe; an explicit policy overrides both (opt-in tables).
        self._stage_window_policy_cached = stage_window_policy
        self._shared_quick_cost_cache: dict[tuple[int, int], float] = {}
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

        self._shapes_explicit = shapes is not None
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
            self._native_quick_planner = self._create_native_quick_planner(planner_threads)
        else:
            if native_cold_planner is True:
                raise ValueError("native cold planner does not yet support stage-specific scoring")
            self._native_planner = None
            self._native_quick_planner = None

    def _create_native_quick_planner(self, planner_threads):
        exporter = getattr(self.model, "native_quick_planner_payload", None)
        if not callable(exporter):
            return None
        native_type = None
        for module_name in ("fused_cpp._moe_C", "fused_cpp._C"):
            try:
                native_type = importlib.import_module(module_name).NativeQuickPlanner
                break
            except (ImportError, AttributeError):
                continue
        if native_type is None:
            return None
        homogeneous_shapes = [shape for shape in self.shapes if len(set(shape)) == 1]
        if not homogeneous_shapes:
            return None
        if planner_threads is None:
            configured_threads = os.environ.get("FUSED_CPP_MOE_PLANNER_THREADS")
            planner_threads = 1 if configured_threads is None else int(configured_threads)
        if planner_threads < 0:
            raise ValueError(f"planner_threads must be non-negative, got {planner_threads}")
        payload = exporter()
        return native_type(
            self.num_cores,
            [list(shape) for shape in homogeneous_shapes],
            int(payload["max_stage_bytes"]),
            payload["window_bytes_by_width"],
            float(payload["relative_error"]),
            int(payload["profile_runs"]),
            planner_threads,
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

    def _assign_lpt(self, experts, lanes):
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

    def _assign_homogeneous_lpt(self, experts, lanes):
        """Assign a homogeneous shape and return its exact isolated lane loads."""
        if not lanes or len({width for _, width in lanes}) != 1:
            raise ValueError("homogeneous LPT requires lanes with one shared width")
        width = lanes[0][1]
        task_times = {
            routes: self._task_time(routes, width)
            for routes in dict.fromkeys(routes for _, routes in experts)
        }
        availability = [(0.0, lane) for lane in range(len(lanes))]
        heapify(availability)
        lane_experts: List[List[int]] = [[] for _ in lanes]
        order = sorted(range(len(experts)), key=lambda index: -experts[index][1])
        for index in order:
            task_time = task_times[experts[index][1]]
            load, lane = heappop(availability)
            score = load + task_time
            tied = []
            while availability and availability[0][0] + task_time == score:
                tied.append(heappop(availability))
            if tied:
                tied.append((load, lane))
                load, lane = min(tied, key=lambda item: item[1])
                for candidate in tied:
                    if candidate[1] != lane:
                        heappush(availability, candidate)
            lane_experts[lane].append(index)
            heappush(availability, (score, lane))
        lane_loads = [0.0] * len(lanes)
        for load, lane in availability:
            lane_loads[lane] = load
        return lane_experts, lane_loads

    def _quick_homogeneous_scale(self, shape: Sequence[int]) -> float:
        """Apply calibrated wide-team dilation to one homogeneous shape.

        Quick still uses isolated LPT packing. The scale only ranks widths; it
        is a whole-task proxy of the event-path GEMM occupancy term and is 1
        when the profile has no wide-team table. A model exposing
        ``quick_homogeneous_scale`` supplies the scale itself.
        """
        signature = tuple(int(part) for part in shape)
        if not signature:
            return 1.0
        occupied = min(sum(signature), self.num_cores)
        model_scale = getattr(self.model, "quick_homogeneous_scale", None)
        if callable(model_scale):
            return float(model_scale(int(signature[0]), occupied, self.num_cores))
        pressure = getattr(getattr(self.model, "calibration", None), "wide_team_pressure", None)
        occupancy = getattr(pressure, "occupancy_scale", None)
        if occupancy is None:
            return 1.0
        return float(occupancy(int(signature[0]), occupied, self.num_cores))

    def _quick_cost_rows(self, experts, shapes):
        rows = []
        for shape in shapes:
            width = int(shape[0])
            scale = self._quick_homogeneous_scale(shape)
            task_times = {
                routes: self._task_time(routes, width) * scale
                for routes in dict.fromkeys(routes for _, routes in experts)
            }
            rows.append([task_times[routes] for _, routes in experts])
        return rows

    def quick_tasks_for_shape(self, experts, shape):
        """Build exact homogeneous LPT tasks using the native helper when available."""
        signature = tuple(int(value) for value in shape)
        lanes = self._lanes(signature)
        if self._native_quick_planner is None:
            assignment, _ = self._assign_homogeneous_lpt(experts, lanes)
            return self._build_tasks(experts, lanes, assignment)
        costs = self._quick_cost_rows(experts, (signature,))[0]
        result = self._native_quick_planner.assign(
            [expert for expert, _ in experts],
            [routes for _, routes in experts],
            list(signature),
            costs,
        )
        return result["tasks"]

    def _shared_quick_shapes(self) -> tuple[tuple[int, ...], ...]:
        candidates: set[tuple[int, ...]] = set()
        for shared_width in self.widths:
            if shared_width == self.num_cores:
                candidates.add((shared_width,))
                continue
            remaining = self.num_cores - shared_width
            if remaining <= 0:
                continue
            for routed_width in self.widths:
                if (
                    routed_width <= shared_width
                    and routed_width <= remaining
                    and remaining % routed_width == 0
                ):
                    candidates.add((shared_width,) + (routed_width,) * (remaining // routed_width))
        return tuple(sorted(candidates, key=lambda shape: (shape[0], shape[1] if len(shape) > 1 else shape[0], shape)))

    def _assign_shared_lpt(self, experts, lanes, shared_expert_id: int):
        shared_indices = [index for index, (expert, _) in enumerate(experts) if expert == shared_expert_id]
        if len(shared_indices) != 1:
            raise ValueError("shared-aware quick planning requires exactly one active synthetic shared expert")
        shared_index = shared_indices[0]
        lane_experts: List[List[int]] = [[] for _ in lanes]
        lane_experts[0].append(shared_index)
        lane_loads = [0.0] * len(lanes)
        lane_loads[0] = self._shared_quick_task_time(experts[shared_index][1], lanes[0][1])
        availability: dict[int, list[tuple[float, int]]] = {}
        for lane, (_, width) in enumerate(lanes):
            availability.setdefault(width, []).append((lane_loads[lane], lane))
        for heap in availability.values():
            heapify(heap)
        order = sorted(
            (index for index in range(len(experts)) if index != shared_index),
            key=lambda index: (-experts[index][1], index),
        )
        for index in order:
            routes = experts[index][1]
            _, lane, width = min(
                (
                    heap[0][0] + self._shared_quick_task_time(routes, width),
                    heap[0][1],
                    width,
                )
                for width, heap in availability.items()
            )
            load, selected_lane = heappop(availability[width])
            assert selected_lane == lane
            lane_experts[lane].append(index)
            lane_loads[lane] = load + self._shared_quick_task_time(routes, width)
            heappush(availability[width], (lane_loads[lane], lane))
        return lane_experts, lane_loads

    def _shared_quick_task_time(self, routes: int, threads: int) -> float:
        key = (int(routes), int(threads))
        cached = self._shared_quick_cost_cache.get(key)
        if cached is None:
            cached = self._task_time(*key)
            self._shared_quick_cost_cache[key] = cached
        return cached

    def _shared_quick_cost_rows(self, experts, shapes):
        widths = sorted({int(width) for shape in shapes for width in shape})
        rows = [
            [self._shared_quick_task_time(routes, width) for _, routes in experts]
            for width in widths
        ]
        return widths, rows

    def quick_tasks_for_shared_shape(self, experts, shape, shared_expert_id: int):
        signature = tuple(int(value) for value in shape)
        lanes = self._lanes(signature)
        assignment, _ = self._assign_shared_lpt(experts, lanes, shared_expert_id)
        return self._build_tasks(experts, lanes, assignment)

    @staticmethod
    def _reverse_lanes(lane_experts, parity: int):
        return [
            list(reversed(expert_indices)) if lane % 2 == parity else list(expert_indices)
            for lane, expert_indices in enumerate(lane_experts)
        ]

    def _assignment_for_order(self, lpt_assignment, assignment_order: str):
        if assignment_order == _ASSIGNMENT_ORDER_LPT:
            return [list(expert_indices) for expert_indices in lpt_assignment]
        if assignment_order == _ASSIGNMENT_ORDER_REVERSE_ODD:
            return self._reverse_lanes(lpt_assignment, 1)
        if assignment_order == _ASSIGNMENT_ORDER_REVERSE_EVEN:
            return self._reverse_lanes(lpt_assignment, 0)
        raise ValueError(f"unknown assignment order: {assignment_order}")

    @staticmethod
    def _assignment_better(candidate: float, reference: float) -> bool:
        tolerance = max(
            _TEMPORAL_ASSIGNMENT_ABS_NS,
            abs(reference) * _TEMPORAL_ASSIGNMENT_REL_TOL,
        )
        return candidate < reference - tolerance

    def _temporal_assignment(self, experts, lanes, lpt_assignment):
        """Greedily orient fixed LPT lanes using the full event-time model.

        Lane membership, width, core interval, and isolated lane work remain
        unchanged. Reversing selected lanes advances their smaller experts and
        delays their larger experts, allowing the DAG model to overlap unlike
        resource phases without adding idling or a new runtime contract.
        """

        def score(assignment):
            return self._score(self._build_tasks(experts, lanes, assignment))

        best = [list(expert_indices) for expert_indices in lpt_assignment]
        best_score = score(best)
        best_order = _ASSIGNMENT_ORDER_LPT
        if (
            len(lanes) <= 1
            or all(len(expert_indices) <= 1 for expert_indices in lpt_assignment)
            or len({routes for _, routes in experts}) <= 1
        ):
            return best, best_score, best_order

        # The two checkerboard seeds can express a mixed first wave even when
        # no single-lane reversal is independently profitable.
        for parity in (1, 0):
            candidate = self._reverse_lanes(lpt_assignment, parity)
            candidate_score = score(candidate)
            if self._assignment_better(candidate_score, best_score):
                best = candidate
                best_score = candidate_score
                best_order = (
                    _ASSIGNMENT_ORDER_REVERSE_ODD if parity == 1 else _ASSIGNMENT_ORDER_REVERSE_EVEN
                )

        return best, best_score, best_order

    def _select_assignment(self, experts, lanes):
        lpt_assignment = self._assign_lpt(experts, lanes)
        shape = tuple(width for _, width in lanes)
        if self._uses_full_workload_anchor(experts, shape):
            return lpt_assignment, None, _ASSIGNMENT_ORDER_LPT
        return self._temporal_assignment(experts, lanes, lpt_assignment)

    def _assign(self, experts, lanes):
        assignment, _, _ = self._select_assignment(experts, lanes)
        return assignment

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
        placed = getattr(self.model, "dag_makespan_placed", None)
        if self.stage is None and callable(placed):
            return placed(
                [
                    (
                        routes,
                        threads,
                        self.cpu_ids[core_begin : core_begin + threads],
                        deps,
                    )
                    for _, routes, core_begin, threads, deps in tasks
                ]
            )
        return self._dag_makespan([(routes, threads, deps) for _, routes, _, threads, deps in tasks])

    def _task_time(self, routes: int, threads: int) -> float:
        """Task time under the windows the plan will carry.

        The cost model prices the full stripe. A window policy with measured
        full-load time scales turns that into T*(M, t) = T(M, t) * r(t, M), so
        width selection sees the windows the lowering emits. Stage planners and
        policies without scales keep r = 1.
        """
        if self.stage is None:
            return self.model.T_iso(routes, threads) * self._window_time_scale(routes, threads)
        return self.model.stage_T_iso(self.stage, routes, threads)

    def _window_time_scale(self, routes: int, threads: int) -> float:
        policy = self._stage_window_policy()
        scale = getattr(policy, "time_scale", None)
        return 1.0 if scale is None else float(scale(int(routes), int(threads)))

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
        independent_analytic_samples: bool = False,
    ) -> float:
        if self.model.schema_version < 2:
            return 0.0
        if use_full_workload_anchor and self._uses_full_workload_anchor(experts, shape):
            relative = self.model.relative_full_call_uncertainty(experts[0][1], shape)
            return makespan * relative / math.sqrt(self.model.profile_runs)
        constant_relative = getattr(self.model, "relative_error", None)
        if constant_relative is None:
            relative = max(
                self.model.relative_uncertainty(routes, shape)
                for routes in dict.fromkeys(routes for _, routes in experts)
            )
        else:
            relative = float(constant_relative)
            if getattr(self.model, "iso_mode", None) == "analytic" and not independent_analytic_samples:
                # Analytical calibration error is systematic model error. More
                # experts or repeated probe runs do not make it independent.
                return makespan * relative
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
            resolver = getattr(self.model, "stage_bytes_per_worker", None)
            if callable(resolver):
                return int(resolver(self.stage, threads, routes))
        resolver = self.model.window_bytes_per_worker
        return int(resolver(threads, routes))

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
                    self.model.stage_bytes_per_worker(self.stage, int(width)) for width in shape
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

    def _score_shape_with_order(self, experts, shape):
        signature = tuple(int(value) for value in shape)
        if self.model.schema_version >= 2 and not self.model.supports_shape(signature):
            raise ProfileCompatibilityError(f"shape {signature} is not supported by {self.model.profile_path.name}")
        lanes = self._lanes(signature)
        assignment, assignment_score, assignment_order = self._select_assignment(experts, lanes)
        tasks = self._build_tasks(experts, lanes, assignment)
        if self._uses_full_workload_anchor(experts, signature):
            return self.model.profiled_full_call_time(experts[0][1], signature), tasks, assignment_order
        assert assignment_score is not None
        return assignment_score, tasks, assignment_order

    def score_shape(self, experts, shape):
        makespan, tasks, _ = self._score_shape_with_order(experts, shape)
        return makespan, tasks

    def _candidate(self, experts, shape) -> dict:
        makespan, tasks, assignment_order = self._score_shape_with_order(experts, shape)
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
            "assignment_order": assignment_order,
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": tasks,
            "active_working_set_bytes": self.active_working_set_bytes(shape, tasks),
            "window_bytes_per_worker": self.window_bytes_per_worker(shape, tasks),
            "resource_groups": resource_groups,
        }

    def _quick_candidate(self, experts, shape) -> dict:
        """Score one homogeneous shape with isolated LPT lane loads.

        Packing stays isolated-LPT. Width ranking multiplies the lane makespan
        by the calibrated wide-team occupancy scale when present. This is not
        the event simulator; full search remains the mixed-width oracle.
        """
        signature = tuple(int(value) for value in shape)
        if len(set(signature)) != 1:
            raise ValueError("quick planning accepts only homogeneous shapes")
        lanes = self._lanes(signature)
        assignment, lane_loads = self._assign_homogeneous_lpt(experts, lanes)
        tasks = self._build_tasks(experts, lanes, assignment)
        makespan = max(lane_loads, default=0.0) * self._quick_homogeneous_scale(signature)
        uncertainty = self._uncertainty(
            experts,
            signature,
            makespan,
            use_full_workload_anchor=False,
            independent_analytic_samples=True,
        )
        return {
            "shape": signature,
            "execution_mode": _ASYNC_EXECUTION_STRICT,
            "tail_pool_threads": None,
            "tail_pool_max_routes": None,
            "tail_pool_tasks": 0,
            "tail_repartition_width": None,
            "tail_repartition_tasks": 0,
            "tail_repartition_route_slices": 1,
            "assignment_order": _ASSIGNMENT_ORDER_LPT,
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": tasks,
            "active_working_set_bytes": self.active_working_set_bytes(signature, tasks),
            "window_bytes_per_worker": self.window_bytes_per_worker(signature, tasks),
            "resource_groups": len(lanes),
        }

    def _uses_analytic_full(self) -> bool:
        return (
            self.stage is None
            and not self._shapes_explicit
            and getattr(self.model, "iso_mode", None) == "analytic"
        )

    def _analytic_full_baseline(self, experts) -> dict:
        """Rescore the quick winner as an additional full-search candidate.

        Full search evaluates the complete modeled candidate space. Keeping the
        quick winner in that space makes the two objectives directly comparable
        and guarantees that the model-optimal full score cannot exceed quick.
        """
        homogeneous_shapes = [shape for shape in self.shapes if len(set(shape)) == 1]
        if not homogeneous_shapes:
            raise ProfileCompatibilityError("analytic full search requires at least one homogeneous shape")
        quick_candidates = [self._quick_candidate(experts, shape) for shape in homogeneous_shapes]
        baseline = min(
            quick_candidates,
            key=lambda candidate: (
                candidate["makespan_ns"],
                candidate["active_working_set_bytes"],
                candidate["resource_groups"],
            ),
        )
        baseline["makespan_ns"] = self._score(baseline["tasks"])
        baseline["uncertainty_ns"] = self._uncertainty(
            experts,
            baseline["shape"],
            baseline["makespan_ns"],
            use_full_workload_anchor=False,
        )
        baseline["pessimistic_ns"] = baseline["makespan_ns"] + baseline["uncertainty_ns"]
        return baseline

    @staticmethod
    def _select_analytic_full(candidates: Sequence[dict]) -> dict:
        """Select expected-best full plan with one bounded uncertainty fallback."""
        if not candidates:
            raise ValueError("analytical full selection requires at least one candidate")
        fastest = min(
            candidates,
            key=lambda candidate: (
                candidate["makespan_ns"],
                candidate["pessimistic_ns"],
                candidate["active_working_set_bytes"],
                candidate.get("resource_groups", len(candidate["shape"])),
            ),
        )
        fastest_width = max(int(width) for width in fastest["shape"])
        if fastest_width <= 8:
            return fastest
        narrower_widths = sorted(
            {
                max(int(width) for width in candidate["shape"])
                for candidate in candidates
                if max(int(width) for width in candidate["shape"]) < fastest_width
            }
        )
        if not narrower_widths:
            return fastest
        target_width = narrower_widths[-1]
        fastest_lower = fastest["makespan_ns"] - fastest["uncertainty_ns"]
        fastest_upper = fastest["pessimistic_ns"]
        narrowed = [
            candidate
            for candidate in candidates
            if max(int(width) for width in candidate["shape"]) <= target_width
            and candidate["makespan_ns"] - candidate["uncertainty_ns"] <= fastest_upper
            and candidate["pessimistic_ns"] >= fastest_lower
        ]
        if not narrowed:
            return fastest
        return min(
            narrowed,
            key=lambda candidate: (
                candidate["makespan_ns"],
                candidate["pessimistic_ns"],
                candidate["active_working_set_bytes"],
                candidate.get("resource_groups", len(candidate["shape"])),
            ),
        )

    def _quick_shared_candidate(self, experts, shape, shared_expert_id: int) -> dict:
        signature = tuple(int(value) for value in shape)
        lanes = self._lanes(signature)
        assignment, lane_loads = self._assign_shared_lpt(experts, lanes, shared_expert_id)
        tasks = self._build_tasks(experts, lanes, assignment)
        makespan = max(lane_loads, default=0.0)
        uncertainty = self._uncertainty(
            experts,
            signature,
            makespan,
            use_full_workload_anchor=False,
            independent_analytic_samples=True,
        )
        return {
            "shape": signature,
            "execution_mode": _ASYNC_EXECUTION_STRICT,
            "tail_pool_threads": None,
            "tail_pool_max_routes": None,
            "tail_pool_tasks": 0,
            "tail_repartition_width": None,
            "tail_repartition_tasks": 0,
            "tail_repartition_route_slices": 1,
            "assignment_order": _ASSIGNMENT_ORDER_LPT,
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": tasks,
            "active_working_set_bytes": None,
            "window_bytes_per_worker": None,
            "resource_groups": len(lanes),
            "shared_expert_id": shared_expert_id,
            "shared_width": signature[0],
            "routed_width": signature[1] if len(signature) > 1 else signature[0],
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

    def _select_shared_quick(self, candidates: list[dict]) -> dict:
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
        for candidate in overlapping:
            candidate["active_working_set_bytes"] = self.active_working_set_bytes(candidate["shape"])
            candidate["window_bytes_per_worker"] = self.window_bytes_per_worker(candidate["shape"])
        return min(
            overlapping,
            key=lambda candidate: (
                candidate["active_working_set_bytes"],
                candidate["resource_groups"],
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
            "assignment_order": strict_candidate["assignment_order"],
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
            "assignment_order": strict_candidate["assignment_order"],
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
        # A pool width the cost model never measured is scored with the nearest one it did, which
        # is how one-thread pools got chosen and then ran 3.2-4.3x their predicted time
        # (tmp/search_reliability_20260920, E8). Keep the automatic search inside the calibration.
        supports_width = getattr(self.model, "supports_width", None)
        if callable(supports_width) and forced_pool_threads is None:
            widths = [width for width in widths if supports_width(int(width))]
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
        topk_ids=None,
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
            else self.to_async_bridge(selected["tasks"], topk_ids=topk_ids)
        )
        return {
            "plan_version": _ASYNC_PLAN_VERSION,
            "stage": self.stage,
            "shape": tuple(selected["shape"]),
            "assignment_order": selected.get("assignment_order", _ASSIGNMENT_ORDER_LPT),
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
            "early_merge": bridge["early_merge"],
            "policy": policy,
            "tasks": selected["tasks"],
            "bridge": bridge,
            "planner_backend": planner_backend,
            "planner_workers": planner_workers,
            "strict_candidates": strict_candidates,
            "dynamic_candidates": dynamic_candidates,
            "tail_repartition_candidates": tail_repartition_candidates,
            "shared_expert_id": selected.get("shared_expert_id"),
            "shared_width": selected.get("shared_width"),
            "routed_width": selected.get("routed_width"),
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
                    "window_bytes_per_worker": (
                        tuple(candidate["window_bytes_per_worker"])
                        if candidate["window_bytes_per_worker"] is not None
                        else None
                    ),
                    "resource_groups": candidate["resource_groups"],
                    "shared_width": candidate.get("shared_width"),
                    "routed_width": candidate.get("routed_width"),
                }
                for candidate in sorted(candidates, key=lambda candidate: candidate["makespan_ns"])
            ],
        }

    def plan(
        self,
        experts: List[Tuple[int, int]],
        *,
        topk_ids=None,
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
        analytic_full = self._uses_analytic_full()
        if self._native_planner is not None and not analytic_full:
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
                topk_ids=topk_ids,
                planner_backend="cpp",
                planner_workers=int(native["configured_workers"]),
                strict_candidates=int(native["strict_candidates"]),
                dynamic_candidates=int(native["dynamic_candidates"]),
                tail_repartition_candidates=int(native["tail_repartition_candidates"]),
            )
        strict_candidates = [self._candidate(experts, shape) for shape in self.shapes]
        analytic_baseline = self._analytic_full_baseline(experts) if analytic_full else None
        if analytic_baseline is not None:
            strict_candidates.append(analytic_baseline)
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
        selected = (
            self._select_analytic_full(candidates)
            if analytic_baseline is not None and forced_tail_pool_threads is None
            else self._select(candidates)
        )
        return self._finalize_plan(
            selected,
            candidates,
            topk_ids=topk_ids,
            planner_backend="python_analytic_full" if analytic_full else "python",
            planner_workers=1,
            strict_candidates=len(strict_candidates),
            dynamic_candidates=len(tail_pool_candidates),
            tail_repartition_candidates=len(tail_repartition_candidates),
        )

    def plan_quick(
        self,
        experts: List[Tuple[int, int]],
        *,
        topk_ids=None,
    ) -> Dict[str, object]:
        """Build a low-overhead strict plan from homogeneous team shapes."""
        experts = [(expert, routes) for expert, routes in experts if routes > 0]
        if not experts:
            raise ValueError("at least one active expert is required")
        homogeneous_shapes = [shape for shape in self.shapes if len(set(shape)) == 1]
        if not homogeneous_shapes:
            raise ProfileCompatibilityError("quick planning requires at least one homogeneous shape")
        if self._native_quick_planner is not None:
            native = self._native_quick_planner.plan(
                [expert for expert, _ in experts],
                [routes for _, routes in experts],
                self._quick_cost_rows(experts, homogeneous_shapes),
            )
            return self._finalize_plan(
                native["selected"],
                native["candidates"],
                topk_ids=topk_ids,
                planner_backend="cpp_quick",
                planner_workers=int(native["configured_workers"]),
                strict_candidates=int(native["strict_candidates"]),
                dynamic_candidates=0,
                tail_repartition_candidates=0,
            )
        candidates = [self._quick_candidate(experts, shape) for shape in homogeneous_shapes]
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate["makespan_ns"],
                candidate["active_working_set_bytes"],
                candidate["resource_groups"],
            ),
        )
        return self._finalize_plan(
            selected,
            candidates,
            topk_ids=topk_ids,
            planner_backend="python_quick",
            planner_workers=1,
            strict_candidates=len(candidates),
            dynamic_candidates=0,
            tail_repartition_candidates=0,
        )

    def plan_quick_fixed(
        self,
        experts: List[Tuple[int, int]],
        threads: int,
        *,
        topk_ids=None,
    ) -> Dict[str, object]:
        """Build one fixed-width homogeneous LPT plan without width search."""
        experts = [(expert, routes) for expert, routes in experts if routes > 0]
        if not experts:
            raise ValueError("at least one active expert is required")
        threads = int(threads)
        if threads <= 0 or self.num_cores % threads:
            raise ValueError(f"fixed threads must be a positive divisor of {self.num_cores}, got {threads}")
        shape = (threads,) * (self.num_cores // threads)
        if shape not in self.shapes:
            raise ProfileCompatibilityError(
                f"fixed {threads}T shape is not supported by {self.model.profile_path.name}"
            )
        if self._native_quick_planner is None:
            candidate = self._quick_candidate(experts, shape)
            backend = "python_fixed_quick"
            workers = 1
        else:
            candidate = self._native_quick_planner.assign(
                [expert for expert, _ in experts],
                [routes for _, routes in experts],
                list(shape),
                self._quick_cost_rows(experts, (shape,))[0],
            )
            backend = "cpp_fixed_quick"
            workers = int(self._native_quick_planner.configured_workers)
        return self._finalize_plan(
            candidate,
            [candidate],
            topk_ids=topk_ids,
            planner_backend=backend,
            planner_workers=workers,
            strict_candidates=1,
            dynamic_candidates=0,
            tail_repartition_candidates=0,
        )

    def plan_quick_with_shared(
        self,
        experts: List[Tuple[int, int]],
        *,
        shared_expert_id: int,
        topk_ids=None,
        allowed_widths: Sequence[int] | None = None,
        homogeneous_only: bool = False,
    ) -> Dict[str, object]:
        """Build a bounded mixed-width plan with one all-token synthetic expert."""
        experts = [(expert, routes) for expert, routes in experts if routes > 0]
        if not experts:
            raise ValueError("at least one active expert is required")
        shared_matches = [
            routes for expert, routes in experts if expert == int(shared_expert_id)
        ]
        if len(shared_matches) != 1:
            raise ValueError(
                "shared-aware quick planning requires exactly one active synthetic shared expert"
            )
        shapes = self._shared_quick_shapes()
        if allowed_widths is not None:
            allowed = {int(width) for width in allowed_widths}
            shapes = tuple(
                shape for shape in shapes if all(int(width) in allowed for width in shape)
            )
        if homogeneous_only:
            shapes = tuple(shape for shape in shapes if len(set(shape)) == 1)
        if len(experts) > 2:
            shapes = tuple(shape for shape in shapes if len(shape) > 1)
            shared_routes = shared_matches[0]
            routed_routes = sum(
                routes for expert, routes in experts if expert != int(shared_expert_id)
            )
            if routed_routes >= shared_routes:
                shapes = tuple(
                    shape for shape in shapes if int(shape[0]) <= self.num_cores // 2
                )
        if not shapes:
            raise ProfileCompatibilityError("shared-aware quick planning has no legal shape")
        if self._native_quick_planner is not None and hasattr(self._native_quick_planner, "plan_shared"):
            widths, cost_rows = self._shared_quick_cost_rows(experts, shapes)
            native = self._native_quick_planner.plan_shared(
                [expert for expert, _ in experts],
                [routes for _, routes in experts],
                int(shared_expert_id),
                [list(shape) for shape in shapes],
                widths,
                cost_rows,
            )
            selected = native["selected"]
            selected["shared_expert_id"] = int(shared_expert_id)
            selected["shared_width"] = int(selected["shape"][0])
            selected["routed_width"] = int(
                selected["shape"][1] if len(selected["shape"]) > 1 else selected["shape"][0]
            )
            for candidate in native["candidates"]:
                candidate["shared_width"] = int(candidate["shape"][0])
                candidate["routed_width"] = int(
                    candidate["shape"][1] if len(candidate["shape"]) > 1 else candidate["shape"][0]
                )
            return self._finalize_plan(
                selected,
                native["candidates"],
                topk_ids=topk_ids,
                planner_backend="cpp_shared_quick",
                planner_workers=int(native["configured_workers"]),
                strict_candidates=int(native["strict_candidates"]),
                dynamic_candidates=0,
                tail_repartition_candidates=0,
            )
        candidates = [
            self._quick_shared_candidate(experts, shape, int(shared_expert_id))
            for shape in shapes
        ]
        selected = self._select_shared_quick(candidates)
        return self._finalize_plan(
            selected,
            candidates,
            topk_ids=topk_ids,
            planner_backend="python_shared_quick",
            planner_workers=1,
            strict_candidates=len(candidates),
            dynamic_candidates=0,
            tail_repartition_candidates=0,
        )

    def _early_merge_decision(self, tasks, topk_ids=None) -> tuple[bool, dict[str, object]]:
        """Enable ready-token merge without running the analytical DAG gate."""
        return True, {"reason": "fixed_on"}

    def _early_merge_policy(self, tasks, topk_ids=None) -> bool | None:
        return self._early_merge_decision(tasks, topk_ids)[0]

    def to_async_bridge(self, tasks, *, topk_ids=None) -> Dict[str, object]:
        dependency_offsets, flat_dependencies = [0], []
        for _, _, _, _, dependencies in tasks:
            flat_dependencies.extend(dependencies)
            dependency_offsets.append(len(flat_dependencies))
        task_threads = [threads for _, _, _, threads, _ in tasks]
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
        w13_windows, w2_windows = self._stage_windows(tasks, task_threads)
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
            "task_w13_window_tiles": w13_windows,
            "task_w2_window_tiles": w2_windows,
            "early_merge": self._early_merge_policy(tasks, topk_ids),
        }

    def _stage_windows(self, tasks, task_threads) -> tuple[list[int], list[int]]:
        """Per-task `(w13, w2)` owner windows in whole N tiles.

        A deterministic read of the calibrated table at the already-selected
        `(routes, threads)`, so this adds no search dimension. Shapes with no
        calibrated policy get the full stripe, which is the pre-window geometry.
        """
        num_tasks = len(tasks)
        policy = self._stage_window_policy()
        if policy is None:
            return ([_FULL_STRIPE_WINDOW] * num_tasks, [_FULL_STRIPE_WINDOW] * num_tasks)
        w13_windows: list[int] = []
        w2_windows: list[int] = []
        for (_, routes, _, _, _), threads in zip(tasks, task_threads):
            w13, w2 = policy.select(int(routes), int(threads))
            w13_windows.append(int(w13))
            w2_windows.append(int(w2))
        return (w13_windows, w2_windows)

    def _stage_window_policy(self):
        if self._stage_window_policy_cached is not _UNSET:
            return self._stage_window_policy_cached
        policy = getattr(self.model, "policy", None)
        resolved = None
        if policy is not None:
            hidden_size = getattr(policy, "hidden_size", None)
            intermediate_size = getattr(policy, "intermediate_size", None)
            backend_n_tile = getattr(policy, "backend_n_tile", None)
            if None not in (hidden_size, intermediate_size, backend_n_tile):
                resolved = default_stage_window_policy(
                    hidden_size=int(hidden_size),
                    intermediate_size=int(intermediate_size),
                    backend_n_tile=int(backend_n_tile),
                    machine_id=getattr(getattr(self.model, "calibration", None), "machine_id", None),
                )
        self._stage_window_policy_cached = resolved
        return resolved

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
        num_tasks = len(tasks)
        tail_pool_w13_windows, tail_pool_w2_windows = self._stage_windows(tasks, task_threads)
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
            "task_w13_window_tiles": tail_pool_w13_windows,
            "task_w2_window_tiles": tail_pool_w2_windows,
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
    ):
        common = {
            "widths": widths,
            "cpu_ids": cpu_ids,
            "shapes": shapes,
            "native_cold_planner": False,
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
