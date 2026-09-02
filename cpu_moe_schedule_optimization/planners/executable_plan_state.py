"""Canonical executable strict-plan state for event-guided MoE search.

The representation intentionally covers the current fixed-lane full/greedy
planner domain.  It is not a replacement for public Plan V2 and does not model
tail-pool, route-sliced, resize, or arbitrary interval-DAG plans.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


EXECUTABLE_PLAN_STATE_VERSION = 1
_PLAN_VERSION = 2
_EXECUTION_STRICT = "strict"
_PLACEMENT_FIXED = 0
_NUMA_UNSPECIFIED = -1
_STAGE_EXPERT = 0
_RESIZE_NONE = 0
_FULL_EXPERT_RANGE = 0

PlannerTask = tuple[int, int, int, int, list[int]]


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a one-dimensional integer sequence")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain integers") from error


@dataclass(frozen=True, order=True)
class ExecutableLlcDomain:
    """One contiguous logical-core interval belonging to an LLC domain."""

    domain_id: str
    core_begin: int
    core_count: int

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("LLC domain id must be non-empty")
        if self.core_begin < 0 or self.core_count <= 0:
            raise ValueError("LLC domain interval must be non-negative and non-empty")

    @property
    def core_end(self) -> int:
        return self.core_begin + self.core_count


@dataclass(frozen=True)
class ExecutableExpertTask:
    """One unsliced whole-expert task inside a serial lane."""

    expert_id: int
    routes: int
    w13_window_tiles: int = 0
    w2_window_tiles: int = 0

    def __post_init__(self) -> None:
        if self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")
        if self.routes <= 0:
            raise ValueError("routes must be positive")
        if min(self.w13_window_tiles, self.w2_window_tiles) < 0:
            raise ValueError("stage window tiles must be non-negative")


@dataclass(frozen=True)
class ExecutableLane:
    """One contiguous fixed-width core interval and its serial task chain."""

    core_begin: int
    threads: int
    tasks: tuple[ExecutableExpertTask, ...] = ()

    def __post_init__(self) -> None:
        if self.core_begin < 0 or self.threads <= 0:
            raise ValueError("lane interval must be non-negative and non-empty")
        object.__setattr__(self, "tasks", tuple(self.tasks))

    @property
    def core_end(self) -> int:
        return self.core_begin + self.threads


@dataclass(frozen=True)
class ExecutablePlanState:
    """Canonical fixed-lane state used by future VND/LNS search.

    Lanes are serialized in increasing logical-core order and tasks are
    serialized lane by lane.  Dependencies are therefore derived rather than
    stored: every task after the first in a lane depends on its predecessor.
    """

    num_threads: int
    thread_cpu_ids: tuple[int, ...]
    lanes: tuple[ExecutableLane, ...]
    llc_domains: tuple[ExecutableLlcDomain, ...] = ()
    early_merge: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "thread_cpu_ids", tuple(int(cpu) for cpu in self.thread_cpu_ids))
        object.__setattr__(self, "lanes", tuple(self.lanes))
        object.__setattr__(self, "llc_domains", tuple(self.llc_domains))
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if len(self.thread_cpu_ids) != self.num_threads:
            raise ValueError("thread_cpu_ids must contain exactly num_threads entries")
        if min(self.thread_cpu_ids) < 0 or len(set(self.thread_cpu_ids)) != self.num_threads:
            raise ValueError("thread_cpu_ids must be unique and non-negative")
        if self.early_merge is not None and type(self.early_merge) is not bool:
            raise TypeError("early_merge must be a bool or None")
        self._validate_partition(self.lanes, self.num_threads, "lanes")
        if self.llc_domains:
            if len({domain.domain_id for domain in self.llc_domains}) != len(self.llc_domains):
                raise ValueError("LLC domain ids must be unique")
            self._validate_partition(self.llc_domains, self.num_threads, "LLC domains")
        tasks = tuple(task for lane in self.lanes for task in lane.tasks)
        if not tasks:
            raise ValueError("executable plan state must contain at least one task")
        expert_ids = [task.expert_id for task in tasks]
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("fixed-lane search state requires one task per active expert")

    @staticmethod
    def _validate_partition(
        intervals: Sequence[ExecutableLane | ExecutableLlcDomain],
        num_threads: int,
        name: str,
    ) -> None:
        if not intervals:
            raise ValueError(f"{name} must form a non-empty partition")
        cursor = 0
        for interval in intervals:
            if interval.core_begin != cursor:
                raise ValueError(f"{name} must be ordered and form a gap-free partition")
            cursor = interval.core_end
        if cursor != num_threads:
            raise ValueError(f"{name} must cover exactly num_threads={num_threads}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(lane.threads for lane in self.lanes)

    @property
    def task_count(self) -> int:
        return sum(len(lane.tasks) for lane in self.lanes)

    def lane_domain_ids(self, lane_index: int) -> tuple[str, ...]:
        lane = self.lanes[int(lane_index)]
        return tuple(
            domain.domain_id
            for domain in self.llc_domains
            if lane.core_begin < domain.core_end and domain.core_begin < lane.core_end
        )

    def to_planner_tasks(self) -> list[PlannerTask]:
        tasks: list[PlannerTask] = []
        for lane in self.lanes:
            previous: int | None = None
            for task in lane.tasks:
                dependencies = [] if previous is None else [previous]
                tasks.append(
                    (
                        task.expert_id,
                        task.routes,
                        lane.core_begin,
                        lane.threads,
                        dependencies,
                    )
                )
                previous = len(tasks) - 1
        return tasks

    def to_bridge(self) -> dict[str, object]:
        tasks = self.to_planner_tasks()
        dependency_offsets = [0]
        dependencies: list[int] = []
        expert_ids = []
        core_begins = []
        widths = []
        w13_windows = []
        w2_windows = []
        for lane in self.lanes:
            for task in lane.tasks:
                task_id = len(expert_ids)
                expert_ids.append(task.expert_id)
                core_begins.append(lane.core_begin)
                widths.append(lane.threads)
                w13_windows.append(task.w13_window_tiles)
                w2_windows.append(task.w2_window_tiles)
                if tasks[task_id][4]:
                    dependencies.extend(tasks[task_id][4])
                dependency_offsets.append(len(dependencies))
        num_tasks = len(expert_ids)
        return {
            "plan_version": _PLAN_VERSION,
            "execution_mode": _EXECUTION_STRICT,
            "num_threads": self.num_threads,
            "thread_cpu_ids": list(self.thread_cpu_ids),
            "task_expert_ids": expert_ids,
            "task_core_begins": core_begins,
            "task_threads": widths,
            "task_dep_offsets": dependency_offsets,
            "task_deps": dependencies,
            "task_preferred_threads": list(widths),
            "task_min_threads": list(widths),
            "task_max_threads": list(widths),
            "task_allowed_thread_offsets": list(range(num_tasks + 1)),
            "task_allowed_threads": list(widths),
            "task_placement_modes": [_PLACEMENT_FIXED] * num_tasks,
            "task_numa_nodes": [_NUMA_UNSPECIFIED] * num_tasks,
            "task_stage_ids": [_STAGE_EXPERT] * num_tasks,
            "task_resize_points": [_RESIZE_NONE] * num_tasks,
            "task_range_granularities": [_FULL_EXPERT_RANGE] * num_tasks,
            "task_w13_window_tiles": w13_windows,
            "task_w2_window_tiles": w2_windows,
            "early_merge": self.early_merge,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "state_schema_version": EXECUTABLE_PLAN_STATE_VERSION,
            "num_threads": self.num_threads,
            "thread_cpu_ids": list(self.thread_cpu_ids),
            "llc_domains": [
                {
                    "id": domain.domain_id,
                    "core_begin": domain.core_begin,
                    "core_count": domain.core_count,
                }
                for domain in self.llc_domains
            ],
            "lanes": [
                {
                    "core_begin": lane.core_begin,
                    "threads": lane.threads,
                    "tasks": [
                        {
                            "expert_id": task.expert_id,
                            "routes": task.routes,
                            "w13_window_tiles": task.w13_window_tiles,
                            "w2_window_tiles": task.w2_window_tiles,
                        }
                        for task in lane.tasks
                    ],
                }
                for lane in self.lanes
            ],
            "early_merge": self.early_merge,
        }

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_planner_result(
        cls,
        result: Mapping[str, object],
        *,
        llc_domains: Sequence[tuple[str, Sequence[int]]] = (),
    ) -> "ExecutablePlanState":
        if result.get("execution_mode") != _EXECUTION_STRICT:
            raise ValueError("executable search state accepts only strict planner results")
        if result.get("stage") is not None:
            raise ValueError("executable search state accepts only whole-expert planner results")
        if result.get("tail_pool_threads") is not None:
            raise ValueError("executable search state does not accept tail-pool plans")
        if int(result.get("tail_pool_tasks", 0)) != 0:
            raise ValueError("executable search state does not accept tail-pool tasks")
        if result.get("tail_repartition_width") is not None or int(
            result.get("tail_repartition_tasks", 0)
        ) != 0:
            raise ValueError("executable search state does not accept tail repartition")
        bridge = result.get("bridge")
        raw_tasks = result.get("tasks")
        if not isinstance(bridge, Mapping):
            raise TypeError("planner result bridge must be a mapping")
        if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
            raise TypeError("planner result tasks must be a sequence")
        shape = _int_tuple(result.get("shape"), "shape")
        if not shape or min(shape) <= 0:
            raise ValueError("planner result shape must contain positive lane widths")
        num_threads = int(bridge.get("num_threads", -1))
        if sum(shape) != num_threads:
            raise ValueError("planner result shape must cover bridge num_threads")
        thread_cpu_ids = _int_tuple(bridge.get("thread_cpu_ids"), "thread_cpu_ids")
        parsed_tasks: list[PlannerTask] = []
        for task_id, raw_task in enumerate(raw_tasks):
            if isinstance(raw_task, (str, bytes)) or not isinstance(raw_task, Sequence):
                raise TypeError(f"planner task {task_id} must be a five-field sequence")
            if len(raw_task) != 5:
                raise ValueError(f"planner task {task_id} must contain five fields")
            expert, routes, core_begin, threads, raw_dependencies = raw_task
            parsed_tasks.append(
                (
                    int(expert),
                    int(routes),
                    int(core_begin),
                    int(threads),
                    list(_int_tuple(raw_dependencies, f"task {task_id} dependencies")),
                )
            )
        cls._validate_bridge(bridge, parsed_tasks)
        domains = cls._domains_from_physical_cpu_ids(thread_cpu_ids, llc_domains)
        w13_windows = _int_tuple(bridge.get("task_w13_window_tiles"), "task_w13_window_tiles")
        w2_windows = _int_tuple(bridge.get("task_w2_window_tiles"), "task_w2_window_tiles")
        lane_specs = []
        cursor = 0
        for width in shape:
            lane_specs.append((cursor, width))
            cursor += width
        task_indices_by_lane: dict[tuple[int, int], list[int]] = {
            lane: [] for lane in lane_specs
        }
        for task_id, (_, _, core_begin, threads, _) in enumerate(parsed_tasks):
            lane = (core_begin, threads)
            if lane not in task_indices_by_lane:
                raise ValueError(f"planner task {task_id} does not belong to a shape lane")
            task_indices_by_lane[lane].append(task_id)
        serialized_indices = [
            task_id
            for lane in lane_specs
            for task_id in task_indices_by_lane[lane]
        ]
        if serialized_indices != list(range(len(parsed_tasks))):
            raise ValueError("planner tasks must be serialized lane by lane")
        lanes = []
        for core_begin, width in lane_specs:
            lane_tasks = []
            previous: int | None = None
            for task_id in task_indices_by_lane[(core_begin, width)]:
                expert, routes, _, _, dependencies = parsed_tasks[task_id]
                expected_dependencies = [] if previous is None else [previous]
                if dependencies != expected_dependencies:
                    raise ValueError(
                        f"task {task_id} must depend exactly on its previous lane task"
                    )
                lane_tasks.append(
                    ExecutableExpertTask(
                        expert,
                        routes,
                        w13_windows[task_id],
                        w2_windows[task_id],
                    )
                )
                previous = task_id
            lanes.append(ExecutableLane(core_begin, width, tuple(lane_tasks)))
        return cls(
            num_threads=num_threads,
            thread_cpu_ids=thread_cpu_ids,
            lanes=tuple(lanes),
            llc_domains=domains,
            early_merge=bridge.get("early_merge"),
        )

    @staticmethod
    def _domains_from_physical_cpu_ids(
        thread_cpu_ids: tuple[int, ...],
        raw_domains: Sequence[tuple[str, Sequence[int]]],
    ) -> tuple[ExecutableLlcDomain, ...]:
        if not raw_domains:
            return ()
        logical_by_cpu = {cpu: logical for logical, cpu in enumerate(thread_cpu_ids)}
        domains = []
        seen_cpus: set[int] = set()
        for domain_id, raw_cpu_ids in raw_domains:
            cpu_ids = _int_tuple(raw_cpu_ids, f"LLC domain {domain_id} CPU ids")
            if not cpu_ids:
                raise ValueError(f"LLC domain {domain_id} must contain at least one CPU")
            if len(set(cpu_ids)) != len(cpu_ids):
                raise ValueError(f"LLC domain {domain_id} CPU ids must be unique")
            unknown = sorted(set(cpu_ids) - logical_by_cpu.keys())
            if unknown:
                raise ValueError(f"LLC domain {domain_id} contains CPUs outside the plan: {unknown}")
            if seen_cpus.intersection(cpu_ids):
                raise ValueError("LLC domain CPU sets must be disjoint")
            seen_cpus.update(cpu_ids)
            logical_ids = sorted(logical_by_cpu[cpu] for cpu in cpu_ids)
            expected = list(range(logical_ids[0], logical_ids[-1] + 1))
            if logical_ids != expected:
                raise ValueError(f"LLC domain {domain_id} is not contiguous in logical CPU order")
            domains.append(
                ExecutableLlcDomain(str(domain_id), logical_ids[0], len(logical_ids))
            )
        if seen_cpus != set(thread_cpu_ids):
            raise ValueError("LLC domains must partition thread_cpu_ids")
        return tuple(sorted(domains, key=lambda domain: domain.core_begin))

    @staticmethod
    def _validate_bridge(
        bridge: Mapping[str, object],
        tasks: Sequence[PlannerTask],
    ) -> None:
        if int(bridge.get("plan_version", -1)) != _PLAN_VERSION:
            raise ValueError(f"bridge plan_version must be {_PLAN_VERSION}")
        if bridge.get("execution_mode") != _EXECUTION_STRICT:
            raise ValueError("bridge execution_mode must be strict")
        expected_experts = tuple(task[0] for task in tasks)
        expected_begins = tuple(task[2] for task in tasks)
        expected_widths = tuple(task[3] for task in tasks)
        dependency_offsets = [0]
        dependencies = []
        for task in tasks:
            dependencies.extend(task[4])
            dependency_offsets.append(len(dependencies))
        exact_fields = {
            "task_expert_ids": expected_experts,
            "task_core_begins": expected_begins,
            "task_threads": expected_widths,
            "task_dep_offsets": tuple(dependency_offsets),
            "task_deps": tuple(dependencies),
            "task_preferred_threads": expected_widths,
            "task_min_threads": expected_widths,
            "task_max_threads": expected_widths,
            "task_allowed_thread_offsets": tuple(range(len(tasks) + 1)),
            "task_allowed_threads": expected_widths,
            "task_placement_modes": (_PLACEMENT_FIXED,) * len(tasks),
            "task_numa_nodes": (_NUMA_UNSPECIFIED,) * len(tasks),
            "task_stage_ids": (_STAGE_EXPERT,) * len(tasks),
            "task_resize_points": (_RESIZE_NONE,) * len(tasks),
            "task_range_granularities": (_FULL_EXPERT_RANGE,) * len(tasks),
        }
        for name, expected in exact_fields.items():
            actual = _int_tuple(bridge.get(name), name)
            if actual != expected:
                raise ValueError(f"bridge {name} is not canonical for the planner tasks")
        for name in ("task_w13_window_tiles", "task_w2_window_tiles"):
            values = _int_tuple(bridge.get(name), name)
            if len(values) != len(tasks) or min(values, default=0) < 0:
                raise ValueError(f"bridge {name} must contain one non-negative value per task")


__all__ = [
    "EXECUTABLE_PLAN_STATE_VERSION",
    "ExecutableExpertTask",
    "ExecutableLane",
    "ExecutableLlcDomain",
    "ExecutablePlanState",
    "PlannerTask",
]
