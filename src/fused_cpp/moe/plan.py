"""Versioned execution plans for the async CPU MoE runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


ASYNC_MOE_PLAN_VERSION = 2
ASYNC_MOE_EXECUTION_STRICT = "strict"
ASYNC_MOE_EXECUTION_TAIL_POOL = "tail_pool"
ASYNC_MOE_PLACEMENT_FIXED = 0
ASYNC_MOE_PLACEMENT_TAIL_POOL = 1
ASYNC_MOE_STAGE_EXPERT = 0
ASYNC_MOE_RESIZE_NONE = 0
ASYNC_MOE_FULL_EXPERT_RANGE = 0

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_V2_SEQUENCE_FIELDS = (
    "thread_cpu_ids",
    "task_expert_ids",
    "task_core_begins",
    "task_threads",
    "task_dep_offsets",
    "task_deps",
    "task_preferred_threads",
    "task_min_threads",
    "task_max_threads",
    "task_allowed_thread_offsets",
    "task_allowed_threads",
    "task_placement_modes",
    "task_numa_nodes",
    "task_stage_ids",
    "task_resize_points",
    "task_range_granularities",
)

_V2_OPTIONAL_PER_TASK_FIELDS = (
    "task_w13_window_bytes",
    "task_w2_window_bytes",
)


def _integer_tensor(value: object, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if value.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"{name} must use an integer dtype, got {value.dtype}")
        if value.dim() != 1:
            raise ValueError(f"{name} must be 1-D")
        return value.to(dtype=torch.int64).contiguous()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a 1-D integer sequence or tensor")
    try:
        return torch.tensor([int(item) for item in value], dtype=torch.int64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain integers") from error


def _values(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.tolist())


@dataclass(frozen=True)
class AsyncMoEPlanV2:
    """Materialized Plan V2 accepted by the native async runtime.

    The runtime executes ``task_threads`` exactly after a task starts. In
    ``tail_pool`` mode, threads may be regrouped only between whole-expert
    tasks; in-task width changes and stage/range resizing remain disabled.
    """

    num_threads: int
    execution_mode: str
    thread_cpu_ids: torch.Tensor
    task_expert_ids: torch.Tensor
    task_core_begins: torch.Tensor
    task_threads: torch.Tensor
    task_dep_offsets: torch.Tensor
    task_deps: torch.Tensor
    task_preferred_threads: torch.Tensor
    task_min_threads: torch.Tensor
    task_max_threads: torch.Tensor
    task_allowed_thread_offsets: torch.Tensor
    task_allowed_threads: torch.Tensor
    task_placement_modes: torch.Tensor
    task_numa_nodes: torch.Tensor
    task_stage_ids: torch.Tensor
    task_resize_points: torch.Tensor
    task_range_granularities: torch.Tensor
    task_w13_window_bytes: torch.Tensor | None = None
    task_w2_window_bytes: torch.Tensor | None = None

    @property
    def plan_version(self) -> int:
        return ASYNC_MOE_PLAN_VERSION

    @property
    def native_execution_mode(self) -> int:
        return {
            ASYNC_MOE_EXECUTION_STRICT: 0,
            ASYNC_MOE_EXECUTION_TAIL_POOL: 1,
        }[self.execution_mode]

    def __post_init__(self) -> None:
        num_tasks = int(self.task_expert_ids.numel())
        for name in _V2_OPTIONAL_PER_TASK_FIELDS:
            if getattr(self, name) is None:
                object.__setattr__(self, name, torch.full((num_tasks,), -1, dtype=torch.int64))
        self.validate()

    @classmethod
    def from_dict(cls, plan: Mapping[str, object]) -> "AsyncMoEPlanV2":
        """Validate and materialize a JSON-compatible Plan V2 bridge."""
        version = int(plan.get("plan_version", -1))
        if version != ASYNC_MOE_PLAN_VERSION:
            raise ValueError(f"plan_version must be {ASYNC_MOE_PLAN_VERSION}, got {version}")
        required = ("num_threads", "execution_mode", *_V2_SEQUENCE_FIELDS)
        missing = [name for name in required if name not in plan]
        if missing:
            raise ValueError(f"Plan V2 is missing required fields: {', '.join(missing)}")
        tensors = {name: _integer_tensor(plan[name], name) for name in _V2_SEQUENCE_FIELDS}
        num_tasks = int(tensors["task_expert_ids"].numel())
        tensors.update(
            {
                name: _integer_tensor(plan.get(name, [-1] * num_tasks), name)
                for name in _V2_OPTIONAL_PER_TASK_FIELDS
            }
        )
        return cls(
            num_threads=int(plan["num_threads"]),
            execution_mode=str(plan["execution_mode"]),
            **tensors,
        )

    def validate(self) -> None:
        if self.execution_mode not in {
            ASYNC_MOE_EXECUTION_STRICT,
            ASYNC_MOE_EXECUTION_TAIL_POOL,
        }:
            raise ValueError(f"execution_mode must be 'strict' or 'tail_pool', got {self.execution_mode!r}")
        if self.num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {self.num_threads}")
        for name in (*_V2_SEQUENCE_FIELDS, *_V2_OPTIONAL_PER_TASK_FIELDS):
            tensor = getattr(self, name)
            assert tensor is not None
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor in a materialized plan")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU tensor")
            if tensor.dtype not in _INTEGER_DTYPES:
                raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}")
            if tensor.dim() != 1:
                raise ValueError(f"{name} must be 1-D")

        cpu_ids = _values(self.thread_cpu_ids)
        if len(cpu_ids) != self.num_threads:
            raise ValueError(
                f"thread_cpu_ids must have exactly num_threads entries: got {len(cpu_ids)} vs {self.num_threads}"
            )
        if any(cpu < 0 for cpu in cpu_ids):
            raise ValueError("thread_cpu_ids must be non-negative")
        if len(set(cpu_ids)) != len(cpu_ids):
            raise ValueError("thread_cpu_ids must not contain duplicates")

        experts = _values(self.task_expert_ids)
        num_tasks = len(experts)
        if num_tasks == 0:
            raise ValueError("Plan V2 must contain at least one task")
        if any(expert < 0 for expert in experts):
            raise ValueError("task_expert_ids must be non-negative")
        if len(set(experts)) != num_tasks:
            raise ValueError("Plan V2 currently requires one task per expert")

        per_task = {
            "task_core_begins": self.task_core_begins,
            "task_threads": self.task_threads,
            "task_preferred_threads": self.task_preferred_threads,
            "task_min_threads": self.task_min_threads,
            "task_max_threads": self.task_max_threads,
            "task_placement_modes": self.task_placement_modes,
            "task_numa_nodes": self.task_numa_nodes,
            "task_stage_ids": self.task_stage_ids,
            "task_resize_points": self.task_resize_points,
            "task_range_granularities": self.task_range_granularities,
            "task_w13_window_bytes": self.task_w13_window_bytes,
            "task_w2_window_bytes": self.task_w2_window_bytes,
        }
        for name, tensor in per_task.items():
            if tensor.numel() != num_tasks:
                raise ValueError(f"{name} must have one entry per task: got {tensor.numel()} vs {num_tasks}")

        core_begins = _values(self.task_core_begins)
        selected_widths = _values(self.task_threads)
        preferred_widths = _values(self.task_preferred_threads)
        min_widths = _values(self.task_min_threads)
        max_widths = _values(self.task_max_threads)
        placements = _values(self.task_placement_modes)
        numa_nodes = _values(self.task_numa_nodes)
        stages = _values(self.task_stage_ids)
        resize_points = _values(self.task_resize_points)
        range_granularities = _values(self.task_range_granularities)
        assert self.task_w13_window_bytes is not None
        assert self.task_w2_window_bytes is not None
        w13_window_bytes = _values(self.task_w13_window_bytes)
        w2_window_bytes = _values(self.task_w2_window_bytes)
        if any(value < -1 for value in (*w13_window_bytes, *w2_window_bytes)):
            raise ValueError("per-task stage windows must be -1 (inherit) or non-negative")
        for task, (core_begin, width) in enumerate(zip(core_begins, selected_widths)):
            if width <= 0:
                raise ValueError(f"task_threads[{task}] must be positive")
            placement = placements[task]
            if placement == ASYNC_MOE_PLACEMENT_FIXED:
                if core_begin < 0:
                    raise ValueError(f"fixed task_core_begins[{task}] must be non-negative")
            elif placement == ASYNC_MOE_PLACEMENT_TAIL_POOL:
                if core_begin != -1:
                    raise ValueError(f"tail-pool task_core_begins[{task}] must be -1")
            else:
                raise ValueError(f"task_placement_modes[{task}] has unsupported value {placement}")
            if placement == ASYNC_MOE_PLACEMENT_FIXED and core_begin + width > self.num_threads:
                raise ValueError(
                    f"task {task} interval [{core_begin}, {core_begin + width}) exceeds num_threads={self.num_threads}"
                )
        if self.execution_mode == ASYNC_MOE_EXECUTION_STRICT and any(
            placement != ASYNC_MOE_PLACEMENT_FIXED for placement in placements
        ):
            raise ValueError("strict Plan V2 requires every task placement to be fixed")
        if any(node != -1 for node in numa_nodes):
            raise ValueError("Plan V2 currently only supports task_numa_nodes=-1")
        if any(stage != ASYNC_MOE_STAGE_EXPERT for stage in stages):
            raise ValueError("Plan V2 currently only supports whole-expert tasks")
        if any(point != ASYNC_MOE_RESIZE_NONE for point in resize_points):
            raise ValueError("Plan V2 currently does not support resize points")
        if any(granularity != ASYNC_MOE_FULL_EXPERT_RANGE for granularity in range_granularities):
            raise ValueError("Plan V2 currently only supports full-expert task ranges")

        dep_offsets = _values(self.task_dep_offsets)
        dependencies = _values(self.task_deps)
        if len(dep_offsets) != num_tasks + 1:
            raise ValueError("task_dep_offsets must have num_tasks + 1 entries")
        if not dep_offsets or dep_offsets[0] != 0:
            raise ValueError("task_dep_offsets[0] must be 0")
        if dep_offsets[-1] != len(dependencies):
            raise ValueError("last task_dep_offsets entry must equal task_deps length")
        for task in range(num_tasks):
            begin, end = dep_offsets[task], dep_offsets[task + 1]
            if begin > end or begin < 0 or end > len(dependencies):
                raise ValueError(f"task {task} dependency range is invalid")
            if any(dep < 0 or dep >= task for dep in dependencies[begin:end]):
                raise ValueError(f"task {task} dependencies must refer to earlier task ids")
            if placements[task] == ASYNC_MOE_PLACEMENT_TAIL_POOL and begin != end:
                raise ValueError(f"tail-pool task {task} must not have dependencies")
            if placements[task] == ASYNC_MOE_PLACEMENT_FIXED and any(
                placements[dep] == ASYNC_MOE_PLACEMENT_TAIL_POOL for dep in dependencies[begin:end]
            ):
                raise ValueError(f"fixed task {task} must not depend on a tail-pool task")

        allowed_offsets = _values(self.task_allowed_thread_offsets)
        allowed_widths = _values(self.task_allowed_threads)
        if len(allowed_offsets) != num_tasks + 1:
            raise ValueError("task_allowed_thread_offsets must have num_tasks + 1 entries")
        if not allowed_offsets or allowed_offsets[0] != 0:
            raise ValueError("task_allowed_thread_offsets[0] must be 0")
        if allowed_offsets[-1] != len(allowed_widths):
            raise ValueError("last task_allowed_thread_offsets entry must equal task_allowed_threads length")
        for task in range(num_tasks):
            begin, end = allowed_offsets[task], allowed_offsets[task + 1]
            if begin > end or begin < 0 or end > len(allowed_widths):
                raise ValueError(f"task {task} allowed-width range is invalid")
            task_allowed = allowed_widths[begin:end]
            if not task_allowed:
                raise ValueError(f"task {task} must allow at least one thread width")
            if any(width <= 0 or width > self.num_threads for width in task_allowed):
                raise ValueError(f"task {task} allowed widths are out of range")
            if any(left >= right for left, right in zip(task_allowed, task_allowed[1:])):
                raise ValueError(f"task {task} allowed widths must be strictly increasing")
            if selected_widths[task] not in task_allowed:
                raise ValueError(f"task_threads[{task}] is not present in its allowed widths")
            if preferred_widths[task] not in task_allowed:
                raise ValueError(f"task_preferred_threads[{task}] is not in its allowed widths")
            if min_widths[task] != task_allowed[0]:
                raise ValueError(f"task_min_threads[{task}] does not match its allowed widths")
            if max_widths[task] != task_allowed[-1]:
                raise ValueError(f"task_max_threads[{task}] does not match its allowed widths")
            if (
                placements[task] == ASYNC_MOE_PLACEMENT_FIXED
                and core_begins[task] + task_allowed[-1] > self.num_threads
            ):
                raise ValueError(f"task {task} allowed widths exceed its fixed logical-core placement")

        if self.execution_mode == ASYNC_MOE_EXECUTION_TAIL_POOL:
            pooled_tasks = [
                task for task, placement in enumerate(placements) if placement == ASYNC_MOE_PLACEMENT_TAIL_POOL
            ]
            if not pooled_tasks:
                raise ValueError("tail_pool execution requires at least one pooled task")
            pool_widths = {selected_widths[task] for task in pooled_tasks}
            if len(pool_widths) != 1:
                raise ValueError("all tail-pool tasks must use the same selected width")
            pool_width = next(iter(pool_widths))
            if pool_width > self.num_threads or self.num_threads % pool_width != 0:
                raise ValueError(
                    "tail-pool task width must divide num_threads: "
                    f"pool_width={pool_width}, num_threads={self.num_threads}"
                )
            for task, placement in enumerate(placements):
                if placement != ASYNC_MOE_PLACEMENT_FIXED:
                    continue
                if core_begins[task] % pool_width != 0 or selected_widths[task] % pool_width != 0:
                    raise ValueError(
                        "fixed task intervals must align to the tail-pool width: "
                        f"task={task}, core_begin={core_begins[task]}, "
                        f"threads={selected_widths[task]}, pool_width={pool_width}"
                    )

    def legacy_schedule(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the fixed-width arrays consumed by the native async executor."""
        if self.execution_mode != ASYNC_MOE_EXECUTION_STRICT:
            raise RuntimeError("only strict Plan V2 can be lowered to the legacy executor")
        return (
            self.task_expert_ids,
            self.task_core_begins,
            self.task_threads,
            self.task_dep_offsets,
            self.task_deps,
        )


def upgrade_legacy_async_plan(plan: Mapping[str, object]) -> dict[str, object]:
    """Upgrade a legacy fixed-width async bridge to strict Plan V2."""
    task_threads = [int(width) for width in plan["task_threads"]]  # type: ignore[index]
    num_tasks = len(task_threads)
    return {
        **plan,
        "plan_version": ASYNC_MOE_PLAN_VERSION,
        "execution_mode": ASYNC_MOE_EXECUTION_STRICT,
        "task_preferred_threads": list(task_threads),
        "task_min_threads": list(task_threads),
        "task_max_threads": list(task_threads),
        "task_allowed_thread_offsets": list(range(num_tasks + 1)),
        "task_allowed_threads": list(task_threads),
        "task_placement_modes": [ASYNC_MOE_PLACEMENT_FIXED] * num_tasks,
        "task_numa_nodes": [-1] * num_tasks,
        "task_stage_ids": [ASYNC_MOE_STAGE_EXPERT] * num_tasks,
        "task_resize_points": [ASYNC_MOE_RESIZE_NONE] * num_tasks,
        "task_range_granularities": [ASYNC_MOE_FULL_EXPERT_RANGE] * num_tasks,
        "task_w13_window_bytes": [-1] * num_tasks,
        "task_w2_window_bytes": [-1] * num_tasks,
    }


__all__ = [
    "ASYNC_MOE_EXECUTION_STRICT",
    "ASYNC_MOE_EXECUTION_TAIL_POOL",
    "ASYNC_MOE_FULL_EXPERT_RANGE",
    "ASYNC_MOE_PLACEMENT_FIXED",
    "ASYNC_MOE_PLACEMENT_TAIL_POOL",
    "ASYNC_MOE_PLAN_VERSION",
    "ASYNC_MOE_RESIZE_NONE",
    "ASYNC_MOE_STAGE_EXPERT",
    "AsyncMoEPlanV2",
    "upgrade_legacy_async_plan",
]
