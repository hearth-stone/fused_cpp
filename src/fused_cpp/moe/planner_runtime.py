"""Explicit process runtime for analytical Plan V2 MoE dispatch."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from cpu_moe_schedule_optimization.cost_model.analytic_model import (
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
)
from cpu_moe_schedule_optimization.planners.planned_moe import PlannedMoE, route_counts
from fused_cpp.moe.plan import AsyncMoEPlanV2


class MoePlannerRuntime:
    """Thread-safe cached planner for one machine rank and expert shape.

    The runtime is thread-safe. Planning and cache mutation are serialized;
    native Plan V2 execution is not covered by this lock.
    """

    def __init__(
        self,
        calibration: AnalyticMachineCalibration | str | Path,
        *,
        hidden_size: int,
        intermediate_size: int,
        global_experts: int,
        local_experts: int,
        mode: str = "standalone",
        degree: int = 1,
        concurrent_ranks: int = 1,
        cpu_ids: Sequence[int] | None = None,
    ) -> None:
        self.model = AnalyticMoeCostModel(
            calibration,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            global_experts=global_experts,
            local_experts=local_experts,
            mode=mode,
            degree=degree,
            concurrent_ranks=concurrent_ranks,
        )
        machine = self.model.calibration
        if cpu_ids is None:
            cpu_ids = machine.rank_cpu_ids or self._current_affinity(machine.cores_per_rank)
        resolved_cpu_ids = tuple(int(cpu) for cpu in cpu_ids)
        if len(resolved_cpu_ids) != machine.cores_per_rank or len(set(resolved_cpu_ids)) != len(resolved_cpu_ids):
            raise ValueError(
                "cpu_ids must contain exactly one unique CPU for every calibrated rank core"
            )
        if machine.rank_cpu_ids and resolved_cpu_ids != machine.rank_cpu_ids:
            raise ValueError("cpu_ids must match the ordered CPU ids recorded by the calibration")

        self.cpu_ids = resolved_cpu_ids
        self.num_cores = len(resolved_cpu_ids)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.global_experts = int(global_experts)
        self.local_experts = int(local_experts)
        self.mode = str(mode)
        self._planner = PlannedMoE(
            self.model,
            num_cores=self.num_cores,
            cpu_ids=self.cpu_ids,
            search_mode="quick",
        )
        self._lock = threading.RLock()

    @staticmethod
    def _current_affinity(core_count: int) -> tuple[int, ...]:
        if not hasattr(os, "sched_getaffinity"):
            return tuple(range(core_count))
        affinity = tuple(sorted(os.sched_getaffinity(0)))
        if len(affinity) != core_count:
            raise ValueError(
                f"current affinity contains {len(affinity)} CPUs but calibration requires {core_count}; pass cpu_ids"
            )
        return affinity

    def _is_compatible(
        self,
        weights: Any,
        *,
        num_threads: int,
        activation: Any,
        global_num_experts: int,
    ) -> bool:
        activation_name = getattr(activation, "value", activation)
        effective_global_experts = self.global_experts if global_num_experts < 0 else int(global_num_experts)
        return (
            self.local_experts == self.global_experts
            and self.mode in {"standalone", "tp"}
            and int(num_threads) == self.num_cores
            and str(activation_name) == "silu"
            and bool(weights.fused_silu)
            and int(weights.gemm_backend) == 1
            and int(weights.backend_n_tile) == self.model.policy.backend_n_tile
            and int(weights.w13[0].shape[0]) == self.local_experts
            and int(weights.w13[1]) == self.hidden_size
            and int(weights.w13[2]) == 2 * self.intermediate_size
            and int(weights.w2[1]) == self.intermediate_size
            and int(weights.w2[2]) == self.hidden_size
            and effective_global_experts == self.global_experts
        )

    def plan_for_dispatch(
        self,
        weights: Any,
        topk_ids: torch.Tensor,
        *,
        num_threads: int,
        activation: Any,
        global_num_experts: int,
    ) -> AsyncMoEPlanV2 | None:
        """Return a plan when this runtime owns the call, otherwise ``None``."""
        if not self._is_compatible(
            weights,
            num_threads=num_threads,
            activation=activation,
            global_num_experts=global_num_experts,
        ):
            return None
        counts = route_counts(topk_ids, self.local_experts)
        if not counts:
            return None
        with self._lock:
            spec = self._planner.plan_spec_for(counts, topk_ids=topk_ids)
        return AsyncMoEPlanV2.from_dict(spec["bridge"])

    @property
    def last_plan(self) -> dict[str, object]:
        """Return a snapshot of planner diagnostics for the latest call."""
        with self._lock:
            return dict(self._planner.last)


_default_runtime_lock = threading.RLock()
_default_runtime: MoePlannerRuntime | None = None


def set_default_moe_planner_runtime(runtime: MoePlannerRuntime | None) -> MoePlannerRuntime | None:
    """Install ``runtime`` for normal fused-MoE dispatch and return the previous value."""
    if runtime is not None and not isinstance(runtime, MoePlannerRuntime):
        raise TypeError("runtime must be a MoePlannerRuntime or None")
    global _default_runtime
    with _default_runtime_lock:
        previous = _default_runtime
        _default_runtime = runtime
        return previous


def get_default_moe_planner_runtime() -> MoePlannerRuntime | None:
    """Return the process-wide planner runtime, if explicitly installed."""
    with _default_runtime_lock:
        return _default_runtime


def calibrate_moe_planner_quick(*args: Any, **kwargs: Any) -> Any:
    """Explicitly run the supported quick machine calibration workflow."""
    from cpu_moe_schedule_optimization.cost_model.quick_calibration import calibrate_moe_planner_quick as calibrate

    return calibrate(*args, **kwargs)


def enable_moe_planner_quick(
    *,
    hidden_size: int,
    intermediate_size: int,
    global_experts: int,
    local_experts: int,
    cpu_ids: Sequence[int] | None = None,
    mode: str = "standalone",
    degree: int = 1,
    concurrent_ranks: int = 1,
    output: str | Path | None = None,
    supported_widths: Sequence[int] | None = None,
    machine_id: str | None = None,
    overwrite: bool = False,
    seed: int = 20260814,
    report: Callable[[str], None] | None = None,
) -> MoePlannerRuntime:
    """Calibrate, construct, and install the default MoE planner runtime."""
    result = calibrate_moe_planner_quick(
        cpu_ids,
        output=output,
        supported_widths=supported_widths,
        machine_id=machine_id,
        overwrite=overwrite,
        seed=seed,
        report=report,
    )
    runtime = MoePlannerRuntime(
        result.calibration,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        global_experts=global_experts,
        local_experts=local_experts,
        mode=mode,
        degree=degree,
        concurrent_ranks=concurrent_ranks,
        cpu_ids=result.cpu_ids,
    )
    set_default_moe_planner_runtime(runtime)
    return runtime


__all__ = [
    "MoePlannerRuntime",
    "calibrate_moe_planner_quick",
    "enable_moe_planner_quick",
    "get_default_moe_planner_runtime",
    "set_default_moe_planner_runtime",
]
