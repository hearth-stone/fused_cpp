"""Hardware-grounded analytical cost model for fused CPU MoE.

The model deliberately separates route-dependent work from machine
calibration:

* :mod:`gemm_cost_model` describes logical W13/W2 work;
* :mod:`sve_bf16_kernel_model` lowers it to physical SVE kernel demand;
* this module maps that demand through a small set of cache capacities,
  saturating service curves, and fixed runtime costs.

Unlike ``ContentionCostModel``, no route/thread latency table or measured
contention shape is required. Concurrent tasks are simulated as W13/W2
setup/cold-B/steady-B phases which consume calibrated aggregate matrix,
frontend, cache, DRAM, and epilogue service ceilings.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from gemm_cost_model import ExecutionSchedule, fused_expert_work
    from sve_bf16_kernel_model import SveBf16KernelExecution, SveBf16KernelProfile
    from weight_window import (
        WeightWindowGeometry,
        fused_moe_task_weight_windows,
        fused_moe_weight_windows,
        stage_weight_window_geometry,
    )
except ImportError:  # pragma: no cover - package-style import
    from .gemm_cost_model import ExecutionSchedule, fused_expert_work
    from .sve_bf16_kernel_model import SveBf16KernelExecution, SveBf16KernelProfile
    from .weight_window import (
        WeightWindowGeometry,
        fused_moe_task_weight_windows,
        fused_moe_weight_windows,
        stage_weight_window_geometry,
    )


ANALYTIC_MACHINE_SCHEMA_VERSION = 1
ANALYTIC_MODEL_SCHEMA_VERSION = 6
ANALYTIC_MODEL_NAME = "phase_ecm_shared_resource_v3"
_SHARED_RESOURCES = (
    "gemm_core_flops",
    "matrix_flops",
    "frontend_instructions",
    "l1_bytes",
    "l2_bytes",
    "llc_bytes",
    "dram_bytes",
    "epilogue_elements",
)
_RESOURCE_PATHS = {
    "gemm_core_flops": "m12_l1_hot_gemm_core",
    "matrix_flops": "bfmmla_execution",
    "frontend_instructions": "frontend_and_issue",
    "l1_bytes": "core_load_delivery",
    "l2_bytes": "private_l2_transfer",
    "llc_bytes": "shared_llc_to_private_l2_refill",
    "dram_bytes": "dram_to_llc_compulsory_and_spill",
    "epilogue_elements": "fused_epilogue_execution",
}


@dataclass(frozen=True)
class SaturatingServiceCurve:
    """Monotone service curve fixed by one-core and saturation measurements.

    The selected curve family interpolates between one-core service and the
    sustainable aggregate knee without storing a value for every active width.
    """

    single_thread_rate: float
    saturated_rate: float
    saturation_threads: int
    curve: str = "power"

    def __post_init__(self) -> None:
        if min(self.single_thread_rate, self.saturated_rate) <= 0.0:
            raise ValueError("service rates must be positive")
        if self.saturation_threads <= 0:
            raise ValueError("saturation_threads must be positive")
        if self.curve not in {"power", "shared_bottleneck"}:
            raise ValueError("curve must be 'power' or 'shared_bottleneck'")
        if self.saturated_rate < self.single_thread_rate:
            raise ValueError("saturated_rate cannot be below single_thread_rate")
        if (
            self.curve == "shared_bottleneck"
            and self.saturated_rate > self.single_thread_rate * self.saturation_threads
        ):
            raise ValueError("shared_bottleneck saturated_rate cannot exceed linear scaling")
        if self.saturation_threads == 1 and not math.isclose(
            self.single_thread_rate,
            self.saturated_rate,
            rel_tol=1e-12,
        ):
            raise ValueError("a one-thread saturation point must equal the single-thread rate")

    @property
    def exponent(self) -> float:
        if self.saturation_threads == 1 or self.saturated_rate == self.single_thread_rate:
            return 0.0
        return math.log(self.saturated_rate / self.single_thread_rate) / math.log(self.saturation_threads)

    def rate(self, active_threads: int) -> float:
        if active_threads <= 0:
            raise ValueError("active_threads must be positive")
        if active_threads >= self.saturation_threads:
            return self.saturated_rate
        if self.curve == "shared_bottleneck":
            linear_ratio = self.single_thread_rate * self.saturation_threads / self.saturated_rate
            beta = (linear_ratio - 1.0) / max(self.saturation_threads - 1, 1)
            return self.single_thread_rate * active_threads / (1.0 + beta * (active_threads - 1))
        return min(
            self.saturated_rate,
            self.single_thread_rate * active_threads**self.exponent,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "SaturatingServiceCurve":
        return cls(
            single_thread_rate=float(payload["single_thread_rate"]),
            saturated_rate=float(payload["saturated_rate"]),
            saturation_threads=int(payload["saturation_threads"]),
            curve=str(payload.get("curve", "power")),
        )


@dataclass(frozen=True)
class CacheCalibration:
    l1d_bytes_per_core: int
    l2_bytes_per_core: int
    llc_bytes_per_rank: int
    l2_effective_fraction: float = 0.75
    llc_effective_fraction: float = 0.75
    l2_b_reuse_effective_fraction: float | None = None
    l2_b_reuse_miss_floor: float = 0.0
    l2_b_reuse_miss_at_capacity: float = 1.0
    l2_b_reuse_miss_ceiling: float = 1.0

    def __post_init__(self) -> None:
        if min(self.l1d_bytes_per_core, self.l2_bytes_per_core, self.llc_bytes_per_rank) <= 0:
            raise ValueError("cache capacities must be positive")
        if not 0.0 < self.l2_effective_fraction <= 1.0:
            raise ValueError("l2_effective_fraction must be in (0, 1]")
        if not 0.0 < self.llc_effective_fraction <= 1.0:
            raise ValueError("llc_effective_fraction must be in (0, 1]")
        if self.l2_b_reuse_effective_fraction is not None and not (
            0.0 < self.l2_b_reuse_effective_fraction <= 1.0
        ):
            raise ValueError("l2_b_reuse_effective_fraction must be in (0, 1]")
        if not 0.0 <= self.l2_b_reuse_miss_floor < 1.0:
            raise ValueError("l2_b_reuse_miss_floor must be in [0, 1)")
        if not (self.l2_b_reuse_miss_floor <= self.l2_b_reuse_miss_at_capacity <= self.l2_b_reuse_miss_ceiling <= 1.0):
            raise ValueError("packed-B L2 miss anchors must be monotone and no larger than one")

    @property
    def effective_l2_bytes_per_core(self) -> float:
        return self.l2_bytes_per_core * self.l2_effective_fraction

    @property
    def effective_llc_bytes_per_rank(self) -> float:
        return self.llc_bytes_per_rank * self.llc_effective_fraction

    @property
    def effective_l2_b_reuse_bytes_per_core(self) -> float:
        fraction = (
            self.l2_effective_fraction
            if self.l2_b_reuse_effective_fraction is None
            else self.l2_b_reuse_effective_fraction
        )
        return self.l2_bytes_per_core * fraction

    @classmethod
    def from_dict(cls, payload: dict) -> "CacheCalibration":
        return cls(
            l1d_bytes_per_core=int(payload["l1d_bytes_per_core"]),
            l2_bytes_per_core=int(payload["l2_bytes_per_core"]),
            llc_bytes_per_rank=int(payload["llc_bytes_per_rank"]),
            l2_effective_fraction=float(payload.get("l2_effective_fraction", 0.75)),
            llc_effective_fraction=float(payload.get("llc_effective_fraction", 0.75)),
            l2_b_reuse_effective_fraction=(
                float(payload["l2_b_reuse_effective_fraction"])
                if payload.get("l2_b_reuse_effective_fraction") is not None
                else None
            ),
            l2_b_reuse_miss_floor=float(payload.get("l2_b_reuse_miss_floor", 0.0)),
            l2_b_reuse_miss_at_capacity=float(payload.get("l2_b_reuse_miss_at_capacity", 1.0)),
            l2_b_reuse_miss_ceiling=float(payload.get("l2_b_reuse_miss_ceiling", 1.0)),
        )


@dataclass(frozen=True)
class RuntimeOverheads:
    call_setup_ns: float = 0.0
    expert_fixed_ns: float = 0.0
    route_ns: float = 0.0
    stage_fixed_ns: float = 0.0
    range_fixed_ns: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.call_setup_ns,
                self.expert_fixed_ns,
                self.route_ns,
                self.stage_fixed_ns,
                self.range_fixed_ns,
            )
            < 0.0
        ):
            raise ValueError("runtime overheads must be non-negative")

    @classmethod
    def from_dict(cls, payload: dict) -> "RuntimeOverheads":
        return cls(
            call_setup_ns=float(payload.get("call_setup_ns", 0.0)),
            expert_fixed_ns=float(payload.get("expert_fixed_ns", 0.0)),
            route_ns=float(payload.get("route_ns", 0.0)),
            stage_fixed_ns=float(payload.get("stage_fixed_ns", 0.0)),
            range_fixed_ns=float(payload.get("range_fixed_ns", 0.0)),
        )


@dataclass(frozen=True)
class AnalyticMachineCalibration:
    """Thin machine calibration independent of route and expert shape."""

    machine_id: str
    cores_per_rank: int
    caches: CacheCalibration
    matrix_flops: SaturatingServiceCurve
    gemm_core_flops: SaturatingServiceCurve
    l1_bytes: SaturatingServiceCurve
    l2_bytes: SaturatingServiceCurve
    llc_bytes: SaturatingServiceCurve
    dram_bytes: SaturatingServiceCurve
    backend_n_tile: int = 8
    frontend_instructions: SaturatingServiceCurve | None = None
    epilogue_elements: SaturatingServiceCurve | None = None
    overheads: RuntimeOverheads = RuntimeOverheads()
    supported_widths: tuple[int, ...] = (1,)
    relative_uncertainty: float = 0.05
    w13_scale: float = 1.0
    w2_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.machine_id:
            raise ValueError("machine_id must be non-empty")
        if self.cores_per_rank <= 0:
            raise ValueError("cores_per_rank must be positive")
        if self.gemm_core_flops is None:
            raise ValueError("gemm_core_flops must provide the L1-hot GEMM compute peak")
        if self.backend_n_tile < 8 or self.backend_n_tile % 8:
            raise ValueError("backend_n_tile must be at least eight and a multiple of eight")
        widths = tuple(sorted(set(int(width) for width in self.supported_widths)))
        if not widths or widths[0] <= 0 or widths[-1] > self.cores_per_rank:
            raise ValueError("supported_widths must be positive and no larger than cores_per_rank")
        if not 0.0 <= self.relative_uncertainty < 1.0:
            raise ValueError("relative_uncertainty must be in [0, 1)")
        if min(self.w13_scale, self.w2_scale) <= 0.0:
            raise ValueError("stage scales must be positive")
        object.__setattr__(self, "supported_widths", widths)

    def service_rate(self, resource: str, active_threads: int) -> float:
        curve = getattr(self, resource)
        if curve is None:
            return math.inf
        return curve.rate(min(active_threads, self.cores_per_rank))

    @classmethod
    def from_dict(cls, payload: dict) -> "AnalyticMachineCalibration":
        if int(payload.get("schema_version", 0)) != ANALYTIC_MACHINE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported analytic machine schema {payload.get('schema_version')!r}; "
                f"expected {ANALYTIC_MACHINE_SCHEMA_VERSION}"
            )
        if payload.get("kind") != "moe_analytic_machine":
            raise ValueError("analytic calibration kind must be 'moe_analytic_machine'")
        machine = payload["machine"]
        services = payload["services"]
        stage_scales = payload.get("stage_scales", {})
        optional_frontend = services.get("frontend_instructions")
        optional_epilogue = services.get("epilogue_elements")
        if "gemm_core_flops" not in services:
            raise ValueError("analytic calibration must contain the L1-hot gemm_core_flops service")
        return cls(
            machine_id=str(machine["id"]),
            cores_per_rank=int(machine["cores_per_rank"]),
            caches=CacheCalibration.from_dict(payload["caches"]),
            matrix_flops=SaturatingServiceCurve.from_dict(services["matrix_flops"]),
            gemm_core_flops=SaturatingServiceCurve.from_dict(services["gemm_core_flops"]),
            l1_bytes=SaturatingServiceCurve.from_dict(services["l1_bytes"]),
            l2_bytes=SaturatingServiceCurve.from_dict(services["l2_bytes"]),
            llc_bytes=SaturatingServiceCurve.from_dict(services["llc_bytes"]),
            dram_bytes=SaturatingServiceCurve.from_dict(services["dram_bytes"]),
            backend_n_tile=int(payload.get("kernel", {}).get("backend_n_tile", 8)),
            frontend_instructions=(
                SaturatingServiceCurve.from_dict(optional_frontend) if optional_frontend is not None else None
            ),
            epilogue_elements=(
                SaturatingServiceCurve.from_dict(optional_epilogue) if optional_epilogue is not None else None
            ),
            overheads=RuntimeOverheads.from_dict(payload.get("overheads", {})),
            supported_widths=tuple(int(value) for value in payload["planner"]["supported_widths"]),
            relative_uncertainty=float(payload.get("uncertainty", {}).get("relative", 0.05)),
            w13_scale=float(stage_scales.get("w13", 1.0)),
            w2_scale=float(stage_scales.get("w2", 1.0)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "AnalyticMachineCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        def service(curve: SaturatingServiceCurve | None) -> dict | None:
            return asdict(curve) if curve is not None else None

        services = {
            "matrix_flops": service(self.matrix_flops),
            "gemm_core_flops": service(self.gemm_core_flops),
            "l1_bytes": service(self.l1_bytes),
            "l2_bytes": service(self.l2_bytes),
            "llc_bytes": service(self.llc_bytes),
            "dram_bytes": service(self.dram_bytes),
        }
        if self.frontend_instructions is not None:
            services["frontend_instructions"] = service(self.frontend_instructions)
        if self.epilogue_elements is not None:
            services["epilogue_elements"] = service(self.epilogue_elements)
        return {
            "schema_version": ANALYTIC_MACHINE_SCHEMA_VERSION,
            "kind": "moe_analytic_machine",
            "machine": {
                "id": self.machine_id,
                "cores_per_rank": self.cores_per_rank,
            },
            "kernel": {"backend_n_tile": self.backend_n_tile},
            "caches": asdict(self.caches),
            "services": services,
            "overheads": asdict(self.overheads),
            "planner": {"supported_widths": list(self.supported_widths)},
            "uncertainty": {"relative": self.relative_uncertainty},
            "stage_scales": {"w13": self.w13_scale, "w2": self.w2_scale},
        }


@dataclass(frozen=True)
class AnalyticPolicy:
    """Planner-visible policy identity without measured profile hashes."""

    machine_id: str
    mode: str
    degree: int
    hidden_size: int
    intermediate_size: int
    global_experts: int
    local_experts: int
    backend: str
    backend_n_tile: int
    sve_implementation: str
    m_tail_policy: str
    activation: str
    dtype: str
    w13_split: bool
    w13_split_chunks: int
    weight_window_bytes: int
    w13_window_ranges: int
    w2_window_ranges: int
    measurement_experts: int
    cores_per_rank: int
    concurrent_ranks: int
    llc_bytes_per_rank: int

    def key_without_kernel_policy(self) -> tuple[object, ...]:
        return (
            "analytic",
            self.machine_id,
            self.mode,
            self.degree,
            self.hidden_size,
            self.intermediate_size,
            self.global_experts,
            self.local_experts,
            self.backend,
            self.backend_n_tile,
            self.sve_implementation,
            self.m_tail_policy,
            self.activation,
            self.dtype,
            self.cores_per_rank,
            self.concurrent_ranks,
            self.llc_bytes_per_rank,
        )

    def key_without_split(self) -> tuple[object, ...]:
        return self.key_without_kernel_policy()

    def kernel_policy_key(self) -> tuple[object, ...]:
        return ("stage_ranges", self.w13_window_ranges, self.w2_window_ranges)


def analytic_candidate_shapes(cores: int, widths: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Generate homogeneous and two-width static interval shapes.

    The analytical model is not restricted to measured shapes.  Limiting the
    online set to at most two widths preserves useful long/short mixtures while
    avoiding the combinatorial integer-partition space.
    """

    widths = tuple(sorted({int(width) for width in widths if 0 < int(width) <= cores}))
    if cores <= 0 or not widths:
        raise ValueError("cores and widths must define a non-empty positive shape space")
    shapes: set[tuple[int, ...]] = set()
    for width in widths:
        if cores % width == 0:
            shapes.add((width,) * (cores // width))
    for small_index, small in enumerate(widths):
        for large in widths[small_index + 1 :]:
            for large_count in range(1, cores // large + 1):
                remaining = cores - large_count * large
                if remaining <= 0 or remaining % small:
                    continue
                shape = (large,) * large_count + (small,) * (remaining // small)
                shapes.add(shape)
    return tuple(sorted(shapes, key=lambda shape: (len(shape), shape)))


def _smooth_capacity_miss(working_set: float, effective_capacity: float, physical_capacity: float) -> float:
    if working_set <= effective_capacity:
        return 0.0
    if working_set >= physical_capacity or physical_capacity <= effective_capacity:
        return 1.0
    x = (working_set - effective_capacity) / (physical_capacity - effective_capacity)
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class AnalyticRangeDemand:
    """Range-local demand.

    ``balanced_work_fraction`` scales the mapper's aggregate busiest-lane
    upper bound. It can sum to less than one when a tail range activates fewer
    N owners than the full team.
    """

    n_tiles: int
    balanced_work_fraction: float
    owner_window_bytes: int
    l2_miss_fraction_b: float
    l2_steady_miss_fraction_b: float
    l2_miss_fraction_a: float
    b_transient_reuses: int
    b_steady_reuses: int
    b_reuse_footprint_bytes: int
    a_residency_footprint_bytes: int
    a_residency_capacity_bytes: int
    a_l2_refill_bytes: float
    b_l2_refill_bytes: float
    c_write_bytes: float
    l2_bytes: float
    llc_bytes: float
    compulsory_dram_bytes: float
    spillable_dram_bytes: float
    reusable_b_bytes: int
    llc_working_set_bytes: float


@dataclass(frozen=True)
class AnalyticStageDemand:
    stage: str
    mapping: SveBf16KernelExecution
    ranges: int
    window_bytes: int
    owner_window_bytes: int
    reusable_b_bytes: int
    l2_miss_fraction_b: float
    l2_miss_fraction_a: float
    a_l2_refill_bytes: float
    b_l2_refill_bytes: float
    l2_refill_bytes: float
    c_write_bytes: float
    l2_bytes: float
    llc_bytes: float
    compulsory_dram_bytes: float
    spillable_dram_bytes: float
    llc_working_set_bytes: float
    range_demands: tuple[AnalyticRangeDemand, ...]


@dataclass(frozen=True)
class AnalyticPhase:
    name: str
    kind: str
    panel_count: int
    active_threads: int
    fixed_ns: float
    gemm_core_ns: float
    matrix_ns: float
    frontend_ns: float
    l1_ns: float
    l2_ns: float
    llc_ns: float
    epilogue_ns: float
    matrix_flops: float
    frontend_instructions: float
    l1_bytes: float
    l2_bytes: float
    llc_bytes: float
    epilogue_elements: float
    compulsory_dram_bytes: float
    spillable_dram_bytes: float
    dram_rate: float
    working_set_bytes: float
    isolated_spill_fraction: float
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"operator", "range_setup", "cold_b", "steady_b"}:
            raise ValueError(f"unsupported analytical phase kind {self.kind!r}")
        if self.panel_count < 0:
            raise ValueError("panel_count must be non-negative")
        if self.kind in {"operator", "range_setup"} and self.panel_count != 0:
            raise ValueError("non-GEMM phases cannot contain GEMM panels")
        if self.kind in {"cold_b", "steady_b"} and self.panel_count == 0:
            raise ValueError("GEMM phases must contain at least one panel")

    def dram_bytes(self, spill_fraction: float) -> float:
        return self.compulsory_dram_bytes + spill_fraction * self.spillable_dram_bytes

    def resource_demand(self, resource: str, spill_fraction: float) -> float:
        if resource == "gemm_core_flops":
            return self.matrix_flops
        if resource in {"matrix_flops", "frontend_instructions", "l1_bytes"}:
            return 0.0
        if resource == "dram_bytes":
            return self.dram_bytes(spill_fraction)
        return float(getattr(self, resource))

    def resource_times_ns(self, spill_fraction: float | None = None) -> dict[str, float]:
        if spill_fraction is None:
            spill_fraction = self.isolated_spill_fraction
        return {
            "gemm_core_flops": self.gemm_core_ns,
            "matrix_flops": self.matrix_ns,
            "frontend_instructions": self.frontend_ns,
            "l1_bytes": self.l1_ns,
            "l2_bytes": self.l2_ns,
            "llc_bytes": self.llc_ns,
            "dram_bytes": self.dram_bytes(spill_fraction) / self.dram_rate * 1e9,
            "epilogue_elements": self.epilogue_ns,
        }

    def duration_ns(
        self,
        *,
        spill_fraction: float | None = None,
        resource_scales: Mapping[str, float] | None = None,
    ) -> float:
        if spill_fraction is None:
            spill_fraction = self.isolated_spill_fraction
        scales = resource_scales or {}
        times = self.resource_times_ns(spill_fraction)
        gemm_core_ns = times["gemm_core_flops"] * scales.get("gemm_core_flops", 1.0)
        # The calibrated load probes measure endpoint-to-register service:
        # LLC already includes LLC->L2->L1 and DRAM includes the whole path.
        # Their lower bounds overlap and therefore compose with max, not sum.
        # The L1-hot M12 GEMM peak already includes BFMMLA, frontend, and L1
        # load delivery. Only lower hierarchy endpoint bounds remain.
        transfer_ns = max(
            times["l2_bytes"] * scales.get("l2_bytes", 1.0),
            times["llc_bytes"] * scales.get("llc_bytes", 1.0),
            times["dram_bytes"] * scales.get("dram_bytes", 1.0),
        )
        body_ns = max(gemm_core_ns, transfer_ns)
        epilogue_ns = times["epilogue_elements"] * scales.get("epilogue_elements", 1.0)
        return self.residual_scale * (self.fixed_ns + body_ns + epilogue_ns)

    @property
    def base_ns(self) -> float:
        return self.duration_ns()


@dataclass(frozen=True)
class ExpertPrediction:
    routes: int
    threads: int
    phases: tuple[AnalyticPhase, ...]
    w13_demand: AnalyticStageDemand
    w2_demand: AnalyticStageDemand

    @property
    def total_ns(self) -> float:
        return sum(phase.base_ns for phase in self.phases)

    @property
    def w13_ns(self) -> float:
        return sum(phase.base_ns for phase in self.phases if phase.name.startswith("w13"))

    @property
    def w2_ns(self) -> float:
        return sum(phase.base_ns for phase in self.phases if phase.name.startswith("w2"))


@dataclass(frozen=True)
class AnalyticStageWindowScore:
    """Analytical objective for one tile-aligned stage-window geometry."""

    stage: str
    routes: int
    threads: int
    target_bytes: int
    ranges: int
    range_bytes: int
    worker_bytes: int
    active_threads: int
    objective_ns: float
    serialized_core_ns: float
    transfer_ns: float
    range_overhead_ns: float
    l2_miss_fraction_a: float
    l2_miss_fraction_b: float
    a_l2_refill_bytes: float
    b_l2_refill_bytes: float
    l2_bytes: float
    llc_bytes: float
    compulsory_dram_bytes: float
    spillable_dram_bytes: float


@dataclass(frozen=True)
class AnalyticResourcePressure:
    """Requested and allocated service for one active shared resource."""

    active_threads: int
    offered_rate: float
    capacity: float
    utilization: float
    dilation: float
    allocated_rate: float
    allocated_utilization: float


class AnalyticMoeCostModel:
    """Planner-compatible analytical SVE fused-expert cost model."""

    schema_version = ANALYTIC_MODEL_SCHEMA_VERSION
    iso_mode = "analytic"
    use_stage_model = True
    has_full_workload_anchors = False
    profile_runs = 1

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
        backend_n_tile: int | None = None,
        activation: str = "silu",
        dtype: str = "bf16",
        w13_split: bool = True,
        w13_split_chunks: int = 2,
        weight_window_bytes: int = 0,
        exact_m: bool = True,
        down_output_element_bytes: int = 2,
        supported_widths: Sequence[int] | None = None,
        supported_shapes: Sequence[Sequence[int]] | None = None,
    ):
        if isinstance(calibration, (str, Path)):
            self.profile_path = Path(calibration)
            self.calibration = AnalyticMachineCalibration.from_path(calibration)
        else:
            self.calibration = calibration
            self.profile_path = Path(f"{calibration.machine_id}.analytic.json")
        if min(hidden_size, intermediate_size, global_experts, local_experts) <= 0:
            raise ValueError("expert dimensions and counts must be positive")
        resolved_n_tile = self.calibration.backend_n_tile if backend_n_tile is None else int(backend_n_tile)
        if degree <= 0 or concurrent_ranks <= 0 or resolved_n_tile <= 0:
            raise ValueError("degree, concurrent_ranks, and backend_n_tile must be positive")
        if resolved_n_tile < 8 or resolved_n_tile % 8:
            raise ValueError(f"SVE backend_n_tile must be at least 8 and a multiple of 8, got {resolved_n_tile}")
        if hidden_size % 8 or intermediate_size % 8:
            raise ValueError("SVE packed K dimensions must be multiples of eight")
        if hidden_size % resolved_n_tile or (2 * intermediate_size) % resolved_n_tile:
            raise ValueError("SVE packed N dimensions must be multiples of backend_n_tile")
        if activation != "silu":
            raise ValueError(f"analytical SVE model supports only activation='silu', got {activation!r}")
        if dtype != "bf16":
            raise ValueError(f"analytical SVE model supports only dtype='bf16', got {dtype!r}")
        if down_output_element_bytes <= 0:
            raise ValueError("down_output_element_bytes must be positive")
        if weight_window_bytes < 0:
            raise ValueError("weight_window_bytes must be non-negative")
        if weight_window_bytes > 0 and w13_split:
            raise ValueError("positive weight_window_bytes uses canonical w13_split=False")
        if not w13_split and w13_split_chunks != 1:
            raise ValueError("w13_split_chunks must be one when w13_split is false")

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.local_experts = int(local_experts)
        self.down_output_element_bytes = int(down_output_element_bytes)
        self.weight_window_bytes = int(weight_window_bytes)
        self.w13_split_chunks = int(w13_split_chunks)
        self._exact_m = bool(exact_m)
        self._mapper = SveBf16KernelProfile(n_tile=resolved_n_tile, exact_m=exact_m)
        self._w13_geometry, self._w2_geometry = fused_moe_weight_windows(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            n_tile=resolved_n_tile,
            target_bytes=self.weight_window_bytes,
            w13_fallback_ranges=self.w13_split_chunks,
        )
        self.w13_window_ranges = self._w13_geometry.ranges
        self.w2_window_ranges = self._w2_geometry.ranges
        self.w13_chunk_bytes = self._w13_geometry.max_range_bytes
        self.w2_chunk_bytes = self._w2_geometry.max_range_bytes
        self.w2_bytes = self.w2_chunk_bytes
        self.max_stage_bytes = max(self.w13_chunk_bytes, self.w2_chunk_bytes)
        self.w13_tile_bytes = self.hidden_size * resolved_n_tile * 2
        self.w2_tile_bytes = self.intermediate_size * resolved_n_tile * 2
        self.call_setup_ns = self.calibration.overheads.call_setup_ns
        self.relative_error = self.calibration.relative_uncertainty
        self.profile = self.calibration.to_dict()

        widths = tuple(supported_widths or self.calibration.supported_widths)
        self.supported_widths = tuple(
            sorted({int(width) for width in widths if 0 < int(width) <= self.calibration.cores_per_rank})
        )
        if not self.supported_widths:
            raise ValueError("no supported thread widths remain")
        self._shape_keys = (
            tuple(tuple(int(value) for value in shape) for shape in supported_shapes)
            if supported_shapes is not None
            else analytic_candidate_shapes(self.calibration.cores_per_rank, self.supported_widths)
        )
        self._shapes_are_explicit = supported_shapes is not None
        if any(sum(shape) != self.calibration.cores_per_rank for shape in self._shape_keys):
            raise ValueError("every supported shape must cover cores_per_rank")
        self.policy = AnalyticPolicy(
            machine_id=self.calibration.machine_id,
            mode=str(mode),
            degree=int(degree),
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            global_experts=int(global_experts),
            local_experts=self.local_experts,
            backend="sve",
            backend_n_tile=resolved_n_tile,
            sve_implementation="jit" if exact_m else "asm",
            m_tail_policy="xbyak_exact_m" if exact_m else "static_bucketed",
            activation=str(activation),
            dtype=str(dtype),
            w13_split=bool(w13_split),
            w13_split_chunks=self.w13_split_chunks,
            weight_window_bytes=self.weight_window_bytes,
            w13_window_ranges=self.w13_window_ranges,
            w2_window_ranges=self.w2_window_ranges,
            measurement_experts=0,
            cores_per_rank=self.calibration.cores_per_rank,
            concurrent_ranks=int(concurrent_ranks),
            llc_bytes_per_rank=self.calibration.caches.llc_bytes_per_rank,
        )
        self.task_stage_window_policy = None

    @property
    def supported_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self._shape_keys

    def candidate_shapes(self, cores: int) -> tuple[tuple[int, ...], ...]:
        cores = int(cores)
        if cores <= 0 or cores > self.calibration.cores_per_rank:
            raise ValueError(f"cores must be in [1, {self.calibration.cores_per_rank}], got {cores}")
        if self._shapes_are_explicit:
            return tuple(shape for shape in self._shape_keys if sum(shape) == cores)
        return analytic_candidate_shapes(cores, self.supported_widths)

    def supports_shape(self, shape) -> bool:
        signature = tuple(int(value) for value in shape)
        if self._shapes_are_explicit:
            return signature in self._shape_keys
        if not signature or sum(signature) > self.calibration.cores_per_rank:
            return False
        return signature in self.candidate_shapes(sum(signature))

    def m12_tail_capacity(self, remainder: int) -> int:
        if remainder <= 0:
            return 0
        if self._exact_m:
            return remainder
        if remainder <= 2:
            return remainder
        if remainder <= 4:
            return 4
        if remainder <= 8:
            return 8
        return 12

    def m12_effective_rows(self, routes: int) -> int:
        blocks, remainder = divmod(max(int(routes), 0), 12)
        return blocks * 12 + self.m12_tail_capacity(remainder)

    def with_task_stage_window_policy(self, policy):
        """Return a clone that analytically lowers the policy-selected windows."""
        if not callable(getattr(policy, "select", None)):
            raise TypeError("task stage-window policy must provide select(routes, threads)")
        bound = copy.copy(self)
        bound.task_stage_window_policy = policy
        return bound

    @lru_cache(maxsize=4096)
    def _task_weight_geometries(
        self,
        routes: int,
        threads: int,
    ) -> tuple[WeightWindowGeometry, WeightWindowGeometry]:
        if self.task_stage_window_policy is None:
            return self._w13_geometry, self._w2_geometry
        w13_target, w2_target = self.task_stage_window_policy.select(int(routes), int(threads))
        if min(w13_target, w2_target) < -1:
            raise ValueError("task stage-window policy must return -1 or non-negative byte counts")
        if w13_target == -1 and w2_target == -1:
            return self._w13_geometry, self._w2_geometry
        return fused_moe_task_weight_windows(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            n_tile=self.policy.backend_n_tile,
            inherited_target_bytes=self.weight_window_bytes,
            w13_target_bytes=int(w13_target),
            w2_target_bytes=int(w2_target),
            w13_fallback_ranges=self.w13_split_chunks,
        )

    def task_max_stage_bytes(self, routes: int, threads: int) -> int:
        w13, w2 = self._task_weight_geometries(int(routes), int(threads))
        return max(w13.max_range_bytes, w2.max_range_bytes)

    def task_stage_ranges(self, routes: int, threads: int) -> tuple[int, int]:
        w13, w2 = self._task_weight_geometries(int(routes), int(threads))
        return w13.ranges, w2.ranges

    def window_bytes_per_worker(self, threads: int, routes: int | None = None) -> int:
        w13, w2 = (
            (self._w13_geometry, self._w2_geometry)
            if routes is None
            else self._task_weight_geometries(int(routes), int(threads))
        )
        return max(
            w13.bytes_per_worker(threads),
            w2.bytes_per_worker(threads),
        )

    def _l2_miss_fraction(self, working_set_bytes: float) -> float:
        cache = self.calibration.caches
        return _smooth_capacity_miss(
            working_set_bytes,
            cache.effective_l2_bytes_per_core,
            cache.l2_bytes_per_core,
        )

    def _l2_b_reuse_miss_fraction(self, working_set_bytes: float) -> float:
        cache = self.calibration.caches
        floor = cache.l2_b_reuse_miss_floor
        at_capacity = cache.l2_b_reuse_miss_at_capacity
        ceiling = cache.l2_b_reuse_miss_ceiling
        if working_set_bytes <= cache.l2_bytes_per_core:
            transition = _smooth_capacity_miss(
                working_set_bytes,
                cache.effective_l2_b_reuse_bytes_per_core,
                cache.l2_bytes_per_core,
            )
            return floor + (at_capacity - floor) * transition
        transition = _smooth_capacity_miss(
            working_set_bytes,
            cache.l2_bytes_per_core,
            2.0 * cache.l2_bytes_per_core,
        )
        return at_capacity + (ceiling - at_capacity) * transition

    def _l2_steady_scan_miss_fraction(self, working_set_bytes: float) -> float:
        """Return the physical resident/streaming state of a cyclic L2 scan."""
        return float(working_set_bytes > self.calibration.caches.l2_bytes_per_core)

    @lru_cache(maxsize=16384)
    def score_stage_window(
        self,
        stage: str,
        routes: int,
        threads: int,
        target_bytes: int,
    ) -> AnalyticStageWindowScore:
        """Score a stage window from physical service demand, not a route table.

        The regular isolated ECM uses ``max(core, transfer)`` because the two
        lower bounds overlap. That is appropriate for absolute latency but is
        too permissive for choosing a cache window: a transfer reduction hidden
        below the compute ceiling would otherwise have zero value, despite
        reducing refill latency and contention. Window selection therefore uses
        a serialized *incremental* objective. Candidate-independent compute and
        compulsory-B terms cancel; tile imbalance, L2/LLC refill, DRAM spill and
        calibrated per-range control remain visible.
        """
        stage = str(stage)
        routes = int(routes)
        threads = int(threads)
        target_bytes = int(target_bytes)
        if stage not in {"w13", "w2"}:
            raise ValueError(f"unsupported stage {stage!r}")
        if routes <= 0 or threads <= 0 or target_bytes <= 0:
            raise ValueError("routes, threads and target_bytes must be positive")
        if threads not in self.supported_widths:
            raise KeyError(f"unsupported analytical thread width {threads}")

        work = fused_expert_work(
            routes,
            self.hidden_size,
            self.intermediate_size,
            down_output_element_bytes=self.down_output_element_bytes,
        )
        if stage == "w13":
            logical = work.w13
            k = self.hidden_size
            n = 2 * self.intermediate_size
            fallback_ranges = self.w13_split_chunks
        else:
            logical = work.w2
            k = self.intermediate_size
            n = self.hidden_size
            fallback_ranges = 1
        geometry = stage_weight_window_geometry(
            k=k,
            n=n,
            n_tile=self.policy.backend_n_tile,
            target_bytes=target_bytes,
            fallback_ranges=fallback_ranges,
        )
        mapping = self._mapper.lower(
            logical,
            ExecutionSchedule(threads=threads, sequential_n_ranges=geometry.ranges),
        )
        demand = self._stage_demand(stage, mapping, geometry)
        phases = self._stage_phases(demand)
        serialized_core_ns = sum(phase.gemm_core_ns + phase.epilogue_ns for phase in phases)
        transfer_ns = 0.0
        range_overhead_ns = 0.0
        for phase in phases:
            if phase.kind == "range_setup":
                range_overhead_ns += phase.fixed_ns
                continue
            times = phase.resource_times_ns()
            transfer_ns += max(times["l2_bytes"], times["llc_bytes"], times["dram_bytes"])
        return AnalyticStageWindowScore(
            stage=stage,
            routes=routes,
            threads=threads,
            target_bytes=target_bytes,
            ranges=geometry.ranges,
            range_bytes=geometry.max_range_bytes,
            worker_bytes=geometry.bytes_per_worker(threads),
            active_threads=geometry.active_threads(threads),
            objective_ns=serialized_core_ns + transfer_ns + range_overhead_ns,
            serialized_core_ns=serialized_core_ns,
            transfer_ns=transfer_ns,
            range_overhead_ns=range_overhead_ns,
            l2_miss_fraction_a=demand.l2_miss_fraction_a,
            l2_miss_fraction_b=demand.l2_miss_fraction_b,
            a_l2_refill_bytes=demand.a_l2_refill_bytes,
            b_l2_refill_bytes=demand.b_l2_refill_bytes,
            l2_bytes=demand.l2_bytes,
            llc_bytes=demand.llc_bytes,
            compulsory_dram_bytes=demand.compulsory_dram_bytes,
            spillable_dram_bytes=demand.spillable_dram_bytes,
        )

    def default_task_stage_window_policy(self, *, num_cores: int, cpu_ids: Sequence[int]):
        """Build the deterministic analytical policy for this model instance."""
        if int(num_cores) <= 0 or int(num_cores) > self.calibration.cores_per_rank:
            return None
        if len(tuple(cpu_ids)) != int(num_cores):
            return None
        try:
            from analytic_stage_window_policy import AnalyticStageWindowPolicy
        except ImportError:  # pragma: no cover - package-style import
            from .analytic_stage_window_policy import AnalyticStageWindowPolicy
        return AnalyticStageWindowPolicy(self)

    def _llc_miss_fraction(self, working_set_bytes: float) -> float:
        cache = self.calibration.caches
        return _smooth_capacity_miss(
            working_set_bytes,
            cache.effective_llc_bytes_per_rank,
            cache.llc_bytes_per_rank,
        )

    def _stage_demand(
        self,
        stage: str,
        mapping: SveBf16KernelExecution,
        geometry: WeightWindowGeometry,
    ) -> AnalyticStageDemand:
        if stage not in {"w13", "w2"}:
            raise ValueError(f"unsupported stage {stage!r}")
        panels = len(mapping.panels)
        if panels == 0:
            return AnalyticStageDemand(
                stage=stage,
                mapping=mapping,
                ranges=geometry.ranges,
                window_bytes=0,
                owner_window_bytes=0,
                reusable_b_bytes=0,
                l2_miss_fraction_b=0.0,
                l2_miss_fraction_a=0.0,
                a_l2_refill_bytes=0.0,
                b_l2_refill_bytes=0.0,
                l2_refill_bytes=0.0,
                c_write_bytes=0.0,
                l2_bytes=0.0,
                llc_bytes=0.0,
                compulsory_dram_bytes=0.0,
                spillable_dram_bytes=0.0,
                llc_working_set_bytes=0.0,
                range_demands=(),
            )

        logical = mapping.logical_work
        a_bytes = mapping.compute_rows * logical.k * logical.input_element_bytes
        max_panel_rows = max(panel.compute_rows for panel in mapping.panels)
        a_panel_bytes = max_panel_rows * logical.k * logical.input_element_bytes
        total_tiles = mapping.allocation.total_tiles
        tile_bytes = logical.k * mapping.n_tile * logical.weight_element_bytes
        balanced_tiles = tuple(
            math.ceil(tiles / mapping.schedule.threads) * min(tiles, mapping.schedule.threads)
            for tiles in mapping.allocation.range_tiles
        )
        aggregate_balanced_tiles = mapping.allocation.busiest_thread_tiles * mapping.allocation.active_threads
        seen_active_threads = 0
        range_demands: list[AnalyticRangeDemand] = []
        for range_tiles, balanced_range_tiles in zip(
            mapping.allocation.range_tiles,
            balanced_tiles,
        ):
            active_threads = min(range_tiles, mapping.schedule.threads)
            owner_tiles = math.ceil(range_tiles / mapping.schedule.threads)
            owner_window_bytes = owner_tiles * tile_bytes
            weight_bytes = range_tiles * tile_bytes
            c_write_bytes = mapping.llc_c_write_bytes * range_tiles / total_tiles

            # Preserve one panel of L2 headroom for the kernel's in-flight
            # load/prefetch state. Below this physical capacity boundary the
            # full packed A survives sequential N ranges; above it, A is a
            # cyclic stream and each owner must refill it.
            a_residency_footprint = owner_window_bytes + a_bytes
            a_residency_capacity = max(
                self.calibration.caches.l2_bytes_per_core - a_panel_bytes,
                0,
            )
            a_l2_miss = float(a_residency_footprint > a_residency_capacity)

            b_reuse_footprint = owner_window_bytes + a_panel_bytes
            b_l2_miss = self._l2_b_reuse_miss_fraction(b_reuse_footprint)
            available_b_reuses = panels - 1
            # Once one effective L2 of unique A panels has streamed past, the
            # cyclic B stripe reaches its physical resident/streaming state.
            # Before that turnover point, use the calibrated repeated-scan
            # miss curve. This derives the transition from cache and panel
            # geometry instead of multiplying one short-route miss forever.
            cache_turnover_reuses = max(
                int(self.calibration.caches.effective_l2_bytes_per_core // a_panel_bytes),
                1,
            )
            transient_b_reuses = min(
                available_b_reuses,
                cache_turnover_reuses,
            )
            steady_b_reuses = available_b_reuses - transient_b_reuses
            b_l2_steady_miss = self._l2_steady_scan_miss_fraction(b_reuse_footprint)
            b_l2_refill = weight_bytes * (
                1.0 + transient_b_reuses * b_l2_miss + steady_b_reuses * b_l2_steady_miss
            )
            new_threads = max(active_threads - seen_active_threads, 0)
            reused_threads = active_threads - new_threads
            a_l2_refill = a_bytes * (new_threads + reused_threads * a_l2_miss)
            seen_active_threads = max(seen_active_threads, active_threads)

            l2_bytes = a_l2_refill + b_l2_refill + c_write_bytes
            reusable_b_bytes = weight_bytes if panels > 1 else 0
            spillable_dram = a_l2_refill + max(b_l2_refill - weight_bytes, 0.0) + c_write_bytes
            range_demands.append(
                AnalyticRangeDemand(
                    n_tiles=range_tiles,
                    balanced_work_fraction=balanced_range_tiles / aggregate_balanced_tiles,
                    owner_window_bytes=owner_window_bytes,
                    l2_miss_fraction_b=b_l2_miss,
                    l2_steady_miss_fraction_b=b_l2_steady_miss,
                    l2_miss_fraction_a=a_l2_miss,
                    b_transient_reuses=transient_b_reuses,
                    b_steady_reuses=steady_b_reuses,
                    b_reuse_footprint_bytes=b_reuse_footprint,
                    a_residency_footprint_bytes=a_residency_footprint,
                    a_residency_capacity_bytes=a_residency_capacity,
                    a_l2_refill_bytes=a_l2_refill,
                    b_l2_refill_bytes=b_l2_refill,
                    c_write_bytes=c_write_bytes,
                    l2_bytes=l2_bytes,
                    llc_bytes=l2_bytes,
                    compulsory_dram_bytes=weight_bytes,
                    spillable_dram_bytes=spillable_dram,
                    reusable_b_bytes=reusable_b_bytes,
                    llc_working_set_bytes=reusable_b_bytes + a_bytes + c_write_bytes,
                )
            )

        a_l2_refill = sum(item.a_l2_refill_bytes for item in range_demands)
        b_l2_refill = sum(item.b_l2_refill_bytes for item in range_demands)
        c_write_bytes = sum(item.c_write_bytes for item in range_demands)
        l2_refill_bytes = a_l2_refill + b_l2_refill
        l2_bytes = sum(item.l2_bytes for item in range_demands)
        llc_bytes = sum(item.llc_bytes for item in range_demands)
        compulsory_dram = sum(item.compulsory_dram_bytes for item in range_demands)
        spillable_dram = sum(item.spillable_dram_bytes for item in range_demands)
        owner_window_bytes = max(item.owner_window_bytes for item in range_demands)
        reusable_b_bytes = max(item.reusable_b_bytes for item in range_demands)
        llc_working_set = max(item.llc_working_set_bytes for item in range_demands)
        b_l2_miss = (
            sum(item.l2_miss_fraction_b * item.compulsory_dram_bytes for item in range_demands) / compulsory_dram
        )
        active_scans = tuple(min(tiles, mapping.schedule.threads) for tiles in mapping.allocation.range_tiles)
        a_l2_miss = sum(item.l2_miss_fraction_a * scans for item, scans in zip(range_demands, active_scans)) / sum(
            active_scans
        )
        return AnalyticStageDemand(
            stage=stage,
            mapping=mapping,
            ranges=geometry.ranges,
            window_bytes=geometry.max_range_bytes,
            owner_window_bytes=owner_window_bytes,
            reusable_b_bytes=reusable_b_bytes,
            l2_miss_fraction_b=b_l2_miss,
            l2_miss_fraction_a=a_l2_miss,
            a_l2_refill_bytes=a_l2_refill,
            b_l2_refill_bytes=b_l2_refill,
            l2_refill_bytes=l2_refill_bytes,
            c_write_bytes=c_write_bytes,
            l2_bytes=l2_bytes,
            llc_bytes=llc_bytes,
            compulsory_dram_bytes=compulsory_dram,
            spillable_dram_bytes=spillable_dram,
            llc_working_set_bytes=llc_working_set,
            range_demands=tuple(range_demands),
        )

    def _stage_phases(self, demand: AnalyticStageDemand) -> tuple[AnalyticPhase, ...]:
        mapping = demand.mapping
        machine = self.calibration
        residual_scale = machine.w13_scale if demand.stage == "w13" else machine.w2_scale
        logical = mapping.logical_work
        vector_bytes = mapping.vector_bytes
        output_columns_per_tile = logical.output_columns * mapping.n_tile // logical.n
        total_compute_rows = sum(panel.compute_rows for panel in mapping.panels)
        total_store_rows = sum(panel.store_rows for panel in mapping.panels)
        phases: list[AnalyticPhase] = []

        def append_phase(
            *,
            range_index: int,
            phase_kind: str,
            panels,
            active_threads: int,
            balanced_tiles: int,
            a_l2_bytes: float,
            b_l2_bytes: float,
            c_write_bytes: float,
            compulsory_dram_bytes: float,
            spillable_dram_bytes: float,
            working_set_bytes: float,
        ) -> None:
            compute_rows = sum(panel.compute_rows for panel in panels)
            store_rows = sum(panel.store_rows for panel in panels)
            panel_count = len(panels)
            bfmmla_instructions = compute_rows * logical.k // 2 * balanced_tiles
            a_load_instructions = compute_rows * logical.k // 8 * balanced_tiles
            b_load_instructions = panel_count * logical.k * balanced_tiles
            matrix_flops = bfmmla_instructions * (2 * vector_bytes)
            frontend_instructions = bfmmla_instructions + a_load_instructions + b_load_instructions
            l1_bytes = a_load_instructions * 16 + b_load_instructions * vector_bytes
            epilogue_elements = store_rows * output_columns_per_tile * balanced_tiles
            refill_bytes = a_l2_bytes + b_l2_bytes
            transfer_bytes = refill_bytes + c_write_bytes
            gemm_core_ns = matrix_flops / machine.service_rate("gemm_core_flops", active_threads) * 1e9
            matrix_ns = matrix_flops / machine.service_rate("matrix_flops", active_threads) * 1e9
            frontend_ns = 0.0
            if machine.frontend_instructions is not None:
                frontend_ns = (
                    frontend_instructions / machine.service_rate("frontend_instructions", active_threads) * 1e9
                )
            l1_ns = l1_bytes / machine.service_rate("l1_bytes", active_threads) * 1e9
            l2_ns = transfer_bytes / machine.service_rate("l2_bytes", active_threads) * 1e9
            llc_ns = transfer_bytes / machine.service_rate("llc_bytes", active_threads) * 1e9
            epilogue_ns = 0.0
            if machine.epilogue_elements is not None:
                epilogue_ns = epilogue_elements / machine.service_rate("epilogue_elements", active_threads) * 1e9
            phases.append(
                AnalyticPhase(
                    name=f"{demand.stage}:r{range_index}:{phase_kind}",
                    kind=phase_kind,
                    panel_count=panel_count,
                    active_threads=active_threads,
                    fixed_ns=0.0,
                    gemm_core_ns=gemm_core_ns,
                    matrix_ns=matrix_ns,
                    frontend_ns=frontend_ns,
                    l1_ns=l1_ns,
                    l2_ns=l2_ns,
                    llc_ns=llc_ns,
                    epilogue_ns=epilogue_ns,
                    matrix_flops=matrix_flops,
                    frontend_instructions=frontend_instructions,
                    l1_bytes=l1_bytes,
                    l2_bytes=transfer_bytes,
                    llc_bytes=transfer_bytes,
                    epilogue_elements=epilogue_elements,
                    compulsory_dram_bytes=compulsory_dram_bytes,
                    spillable_dram_bytes=spillable_dram_bytes,
                    dram_rate=machine.service_rate("dram_bytes", active_threads),
                    working_set_bytes=working_set_bytes,
                    isolated_spill_fraction=self._llc_miss_fraction(working_set_bytes),
                    residual_scale=residual_scale,
                )
            )

        for index, range_demand in enumerate(demand.range_demands):
            stage_fraction = range_demand.n_tiles / mapping.allocation.total_tiles
            active_threads = min(range_demand.n_tiles, mapping.schedule.threads)
            balanced_tiles = math.ceil(range_demand.n_tiles / mapping.schedule.threads) * active_threads
            setup_ns = machine.overheads.stage_fixed_ns * stage_fraction + machine.overheads.range_fixed_ns
            if setup_ns > 0.0:
                phases.append(
                    AnalyticPhase(
                        name=f"{demand.stage}:r{index}:setup",
                        kind="range_setup",
                        panel_count=0,
                        active_threads=active_threads,
                        fixed_ns=setup_ns,
                        gemm_core_ns=0.0,
                        matrix_ns=0.0,
                        frontend_ns=0.0,
                        l1_ns=0.0,
                        l2_ns=0.0,
                        llc_ns=0.0,
                        epilogue_ns=0.0,
                        matrix_flops=0.0,
                        frontend_instructions=0.0,
                        l1_bytes=0.0,
                        l2_bytes=0.0,
                        llc_bytes=0.0,
                        epilogue_elements=0.0,
                        compulsory_dram_bytes=0.0,
                        spillable_dram_bytes=0.0,
                        dram_rate=machine.service_rate("dram_bytes", active_threads),
                        working_set_bytes=0.0,
                        isolated_spill_fraction=0.0,
                        residual_scale=residual_scale,
                    )
                )
            cold_panels = mapping.panels[:1]
            steady_panels = mapping.panels[1:]
            cold_compute_fraction = cold_panels[0].compute_rows / total_compute_rows
            cold_store_fraction = cold_panels[0].store_rows / total_store_rows
            cold_a_l2_bytes = range_demand.a_l2_refill_bytes * cold_compute_fraction
            cold_c_write_bytes = range_demand.c_write_bytes * cold_store_fraction
            cold_b_l2_bytes = range_demand.compulsory_dram_bytes
            append_phase(
                range_index=index,
                phase_kind="cold_b",
                panels=cold_panels,
                active_threads=active_threads,
                balanced_tiles=balanced_tiles,
                a_l2_bytes=cold_a_l2_bytes,
                b_l2_bytes=cold_b_l2_bytes,
                c_write_bytes=cold_c_write_bytes,
                compulsory_dram_bytes=range_demand.compulsory_dram_bytes,
                spillable_dram_bytes=cold_a_l2_bytes + cold_c_write_bytes,
                working_set_bytes=range_demand.llc_working_set_bytes,
            )
            if steady_panels:
                append_phase(
                    range_index=index,
                    phase_kind="steady_b",
                    panels=steady_panels,
                    active_threads=active_threads,
                    balanced_tiles=balanced_tiles,
                    a_l2_bytes=range_demand.a_l2_refill_bytes - cold_a_l2_bytes,
                    b_l2_bytes=range_demand.b_l2_refill_bytes - cold_b_l2_bytes,
                    c_write_bytes=range_demand.c_write_bytes - cold_c_write_bytes,
                    compulsory_dram_bytes=0.0,
                    spillable_dram_bytes=range_demand.spillable_dram_bytes - cold_a_l2_bytes - cold_c_write_bytes,
                    working_set_bytes=range_demand.llc_working_set_bytes,
                )
        return tuple(phases)

    @lru_cache(maxsize=4096)
    def predict_expert(self, routes: int, threads: int) -> ExpertPrediction:
        routes = int(routes)
        threads = int(threads)
        if routes <= 0:
            raise ValueError("predict_expert requires positive routes")
        if threads not in self.supported_widths:
            raise KeyError(f"unsupported analytical thread width {threads}")
        work = fused_expert_work(
            routes,
            self.hidden_size,
            self.intermediate_size,
            down_output_element_bytes=self.down_output_element_bytes,
        )
        w13_geometry, w2_geometry = self._task_weight_geometries(routes, threads)
        w13_mapping = self._mapper.lower(
            work.w13,
            ExecutionSchedule(threads=threads, sequential_n_ranges=w13_geometry.ranges),
        )
        w2_mapping = self._mapper.lower(
            work.w2,
            ExecutionSchedule(threads=threads, sequential_n_ranges=w2_geometry.ranges),
        )
        w13_demand = self._stage_demand("w13", w13_mapping, w13_geometry)
        w2_demand = self._stage_demand("w2", w2_mapping, w2_geometry)
        overhead_ns = self.calibration.overheads.expert_fixed_ns + routes * self.calibration.overheads.route_ns
        phases: list[AnalyticPhase] = []
        if overhead_ns > 0.0:
            phases.append(
                AnalyticPhase(
                    name="operator",
                    kind="operator",
                    panel_count=0,
                    active_threads=threads,
                    fixed_ns=overhead_ns,
                    gemm_core_ns=0.0,
                    matrix_ns=0.0,
                    frontend_ns=0.0,
                    l1_ns=0.0,
                    l2_ns=0.0,
                    llc_ns=0.0,
                    epilogue_ns=0.0,
                    matrix_flops=0.0,
                    frontend_instructions=0.0,
                    l1_bytes=0.0,
                    l2_bytes=0.0,
                    llc_bytes=0.0,
                    epilogue_elements=0.0,
                    compulsory_dram_bytes=0.0,
                    spillable_dram_bytes=0.0,
                    dram_rate=self.calibration.service_rate("dram_bytes", threads),
                    working_set_bytes=0.0,
                    isolated_spill_fraction=0.0,
                )
            )
        phases.extend(self._stage_phases(w13_demand))
        phases.extend(self._stage_phases(w2_demand))
        return ExpertPrediction(
            routes=routes,
            threads=threads,
            phases=tuple(phases),
            w13_demand=w13_demand,
            w2_demand=w2_demand,
        )

    def T_iso(self, routes: int, threads: int) -> float:
        if int(routes) <= 0:
            return 0.0
        return self.predict_expert(int(routes), int(threads)).total_ns

    def _task_phases(self, routes: int, threads: int) -> list[tuple[float, int]]:
        prediction = self.predict_expert(routes, threads)
        return [(phase.base_ns, int(phase.working_set_bytes)) for phase in prediction.phases]

    def relative_uncertainty(self, routes: int, shape) -> float:
        del routes, shape
        return self.relative_error

    def relative_full_call_uncertainty(self, routes: int, shape) -> float:
        return self.relative_uncertainty(routes, shape)

    def profiled_full_call_time(self, routes: int, shape) -> float:
        del routes, shape
        raise RuntimeError("analytical models do not contain full-workload timing anchors")

    @staticmethod
    def _dag_state(tasks):
        count = len(tasks)
        dependency_count = [len(dependencies) for _, _, dependencies in tasks]
        successors = [[] for _ in range(count)]
        for task_id, (_, _, dependencies) in enumerate(tasks):
            for dependency in dependencies:
                if dependency < 0 or dependency >= task_id:
                    raise ValueError(
                        f"task dependencies must refer to earlier tasks: task={task_id}, dependency={dependency}"
                    )
                successors[dependency].append(task_id)
        return dependency_count, successors, [count == 0 or value == 0 for value in dependency_count]

    def _active_phase_state(
        self,
        current: Mapping[int, AnalyticPhase],
    ) -> tuple[
        float,
        dict[int, float],
        dict[str, AnalyticResourcePressure],
        dict[int, float],
    ]:
        total_working_set = sum(phase.working_set_bytes for phase in current.values())
        spill_fraction = self._llc_miss_fraction(total_working_set)
        provisional = {index: phase.duration_ns(spill_fraction=spill_fraction) for index, phase in current.items()}
        pressures: dict[str, AnalyticResourcePressure] = {}
        resource_scales: dict[str, float] = {}
        for resource in _SHARED_RESOURCES:
            demands = {index: phase.resource_demand(resource, spill_fraction) for index, phase in current.items()}
            active_threads = min(
                sum(current[index].active_threads for index, demand in demands.items() if demand > 0.0),
                self.calibration.cores_per_rank,
            )
            capacity = self.calibration.service_rate(resource, active_threads) if active_threads > 0 else math.inf
            # Request rate is measured over the interval in which this resource
            # is active, not averaged over the whole ECM phase.  Averaging over
            # compute time and then scaling only the resource term can violate
            # the calibrated aggregate capacity.  Dividing by the resource's
            # own service time gives the independently requested service; the
            # ECM max/sum composition below decides whether that service is
            # hidden by another component.
            if math.isfinite(capacity):
                offered_rate = sum(
                    demand
                    / max(
                        current[index].residual_scale
                        * current[index].resource_times_ns(spill_fraction)[resource]
                        * 1e-9,
                        1e-30,
                    )
                    for index, demand in demands.items()
                    if demand > 0.0
                )
            else:
                offered_rate = sum(demand / max(provisional[index] * 1e-9, 1e-30) for index, demand in demands.items())
            utilization = offered_rate / capacity if math.isfinite(capacity) else 0.0
            dilation = max(1.0, utilization)
            allocated_rate = offered_rate / dilation
            allocated_utilization = allocated_rate / capacity if math.isfinite(capacity) else 0.0
            pressures[resource] = AnalyticResourcePressure(
                active_threads=active_threads,
                offered_rate=offered_rate,
                capacity=capacity,
                utilization=utilization,
                dilation=dilation,
                allocated_rate=allocated_rate,
                allocated_utilization=allocated_utilization,
            )
            resource_scales[resource] = dilation
        multipliers = {
            index: (
                phase.duration_ns(
                    spill_fraction=spill_fraction,
                    resource_scales=resource_scales,
                )
                / phase.base_ns
            )
            for index, phase in current.items()
        }
        return spill_fraction, provisional, pressures, multipliers

    def active_resource_pressure(self, phases: Sequence[AnalyticPhase]) -> dict[str, AnalyticResourcePressure]:
        """Return physically named offered-load pressure for one concurrent phase set."""
        current = {index: phase for index, phase in enumerate(phases)}
        if not current:
            return {}
        return self._active_phase_state(current)[2]

    @staticmethod
    def _pressure_dict(pressure: AnalyticResourcePressure) -> dict[str, float | int | None]:
        return {
            "active_threads": pressure.active_threads,
            "offered_rate": pressure.offered_rate,
            "capacity": pressure.capacity if math.isfinite(pressure.capacity) else None,
            "utilization": pressure.utilization,
            "dilation": pressure.dilation,
            "allocated_rate": pressure.allocated_rate,
            "allocated_utilization": pressure.allocated_utilization,
        }

    def _dag_result(
        self,
        tasks,
        *,
        event_log: list[dict] | None = None,
    ) -> tuple[float, tuple[float, ...]]:
        tasks = [(int(routes), int(threads), list(dependencies)) for routes, threads, dependencies in tasks]
        if not tasks:
            return 0.0, ()
        phases = [self.predict_expert(routes, threads).phases for routes, threads, _ in tasks]
        if any(not task_phases for task_phases in phases):
            raise ValueError("analytical DAG tasks must have positive route counts")
        phase_index = [0] * len(tasks)
        remaining = [task_phases[0].base_ns for task_phases in phases]
        dependency_count, successors, started = self._dag_state(tasks)
        finished = [False] * len(tasks)
        finish_times = [0.0] * len(tasks)
        wall_ns = self.call_setup_ns
        max_events = sum(len(task_phases) for task_phases in phases) + len(tasks) + 2
        guard = 0

        while not all(finished):
            guard += 1
            if guard > 2 * max_events:
                raise RuntimeError("analytical stage simulator did not converge")
            active = [index for index in range(len(tasks)) if started[index] and not finished[index]]
            if not active:
                raise ValueError("DAG deadlock (cycle or unreachable task)")
            current = {index: phases[index][phase_index[index]] for index in active}
            spill_fraction, _, pressures, multipliers = self._active_phase_state(current)
            elapsed = min(remaining[index] * multipliers[index] for index in active)
            if event_log is not None:
                event_log.append(
                    {
                        "start_ns": wall_ns,
                        "duration_ns": elapsed,
                        "active_tasks": list(active),
                        "phases": {str(index): current[index].name for index in active},
                        "phase_kinds": {str(index): current[index].kind for index in active},
                        "phase_dilation": {str(index): multipliers[index] for index in active},
                        "working_set_bytes": sum(phase.working_set_bytes for phase in current.values()),
                        "llc_spill_fraction": spill_fraction,
                        "resources": {
                            resource: {
                                "path": _RESOURCE_PATHS[resource],
                                **self._pressure_dict(pressure),
                            }
                            for resource, pressure in pressures.items()
                            if pressure.offered_rate > 0.0
                        },
                    }
                )
            wall_ns += elapsed
            for index in active:
                remaining[index] -= elapsed / multipliers[index]

            completed = [index for index in active if remaining[index] <= 1e-6]
            for index in completed:
                phase_index[index] += 1
                if phase_index[index] < len(phases[index]):
                    remaining[index] = phases[index][phase_index[index]].base_ns
                    continue
                finished[index] = True
                finish_times[index] = wall_ns
                for successor in successors[index]:
                    dependency_count[successor] -= 1
                    if dependency_count[successor] == 0:
                        started[successor] = True
        return wall_ns, tuple(finish_times)

    def dag_makespan(self, tasks) -> float:
        return self._dag_result(tasks)[0]

    def dag_task_finish_times(self, tasks) -> tuple[float, ...]:
        """Return task completion timestamps from the analytical simulator."""
        return self._dag_result(tasks)[1]

    def explain_dag(self, tasks) -> dict:
        """Run the analytical DAG and expose each resource-allocation event."""
        normalized = [(int(routes), int(threads), list(dependencies)) for routes, threads, dependencies in tasks]
        events: list[dict] = []
        makespan_ns, finish_times = self._dag_result(normalized, event_log=events)
        return {
            "model": ANALYTIC_MODEL_NAME,
            "machine_id": self.calibration.machine_id,
            "makespan_ns": makespan_ns,
            "task_finish_ns": list(finish_times),
            "events": events,
        }

    def phase_makespan(self, tasks) -> float:
        return self.dag_makespan((routes, threads, []) for routes, threads in tasks)

    def flat_dag_makespan(self, tasks) -> float:
        return self.dag_makespan(tasks)

    def scalar_makespan(self, tasks) -> float:
        return self.phase_makespan(tasks)

    def derate(
        self,
        n: int,
        routes: int | None = None,
        max_threads: int | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> float:
        if n <= 1:
            return 1.0
        route_count = max(int(routes or 1), 1)
        widths = tuple(shape or (max_threads or self.supported_widths[0],) * n)
        if len(widths) != n:
            widths = (max(widths),) * n
        concurrent = self.phase_makespan([(route_count, width) for width in widths])
        isolated = max(self.T_iso(route_count, width) for width in widths)
        return max(concurrent / isolated, 1.0)

    def explain(self, routes: int, threads: int) -> dict:
        prediction = self.predict_expert(routes, threads)

        def demand(stage: AnalyticStageDemand) -> dict:
            stage_phases = tuple(phase for phase in prediction.phases if phase.name.startswith(f"{stage.stage}:"))
            return {
                "ranges": stage.ranges,
                "window_bytes": stage.window_bytes,
                "owner_window_bytes": stage.owner_window_bytes,
                "reusable_b_bytes": stage.reusable_b_bytes,
                "l2_miss_fraction_b": stage.l2_miss_fraction_b,
                "l2_miss_fraction_a": stage.l2_miss_fraction_a,
                "a_l2_refill_bytes": stage.a_l2_refill_bytes,
                "b_l2_refill_bytes": stage.b_l2_refill_bytes,
                "l2_refill_bytes": stage.l2_refill_bytes,
                "c_write_bytes": stage.c_write_bytes,
                "l2_bytes": stage.l2_bytes,
                "llc_bytes": stage.llc_bytes,
                "compulsory_dram_bytes": stage.compulsory_dram_bytes,
                "spillable_dram_bytes": stage.spillable_dram_bytes,
                "llc_working_set_bytes": stage.llc_working_set_bytes,
                "executed_flops": sum(phase.matrix_flops for phase in stage_phases),
                "mapping_balanced_executed_flops_upper_bound": (stage.mapping.demand.balanced_executed_flops),
                "active_threads": stage.mapping.demand.active_threads,
                "range_demands": [
                    {
                        "n_tiles": item.n_tiles,
                        "balanced_work_fraction": item.balanced_work_fraction,
                        "owner_window_bytes": item.owner_window_bytes,
                        "b_reuse_footprint_bytes": item.b_reuse_footprint_bytes,
                        "b_transient_reuses": item.b_transient_reuses,
                        "b_steady_reuses": item.b_steady_reuses,
                        "l2_steady_miss_fraction_b": item.l2_steady_miss_fraction_b,
                        "a_residency_footprint_bytes": item.a_residency_footprint_bytes,
                        "a_residency_capacity_bytes": item.a_residency_capacity_bytes,
                        "a_l2_refill_bytes": item.a_l2_refill_bytes,
                        "b_l2_refill_bytes": item.b_l2_refill_bytes,
                        "compulsory_dram_bytes": item.compulsory_dram_bytes,
                        "spillable_dram_bytes": item.spillable_dram_bytes,
                        "llc_working_set_bytes": item.llc_working_set_bytes,
                    }
                    for item in stage.range_demands
                ],
            }

        def phase_detail(phase: AnalyticPhase) -> dict:
            pressures = self.active_resource_pressure((phase,))
            resource_times = phase.resource_times_ns()
            if phase.gemm_core_ns > 0.0:
                transfer_ns = max(resource_times[name] for name in ("l2_bytes", "llc_bytes", "dram_bytes"))
                body_components = {
                    "gemm_core": resource_times["gemm_core_flops"],
                    "transfer": transfer_ns,
                }
            else:
                transfer_ns = max(resource_times[name] for name in ("l1_bytes", "l2_bytes", "llc_bytes", "dram_bytes"))
                body_components = {
                    "matrix": resource_times["matrix_flops"],
                    "frontend": resource_times["frontend_instructions"],
                    "transfer": transfer_ns,
                }
            bottleneck = max(body_components, key=body_components.get)
            return {
                "name": phase.name,
                "kind": phase.kind,
                "panel_count": phase.panel_count,
                "base_ns": phase.base_ns,
                "fixed_ns": phase.fixed_ns,
                "gemm_core_ns": phase.gemm_core_ns,
                "matrix_ns": phase.matrix_ns,
                "frontend_ns": phase.frontend_ns,
                "l1_ns": phase.l1_ns,
                "l2_ns": phase.l2_ns,
                "llc_ns": phase.llc_ns,
                "dram_ns": resource_times["dram_bytes"],
                "epilogue_ns": phase.epilogue_ns,
                "ecm_body_bottleneck": bottleneck,
                "working_set_bytes": phase.working_set_bytes,
                "matrix_flops": phase.matrix_flops,
                "l1_bytes": phase.l1_bytes,
                "l2_bytes": phase.l2_bytes,
                "llc_bytes": phase.llc_bytes,
                "compulsory_dram_bytes": phase.compulsory_dram_bytes,
                "spillable_dram_bytes": phase.spillable_dram_bytes,
                "resource_pressure": {
                    resource: {
                        "path": _RESOURCE_PATHS[resource],
                        **self._pressure_dict(pressure),
                    }
                    for resource, pressure in pressures.items()
                    if pressure.offered_rate > 0.0
                },
            }

        return {
            "model": ANALYTIC_MODEL_NAME,
            "machine_id": self.calibration.machine_id,
            "routes": routes,
            "threads": threads,
            "total_ns": prediction.total_ns,
            "w13_ns": prediction.w13_ns,
            "w2_ns": prediction.w2_ns,
            "w13": demand(prediction.w13_demand),
            "w2": demand(prediction.w2_demand),
            "phases": [phase_detail(phase) for phase in prediction.phases],
        }
