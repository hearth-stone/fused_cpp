"""Hardware-grounded analytical cost model for fused CPU MoE.

The model deliberately separates route-dependent work from machine
calibration:

* :mod:`gemm_cost_model` describes logical W13/W2 work;
* :mod:`sve_bf16_kernel_model` lowers it to physical SVE kernel demand;
* this module maps that demand through a small set of cache capacities,
  measured service curves, and fixed runtime costs.

Unlike ``ContentionCostModel``, no route/thread latency table or measured
contention shape is required. Concurrent tasks are simulated as W13/W2
setup/cold-B/steady-B phases which consume calibrated aggregate matrix,
frontend, cache, DRAM, and epilogue service ceilings.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from full_stage_geometry import FullStageGeometry, StageWindowPlan, full_stage_geometry
    from gemm_cost_model import ExecutionSchedule, fused_expert_work
    from sve_bf16_kernel_model import SveBf16KernelExecution, SveBf16KernelProfile
except ImportError:  # pragma: no cover - package-style import
    from .full_stage_geometry import FullStageGeometry, StageWindowPlan, full_stage_geometry
    from .gemm_cost_model import ExecutionSchedule, fused_expert_work
    from .sve_bf16_kernel_model import SveBf16KernelExecution, SveBf16KernelProfile


ANALYTIC_MACHINE_SCHEMA_VERSION = 2
SUPPORTED_ANALYTIC_MACHINE_SCHEMA_VERSIONS = frozenset({1, 2})
ANALYTIC_MODEL_SCHEMA_VERSION = 8
ANALYTIC_MODEL_NAME = "phase_ecm_llc_domain_team_pressure_v5"
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


@lru_cache(maxsize=1)
def _formula_source_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("analytic_model.py", "full_stage_geometry.py", "gemm_cost_model.py", "sve_bf16_kernel_model.py"):
        digest.update(name.encode("ascii"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SaturatingServiceCurve:
    """Monotone smooth or measured-point service curve.

    Compute resources may use a compact smooth family. Shared LLC/DRAM
    resources use piecewise interpolation so topology knees remain explicit.
    """

    single_thread_rate: float
    saturated_rate: float
    saturation_threads: int
    curve: str = "power"
    points: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if min(self.single_thread_rate, self.saturated_rate) <= 0.0:
            raise ValueError("service rates must be positive")
        if self.saturation_threads <= 0:
            raise ValueError("saturation_threads must be positive")
        if self.curve not in {"power", "shared_bottleneck", "piecewise_linear"}:
            raise ValueError("curve must be 'power', 'shared_bottleneck', or 'piecewise_linear'")
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
        if self.curve == "piecewise_linear":
            if not self.points or self.points[0][0] != 1:
                raise ValueError("piecewise_linear service requires a one-thread point")
            if any(thread <= 0 or rate <= 0.0 for thread, rate in self.points):
                raise ValueError("piecewise_linear points must be positive")
            if any(right[0] <= left[0] for left, right in zip(self.points, self.points[1:])):
                raise ValueError("piecewise_linear thread points must be strictly increasing")
            if any(right[1] < left[1] for left, right in zip(self.points, self.points[1:])):
                raise ValueError("piecewise_linear rates must be monotone")
            if not math.isclose(self.single_thread_rate, self.points[0][1], rel_tol=1e-12):
                raise ValueError("single_thread_rate must match the first piecewise point")
            if self.saturation_threads != self.points[-1][0] or not math.isclose(
                self.saturated_rate,
                self.points[-1][1],
                rel_tol=1e-12,
            ):
                raise ValueError("piecewise saturation anchor must match the final point")
        elif self.points:
            raise ValueError("service points are only valid for piecewise_linear curves")

    @property
    def exponent(self) -> float:
        if self.saturation_threads == 1 or self.saturated_rate == self.single_thread_rate:
            return 0.0
        return math.log(self.saturated_rate / self.single_thread_rate) / math.log(self.saturation_threads)

    def rate(self, active_threads: int) -> float:
        if active_threads <= 0:
            raise ValueError("active_threads must be positive")
        if self.curve == "piecewise_linear":
            if active_threads >= self.points[-1][0]:
                return self.points[-1][1]
            for (left_threads, left_rate), (right_threads, right_rate) in zip(
                self.points,
                self.points[1:],
            ):
                if active_threads <= right_threads:
                    fraction = (active_threads - left_threads) / (right_threads - left_threads)
                    return left_rate + fraction * (right_rate - left_rate)
            raise AssertionError("piecewise service interpolation did not find an interval")
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
        points = []
        for point in payload.get("points", ()):
            if isinstance(point, Mapping):
                points.append((int(point["threads"]), float(point["rate"])))
            else:
                threads, rate = point
                points.append((int(threads), float(rate)))
        return cls(
            single_thread_rate=float(payload["single_thread_rate"]),
            saturated_rate=float(payload["saturated_rate"]),
            saturation_threads=int(payload["saturation_threads"]),
            curve=str(payload.get("curve", "power")),
            points=tuple(points),
        )

    def to_dict(self) -> dict:
        payload = {
            "single_thread_rate": self.single_thread_rate,
            "saturated_rate": self.saturated_rate,
            "saturation_threads": self.saturation_threads,
            "curve": self.curve,
        }
        if self.curve == "piecewise_linear":
            payload["points"] = [
                {"threads": threads, "rate": rate}
                for threads, rate in self.points
            ]
        return payload


@dataclass(frozen=True)
class LlcDomainCalibration:
    """One shared LLC domain inside a NUMA scheduling rank."""

    domain_id: str
    cpu_ids: tuple[int, ...]
    capacity_bytes: int
    service: SaturatingServiceCurve

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("LLC domain id must be non-empty")
        if self.capacity_bytes <= 0:
            raise ValueError("LLC domain capacity must be positive")
        if not self.cpu_ids or min(self.cpu_ids) < 0 or len(set(self.cpu_ids)) != len(self.cpu_ids):
            raise ValueError("LLC domain CPU ids must be unique and non-negative")
        if self.service.saturation_threads > len(self.cpu_ids):
            raise ValueError("LLC domain service cannot saturate beyond its CPU count")

    @classmethod
    def from_dict(cls, payload: dict) -> "LlcDomainCalibration":
        return cls(
            domain_id=str(payload["id"]),
            cpu_ids=tuple(int(cpu) for cpu in payload["cpu_ids"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            service=SaturatingServiceCurve.from_dict(payload["service"]),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.domain_id,
            "cpu_ids": list(self.cpu_ids),
            "capacity_bytes": self.capacity_bytes,
            "service": self.service.to_dict(),
        }

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
    panel_range_restart_ns: float = 0.0
    w13_panel_range_restart_ns: float | None = None
    w2_panel_range_restart_ns: float | None = None
    by_width: tuple[tuple[int, float, float], ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.call_setup_ns,
            self.expert_fixed_ns,
            self.route_ns,
            self.stage_fixed_ns,
            self.range_fixed_ns,
            self.panel_range_restart_ns,
        ) + tuple(
            value
            for value in (self.w13_panel_range_restart_ns, self.w2_panel_range_restart_ns)
            if value is not None
        )
        if min(values) < 0.0:
            raise ValueError("runtime overheads must be non-negative")
        by_width = tuple(
            sorted(
                (int(width), float(expert_fixed), float(route))
                for width, expert_fixed, route in self.by_width
            )
        )
        if any(width <= 0 or min(expert_fixed, route) < 0.0 for width, expert_fixed, route in by_width):
            raise ValueError("width-specific runtime overheads require positive widths and non-negative values")
        if len({width for width, _, _ in by_width}) != len(by_width):
            raise ValueError("width-specific runtime overhead widths must be unique")
        object.__setattr__(self, "by_width", by_width)

    def expert_overhead_ns(self, routes: int, threads: int) -> float:
        expert_fixed_ns = self.expert_fixed_ns
        route_ns = self.route_ns
        for width, width_expert_fixed_ns, width_route_ns in self.by_width:
            if width == threads:
                expert_fixed_ns = width_expert_fixed_ns
                route_ns = width_route_ns
                break
        return expert_fixed_ns + routes * route_ns

    def panel_range_restart_for_stage(self, stage: str) -> float:
        if stage == "w13":
            value = self.w13_panel_range_restart_ns
        elif stage == "w2":
            value = self.w2_panel_range_restart_ns
        else:
            raise ValueError(f"unsupported stage {stage!r}")
        return self.panel_range_restart_ns if value is None else value

    @classmethod
    def from_dict(cls, payload: dict) -> "RuntimeOverheads":
        return cls(
            call_setup_ns=float(payload.get("call_setup_ns", 0.0)),
            expert_fixed_ns=float(payload.get("expert_fixed_ns", 0.0)),
            route_ns=float(payload.get("route_ns", 0.0)),
            stage_fixed_ns=float(payload.get("stage_fixed_ns", 0.0)),
            range_fixed_ns=float(payload.get("range_fixed_ns", 0.0)),
            panel_range_restart_ns=float(payload.get("panel_range_restart_ns", 0.0)),
            w13_panel_range_restart_ns=(
                float(payload["w13_panel_range_restart_ns"])
                if payload.get("w13_panel_range_restart_ns") is not None
                else None
            ),
            w2_panel_range_restart_ns=(
                float(payload["w2_panel_range_restart_ns"])
                if payload.get("w2_panel_range_restart_ns") is not None
                else None
            ),
            by_width=tuple(
                (
                    int(point["threads"]),
                    float(point.get("expert_fixed_ns", 0.0)),
                    float(point.get("route_ns", 0.0)),
                )
                for point in payload.get("by_width", ())
            ),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "call_setup_ns": self.call_setup_ns,
            "expert_fixed_ns": self.expert_fixed_ns,
            "route_ns": self.route_ns,
            "stage_fixed_ns": self.stage_fixed_ns,
            "range_fixed_ns": self.range_fixed_ns,
            "panel_range_restart_ns": self.panel_range_restart_ns,
            "w13_panel_range_restart_ns": self.w13_panel_range_restart_ns,
            "w2_panel_range_restart_ns": self.w2_panel_range_restart_ns,
        }
        if self.by_width:
            payload["by_width"] = [
                {
                    "threads": width,
                    "expert_fixed_ns": expert_fixed_ns,
                    "route_ns": route_ns,
                }
                for width, expert_fixed_ns, route_ns in self.by_width
            ]
        return payload


@dataclass(frozen=True)
class WideTeamPressureCalibration:
    """Residual full-cohort dilation indexed by fixed team width."""

    isolated_dilation: tuple[tuple[int, float], ...] = ()
    full_cohort_dilation: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        isolated = tuple(sorted((int(width), float(scale)) for width, scale in self.isolated_dilation))
        full = tuple(sorted((int(width), float(scale)) for width, scale in self.full_cohort_dilation))
        for name, points in (("isolated", isolated), ("full-cohort", full)):
            if any(width <= 0 or scale < 1.0 for width, scale in points):
                raise ValueError(
                    f"wide-team {name} widths must be positive and dilation must be at least one"
                )
            if len({width for width, _ in points}) != len(points):
                raise ValueError(f"wide-team {name} widths must be unique")
        isolated_by_width = dict(isolated)
        if any(scale < isolated_by_width.get(width, 1.0) for width, scale in full):
            raise ValueError("full-cohort dilation cannot be below isolated dilation")
        object.__setattr__(self, "isolated_dilation", isolated)
        object.__setattr__(self, "full_cohort_dilation", full)

    @staticmethod
    def _scale(points: tuple[tuple[int, float], ...], team_width: int) -> float:
        if team_width <= 0:
            raise ValueError("team_width must be positive")
        return dict(points).get(team_width, 1.0)

    def isolated_scale(self, team_width: int) -> float:
        return self._scale(self.isolated_dilation, team_width)

    def full_cohort_scale(self, team_width: int) -> float:
        return max(
            self._scale(self.full_cohort_dilation, team_width),
            self.isolated_scale(team_width),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "WideTeamPressureCalibration":
        return cls(
            isolated_dilation=tuple(
                (int(point["threads"]), float(point["dilation"]))
                for point in payload.get("isolated_dilation", ())
            ),
            full_cohort_dilation=tuple(
                (int(point["threads"]), float(point["dilation"]))
                for point in payload.get("full_cohort_dilation", ())
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "isolated_dilation": [
                {"threads": width, "dilation": scale}
                for width, scale in self.isolated_dilation
            ],
            "full_cohort_dilation": [
                {"threads": width, "dilation": scale}
                for width, scale in self.full_cohort_dilation
            ]
        }


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
    rank_cpu_ids: tuple[int, ...] = ()
    llc_domains: tuple[LlcDomainCalibration, ...] = ()
    dram_scope: str = "numa_rank"
    wide_team_pressure: WideTeamPressureCalibration = WideTeamPressureCalibration()

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
        if self.dram_scope != "numa_rank":
            raise ValueError("only NUMA-rank shared DRAM service is supported")
        rank_cpu_ids = tuple(int(cpu) for cpu in self.rank_cpu_ids)
        if rank_cpu_ids and (
            len(rank_cpu_ids) != self.cores_per_rank
            or min(rank_cpu_ids) < 0
            or len(set(rank_cpu_ids)) != len(rank_cpu_ids)
        ):
            raise ValueError("rank_cpu_ids must contain one unique non-negative id per rank core")
        if self.llc_domains:
            domain_ids = [domain.domain_id for domain in self.llc_domains]
            domain_cpus = [cpu for domain in self.llc_domains for cpu in domain.cpu_ids]
            if len(set(domain_ids)) != len(domain_ids):
                raise ValueError("LLC domain ids must be unique")
            if len(set(domain_cpus)) != len(domain_cpus):
                raise ValueError("LLC domain CPU sets must be disjoint")
            if not rank_cpu_ids:
                rank_cpu_ids = tuple(domain_cpus)
            if set(domain_cpus) != set(rank_cpu_ids):
                raise ValueError("LLC domains must partition rank_cpu_ids")
            if sum(domain.capacity_bytes for domain in self.llc_domains) != self.caches.llc_bytes_per_rank:
                raise ValueError("LLC domain capacities must sum to llc_bytes_per_rank")
        object.__setattr__(self, "supported_widths", widths)
        object.__setattr__(self, "rank_cpu_ids", rank_cpu_ids)

    def llc_domain_thread_counts(self, active_cpu_ids: Iterable[int]) -> dict[str, int]:
        if not self.llc_domains:
            raise ValueError("calibration does not describe LLC domains")
        requested = tuple(int(cpu) for cpu in active_cpu_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("active_cpu_ids must be non-empty and unique")
        cpu_to_domain = {
            cpu: domain.domain_id
            for domain in self.llc_domains
            for cpu in domain.cpu_ids
        }
        unknown = sorted(set(requested) - cpu_to_domain.keys())
        if unknown:
            raise ValueError(f"active CPUs are outside the calibrated rank: {unknown}")
        counts = {domain.domain_id: 0 for domain in self.llc_domains}
        for cpu in requested:
            counts[cpu_to_domain[cpu]] += 1
        return counts

    def llc_capacity_bytes(
        self,
        *,
        active_cpu_ids: Iterable[int] | None = None,
        llc_domain_threads: Mapping[str, int] | None = None,
    ) -> int:
        if not self.llc_domains or (active_cpu_ids is None and llc_domain_threads is None):
            return self.caches.llc_bytes_per_rank
        counts = self._resolve_llc_domain_threads(
            active_cpu_ids=active_cpu_ids,
            llc_domain_threads=llc_domain_threads,
        )
        return sum(
            domain.capacity_bytes
            for domain in self.llc_domains
            if counts[domain.domain_id] > 0
        )

    def _resolve_llc_domain_threads(
        self,
        *,
        active_cpu_ids: Iterable[int] | None,
        llc_domain_threads: Mapping[str, int] | None,
    ) -> dict[str, int]:
        if active_cpu_ids is not None and llc_domain_threads is not None:
            raise ValueError("provide active_cpu_ids or llc_domain_threads, not both")
        if active_cpu_ids is not None:
            return self.llc_domain_thread_counts(active_cpu_ids)
        if llc_domain_threads is None:
            raise ValueError("LLC domain placement is missing")
        provided = {str(domain_id): int(count) for domain_id, count in llc_domain_threads.items()}
        known = {domain.domain_id: len(domain.cpu_ids) for domain in self.llc_domains}
        unknown = sorted(set(provided) - known.keys())
        if unknown:
            raise ValueError(f"unknown LLC domains: {unknown}")
        counts = {domain_id: provided.get(domain_id, 0) for domain_id in known}
        if any(count < 0 or count > known[domain_id] for domain_id, count in counts.items()):
            raise ValueError("LLC domain thread counts must fit each domain")
        if sum(counts.values()) <= 0:
            raise ValueError("at least one LLC-domain thread must be active")
        return counts

    def service_rate(
        self,
        resource: str,
        active_threads: int,
        *,
        active_cpu_ids: Iterable[int] | None = None,
        llc_domain_threads: Mapping[str, int] | None = None,
    ) -> float:
        if active_threads <= 0:
            raise ValueError("active_threads must be positive")
        placement_counts = None
        if active_cpu_ids is not None:
            active_cpu_ids = tuple(int(cpu) for cpu in active_cpu_ids)
            if len(active_cpu_ids) != active_threads:
                raise ValueError("active_cpu_ids count must match active_threads")
        if active_cpu_ids is not None or llc_domain_threads is not None:
            if not self.llc_domains:
                raise ValueError("placement-aware service requires calibrated LLC domains")
            placement_counts = self._resolve_llc_domain_threads(
                active_cpu_ids=active_cpu_ids,
                llc_domain_threads=llc_domain_threads,
            )
            if sum(placement_counts.values()) != active_threads:
                raise ValueError("LLC domain thread counts must sum to active_threads")
        curve = getattr(self, resource)
        if curve is None:
            return math.inf
        capped_threads = min(active_threads, self.cores_per_rank)
        if resource != "llc_bytes" or placement_counts is None:
            return curve.rate(capped_threads)
        domain_rate = sum(
            domain.service.rate(placement_counts[domain.domain_id])
            for domain in self.llc_domains
            if placement_counts[domain.domain_id] > 0
        )
        active_domains = sum(count > 0 for count in placement_counts.values())
        if active_domains == 1:
            return domain_rate
        # Domain curves already model low-width injection. Multiple domains
        # only share the measured rank-level fabric ceiling; prefix points of
        # the rank curve may still describe a single-domain placement.
        return min(domain_rate, curve.saturated_rate)

    @classmethod
    def from_dict(cls, payload: dict) -> "AnalyticMachineCalibration":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version not in SUPPORTED_ANALYTIC_MACHINE_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported analytic machine schema {payload.get('schema_version')!r}; "
                f"expected one of {sorted(SUPPORTED_ANALYTIC_MACHINE_SCHEMA_VERSIONS)}"
            )
        if payload.get("kind") != "moe_analytic_machine":
            raise ValueError("analytic calibration kind must be 'moe_analytic_machine'")
        machine = payload["machine"]
        services = payload["services"]
        stage_scales = payload.get("stage_scales", {})
        optional_frontend = services.get("frontend_instructions")
        optional_epilogue = services.get("epilogue_elements")
        topology = payload.get("topology", {}) if schema_version >= 2 else {}
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
            rank_cpu_ids=tuple(int(cpu) for cpu in topology.get("rank_cpu_ids", ())),
            llc_domains=tuple(
                LlcDomainCalibration.from_dict(domain)
                for domain in topology.get("llc_domains", ())
            ),
            dram_scope=str(topology.get("dram_scope", "numa_rank")),
            wide_team_pressure=WideTeamPressureCalibration.from_dict(
                payload.get("planner", {}).get("wide_team_pressure", {})
            ),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "AnalyticMachineCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        def service(curve: SaturatingServiceCurve | None) -> dict | None:
            return curve.to_dict() if curve is not None else None

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
        payload = {
            "schema_version": ANALYTIC_MACHINE_SCHEMA_VERSION,
            "kind": "moe_analytic_machine",
            "machine": {
                "id": self.machine_id,
                "cores_per_rank": self.cores_per_rank,
            },
            "kernel": {"backend_n_tile": self.backend_n_tile},
            "caches": asdict(self.caches),
            "services": services,
            "overheads": self.overheads.to_dict(),
            "planner": {"supported_widths": list(self.supported_widths)},
            "uncertainty": {"relative": self.relative_uncertainty},
            "stage_scales": {"w13": self.w13_scale, "w2": self.w2_scale},
        }
        if self.rank_cpu_ids or self.llc_domains:
            payload["topology"] = {
                "rank_cpu_ids": list(self.rank_cpu_ids),
                "llc_domains": [domain.to_dict() for domain in self.llc_domains],
                "dram_scope": self.dram_scope,
            }
        if (
            self.wide_team_pressure.isolated_dilation
            or self.wide_team_pressure.full_cohort_dilation
        ):
            payload["planner"]["wide_team_pressure"] = self.wide_team_pressure.to_dict()
        return payload


@dataclass(frozen=True)
class AnalyticPolicy:
    """Planner-visible machine and kernel domain identity."""

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
    measurement_experts: int
    cores_per_rank: int
    concurrent_ranks: int
    llc_bytes_per_rank: int
    llc_topology_signature: tuple[tuple[tuple[int, ...], int], ...]

    def identity_key(self) -> tuple[object, ...]:
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
            self.llc_topology_signature,
        )

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
class AnalyticStripeDemand:
    """Demand for one worker stripe of a full-N stage.

    ``balanced_work_fraction`` scales the mapper's aggregate busiest-lane
    upper bound when N tiles do not divide the team evenly.
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
    stage_bytes: int
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
    stripe_demand: AnalyticStripeDemand | None


@dataclass(frozen=True)
class AnalyticWindowDemand:
    """Physical demand of one exact runtime stage window.

    Byte counts are for one expert. Timing fields describe one homogeneous
    cohort wave, so aggregate cache pressure is visible without turning the
    selector into a route/thread latency table.
    """

    window_index: int
    begin_tile: int
    n_tiles: int
    active_threads: int
    balanced_tiles: int
    owner_tiles: int
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
    aggregate_working_set_bytes: float
    serialized_core_ns: float
    transfer_ns: float
    ecm_ns: float


@dataclass(frozen=True)
class AnalyticStageWindowScore:
    """Interpretable objective for one ``(stage, M, T, window_tiles)`` point."""

    stage: str
    routes: int
    threads: int
    cohort_threads: int
    cohort_tasks: int
    requested_window_tiles: int
    window_tiles: int
    full_stripe_window_tiles: int
    range_tiles: int
    windows: int
    owner_window_bytes: int
    objective_ns: float
    ecm_ns: float
    serialized_core_ns: float
    transfer_ns: float
    window_overhead_ns: float
    panel_range_restart_ns: float
    range_restart_ns: float
    l2_ns: float
    llc_ns: float
    dram_ns: float
    l2_miss_fraction_a: float
    l2_miss_fraction_b: float
    a_l2_refill_bytes: float
    b_l2_refill_bytes: float
    l2_bytes: float
    llc_bytes: float
    compulsory_dram_bytes: float
    spillable_dram_bytes: float
    max_llc_working_set_bytes: float
    max_aggregate_working_set_bytes: float
    starves_any_thread: bool
    window_demands: tuple[AnalyticWindowDemand, ...]

    @property
    def is_full_stripe(self) -> bool:
        return self.window_tiles == self.full_stripe_window_tiles


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
        if self.kind not in {"operator", "stage_setup", "cold_b", "steady_b"}:
            raise ValueError(f"unsupported analytical phase kind {self.kind!r}")
        if self.panel_count < 0:
            raise ValueError("panel_count must be non-negative")
        if self.kind in {"operator", "stage_setup"} and self.panel_count != 0:
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
        return dict(zip(_SHARED_RESOURCES, self.resource_time_vector(spill_fraction)))

    def resource_vectors(self, spill_fraction: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return fixed-order demand/time vectors for contention simulation."""
        dram_bytes = self.dram_bytes(spill_fraction)
        return (
            (
                self.matrix_flops,
                0.0,
                0.0,
                0.0,
                self.l2_bytes,
                self.llc_bytes,
                dram_bytes,
                self.epilogue_elements,
            ),
            (
                self.gemm_core_ns,
                self.matrix_ns,
                self.frontend_ns,
                self.l1_ns,
                self.l2_ns,
                self.llc_ns,
                dram_bytes / self.dram_rate * 1e9,
                self.epilogue_ns,
            ),
        )

    def resource_time_vector(self, spill_fraction: float) -> tuple[float, ...]:
        return self.resource_vectors(spill_fraction)[1]

    def _duration_from_resource_times(
        self,
        times: Sequence[float],
        resource_scales: Mapping[str, float] | None = None,
    ) -> float:
        scales = resource_scales or {}
        gemm_core_ns = times[0] * scales.get("gemm_core_flops", 1.0)
        transfer_ns = max(
            times[4] * scales.get("l2_bytes", 1.0),
            times[5] * scales.get("llc_bytes", 1.0),
            times[6] * scales.get("dram_bytes", 1.0),
        )
        body_ns = max(gemm_core_ns, transfer_ns)
        epilogue_ns = times[7] * scales.get("epilogue_elements", 1.0)
        return self.residual_scale * (self.fixed_ns + body_ns + epilogue_ns)

    def duration_ns(
        self,
        *,
        spill_fraction: float | None = None,
        resource_scales: Mapping[str, float] | None = None,
    ) -> float:
        if spill_fraction is None:
            spill_fraction = self.isolated_spill_fraction
        times = self.resource_time_vector(spill_fraction)
        # The calibrated load probes measure endpoint-to-register service:
        # LLC already includes LLC->L2->L1 and DRAM includes the whole path.
        # Their lower bounds overlap and therefore compose with max, not sum.
        # The L1-hot M12 GEMM peak already includes BFMMLA, frontend, and L1
        # load delivery. Only lower hierarchy endpoint bounds remain.
        return self._duration_from_resource_times(times, resource_scales)

    @cached_property
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
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.local_experts = int(local_experts)
        self.down_output_element_bytes = int(down_output_element_bytes)
        self._exact_m = bool(exact_m)
        self._mapper = SveBf16KernelProfile(n_tile=resolved_n_tile, exact_m=exact_m)
        self._w13_geometry = full_stage_geometry(
            k=self.hidden_size,
            n=2 * self.intermediate_size,
            n_tile=resolved_n_tile,
        )
        self._w2_geometry = full_stage_geometry(
            k=self.intermediate_size,
            n=self.hidden_size,
            n_tile=resolved_n_tile,
        )
        self.w13_stage_bytes = self._w13_geometry.stage_bytes
        self.w2_stage_bytes = self._w2_geometry.stage_bytes
        self.w2_bytes = self.w2_stage_bytes
        self.max_stage_bytes = max(self.w13_stage_bytes, self.w2_stage_bytes)
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
            measurement_experts=0,
            cores_per_rank=self.calibration.cores_per_rank,
            concurrent_ranks=int(concurrent_ranks),
            llc_bytes_per_rank=self.calibration.caches.llc_bytes_per_rank,
            llc_topology_signature=tuple(
                (domain.cpu_ids, domain.capacity_bytes)
                for domain in self.calibration.llc_domains
            ),
        )
        self._t_iso_scalar_cache: dict[tuple[int, int], float] = {}

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

    def task_max_stage_bytes(self, routes: int, threads: int) -> int:
        del routes, threads
        return self.max_stage_bytes

    def window_bytes_per_worker(self, threads: int, routes: int | None = None) -> int:
        del routes
        return max(
            self._w13_geometry.bytes_per_worker(threads),
            self._w2_geometry.bytes_per_worker(threads),
        )

    def native_quick_planner_payload(self) -> dict[str, object]:
        """Export immutable metadata used around native homogeneous LPT."""
        return {
            "max_stage_bytes": self.max_stage_bytes,
            "relative_error": self.relative_error,
            "profile_runs": self.profile_runs,
            "window_bytes_by_width": [
                (width, self.window_bytes_per_worker(width))
                for width in self.supported_widths
            ],
        }

    def stage_bytes_per_worker(self, stage: str, threads: int, routes: int | None = None) -> int:
        del routes
        if stage == "w13":
            geometry = self._w13_geometry
        elif stage == "w2":
            geometry = self._w2_geometry
        else:
            raise ValueError(f"stage must be 'w13' or 'w2', got {stage!r}")
        return geometry.bytes_per_worker(threads)

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

    def _llc_miss_fraction(
        self,
        working_set_bytes: float,
        *,
        active_cpu_ids: Iterable[int] | None = None,
        llc_domain_threads: Mapping[str, int] | None = None,
    ) -> float:
        cache = self.calibration.caches
        physical_capacity = self.calibration.llc_capacity_bytes(
            active_cpu_ids=active_cpu_ids,
            llc_domain_threads=llc_domain_threads,
        )
        return _smooth_capacity_miss(
            working_set_bytes,
            physical_capacity * cache.llc_effective_fraction,
            physical_capacity,
        )

    def _stage_mapping_and_geometry(
        self,
        stage: str,
        routes: int,
        threads: int,
    ) -> tuple[SveBf16KernelExecution, FullStageGeometry]:
        work = fused_expert_work(
            routes,
            self.hidden_size,
            self.intermediate_size,
            down_output_element_bytes=self.down_output_element_bytes,
        )
        if stage == "w13":
            logical = work.w13
            geometry = self._w13_geometry
        elif stage == "w2":
            logical = work.w2
            geometry = self._w2_geometry
        else:
            raise ValueError(f"unsupported stage {stage!r}")
        mapping = self._mapper.lower(logical, ExecutionSchedule(threads=threads))
        return mapping, geometry

    @lru_cache(maxsize=16384)
    def score_stage_window(
        self,
        stage: str,
        routes: int,
        threads: int,
        window_tiles: int,
        cohort_threads: int | None = None,
    ) -> AnalyticStageWindowScore:
        """Score the exact tile-window geometry executed by the native runtime.

        ``window_tiles=0`` is the full-stripe endpoint. The demand fields are
        per expert, while timing uses a homogeneous cohort occupying
        ``cohort_threads``. This keeps cache-capacity effects explicit: private
        L2 retention is per owner, and LLC spill is evaluated on the aggregate
        simultaneous footprint. Candidate-independent compute and compulsory
        B traffic remain in the report, but the selector uses a serialized
        incremental objective so hidden refill reductions are not assigned
        zero value merely because GEMM compute is the ECM maximum.
        """
        stage = str(stage)
        routes = int(routes)
        threads = int(threads)
        requested_window_tiles = int(window_tiles)
        if stage not in {"w13", "w2"}:
            raise ValueError(f"unsupported stage {stage!r}")
        if routes <= 0 or threads <= 0 or requested_window_tiles < 0:
            raise ValueError("routes and threads must be positive and window_tiles non-negative")
        if threads not in self.supported_widths:
            raise KeyError(f"unsupported analytical thread width {threads}")

        requested_cohort_threads = threads if cohort_threads is None else int(cohort_threads)
        if requested_cohort_threads < threads:
            raise ValueError("cohort_threads cannot be smaller than the task width")
        if requested_cohort_threads > self.calibration.cores_per_rank:
            raise ValueError("cohort_threads cannot exceed cores_per_rank")
        cohort_tasks = requested_cohort_threads // threads
        if cohort_tasks <= 0:
            raise ValueError("cohort_threads must contain at least one complete task team")
        used_cohort_threads = cohort_tasks * threads

        mapping, geometry = self._stage_mapping_and_geometry(stage, routes, threads)
        plan: StageWindowPlan
        if requested_window_tiles == 0:
            plan = geometry.full_stripe_plan(threads)
        else:
            plan = geometry.window_plan(threads, requested_window_tiles)
        if plan.total_tiles != mapping.allocation.total_tiles:
            raise RuntimeError("stage-window geometry and SVE kernel mapping disagree on N tiles")

        logical = mapping.logical_work
        panels = len(mapping.panels)
        if panels <= 0:
            raise RuntimeError("positive-route stage must contain at least one M panel")
        a_bytes = mapping.compute_rows * logical.k * logical.input_element_bytes
        a_panel_bytes = max(panel.compute_rows for panel in mapping.panels) * logical.k * logical.input_element_bytes
        tile_bytes = logical.k * mapping.n_tile * logical.weight_element_bytes
        if tile_bytes != plan.bytes_per_tile:
            raise RuntimeError("stage-window geometry and SVE kernel mapping disagree on tile bytes")
        total_tiles = plan.total_tiles
        output_columns_per_tile = logical.output_columns * mapping.n_tile // logical.n
        total_store_rows = sum(panel.store_rows for panel in mapping.panels)
        available_b_reuses = panels - 1
        cache_turnover_reuses = max(
            int(self.calibration.caches.effective_l2_bytes_per_core // a_panel_bytes),
            1,
        )
        transient_b_reuses = min(available_b_reuses, cache_turnover_reuses)
        steady_b_reuses = available_b_reuses - transient_b_reuses
        # A must survive the complete B owner window between successive N
        # windows.  Use the calibrated usable L2 capacity here, not the nominal
        # cache size: replacement, prefetch, and the in-flight A panel consume
        # the headroom that the effective-capacity calibration represents.
        a_residency_capacity = max(
            self.calibration.caches.effective_l2_bytes_per_core - a_panel_bytes,
            0,
        )

        seen_active_threads = 0
        windows: list[AnalyticWindowDemand] = []
        for window_index in range(plan.windows):
            begin_tile, n_tiles = plan.window_tile_span(window_index)
            active_threads = min(n_tiles, threads)
            owner_tiles = math.ceil(n_tiles / threads)
            balanced_tiles = owner_tiles * active_threads
            owner_window_bytes = owner_tiles * tile_bytes
            weight_bytes = n_tiles * tile_bytes
            c_write_bytes = mapping.llc_c_write_bytes * n_tiles / total_tiles

            a_residency_footprint = owner_window_bytes + a_bytes
            a_l2_miss = _smooth_capacity_miss(
                a_residency_footprint,
                a_residency_capacity,
                max(self.calibration.caches.l2_bytes_per_core - a_panel_bytes, 0),
            )
            b_reuse_footprint = owner_window_bytes + a_panel_bytes
            b_l2_miss = self._l2_b_reuse_miss_fraction(b_reuse_footprint)
            b_l2_steady_miss = self._l2_steady_scan_miss_fraction(b_reuse_footprint)
            b_l2_refill = weight_bytes * (
                1.0
                + transient_b_reuses * b_l2_miss
                + steady_b_reuses * b_l2_steady_miss
            )

            new_threads = max(active_threads - seen_active_threads, 0)
            reused_threads = active_threads - new_threads
            a_l2_refill = a_bytes * (new_threads + reused_threads * a_l2_miss)
            seen_active_threads = max(seen_active_threads, active_threads)

            l2_bytes = a_l2_refill + b_l2_refill + c_write_bytes
            reusable_b_bytes = weight_bytes if panels > 1 else 0
            spillable_dram_bytes = (
                a_l2_refill + max(b_l2_refill - weight_bytes, 0.0) + c_write_bytes
            )
            llc_working_set_bytes = reusable_b_bytes + a_bytes + c_write_bytes
            aggregate_working_set_bytes = llc_working_set_bytes * cohort_tasks
            aggregate_active_threads = active_threads * cohort_tasks

            bfmmla_instructions = mapping.compute_rows * logical.k // 2 * balanced_tiles
            matrix_flops = bfmmla_instructions * (2 * mapping.vector_bytes)
            epilogue_elements = total_store_rows * output_columns_per_tile * balanced_tiles
            gemm_core_ns = (
                matrix_flops
                * cohort_tasks
                / self.calibration.service_rate("gemm_core_flops", aggregate_active_threads)
                * 1e9
            )
            epilogue_ns = 0.0
            if self.calibration.epilogue_elements is not None:
                epilogue_ns = (
                    epilogue_elements
                    * cohort_tasks
                    / self.calibration.service_rate("epilogue_elements", aggregate_active_threads)
                    * 1e9
                )
            l2_ns = (
                l2_bytes
                * cohort_tasks
                / self.calibration.service_rate("l2_bytes", aggregate_active_threads)
                * 1e9
            )
            llc_ns = (
                l2_bytes
                * cohort_tasks
                / self.calibration.service_rate("llc_bytes", aggregate_active_threads)
                * 1e9
            )
            llc_spill_fraction = self._llc_miss_fraction(aggregate_working_set_bytes)
            aggregate_dram_bytes = (
                weight_bytes + llc_spill_fraction * spillable_dram_bytes
            ) * cohort_tasks
            dram_ns = (
                aggregate_dram_bytes
                / self.calibration.service_rate("dram_bytes", aggregate_active_threads)
                * 1e9
            )
            transfer_ns = max(l2_ns, llc_ns, dram_ns)
            serialized_core_ns = gemm_core_ns + epilogue_ns
            windows.append(
                AnalyticWindowDemand(
                    window_index=window_index,
                    begin_tile=begin_tile,
                    n_tiles=n_tiles,
                    active_threads=active_threads,
                    balanced_tiles=balanced_tiles,
                    owner_tiles=owner_tiles,
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
                    spillable_dram_bytes=spillable_dram_bytes,
                    reusable_b_bytes=reusable_b_bytes,
                    llc_working_set_bytes=llc_working_set_bytes,
                    aggregate_working_set_bytes=aggregate_working_set_bytes,
                    serialized_core_ns=serialized_core_ns,
                    transfer_ns=transfer_ns,
                    ecm_ns=max(gemm_core_ns, transfer_ns) + epilogue_ns,
                )
            )

        compulsory_dram_bytes = sum(window.compulsory_dram_bytes for window in windows)
        active_scans = sum(window.active_threads for window in windows)
        panel_range_restart_ns = self.calibration.overheads.panel_range_restart_for_stage(stage)
        range_restart_ns = panels * max(plan.windows - 1, 0) * panel_range_restart_ns
        window_overhead_ns = (
            self.calibration.overheads.stage_fixed_ns
            + plan.windows * self.calibration.overheads.range_fixed_ns
            + range_restart_ns
        )
        serialized_core_ns = sum(window.serialized_core_ns for window in windows)
        transfer_ns = sum(window.transfer_ns for window in windows)
        return AnalyticStageWindowScore(
            stage=stage,
            routes=routes,
            threads=threads,
            cohort_threads=used_cohort_threads,
            cohort_tasks=cohort_tasks,
            requested_window_tiles=requested_window_tiles,
            window_tiles=plan.window_tiles,
            full_stripe_window_tiles=geometry.tiles_per_worker(threads),
            range_tiles=plan.range_tiles,
            windows=plan.windows,
            owner_window_bytes=max(window.owner_window_bytes for window in windows),
            objective_ns=serialized_core_ns + transfer_ns + window_overhead_ns,
            ecm_ns=sum(window.ecm_ns for window in windows) + window_overhead_ns,
            serialized_core_ns=serialized_core_ns,
            transfer_ns=transfer_ns,
            window_overhead_ns=window_overhead_ns,
            panel_range_restart_ns=panel_range_restart_ns,
            range_restart_ns=range_restart_ns,
            l2_ns=sum(
                window.l2_bytes
                * cohort_tasks
                / self.calibration.service_rate(
                    "l2_bytes", window.active_threads * cohort_tasks
                )
                * 1e9
                for window in windows
            ),
            llc_ns=sum(
                window.llc_bytes
                * cohort_tasks
                / self.calibration.service_rate(
                    "llc_bytes", window.active_threads * cohort_tasks
                )
                * 1e9
                for window in windows
            ),
            dram_ns=sum(
                (
                    window.compulsory_dram_bytes
                    + self._llc_miss_fraction(window.aggregate_working_set_bytes)
                    * window.spillable_dram_bytes
                )
                * cohort_tasks
                / self.calibration.service_rate(
                    "dram_bytes", window.active_threads * cohort_tasks
                )
                * 1e9
                for window in windows
            ),
            l2_miss_fraction_a=(
                sum(window.l2_miss_fraction_a * window.active_threads for window in windows)
                / active_scans
            ),
            l2_miss_fraction_b=(
                sum(
                    window.l2_miss_fraction_b * window.compulsory_dram_bytes
                    for window in windows
                )
                / compulsory_dram_bytes
            ),
            a_l2_refill_bytes=sum(window.a_l2_refill_bytes for window in windows),
            b_l2_refill_bytes=sum(window.b_l2_refill_bytes for window in windows),
            l2_bytes=sum(window.l2_bytes for window in windows),
            llc_bytes=sum(window.llc_bytes for window in windows),
            compulsory_dram_bytes=compulsory_dram_bytes,
            spillable_dram_bytes=sum(window.spillable_dram_bytes for window in windows),
            max_llc_working_set_bytes=max(window.llc_working_set_bytes for window in windows),
            max_aggregate_working_set_bytes=max(
                window.aggregate_working_set_bytes for window in windows
            ),
            starves_any_thread=plan.starves_any_thread(),
            window_demands=tuple(windows),
        )

    def shadow_stage_window_policy(self, *, cohort_threads: int | None = None):
        """Build the analytical selector without binding it to production."""
        try:
            from analytic_stage_window_policy import AnalyticStageWindowPolicy
        except ImportError:  # pragma: no cover - package-style import
            from .analytic_stage_window_policy import AnalyticStageWindowPolicy
        return AnalyticStageWindowPolicy(self, cohort_threads=cohort_threads)

    def _stage_demand(
        self,
        stage: str,
        mapping: SveBf16KernelExecution,
        geometry: FullStageGeometry,
    ) -> AnalyticStageDemand:
        if stage not in {"w13", "w2"}:
            raise ValueError(f"unsupported stage {stage!r}")
        if mapping.schedule.sequential_n_ranges != 1:
            raise ValueError("the production analytical model requires one full-N stage")
        panels = len(mapping.panels)
        if panels == 0:
            return AnalyticStageDemand(
                stage=stage,
                mapping=mapping,
                stage_bytes=0,
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
                stripe_demand=None,
            )

        logical = mapping.logical_work
        a_bytes = mapping.compute_rows * logical.k * logical.input_element_bytes
        max_panel_rows = max(panel.compute_rows for panel in mapping.panels)
        a_panel_bytes = max_panel_rows * logical.k * logical.input_element_bytes
        total_tiles = mapping.allocation.total_tiles
        tile_bytes = logical.k * mapping.n_tile * logical.weight_element_bytes
        active_threads = min(total_tiles, mapping.schedule.threads)
        owner_tiles = math.ceil(total_tiles / mapping.schedule.threads)
        balanced_tiles = owner_tiles * active_threads
        owner_window_bytes = owner_tiles * tile_bytes
        weight_bytes = total_tiles * tile_bytes
        c_write_bytes = mapping.llc_c_write_bytes

        # Team width is the only cache-window control: each owner repeatedly
        # scans its full-N packed-B stripe while consuming successive A panels.
        a_residency_footprint = owner_window_bytes + a_bytes
        a_residency_capacity = max(
            self.calibration.caches.l2_bytes_per_core - a_panel_bytes,
            0,
        )
        a_l2_miss = float(a_residency_footprint > a_residency_capacity)
        b_reuse_footprint = owner_window_bytes + a_panel_bytes
        b_l2_miss = self._l2_b_reuse_miss_fraction(b_reuse_footprint)
        available_b_reuses = panels - 1
        cache_turnover_reuses = max(
            int(self.calibration.caches.effective_l2_bytes_per_core // a_panel_bytes),
            1,
        )
        transient_b_reuses = min(available_b_reuses, cache_turnover_reuses)
        steady_b_reuses = available_b_reuses - transient_b_reuses
        b_l2_steady_miss = self._l2_steady_scan_miss_fraction(b_reuse_footprint)
        b_l2_refill = weight_bytes * (
            1.0 + transient_b_reuses * b_l2_miss + steady_b_reuses * b_l2_steady_miss
        )
        a_l2_refill = a_bytes * active_threads
        l2_refill_bytes = a_l2_refill + b_l2_refill
        l2_bytes = l2_refill_bytes + c_write_bytes
        reusable_b_bytes = weight_bytes if panels > 1 else 0
        spillable_dram = a_l2_refill + max(b_l2_refill - weight_bytes, 0.0) + c_write_bytes
        llc_working_set = reusable_b_bytes + a_bytes + c_write_bytes
        stripe = AnalyticStripeDemand(
            n_tiles=total_tiles,
            balanced_work_fraction=balanced_tiles
            / (mapping.allocation.busiest_thread_tiles * mapping.allocation.active_threads),
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
            llc_working_set_bytes=llc_working_set,
        )
        return AnalyticStageDemand(
            stage=stage,
            mapping=mapping,
            stage_bytes=weight_bytes,
            owner_window_bytes=owner_window_bytes,
            reusable_b_bytes=reusable_b_bytes,
            l2_miss_fraction_b=b_l2_miss,
            l2_miss_fraction_a=a_l2_miss,
            a_l2_refill_bytes=a_l2_refill,
            b_l2_refill_bytes=b_l2_refill,
            l2_refill_bytes=l2_refill_bytes,
            c_write_bytes=c_write_bytes,
            l2_bytes=l2_bytes,
            llc_bytes=l2_bytes,
            compulsory_dram_bytes=weight_bytes,
            spillable_dram_bytes=spillable_dram,
            llc_working_set_bytes=llc_working_set,
            stripe_demand=stripe,
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
                    name=f"{demand.stage}:{phase_kind}",
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

        stripe = demand.stripe_demand
        if stripe is None:
            return ()
        active_threads = min(stripe.n_tiles, mapping.schedule.threads)
        balanced_tiles = math.ceil(stripe.n_tiles / mapping.schedule.threads) * active_threads
        if machine.overheads.stage_fixed_ns > 0.0:
            phases.append(
                AnalyticPhase(
                    name=f"{demand.stage}:setup",
                    kind="stage_setup",
                    panel_count=0,
                    active_threads=active_threads,
                    fixed_ns=machine.overheads.stage_fixed_ns,
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
        cold_a_l2_bytes = stripe.a_l2_refill_bytes * cold_compute_fraction
        cold_c_write_bytes = stripe.c_write_bytes * cold_store_fraction
        cold_b_l2_bytes = stripe.compulsory_dram_bytes
        append_phase(
            phase_kind="cold_b",
            panels=cold_panels,
            active_threads=active_threads,
            balanced_tiles=balanced_tiles,
            a_l2_bytes=cold_a_l2_bytes,
            b_l2_bytes=cold_b_l2_bytes,
            c_write_bytes=cold_c_write_bytes,
            compulsory_dram_bytes=stripe.compulsory_dram_bytes,
            spillable_dram_bytes=cold_a_l2_bytes + cold_c_write_bytes,
            working_set_bytes=stripe.llc_working_set_bytes,
        )
        if steady_panels:
            append_phase(
                phase_kind="steady_b",
                panels=steady_panels,
                active_threads=active_threads,
                balanced_tiles=balanced_tiles,
                a_l2_bytes=stripe.a_l2_refill_bytes - cold_a_l2_bytes,
                b_l2_bytes=stripe.b_l2_refill_bytes - cold_b_l2_bytes,
                c_write_bytes=stripe.c_write_bytes - cold_c_write_bytes,
                compulsory_dram_bytes=0.0,
                spillable_dram_bytes=stripe.spillable_dram_bytes - cold_a_l2_bytes - cold_c_write_bytes,
                working_set_bytes=stripe.llc_working_set_bytes,
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
        w13_mapping = self._mapper.lower(
            work.w13,
            ExecutionSchedule(threads=threads),
        )
        w2_mapping = self._mapper.lower(
            work.w2,
            ExecutionSchedule(threads=threads),
        )
        w13_demand = self._stage_demand("w13", w13_mapping, self._w13_geometry)
        w2_demand = self._stage_demand("w2", w2_mapping, self._w2_geometry)
        overhead_ns = self.calibration.overheads.expert_overhead_ns(routes, threads)
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
        routes = int(routes)
        threads = int(threads)
        if routes <= 0:
            return 0.0
        key = (routes, threads)
        cached = self._t_iso_scalar_cache.get(key)
        if cached is not None:
            return cached
        value = self.predict_expert(routes, threads).total_ns
        self._t_iso_scalar_cache[key] = value
        return value

    def task_stage_phases(
        self,
        stage: str,
        routes: int,
        threads: int,
    ) -> tuple[tuple[float, int], ...]:
        """Expose stage-isolated time and compulsory packed-B bytes.

        This is the narrow adapter consumed by the offline cold-phase oracle.
        It does not expose analytical cache phases as CP-SAT decision variables:
        the master retains one cold-B phase per GEMM stage, while the complete
        event model remains responsible for cache/refill contention in reranking.
        """

        prediction = self.predict_expert(int(routes), int(threads))
        if stage == "w13":
            return ((prediction.w13_ns, prediction.w13_demand.stage_bytes),)
        if stage == "w2":
            return ((prediction.w2_ns, prediction.w2_demand.stage_bytes),)
        raise ValueError(f"unsupported stage {stage!r}")

    def t_iso_cache_identity(self) -> dict[str, object]:
        """Return every stable input needed to validate persisted T_iso values."""
        return {
            "analytic_model_schema_version": ANALYTIC_MODEL_SCHEMA_VERSION,
            "analytic_model_name": ANALYTIC_MODEL_NAME,
            "formula_source_sha256": _formula_source_sha256(),
            "calibration": self.calibration.to_dict(),
            "policy": list(self.policy.identity_key()),
            "supported_widths": list(self.supported_widths),
            "down_output_element_bytes": self.down_output_element_bytes,
            "exact_m": self._exact_m,
        }

    def import_t_iso_cache(self, entries: Mapping[tuple[int, int], float]) -> int:
        """Load validated scalar costs for this model's legal thread widths."""
        loaded = 0
        for (routes, threads), value in entries.items():
            routes = int(routes)
            threads = int(threads)
            value = float(value)
            if routes <= 0 or threads not in self.supported_widths or not math.isfinite(value) or value <= 0.0:
                continue
            self._t_iso_scalar_cache[(routes, threads)] = value
            loaded += 1
        return loaded

    def export_t_iso_cache(self) -> dict[tuple[int, int], float]:
        """Return a copy suitable for an atomic disk-cache write."""
        return dict(self._t_iso_scalar_cache)

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
        current_items = tuple(current.items())
        total_working_set = sum(phase.working_set_bytes for _, phase in current_items)
        spill_fraction = self._llc_miss_fraction(total_working_set)
        resource_vectors = {
            index: phase.resource_vectors(spill_fraction)
            for index, phase in current_items
        }
        provisional = {
            index: phase._duration_from_resource_times(resource_vectors[index][1])
            for index, phase in current_items
        }
        pressures: dict[str, AnalyticResourcePressure] = {}
        resource_scales: dict[str, float] = {}
        for resource_index, resource in enumerate(_SHARED_RESOURCES):
            demands = {
                index: resource_vectors[index][0][resource_index]
                for index, _ in current_items
            }
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
                        * resource_vectors[index][1][resource_index]
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
                phase._duration_from_resource_times(
                    resource_vectors[index][1], resource_scales
                )
                / phase.base_ns
            )
            for index, phase in current_items
        }
        return spill_fraction, provisional, pressures, multipliers

    def _active_phase_state_placed(
        self,
        current: Mapping[int, AnalyticPhase],
        task_cpu_ids: Sequence[Sequence[int]],
    ) -> tuple[
        dict[int, float],
        dict[int, float],
        dict[str, AnalyticResourcePressure],
        dict[int, float],
        dict[str, dict[str, float | int | None]],
        dict[int, float],
    ]:
        """Allocate LLC capacity and service by the tasks' physical domains."""

        domains = self.calibration.llc_domains
        if not domains:
            raise ValueError("placement-aware phase state requires calibrated LLC domains")
        cpu_to_domain = {
            cpu: domain.domain_id
            for domain in domains
            for cpu in domain.cpu_ids
        }
        current_items = tuple(current.items())
        task_domain_counts: dict[int, dict[str, int]] = {}
        domain_thread_counts = {domain.domain_id: 0 for domain in domains}
        domain_working_sets = {domain.domain_id: 0.0 for domain in domains}
        for index, phase in current_items:
            active_cpu_ids = tuple(task_cpu_ids[index][: phase.active_threads])
            if len(active_cpu_ids) != phase.active_threads:
                raise ValueError(f"task {index} placement is narrower than its active phase")
            counts = {domain.domain_id: 0 for domain in domains}
            for cpu in active_cpu_ids:
                try:
                    counts[cpu_to_domain[cpu]] += 1
                except KeyError as error:
                    raise ValueError(f"task {index} uses CPU {cpu} outside the calibrated rank") from error
            task_domain_counts[index] = counts
            for domain_id, count in counts.items():
                if count <= 0:
                    continue
                share = count / phase.active_threads
                domain_thread_counts[domain_id] += count
                domain_working_sets[domain_id] += phase.working_set_bytes * share

        domain_spill = {}
        for domain in domains:
            domain_id = domain.domain_id
            threads = domain_thread_counts[domain_id]
            domain_spill[domain_id] = (
                self._llc_miss_fraction(
                    domain_working_sets[domain_id],
                    llc_domain_threads={domain_id: threads},
                )
                if threads > 0
                else 0.0
            )
        task_spill = {
            index: sum(
                count / phase.active_threads * domain_spill[domain_id]
                for domain_id, count in task_domain_counts[index].items()
                if count > 0
            )
            for index, phase in current_items
        }
        resource_vectors = {
            index: phase.resource_vectors(task_spill[index])
            for index, phase in current_items
        }
        provisional = {
            index: phase._duration_from_resource_times(resource_vectors[index][1])
            for index, phase in current_items
        }
        pressures: dict[str, AnalyticResourcePressure] = {}
        resource_scales = {index: {} for index, _ in current_items}
        for resource_index, resource in enumerate(_SHARED_RESOURCES):
            if resource == "llc_bytes":
                continue
            demands = {
                index: resource_vectors[index][0][resource_index]
                for index, _ in current_items
            }
            active_threads = min(
                sum(current[index].active_threads for index, demand in demands.items() if demand > 0.0),
                self.calibration.cores_per_rank,
            )
            capacity = self.calibration.service_rate(resource, active_threads) if active_threads > 0 else math.inf
            if math.isfinite(capacity):
                offered_rate = sum(
                    demand
                    / max(
                        current[index].residual_scale
                        * resource_vectors[index][1][resource_index]
                        * 1e-9,
                        1e-30,
                    )
                    for index, demand in demands.items()
                    if demand > 0.0
                )
            else:
                offered_rate = sum(
                    demand / max(provisional[index] * 1e-9, 1e-30)
                    for index, demand in demands.items()
                    if demand > 0.0
                )
            utilization = offered_rate / capacity if math.isfinite(capacity) else 0.0
            dilation = max(1.0, utilization)
            pressures[resource] = AnalyticResourcePressure(
                active_threads=active_threads,
                offered_rate=offered_rate,
                capacity=capacity,
                utilization=utilization,
                dilation=dilation,
                allocated_rate=offered_rate / dilation,
                allocated_utilization=(offered_rate / dilation / capacity if math.isfinite(capacity) else 0.0),
            )
            for index, demand in demands.items():
                if demand > 0.0:
                    resource_scales[index][resource] = dilation

        llc_index = _SHARED_RESOURCES.index("llc_bytes")
        domain_details: dict[str, dict[str, float | int | None]] = {}
        domain_dilations = {}
        total_llc_offered = 0.0
        requesting_domain_threads = {domain.domain_id: 0 for domain in domains}
        for domain in domains:
            domain_id = domain.domain_id
            offered_rate = 0.0
            active_threads = 0
            for index, phase in current_items:
                count = task_domain_counts[index][domain_id]
                demand = resource_vectors[index][0][llc_index]
                if count <= 0 or demand <= 0.0:
                    continue
                share = count / phase.active_threads
                offered_rate += (
                    demand
                    * share
                    / max(
                        phase.residual_scale * resource_vectors[index][1][llc_index] * 1e-9,
                        1e-30,
                    )
                )
                active_threads += count
            requesting_domain_threads[domain_id] = active_threads
            capacity = (
                self.calibration.service_rate(
                    "llc_bytes",
                    active_threads,
                    llc_domain_threads={domain_id: active_threads},
                )
                if active_threads > 0
                else math.inf
            )
            utilization = offered_rate / capacity if math.isfinite(capacity) else 0.0
            dilation = max(1.0, utilization)
            domain_dilations[domain_id] = dilation
            total_llc_offered += offered_rate
            domain_details[domain_id] = {
                "active_threads": active_threads,
                "working_set_bytes": domain_working_sets[domain_id],
                "spill_fraction": domain_spill[domain_id],
                "offered_rate": offered_rate,
                "capacity": capacity if math.isfinite(capacity) else None,
                "utilization": utilization,
                "dilation": dilation,
            }
        total_requesting_threads = sum(requesting_domain_threads.values())
        rank_llc_capacity = (
            self.calibration.service_rate(
                "llc_bytes",
                total_requesting_threads,
                llc_domain_threads=requesting_domain_threads,
            )
            if total_requesting_threads > 0
            else math.inf
        )
        rank_llc_utilization = (
            total_llc_offered / rank_llc_capacity if math.isfinite(rank_llc_capacity) else 0.0
        )
        rank_llc_dilation = max(1.0, rank_llc_utilization)
        pressures["llc_bytes"] = AnalyticResourcePressure(
            active_threads=total_requesting_threads,
            offered_rate=total_llc_offered,
            capacity=rank_llc_capacity,
            utilization=rank_llc_utilization,
            dilation=rank_llc_dilation,
            allocated_rate=total_llc_offered / rank_llc_dilation,
            allocated_utilization=(
                total_llc_offered / rank_llc_dilation / rank_llc_capacity
                if math.isfinite(rank_llc_capacity)
                else 0.0
            ),
        )
        for index, phase in current_items:
            if resource_vectors[index][0][llc_index] <= 0.0:
                continue
            local_dilation = max(
                (
                    domain_dilations[domain_id]
                    for domain_id, count in task_domain_counts[index].items()
                    if count > 0
                ),
                default=1.0,
            )
            resource_scales[index]["llc_bytes"] = max(rank_llc_dilation, local_dilation)

        multipliers = {
            index: (
                phase._duration_from_resource_times(
                    resource_vectors[index][1], resource_scales[index]
                )
                / phase.base_ns
            )
            for index, phase in current_items
        }
        occupied_team_threads = min(
            sum(len(task_cpu_ids[index]) for index, _ in current_items),
            self.calibration.cores_per_rank,
        )
        team_pressure_dilation = {}
        for index, phase in current_items:
            team_width = len(task_cpu_ids[index])
            available_peer_threads = self.calibration.cores_per_rank - team_width
            peer_threads = max(occupied_team_threads - team_width, 0)
            peer_fraction = (
                min(peer_threads / available_peer_threads, 1.0)
                if available_peer_threads > 0
                else 0.0
            )
            isolated_scale = self.calibration.wide_team_pressure.isolated_scale(team_width)
            full_cohort_scale = self.calibration.wide_team_pressure.full_cohort_scale(team_width)
            dilation = (
                isolated_scale + (full_cohort_scale - isolated_scale) * peer_fraction
                if phase.kind in {"cold_b", "steady_b"}
                else 1.0
            )
            team_pressure_dilation[index] = dilation
            multipliers[index] *= dilation
        return (
            task_spill,
            provisional,
            pressures,
            multipliers,
            domain_details,
            team_pressure_dilation,
        )

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
        task_cpu_ids: Sequence[Sequence[int]] | None = None,
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
            if task_cpu_ids is None:
                spill_fraction, _, pressures, multipliers = self._active_phase_state(current)
                spill_log: float | dict[str, float] = spill_fraction
                domain_log = None
            else:
                (
                    task_spill,
                    _,
                    pressures,
                    multipliers,
                    domain_log,
                    team_pressure_dilation,
                ) = self._active_phase_state_placed(
                    current,
                    task_cpu_ids,
                )
                spill_log = {str(index): task_spill[index] for index in active}
            elapsed = min(remaining[index] * multipliers[index] for index in active)
            if event_log is not None:
                event = {
                    "start_ns": wall_ns,
                    "duration_ns": elapsed,
                    "active_tasks": list(active),
                    "phases": {str(index): current[index].name for index in active},
                    "phase_kinds": {str(index): current[index].kind for index in active},
                    "phase_dilation": {str(index): multipliers[index] for index in active},
                    "working_set_bytes": sum(phase.working_set_bytes for phase in current.values()),
                    "llc_spill_fraction": spill_log,
                    "resources": {
                        resource: {
                            "path": _RESOURCE_PATHS[resource],
                            **self._pressure_dict(pressure),
                        }
                        for resource, pressure in pressures.items()
                        if pressure.offered_rate > 0.0
                    },
                }
                if domain_log is not None:
                    event["llc_domains"] = domain_log
                    event["team_pressure_dilation"] = {
                        str(index): team_pressure_dilation[index]
                        for index in active
                    }
                event_log.append(event)
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

    @staticmethod
    def _normalize_placed_tasks(
        tasks,
    ) -> tuple[list[tuple[int, int, list[int]]], list[tuple[int, ...]]]:
        normalized = []
        task_cpu_ids = []
        ancestors: list[set[int]] = []
        for task_index, (routes, threads, raw_cpu_ids, raw_dependencies) in enumerate(tasks):
            threads = int(threads)
            cpu_ids = tuple(int(cpu) for cpu in raw_cpu_ids)
            dependencies = tuple(sorted({int(value) for value in raw_dependencies}))
            if threads <= 0 or len(cpu_ids) != threads or len(set(cpu_ids)) != threads:
                raise ValueError(f"task {task_index} placement must contain one unique CPU per thread")
            if dependencies and (dependencies[0] < 0 or dependencies[-1] >= task_index):
                raise ValueError("placed task dependencies must refer to earlier tasks")
            reachable = set(dependencies)
            for dependency in dependencies:
                reachable.update(ancestors[dependency])
            for earlier, earlier_cpu_ids in enumerate(task_cpu_ids):
                if set(cpu_ids).intersection(earlier_cpu_ids) and earlier not in reachable:
                    raise ValueError(
                        "overlapping placed tasks must be ordered by dependencies: "
                        f"earlier_task={earlier}, task={task_index}"
                    )
            ancestors.append(reachable)
            normalized.append((int(routes), threads, list(dependencies)))
            task_cpu_ids.append(cpu_ids)
        return normalized, task_cpu_ids

    def dag_makespan_placed(self, tasks) -> float:
        """Score a DAG whose tasks carry exact physical CPU placements."""

        normalized, task_cpu_ids = self._normalize_placed_tasks(tasks)
        if not self.calibration.llc_domains:
            return self.dag_makespan(normalized)
        return self._dag_result(normalized, task_cpu_ids=task_cpu_ids)[0]

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

    def explain_dag_placed(self, tasks) -> dict:
        """Explain a DAG while preserving every task's physical CPU placement."""

        normalized, task_cpu_ids = self._normalize_placed_tasks(tasks)
        events: list[dict] = []
        makespan_ns, finish_times = self._dag_result(
            normalized,
            event_log=events,
            task_cpu_ids=task_cpu_ids if self.calibration.llc_domains else None,
        )
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
            stripe = stage.stripe_demand
            return {
                "stage_bytes": stage.stage_bytes,
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
                "worker_stripe": (
                    {
                        "n_tiles": stripe.n_tiles,
                        "balanced_work_fraction": stripe.balanced_work_fraction,
                        "owner_window_bytes": stripe.owner_window_bytes,
                        "b_reuse_footprint_bytes": stripe.b_reuse_footprint_bytes,
                        "b_transient_reuses": stripe.b_transient_reuses,
                        "b_steady_reuses": stripe.b_steady_reuses,
                        "l2_steady_miss_fraction_b": stripe.l2_steady_miss_fraction_b,
                        "a_residency_footprint_bytes": stripe.a_residency_footprint_bytes,
                        "a_residency_capacity_bytes": stripe.a_residency_capacity_bytes,
                        "a_l2_refill_bytes": stripe.a_l2_refill_bytes,
                        "b_l2_refill_bytes": stripe.b_l2_refill_bytes,
                        "compulsory_dram_bytes": stripe.compulsory_dram_bytes,
                        "spillable_dram_bytes": stripe.spillable_dram_bytes,
                        "llc_working_set_bytes": stripe.llc_working_set_bytes,
                    }
                    if stripe is not None
                    else None
                ),
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
