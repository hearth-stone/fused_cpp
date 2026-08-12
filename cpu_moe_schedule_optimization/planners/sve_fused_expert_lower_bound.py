"""Build resource lower-bound problems for the current SVE fused expert.

The first version covers the W13 -> W2 GEMM sub-DAG only. Omitting gather,
publication, and merge work weakens the bound but cannot raise it. For each
thread width, the mode retains only work common to every legal stage window;
cache replay, range restart, and other window-specific costs are omitted.

This adapter certifies a scheduling domain built from the current exact-M SVE
kernel. It is not a lower bound for an implementation allowed to replace that
kernel with an arbitrary GEMM algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

if __package__:
    from .resource_lower_bound import CriticalChain, LowerBoundProblem, MoldableStage, ResourceCapacity, StageMode
    from ..cost_model.gemm_cost_model import ExecutionSchedule, fused_expert_work
    from ..cost_model.sve_bf16_kernel_model import SveBf16KernelExecution, SveBf16KernelProfile
else:
    from gemm_cost_model import ExecutionSchedule, fused_expert_work
    from resource_lower_bound import CriticalChain, LowerBoundProblem, MoldableStage, ResourceCapacity, StageMode
    from sve_bf16_kernel_model import SveBf16KernelExecution, SveBf16KernelProfile


@dataclass(frozen=True)
class ServiceUpperBound:
    """Aggregate and optional per-core upper service rates."""

    resource: str
    aggregate_per_second: float
    per_core_per_second: float | None = None

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("service resource name must be non-empty")
        if not math.isfinite(self.aggregate_per_second) or self.aggregate_per_second <= 0.0:
            raise ValueError(f"aggregate service for {self.resource!r} must be finite and positive")
        if self.per_core_per_second is not None and (
            not math.isfinite(self.per_core_per_second) or self.per_core_per_second <= 0.0
        ):
            raise ValueError(f"per-core service for {self.resource!r} must be finite and positive")

def _round_fraction_down(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def _ratio_down(amount: float, capacity: float) -> float:
    return _round_fraction_down(Fraction.from_float(amount) / Fraction.from_float(capacity))


def _integer_to_float_down(value: int) -> float:
    """Represent a non-negative work count without rounding it upward."""

    if value < 0:
        raise ValueError("work count must be non-negative")
    rounded = float(value)
    if int(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


@dataclass(frozen=True)
class SveFusedExpertHardwareEnvelope:
    """Upper service envelope used by the certified GEMM-only bound.

    Rates must be architectural upper bounds for a hardware-level certificate.
    Sustainable measured peaks instead define only a calibrated-model bound.
    """

    num_cores: int
    bfmmla: ServiceUpperBound
    key_instructions: ServiceUpperBound
    l1_load_bytes: ServiceUpperBound
    epilogue_elements: ServiceUpperBound
    dram_bytes: ServiceUpperBound | None = None

    def __post_init__(self) -> None:
        if self.num_cores <= 0:
            raise ValueError("num_cores must be positive")
        expected = {
            "bfmmla_flops": self.bfmmla,
            "key_instructions": self.key_instructions,
            "l1_load_bytes": self.l1_load_bytes,
            "epilogue_elements": self.epilogue_elements,
        }
        for resource, service in expected.items():
            if service.resource != resource:
                raise ValueError(f"expected service {resource!r}, got {service.resource!r}")
        if self.dram_bytes is not None and self.dram_bytes.resource != "dram_bytes":
            raise ValueError(f"expected service 'dram_bytes', got {self.dram_bytes.resource!r}")

    def services(self, *, include_weight_dram: bool) -> tuple[ServiceUpperBound, ...]:
        values = [self.bfmmla, self.key_instructions, self.l1_load_bytes, self.epilogue_elements]
        if include_weight_dram:
            if self.dram_bytes is None:
                raise ValueError("include_weight_dram requires a DRAM service upper bound")
            values.append(self.dram_bytes)
        return tuple(values)


def _mapping_demands(
    mapping: SveBf16KernelExecution,
    *,
    compulsory_weight_bytes: int,
    include_weight_dram: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    demand = mapping.demand
    aggregate = {
        "bfmmla_flops": _integer_to_float_down(demand.executed_flops),
        "key_instructions": _integer_to_float_down(demand.key_body_instructions),
        "l1_load_bytes": _integer_to_float_down(demand.l1_load_bytes),
        "epilogue_elements": _integer_to_float_down(demand.epilogue_elements),
    }
    balanced = {
        "bfmmla_flops": _integer_to_float_down(demand.balanced_executed_flops),
        "key_instructions": _integer_to_float_down(demand.balanced_key_body_instructions),
        "l1_load_bytes": _integer_to_float_down(demand.balanced_l1_load_bytes),
        "epilogue_elements": _integer_to_float_down(demand.balanced_epilogue_elements),
    }
    if include_weight_dram:
        aggregate["dram_bytes"] = _integer_to_float_down(compulsory_weight_bytes)
        balanced["dram_bytes"] = _integer_to_float_down(compulsory_weight_bytes)
    return aggregate, balanced


def _stage_mode(
    *,
    name: str,
    mapping: SveBf16KernelExecution,
    compulsory_weight_bytes: int,
    envelope: SveFusedExpertHardwareEnvelope,
    include_weight_dram: bool,
) -> StageMode:
    aggregate, balanced = _mapping_demands(
        mapping,
        compulsory_weight_bytes=compulsory_weight_bytes,
        include_weight_dram=include_weight_dram,
    )
    services = {
        service.resource: service for service in envelope.services(include_weight_dram=include_weight_dram)
    }
    active_threads = mapping.demand.active_threads
    duration_terms = []
    core_time_terms = []
    for resource, service in services.items():
        duration_terms.append(_ratio_down(aggregate[resource], service.aggregate_per_second))
        if service.per_core_per_second is not None:
            duration_terms.append(
                _ratio_down(balanced[resource], active_threads * service.per_core_per_second)
            )
            core_time_terms.append(_ratio_down(aggregate[resource], service.per_core_per_second))
    duration_lower_bound = max(duration_terms)

    # Total core-time follows from aggregate work and a per-core service
    # ceiling.  Do not multiply the busiest-lane duration by active_threads:
    # shorter lanes may finish earlier when N tiles are imbalanced.
    aggregate["core_seconds"] = max(core_time_terms, default=0.0)
    return StageMode.from_mapping(
        name=name,
        threads=mapping.schedule.threads,
        demands=aggregate,
        duration_lower_bound_s=duration_lower_bound,
    )


def build_sve_fused_expert_lower_bound_problem(
    route_counts: Sequence[int] | Mapping[int, int],
    *,
    hidden_size: int,
    intermediate_size: int,
    widths: Sequence[int],
    n_tile: int,
    envelope: SveFusedExpertHardwareEnvelope,
    include_weight_dram: bool = False,
) -> LowerBoundProblem:
    """Lower a route histogram to the first-version GEMM-only problem.

    ``widths`` must enumerate every thread width legal in the scheduling domain
    being certified.  Omitting a legal mode can raise the result and would make
    it a bound only for that explicitly restricted domain.

    ``include_weight_dram`` requires an initial-state contract that every
    active expert's W13 and W2 weights enter from DRAM at least once.
    """

    if min(hidden_size, intermediate_size, n_tile) <= 0:
        raise ValueError("hidden_size, intermediate_size, and n_tile must be positive")
    legal_widths = tuple(sorted({int(width) for width in widths}))
    if not legal_widths or legal_widths[0] <= 0 or legal_widths[-1] > envelope.num_cores:
        raise ValueError("widths must be positive and no larger than the hardware core count")
    if isinstance(route_counts, Mapping):
        experts = sorted((int(expert_id), int(routes)) for expert_id, routes in route_counts.items())
    else:
        experts = list(enumerate(int(routes) for routes in route_counts))
    if any(routes < 0 for _, routes in experts):
        raise ValueError("route counts must be non-negative")

    mapper = SveBf16KernelProfile(n_tile=n_tile)
    stages = []
    chains = []
    for expert_id, routes in experts:
        if routes == 0:
            continue
        logical = fused_expert_work(routes, hidden_size, intermediate_size)
        stage_ids = []
        for stage_name, stage_work in (("w13", logical.w13), ("w2", logical.w2)):
            stage_id = f"expert-{expert_id}:{stage_name}"
            modes = []
            for width in legal_widths:
                mapping = mapper.lower(stage_work, ExecutionSchedule(threads=width))
                modes.append(
                    _stage_mode(
                        name=f"{width}t-minimum-work",
                        mapping=mapping,
                        compulsory_weight_bytes=stage_work.compulsory_b_read_bytes,
                        envelope=envelope,
                        include_weight_dram=include_weight_dram,
                    )
                )
            stages.append(MoldableStage(stage_id=stage_id, modes=tuple(modes)))
            stage_ids.append(stage_id)
        chains.append(CriticalChain(name=f"expert-{expert_id}", stage_ids=tuple(stage_ids)))

    resources = [
        ResourceCapacity(service.resource, service.aggregate_per_second)
        for service in envelope.services(include_weight_dram=include_weight_dram)
    ]
    resources.append(ResourceCapacity("core_seconds", float(envelope.num_cores)))
    return LowerBoundProblem(
        resources=tuple(resources),
        stages=tuple(stages),
        critical_chains=tuple(chains),
    )
