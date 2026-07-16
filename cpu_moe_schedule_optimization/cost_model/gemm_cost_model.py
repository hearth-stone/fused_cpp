"""Implementation-weak contracts for layered GEMM cost models.

The model core separates three concerns:

* :class:`LogicalGemmWork` describes algorithmic work and compulsory traffic;
* a kernel mapper lowers that work and a schedule to :class:`KernelDemand`;
* :class:`MeasuredMachineProfile` converts the demand to a time prediction.

No class in this module knows about SVE, BFMMLA, or a particular tile shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LogicalGemmWork:
    """Implementation-independent work for one logical GEMM stage."""

    name: str
    routes: int
    k: int
    n: int
    output_columns: int
    input_element_bytes: int
    weight_element_bytes: int
    output_element_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("logical GEMM stage name must be non-empty")
        if self.routes < 0:
            raise ValueError(f"routes must be non-negative, got {self.routes}")
        dimensions = (
            self.k,
            self.n,
            self.output_columns,
            self.input_element_bytes,
            self.weight_element_bytes,
            self.output_element_bytes,
        )
        if min(dimensions) <= 0:
            raise ValueError("logical GEMM dimensions and element sizes must be positive")

    @property
    def useful_flops(self) -> int:
        return 2 * self.routes * self.k * self.n

    @property
    def compulsory_a_read_bytes(self) -> int:
        return self.routes * self.k * self.input_element_bytes

    @property
    def compulsory_b_read_bytes(self) -> int:
        if self.routes == 0:
            return 0
        return self.k * self.n * self.weight_element_bytes

    @property
    def compulsory_c_write_bytes(self) -> int:
        return self.routes * self.output_columns * self.output_element_bytes

    @property
    def compulsory_bytes(self) -> int:
        return self.compulsory_a_read_bytes + self.compulsory_b_read_bytes + self.compulsory_c_write_bytes

    @property
    def output_elements(self) -> int:
        return self.routes * self.output_columns


@dataclass(frozen=True)
class FusedExpertWork:
    """Logical W13/W2 work and the algorithmic dependency between stages."""

    w13: LogicalGemmWork
    w2: LogicalGemmWork
    dependencies: tuple[tuple[str, str], ...] = (("w13", "w2"),)

    @property
    def useful_flops(self) -> int:
        return self.w13.useful_flops + self.w2.useful_flops

    @property
    def compulsory_bytes(self) -> int:
        return self.w13.compulsory_bytes + self.w2.compulsory_bytes


def fused_expert_work(
    routes: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    input_element_bytes: int = 2,
    weight_element_bytes: int = 2,
    intermediate_element_bytes: int = 2,
    down_output_element_bytes: int = 4,
) -> FusedExpertWork:
    """Build logical fused-expert work without choosing a kernel or schedule."""
    if min(hidden_size, intermediate_size) <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive")
    w13 = LogicalGemmWork(
        name="w13",
        routes=routes,
        k=hidden_size,
        n=2 * intermediate_size,
        output_columns=intermediate_size,
        input_element_bytes=input_element_bytes,
        weight_element_bytes=weight_element_bytes,
        output_element_bytes=intermediate_element_bytes,
    )
    w2 = LogicalGemmWork(
        name="w2",
        routes=routes,
        k=intermediate_size,
        n=hidden_size,
        output_columns=hidden_size,
        input_element_bytes=intermediate_element_bytes,
        weight_element_bytes=weight_element_bytes,
        output_element_bytes=down_output_element_bytes,
    )
    return FusedExpertWork(w13=w13, w2=w2)


@dataclass(frozen=True)
class ExecutionSchedule:
    """Implementation-neutral scheduling decisions supplied to a kernel mapper."""

    threads: int
    parallel_axis: str = "N"
    sequential_n_ranges: int = 1

    def __post_init__(self) -> None:
        if self.threads <= 0:
            raise ValueError(f"threads must be positive, got {self.threads}")
        if self.parallel_axis not in {"M", "N"}:
            raise ValueError(f"parallel_axis must be 'M' or 'N', got {self.parallel_axis!r}")
        if self.sequential_n_ranges <= 0:
            raise ValueError("sequential_n_ranges must be positive")


@dataclass(frozen=True)
class KernelDemand:
    """Physical resource demand emitted by one implementation mapper."""

    logical_work: LogicalGemmWork
    schedule: ExecutionSchedule
    implementation_id: str
    compute_kind: str
    active_threads: int
    executed_flops: int
    balanced_executed_flops: int
    key_body_instructions: int
    balanced_key_body_instructions: int
    l1_load_bytes: int
    balanced_l1_load_bytes: int
    private_refill_bytes: int
    shared_cache_read_bytes: int
    shared_cache_write_bytes: int
    epilogue_elements: int
    balanced_epilogue_elements: int
    transient_working_set_bytes: int
    stage_invocations: int = 1
    range_invocations: int = 1

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.compute_kind:
            raise ValueError("implementation_id and compute_kind must be non-empty")
        values = (
            self.active_threads,
            self.executed_flops,
            self.balanced_executed_flops,
            self.key_body_instructions,
            self.balanced_key_body_instructions,
            self.l1_load_bytes,
            self.balanced_l1_load_bytes,
            self.private_refill_bytes,
            self.shared_cache_read_bytes,
            self.shared_cache_write_bytes,
            self.epilogue_elements,
            self.balanced_epilogue_elements,
            self.transient_working_set_bytes,
            self.stage_invocations,
            self.range_invocations,
        )
        if min(values) < 0:
            raise ValueError("kernel demand values must be non-negative")
        if self.active_threads > self.schedule.threads:
            raise ValueError("active_threads cannot exceed scheduled threads")
        if self.balanced_executed_flops < self.executed_flops:
            raise ValueError("balanced executed FLOPs cannot be smaller than physical FLOPs")
        if self.balanced_key_body_instructions < self.key_body_instructions:
            raise ValueError("balanced key instructions cannot be smaller than physical instructions")
        if self.balanced_l1_load_bytes < self.l1_load_bytes:
            raise ValueError("balanced L1 bytes cannot be smaller than physical L1 bytes")
        if self.balanced_epilogue_elements < self.epilogue_elements:
            raise ValueError("balanced epilogue work cannot be smaller than physical epilogue work")

    @property
    def shared_cache_bytes(self) -> int:
        return self.shared_cache_read_bytes + self.shared_cache_write_bytes

    @property
    def compute_efficiency(self) -> float:
        if self.executed_flops == 0:
            return 1.0
        return self.logical_work.useful_flops / self.executed_flops


class KernelMapping(Protocol):
    """Structural contract returned by implementation-specific mappers."""

    demand: KernelDemand


class KernelMapper(Protocol):
    """Map logical work and a schedule to implementation-specific demand."""

    implementation_id: str

    def lower(self, logical_work: LogicalGemmWork, schedule: ExecutionSchedule) -> KernelMapping: ...


@dataclass(frozen=True)
class MeasuredMachineProfile:
    """Measured service rates for one implementation and active thread width."""

    machine_id: str
    implementation_id: str
    threads: int
    matrix_flops_per_second: float
    l1_load_bytes_per_second: float
    shared_cache_bytes_per_second: float
    private_refill_bytes_per_second: float | None = None
    key_instructions_per_second: float | None = None
    epilogue_elements_per_second: float | None = None
    stage_fixed_ns: float = 0.0
    range_fixed_ns: float = 0.0

    def __post_init__(self) -> None:
        if not self.machine_id or not self.implementation_id:
            raise ValueError("machine_id and implementation_id must be non-empty")
        if self.threads <= 0:
            raise ValueError("profile threads must be positive")
        mandatory = (
            self.matrix_flops_per_second,
            self.l1_load_bytes_per_second,
            self.shared_cache_bytes_per_second,
        )
        optional = (
            self.private_refill_bytes_per_second,
            self.key_instructions_per_second,
            self.epilogue_elements_per_second,
        )
        if any(rate <= 0.0 for rate in mandatory):
            raise ValueError("mandatory measured service rates must be positive")
        if any(rate is not None and rate <= 0.0 for rate in optional):
            raise ValueError("optional measured service rates must be positive")
        if min(self.stage_fixed_ns, self.range_fixed_ns) < 0.0:
            raise ValueError("measured fixed costs must be non-negative")


@dataclass(frozen=True)
class StagePrediction:
    matrix_ns: float
    frontend_ns: float
    l1_load_ns: float
    private_refill_ns: float
    shared_cache_ns: float
    nonoverlap_ns: float
    body_ns: float
    epilogue_ns: float
    fixed_ns: float
    total_ns: float
    bottleneck: str

    @property
    def llc_ns(self) -> float:
        """Compatibility name used by the original GEMM ECM report."""
        return self.shared_cache_ns


def predict_stage(demand: KernelDemand, profile: MeasuredMachineProfile) -> StagePrediction:
    """Evaluate the measured machine response for one lowered kernel demand."""
    if demand.implementation_id != profile.implementation_id:
        raise ValueError(
            "machine profile implementation mismatch: "
            f"demand={demand.implementation_id!r} profile={profile.implementation_id!r}"
        )
    if demand.active_threads != profile.threads:
        raise ValueError(f"machine profile thread mismatch: demand={demand.active_threads} profile={profile.threads}")

    matrix_ns = demand.balanced_executed_flops / profile.matrix_flops_per_second * 1e9
    frontend_ns = 0.0
    if profile.key_instructions_per_second is not None:
        frontend_ns = demand.balanced_key_body_instructions / profile.key_instructions_per_second * 1e9
    l1_load_ns = demand.balanced_l1_load_bytes / profile.l1_load_bytes_per_second * 1e9
    private_refill_ns = 0.0
    if profile.private_refill_bytes_per_second is not None:
        private_refill_ns = demand.private_refill_bytes / profile.private_refill_bytes_per_second * 1e9
    shared_cache_ns = demand.shared_cache_bytes / profile.shared_cache_bytes_per_second * 1e9
    nonoverlap_ns = l1_load_ns + private_refill_ns + shared_cache_ns
    body_candidates = {
        "matrix": matrix_ns,
        "frontend": frontend_ns,
        "load_transfer": nonoverlap_ns,
    }
    bottleneck = max(body_candidates, key=body_candidates.get)
    body_ns = body_candidates[bottleneck]
    epilogue_ns = 0.0
    if profile.epilogue_elements_per_second is not None:
        epilogue_ns = demand.balanced_epilogue_elements / profile.epilogue_elements_per_second * 1e9
    fixed_ns = profile.stage_fixed_ns * demand.stage_invocations + profile.range_fixed_ns * demand.range_invocations
    return StagePrediction(
        matrix_ns=matrix_ns,
        frontend_ns=frontend_ns,
        l1_load_ns=l1_load_ns,
        private_refill_ns=private_refill_ns,
        shared_cache_ns=shared_cache_ns,
        nonoverlap_ns=nonoverlap_ns,
        body_ns=body_ns,
        epilogue_ns=epilogue_ns,
        fixed_ns=fixed_ns,
        total_ns=fixed_ns + body_ns + epilogue_ns,
        bottleneck=bottleneck,
    )
