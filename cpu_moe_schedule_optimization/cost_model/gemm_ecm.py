#!/usr/bin/env python3
"""Compatibility CLI for the layered, implementation-weak GEMM cost model.

The generic contracts live in ``gemm_cost_model.py``.  The current SVE BF16
assembly mapping lives in ``sve_bf16_kernel_model.py``.  This module preserves
the original report CLI and helper API while delegating all predictions through
``LogicalGemmWork -> KernelDemand -> MeasuredMachineProfile``.
"""

from __future__ import annotations

__all__ = [
    "BF16_BYTES",
    "EcmCaps",
    "EcmPrediction",
    "ExecutionSchedule",
    "GemmEcmWork",
    "GemmStage",
    "KernelDemand",
    "KernelPanel",
    "LogicalGemmWork",
    "M_PANEL",
    "MeasuredMachineProfile",
    "NTileAllocation",
    "SveBf16KernelProfile",
    "allocate_n_tiles",
    "build_stage_report",
    "fused_expert_work",
    "gemm_ecm_work",
    "kernel_panels",
    "predict_ecm",
    "predict_stage",
    "w13_stage",
    "w2_stage",
]

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__:
    from .gemm_cost_model import (
        ExecutionSchedule,
        KernelDemand,
        LogicalGemmWork,
        MeasuredMachineProfile,
        fused_expert_work,
        predict_stage,
    )
    from .sve_bf16_kernel_model import (
        BF16_BYTES,
        M_PANEL,
        KernelPanel,
        NTileAllocation,
        SveBf16KernelExecution,
        SveBf16KernelProfile,
        allocate_n_tiles,
        kernel_panels,
    )
else:
    from gemm_cost_model import (
        ExecutionSchedule,
        KernelDemand,
        LogicalGemmWork,
        MeasuredMachineProfile,
        fused_expert_work,
        predict_stage,
    )
    from sve_bf16_kernel_model import (
        BF16_BYTES,
        M_PANEL,
        KernelPanel,
        NTileAllocation,
        SveBf16KernelExecution,
        SveBf16KernelProfile,
        allocate_n_tiles,
        kernel_panels,
    )


FP32_BYTES = 4


@dataclass(frozen=True)
class GemmStage:
    """Legacy facade combining logical work, schedule, and SVE tile settings."""

    name: str
    routes: int
    k: int
    n: int
    output_columns: int
    output_element_bytes: int
    threads: int
    n_tile: int
    n_ranges: int = 1

    def __post_init__(self) -> None:
        if self.name not in {"w13", "w2"}:
            raise ValueError(f"name must be 'w13' or 'w2', got {self.name!r}")
        if self.routes < 0:
            raise ValueError(f"routes must be non-negative, got {self.routes}")
        if (
            min(
                self.k,
                self.n,
                self.output_columns,
                self.output_element_bytes,
                self.threads,
                self.n_tile,
                self.n_ranges,
            )
            <= 0
        ):
            raise ValueError("all GEMM dimensions and execution parameters must be positive")
        if self.k % 8 != 0:
            raise ValueError(f"k must be padded to a multiple of eight, got {self.k}")
        if self.n % self.n_tile != 0:
            raise ValueError(f"n must be padded to n_tile: {self.n} % {self.n_tile} != 0")
        if (self.output_columns * self.n_tile) % self.n != 0:
            raise ValueError("each N tile must map to an integer number of outputs")

    @property
    def vector_bytes(self) -> int:
        return self.n_tile * BF16_BYTES

    def logical_work(self) -> LogicalGemmWork:
        return LogicalGemmWork(
            name=self.name,
            routes=self.routes,
            k=self.k,
            n=self.n,
            output_columns=self.output_columns,
            input_element_bytes=BF16_BYTES,
            weight_element_bytes=BF16_BYTES,
            output_element_bytes=self.output_element_bytes,
        )

    def schedule(self) -> ExecutionSchedule:
        return ExecutionSchedule(
            threads=self.threads,
            parallel_axis="N",
            sequential_n_ranges=self.n_ranges,
        )


def w13_stage(
    routes: int,
    threads: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    n_tile: int,
    n_ranges: int = 2,
) -> GemmStage:
    return GemmStage(
        name="w13",
        routes=routes,
        k=hidden_size,
        n=2 * intermediate_size,
        output_columns=intermediate_size,
        output_element_bytes=BF16_BYTES,
        threads=threads,
        n_tile=n_tile,
        n_ranges=n_ranges,
    )


def w2_stage(
    routes: int,
    threads: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    n_tile: int,
    output_element_bytes: int = FP32_BYTES,
) -> GemmStage:
    return GemmStage(
        name="w2",
        routes=routes,
        k=intermediate_size,
        n=hidden_size,
        output_columns=hidden_size,
        output_element_bytes=output_element_bytes,
        threads=threads,
        n_tile=n_tile,
    )


@dataclass(frozen=True)
class GemmEcmWork:
    """Legacy view over the implementation-specific SVE mapping result."""

    stage: GemmStage
    mapping: SveBf16KernelExecution

    @property
    def demand(self) -> KernelDemand:
        return self.mapping.demand

    @property
    def algorithm_work(self) -> LogicalGemmWork:
        return self.mapping.logical_work

    @property
    def execution_schedule(self) -> ExecutionSchedule:
        return self.mapping.schedule

    def __getattr__(self, name: str) -> object:
        # Preserve the original diagnostic attributes while keeping the generic
        # model independent of SVE details.
        return getattr(self.mapping, name)


def gemm_ecm_work(stage: GemmStage) -> GemmEcmWork:
    """Lower a legacy stage through the current SVE implementation mapper."""
    profile = SveBf16KernelProfile(n_tile=stage.n_tile)
    mapping = profile.lower(stage.logical_work(), stage.schedule())
    return GemmEcmWork(stage=stage, mapping=mapping)


@dataclass(frozen=True)
class EcmCaps:
    """Independent aggregate service rates for ``work.allocation.active_threads``."""

    bfmmla_flops_per_second: float
    l1_load_bytes_per_second: float
    llc_bytes_per_second: float
    private_refill_bytes_per_second: float | None = None
    key_instructions_per_second: float | None = None
    epilogue_elements_per_second: float | None = None
    stage_fixed_ns: float = 0.0
    range_fixed_ns: float = 0.0

    def __post_init__(self) -> None:
        mandatory = (
            self.bfmmla_flops_per_second,
            self.l1_load_bytes_per_second,
            self.llc_bytes_per_second,
        )
        optional = (
            self.private_refill_bytes_per_second,
            self.key_instructions_per_second,
            self.epilogue_elements_per_second,
        )
        if any(rate <= 0.0 for rate in mandatory):
            raise ValueError("mandatory ECM service rates must be positive")
        if any(rate is not None and rate <= 0.0 for rate in optional):
            raise ValueError("optional ECM service rates must be positive")
        if min(self.stage_fixed_ns, self.range_fixed_ns) < 0.0:
            raise ValueError("ECM fixed costs must be non-negative")

    def measured_profile(self, work: GemmEcmWork) -> MeasuredMachineProfile:
        """Adapt legacy caps to the measured machine-profile contract."""
        return MeasuredMachineProfile(
            machine_id="legacy_ecm_caps",
            implementation_id=work.demand.implementation_id,
            threads=work.demand.active_threads,
            matrix_flops_per_second=self.bfmmla_flops_per_second,
            l1_load_bytes_per_second=self.l1_load_bytes_per_second,
            shared_cache_bytes_per_second=self.llc_bytes_per_second,
            private_refill_bytes_per_second=self.private_refill_bytes_per_second,
            key_instructions_per_second=self.key_instructions_per_second,
            epilogue_elements_per_second=self.epilogue_elements_per_second,
            stage_fixed_ns=self.stage_fixed_ns,
            range_fixed_ns=self.range_fixed_ns,
        )


@dataclass(frozen=True)
class EcmPrediction:
    matrix_ns: float
    frontend_ns: float
    l1_load_ns: float
    private_refill_ns: float
    llc_ns: float
    nonoverlap_ns: float
    body_ns: float
    epilogue_ns: float
    fixed_ns: float
    total_ns: float
    bottleneck: str


def predict_ecm(work: GemmEcmWork, caps: EcmCaps) -> EcmPrediction:
    """Evaluate the generic machine response through the legacy API."""
    prediction = predict_stage(work.demand, caps.measured_profile(work))
    return EcmPrediction(
        matrix_ns=prediction.matrix_ns,
        frontend_ns=prediction.frontend_ns,
        l1_load_ns=prediction.l1_load_ns,
        private_refill_ns=prediction.private_refill_ns,
        llc_ns=prediction.shared_cache_ns,
        nonoverlap_ns=prediction.nonoverlap_ns,
        body_ns=prediction.body_ns,
        epilogue_ns=prediction.epilogue_ns,
        fixed_ns=prediction.fixed_ns,
        total_ns=prediction.total_ns,
        bottleneck=prediction.bottleneck,
    )


@dataclass(frozen=True)
class StageObservation:
    stage: str
    threads: int
    train_points: int
    holdout_points: int
    intercept_ns: float
    panel_ns: float
    required_tflops: float
    required_bfmmla_gips: float
    required_l1_gbs: float
    required_llc_gbs: float
    holdout_median_error: float
    holdout_max_error: float


def _linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("linear fit requires at least two points")
    count = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denominator = count * sxx - sx * sx
    if denominator == 0.0:
        raise ValueError("linear fit requires distinct route values")
    slope = (count * sxy - sx * sy) / denominator
    return (sy - slope * sx) / count, slope


def fit_stage_observations(
    profile: dict,
    *,
    n_tile: int,
    w13_n_ranges: int,
    train_routes: tuple[int, ...] = (192, 384, 768),
    holdout_routes: tuple[int, ...] = (1536, 2040),
) -> list[StageObservation]:
    """Fit M12 stage slopes and expose equivalent service-rate lower bounds."""
    shape = profile["shape"]
    hidden_size = int(shape["hidden_size"])
    intermediate_size = int(shape["ffn_hidden_size"])
    rows = profile["rows"]
    thread_values = sorted({int(row["threads"]) for row in rows})
    observations: list[StageObservation] = []
    for stage_name, field_name in (("w13", "w13_ms"), ("w2", "w2_ms")):
        for threads in thread_values:
            train = [
                (int(row["routes"]) / M_PANEL, float(row[field_name]) * 1e6)
                for row in rows
                if int(row["threads"]) == threads
                and int(row["routes"]) in train_routes
                and int(row["routes"]) % M_PANEL == 0
            ]
            intercept_ns, panel_ns = _linear_fit(train)
            if panel_ns <= 0.0:
                raise ValueError(f"non-positive {stage_name} panel slope for threads={threads}")
            holdout = [
                (int(row["routes"]) / M_PANEL, float(row[field_name]) * 1e6)
                for row in rows
                if int(row["threads"]) == threads
                and int(row["routes"]) in holdout_routes
                and int(row["routes"]) % M_PANEL == 0
            ]
            errors = [abs((intercept_ns + panel_ns * panels) / measured_ns - 1.0) for panels, measured_ns in holdout]
            if not errors:
                raise ValueError(f"no {stage_name} holdout points for threads={threads}")
            stage = (
                w13_stage(
                    M_PANEL,
                    threads,
                    hidden_size,
                    intermediate_size,
                    n_tile=n_tile,
                    n_ranges=w13_n_ranges,
                )
                if stage_name == "w13"
                else w2_stage(
                    M_PANEL,
                    threads,
                    hidden_size,
                    intermediate_size,
                    n_tile=n_tile,
                )
            )
            work = gemm_ecm_work(stage)
            seconds = panel_ns / 1e9
            observations.append(
                StageObservation(
                    stage=stage_name,
                    threads=threads,
                    train_points=len(train),
                    holdout_points=len(holdout),
                    intercept_ns=intercept_ns,
                    panel_ns=panel_ns,
                    required_tflops=(work.demand.balanced_executed_flops / seconds / 1e12),
                    required_bfmmla_gips=(work.balanced_bfmmla_instructions / seconds / 1e9),
                    required_l1_gbs=work.demand.balanced_l1_load_bytes / seconds / 1e9,
                    required_llc_gbs=work.demand.shared_cache_bytes / seconds / 1e9,
                    holdout_median_error=statistics.median(errors),
                    holdout_max_error=max(errors),
                )
            )
    return observations


def build_stage_report(
    profile: dict,
    *,
    source_profile: str,
    n_tile: int,
    w13_n_ranges: int,
    identity_profile: dict | None = None,
    identity_source: str | None = None,
    train_routes: tuple[int, ...] = (192, 384, 768),
    holdout_routes: tuple[int, ...] = (1536, 2040),
) -> dict:
    shape = profile["shape"]
    hidden_size = int(shape["hidden_size"])
    intermediate_size = int(shape["ffn_hidden_size"])
    logical_expert = fused_expert_work(
        M_PANEL,
        hidden_size,
        intermediate_size,
    )
    kernel_profile = SveBf16KernelProfile(n_tile=n_tile)
    observations = fit_stage_observations(
        profile,
        n_tile=n_tile,
        w13_n_ranges=w13_n_ranges,
        train_routes=train_routes,
        holdout_routes=holdout_routes,
    )
    identity_observations: list[StageObservation] = []
    silu_deltas: list[dict] = []
    if identity_profile is not None:
        if identity_profile["shape"] != profile["shape"]:
            raise ValueError("SiLU and identity profiles must have identical shapes")
        identity_observations = fit_stage_observations(
            identity_profile,
            n_tile=n_tile,
            w13_n_ranges=w13_n_ranges,
            train_routes=train_routes,
            holdout_routes=holdout_routes,
        )
        base_by_key = {(row.stage, row.threads): row for row in observations}
        identity_by_key = {(row.stage, row.threads): row for row in identity_observations}
        for threads in sorted({row.threads for row in observations}):
            silu = base_by_key[("w13", threads)]
            identity = identity_by_key[("w13", threads)]
            w2 = identity_by_key[("w2", threads)]
            panel_work = gemm_ecm_work(
                w13_stage(
                    M_PANEL,
                    threads,
                    hidden_size,
                    intermediate_size,
                    n_tile=n_tile,
                    n_ranges=w13_n_ranges,
                )
            )
            delta_ns = silu.panel_ns - identity.panel_ns
            silu_deltas.append(
                {
                    "threads": threads,
                    "silu_extra_panel_ns": delta_ns,
                    "silu_extra_fraction_of_w13": delta_ns / silu.panel_ns,
                    "silu_extra_ns_per_output": (delta_ns / panel_work.output_elements),
                    "identity_w13_to_w2_panel_ratio": (identity.panel_ns / w2.panel_ns),
                }
            )

    return {
        "schema_version": 2,
        "kind": "sve_bf16_gemm_ecm_shadow",
        "source_profile": source_profile,
        "identity_source_profile": identity_source,
        "shape": profile["shape"],
        "model_layers": {
            "algorithm": {
                "contract": "LogicalGemmWork",
                "formula": "w_s = A_s(x)",
                "implementation_independent": True,
                "route_12_example": {
                    "w13_useful_flops": logical_expert.w13.useful_flops,
                    "w2_useful_flops": logical_expert.w2.useful_flops,
                    "w13_compulsory_bytes": logical_expert.w13.compulsory_bytes,
                    "w2_compulsory_bytes": logical_expert.w2.compulsory_bytes,
                    "dependencies": [list(edge) for edge in logical_expert.dependencies],
                },
            },
            "implementation": {
                "contract": "KernelDemand",
                "formula": "d_s = Phi_kappa(w_s, sigma_s)",
                "implementation_id": kernel_profile.implementation_id,
                "mapping_source": "static implementation contract",
            },
            "machine_response": {
                "contract": "MeasuredMachineProfile",
                "formula": "T_hat_s = Psi_mu(d_s, sigma_s) + O_kappa_mu(sigma_s)",
                "measured": True,
                "profile_scope": "one implementation and active thread width",
            },
        },
        "kernel": {
            "m_panel": M_PANEL,
            "n_tile": n_tile,
            "vector_bytes": n_tile * BF16_BYTES,
            "w13_n_ranges": w13_n_ranges,
            "m12_k4_n_tile": {
                "bfmmla_instructions": 24,
                "a_broadcast_loads": 6,
                "b_vector_loads": 4,
            },
        },
        "formula": {
            "full": "w_s=A_s(x); d_s=Phi_kappa(w_s,sigma_s); T_hat_s=Psi_mu(d_s,sigma_s)+O_kappa_mu(sigma_s)",
            "body": "max(T_bfmmla, T_frontend, T_l1_load + T_private + T_llc)",
            "stage": "T_stage_fixed + N_range*T_range_fixed + T_body + T_epilogue",
            "l1_a": "2 * compute_rows * K * N_tiles",
            "l1_b": "2 * panel_count * K * N",
            "llc_a": "2 * compute_rows * K * sum_ranges(active_threads)",
            "llc_b": "2 * panel_count * K * N",
        },
        "fit_scope": {
            "train_routes": list(train_routes),
            "holdout_routes": list(holdout_routes),
            "full_m12_only": True,
        },
        "identifiability": (
            "The required rates are equivalent lower-bound views of one stage "
            "slope. They do not become independent hardware ceilings until "
            "pure BFMMLA, load/cache, and epilogue rates are measured separately."
        ),
        "observations": [asdict(row) for row in observations],
        "identity_observations": [asdict(row) for row in identity_observations],
        "silu_deltas": silu_deltas,
    }


def _print_report(report: dict) -> None:
    print("stage T panel_us req_TF/s BFMMLA_GI/s L1_GB/s LLC_GB/s hold_med% hold_max%")
    for row in report["observations"]:
        print(
            f"{row['stage']:<4s} {row['threads']:>2d} "
            f"{row['panel_ns'] / 1e3:>8.3f} "
            f"{row['required_tflops']:>8.3f} "
            f"{row['required_bfmmla_gips']:>12.3f} "
            f"{row['required_l1_gbs']:>9.1f} "
            f"{row['required_llc_gbs']:>10.1f} "
            f"{100 * row['holdout_median_error']:>9.2f} "
            f"{100 * row['holdout_max_error']:>9.2f}"
        )
    if report["silu_deltas"]:
        print("\nT SiLU_extra_us W13_extra% ns/output identity_W13/W2")
        for row in report["silu_deltas"]:
            print(
                f"{row['threads']:>2d} "
                f"{row['silu_extra_panel_ns'] / 1e3:>13.3f} "
                f"{100 * row['silu_extra_fraction_of_w13']:>10.2f} "
                f"{row['silu_extra_ns_per_output']:>9.4f} "
                f"{row['identity_w13_to_w2_panel_ratio']:>15.3f}"
            )


def _parse_routes(value: str) -> tuple[int, ...]:
    routes = tuple(int(item) for item in value.split(",") if item)
    if len(routes) < 2 or any(route <= 0 for route in routes):
        raise argparse.ArgumentTypeError("route list needs at least two positive values")
    return routes


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a layered GEMM cost shadow report from stage traces")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--identity-profile", type=Path)
    parser.add_argument(
        "--n-tile",
        type=int,
        help="override profile kernel.backend_n_tile (legacy default: 8)",
    )
    parser.add_argument(
        "--w13-n-ranges",
        type=int,
        help="override profile kernel.w13_n_ranges (legacy default: 2)",
    )
    parser.add_argument(
        "--train-routes",
        type=_parse_routes,
        default=(192, 384, 768),
    )
    parser.add_argument(
        "--holdout-routes",
        type=_parse_routes,
        default=(1536, 2040),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    kernel = profile.get("kernel", {})
    n_tile = int(args.n_tile if args.n_tile is not None else kernel.get("backend_n_tile", 8))
    w13_n_ranges = int(args.w13_n_ranges if args.w13_n_ranges is not None else kernel.get("w13_n_ranges", 2))
    identity_profile = None
    if args.identity_profile is not None:
        identity_profile = json.loads(args.identity_profile.read_text(encoding="utf-8"))
    report = build_stage_report(
        profile,
        source_profile=str(args.profile),
        n_tile=n_tile,
        w13_n_ranges=w13_n_ranges,
        identity_profile=identity_profile,
        identity_source=(str(args.identity_profile) if args.identity_profile is not None else None),
        train_routes=args.train_routes,
        holdout_routes=args.holdout_routes,
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
