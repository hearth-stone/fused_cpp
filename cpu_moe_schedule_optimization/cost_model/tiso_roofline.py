#!/usr/bin/env python3
"""Explainable work and roofline decomposition for isolated fused experts.

This module is deliberately a shadow model.  It computes kernel-visible work
from the M12/M8/M4/M2/M1 execution structure and can evaluate a roofline when
independently measured compute and cache-bandwidth ceilings are supplied.  It
does not infer both ceilings from one latency observation: only their active
minimum is identifiable from ``T_iso`` alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__:
    from .gemm_ecm import kernel_panels
else:
    from gemm_ecm import kernel_panels


M_PANEL = 12
BF16_BYTES = 2
FP32_BYTES = 4


def m12_tail_capacity(remainder: int) -> int:
    """Return the rows actually computed by the dispatched tail kernel."""
    if remainder < 0 or remainder >= M_PANEL:
        raise ValueError(f"remainder must be in [0, 11], got {remainder}")
    if remainder == 0:
        return 0
    return kernel_panels(remainder)[0].compute_rows


def panel_histogram(routes: int) -> dict[int, int]:
    """Map physical panel rows to occurrence count for ``routes`` logical rows."""
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    blocks, remainder = divmod(int(routes), M_PANEL)
    panels: dict[int, int] = {}
    if blocks:
        panels[M_PANEL] = blocks
    tail = m12_tail_capacity(remainder)
    if tail:
        panels[tail] = panels.get(tail, 0) + 1
    return panels


@dataclass(frozen=True)
class GemmWork:
    flops: int
    a_read_bytes: int
    b_read_bytes: int
    c_write_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.a_read_bytes + self.b_read_bytes + self.c_write_bytes


@dataclass(frozen=True)
class AuxWork:
    input_read_bytes: int
    packed_a_write_bytes: int
    down_read_bytes: int
    output_write_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.input_read_bytes
            + self.packed_a_write_bytes
            + self.down_read_bytes
            + self.output_write_bytes
        )


@dataclass(frozen=True)
class TisoWork:
    routes: int
    effective_rows: int
    packed_rows: int
    store_rows: int
    panel_count: int
    panel_histogram: dict[int, int]
    threads: int
    w13_n_ranges: int
    hidden_size: int
    intermediate_size: int
    w13: GemmWork
    w2: GemmWork
    aux: AuxWork

    @property
    def gemm_flops(self) -> int:
        return self.w13.flops + self.w2.flops

    @property
    def gemm_bytes(self) -> int:
        return self.w13.total_bytes + self.w2.total_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.gemm_flops / self.gemm_bytes if self.gemm_bytes else 0.0


def fused_expert_work(
    routes: int,
    threads: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    parallel_axis: str = "N",
    w13_n_ranges: int = 1,
    w2_output_bytes: int = FP32_BYTES,
) -> TisoWork:
    """Compute fused W13+W2 work using the current packed-A kernel loops.

    For N-split, every N partition scans packed A while packed B is partitioned
    across workers. Aggregate A traffic is multiplied by ``threads`` and by
    sequential W13 N ranges; B traffic remains one full weight pass per
    physical M panel.
    """
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    if min(threads, hidden_size, intermediate_size) <= 0:
        raise ValueError("threads, hidden_size, and intermediate_size must be positive")
    if parallel_axis not in {"M", "N"}:
        raise ValueError(f"parallel_axis must be 'M' or 'N', got {parallel_axis!r}")
    if w13_n_ranges <= 0:
        raise ValueError(f"w13_n_ranges must be positive, got {w13_n_ranges}")
    if w2_output_bytes not in {BF16_BYTES, FP32_BYTES}:
        raise ValueError("w2_output_bytes must be 2 (bf16) or 4 (fp32)")

    panel_sequence = kernel_panels(routes)
    panels = panel_histogram(routes)
    effective_rows = sum(panel.compute_rows for panel in panel_sequence)
    packed_rows = sum(panel.packed_rows for panel in panel_sequence)
    store_rows = sum(panel.store_rows for panel in panel_sequence)
    panel_count = len(panel_sequence)
    n_partitions = threads if parallel_axis == "N" else 1
    h = int(hidden_size)
    f = int(intermediate_size)

    # W13 is [M,H] x [H,2F], then fused SiLU writes BF16 [M,F].
    w13 = GemmWork(
        flops=4 * effective_rows * h * f,
        a_read_bytes=(
            BF16_BYTES * effective_rows * h * n_partitions * int(w13_n_ranges)
        ),
        b_read_bytes=BF16_BYTES * panel_count * h * (2 * f),
        c_write_bytes=BF16_BYTES * store_rows * f,
    )
    # W2 is [M,F] x [F,H]. The profiled skip_weighted path materializes FP32
    # down rows before converting/scattering them to the BF16 output.
    w2 = GemmWork(
        flops=2 * effective_rows * h * f,
        a_read_bytes=BF16_BYTES * effective_rows * f * n_partitions,
        b_read_bytes=BF16_BYTES * panel_count * f * h,
        c_write_bytes=w2_output_bytes * store_rows * h,
    )
    aux = AuxWork(
        input_read_bytes=BF16_BYTES * routes * h,
        packed_a_write_bytes=BF16_BYTES * packed_rows * h,
        down_read_bytes=w2_output_bytes * routes * h,
        output_write_bytes=BF16_BYTES * routes * h,
    )
    return TisoWork(
        routes=int(routes),
        effective_rows=effective_rows,
        packed_rows=packed_rows,
        store_rows=store_rows,
        panel_count=panel_count,
        panel_histogram=panels,
        threads=int(threads),
        w13_n_ranges=int(w13_n_ranges),
        hidden_size=h,
        intermediate_size=f,
        w13=w13,
        w2=w2,
        aux=aux,
    )


@dataclass(frozen=True)
class RooflineCaps:
    """Independently measured ceilings for one thread width."""

    w13_flops_per_second: float
    w2_flops_per_second: float
    l3_bytes_per_second: float
    copy_bytes_per_second: float
    fixed_ns: float = 0.0

    def __post_init__(self) -> None:
        rates = (
            self.w13_flops_per_second,
            self.w2_flops_per_second,
            self.l3_bytes_per_second,
            self.copy_bytes_per_second,
        )
        if any(value <= 0.0 for value in rates):
            raise ValueError("all roofline rates must be positive")
        if self.fixed_ns < 0.0:
            raise ValueError("fixed_ns must be non-negative")


@dataclass(frozen=True)
class RooflinePrediction:
    fixed_ns: float
    w13_compute_ns: float
    w13_memory_ns: float
    w13_ns: float
    w2_compute_ns: float
    w2_memory_ns: float
    w2_ns: float
    aux_ns: float
    total_ns: float


def predict_roofline(work: TisoWork, caps: RooflineCaps) -> RooflinePrediction:
    """Evaluate ``fixed + max(compute,memory)`` for W13 and W2."""
    w13_compute = work.w13.flops / caps.w13_flops_per_second * 1e9
    w13_memory = work.w13.total_bytes / caps.l3_bytes_per_second * 1e9
    w13 = max(w13_compute, w13_memory)
    w2_compute = work.w2.flops / caps.w2_flops_per_second * 1e9
    w2_memory = work.w2.total_bytes / caps.l3_bytes_per_second * 1e9
    w2 = max(w2_compute, w2_memory)
    aux = work.aux.total_bytes / caps.copy_bytes_per_second * 1e9
    total = caps.fixed_ns + w13 + w2 + aux
    return RooflinePrediction(
        fixed_ns=caps.fixed_ns,
        w13_compute_ns=w13_compute,
        w13_memory_ns=w13_memory,
        w13_ns=w13,
        w2_compute_ns=w2_compute,
        w2_memory_ns=w2_memory,
        w2_ns=w2,
        aux_ns=aux,
        total_ns=total,
    )


@dataclass(frozen=True)
class BulkObservation:
    threads: int
    points: int
    intercept_ns: float
    steady_panel_ns: float
    required_tflops: float
    required_l3_gbs: float
    arithmetic_intensity: float
    median_abs_error: float
    max_abs_error: float


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
        raise ValueError("linear fit requires distinct x values")
    slope = (count * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / count
    return intercept, slope


def fit_bulk_observations(
    profile: dict,
    *,
    min_routes: int = 192,
) -> list[BulkObservation]:
    """Extract the identifiable steady-panel service rate from ``T_iso``.

    ``required_tflops`` and ``required_l3_gbs`` are two views of the same
    observed panel duration. They are not independent compute/bandwidth
    measurements and must not both be treated as fitted hardware ceilings.
    """
    expert = profile["expert_shape"]
    hidden_size = int(expert["hidden_size"])
    intermediate_size = int(expert["intermediate_size"])
    axis = str(profile.get("kernel", {}).get("parallel_axis", "N"))
    kernel = profile.get("kernel", {})
    w13_n_ranges = (
        int(kernel.get("w13_split_chunks", 2))
        if bool(kernel.get("w13_split", False))
        else 1
    )
    isolated = profile["isolated"]
    thread_values = sorted({int(entry["threads"]) for entry in isolated})
    output: list[BulkObservation] = []
    for threads in thread_values:
        samples = [
            (int(entry["routes"]), float(entry["median_ns"]))
            for entry in isolated
            if int(entry["threads"]) == threads
            and int(entry["routes"]) >= min_routes
            and int(entry["routes"]) % M_PANEL == 0
        ]
        points = [(routes / M_PANEL, latency) for routes, latency in samples]
        intercept, panel_ns = _linear_fit(points)
        if panel_ns <= 0.0:
            raise ValueError(
                f"non-positive steady panel time for threads={threads}: {panel_ns}"
            )
        panel_work = fused_expert_work(
            M_PANEL,
            threads,
            hidden_size,
            intermediate_size,
            parallel_axis=axis,
            w13_n_ranges=w13_n_ranges,
        )
        errors = [
            abs((intercept + panels * panel_ns) / measured - 1.0)
            for panels, measured in points
        ]
        seconds = panel_ns / 1e9
        output.append(
            BulkObservation(
                threads=threads,
                points=len(points),
                intercept_ns=intercept,
                steady_panel_ns=panel_ns,
                required_tflops=panel_work.gemm_flops / seconds / 1e12,
                required_l3_gbs=panel_work.gemm_bytes / seconds / 1e9,
                arithmetic_intensity=panel_work.arithmetic_intensity,
                median_abs_error=statistics.median(errors),
                max_abs_error=max(errors),
            )
        )
    return output


def build_report(profile: dict, profile_path: str, min_routes: int) -> dict:
    observations = fit_bulk_observations(profile, min_routes=min_routes)
    kernel = profile.get("kernel", {})
    w13_n_ranges = (
        int(kernel.get("w13_split_chunks", 2))
        if bool(kernel.get("w13_split", False))
        else 1
    )
    return {
        "schema_version": 1,
        "kind": "explainable_tiso_roofline_shadow",
        "source_profile": profile_path,
        "formula": {
            "m_panel": M_PANEL,
            "w13_n_ranges": w13_n_ranges,
            "w13_flops": "4*m_compute*H*F",
            "w2_flops": "2*m_compute*H*F",
            "w13_bytes_nsplit": ("4*H*F + 2*c13*t*m_compute*H + 2*m_store*F"),
            "w2_bytes_nsplit": ("2*H*F + 2*t*m_compute*F + 4*m_store*H"),
            "stage_time": "max(flops / P_stage(t), bytes / B_L3(t))",
            "total_time": "O(t) + T_aux + sum_panels(T_w13 + T_w2)",
        },
        "fit_scope": {
            "min_routes": int(min_routes),
            "only_full_m12_panels": True,
        },
        "identifiability": (
            "Each T_iso slope identifies one effective panel service time. "
            "required_tflops and required_l3_gbs are equivalent views of that "
            "same observation; independent pure-GEMM and L3-bandwidth data are "
            "required to select the active roof."
        ),
        "observations": [asdict(observation) for observation in observations],
    }


def _print_report(report: dict) -> None:
    print(
        "T  points  intercept_ms  panel_us  req_TF/s  req_L3_GB/s  "
        "AI(F/B)  fit_med%  fit_max%"
    )
    for row in report["observations"]:
        print(
            f"{row['threads']:<2d} {row['points']:>6d} "
            f"{row['intercept_ns'] / 1e6:>13.3f} "
            f"{row['steady_panel_ns'] / 1e3:>9.3f} "
            f"{row['required_tflops']:>9.3f} "
            f"{row['required_l3_gbs']:>13.1f} "
            f"{row['arithmetic_intensity']:>8.2f} "
            f"{100 * row['median_abs_error']:>8.2f} "
            f"{100 * row['max_abs_error']:>8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive explainable bulk T_iso roofline observations"
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("--min-routes", type=int, default=192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.min_routes < M_PANEL:
        raise ValueError(f"--min-routes must be at least {M_PANEL}")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    report = build_report(profile, str(args.profile), args.min_routes)
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
