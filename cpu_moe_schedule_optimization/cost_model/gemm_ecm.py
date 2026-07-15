#!/usr/bin/env python3
"""Microkernel-aware ECM shadow model for the fused SVE BF16 GEMMs.

The model separates facts derived from the assembly from machine service rates:

* M12/M8/M4/M2 instruction counts and logical/physical tail rows;
* L1 traffic caused by every N-tile visit;
* shared-cache traffic under the current "A once per worker/range, B once per
  M panel" convention;
* independently calibrated matrix, load, cache, and epilogue service rates.

It intentionally remains outside the active planner cost path until the
service rates have been measured on the target machine and held-out errors are
acceptable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


M_PANEL = 12
BF16_BYTES = 2
FP32_BYTES = 4


@dataclass(frozen=True)
class KernelPanel:
    """One dispatched M panel and its distinct row-count meanings."""

    logical_rows: int
    compute_rows: int
    packed_rows: int
    store_rows: int
    kernel: str


def kernel_panels(routes: int) -> tuple[KernelPanel, ...]:
    """Return the exact hybrid M12/tail dispatch for ``routes`` rows."""
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")

    full_panels, remainder = divmod(int(routes), M_PANEL)
    panels = [KernelPanel(12, 12, 12, 12, "M12")] * full_panels
    if remainder == 0:
        return tuple(panels)
    if remainder == 1:
        # The exported M1 symbol aliases the M2 body. Stores are predicated to
        # one logical row, while gather reserves one eight-row tail block.
        tail = KernelPanel(1, 2, 8, 1, "M2-as-M1")
    elif remainder == 2:
        tail = KernelPanel(2, 2, 8, 2, "M2")
    elif remainder <= 4:
        tail = KernelPanel(remainder, 4, 8, 4, "M4")
    elif remainder <= 8:
        tail = KernelPanel(remainder, 8, 8, 8, "M8")
    else:
        # 9--11 rows are padded upward so B is streamed once by M12 instead of
        # once by M8 and again by a second tail kernel.
        tail = KernelPanel(remainder, 12, 12, 12, "M12-padded")
    return (*panels, tail)


@dataclass(frozen=True)
class NTileAllocation:
    total_tiles: int
    range_tiles: tuple[int, ...]
    per_thread_tiles: tuple[int, ...]
    active_threads: int
    busiest_thread_tiles: int
    active_thread_range_scans: int


def allocate_n_tiles(
    n_columns: int,
    n_tile: int,
    threads: int,
    n_ranges: int,
) -> NTileAllocation:
    """Mirror the contiguous N-tile split used by each sequential range."""
    if min(n_columns, n_tile, threads, n_ranges) <= 0:
        raise ValueError("n_columns, n_tile, threads, and n_ranges must be positive")
    if n_columns % n_tile != 0:
        raise ValueError(f"n_columns must be padded to n_tile: {n_columns} % {n_tile} != 0")

    total_tiles = n_columns // n_tile
    range_base, range_extra = divmod(total_tiles, n_ranges)
    if range_extra:
        raise ValueError(f"N tiles={total_tiles} cannot form {n_ranges} equal ranges")
    range_tiles = (range_base,) * n_ranges
    if not range_tiles or range_base <= 0:
        raise ValueError(f"n_ranges={n_ranges} exceeds available N tiles={total_tiles}")

    per_thread = [0] * threads
    active_thread_range_scans = 0
    for tiles in range_tiles:
        thread_base, thread_extra = divmod(tiles, threads)
        active_thread_range_scans += min(tiles, threads)
        for thread_id in range(threads):
            per_thread[thread_id] += thread_base + (thread_id < thread_extra)

    active_threads = sum(value > 0 for value in per_thread)
    return NTileAllocation(
        total_tiles=total_tiles,
        range_tiles=range_tiles,
        per_thread_tiles=tuple(per_thread),
        active_threads=active_threads,
        busiest_thread_tiles=max(per_thread),
        active_thread_range_scans=active_thread_range_scans,
    )


@dataclass(frozen=True)
class GemmStage:
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
    stage: GemmStage
    panels: tuple[KernelPanel, ...]
    allocation: NTileAllocation
    logical_rows: int
    compute_rows: int
    packed_rows: int
    store_rows: int
    useful_flops: int
    executed_flops: int
    bfmmla_instructions: int
    a_load_instructions: int
    b_load_instructions: int
    balanced_bfmmla_instructions: int
    balanced_key_instructions: int
    l1_a_read_bytes: int
    l1_b_read_bytes: int
    l1_c_write_bytes: int
    balanced_l1_load_bytes: int
    private_refill_bytes: int
    llc_a_read_bytes: int
    llc_b_read_bytes: int
    llc_c_write_bytes: int
    output_elements: int
    balanced_output_elements: int

    @property
    def key_body_instructions(self) -> int:
        return self.bfmmla_instructions + self.a_load_instructions + self.b_load_instructions

    @property
    def l1_load_bytes(self) -> int:
        return self.l1_a_read_bytes + self.l1_b_read_bytes

    @property
    def llc_bytes(self) -> int:
        return self.llc_a_read_bytes + self.llc_b_read_bytes + self.llc_c_write_bytes

    @property
    def compute_efficiency(self) -> float:
        return self.useful_flops / self.executed_flops if self.executed_flops else 1.0


def gemm_ecm_work(stage: GemmStage) -> GemmEcmWork:
    """Derive instruction counts and traffic from the current SVE assembly."""
    panels = kernel_panels(stage.routes)
    allocation = allocate_n_tiles(
        stage.n,
        stage.n_tile,
        stage.threads,
        stage.n_ranges,
    )
    logical_rows = sum(panel.logical_rows for panel in panels)
    compute_rows = sum(panel.compute_rows for panel in panels)
    packed_rows = sum(panel.packed_rows for panel in panels)
    store_rows = sum(panel.store_rows for panel in panels)

    # For one N tile and one K4 body, an Mr kernel executes 2*Mr BFMMLA,
    # Mr/2 16-byte A broadcasts, and four VL-byte B loads.
    bfmmla_per_tile = compute_rows * stage.k // 2
    a_loads_per_tile = compute_rows * stage.k // 8
    b_loads_per_tile = len(panels) * stage.k
    bfmmla = bfmmla_per_tile * allocation.total_tiles
    a_loads = a_loads_per_tile * allocation.total_tiles
    b_loads = b_loads_per_tile * allocation.total_tiles

    busiest_bfmmla = bfmmla_per_tile * allocation.busiest_thread_tiles
    busiest_a_loads = a_loads_per_tile * allocation.busiest_thread_tiles
    busiest_b_loads = b_loads_per_tile * allocation.busiest_thread_tiles
    balanced_bfmmla = busiest_bfmmla * allocation.active_threads
    balanced_key = (busiest_bfmmla + busiest_a_loads + busiest_b_loads) * allocation.active_threads

    l1_a = a_loads * 16
    l1_b = b_loads * stage.vector_bytes
    l1_c = store_rows * stage.output_columns * stage.output_element_bytes
    balanced_l1 = (busiest_a_loads * 16 + busiest_b_loads * stage.vector_bytes) * allocation.active_threads

    # Shared-cache convention: every active worker/range brings each A panel
    # once from shared cache, while disjoint N owners together stream one full
    # B matrix for every physical M panel. Repeated A visits for later N tiles
    # are represented at L1, not counted again here.
    a_panel_bytes = compute_rows * stage.k * BF16_BYTES
    llc_a = a_panel_bytes * allocation.active_thread_range_scans
    llc_b = len(panels) * stage.k * stage.n * BF16_BYTES
    private_refill = llc_a + llc_b
    output_elements = store_rows * stage.output_columns
    output_columns_per_tile = stage.output_columns * stage.n_tile // stage.n
    balanced_output_elements = (
        store_rows * output_columns_per_tile * allocation.busiest_thread_tiles * allocation.active_threads
    )

    executed_flops = bfmmla * (2 * stage.vector_bytes)
    useful_flops = 2 * logical_rows * stage.k * stage.n
    return GemmEcmWork(
        stage=stage,
        panels=panels,
        allocation=allocation,
        logical_rows=logical_rows,
        compute_rows=compute_rows,
        packed_rows=packed_rows,
        store_rows=store_rows,
        useful_flops=useful_flops,
        executed_flops=executed_flops,
        bfmmla_instructions=bfmmla,
        a_load_instructions=a_loads,
        b_load_instructions=b_loads,
        balanced_bfmmla_instructions=balanced_bfmmla,
        balanced_key_instructions=balanced_key,
        l1_a_read_bytes=l1_a,
        l1_b_read_bytes=l1_b,
        l1_c_write_bytes=l1_c,
        balanced_l1_load_bytes=balanced_l1,
        private_refill_bytes=private_refill,
        llc_a_read_bytes=llc_a,
        llc_b_read_bytes=llc_b,
        llc_c_write_bytes=l1_c,
        output_elements=output_elements,
        balanced_output_elements=balanced_output_elements,
    )


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
    """Evaluate ``fixed + max(overlap, non-overlap transfers) + epilogue``."""
    matrix_flops = work.balanced_bfmmla_instructions * 2 * work.stage.vector_bytes
    matrix_ns = matrix_flops / caps.bfmmla_flops_per_second * 1e9
    frontend_ns = 0.0
    if caps.key_instructions_per_second is not None:
        frontend_ns = work.balanced_key_instructions / caps.key_instructions_per_second * 1e9
    l1_load_ns = work.balanced_l1_load_bytes / caps.l1_load_bytes_per_second * 1e9
    private_refill_ns = 0.0
    if caps.private_refill_bytes_per_second is not None:
        private_refill_ns = work.private_refill_bytes / caps.private_refill_bytes_per_second * 1e9
    llc_ns = work.llc_bytes / caps.llc_bytes_per_second * 1e9
    nonoverlap_ns = l1_load_ns + private_refill_ns + llc_ns
    body_candidates = {
        "matrix": matrix_ns,
        "frontend": frontend_ns,
        "load_transfer": nonoverlap_ns,
    }
    bottleneck = max(body_candidates, key=body_candidates.get)
    body_ns = body_candidates[bottleneck]
    epilogue_ns = 0.0
    if caps.epilogue_elements_per_second is not None:
        epilogue_ns = work.balanced_output_elements / caps.epilogue_elements_per_second * 1e9
    fixed_ns = caps.stage_fixed_ns + caps.range_fixed_ns * work.stage.n_ranges
    return EcmPrediction(
        matrix_ns=matrix_ns,
        frontend_ns=frontend_ns,
        l1_load_ns=l1_load_ns,
        private_refill_ns=private_refill_ns,
        llc_ns=llc_ns,
        nonoverlap_ns=nonoverlap_ns,
        body_ns=body_ns,
        epilogue_ns=epilogue_ns,
        fixed_ns=fixed_ns,
        total_ns=fixed_ns + body_ns + epilogue_ns,
        bottleneck=bottleneck,
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
                    required_tflops=(work.balanced_bfmmla_instructions * 2 * work.stage.vector_bytes / seconds / 1e12),
                    required_bfmmla_gips=(work.balanced_bfmmla_instructions / seconds / 1e9),
                    required_l1_gbs=work.balanced_l1_load_bytes / seconds / 1e9,
                    required_llc_gbs=work.llc_bytes / seconds / 1e9,
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
        shape = profile["shape"]
        for threads in sorted({row.threads for row in observations}):
            silu = base_by_key[("w13", threads)]
            identity = identity_by_key[("w13", threads)]
            w2 = identity_by_key[("w2", threads)]
            panel_work = gemm_ecm_work(
                w13_stage(
                    M_PANEL,
                    threads,
                    int(shape["hidden_size"]),
                    int(shape["ffn_hidden_size"]),
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
        "schema_version": 1,
        "kind": "sve_bf16_gemm_ecm_shadow",
        "source_profile": source_profile,
        "identity_source_profile": identity_source,
        "shape": profile["shape"],
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
            "body": "max(T_bfmmla, T_frontend, T_l1_load + T_private + T_llc)",
            "stage": "T_fixed + T_body + T_epilogue",
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
    parser = argparse.ArgumentParser(description="Build a microkernel-aware ECM shadow report from stage traces")
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
