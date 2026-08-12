#!/usr/bin/env python3
"""Build a thin analytical MoE calibration from independent service probes.

An optional isolated training profile contributes only three scalar residuals:
one expert-fixed cost, one per-route cost, and one common W13/W2 stage scale.
Contention rows are never read during fitting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

try:
    from analytic_model import AnalyticMachineCalibration, SaturatingServiceCurve
    from validate_analytic_model import _model_from_profile
except ImportError:  # pragma: no cover - package-style import
    from .analytic_model import AnalyticMachineCalibration, SaturatingServiceCurve
    from .validate_analytic_model import _model_from_profile


POWER_RESOURCES = {"gemm_core_flops", "matrix_flops", "l1_bytes", "l2_bytes", "llc_bytes"}
SHARED_BOTTLENECK_RESOURCES = {"dram_bytes"}


def parse_int_set(value: str) -> set[int]:
    result = {int(item) for item in value.split(",") if item.strip()}
    if not result or min(result) <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {value!r}")
    return result


def _service_rows(probe: dict, resource: str) -> list[dict]:
    rows = sorted(probe["services"][resource]["rows"], key=lambda row: int(row["threads"]))
    if not rows or int(rows[0]["threads"]) != 1:
        raise ValueError(f"{resource} probe must contain a one-thread point")
    if any(float(row["aggregate_rate"]) <= 0.0 for row in rows):
        raise ValueError(f"{resource} rates must be positive")
    return rows


def select_curve(probe: dict, resource: str) -> tuple[dict, dict]:
    rows = _service_rows(probe, resource)
    single = float(rows[0]["aggregate_rate"])
    if resource in POWER_RESOURCES or resource == "gemm_l2_flops":
        saturation_threads = int(rows[-1]["threads"])
        saturated = max(single, float(rows[-1]["aggregate_rate"]))
        curve_name = "power"
    elif resource in SHARED_BOTTLENECK_RESOURCES:
        maximum = max(float(row["aggregate_rate"]) for row in rows)
        knee_index = len(rows) - 1
        for index, row in enumerate(rows):
            tail = [float(item["aggregate_rate"]) for item in rows[index:]]
            if float(row["aggregate_rate"]) >= 0.95 * maximum and min(tail) >= 0.85 * maximum:
                knee_index = index
                break
        saturation_threads = int(rows[knee_index]["threads"])
        saturated = statistics.median(float(row["aggregate_rate"]) for row in rows[knee_index:])
        saturated = min(max(single, saturated), single * saturation_threads)
        curve_name = "shared_bottleneck"
    else:  # pragma: no cover - guarded by the caller
        raise KeyError(resource)

    payload = {
        "single_thread_rate": single,
        "saturated_rate": saturated,
        "saturation_threads": saturation_threads,
        "curve": curve_name,
    }
    curve = SaturatingServiceCurve.from_dict(payload)
    residuals = []
    for row in rows:
        measured = float(row["aggregate_rate"])
        predicted = curve.rate(int(row["threads"]))
        residuals.append(
            {
                "threads": int(row["threads"]),
                "measured_rate": measured,
                "predicted_rate": predicted,
                "relative_error": predicted / measured - 1.0,
            }
        )
    fit = {
        "curve": payload,
        "mape": statistics.fmean(abs(row["relative_error"]) for row in residuals),
        "max_absolute_relative_error": max(abs(row["relative_error"]) for row in residuals),
        "rows": residuals,
    }
    return payload, fit


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    count = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(count)]
    for column in range(count):
        pivot = max(range(column, count), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("singular least-squares system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(count):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(count)]


def _least_squares(features: list[list[float]], targets: list[float], columns: tuple[int, ...]) -> list[float]:
    scales = [max(abs(row[column]) for row in features) or 1.0 for column in columns]
    normalized = [[row[column] / scale for column, scale in zip(columns, scales)] for row in features]
    normal = [
        [sum(row[left] * row[right] for row in normalized) for right in range(len(columns))]
        for left in range(len(columns))
    ]
    rhs = [sum(row[column] * target for row, target in zip(normalized, targets)) for column in range(len(columns))]
    normalized_coefficients = _solve_linear_system(normal, rhs)
    coefficients = [0.0, 0.0, 0.0]
    for column, coefficient, scale in zip(columns, normalized_coefficients, scales):
        coefficients[column] = coefficient / scale
    return coefficients


def fit_nonnegative_residuals(features: list[list[float]], targets: list[float]) -> tuple[list[float], float]:
    """Fit fixed + route_ns * M + gamma * physical_ns with non-negative terms."""
    if len(features) != len(targets) or len(features) < 3:
        raise ValueError("residual fit requires at least three matching points")
    best: tuple[float, list[float]] | None = None
    # The physical stage scale is mandatory. Fixed and route terms may land on
    # the non-negative boundary when the explicit physical model explains them.
    for optional_count in range(3):
        for optional in itertools.combinations((0, 1), optional_count):
            columns = tuple(sorted((*optional, 2)))
            try:
                coefficients = _least_squares(features, targets, columns)
            except ValueError:
                continue
            if coefficients[2] <= 0.0 or any(value < -1e-9 for value in coefficients):
                continue
            coefficients = [max(0.0, value) for value in coefficients]
            sse = sum(
                (sum(coefficient * value for coefficient, value in zip(coefficients, row)) - target) ** 2
                for row, target in zip(features, targets)
            )
            if best is None or sse < best[0]:
                best = (sse, coefficients)
    if best is None:
        raise ValueError("failed to find a non-negative residual calibration")
    return best[1], best[0]


def fit_operator_residuals(
    calibration_payload: dict,
    profile: dict,
    *,
    train_routes: set[int],
    train_threads: set[int],
) -> dict:
    initial = AnalyticMachineCalibration.from_dict(calibration_payload)
    model = _model_from_profile(initial, profile, down_output_element_bytes=2)
    selected = [
        row
        for row in profile["isolated"]
        if int(row["routes"]) in train_routes and int(row["threads"]) in train_threads
    ]
    expected = len(train_routes) * len(train_threads)
    if len(selected) != expected:
        available = {(int(row["routes"]), int(row["threads"])) for row in selected}
        missing = sorted(set(itertools.product(train_routes, train_threads)) - available)
        raise ValueError(f"training profile is missing isolated points: {missing}")

    features = []
    targets = []
    for row in selected:
        routes = int(row["routes"])
        threads = int(row["threads"])
        physical_ns = model.T_iso(routes, threads)
        features.append([1.0, float(routes), physical_ns])
        targets.append(float(row["median_ns"]))
    coefficients, sse = fit_nonnegative_residuals(features, targets)
    expert_fixed_ns, route_ns, stage_scale = coefficients

    rows = []
    for source, feature, target in zip(selected, features, targets):
        predicted = sum(coefficient * value for coefficient, value in zip(coefficients, feature))
        rows.append(
            {
                "routes": int(source["routes"]),
                "threads": int(source["threads"]),
                "measured_ns": target,
                "predicted_ns": predicted,
                "relative_error": predicted / target - 1.0,
            }
        )
    return {
        "expert_fixed_ns": expert_fixed_ns,
        "route_ns": route_ns,
        "stage_scale": stage_scale,
        "sum_squared_error_ns2": sse,
        "mape": statistics.fmean(abs(row["relative_error"]) for row in rows),
        "max_absolute_relative_error": max(abs(row["relative_error"]) for row in rows),
        "train_routes": sorted(train_routes),
        "train_threads": sorted(train_threads),
        "rows": rows,
    }


def build_calibration(
    probe: dict,
    *,
    machine_id: str,
    l2_effective_fraction: float,
    llc_effective_fraction: float,
    l2_b_reuse_effective_fraction: float | None = None,
    l2_b_reuse_miss_floor: float,
    l2_b_reuse_miss_at_capacity: float,
    l2_b_reuse_miss_ceiling: float,
    relative_uncertainty: float,
    backend_n_tile: int = 8,
) -> tuple[dict, dict]:
    if probe.get("kind") != "moe_analytic_service_probe":
        raise ValueError("input is not an analytical service probe")
    services: dict[str, dict] = {}
    service_fit: dict[str, dict] = {}
    for resource in sorted(POWER_RESOURCES | SHARED_BOTTLENECK_RESOURCES):
        services[resource], service_fit[resource] = select_curve(probe, resource)
    if "gemm_l2_flops" in probe["services"]:
        _, service_fit["gemm_l2_flops"] = select_curve(probe, "gemm_l2_flops")
    matrix = services["matrix_flops"]
    kernel = probe["kernel"]
    frontend_per_matrix_instruction = float(kernel["frontend_instructions_per_cycle"]) / float(
        kernel["bfmmla_instructions_per_cycle"]
    )
    flops_per_bfmmla = float(kernel["bfmmla_flops_per_instruction"])
    services["frontend_instructions"] = {
        "single_thread_rate": matrix["single_thread_rate"] / flops_per_bfmmla * frontend_per_matrix_instruction,
        "saturated_rate": matrix["saturated_rate"] / flops_per_bfmmla * frontend_per_matrix_instruction,
        "saturation_threads": matrix["saturation_threads"],
        "curve": matrix["curve"],
    }
    service_fit["frontend_instructions"] = {
        "curve": services["frontend_instructions"],
        "source": "matrix probe scaled by calibrated frontend/BFMMLA issue widths",
        "frontend_instructions_per_cycle": kernel["frontend_instructions_per_cycle"],
        "bfmmla_instructions_per_cycle": kernel["bfmmla_instructions_per_cycle"],
    }
    caches = probe["caches"]
    widths = sorted(int(row["threads"]) for row in probe["services"]["gemm_core_flops"]["rows"])
    service_payload = probe.get("services", {})
    panel_range_restart_ns = float(
        service_payload.get("panel_range_restart", {}).get("panel_range_restart_ns", 0.0)
    )
    has_fused_w13_restart = "w13_fused_panel_range_restart" in service_payload
    w13_panel_range_restart_ns = float(
        service_payload.get("w13_fused_panel_range_restart", {})
        .get("panel_range_restart_ns", panel_range_restart_ns)
    )
    payload = {
        "schema_version": 1,
        "kind": "moe_analytic_machine",
        "machine": {
            "id": machine_id,
            "cores_per_rank": int(probe["machine"]["cores_per_rank"]),
        },
        "kernel": {"backend_n_tile": int(backend_n_tile)},
        "caches": {
            "l1d_bytes_per_core": int(caches["l1d_bytes_per_core"]),
            "l2_bytes_per_core": int(caches["l2_bytes_per_core"]),
            "llc_bytes_per_rank": int(caches["llc_bytes_per_rank"]),
            "l2_effective_fraction": l2_effective_fraction,
            "llc_effective_fraction": llc_effective_fraction,
            "l2_b_reuse_effective_fraction": (
                l2_effective_fraction
                if l2_b_reuse_effective_fraction is None
                else l2_b_reuse_effective_fraction
            ),
            "l2_b_reuse_miss_floor": l2_b_reuse_miss_floor,
            "l2_b_reuse_miss_at_capacity": l2_b_reuse_miss_at_capacity,
            "l2_b_reuse_miss_ceiling": l2_b_reuse_miss_ceiling,
        },
        "services": services,
        "overheads": {
            "call_setup_ns": 0.0,
            "expert_fixed_ns": 0.0,
            "route_ns": 0.0,
            "stage_fixed_ns": 0.0,
            "range_fixed_ns": 0.0,
            "panel_range_restart_ns": panel_range_restart_ns,
            "w13_panel_range_restart_ns": w13_panel_range_restart_ns,
            "w2_panel_range_restart_ns": panel_range_restart_ns,
        },
        "planner": {"supported_widths": widths},
        "uncertainty": {"relative": relative_uncertainty},
        "stage_scales": {"w13": 1.0, "w2": 1.0},
        "provenance": {
            "service_probe_kind": probe["kind"],
            "service_probe_machine": probe["machine"],
            "gemm_core_service": "m12_l1_hot_full_no_store",
            "matrix_service": "register_only_bfmmla_diagnostic",
            "contention_measurements_used": False,
            "panel_range_restart_service": {
                "w13": (
                    "m12_l1_hot_fused_w13_extra_n_range"
                    if has_fused_w13_restart
                    else "fallback_to_m12_l1_hot_full_no_store_extra_n_range"
                ),
                "w2": "m12_l1_hot_full_no_store_extra_n_range",
            },
        },
    }
    if l2_b_reuse_miss_floor > 0.0 or l2_b_reuse_miss_at_capacity < 1.0 or l2_b_reuse_miss_ceiling < 1.0:
        payload["provenance"]["l2_b_retention_calibration"] = {
            "kind": "independent_packed_b_repeated_scan_probe",
            "effective_fraction": payload["caches"]["l2_b_reuse_effective_fraction"],
            "miss_floor": l2_b_reuse_miss_floor,
            "miss_at_nominal_capacity": l2_b_reuse_miss_at_capacity,
            "miss_at_twice_nominal_capacity": l2_b_reuse_miss_ceiling,
        }
    return payload, service_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service_probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--machine-id")
    parser.add_argument("--backend-n-tile", type=int, default=8)
    parser.add_argument("--training-profile", type=Path)
    parser.add_argument("--train-routes", type=parse_int_set, default=parse_int_set("12,192,2040"))
    parser.add_argument("--train-threads", type=parse_int_set, default=parse_int_set("1,4,16,48"))
    parser.add_argument("--l2-effective-fraction", type=float, default=0.75)
    parser.add_argument("--llc-effective-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--l2-b-reuse-effective-fraction",
        type=float,
        help="private-L2 fraction below which repeated packed-B scans reach the miss floor",
    )
    parser.add_argument(
        "--l2-b-reuse-miss-floor",
        type=float,
        default=0.0,
        help="packed-B repeated-scan L2 miss floor from an independent retention probe",
    )
    parser.add_argument(
        "--l2-b-reuse-miss-at-capacity",
        type=float,
        default=1.0,
        help="packed-B repeated-scan miss fraction at the nominal private-L2 capacity",
    )
    parser.add_argument(
        "--l2-b-reuse-miss-ceiling",
        type=float,
        default=1.0,
        help="packed-B repeated-scan miss fraction at twice the nominal private-L2 capacity",
    )
    parser.add_argument("--relative-uncertainty", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe = json.loads(args.service_probe.read_text(encoding="utf-8"))
    machine_id = args.machine_id or f"{probe['machine']['id']}-numa0-sve-jit-thin-v1"
    calibration, service_fit = build_calibration(
        probe,
        machine_id=machine_id,
        l2_effective_fraction=args.l2_effective_fraction,
        llc_effective_fraction=args.llc_effective_fraction,
        l2_b_reuse_effective_fraction=args.l2_b_reuse_effective_fraction,
        l2_b_reuse_miss_floor=args.l2_b_reuse_miss_floor,
        l2_b_reuse_miss_at_capacity=args.l2_b_reuse_miss_at_capacity,
        l2_b_reuse_miss_ceiling=args.l2_b_reuse_miss_ceiling,
        relative_uncertainty=args.relative_uncertainty,
        backend_n_tile=args.backend_n_tile,
    )
    residual_fit = None
    if args.training_profile is not None:
        profile = json.loads(args.training_profile.read_text(encoding="utf-8"))
        residual_fit = fit_operator_residuals(
            calibration,
            profile,
            train_routes=args.train_routes,
            train_threads=args.train_threads,
        )
        calibration["overheads"]["expert_fixed_ns"] = residual_fit["expert_fixed_ns"]
        calibration["overheads"]["route_ns"] = residual_fit["route_ns"]
        calibration["stage_scales"] = {
            "w13": residual_fit["stage_scale"],
            "w2": residual_fit["stage_scale"],
        }
        calibration["provenance"]["isolated_residual_training"] = {
            "routes": residual_fit["train_routes"],
            "threads": residual_fit["train_threads"],
            "points": len(residual_fit["rows"]),
        }
    AnalyticMachineCalibration.from_dict(calibration)

    report = {
        "kind": "moe_analytic_thin_calibration_report",
        "machine_id": machine_id,
        "service_curve_fit": service_fit,
        "operator_residual_fit": residual_fit,
        "contention_measurements_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if args.report is not None:
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
