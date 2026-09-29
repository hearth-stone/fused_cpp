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
from typing import Mapping, Sequence

try:
    from analytic_model import (
        ANALYTIC_MACHINE_SCHEMA_VERSION,
        AnalyticMachineCalibration,
        SaturatingServiceCurve,
    )
    from validate_analytic_model import _model_from_profile
except ImportError:  # pragma: no cover - package-style import
    from .analytic_model import (
        ANALYTIC_MACHINE_SCHEMA_VERSION,
        AnalyticMachineCalibration,
        SaturatingServiceCurve,
    )
    from .validate_analytic_model import _model_from_profile


POWER_RESOURCES = {"gemm_core_flops", "matrix_flops", "l1_bytes", "l2_bytes"}
PIECEWISE_RESOURCES = {"llc_bytes", "dram_bytes"}


def parse_int_set(value: str) -> set[int]:
    result = {int(item) for item in value.split(",") if item.strip()}
    if not result or min(result) <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {value!r}")
    return result


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected DOMAIN_ID=PATH") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty DOMAIN_ID=PATH")
    return name, Path(path)


def _service_rows(probe: dict, resource: str) -> list[dict]:
    rows = sorted(probe["services"][resource]["rows"], key=lambda row: int(row["threads"]))
    if not rows or int(rows[0]["threads"]) != 1:
        raise ValueError(f"{resource} probe must contain a one-thread point")
    if any(float(row["aggregate_rate"]) <= 0.0 for row in rows):
        raise ValueError(f"{resource} rates must be positive")
    return rows


def _isotonic_rates(rates: Sequence[float]) -> list[float]:
    """Least-squares monotone projection using equal-weight PAVA."""
    blocks: list[list[float]] = []
    for rate in rates:
        blocks.append([float(rate), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right_sum, right_count = blocks.pop()
            left_sum, left_count = blocks.pop()
            blocks.append([left_sum + right_sum, left_count + right_count])
    projected = []
    for total, count in blocks:
        projected.extend([total / count] * int(count))
    return projected


def _piecewise_curve(rows: Sequence[dict]) -> tuple[dict, dict]:
    measured_rates = [float(row["aggregate_rate"]) for row in rows]
    projected_rates = _isotonic_rates(measured_rates)
    points = [
        {"threads": int(row["threads"]), "rate": rate}
        for row, rate in zip(rows, projected_rates)
    ]
    payload = {
        "single_thread_rate": projected_rates[0],
        "saturated_rate": projected_rates[-1],
        "saturation_threads": int(rows[-1]["threads"]),
        "curve": "piecewise_linear",
        "points": points,
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
    leave_one_out = []
    for omitted_index in range(1, len(rows) - 1):
        retained = list(rows[:omitted_index]) + list(rows[omitted_index + 1 :])
        retained_rates = _isotonic_rates([float(row["aggregate_rate"]) for row in retained])
        retained_payload = {
            "single_thread_rate": retained_rates[0],
            "saturated_rate": retained_rates[-1],
            "saturation_threads": int(retained[-1]["threads"]),
            "curve": "piecewise_linear",
            "points": [
                {"threads": int(row["threads"]), "rate": rate}
                for row, rate in zip(retained, retained_rates)
            ],
        }
        retained_curve = SaturatingServiceCurve.from_dict(retained_payload)
        omitted = rows[omitted_index]
        measured = float(omitted["aggregate_rate"])
        predicted = retained_curve.rate(int(omitted["threads"]))
        leave_one_out.append(
            {
                "threads": int(omitted["threads"]),
                "measured_rate": measured,
                "predicted_rate": predicted,
                "relative_error": predicted / measured - 1.0,
            }
        )
    return payload, {
        "curve": payload,
        "fit_kind": "isotonic_piecewise_interpolation",
        "evaluation": "in_sample_service_points",
        "mape": statistics.fmean(abs(row["relative_error"]) for row in residuals),
        "max_absolute_relative_error": max(abs(row["relative_error"]) for row in residuals),
        "rows": residuals,
        "leave_one_sampled_width_out": {
            "scope": "internal_widths_only; sampling-density diagnostic, not an independent rerun",
            "points": len(leave_one_out),
            "mape": (
                statistics.fmean(abs(row["relative_error"]) for row in leave_one_out)
                if leave_one_out
                else None
            ),
            "max_absolute_relative_error": (
                max(abs(row["relative_error"]) for row in leave_one_out)
                if leave_one_out
                else None
            ),
            "rows": leave_one_out,
        },
    }


def select_curve(probe: dict, resource: str) -> tuple[dict, dict]:
    rows = _service_rows(probe, resource)
    if resource in PIECEWISE_RESOURCES:
        return _piecewise_curve(rows)
    single = float(rows[0]["aggregate_rate"])
    if resource in POWER_RESOURCES or resource == "gemm_l2_flops":
        saturation_threads = int(rows[-1]["threads"])
        saturated = max(single, float(rows[-1]["aggregate_rate"]))
        curve_name = "power"
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


def _subset_service_probe(probe: dict, *, resource: str, cpu_ids: set[int]) -> dict | None:
    ordered_cpus = [int(cpu) for cpu in probe["machine"].get("cpu_ids", ())]
    if not ordered_cpus:
        return None
    rows = [
        row
        for row in _service_rows(probe, resource)
        if set(ordered_cpus[: int(row["threads"])]) <= cpu_ids
    ]
    if (
        not rows
        or int(rows[0]["threads"]) != 1
        or int(rows[-1]["threads"]) != len(cpu_ids)
    ):
        return None
    return {"services": {resource: {"rows": rows}}}


def _llc_domain_services(
    probe: dict,
    topology: dict,
    domain_probes: Mapping[str, dict],
) -> tuple[list[dict], dict]:
    domains = topology.get("llc_domains", ())
    if not domains:
        return [], {}
    curves: dict[str, tuple[dict, dict, str]] = {}
    for domain in domains:
        domain_id = str(domain["id"])
        domain_cpus = {int(cpu) for cpu in domain["cpu_ids"]}
        source = domain_probes.get(domain_id)
        source_name = f"explicit_domain_probe:{domain_id}"
        if source is not None:
            source_cpus = {int(cpu) for cpu in source["machine"].get("cpu_ids", ())}
            if source_cpus and not source_cpus <= domain_cpus:
                raise ValueError(
                    f"LLC domain probe {domain_id!r} contains CPUs outside its topology domain"
                )
        if source is None:
            source = _subset_service_probe(probe, resource="llc_bytes", cpu_ids=domain_cpus)
            source_name = "rank_probe_single_domain_prefix"
        if source is not None:
            curve, fit = select_curve(source, "llc_bytes")
            curves[domain_id] = (curve, fit, source_name)

    for domain in domains:
        domain_id = str(domain["id"])
        if domain_id in curves:
            continue
        signature = (len(domain["cpu_ids"]), int(domain["capacity_bytes"]))
        matches = [
            (source_domain, value)
            for source_domain, value in curves.items()
            if next(
                (
                    len(item["cpu_ids"]),
                    int(item["capacity_bytes"]),
                )
                for item in domains
                if str(item["id"]) == source_domain
            )
            == signature
        ]
        if not matches:
            raise ValueError(
                f"LLC domain {domain_id!r} has no service probe and no symmetric calibrated domain"
            )
        source_domain, (curve, fit, _) = matches[0]
        curves[domain_id] = (curve, fit, f"symmetric_clone:{source_domain}")

    payload = []
    report = {}
    for domain in domains:
        domain_id = str(domain["id"])
        curve, fit, source = curves[domain_id]
        payload.append(
            {
                "id": domain_id,
                "cpu_ids": [int(cpu) for cpu in domain["cpu_ids"]],
                "capacity_bytes": int(domain["capacity_bytes"]),
                "service": curve,
            }
        )
        report[domain_id] = {"source": source, **fit}
    return payload, report


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
    llc_domain_probes: Mapping[str, dict] | None = None,
    supported_widths: Sequence[int] | None = None,
    topology_override: dict | None = None,
) -> tuple[dict, dict]:
    if probe.get("kind") != "moe_analytic_service_probe":
        raise ValueError("input is not an analytical service probe")
    services: dict[str, dict] = {}
    service_fit: dict[str, dict] = {}
    for resource in sorted(POWER_RESOURCES | PIECEWISE_RESOURCES):
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
    measured_widths = sorted(int(row["threads"]) for row in probe["services"]["gemm_core_flops"]["rows"])
    probe_schema = int(probe.get("schema_version", 1))
    if supported_widths is None and probe_schema >= 2:
        raise ValueError("schema-v2 service probes require explicit supported_widths")
    widths = sorted({int(width) for width in (supported_widths or measured_widths)})
    cores_per_rank = int(probe["machine"]["cores_per_rank"])
    if not widths or widths[0] <= 0 or widths[-1] > cores_per_rank:
        raise ValueError("supported_widths must be positive and no larger than cores_per_rank")
    topology = topology_override if topology_override is not None else probe.get("topology", {})
    rank_cpu_ids = [int(cpu) for cpu in topology.get("rank_cpu_ids", probe["machine"].get("cpu_ids", ()))]
    if rank_cpu_ids and len(rank_cpu_ids) != cores_per_rank:
        raise ValueError("rank_cpu_ids count must match cores_per_rank")
    llc_domains, llc_domain_fit = _llc_domain_services(
        probe,
        topology,
        llc_domain_probes or {},
    )
    if llc_domains:
        detected_capacity = sum(int(domain["capacity_bytes"]) for domain in llc_domains)
        service_fit["llc_capacity"] = {
            "probe_llc_bytes_per_rank": int(caches["llc_bytes_per_rank"]),
            "topology_llc_bytes_per_rank": detected_capacity,
            "corrected_from_legacy_single_domain_value": (
                int(caches["llc_bytes_per_rank"]) != detected_capacity
            ),
        }
        service_fit["llc_domains"] = llc_domain_fit
        summed_domain_rate = sum(float(domain["service"]["saturated_rate"]) for domain in llc_domains)
        rank_llc_rate = float(services["llc_bytes"]["saturated_rate"])
        service_fit["llc_topology_composition"] = {
            "summed_domain_saturated_rate": summed_domain_rate,
            "rank_saturated_rate": rank_llc_rate,
            "rank_to_summed_domain_ratio": rank_llc_rate / summed_domain_rate,
            "composition": "min(sum(active_domain_rates), rank_saturated_rate)",
        }
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
        "schema_version": ANALYTIC_MACHINE_SCHEMA_VERSION,
        "kind": "moe_analytic_machine",
        "machine": {
            "id": machine_id,
            "cores_per_rank": cores_per_rank,
        },
        "kernel": {"backend_n_tile": int(backend_n_tile)},
        "caches": {
            "l1d_bytes_per_core": int(caches["l1d_bytes_per_core"]),
            "l2_bytes_per_core": int(caches["l2_bytes_per_core"]),
            "llc_bytes_per_rank": (
                sum(int(domain["capacity_bytes"]) for domain in llc_domains)
                if llc_domains
                else int(caches["llc_bytes_per_rank"])
            ),
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
            "shared_service_interpolation": {
                "resources": sorted(PIECEWISE_RESOURCES),
                "kind": "isotonic_piecewise_linear",
                "validation": "in_sample_only_until_held_out_widths_are_measured",
            },
            "llc_topology_model": (
                "explicit_domains_with_rank_fabric_cap"
                if llc_domains
                else "rank_aggregate_fallback"
            ),
            "topology_source": (
                "explicit_override"
                if topology_override is not None
                else ("service_probe" if topology else "absent")
            ),
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
    if rank_cpu_ids or llc_domains:
        payload["topology"] = {
            "rank_cpu_ids": rank_cpu_ids,
            "llc_domains": llc_domains,
            "dram_scope": str(topology.get("dram_scope", "numa_rank")),
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
    parser.add_argument(
        "--llc-domain-probe",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="DOMAIN_ID=PATH",
        help="optional single-LLC-domain service probe; repeat for heterogeneous domains",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        help="optional topology JSON for replaying a legacy probe without embedded LLC domains",
    )
    parser.add_argument(
        "--supported-widths",
        type=parse_int_set,
        help="planner-legal widths; required for schema-v2 probes",
    )
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
    machine_id = args.machine_id or f"{probe['machine']['id']}-rank-sve-jit-thin-v2"
    llc_domain_probes = {
        domain_id: json.loads(path.read_text(encoding="utf-8"))
        for domain_id, path in args.llc_domain_probe
    }
    if len(llc_domain_probes) != len(args.llc_domain_probe):
        raise ValueError("each --llc-domain-probe must use a unique domain id")
    topology_override = (
        json.loads(args.topology.read_text(encoding="utf-8"))
        if args.topology is not None
        else None
    )
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
        llc_domain_probes=llc_domain_probes,
        supported_widths=sorted(args.supported_widths) if args.supported_widths else None,
        topology_override=topology_override,
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
        "service_fit_scope": "in_sample_points; held-out widths required for predictive error",
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
