#!/usr/bin/env python3
"""Jointly fit domain injection and gather coupling from absolute pressure."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(COST_MODEL_DIR)]

from analytic_model import (  # noqa: E402
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    DramDomainInjectionCalibration,
    NarrowTeamContentionCorrection,
    WideTeamPressureCalibration,
)
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (  # noqa: E402
    OUTPUT_KIND,
    SCHEMA_VERSION,
    SPLIT_AGGRESSOR_COUNTS,
    enumerate_probe_modes,
    mode_name,
    parse_mode,
    _build_bridge,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _model_target_span_ms,
)
from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (  # noqa: E402
    LOCKED_HOLDOUT_SHA256,
    _fit_phase_calibration,
    _read_training_artifact,
    _sha256,
    _validate_phase_artifact,
)


RANK_CPU_IDS = tuple(range(240, 320))
HIDDEN = 4096
INTERMEDIATE = 512
EXPERTS = 17
REJECTED_CAPACITY_SCALE = 0.787
REJECTED_TRAFFIC_MULTIPLIER = 22.26
PRESSURE_KIND = OUTPUT_KIND
COARSE_CAPACITY = tuple(round(0.20 + step * 0.05, 2) for step in range(37))
COARSE_MULTIPLIER = tuple(round(0.25 + step * 0.25, 2) for step in range(32))
FINE_CAPACITY_HALF_WIDTH = 0.15
FINE_CAPACITY_STEP = 0.01
FINE_MULTIPLIER_HALF_WIDTH = 1.0
FINE_MULTIPLIER_STEP = 0.05
NEAR_LOSS_RATIO = 1.05
BOUNDARY_EPS = 1e-9
SYSTEMATIC_MS = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--phase-fit", type=Path, required=True)
    parser.add_argument("--pressure-fit", type=Path, required=True)
    parser.add_argument("--pressure-validation", type=Path, required=True)
    parser.add_argument("--output-calibration", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _validate_pressure_artifact(payload: dict) -> None:
    if payload.get("kind") != PRESSURE_KIND:
        raise ValueError(f"pressure artifact has the wrong kind: {payload.get('kind')!r}")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("pressure artifact has the wrong schema_version")
    modes = payload.get("modes", {})
    if "isolated_head" not in modes or "isolated_after_1" not in modes:
        raise ValueError("pressure artifact is missing isolated controls")
    samples = modes["isolated_head"]["target_span"]["samples_ms"]
    if len(samples) != 31:
        raise ValueError("pressure artifact must contain 31 paired samples")


def _read_pressure_artifact(path: Path) -> tuple[dict, str]:
    payload, digest = _read_training_artifact(path)
    _validate_pressure_artifact(payload)
    return payload, digest


def _assert_exclusive_inputs(*digests: str) -> None:
    unique = set(digests)
    if len(unique) != len(digests):
        raise ValueError("fit, validation, and phase artifacts must have distinct SHA256 values")
    overlap = unique & LOCKED_HOLDOUT_SHA256
    if overlap:
        raise ValueError(f"training input overlaps locked holdout SHA256: {sorted(overlap)}")


def _span(payload: dict, mode: str) -> float:
    return float(payload["modes"][mode]["target_span"]["median_ms"])


def absolute_rows(payload: dict) -> dict[str, float]:
    rows: dict[str, float] = {}
    for mode in payload["modes"]:
        _, phase, count = parse_mode(mode)
        if count == 0:
            continue
        rows[mode] = _span(payload, mode) - _span(payload, f"isolated_{phase}")
    return rows


def contrast_rows(payload: dict) -> dict[str, float]:
    counts = sorted(
        {
            parse_mode(mode)[2]
            for mode in payload["modes"]
            if parse_mode(mode)[0] == "same_llc"
        }
    )
    rows: dict[str, float] = {}
    for phase in ("head", "after_1"):
        for count in counts:
            same = mode_name("same_llc", phase, count)
            cross = mode_name("cross_llc", phase, count)
            if same in payload["modes"] and cross in payload["modes"]:
                rows[f"same_minus_cross_{phase}_n{count}"] = _span(payload, same) - _span(
                    payload, cross
                )
    return rows


def joint_loss(
    measured_absolute: dict[str, float],
    predicted_absolute: dict[str, float],
    measured_contrast: dict[str, float],
    predicted_contrast: dict[str, float],
) -> dict[str, float]:
    abs_errors = [
        abs(predicted_absolute[name] - measured_absolute[name])
        for name in measured_absolute
    ]
    contrast_errors = [
        abs(predicted_contrast[name] - measured_contrast[name])
        for name in measured_contrast
    ]
    if not abs_errors or not contrast_errors:
        raise ValueError("joint loss requires both absolute and contrast rows")
    absolute_mae = statistics.fmean(abs_errors)
    contrast_mae = statistics.fmean(contrast_errors)
    return {
        "absolute_mae_ms": absolute_mae,
        "contrast_mae_ms": contrast_mae,
        "joint_mae_ms": absolute_mae + contrast_mae,
    }


def _inclusive_grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    if step <= 0.0 or stop < start:
        raise ValueError("grid requires start <= stop and positive step")
    count = int(round((stop - start) / step))
    return tuple(round(start + index * step, 10) for index in range(count + 1))


def identifiability_report(
    best: tuple[float, float, float],
    points: list[tuple[float, float, float]],
    *,
    capacity_bounds: tuple[float, float],
    multiplier_bounds: tuple[float, float],
) -> dict[str, object]:
    loss, capacity, multiplier = best
    on_boundary = (
        capacity <= capacity_bounds[0] + BOUNDARY_EPS
        or capacity >= capacity_bounds[1] - BOUNDARY_EPS
        or multiplier <= multiplier_bounds[0] + BOUNDARY_EPS
        or multiplier >= multiplier_bounds[1] - BOUNDARY_EPS
    )
    near = [point for point in points if point[0] <= NEAR_LOSS_RATIO * loss]
    capacities = [point[1] for point in near]
    multipliers = [point[2] for point in near]
    capacity_ratio = max(capacities) / min(capacities)
    multiplier_ratio = max(multipliers) / min(multipliers)
    degenerate = capacity_ratio >= 2.0 or multiplier_ratio >= 2.0
    neighbors = [
        point
        for point in points
        if point != best
        and abs(point[1] - capacity) <= FINE_CAPACITY_STEP + BOUNDARY_EPS
        and abs(point[2] - multiplier) <= FINE_MULTIPLIER_STEP + BOUNDARY_EPS
    ]
    curved = bool(neighbors) and all(point[0] > loss + BOUNDARY_EPS for point in neighbors)
    identifiable = not on_boundary and not degenerate and curved
    return {
        "identifiable": identifiable,
        "on_search_boundary": on_boundary,
        "near_optima_count": len(near),
        "capacity_scale_span_ratio": capacity_ratio,
        "traffic_multiplier_span_ratio": multiplier_ratio,
        "degenerate_flat_valley": degenerate,
        "local_curvature": curved,
        "rejected_contrast_only_params": {
            "capacity_scale": abs(capacity - REJECTED_CAPACITY_SCALE) < 0.005
            and abs(multiplier - REJECTED_TRAFFIC_MULTIPLIER) < 0.05,
            "note": "effective coupling is not a measured byte ratio",
        },
    }


def signed_bias(residuals: list[float]) -> dict[str, object]:
    if not residuals:
        raise ValueError("signed bias requires residuals")
    mean = statistics.fmean(residuals)
    all_nonneg = all(value >= 0.0 for value in residuals)
    all_nonpos = all(value <= 0.0 for value in residuals)
    return {
        "mean_ms": mean,
        "all_same_sign": all_nonneg or all_nonpos,
        "systematic": (all_nonneg or all_nonpos) and abs(mean) >= SYSTEMATIC_MS,
    }


def _cost_model(calibration: AnalyticMachineCalibration) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        global_experts=EXPERTS,
        local_experts=EXPERTS,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )


def _trial(
    base: AnalyticMachineCalibration,
    *,
    capacity_scale: float,
    traffic_multiplier: float,
) -> AnalyticMachineCalibration:
    return replace(
        base,
        dram_domain_injection=DramDomainInjectionCalibration(
            enabled=True,
            capacity_scale=capacity_scale,
        ),
        gather_pressure=replace(
            base.gather_pressure,
            effective_traffic_multiplier=traffic_multiplier,
        ),
    )


def _mode_bridges(base: AnalyticMachineCalibration) -> dict[str, tuple[dict[str, object], dict[int, int]]]:
    model = _cost_model(base)
    counts = tuple(sorted({0, 1, 2, 4, 8, 15}))
    return {
        mode: _build_bridge(model, thread_cpu_ids=RANK_CPU_IDS, mode=mode)
        for mode in enumerate_probe_modes(counts)
    }


def _predict_spans(
    calibration: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
) -> dict[str, float]:
    model = _cost_model(calibration)
    return {
        mode: _model_target_span_ms(model, bridge, routes)
        for mode, (bridge, routes) in bridges.items()
    }


def _families_from_spans(spans: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    payload = {
        "modes": {
            mode: {"target_span": {"median_ms": value}}
            for mode, value in spans.items()
        }
    }
    return absolute_rows(payload), contrast_rows(payload)


def _evaluate_prediction(
    measured_absolute: dict[str, float],
    measured_contrast: dict[str, float],
    predicted_spans: dict[str, float],
) -> dict[str, object]:
    predicted_absolute, predicted_contrast = _families_from_spans(predicted_spans)
    loss = joint_loss(
        measured_absolute,
        predicted_absolute,
        measured_contrast,
        predicted_contrast,
    )
    abs_residuals = {
        name: measured_absolute[name] - predicted_absolute[name]
        for name in measured_absolute
    }
    contrast_residuals = {
        name: measured_contrast[name] - predicted_contrast[name]
        for name in measured_contrast
    }
    return {
        "loss": loss,
        "predicted_absolute_ms": predicted_absolute,
        "predicted_contrast_ms": predicted_contrast,
        "absolute_residual_ms": abs_residuals,
        "contrast_residual_ms": contrast_residuals,
        "absolute_bias": signed_bias(list(abs_residuals.values())),
        "contrast_bias": signed_bias(list(contrast_residuals.values())),
        "split_looks_like_local": {
            name: abs(
                predicted_absolute[name]
                - predicted_absolute[name.replace("split_", "same_llc_")]
            )
            for name in predicted_absolute
            if name.startswith("split_") and name.replace("split_", "same_llc_") in predicted_absolute
        },
    }


def _search_grid(
    base: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
    measured_absolute: dict[str, float],
    measured_contrast: dict[str, float],
    capacities: tuple[float, ...],
    multipliers: tuple[float, ...],
) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
    points: list[tuple[float, float, float]] = []
    best: tuple[float, float, float] | None = None
    for multiplier in multipliers:
        for capacity in capacities:
            predicted = _predict_spans(
                _trial(base, capacity_scale=capacity, traffic_multiplier=multiplier),
                bridges,
            )
            predicted_absolute, predicted_contrast = _families_from_spans(predicted)
            loss = joint_loss(
                measured_absolute,
                predicted_absolute,
                measured_contrast,
                predicted_contrast,
            )["joint_mae_ms"]
            row = (loss, capacity, multiplier)
            points.append(row)
            if best is None or row < best:
                best = row
    if best is None:
        raise AssertionError("parameter grid produced no fit")
    return best, points


def _phase_reaccounted(base: AnalyticMachineCalibration, phase_payload: dict) -> AnalyticMachineCalibration:
    _validate_phase_artifact(phase_payload)
    gather, stage, _report = _fit_phase_calibration(base, phase_payload)
    return replace(
        base,
        overheads=replace(
            base.overheads,
            expert_fixed_ns=0.0,
            route_ns=0.0,
            by_width=(),
        ),
        wide_team_pressure=WideTeamPressureCalibration(),
        narrow_team_contention_correction=NarrowTeamContentionCorrection(),
        gather_pressure=gather,
        stage_phase_calibration=stage,
        dram_domain_injection=DramDomainInjectionCalibration(),
    )


def main() -> int:
    args = parse_args()
    phase_fit, phase_sha = _read_training_artifact(args.phase_fit)
    pressure_fit, fit_sha = _read_pressure_artifact(args.pressure_fit)
    pressure_validation, validation_sha = _read_pressure_artifact(args.pressure_validation)
    base_sha = _sha256(args.base_calibration)
    _assert_exclusive_inputs(phase_sha, fit_sha, validation_sha, base_sha)
    if fit_sha == validation_sha:
        raise ValueError("pressure fit and validation artifacts must be independent")

    base = AnalyticMachineCalibration.from_path(args.base_calibration)
    reaccounted = _phase_reaccounted(base, phase_fit)
    bridges = _mode_bridges(reaccounted)
    measured_fit_abs = absolute_rows(pressure_fit)
    measured_fit_contrast = contrast_rows(pressure_fit)
    measured_val_abs = absolute_rows(pressure_validation)
    measured_val_contrast = contrast_rows(pressure_validation)

    coarse_best, coarse_points = _search_grid(
        reaccounted,
        bridges,
        measured_fit_abs,
        measured_fit_contrast,
        COARSE_CAPACITY,
        COARSE_MULTIPLIER,
    )
    fine_capacities = _inclusive_grid(
        max(COARSE_CAPACITY[0], coarse_best[1] - FINE_CAPACITY_HALF_WIDTH),
        min(COARSE_CAPACITY[-1], coarse_best[1] + FINE_CAPACITY_HALF_WIDTH),
        FINE_CAPACITY_STEP,
    )
    fine_multipliers = _inclusive_grid(
        max(COARSE_MULTIPLIER[0], coarse_best[2] - FINE_MULTIPLIER_HALF_WIDTH),
        min(COARSE_MULTIPLIER[-1], coarse_best[2] + FINE_MULTIPLIER_HALF_WIDTH),
        FINE_MULTIPLIER_STEP,
    )
    best, fine_points = _search_grid(
        reaccounted,
        bridges,
        measured_fit_abs,
        measured_fit_contrast,
        fine_capacities,
        fine_multipliers,
    )
    _loss, capacity_scale, traffic_multiplier = best
    candidate = _trial(
        reaccounted,
        capacity_scale=capacity_scale,
        traffic_multiplier=traffic_multiplier,
    )
    predicted_fit = _predict_spans(candidate, bridges)
    predicted_validation = predicted_fit
    fit_eval = _evaluate_prediction(measured_fit_abs, measured_fit_contrast, predicted_fit)
    validation_eval = _evaluate_prediction(
        measured_val_abs,
        measured_val_contrast,
        predicted_validation,
    )
    identity = identifiability_report(
        best,
        [*coarse_points, *fine_points],
        capacity_bounds=(COARSE_CAPACITY[0], COARSE_CAPACITY[-1]),
        multiplier_bounds=(COARSE_MULTIPLIER[0], COARSE_MULTIPLIER[-1]),
    )
    freeze_ok = (
        identity["identifiable"]
        and not validation_eval["absolute_bias"]["systematic"]
        and not validation_eval["contrast_bias"]["systematic"]
        and not identity["rejected_contrast_only_params"]["capacity_scale"]
    )
    report = {
        "kind": "moe_absolute_pressure_calibration_report",
        "artifact_role": "fit_only_session2_validation_no_holdout",
        "holdout_read": False,
        "inputs": {
            "base_calibration": {"path": str(args.base_calibration), "sha256": base_sha},
            "phase_fit": {"path": str(args.phase_fit), "sha256": phase_sha},
            "pressure_fit": {"path": str(args.pressure_fit), "sha256": fit_sha},
            "pressure_validation": {
                "path": str(args.pressure_validation),
                "sha256": validation_sha,
            },
        },
        "fit_policy": {
            "phase_floor_scale_refit": False,
            "wide_team_pressure_reset": True,
            "narrow_team_contention_correction_reset": True,
            "whole_expert_total_used": False,
            "contrast_only_forbidden": True,
            "locked_holdout_sha256": sorted(LOCKED_HOLDOUT_SHA256),
            "split_counts": sorted(SPLIT_AGGRESSOR_COUNTS),
        },
        "parameters": {
            "capacity_scale": capacity_scale,
            "effective_traffic_multiplier": traffic_multiplier,
            "effective_traffic_multiplier_meaning": "effective_gather_stream_coupling_not_byte_ratio",
        },
        "search": {
            "coarse_best": {
                "joint_mae_ms": coarse_best[0],
                "capacity_scale": coarse_best[1],
                "effective_traffic_multiplier": coarse_best[2],
            },
            "fine_best": {
                "joint_mae_ms": best[0],
                "capacity_scale": capacity_scale,
                "effective_traffic_multiplier": traffic_multiplier,
            },
            "coarse_grid": {
                "capacity_scale": list(COARSE_CAPACITY),
                "effective_traffic_multiplier": list(COARSE_MULTIPLIER),
            },
        },
        "identifiability": identity,
        "fit": {
            "measured_absolute_ms": measured_fit_abs,
            "measured_contrast_ms": measured_fit_contrast,
            **fit_eval,
        },
        "validation": {
            "measured_absolute_ms": measured_val_abs,
            "measured_contrast_ms": measured_val_contrast,
            **validation_eval,
        },
        "decision": {
            "replace_frozen_v8": False,
            "open_vnd_lns": False,
            "read_holdout": False,
            "session2_validation_passed": freeze_ok,
            "reason": (
                "identifiable joint fit with no systematic session-2 bias"
                if freeze_ok
                else "reject freeze: identifiability or session-2 systematic residual failed"
            ),
        },
    }
    args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_calibration.write_text(
        json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "capacity_scale": capacity_scale,
                "effective_traffic_multiplier": traffic_multiplier,
                "fit_joint_mae_ms": fit_eval["loss"]["joint_mae_ms"],
                "fit_absolute_mae_ms": fit_eval["loss"]["absolute_mae_ms"],
                "fit_contrast_mae_ms": fit_eval["loss"]["contrast_mae_ms"],
                "validation_joint_mae_ms": validation_eval["loss"]["joint_mae_ms"],
                "identifiable": identity["identifiable"],
                "session2_validation_passed": freeze_ok,
                "output_calibration": str(args.output_calibration),
                "output_report": str(args.output_report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
