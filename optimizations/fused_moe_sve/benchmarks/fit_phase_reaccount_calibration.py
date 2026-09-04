#!/usr/bin/env python3
"""Fit phase-separated CPU MoE calibration without reading locked holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(COST_MODEL_DIR)]

from analytic_model import (  # noqa: E402
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    DramDomainInjectionCalibration,
    GatherPressureCalibration,
    NarrowTeamContentionCorrection,
    StagePhaseCalibration,
    WideTeamPressureCalibration,
)
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (  # noqa: E402
    LEGACY_MODES as CROSS_LLC_MODES,
    _build_bridge as _build_cross_llc_bridge,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _model_target_span_ms,
)


LOCKED_HOLDOUT_ROUTES = frozenset({1, 2, 5, 6, 12})
LOCKED_HOLDOUT_SHA256 = frozenset(
    {
        "61c0e929ad7a575831f972bdbb2c690df243e028062cdbb10044e66d15867e34",
        "90f34a8498a2a3ad1abfdae69b21ff4b9ab97f08cef1192dcc62c23bf8c07d4f",
        "be6b83475360c330ba4e053c90f19fc162465228eef835661b692fac569ca1a1",
        "277d3896e5d04bafedbf931edba38c0ff29ea6a6b9ad542ad3ff5d78d261df74",
        "28cf59e56d6c281df3245fcc941e19885496ec6e7df9bb4b7f3634d59e1b4376",
        "81c195d34902e0d9002a59f85d0c8bccfc884a6395bacb22031aa69a5c01b0dd",
        "b23ece71ffecbbbc839747ac4be679b34c02c7998a3737764aeee63964ddb67f",
        "fe103f49f816d930b12c021e2ca5d8966bbf0283670c61f245f3464f5d3f92d3",
        "bba234ae99508af9fc9f7026b43150e8c6e09f50026049e53f4392a4d908a8d8",
    }
)
STAGE_KEYS = {
    "w13": "w13_fused_silu_packc",
    "w2": "w2_direct_route",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--phase-fit", type=Path, required=True)
    parser.add_argument("--phase-validation", type=Path, required=True)
    parser.add_argument("--cross-llc-fit", type=Path, required=True)
    parser.add_argument("--cross-llc-validation", type=Path, required=True)
    parser.add_argument("--output-calibration", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_training_artifact(path: Path) -> tuple[dict, str]:
    digest = _sha256(path)
    if digest in LOCKED_HOLDOUT_SHA256:
        raise ValueError(f"training input is a locked holdout artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def _validate_phase_artifact(payload: dict) -> None:
    if payload.get("kind") != "moe_isolated_phase_reaccount_fit":
        raise ValueError("phase artifact has the wrong kind")
    if payload.get("artifact_role") != "fit_only":
        raise ValueError("phase artifact must declare artifact_role=fit_only")
    routes = {int(point["routes"]) for point in payload.get("points", ())}
    overlap = sorted(routes & LOCKED_HOLDOUT_ROUTES)
    if overlap:
        raise ValueError(f"phase artifact overlaps locked holdout routes: {overlap}")


def _best_floor_scale(
    physical_ns: list[float],
    measured_ns: list[float],
    *,
    denominator: Callable[[float], float],
) -> tuple[float, float, float]:
    if len(physical_ns) != len(measured_ns) or not physical_ns:
        raise ValueError("floor/scale fit requires matching non-empty observations")
    floors = sorted({0.0, *measured_ns})
    scales = sorted({measured / physical for measured, physical in zip(measured_ns, physical_ns)})
    best: tuple[float, float, float] | None = None
    for floor in floors:
        for scale in scales:
            loss = statistics.fmean(
                abs(max(floor, scale * physical) - measured) / denominator(measured)
                for physical, measured in zip(physical_ns, measured_ns)
            )
            candidate = (loss, floor, scale)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise AssertionError("non-empty candidate grid produced no fit")
    return best[1], best[2], best[0]


def _phase_rows(payload: dict, threads: int) -> list[dict]:
    return sorted(
        (point for point in payload["points"] if int(point["threads"]) == threads),
        key=lambda point: int(point["routes"]),
    )


def _fit_phase_calibration(
    base: AnalyticMachineCalibration,
    payload: dict,
) -> tuple[GatherPressureCalibration, StagePhaseCalibration, dict]:
    _validate_phase_artifact(payload)
    widths = tuple(int(width) for width in payload["shape"]["widths"])
    model = AnalyticMoeCostModel(
        base,
        hidden_size=int(payload["shape"]["hidden"]),
        intermediate_size=int(payload["shape"]["intermediate"]),
        global_experts=int(payload["shape"]["experts"]),
        local_experts=int(payload["shape"]["experts"]),
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    gather_points = []
    stage_points = {"w13": [], "w2": []}
    report = {"gather": {}, "stages": {"w13": {}, "w2": {}}}
    for threads in widths:
        rows = _phase_rows(payload, threads)
        gather_x = [float(math.ceil(int(row["routes"]) / threads)) for row in rows]
        gather_y = [float(row["phases"]["gather_pack_a"]["median_ms"]) * 1.0e6 for row in rows]
        minimum, row_scale, loss = _best_floor_scale(
            gather_x,
            gather_y,
            denominator=lambda measured: max(measured, 20_000.0),
        )
        gather_points.append((threads, minimum, 0.0, row_scale))
        report["gather"][str(threads)] = {
            "minimum_ns": minimum,
            "fixed_ns": 0.0,
            "row_ns": row_scale,
            "fit_normalized_mae": loss,
        }
        for stage, trace_key in STAGE_KEYS.items():
            physical = [
                getattr(model.predict_expert(int(row["routes"]), threads), f"{stage}_ns")
                for row in rows
            ]
            measured = [float(row["phases"][trace_key]["median_ms"]) * 1.0e6 for row in rows]
            floor, scale, stage_loss = _best_floor_scale(
                physical,
                measured,
                denominator=lambda value: value,
            )
            stage_points[stage].append((threads, floor, scale))
            report["stages"][stage][str(threads)] = {
                "minimum_ns": floor,
                "scale": scale,
                "fit_mape": stage_loss,
            }
    return (
        GatherPressureCalibration(
            enabled=True,
            effective_traffic_multiplier=1.0,
            by_width=tuple(gather_points),
        ),
        StagePhaseCalibration(
            w13_by_width=tuple(stage_points["w13"]),
            w2_by_width=tuple(stage_points["w2"]),
        ),
        report,
    )


def _phase_validation_report(
    calibration: AnalyticMachineCalibration,
    payload: dict,
) -> dict:
    _validate_phase_artifact(payload)
    model = AnalyticMoeCostModel(
        calibration,
        hidden_size=int(payload["shape"]["hidden"]),
        intermediate_size=int(payload["shape"]["intermediate"]),
        global_experts=int(payload["shape"]["experts"]),
        local_experts=int(payload["shape"]["experts"]),
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    errors = {"gather": [], "w13": [], "w2": [], "total": []}
    rows = []
    for point in payload["points"]:
        routes = int(point["routes"])
        threads = int(point["threads"])
        prediction = model.predict_expert(routes, threads)
        predicted = {
            "gather": prediction.phases[0].base_ns,
            "w13": prediction.w13_ns,
            "w2": prediction.w2_ns,
            "total": prediction.total_ns,
        }
        measured = {
            "gather": float(point["phases"]["gather_pack_a"]["median_ms"]) * 1.0e6,
            "w13": float(point["phases"][STAGE_KEYS["w13"]]["median_ms"]) * 1.0e6,
            "w2": float(point["phases"][STAGE_KEYS["w2"]]["median_ms"]) * 1.0e6,
            "total": float(point["target_span"]["median_ms"]) * 1.0e6,
        }
        relative = {name: predicted[name] / measured[name] - 1.0 for name in errors}
        for name, value in relative.items():
            errors[name].append(abs(value))
        rows.append(
            {
                "routes": routes,
                "threads": threads,
                "predicted_ns": predicted,
                "measured_ns": measured,
                "relative_error": relative,
            }
        )
    return {
        "metrics": {
            name: {
                "mape": statistics.fmean(values),
                "max_absolute_relative_error": max(values),
            }
            for name, values in errors.items()
        },
        "rows": rows,
    }


def _cross_contrasts(payload: dict) -> dict[str, float]:
    return {
        "head": float(payload["comparisons"]["local_vs_remote_head"]["delta"]["median_ms"]),
        "after_1": float(
            payload["comparisons"]["local_vs_remote_after_1"]["delta"]["median_ms"]
        ),
    }


def _model_cross_contrasts(calibration: AnalyticMachineCalibration) -> dict[str, float]:
    model = AnalyticMoeCostModel(
        calibration,
        hidden_size=4096,
        intermediate_size=512,
        global_experts=17,
        local_experts=17,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    spans = {}
    for mode in CROSS_LLC_MODES:
        bridge, routes_by_expert = _build_cross_llc_bridge(
            model,
            thread_cpu_ids=tuple(range(240, 320)),
            mode=mode,
        )
        spans[mode] = _model_target_span_ms(model, bridge, routes_by_expert)
    return {
        "head": spans["local_head"] - spans["remote_head"],
        "after_1": spans["local_after_1"] - spans["remote_after_1"],
    }


def _fit_domain_and_gather_coupling(
    base: AnalyticMachineCalibration,
    fit_payload: dict,
    validation_payload: dict,
) -> tuple[AnalyticMachineCalibration, dict]:
    measured_fit = _cross_contrasts(fit_payload)
    measured_validation = _cross_contrasts(validation_payload)
    best: tuple[float, float, dict[str, float]] | None = None
    for step in range(200, 2001):
        capacity_scale = step / 1000.0
        candidate = replace(
            base,
            dram_domain_injection=DramDomainInjectionCalibration(
                enabled=True,
                capacity_scale=capacity_scale,
            ),
        )
        predicted = _model_cross_contrasts(candidate)
        loss = abs(predicted["after_1"] - measured_fit["after_1"])
        row = (loss, capacity_scale, predicted)
        if best is None or row < best:
            best = row
    if best is None:
        raise AssertionError("domain capacity grid produced no fit")
    capacity_scale = best[1]
    candidate = replace(
        base,
        dram_domain_injection=DramDomainInjectionCalibration(
            enabled=True,
            capacity_scale=capacity_scale,
        ),
    )
    before_coupling = _model_cross_contrasts(candidate)
    fit_head_residual = measured_fit["head"] - before_coupling["head"]
    head_delta = fit_payload["comparisons"]["local_vs_remote_head"]["delta"]
    head_p10 = float(head_delta["p10_ms"])
    head_p90 = float(head_delta["p90_ms"])
    coupling_needed = (
        abs(fit_head_residual) >= 0.02
        and not head_p10 <= before_coupling["head"] <= head_p90
    )
    traffic_multiplier = 1.0
    if coupling_needed:
        gather_best: tuple[float, float, dict[str, float]] | None = None
        for step in range(25, 2501):
            multiplier = step / 100.0
            gather = replace(
                candidate.gather_pressure,
                effective_traffic_multiplier=multiplier,
            )
            trial = replace(candidate, gather_pressure=gather)
            predicted = _model_cross_contrasts(trial)
            loss = abs(predicted["head"] - measured_fit["head"])
            row = (loss, multiplier, predicted)
            if gather_best is None or row < gather_best:
                gather_best = row
        if gather_best is None:
            raise AssertionError("gather coupling grid produced no fit")
        traffic_multiplier = gather_best[1]
        candidate = replace(
            candidate,
            gather_pressure=replace(
                candidate.gather_pressure,
                effective_traffic_multiplier=traffic_multiplier,
            ),
        )
    predicted_fit = _model_cross_contrasts(candidate)
    predicted_validation = predicted_fit
    return candidate, {
        "capacity_scale": capacity_scale,
        "gather_stream_coupling_needed": coupling_needed,
        "effective_traffic_multiplier": traffic_multiplier,
        "before_coupling": before_coupling,
        "fit_head_interval_ms": {"p10": head_p10, "p90": head_p90},
        "fit": {
            "measured_ms": measured_fit,
            "predicted_ms": predicted_fit,
            "residual_ms": {
                name: measured_fit[name] - predicted_fit[name]
                for name in measured_fit
            },
        },
        "validation": {
            "measured_ms": measured_validation,
            "predicted_ms": predicted_validation,
            "residual_ms": {
                name: measured_validation[name] - predicted_validation[name]
                for name in measured_validation
            },
        },
    }


def main() -> int:
    args = parse_args()
    phase_fit, phase_fit_sha = _read_training_artifact(args.phase_fit)
    phase_validation, phase_validation_sha = _read_training_artifact(args.phase_validation)
    cross_fit, cross_fit_sha = _read_training_artifact(args.cross_llc_fit)
    cross_validation, cross_validation_sha = _read_training_artifact(args.cross_llc_validation)
    base = AnalyticMachineCalibration.from_path(args.base_calibration)
    gather, stage, phase_fit_report = _fit_phase_calibration(base, phase_fit)
    reaccounted = replace(
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
    phase_fit_validation = _phase_validation_report(reaccounted, phase_fit)
    phase_repeat_validation = _phase_validation_report(reaccounted, phase_validation)
    candidate, cross_report = _fit_domain_and_gather_coupling(
        reaccounted,
        cross_fit,
        cross_validation,
    )
    report = {
        "kind": "moe_phase_reaccount_calibration_report",
        "artifact_role": "fit_only_candidate_freeze",
        "inputs": {
            "base_calibration": {"path": str(args.base_calibration), "sha256": _sha256(args.base_calibration)},
            "phase_fit": {"path": str(args.phase_fit), "sha256": phase_fit_sha},
            "phase_validation": {"path": str(args.phase_validation), "sha256": phase_validation_sha},
            "cross_llc_fit": {"path": str(args.cross_llc_fit), "sha256": cross_fit_sha},
            "cross_llc_validation": {"path": str(args.cross_llc_validation), "sha256": cross_validation_sha},
        },
        "fit_policy": {
            "whole_expert_total_used": False,
            "locked_holdout_routes": sorted(LOCKED_HOLDOUT_ROUTES),
            "locked_holdout_sha256": sorted(LOCKED_HOLDOUT_SHA256),
            "wide_team_pressure_reset": True,
            "narrow_team_contention_correction_reset": True,
        },
        "phase_fit": phase_fit_report,
        "phase_fit_evaluation": phase_fit_validation,
        "phase_validation": phase_repeat_validation,
        "cross_llc": cross_report,
        "candidate": {
            "analytic_model_schema_version": AnalyticMoeCostModel.schema_version,
            "calibration": candidate.to_dict(),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
