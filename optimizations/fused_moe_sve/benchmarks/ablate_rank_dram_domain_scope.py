#!/usr/bin/env python3
"""Ablate rank DRAM dilation versus domain-only injection on locked DAGs.

This diagnostic does not change production formulas. Rank DRAM is disabled by
returning infinite capacity on the first ``service_rate("dram_bytes")`` call in
each placed-phase allocation, which is the rank-wide allocator. Later calls
keep the original ``C_R(n_d)`` used by domain injection. Inflating the
``dram_bytes`` curve itself would also lift the domain cap and is forbidden.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(COST_MODEL_DIR)]

from analytic_model import (  # noqa: E402
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    DramDomainInjectionCalibration,
)
from optimizations.fused_moe_sve.benchmarks.fit_absolute_pressure_calibration import (  # noqa: E402
    _assert_exclusive_inputs,
    _evaluate_prediction,
    _mode_bridges,
    _phase_reaccounted,
    _predict_spans,
    _read_pressure_artifact,
    _read_training_artifact,
    _sha256,
    _trial,
    absolute_rows,
    contrast_rows,
)


REMOTE_NEAR_ZERO_MS = 0.08
LOCAL_PLATEAU_MS = 0.05
CONTRAST_PRESENT_MS = 0.10
COUNT_HEAD = (1, 2, 4, 8, 15)


@dataclass(frozen=True)
class ScopeArm:
    name: str
    disable_rank_dram: bool
    capacity_scale: float | None
    traffic_multiplier: float


DIAGNOSTIC_ARMS = (
    ScopeArm("rank_only", False, None, 1.0),
    ScopeArm("no_rank_dram", True, None, 1.0),
    ScopeArm("domain_only_beta0.20", True, 0.20, 1.0),
    ScopeArm("domain_only_beta0.50", True, 0.50, 1.0),
    ScopeArm("domain_only_beta0.78", True, 0.78, 1.0),
    ScopeArm("domain_only_beta1.00", True, 1.00, 1.0),
    ScopeArm("rank_plus_domain_rejected_joint", False, 0.78, 0.25),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--phase-fit", type=Path, required=True)
    parser.add_argument("--pressure-fit", type=Path, required=True)
    parser.add_argument("--pressure-validation", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


@contextmanager
def disable_rank_dram_dilation() -> Iterator[None]:
    """Keep domain ``C_R(n_d)`` while removing rank-wide DRAM dilation."""

    original_placed = AnalyticMoeCostModel._active_phase_state_placed
    original_service = AnalyticMachineCalibration.service_rate

    def wrapped_placed(self, current, task_cpu_ids):
        rank_dram_consumed = False

        def service_rate(
            calibration,
            resource,
            active_threads,
            *,
            active_cpu_ids=None,
            llc_domain_threads=None,
        ):
            nonlocal rank_dram_consumed
            if resource == "dram_bytes" and not rank_dram_consumed:
                rank_dram_consumed = True
                return math.inf
            return original_service(
                calibration,
                resource,
                active_threads,
                active_cpu_ids=active_cpu_ids,
                llc_domain_threads=llc_domain_threads,
            )

        AnalyticMachineCalibration.service_rate = service_rate
        try:
            return original_placed(self, current, task_cpu_ids)
        finally:
            AnalyticMachineCalibration.service_rate = original_service

    AnalyticMoeCostModel._active_phase_state_placed = wrapped_placed
    try:
        yield
    finally:
        AnalyticMoeCostModel._active_phase_state_placed = original_placed
        AnalyticMachineCalibration.service_rate = original_service


def apply_arm(base: AnalyticMachineCalibration, arm: ScopeArm) -> AnalyticMachineCalibration:
    if arm.capacity_scale is None:
        return replace(
            base,
            dram_domain_injection=DramDomainInjectionCalibration(),
            gather_pressure=replace(
                base.gather_pressure,
                effective_traffic_multiplier=arm.traffic_multiplier,
            ),
        )
    return _trial(
        base,
        capacity_scale=arm.capacity_scale,
        traffic_multiplier=arm.traffic_multiplier,
    )


def predict_arm_spans(
    base: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
    arm: ScopeArm,
) -> dict[str, float]:
    calibration = apply_arm(base, arm)
    if arm.disable_rank_dram:
        with disable_rank_dram_dilation():
            return _predict_spans(calibration, bridges)
    return _predict_spans(calibration, bridges)


def _mean_abs(values: dict[str, float], prefix: str) -> float:
    selected = [abs(value) for name, value in values.items() if name.startswith(prefix)]
    if not selected:
        raise ValueError(f"no rows start with {prefix!r}")
    return sum(selected) / len(selected)


def count_curve(predicted_absolute: dict[str, float], placement: str, phase: str) -> dict[str, float]:
    return {
        f"n{count}": predicted_absolute[f"{placement}_{phase}_n{count}"]
        for count in COUNT_HEAD
        if f"{placement}_{phase}_n{count}" in predicted_absolute
    }


def arm_findings(
    predicted_absolute: dict[str, float],
    predicted_contrast: dict[str, float],
) -> dict[str, object]:
    same_n4 = predicted_absolute.get("same_llc_head_n4")
    same_n15 = predicted_absolute.get("same_llc_head_n15")
    cross_n15 = predicted_absolute.get("cross_llc_head_n15")
    contrast_n4 = predicted_contrast.get("same_minus_cross_head_n4")
    plateau = (
        same_n4 is not None
        and same_n15 is not None
        and abs(same_n15 - same_n4) <= LOCAL_PLATEAU_MS
    )
    remote_near_zero = cross_n15 is not None and abs(cross_n15) <= REMOTE_NEAR_ZERO_MS
    locality = contrast_n4 is not None and contrast_n4 >= CONTRAST_PRESENT_MS
    return {
        "same_llc_head_n4_ms": same_n4,
        "same_llc_head_n15_ms": same_n15,
        "cross_llc_head_n15_ms": cross_n15,
        "same_minus_cross_head_n4_ms": contrast_n4,
        "n15_minus_n4_same_head_ms": (
            None if same_n4 is None or same_n15 is None else same_n15 - same_n4
        ),
        "local_plateau_at_n4": plateau,
        "remote_near_zero": remote_near_zero,
        "locality_contrast_at_n4": locality,
    }


def decide_scope(arm_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    rank_only = arm_reports["rank_only"]["findings"]
    no_rank = arm_reports["no_rank_dram"]["findings"]
    domain_candidates = [
        name
        for name in arm_reports
        if name.startswith("domain_only_")
        and arm_reports[name]["findings"]["remote_near_zero"]
        and arm_reports[name]["findings"]["locality_contrast_at_n4"]
    ]
    saturating = [
        name
        for name in domain_candidates
        if arm_reports[name]["findings"]["local_plateau_at_n4"]
    ]
    rank_explains_remote = (
        not rank_only["remote_near_zero"] and no_rank["remote_near_zero"]
    )
    rank_cross = rank_only["cross_llc_head_n15_ms"]
    no_rank_cross = no_rank["cross_llc_head_n15_ms"]
    rank_fraction = None
    if rank_cross not in (None, 0.0) and no_rank_cross is not None:
        rank_fraction = (rank_cross - no_rank_cross) / rank_cross
    leftover_local = no_rank["same_llc_head_n15_ms"]
    single_resource = bool(saturating) and rank_explains_remote
    return {
        "rank_dram_explains_remote_common_mode": rank_explains_remote,
        "rank_dram_fraction_of_remote_n15": rank_fraction,
        "no_rank_dram_local_n15_ms": leftover_local,
        "no_rank_dram_remote_n15_ms": no_rank_cross,
        "domain_only_restores_locality_without_remote": domain_candidates,
        "domain_only_also_saturates_at_n4": saturating,
        "single_missing_resource": single_resource,
        "add_default_off_structure": single_resource,
        "reason": (
            "domain-local DRAM without rank sharing matches remote-zero, local "
            "contrast, and the n≈4 plateau"
            if single_resource
            else "rank DRAM is only part of the remote common mode; leftover "
            "LLC/L2 dilation and the n≈4 plateau remain, so do not add a new "
            "structure"
        ),
    }


def _compact_absolute(predicted_absolute: dict[str, float]) -> dict[str, float]:
    names: list[str] = []
    for phase in ("head", "after_1"):
        for count in COUNT_HEAD:
            names.append(f"same_llc_{phase}_n{count}")
            names.append(f"cross_llc_{phase}_n{count}")
        for count in (8, 15):
            names.append(f"split_{phase}_n{count}")
    return {name: predicted_absolute[name] for name in names if name in predicted_absolute}


def main() -> int:
    args = parse_args()
    phase_fit, phase_sha = _read_training_artifact(args.phase_fit)
    pressure_fit, fit_sha = _read_pressure_artifact(args.pressure_fit)
    pressure_validation, validation_sha = _read_pressure_artifact(args.pressure_validation)
    base_sha = _sha256(args.base_calibration)
    _assert_exclusive_inputs(phase_sha, fit_sha, validation_sha, base_sha)

    base = AnalyticMachineCalibration.from_path(args.base_calibration)
    reaccounted = _phase_reaccounted(base, phase_fit)
    bridges = _mode_bridges(reaccounted)
    measured_fit_abs = absolute_rows(pressure_fit)
    measured_fit_contrast = contrast_rows(pressure_fit)
    measured_val_abs = absolute_rows(pressure_validation)
    measured_val_contrast = contrast_rows(pressure_validation)

    arm_reports: dict[str, dict[str, object]] = {}
    for arm in DIAGNOSTIC_ARMS:
        predicted = predict_arm_spans(reaccounted, bridges, arm)
        fit_eval = _evaluate_prediction(measured_fit_abs, measured_fit_contrast, predicted)
        validation_eval = _evaluate_prediction(
            measured_val_abs,
            measured_val_contrast,
            predicted,
        )
        predicted_absolute = fit_eval["predicted_absolute_ms"]
        predicted_contrast = fit_eval["predicted_contrast_ms"]
        arm_reports[arm.name] = {
            "arm": {
                "disable_rank_dram": arm.disable_rank_dram,
                "capacity_scale": arm.capacity_scale,
                "traffic_multiplier": arm.traffic_multiplier,
                "traffic_multiplier_meaning": "effective_gather_stream_coupling_not_byte_ratio",
            },
            "fit": {
                "loss": fit_eval["loss"],
                "predicted_absolute_ms": _compact_absolute(predicted_absolute),
                "predicted_contrast_ms": predicted_contrast,
                "absolute_mae_cross_llc_ms": _mean_abs(fit_eval["absolute_residual_ms"], "cross_llc_"),
                "absolute_mae_same_llc_ms": _mean_abs(fit_eval["absolute_residual_ms"], "same_llc_"),
            },
            "validation": {
                "loss": validation_eval["loss"],
                "absolute_mae_cross_llc_ms": _mean_abs(
                    validation_eval["absolute_residual_ms"],
                    "cross_llc_",
                ),
                "absolute_mae_same_llc_ms": _mean_abs(
                    validation_eval["absolute_residual_ms"],
                    "same_llc_",
                ),
            },
            "count_curve": {
                "same_llc_head": count_curve(predicted_absolute, "same_llc", "head"),
                "cross_llc_head": count_curve(predicted_absolute, "cross_llc", "head"),
                "same_llc_after_1": count_curve(predicted_absolute, "same_llc", "after_1"),
                "cross_llc_after_1": count_curve(predicted_absolute, "cross_llc", "after_1"),
            },
            "findings": arm_findings(predicted_absolute, predicted_contrast),
        }

    report = {
        "kind": "moe_rank_dram_domain_scope_ablation",
        "artifact_role": "offline_scope_ablation_no_holdout",
        "holdout_read": False,
        "replace_frozen_v8": False,
        "open_vnd_lns": False,
        "inputs": {
            "base_calibration": {"path": str(args.base_calibration), "sha256": base_sha},
            "phase_fit": {"path": str(args.phase_fit), "sha256": phase_sha},
            "pressure_fit": {"path": str(args.pressure_fit), "sha256": fit_sha},
            "pressure_validation": {
                "path": str(args.pressure_validation),
                "sha256": validation_sha,
            },
        },
        "method": {
            "rank_dram_disabled_by": (
                "first_service_rate_dram_bytes_in_placed_allocator_returns_inf"
            ),
            "domain_cap_keeps_original_cr": True,
            "phase_floor_scale_refit": False,
            "parameter_search": False,
            "thresholds_ms": {
                "remote_near_zero": REMOTE_NEAR_ZERO_MS,
                "local_plateau": LOCAL_PLATEAU_MS,
                "contrast_present": CONTRAST_PRESENT_MS,
            },
        },
        "measured_absolute_ms": {
            "fit": {
                name: measured_fit_abs[name]
                for name in (
                    "same_llc_head_n1",
                    "same_llc_head_n4",
                    "same_llc_head_n15",
                    "cross_llc_head_n1",
                    "cross_llc_head_n4",
                    "cross_llc_head_n15",
                )
            }
        },
        "arms": arm_reports,
        "decision": decide_scope(arm_reports),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_report": str(args.output_report),
                **report["decision"],
                "rank_only_cross_n15_ms": arm_reports["rank_only"]["findings"][
                    "cross_llc_head_n15_ms"
                ],
                "no_rank_dram_cross_n15_ms": arm_reports["no_rank_dram"]["findings"][
                    "cross_llc_head_n15_ms"
                ],
                "no_rank_dram_same_n15_ms": arm_reports["no_rank_dram"]["findings"][
                    "same_llc_head_n15_ms"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
