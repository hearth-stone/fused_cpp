#!/usr/bin/env python3
"""Ablate victim-asymmetric dilation on the locked absolute-pressure DAGs.

This diagnostic does not change production formulas. Symmetric sharing applies
the cohort DRAM/L2/LLC dilation to every concurrent task. Victim-asymmetric
rules either drop peer-byte inflation or skip transfer dilation for
compute-bound phases. No saturating additive tax is fitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(COST_MODEL_DIR)]

from analytic_model import (  # noqa: E402
    _SHARED_RESOURCES,
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    AnalyticPhase,
)
from optimizations.fused_moe_sve.benchmarks.ablate_rank_dram_domain_scope import (  # noqa: E402
    CONTRAST_PRESENT_MS,
    LOCAL_PLATEAU_MS,
    REMOTE_NEAR_ZERO_MS,
    _compact_absolute,
    _mean_abs,
    arm_findings,
    count_curve,
)
from optimizations.fused_moe_sve.benchmarks.ablate_rank_llc_domain_scope import (  # noqa: E402
    target_phase_resources,
)
from optimizations.fused_moe_sve.benchmarks.fit_absolute_pressure_calibration import (  # noqa: E402
    _assert_exclusive_inputs,
    _cost_model,
    _evaluate_prediction,
    _mode_bridges,
    _phase_reaccounted,
    _predict_spans,
    _read_pressure_artifact,
    _read_training_artifact,
    _sha256,
    absolute_rows,
    contrast_rows,
)


TRANSFER_RESOURCES = ("l2_bytes", "llc_bytes", "dram_bytes")
ASYMMETRIC_ARMS = ("symmetric", "compute_bound_skip", "same_llc_peers", "own_demand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--phase-fit", type=Path, required=True)
    parser.add_argument("--pressure-fit", type=Path, required=True)
    parser.add_argument("--pressure-validation", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _own_offered_rate(phase: AnalyticPhase, resource: str, spill: float) -> float:
    demand = phase.resource_demand(resource, spill)
    if demand <= 0.0:
        return 0.0
    times = phase.resource_times_ns(spill)
    return demand / max(phase.residual_scale * times[resource] * 1e-9, 1e-30)


def _finite_capacity(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return math.inf
    return float(value)


def _detail_capacity(details: dict[str, float | int | None], key: str = "capacity") -> float:
    return _finite_capacity(None if details[key] is None else float(details[key]))


def _task_domains(task_cpu_ids: tuple[int, ...], cpu_to_domain: dict[int, str]) -> tuple[str, ...]:
    return tuple(sorted({cpu_to_domain[cpu] for cpu in task_cpu_ids}))


def _shares_llc_domain(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return bool(set(left) & set(right))


def _isolated_transfer_bound(phase: AnalyticPhase, times: tuple[float, ...]) -> bool:
    gemm = times[_SHARED_RESOURCES.index("gemm_core_flops")]
    transfer = max(times[_SHARED_RESOURCES.index(resource)] for resource in TRANSFER_RESOURCES)
    return gemm < transfer


def rewrite_multipliers(
    model: AnalyticMoeCostModel,
    current: dict[int, AnalyticPhase],
    task_cpu_ids,
    result: tuple,
    mode: str,
) -> tuple:
    task_spill, provisional, pressures, multipliers, domain_details, team_pressure = result
    if mode == "symmetric":
        return result

    cpu_to_domain = {
        cpu: domain.domain_id
        for domain in model.calibration.llc_domains
        for cpu in domain.cpu_ids
    }
    domains_by_index = {
        index: _task_domains(tuple(task_cpu_ids[index][: current[index].active_threads]), cpu_to_domain)
        for index in current
    }
    rewritten = dict(multipliers)
    for index, phase in current.items():
        spill = task_spill[index]
        _, times = phase.resource_vectors(spill)
        scales: dict[str, float] = {}
        for resource in TRANSFER_RESOURCES:
            pressure = pressures.get(resource)
            scales[resource] = 1.0 if pressure is None else float(pressure.dilation)
        local_llc = max(
            (
                float(domain_details[domain_id]["dilation"] or 1.0)
                for domain_id in domains_by_index[index]
            ),
            default=1.0,
        )
        rank_llc = 1.0 if "llc_bytes" not in pressures else float(pressures["llc_bytes"].dilation)
        scales["llc_bytes"] = max(rank_llc, local_llc)

        if mode == "compute_bound_skip":
            if not _isolated_transfer_bound(phase, times):
                for resource in TRANSFER_RESOURCES:
                    scales[resource] = 1.0
        elif mode == "own_demand":
            for resource in TRANSFER_RESOURCES:
                if resource == "llc_bytes":
                    capacity = min(
                        (
                            _detail_capacity(domain_details[domain_id])
                            for domain_id in domains_by_index[index]
                        ),
                        default=math.inf,
                    )
                else:
                    pressure = pressures.get(resource)
                    capacity = math.inf if pressure is None else _finite_capacity(pressure.capacity)
                own = _own_offered_rate(phase, resource, spill)
                scales[resource] = max(1.0, own / capacity) if math.isfinite(capacity) and capacity > 0.0 else 1.0
        elif mode == "same_llc_peers":
            peer_indices = [
                other
                for other, other_phase in current.items()
                if other_phase.resource_demand("dram_bytes", task_spill[other]) > 0.0
                and _shares_llc_domain(domains_by_index[index], domains_by_index[other])
            ]
            dram_pressure = pressures.get("dram_bytes")
            dram_capacity = math.inf if dram_pressure is None else _finite_capacity(dram_pressure.capacity)
            dram_offered = sum(
                _own_offered_rate(current[other], "dram_bytes", task_spill[other]) for other in peer_indices
            )
            scales["dram_bytes"] = (
                max(1.0, dram_offered / dram_capacity)
                if math.isfinite(dram_capacity) and dram_capacity > 0.0
                else 1.0
            )
            scales["llc_bytes"] = local_llc
        else:
            raise ValueError(f"unsupported asymmetric mode {mode!r}")

        duration = phase._duration_from_resource_times(times, scales)
        rewritten[index] = max(1.0, duration / phase.base_ns * team_pressure[index])
    return task_spill, provisional, pressures, rewritten, domain_details, team_pressure


@contextmanager
def apply_asymmetric_mode(mode: str) -> Iterator[None]:
    if mode == "symmetric":
        yield
        return

    original_placed = AnalyticMoeCostModel._active_phase_state_placed

    def wrapped_placed(self, current, task_cpu_ids):
        result = original_placed(self, current, task_cpu_ids)
        return rewrite_multipliers(self, current, task_cpu_ids, result, mode)

    AnalyticMoeCostModel._active_phase_state_placed = wrapped_placed
    try:
        yield
    finally:
        AnalyticMoeCostModel._active_phase_state_placed = original_placed


def predict_asymmetric_spans(
    calibration: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
    mode: str,
) -> dict[str, float]:
    with apply_asymmetric_mode(mode):
        return _predict_spans(calibration, bridges)


def leftover_resource_snapshot(
    calibration: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
    mode: str,
    modes: tuple[str, ...] = ("same_llc_head_n15", "cross_llc_head_n15"),
) -> dict[str, list[dict[str, object]]]:
    model = _cost_model(calibration)
    snapshot: dict[str, list[dict[str, object]]] = {}
    with apply_asymmetric_mode(mode):
        for name in modes:
            bridge, routes = bridges[name]
            snapshot[name] = target_phase_resources(model, bridge, routes)
    return snapshot


def decide_asymmetric(arm_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    symmetric = arm_reports["symmetric"]["findings"]
    skip = arm_reports["compute_bound_skip"]["findings"]
    same_llc = arm_reports["same_llc_peers"]["findings"]
    own = arm_reports["own_demand"]["findings"]
    passing = [
        name
        for name, report in arm_reports.items()
        if report["findings"]["remote_near_zero"]
        and report["findings"]["locality_contrast_at_n4"]
        and report["findings"]["local_plateau_at_n4"]
    ]
    own_clears_peer_bytes = (
        own["remote_near_zero"]
        and own["same_llc_head_n15_ms"] is not None
        and abs(own["same_llc_head_n15_ms"]) <= REMOTE_NEAR_ZERO_MS
    )
    skip_matches_symmetric = (
        skip["same_llc_head_n15_ms"] is not None
        and symmetric["same_llc_head_n15_ms"] is not None
        and abs(skip["same_llc_head_n15_ms"] - symmetric["same_llc_head_n15_ms"])
        <= LOCAL_PLATEAU_MS
    )
    add_structure = bool(passing)
    if add_structure:
        reason = (
            "a victim-asymmetric rule matches remote-zero, local contrast, "
            "and the n≈4 plateau without a new fitted residual"
        )
    elif own_clears_peer_bytes:
        reason = (
            "own-demand dilation stops the 1-route victim inheriting 68-route "
            "GEMM DRAM/LLC pressure, so predicted local and remote taxes fall "
            "to isolated; the remaining hardware same-LLC tax is a saturating "
            "occupancy effect that existing service curves cannot identify"
        )
    else:
        reason = "victim-asymmetric dilation does not isolate one missing resource"
    return {
        "compute_bound_skip_matches_symmetric": skip_matches_symmetric,
        "same_llc_peers_zeros_remote_not_local": (
            same_llc["remote_near_zero"] and not same_llc["local_plateau_at_n4"]
        ),
        "own_demand_clears_peer_byte_inheritance": own_clears_peer_bytes,
        "single_missing_resource": add_structure,
        "add_default_off_structure": add_structure,
        "passing_arms": passing,
        "reason": reason,
    }


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
    for mode in ASYMMETRIC_ARMS:
        predicted = predict_asymmetric_spans(reaccounted, bridges, mode)
        fit_eval = _evaluate_prediction(measured_fit_abs, measured_fit_contrast, predicted)
        validation_eval = _evaluate_prediction(
            measured_val_abs,
            measured_val_contrast,
            predicted,
        )
        predicted_absolute = fit_eval["predicted_absolute_ms"]
        predicted_contrast = fit_eval["predicted_contrast_ms"]
        arm_reports[mode] = {
            "arm": {"mode": mode},
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
            "leftover_resource": leftover_resource_snapshot(reaccounted, bridges, mode),
        }

    report = {
        "kind": "moe_victim_asymmetric_dilation_ablation",
        "artifact_role": "offline_victim_asymmetric_ablation_no_holdout",
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
            "symmetric": "cohort_dram_l2_llc_dilation_applied_to_every_task",
            "compute_bound_skip": "isolated_gemm_ge_transfer_sets_transfer_scales_to_1",
            "same_llc_peers": "victim_dram_offered_from_same_llc_tasks_only_and_no_rank_llc",
            "own_demand": "per_task_dilation_is_own_offered_over_capacity",
            "parameter_search": False,
            "saturating_additive_tax": False,
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
        "decision": decide_asymmetric(arm_reports),
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
                "symmetric_same_n15_ms": arm_reports["symmetric"]["findings"]["same_llc_head_n15_ms"],
                "symmetric_cross_n15_ms": arm_reports["symmetric"]["findings"]["cross_llc_head_n15_ms"],
                "same_llc_peers_same_n15_ms": arm_reports["same_llc_peers"]["findings"][
                    "same_llc_head_n15_ms"
                ],
                "same_llc_peers_cross_n15_ms": arm_reports["same_llc_peers"]["findings"][
                    "cross_llc_head_n15_ms"
                ],
                "own_demand_same_n15_ms": arm_reports["own_demand"]["findings"]["same_llc_head_n15_ms"],
                "own_demand_cross_n15_ms": arm_reports["own_demand"]["findings"]["cross_llc_head_n15_ms"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
