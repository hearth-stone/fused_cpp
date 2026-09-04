#!/usr/bin/env python3
"""Ablate rank LLC versus domain LLC on the locked absolute-pressure DAGs.

This diagnostic does not change production formulas. Rank DRAM stays disabled
from the previous scope ablation. Rank LLC is the ``service_rate("llc_bytes")``
call that passes every domain id; domain LLC is the one-key per-domain call.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(COST_MODEL_DIR)]

from analytic_model import AnalyticMachineCalibration, AnalyticMoeCostModel  # noqa: E402
from optimizations.fused_moe_sve.benchmarks.ablate_rank_dram_domain_scope import (  # noqa: E402
    CONTRAST_PRESENT_MS,
    LOCAL_PLATEAU_MS,
    REMOTE_NEAR_ZERO_MS,
    _compact_absolute,
    _mean_abs,
    arm_findings,
    count_curve,
)
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (  # noqa: E402
    TARGET_EXPERT,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _placed_tasks,
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


@dataclass(frozen=True)
class LlcScopeArm:
    name: str
    disable_rank_dram: bool
    disable_rank_llc: bool
    disable_all_llc: bool
    disable_l2: bool


LLC_ARMS = (
    LlcScopeArm("no_rank_dram", True, False, False, False),
    LlcScopeArm("domain_llc_only", True, True, False, False),
    LlcScopeArm("no_llc", True, True, True, False),
    LlcScopeArm("no_llc_no_l2", True, True, True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-calibration", type=Path, required=True)
    parser.add_argument("--phase-fit", type=Path, required=True)
    parser.add_argument("--pressure-fit", type=Path, required=True)
    parser.add_argument("--pressure-validation", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def is_rank_llc_service(llc_domain_threads: Mapping[str, int] | None) -> bool:
    """Rank LLC passes every domain id; domain LLC passes a single domain.

    Do not use DRAM's first-call rule. Domain LLC is invoked before rank LLC,
    so the first ``llc_bytes`` call is the one-key domain allocator.
    """

    return llc_domain_threads is not None and len(llc_domain_threads) > 1


@contextmanager
def disable_placed_dilations(
    *,
    rank_dram: bool = False,
    rank_llc: bool = False,
    all_llc: bool = False,
    l2: bool = False,
) -> Iterator[None]:
    if not any((rank_dram, rank_llc, all_llc, l2)):
        yield
        return

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
            if rank_dram and resource == "dram_bytes" and not rank_dram_consumed:
                rank_dram_consumed = True
                return math.inf
            if resource == "llc_bytes" and all_llc:
                return math.inf
            if resource == "llc_bytes" and rank_llc and is_rank_llc_service(llc_domain_threads):
                return math.inf
            if l2 and resource == "l2_bytes":
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


def target_phase_resources(
    model: AnalyticMoeCostModel,
    bridge: dict[str, object],
    routes: dict[int, int],
) -> list[dict[str, object]]:
    explanation = model.explain_dag_placed(_placed_tasks(bridge, routes))
    experts = [int(value) for value in bridge["task_expert_ids"]]
    target = str(experts.index(TARGET_EXPERT))
    rows: list[dict[str, object]] = []
    for event in explanation["events"]:
        if int(target) not in event["active_tasks"]:
            continue
        resources = event.get("resources", {})
        domain_details = event.get("llc_domains") or {}
        rows.append(
            {
                "phase": event["phases"][target],
                "kind": event["phase_kinds"][target],
                "phase_dilation": event["phase_dilation"][target],
                "llc_bytes_dilation": resources.get("llc_bytes", {}).get("dilation", 1.0),
                "l2_bytes_dilation": resources.get("l2_bytes", {}).get("dilation", 1.0),
                "dram_bytes_dilation": resources.get("dram_bytes", {}).get("dilation", 1.0),
                "llc_domain_dilations": {
                    domain_id: details.get("dilation")
                    for domain_id, details in domain_details.items()
                },
            }
        )
    return rows


def leftover_resource_snapshot(
    calibration: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
    arm: LlcScopeArm,
    modes: tuple[str, ...] = ("same_llc_head_n15", "cross_llc_head_n15"),
) -> dict[str, list[dict[str, object]]]:
    model = _cost_model(calibration)
    snapshot: dict[str, list[dict[str, object]]] = {}
    with disable_placed_dilations(
        rank_dram=arm.disable_rank_dram,
        rank_llc=arm.disable_rank_llc,
        all_llc=arm.disable_all_llc,
        l2=arm.disable_l2,
    ):
        for mode in modes:
            bridge, routes = bridges[mode]
            snapshot[mode] = target_phase_resources(model, bridge, routes)
    return snapshot


def predict_llc_arm_spans(
    base: AnalyticMachineCalibration,
    bridges: dict[str, tuple[dict[str, object], dict[int, int]]],
    arm: LlcScopeArm,
) -> dict[str, float]:
    with disable_placed_dilations(
        rank_dram=arm.disable_rank_dram,
        rank_llc=arm.disable_rank_llc,
        all_llc=arm.disable_all_llc,
        l2=arm.disable_l2,
    ):
        return _predict_spans(base, bridges)


def decide_llc_scope(arm_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    leftover = arm_reports["no_rank_dram"]["findings"]
    domain_llc = arm_reports["domain_llc_only"]["findings"]
    no_llc = arm_reports["no_llc"]["findings"]
    no_l2 = arm_reports["no_llc_no_l2"]["findings"]
    leftover_cross = leftover["cross_llc_head_n15_ms"]
    domain_cross = domain_llc["cross_llc_head_n15_ms"]
    fraction = None
    if leftover_cross not in (None, 0.0) and domain_cross is not None:
        fraction = (leftover_cross - domain_cross) / leftover_cross
    rank_llc_explains_remote = (
        not leftover["remote_near_zero"] and domain_llc["remote_near_zero"]
    )
    domain_llc_carries_local = (
        domain_llc["same_llc_head_n15_ms"] is not None
        and no_llc["same_llc_head_n15_ms"] is not None
        and domain_llc["same_llc_head_n15_ms"] - no_llc["same_llc_head_n15_ms"]
        > REMOTE_NEAR_ZERO_MS
    )
    victim_free_of_peer_cache = (
        no_l2["remote_near_zero"]
        and no_l2["same_llc_head_n15_ms"] is not None
        and abs(no_l2["same_llc_head_n15_ms"]) <= REMOTE_NEAR_ZERO_MS
    )
    add_structure = (
        domain_llc["remote_near_zero"]
        and domain_llc["locality_contrast_at_n4"]
        and domain_llc["local_plateau_at_n4"]
    )
    if add_structure:
        reason = (
            "domain LLC without rank sharing matches remote-zero, local contrast, "
            "and the n≈4 plateau"
        )
    elif victim_free_of_peer_cache:
        reason = (
            "after rank DRAM, rank LLC, domain LLC, and L2 are removed, the 1-route "
            "1T victim no longer inherits 68-route GEMM dilation; the missing "
            "mechanism is victim-asymmetric saturating same-LLC tax, not another "
            "shared-capacity scalar"
        )
    else:
        reason = (
            "rank LLC versus domain LLC does not isolate one missing resource; "
            "do not add a new structure"
        )
    return {
        "rank_llc_explains_remote_common_mode": rank_llc_explains_remote,
        "rank_llc_fraction_of_leftover_remote_n15": fraction,
        "domain_llc_carries_remaining_local": domain_llc_carries_local,
        "victim_free_of_peer_cache_after_no_llc_no_l2": victim_free_of_peer_cache,
        "single_missing_resource": add_structure,
        "add_default_off_structure": add_structure,
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
    for arm in LLC_ARMS:
        predicted = predict_llc_arm_spans(reaccounted, bridges, arm)
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
                "disable_rank_llc": arm.disable_rank_llc,
                "disable_all_llc": arm.disable_all_llc,
                "disable_l2": arm.disable_l2,
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
            "leftover_resource": leftover_resource_snapshot(reaccounted, bridges, arm),
        }

    report = {
        "kind": "moe_rank_llc_domain_scope_ablation",
        "artifact_role": "offline_llc_scope_ablation_no_holdout",
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
            "rank_llc_disabled_by": "service_rate_llc_bytes_with_all_domain_ids_returns_inf",
            "domain_llc_keeps_one_key_call": True,
            "rank_dram_disabled": True,
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
        "decision": decide_llc_scope(arm_reports),
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
                "no_rank_dram_cross_n15_ms": arm_reports["no_rank_dram"]["findings"][
                    "cross_llc_head_n15_ms"
                ],
                "domain_llc_only_cross_n15_ms": arm_reports["domain_llc_only"]["findings"][
                    "cross_llc_head_n15_ms"
                ],
                "domain_llc_only_same_n15_ms": arm_reports["domain_llc_only"]["findings"][
                    "same_llc_head_n15_ms"
                ],
                "no_llc_same_n15_ms": arm_reports["no_llc"]["findings"]["same_llc_head_n15_ms"],
                "no_llc_no_l2_same_n15_ms": arm_reports["no_llc_no_l2"]["findings"][
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
