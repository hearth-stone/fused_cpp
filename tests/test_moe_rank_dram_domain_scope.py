from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(COST_MODEL)]

from analytic_model import (  # noqa: E402
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    CacheCalibration,
    DramDomainInjectionCalibration,
    LlcDomainCalibration,
    SaturatingServiceCurve,
)
from optimizations.fused_moe_sve.benchmarks.ablate_rank_dram_domain_scope import (  # noqa: E402
    DIAGNOSTIC_ARMS,
    apply_arm,
    arm_findings,
    decide_scope,
    disable_rank_dram_dilation,
)
from optimizations.fused_moe_sve.benchmarks.fit_absolute_pressure_calibration import (  # noqa: E402
    _assert_exclusive_inputs,
)
from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (  # noqa: E402
    LOCKED_HOLDOUT_SHA256,
)


def _curve(single: float, saturated: float, threads: int) -> SaturatingServiceCurve:
    return SaturatingServiceCurve(
        single_thread_rate=single,
        saturated_rate=saturated,
        saturation_threads=threads,
        curve="power",
    )


def _dram_bound_model() -> AnalyticMoeCostModel:
    domain_curve = _curve(1e15, 4e15, 4)
    dram_curve = SaturatingServiceCurve(
        single_thread_rate=10e9,
        saturated_rate=30e9,
        saturation_threads=8,
        curve="piecewise_linear",
        points=((1, 10e9), (2, 15e9), (4, 18e9), (8, 30e9)),
    )
    calibration = AnalyticMachineCalibration(
        machine_id="scope-ablation-synthetic",
        cores_per_rank=8,
        caches=CacheCalibration(
            l1d_bytes_per_core=64 * 1024,
            l2_bytes_per_core=256 * 1024,
            llc_bytes_per_rank=64 * 1024 * 1024,
        ),
        matrix_flops=domain_curve,
        gemm_core_flops=domain_curve,
        frontend_instructions=domain_curve,
        l1_bytes=domain_curve,
        l2_bytes=domain_curve,
        llc_bytes=domain_curve,
        dram_bytes=dram_curve,
        epilogue_elements=domain_curve,
        supported_widths=(1, 2, 4, 8),
        rank_cpu_ids=tuple(range(8)),
        llc_domains=(
            LlcDomainCalibration("0", (0, 1, 2, 3), 32 * 1024 * 1024, domain_curve),
            LlcDomainCalibration("1", (4, 5, 6, 7), 32 * 1024 * 1024, domain_curve),
        ),
        dram_domain_injection=DramDomainInjectionCalibration(enabled=True, capacity_scale=0.5),
    )
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
    )


def _makespan(model: AnalyticMoeCostModel, cpus: tuple[int, ...]) -> float:
    return float(model.explain_dag_placed([(12, 1, (cpu,), ()) for cpu in cpus])["makespan_ns"])


def test_exclusive_inputs_reject_locked_holdout() -> None:
    holdout = next(iter(LOCKED_HOLDOUT_SHA256))

    with pytest.raises(ValueError, match="locked holdout"):
        _assert_exclusive_inputs("aaa", holdout)


def test_apply_arm_does_not_enable_domain_when_capacity_is_absent() -> None:
    model = _dram_bound_model()
    rank_only = DIAGNOSTIC_ARMS[0]
    no_rank = DIAGNOSTIC_ARMS[1]

    rank_cal = apply_arm(model.calibration, rank_only)
    no_rank_cal = apply_arm(model.calibration, no_rank)

    assert rank_only.disable_rank_dram is False
    assert no_rank.disable_rank_dram is True
    assert rank_cal.dram_domain_injection.enabled is False
    assert no_rank_cal.dram_domain_injection.enabled is False


def test_disable_rank_dram_makes_colocated_and_split_match() -> None:
    model = _dram_bound_model()
    colocated = (0, 1, 2, 3)
    split = (0, 1, 4, 5)

    with disable_rank_dram_dilation():
        off_model = AnalyticMoeCostModel(
            replace(
                model.calibration,
                dram_domain_injection=DramDomainInjectionCalibration(),
            ),
            hidden_size=64,
            intermediate_size=32,
            global_experts=8,
            local_experts=8,
        )
        colocated_ns = _makespan(off_model, colocated)
        split_ns = _makespan(off_model, split)

    assert colocated_ns == pytest.approx(split_ns, rel=1e-9)


def test_domain_only_restores_colocated_penalty_after_rank_dram_is_removed() -> None:
    model = _dram_bound_model()
    colocated = (0, 1, 2, 3)
    split = (0, 1, 4, 5)

    with disable_rank_dram_dilation():
        colocated_ns = _makespan(model, colocated)
        split_ns = _makespan(model, split)

    assert split_ns < colocated_ns


def test_decide_scope_requires_plateau_before_adding_structure() -> None:
    def report(*, remote: bool, locality: bool, plateau: bool, n15: float, cross_n15: float = 0.02) -> dict:
        return {
            "findings": {
                "remote_near_zero": remote,
                "locality_contrast_at_n4": locality,
                "local_plateau_at_n4": plateau,
                "same_llc_head_n15_ms": n15,
                "cross_llc_head_n15_ms": cross_n15,
            }
        }

    rejected = decide_scope(
        {
            "rank_only": report(remote=False, locality=False, plateau=False, n15=1.0, cross_n15=0.90),
            "no_rank_dram": report(remote=True, locality=False, plateau=True, n15=0.02, cross_n15=0.02),
            "domain_only_beta0.78": report(remote=True, locality=True, plateau=False, n15=1.2),
        }
    )
    accepted = decide_scope(
        {
            "rank_only": report(remote=False, locality=False, plateau=False, n15=1.0, cross_n15=0.90),
            "no_rank_dram": report(remote=True, locality=False, plateau=True, n15=0.02, cross_n15=0.02),
            "domain_only_beta0.78": report(remote=True, locality=True, plateau=True, n15=0.31),
        }
    )

    assert rejected["rank_dram_explains_remote_common_mode"] is True
    assert rejected["add_default_off_structure"] is False
    assert rejected["rank_dram_fraction_of_remote_n15"] == pytest.approx((0.90 - 0.02) / 0.90)
    assert accepted["add_default_off_structure"] is True
    assert accepted["domain_only_also_saturates_at_n4"] == ["domain_only_beta0.78"]


def test_arm_findings_detect_plateau_and_remote() -> None:
    findings = arm_findings(
        {
            "same_llc_head_n4": 0.29,
            "same_llc_head_n15": 0.31,
            "cross_llc_head_n15": 0.02,
        },
        {"same_minus_cross_head_n4": 0.27},
    )

    assert findings["local_plateau_at_n4"] is True
    assert findings["remote_near_zero"] is True
    assert findings["locality_contrast_at_n4"] is True
