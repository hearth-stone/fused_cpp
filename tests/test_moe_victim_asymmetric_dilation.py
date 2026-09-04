from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(COST_MODEL)]

from analytic_model import (  # noqa: E402
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    CacheCalibration,
    LlcDomainCalibration,
    SaturatingServiceCurve,
)
from optimizations.fused_moe_sve.benchmarks.ablate_victim_asymmetric_dilation import (  # noqa: E402
    apply_asymmetric_mode,
    decide_asymmetric,
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
    fast = _curve(1e15, 8e15, 8)
    domain_fast = _curve(1e15, 4e15, 4)
    dram_curve = SaturatingServiceCurve(
        single_thread_rate=10e9,
        saturated_rate=30e9,
        saturation_threads=8,
        curve="piecewise_linear",
        points=((1, 10e9), (2, 15e9), (4, 18e9), (8, 30e9)),
    )
    calibration = AnalyticMachineCalibration(
        machine_id="victim-asymmetric-synthetic",
        cores_per_rank=8,
        caches=CacheCalibration(
            l1d_bytes_per_core=64 * 1024,
            l2_bytes_per_core=256 * 1024,
            llc_bytes_per_rank=64 * 1024 * 1024,
        ),
        matrix_flops=fast,
        gemm_core_flops=fast,
        frontend_instructions=fast,
        l1_bytes=fast,
        l2_bytes=fast,
        llc_bytes=fast,
        dram_bytes=dram_curve,
        epilogue_elements=fast,
        supported_widths=(1, 2, 4, 8),
        rank_cpu_ids=tuple(range(8)),
        llc_domains=(
            LlcDomainCalibration("0", (0, 1, 2, 3), 32 * 1024 * 1024, domain_fast),
            LlcDomainCalibration("1", (4, 5, 6, 7), 32 * 1024 * 1024, domain_fast),
        ),
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


def test_own_demand_removes_cohort_dram_dilation() -> None:
    model = _dram_bound_model()
    colocated = (0, 1, 2, 3)

    symmetric = _makespan(model, colocated)
    with apply_asymmetric_mode("own_demand"):
        own_demand = _makespan(model, colocated)

    assert own_demand < symmetric


def test_compute_bound_skip_keeps_transfer_bound_cohort_dilation() -> None:
    model = _dram_bound_model()
    colocated = (0, 1, 2, 3)

    symmetric = _makespan(model, colocated)
    with apply_asymmetric_mode("compute_bound_skip"):
        skipped = _makespan(model, colocated)

    assert skipped == pytest.approx(symmetric, rel=1e-9)


def test_same_llc_peers_drops_cross_domain_dram_sharing() -> None:
    model = _dram_bound_model()
    colocated = (0, 1, 2, 3)
    split = (0, 1, 4, 5)

    with apply_asymmetric_mode("same_llc_peers"):
        local_ns = _makespan(model, colocated)
        remote_ns = _makespan(model, split)

    assert remote_ns < local_ns


def test_decide_asymmetric_requires_all_gates_before_adding_structure() -> None:
    def report(*, remote: bool, locality: bool, plateau: bool, same_n15: float, cross_n15: float) -> dict:
        return {
            "findings": {
                "remote_near_zero": remote,
                "locality_contrast_at_n4": locality,
                "local_plateau_at_n4": plateau,
                "same_llc_head_n15_ms": same_n15,
                "cross_llc_head_n15_ms": cross_n15,
            }
        }

    decision = decide_asymmetric(
        {
            "symmetric": report(
                remote=False, locality=False, plateau=False, same_n15=0.90, cross_n15=0.89
            ),
            "compute_bound_skip": report(
                remote=False, locality=False, plateau=False, same_n15=0.90, cross_n15=0.89
            ),
            "same_llc_peers": report(
                remote=True, locality=True, plateau=False, same_n15=0.90, cross_n15=0.02
            ),
            "own_demand": report(
                remote=True, locality=False, plateau=True, same_n15=0.00, cross_n15=0.00
            ),
        }
    )

    assert decision["compute_bound_skip_matches_symmetric"] is True
    assert decision["same_llc_peers_zeros_remote_not_local"] is True
    assert decision["own_demand_clears_peer_byte_inheritance"] is True
    assert decision["add_default_off_structure"] is False
    assert "saturating occupancy" in decision["reason"]
