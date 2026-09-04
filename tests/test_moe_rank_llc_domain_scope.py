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
from optimizations.fused_moe_sve.benchmarks.ablate_rank_llc_domain_scope import (  # noqa: E402
    decide_llc_scope,
    disable_placed_dilations,
    is_rank_llc_service,
)
from optimizations.fused_moe_sve.benchmarks.fit_absolute_pressure_calibration import (  # noqa: E402
    _assert_exclusive_inputs,
)
from optimizations.fused_moe_sve.benchmarks.fit_phase_reaccount_calibration import (  # noqa: E402
    LOCKED_HOLDOUT_SHA256,
)


def _fast() -> SaturatingServiceCurve:
    return SaturatingServiceCurve(
        single_thread_rate=1e15,
        saturated_rate=8e15,
        saturation_threads=8,
        curve="power",
    )


def _llc_bound_model() -> AnalyticMoeCostModel:
    domain_curve = SaturatingServiceCurve(
        single_thread_rate=40.0,
        saturated_rate=50.0,
        saturation_threads=4,
        curve="piecewise_linear",
        points=((1, 40.0), (2, 45.0), (4, 50.0)),
    )
    rank_curve = SaturatingServiceCurve(
        single_thread_rate=40.0,
        saturated_rate=40.0,
        saturation_threads=8,
        curve="piecewise_linear",
        points=((1, 40.0), (8, 40.0)),
    )
    fast = _fast()
    calibration = AnalyticMachineCalibration(
        machine_id="llc-scope-ablation-synthetic",
        cores_per_rank=8,
        caches=CacheCalibration(
            l1d_bytes_per_core=64 * 1024,
            l2_bytes_per_core=256 * 1024,
            llc_bytes_per_rank=16 * 1024 * 1024,
        ),
        matrix_flops=fast,
        gemm_core_flops=fast,
        frontend_instructions=fast,
        l1_bytes=fast,
        l2_bytes=fast,
        llc_bytes=rank_curve,
        dram_bytes=fast,
        epilogue_elements=fast,
        supported_widths=(1, 2, 4, 8),
        rank_cpu_ids=tuple(range(8)),
        llc_domains=(
            LlcDomainCalibration("0", (0, 1, 2, 3), 8 * 1024 * 1024, domain_curve),
            LlcDomainCalibration("1", (4, 5, 6, 7), 8 * 1024 * 1024, domain_curve),
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


def test_rank_llc_service_uses_every_domain_id() -> None:
    assert is_rank_llc_service({"0": 4, "1": 0}) is True
    assert is_rank_llc_service({"0": 4}) is False
    assert is_rank_llc_service(None) is False


def test_same_llc_makespan_is_unchanged_when_rank_llc_is_removed() -> None:
    model = _llc_bound_model()
    colocated = (0, 1, 2, 3)

    baseline = _makespan(model, colocated)
    with disable_placed_dilations(rank_llc=True):
        patched = _makespan(model, colocated)

    assert patched == pytest.approx(baseline, rel=1e-9)


def test_cross_llc_makespan_drops_when_rank_llc_is_removed() -> None:
    model = _llc_bound_model()
    split = (0, 1, 4, 5)

    baseline = _makespan(model, split)
    with disable_placed_dilations(rank_llc=True):
        patched = _makespan(model, split)

    assert patched < baseline


def test_all_llc_off_drops_same_llc_makespan_after_rank_llc_is_gone() -> None:
    model = _llc_bound_model()
    colocated = (0, 1, 2, 3)

    with disable_placed_dilations(rank_llc=True):
        domain_only = _makespan(model, colocated)
    with disable_placed_dilations(rank_llc=True, all_llc=True):
        no_llc = _makespan(model, colocated)

    assert no_llc < domain_only


def test_decide_llc_scope_flags_victim_asymmetric_when_peer_cache_is_gone() -> None:
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

    decision = decide_llc_scope(
        {
            "no_rank_dram": report(
                remote=False, locality=True, plateau=False, same_n15=0.59, cross_n15=0.45
            ),
            "domain_llc_only": report(
                remote=True, locality=True, plateau=False, same_n15=0.59, cross_n15=0.02
            ),
            "no_llc": report(
                remote=True, locality=False, plateau=True, same_n15=0.12, cross_n15=0.02
            ),
            "no_llc_no_l2": report(
                remote=True, locality=False, plateau=True, same_n15=0.01, cross_n15=0.00
            ),
        }
    )

    assert decision["rank_llc_explains_remote_common_mode"] is True
    assert decision["domain_llc_carries_remaining_local"] is True
    assert decision["victim_free_of_peer_cache_after_no_llc_no_l2"] is True
    assert decision["add_default_off_structure"] is False
    assert "victim-asymmetric" in decision["reason"]


def test_decide_llc_scope_adds_structure_only_when_one_arm_hits_all_gates() -> None:
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

    decision = decide_llc_scope(
        {
            "no_rank_dram": report(
                remote=False, locality=True, plateau=False, same_n15=0.59, cross_n15=0.45
            ),
            "domain_llc_only": report(
                remote=True, locality=True, plateau=True, same_n15=0.31, cross_n15=0.02
            ),
            "no_llc": report(
                remote=True, locality=False, plateau=True, same_n15=0.02, cross_n15=0.02
            ),
            "no_llc_no_l2": report(
                remote=True, locality=False, plateau=True, same_n15=0.01, cross_n15=0.00
            ),
        }
    )

    assert decision["add_default_off_structure"] is True
    assert "matches remote-zero" in decision["reason"]
