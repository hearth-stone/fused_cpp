from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from analytic_model import AnalyticMachineCalibration, AnalyticMoeCostModel, DramDomainInjectionCalibration  # noqa: E402
from stage_window_policy import StageWindowBand, StageWindowPolicy  # noqa: E402
from test_moe_analytic_model import _calibration  # noqa: E402
from test_moe_native_placed_dag import _contended_model, _native, _piecewise, _random_plan  # noqa: E402

POLICY = StageWindowPolicy(
    name="synthetic_windows",
    hidden_size=64,
    intermediate_size=32,
    backend_n_tile=8,
    bands=(StageWindowBand(min_routes=24, max_routes=4096, w13_tiles=2, w2_tiles=2, widths=(1, 2, 4, 8)),),
)
MULTI = _piecewise(((1, 1e9), (4, 1.5e9), (8, 1.8e9)))


def _window_model(base=None, **terms):
    calibration = replace(base or _contended_model().calibration, **terms)
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
        stage_window_policy=POLICY,
    )


def test_window_model_terms_default_off_and_round_trip() -> None:
    default = _calibration()
    payload = default.to_dict()
    assert "dram_multi_stream_bytes" not in payload["services"]
    assert "llc_spill_rule" not in payload["planner"] and "executed_window_geometry" not in payload["planner"]
    assert (default.dram_multi_stream_bytes, default.llc_spill_rule, default.executed_window_geometry) == (
        None, "aggregate", False)

    terms = replace(default, dram_multi_stream_bytes=MULTI, llc_spill_rule="private_l2_window",
                    executed_window_geometry=True)
    assert AnalyticMachineCalibration.from_dict(terms.to_dict()) == terms
    with pytest.raises(ValueError, match="llc_spill_rule"):
        replace(default, llc_spill_rule="other")


def test_private_l2_window_rule_folds_spill_into_fixed_bytes() -> None:
    aggregate = _window_model()
    private = _window_model(llc_spill_rule="private_l2_window")
    for routes, threads in ((12, 2), (48, 4), (300, 8)):
        phases = private.predict_expert(routes, threads).phases
        assert all(phase.working_set_bytes == 0.0 and phase.spillable_dram_bytes == 0.0
                   for phase in phases if phase.kind in {"cold_b", "steady_b"})
    # With no aggregation, concurrent phases can no longer push each other into LLC spill.
    tasks = [(300, 2, (2 * lane, 2 * lane + 1), ()) for lane in range(4)]
    assert private.dag_makespan_placed(tasks) <= aggregate.dag_makespan_placed(tasks)


def test_multi_stream_dram_capacity_binds_only_for_two_or_more_streams() -> None:
    # Rank-level DRAM only: domain injection would bind first on the contended fixture.
    rank_only = replace(_contended_model().calibration, dram_domain_injection=DramDomainInjectionCalibration())
    base = _window_model(rank_only)
    multi = _window_model(rank_only, dram_multi_stream_bytes=MULTI)
    alone = [(96, 4, (0, 1, 2, 3), ())]
    pair = [(96, 4, (0, 1, 2, 3), ()), (96, 4, (4, 5, 6, 7), ())]
    assert multi.dag_makespan_placed(alone) == base.dag_makespan_placed(alone)
    assert multi.dag_makespan_placed(pair) > base.dag_makespan_placed(pair)
    assert multi.calibration.contended_dram_capacity(5e9, 8, 1) == 5e9
    assert multi.calibration.contended_dram_capacity(5e9, 8, 2) == MULTI.rate(8)


def test_executed_window_geometry_builds_one_cold_and_steady_phase_per_window() -> None:
    windowed = _window_model(executed_window_geometry=True)
    full = _window_model()
    phases = windowed.predict_expert(48, 1).phases
    windows = windowed.score_stage_window("w13", 48, 1, 2).windows
    assert [p.name for p in phases if p.name.startswith("w13:") and p.kind in {"cold_b", "steady_b"}] == [
        f"w13:w{i}:{kind}" for i in range(windows) for kind in ("cold_b", "steady_b")]
    assert windowed.t_iso_cache_identity()["executed_window_policy"] == "synthetic_windows"
    assert full.t_iso_cache_identity()["executed_window_policy"] is None
    # Uncovered routes keep the full-stripe phases.
    assert [p.name for p in windowed.predict_expert(12, 1).phases] == [p.name for p in full.predict_expert(12, 1).phases]


@pytest.mark.parametrize("seed", range(20))
def test_native_matches_python_with_every_window_term(seed: int) -> None:
    model = _window_model(dram_multi_stream_bytes=MULTI, llc_spill_rule="private_l2_window",
                          executed_window_geometry=True)
    _native(model)
    tasks = _random_plan(random.Random(seed))
    assert model.dag_makespan_placed(tasks) == pytest.approx(model._python_dag_makespan_placed(tasks), rel=1e-12)
    aggregate = [(routes, threads, deps) for routes, threads, _, deps in tasks]
    reference_makespan, reference_finish = model._dag_result(aggregate)
    assert model.dag_makespan(aggregate) == pytest.approx(reference_makespan, rel=1e-12)
    assert model.dag_task_finish_times(aggregate) == pytest.approx(reference_finish, rel=1e-12)
