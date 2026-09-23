"""Widths a whole-plan measurement refuted stay runnable but leave the planner's own search."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(ROOT / "cpu_moe_schedule_optimization" / "planners"),
    str(Path(__file__).resolve().parent),
]

from analytic_model import AnalyticMachineCalibration, AnalyticMoeCostModel  # noqa: E402
from interval_planner import IntervalPlanner  # noqa: E402
from test_moe_analytic_model import _calibration  # noqa: E402

# Many light experts: isolated LPT favours the narrowest lanes here.
UNIFORM = [(expert, 4) for expert in range(64)]


def _model(calibration: AnalyticMachineCalibration) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(calibration, hidden_size=64, intermediate_size=32, global_experts=64,
                                local_experts=64)


def test_unreliable_widths_round_trip_and_default_to_none() -> None:
    plain = _calibration()
    assert plain.unreliable_widths == ()
    assert "unreliable_widths" not in plain.to_dict()["planner"]
    marked = replace(plain, unreliable_widths=(2, 1))
    payload = marked.to_dict()
    assert payload["planner"]["unreliable_widths"] == [1, 2]
    assert AnalyticMachineCalibration.from_dict(payload).unreliable_widths == (1, 2)


@pytest.mark.parametrize("widths, message", [((16,), "subset"), ((1, 2, 4, 8), "at least one")])
def test_unreliable_widths_are_validated(widths: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_calibration(), unreliable_widths=widths)


def test_planner_skips_unreliable_widths_but_a_fixed_width_still_runs() -> None:
    plain = _model(_calibration())
    marked = _model(replace(_calibration(), unreliable_widths=(1, 2)))
    assert marked.supported_widths == (1, 2, 4, 8)
    assert marked.planner_widths == (4, 8)

    free = IntervalPlanner(plain, 8, native_cold_planner=False)
    chosen = free.plan_quick(UNIFORM)
    assert set(chosen["shape"]) <= {1, 2}, "the test needs a workload the unmarked model plans narrow"

    guarded = IntervalPlanner(marked, 8, native_cold_planner=False)
    assert guarded.widths == (4, 8)
    assert set(guarded.plan_quick(UNIFORM)["shape"]) <= {4, 8}
    assert set(guarded.plan_quick_fixed(UNIFORM, 1)["shape"]) == {1}
