from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from width_admission import (  # noqa: E402
    admitted_set,
    apply_width_admission,
    evaluate,
    judge,
    select_width,
)

PROFILE = COST_MODEL / "profiles" / "analytic_machine_amazon_c5_192c_numa0_sve_jit_hot_gemm_20260802.json"


def _key(makespan: float) -> tuple[float, float, float]:
    return (makespan, 0.0, 0.0)


# Three workloads. Base widths (4, 8). The model prefers 6 on "a" (a real gain), 2 on
# "b" (a real loss) and never prefers 3.
MEASURED = {
    "a": {2: 12.0, 3: 12.0, 4: 10.0, 6: 9.0, 8: 11.0},
    "b": {2: 11.0, 3: 12.0, 4: 10.0, 6: 10.5, 8: 10.2},
    "c": {2: 9.0, 3: 9.9, 4: 9.5, 6: 9.6, 8: 9.0},
}
SCORES = {
    "a": {2: _key(9.0), 3: _key(9.5), 4: _key(8.0), 6: _key(7.0), 8: _key(8.5)},
    "b": {2: _key(7.0), 3: _key(9.5), 4: _key(8.0), 6: _key(9.0), 8: _key(8.5)},
    "c": {2: _key(9.0), 3: _key(9.5), 4: _key(8.5), 6: _key(9.0), 8: _key(8.0)},
}


def test_select_width_follows_quick_ranking_with_a_stable_tie_break() -> None:
    assert select_width({4: _key(1.0), 8: _key(1.0)}, (8, 4)) == 4
    assert select_width({4: (1.0, 2.0, 0.0), 8: (1.0, 1.0, 0.0)}, (4, 8)) == 8
    assert select_width(SCORES["a"], (4, 8, 99)) == 4
    with pytest.raises(ValueError, match="no scored width"):
        select_width(SCORES["a"], (99,))


def test_judge_admits_gains_excludes_regressions_and_leaves_unused_widths_neutral() -> None:
    assert evaluate(MEASURED, SCORES, (4, 8)) == {"a": 10.0, "b": 10.0, "c": 9.0}
    verdicts = {v.width: v for v in judge(MEASURED, SCORES, (4, 8), (2, 3, 6))}

    assert verdicts[6].verdict == "admit" and verdicts[6].best_change == pytest.approx(-0.1)
    assert verdicts[2].verdict == "exclude" and verdicts[2].worst_change == pytest.approx(0.1)
    assert verdicts[3].verdict == "neutral" and verdicts[3].changed_workloads == 0
    assert admitted_set(MEASURED, SCORES, (4, 8), list(verdicts.values())) == (6,)


def test_admitted_set_keeps_only_admitted_widths_that_hold_jointly() -> None:
    measured = {"a": {4: 10.0, 6: 9.0, 12: 9.5}, "b": {4: 10.0, 6: 9.5, 12: 11.0}}
    scores = {
        "a": {4: _key(3.0), 6: _key(2.0), 12: _key(2.5)},
        "b": {4: _key(3.0), 6: _key(2.9), 12: _key(2.8)},
    }
    verdicts = judge(measured, scores, (4,), (6, 12))
    assert {v.width: v.verdict for v in verdicts} == {6: "admit", 12: "exclude"}
    assert admitted_set(measured, scores, (4,), verdicts) == (6,)
    assert evaluate(measured, scores, (4, 6)) == {"a": 9.0, "b": 9.5}


def _payload_with_overheads(widths) -> dict:
    payload = json.loads(PROFILE.read_text())
    payload["planner"] = {"supported_widths": [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96]}
    payload["overheads"]["by_width"] = [
        {"threads": width, "expert_fixed_ns": 1000.0, "route_ns": 10.0} for width in widths
    ]
    return payload


def test_apply_width_admission_sets_the_planner_search_and_keeps_every_width_runnable(tmp_path: Path) -> None:
    payload = _payload_with_overheads((1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96))
    admitted = apply_width_admission(payload, planner_widths=(4, 6, 8, 16, 24, 32, 48, 96))

    assert payload["planner"] == {"supported_widths": [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96]}
    assert admitted["planner"]["unreliable_widths"] == [1, 2, 12]
    path = tmp_path / "admitted.json"
    path.write_text(json.dumps(admitted))
    model = AnalyticMoeCostModel(
        path, hidden_size=4096, intermediate_size=512, global_experts=256, local_experts=256
    )
    assert model.planner_widths == (4, 6, 8, 16, 24, 32, 48, 96)
    assert model.supported_widths == (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96)
    everything = apply_width_admission(payload, planner_widths=payload["planner"]["supported_widths"])
    assert "unreliable_widths" not in everything["planner"]


def test_apply_width_admission_refuses_widths_without_their_own_overheads() -> None:
    payload = _payload_with_overheads((4, 8, 16))
    with pytest.raises(ValueError, match=r"\[6\] have no fitted per-width overheads"):
        apply_width_admission(payload, planner_widths=(4, 6, 8, 16))
    with pytest.raises(ValueError, match="not supported"):
        apply_width_admission(payload, planner_widths=(4, 5))
    with pytest.raises(ValueError, match="must not be empty"):
        apply_width_admission(payload, planner_widths=())

