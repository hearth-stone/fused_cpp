from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
PLANNER_DIR = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(BENCHMARK_DIR), str(PLANNER_DIR)]

import bench_executable_neighborhood_audit as runner  # noqa: E402
from executable_plan_neighborhood import ExecutablePlanScore  # noqa: E402
from pairwise_plan_ordering import (  # noqa: E402
    AnchorRelativePartialOrder,
    PairwiseResidualCalibration,
)


class _State:
    def __init__(self, key: str) -> None:
        self.key = key
        self.shape = (8, 8)

    def canonical_hash(self) -> str:
        return self.key

    def canonical_payload(self) -> dict[str, object]:
        return {"key": self.key}

    def to_bridge(self) -> dict[str, object]:
        return {"key": self.key}


class _Model:
    def explain_dag_placed(self, _tasks: object) -> dict[str, object]:
        return {}


class _Evaluator:
    def __init__(self, _model: object) -> None:
        self.exact_calls = 0
        self._cache: dict[str, ExecutablePlanScore] = {}

    def exact(self, state: _State) -> ExecutablePlanScore:
        cached = self._cache.get(state.key)
        if cached is not None:
            return cached
        robust = {"anchor": 100.0, "step-1": 90.0, "noise": 89.0}[state.key]
        score = ExecutablePlanScore(robust, robust, robust)
        self._cache[state.key] = score
        self.exact_calls += 1
        return score


def test_partial_order_vnd_updates_incumbent_only_for_confident_move(monkeypatch) -> None:
    states = {key: _State(key) for key in ("anchor", "step-1", "noise")}

    def score_neighborhood(_evaluator, anchor, **_kwargs):
        candidate_key = "step-1" if anchor.key == "anchor" else "noise"
        gain = 5.0 if candidate_key == "step-1" else 1.0
        robust = 90.0 if candidate_key == "step-1" else 89.0
        row = {
            "state_hash": candidate_key,
            "operator": "same_lane_insertion",
            "moved_experts": [1],
            "event_ns": robust,
            "robust_ns": robust,
            "event_gain_pct": gain,
            "robust_gain_pct": gain,
        }
        report = {
            "operators": {
                "same_lane_insertion": {
                    "proposed": 1,
                    "unique": 1,
                    "sampled": 1,
                }
            },
            "scored": [row],
        }
        return report, {candidate_key: states[candidate_key]}

    monkeypatch.setattr(runner, "ExecutablePlanEvaluator", _Evaluator)
    monkeypatch.setattr(runner, "_score_neighborhood", score_neighborhood)
    monkeypatch.setattr(runner, "critical_expert_scores", lambda *_: {1: 2.0, 2: 1.0})
    monkeypatch.setattr(runner, "placed_tasks", lambda *_: [])
    monkeypatch.setattr(runner, "_candidate_context", lambda *_: "same_lane_insertion")
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(0.9, 1, 1.0),
        minimum_gain_pct=2.0,
    )
    args = SimpleNamespace(
        partial_order_max_iterations=4,
        critical_experts=2,
        seed=7,
        neighborhood_mode="order",
        neighbors_per_operator=1,
        event_top_per_strategy=1,
        partial_order_shortlist_budget=2,
    )

    result = runner._run_partial_order_vnd_start(
        args,
        start_name="full",
        initial_state=states["anchor"],
        interval=object(),
        proposal_model=_Model(),
        score_model=_Model(),
        comparator=comparator,
        screen_budgets=(1,),
    )

    assert result["accepted_moves"] == 1
    assert result["accepted_by_operator"] == {"same_lane_insertion": 1}
    assert result["final_state_hash"] == "step-1"
    assert result["stop_reason"] == "no_confident_improvement"
    assert [row["state_hash"] for row in result["best_so_far_curve"]] == ["anchor", "step-1"]
    assert result["iterations"][1]["selected_frontier"][0]["state_hash"] == "noise"
    assert result["iterations"][1]["accepted_move"] is None
    assert result["initial_canonical_state"] == {"key": "anchor"}
    assert result["initial_plan_v2_bridge"] == {"key": "anchor"}
    assert result["iterations"][0]["selected_frontier"][0]["plan_v2_bridge"] == {
        "key": "step-1"
    }
    assert result["iterations"][1]["partial_order"]["relation_counts"] == {
        "candidate_better": 0,
        "candidate_worse": 0,
        "incomparable": 1,
    }
    assert result["placed_context_event_calls"] == 2


def test_lns_diagnostic_mode_never_accepts_a_model_better_candidate(monkeypatch) -> None:
    states = {key: _State(key) for key in ("anchor", "step-1", "noise")}

    def score_neighborhood(_evaluator, _anchor, **_kwargs):
        row = {
            "state_hash": "step-1",
            "operator": "critical_window_template_repartition_cross_domain_d4_b16",
            "moved_experts": [1],
            "event_ns": 90.0,
            "robust_ns": 90.0,
            "event_gain_pct": 5.0,
            "robust_gain_pct": 5.0,
        }
        return {
            "operators": {
                row["operator"]: {"proposed": 1, "unique": 1, "sampled": 1}
            },
            "scored": [row],
        }, {"step-1": states["step-1"]}

    monkeypatch.setattr(runner, "ExecutablePlanEvaluator", _Evaluator)
    monkeypatch.setattr(runner, "_score_neighborhood", score_neighborhood)
    monkeypatch.setattr(runner, "critical_expert_scores", lambda *_: {1: 2.0})
    monkeypatch.setattr(runner, "placed_tasks", lambda *_: [])
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(0.9, 1, 1.0),
        minimum_gain_pct=2.0,
    )
    args = SimpleNamespace(
        partial_order_max_iterations=1,
        critical_experts=1,
        seed=7,
        neighborhood_mode="lns",
        neighbors_per_operator=1,
        event_top_per_strategy=1,
        partial_order_shortlist_budget=1,
    )

    result = runner._run_partial_order_vnd_start(
        args,
        start_name="full",
        initial_state=states["anchor"],
        interval=object(),
        proposal_model=_Model(),
        score_model=_Model(),
        comparator=comparator,
        screen_budgets=(1,),
    )

    iteration = result["iterations"][0]
    partial = iteration["partial_order"]
    assert result["accepted_moves"] == 0
    assert result["final_state_hash"] == "anchor"
    assert result["stop_reason"] == "diagnostic_only_no_model_acceptance"
    assert partial["automatic_acceptance_enabled"] is False
    assert partial["dominance_pruning_enabled"] is False
    assert partial["dominated_keys"] == []
    assert iteration["worse_sentinels"] == []
    assert iteration["lns_diverse_shortlist"]["selected_keys"] == ["step-1"]
    assert iteration["lns_diverse_shortlist"]["audit_keys"][:1] == ["step-1"]
    assert iteration["candidate_features"][0]["actual_closure_size"] == 1


def test_calibrated_operator_context_skips_unused_placed_event_replay(monkeypatch) -> None:
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(
            0.9,
            1,
            4.0,
            context_radii_pct=(("same_lane_insertion", 1.0),),
        ),
        minimum_gain_pct=2.0,
    )

    def unexpected_context(*_args):
        raise AssertionError("placed-event context should not run for an exact operator key")

    monkeypatch.setattr(runner, "_candidate_context", unexpected_context)

    context, used_detailed = runner._candidate_context_for_calibration(
        comparator,
        _Model(),
        _State("anchor"),
        {},
        _State("noise"),
        "same_lane_insertion",
    )

    assert context == "same_lane_insertion"
    assert used_detailed is False


def test_operator_only_report_uses_family_fallback_without_placed_event_replay(monkeypatch) -> None:
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(
            0.9,
            1,
            4.0,
            family_radii_pct=(("order", 3.0),),
            context_radii_pct=(("same_lane_insertion", 1.0),),
        ),
        minimum_gain_pct=2.0,
    )

    def unexpected_context(*_args):
        raise AssertionError("operator-only calibration cannot use a placed-event key")

    monkeypatch.setattr(runner, "_candidate_context", unexpected_context)

    context, used_detailed = runner._candidate_context_for_calibration(
        comparator,
        _Model(),
        _State("anchor"),
        {},
        _State("noise"),
        "same_width_cross_lane_relocation",
    )

    assert context == "same_width_cross_lane_relocation"
    assert used_detailed is False
