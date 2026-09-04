from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLANNER_DIR = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNER_DIR))

from pairwise_plan_ordering import (  # noqa: E402
    CANDIDATE_BETTER,
    CANDIDATE_WORSE,
    INCOMPARABLE,
    AnchorRelativePartialOrder,
    PartialOrderCandidate,
    PairwiseObservation,
    PairwiseResidualCalibration,
    comparator_from_validated_report,
    neighborhood_context,
    run_partial_order_search,
    select_partial_order_shortlist,
)


def _observations() -> tuple[PairwiseObservation, ...]:
    return (
        PairwiseObservation("order", "same_lane", 1.0, 1.5),
        PairwiseObservation("order", "same_lane", 1.0, 0.0),
        PairwiseObservation("order", "cross_lane", 1.0, 3.0),
        PairwiseObservation("width", "split", 0.0, 4.0),
    )


def test_pairwise_calibration_uses_context_family_then_global_fallback() -> None:
    calibration = PairwiseResidualCalibration.fit(
        _observations(),
        coverage=1.0,
        minimum_context_samples=2,
    )

    context_radius, context_scope = calibration.radius_pct(family="order", context="same_lane")
    family_radius, family_scope = calibration.radius_pct(family="order", context="unseen_order")
    global_radius, global_scope = calibration.radius_pct(family="unseen", context="unseen")

    assert context_radius == pytest.approx(1.0)
    assert context_scope == "context"
    assert family_radius == pytest.approx(2.0)
    assert family_scope == "family"
    assert global_radius == pytest.approx(4.0)
    assert global_scope == "global"


@pytest.mark.parametrize(
    ("predicted_gain_pct", "expected"),
    [
        (4.1, CANDIDATE_BETTER),
        (-4.1, CANDIDATE_WORSE),
        (2.9, INCOMPARABLE),
        (-2.9, INCOMPARABLE),
    ],
    ids=("better", "worse", "positive-overlap", "negative-overlap"),
)
def test_anchor_partial_order_requires_the_complete_residual_interval_to_clear_margin(
    predicted_gain_pct: float,
    expected: str,
) -> None:
    calibration = PairwiseResidualCalibration(
        coverage=0.9,
        minimum_context_samples=2,
        global_radius_pct=1.0,
    )
    comparator = AnchorRelativePartialOrder(calibration, minimum_gain_pct=2.0)

    evidence = comparator.compare(predicted_gain_pct, family="order", context="same_lane")

    assert evidence.relation == expected
    assert evidence.lower_gain_pct == pytest.approx(predicted_gain_pct - 1.0)
    assert evidence.upper_gain_pct == pytest.approx(predicted_gain_pct + 1.0)


def test_pairwise_calibration_round_trip_preserves_fallback_scopes() -> None:
    calibration = PairwiseResidualCalibration.fit(
        _observations(),
        coverage=0.75,
        minimum_context_samples=2,
    )

    repeated = PairwiseResidualCalibration.from_dict(calibration.to_dict())

    assert repeated == calibration


def test_pairwise_observation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="predicted_gain_pct must be finite"):
        PairwiseObservation("order", "same_lane", float("nan"), 0.0)


def test_partial_order_shortlist_prunes_only_confidently_worse_candidates() -> None:
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(0.9, 1, 1.0),
        minimum_gain_pct=2.0,
    )
    candidates = (
        PartialOrderCandidate("better", "order", "a", 4.0),
        PartialOrderCandidate("uncertain-a", "order", "a", 1.0),
        PartialOrderCandidate("uncertain-b", "order", "b", -1.0),
        PartialOrderCandidate("worse", "order", "c", -4.0),
    )

    result = select_partial_order_shortlist(comparator, candidates, budget=2)

    assert result.selected_keys == ("better", "uncertain-b")
    assert result.accepted_key == "better"
    assert result.dominated_keys == ("worse",)
    assert result.budget_deferred_keys == ("uncertain-a",)


def test_partial_order_search_keeps_incomparable_frontier_without_accepting_it() -> None:
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(0.9, 1, 1.0),
        minimum_gain_pct=2.0,
    )

    result = run_partial_order_search(
        comparator,
        "anchor",
        lambda _: (PartialOrderCandidate("uncertain", "order", "same_lane", 2.5),),
        shortlist_budget=1,
        maximum_iterations=4,
    )

    assert result.incumbent_key == "anchor"
    assert result.stop_reason == "no_confident_improvement"
    assert result.iterations[0].shortlist.selected_keys == ("uncertain",)


def test_partial_order_search_updates_anchor_until_no_confident_improvement() -> None:
    comparator = AnchorRelativePartialOrder(
        PairwiseResidualCalibration(0.9, 1, 0.5),
        minimum_gain_pct=2.0,
    )
    proposals = {
        "anchor": (PartialOrderCandidate("step-1", "order", "same_lane", 3.0),),
        "step-1": (PartialOrderCandidate("step-2", "order", "same_lane", 4.0),),
        "step-2": (PartialOrderCandidate("noise", "order", "same_lane", 1.0),),
    }

    result = run_partial_order_search(
        comparator,
        "anchor",
        proposals.__getitem__,
        shortlist_budget=2,
        maximum_iterations=4,
    )

    assert result.incumbent_key == "step-2"
    assert result.stop_reason == "no_confident_improvement"
    assert [iteration.anchor_key for iteration in result.iterations] == ["anchor", "step-1", "step-2"]


def test_neighborhood_context_matches_calibration_key_contract() -> None:
    context = neighborhood_context(
        "same_width_cross_lane_swap",
        {
            "placed_critical_lane_switched": False,
            "placed_critical_expert_switched": True,
            "affected_head_direction": "decrease",
            "affected_tail_direction": "increase",
            "cohort_transition_direction": "unchanged",
        },
    )

    assert context == (
        "same_width_cross_lane_swap|critical_lane_switch=0|critical_expert_switch=1|"
        "head=decrease|tail=increase|cohort=unchanged"
    )


def test_validated_report_requires_false_pruning_and_requested_top_k_gates() -> None:
    payload = {
        "calibration": PairwiseResidualCalibration(0.9, 1, 1.0).to_dict(),
        "gates": {
            "zero_false_pruning": True,
            "measured_best_retained": {"16": True, "8": False},
        },
    }

    comparator = comparator_from_validated_report(
        payload,
        shortlist_budget=16,
        minimum_gain_pct=2.0,
    )

    assert comparator.minimum_gain_pct == pytest.approx(2.0)
    with pytest.raises(ValueError, match="top-8"):
        comparator_from_validated_report(
            payload,
            shortlist_budget=8,
            minimum_gain_pct=2.0,
        )
