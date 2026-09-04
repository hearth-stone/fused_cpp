from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from analyze_plan_pairwise_ordering import build_report, extract_pairwise_rows  # noqa: E402


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_extract_neighborhood_pair_preserves_event_and_paired_gain(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "neighborhood.json",
        {
            "kind": "executable_neighborhood_audit",
            "identity": {"calibration_sha256": "cal", "extension_sha256": "ext"},
            "route": {"sha256": "route", "layer_index": 3},
            "baseline": {"state_hash": "anchor"},
            "strategies": {
                "critical": {
                    "scored": [
                        {
                            "state_hash": "candidate",
                            "operator": "same_lane_insertion",
                            "event_gain_pct": 0.5,
                            "robust_gain_pct": 0.25,
                        }
                    ]
                }
            },
            "hardware_shortlist": {
                "measurement": {
                    "baseline": {"samples_ms": [10.0, 10.0, 10.0]},
                    "candidates": [
                        {
                            "name": "neighbor_00",
                            "state_hash": "candidate",
                            "stats": {"samples_ms": [9.0, 9.0, 9.0]},
                            "paired_speedup_pct": {
                                "median": 11.0,
                                "p10": 10.0,
                                "p90": 12.0,
                                "wins": 3,
                                "runs": 3,
                            },
                        }
                    ],
                }
            },
        },
    )

    rows = extract_pairwise_rows(path)

    assert len(rows) == 1
    assert rows[0]["family"] == "order"
    assert rows[0]["context"] == "same_lane_insertion"
    assert rows[0]["event_gain_pct"] == pytest.approx(0.5)
    assert rows[0]["prediction_basis"] == "robust_gain"
    assert rows[0]["predicted_gain_pct"] == pytest.approx(0.25)
    assert rows[0]["measured_gain_pct"] == pytest.approx(11.0)
    assert rows[0]["residual_pct"] == pytest.approx(10.75)


def test_extract_width_migration_uses_width_family(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "width-migration.json",
        {
            "kind": "executable_neighborhood_audit",
            "identity": {},
            "route": {},
            "baseline": {"state_hash": "anchor"},
            "strategies": {
                "critical": {
                    "scored": [
                        {
                            "state_hash": "candidate",
                            "operator": "domain_local_adjacent_width_migration",
                            "event_gain_pct": 0.5,
                        }
                    ]
                }
            },
            "hardware_shortlist": {
                "measurement": {
                    "baseline": {"samples_ms": [10.0]},
                    "candidates": [
                        {
                            "name": "neighbor_00",
                            "state_hash": "candidate",
                            "stats": {"samples_ms": [9.9]},
                        }
                    ],
                }
            },
        },
    )

    row = extract_pairwise_rows(path)[0]

    assert row["family"] == "width"


def test_extract_neighborhood_pair_classifies_placed_event_context(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "context.json",
        {
            "kind": "executable_neighborhood_audit",
            "identity": {},
            "route": {},
            "baseline": {"state_hash": "anchor"},
            "strategies": {
                "critical": {
                    "scored": [
                        {
                            "state_hash": "candidate",
                            "operator": "same_width_cross_lane_swap",
                            "event_gain_pct": 1.0,
                        }
                    ]
                }
            },
            "hardware_shortlist": {
                "measurement": {
                    "baseline": {"samples_ms": [10.0, 10.0, 10.0]},
                    "candidates": [
                        {
                            "name": "neighbor_00",
                            "state_hash": "candidate",
                            "stats": {"samples_ms": [9.0, 9.0, 9.0]},
                            "affected_lanes": {"critical_isolated_lane_switched": True},
                            "placed_event_context": {
                                "before": {
                                    "critical_lane": {"lane_index": 1},
                                    "critical_task": {"expert_id": 3},
                                },
                                "after": {
                                    "critical_lane": {"lane_index": 2},
                                    "critical_task": {"expert_id": 7},
                                },
                                "delta": {
                                    "critical_lane_switched": True,
                                    "critical_expert_switched": True,
                                    "affected_head_ns": -4.0,
                                    "affected_tail_ns": 8.0,
                                    "mean_affected_phase_dilation": 0.2,
                                    "mean_affected_team_pressure_dilation": -0.1,
                                    "cohort_transition_count": 2,
                                },
                            },
                        }
                    ],
                }
            },
        },
    )

    row = extract_pairwise_rows(path)[0]

    assert row["context"] == (
        "same_width_cross_lane_swap|critical_lane_switch=1|critical_expert_switch=1|"
        "head=decrease|tail=increase|cohort=increase"
    )
    assert row["event_context_features"] == {
        "isolated_critical_lane_switched": True,
        "placed_critical_lane_switched": True,
        "placed_critical_expert_switched": True,
        "affected_head_direction": "decrease",
        "affected_tail_direction": "increase",
        "cohort_transition_direction": "increase",
        "before_critical_lane": 1,
        "after_critical_lane": 2,
        "before_critical_expert": 3,
        "after_critical_expert": 7,
        "affected_tail_delta_ns": 8.0,
        "affected_head_delta_ns": -4.0,
        "affected_phase_dilation_delta": 0.2,
        "affected_team_pressure_delta": -0.1,
        "cohort_transition_count_delta": 2,
    }


def test_extract_strict_pair_computes_anchor_relative_prediction_and_measurement(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "strict.json",
        {
            "kind": "high_skew_planner_closure",
            "identity": {},
            "route": {},
            "cp_sat_measured": {"event_selected": "full_selected"},
            "candidates": {
                "full_selected": {
                    "predicted_ms": 10.0,
                    "plan": {"width_histogram": {"8": 10}},
                },
                "cp_sat_00": {
                    "event_model_ms": 10.2,
                    "plan": {"width_histogram": {"4": 4, "8": 8}},
                },
            },
            "stats": {
                "full_selected": {"samples_ms": [10.0, 10.0, 10.0]},
                "cp_sat_00": {"samples_ms": [9.0, 9.0, 9.0]},
            },
        },
    )

    rows = extract_pairwise_rows(path)

    assert len(rows) == 1
    assert rows[0]["family"] == "global"
    assert rows[0]["context"] == "cp_sat_width_order_domain"
    assert rows[0]["predicted_gain_pct"] == pytest.approx(100.0 * (10.0 / 10.2 - 1.0))
    assert rows[0]["measured_gain_pct"] == pytest.approx(100.0 * (10.0 / 9.0 - 1.0))


def test_extract_truncated_neighborhood_recovers_ablation_decision_context(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "truncated.json",
        {
            "kind": "executable_neighborhood_audit",
            "identity": {},
            "route": {},
            "baseline": {"state_hash": "anchor"},
            "strategies": {"critical": {"scored": []}},
            "ablation_decisions": {
                "width_only": {
                    "candidate_state_hash": "candidate",
                    "candidate_event_gain_pct": 0.25,
                }
            },
            "hardware_shortlist": {
                "sources": {"candidate": ["critical:width_only", "decision:width_only"]},
                "measurement": {
                    "baseline": {"samples_ms": [10.0, 10.0, 10.0]},
                    "candidates": [
                        {
                            "name": "neighbor_00",
                            "state_hash": "candidate",
                            "stats": {"samples_ms": [9.9, 9.9, 9.9]},
                            "paired_speedup_pct": {
                                "median": 1.0,
                                "p10": 0.5,
                                "p90": 1.5,
                                "wins": 3,
                                "runs": 3,
                            },
                        }
                    ],
                },
            },
        },
    )

    rows = extract_pairwise_rows(path)

    assert rows[0]["family"] == "width"
    assert rows[0]["context"] == "unknown_width_neighbor"
    assert rows[0]["predicted_gain_pct"] == pytest.approx(0.25)


def test_extract_width_order_oracle_uses_full_selected_anchor(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "width-order.json",
        {
            "kind": "high_skew_width_order_measured_oracle",
            "profile_sha256": "profile",
            "extension_sha256": "extension",
            "route_sha256": "route",
            "metadata": {
                "full_selected": {
                    "predicted_full_ms": 8.0,
                    "plan": {"width_histogram": {"16": 10}},
                },
                "w8_reverse_even": {
                    "predicted_full_ms": 12.0,
                    "plan": {"width_histogram": {"8": 20}},
                },
            },
            "stats": {
                "full_selected": {"samples_ms": [24.0, 24.0, 24.0]},
                "w8_reverse_even": {"samples_ms": [18.0, 18.0, 18.0]},
            },
        },
    )

    rows = extract_pairwise_rows(path)

    assert len(rows) == 1
    assert rows[0]["calibration_sha256"] == "profile"
    assert rows[0]["family"] == "width"
    assert rows[0]["context"] == "homogeneous_width_change"
    assert rows[0]["predicted_gain_pct"] == pytest.approx(100.0 * (8.0 / 12.0 - 1.0))
    assert rows[0]["measured_gain_pct"] == pytest.approx(100.0 * (24.0 / 18.0 - 1.0))


def test_development_replay_marks_calibrated_direction_reversal_incomparable() -> None:
    row = {
        "artifact": "known-counterexample.json",
        "anchor": "full_selected",
        "candidate": "cp_sat_06",
        "family": "global",
        "context": "cp_sat_width_order_domain",
        "predicted_gain_pct": -0.6,
        "measured_gain_pct": 4.0,
        "measured_p10_gain_pct": 3.0,
        "measured_p90_gain_pct": 5.0,
        "residual_pct": 4.6,
    }

    report = build_report(
        [row],
        [row],
        coverage=1.0,
        minimum_context_samples=1,
        minimum_gain_pct=2.0,
        dominance_margin_pct=0.0,
        top_ks=(1,),
        evaluation_role="development_replay_not_holdout",
    )

    evaluated = report["evaluation"]["rows"][0]
    recall = report["evaluation"]["top_k_recall"]["known-counterexample.json"]["budgets"]["1"]
    assert evaluated["partial_order"]["relation"] == "incomparable"
    assert report["evaluation"]["false_dominance_pairs"] == 0
    assert report["evaluation"]["false_pruning_pairs"] == 0
    assert report["gates"] == {
        "zero_false_pruning": True,
        "measured_best_retained": {"1": True},
    }
    assert recall["measured_best_retained"] is True
