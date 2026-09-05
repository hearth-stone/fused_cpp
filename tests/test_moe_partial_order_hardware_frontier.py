from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from analyze_lns_diverse_hardware_frontier import analyze_lns_diverse_sessions  # noqa: E402
from analyze_partial_order_hardware_frontier import analyze_sessions  # noqa: E402
from analyze_template_lns_frontier import analyze_template_lns  # noqa: E402
from analyze_template_lns_suite import analyze_suite  # noqa: E402
from bench_template_lns_layer import (  # noqa: E402
    _deduplicate_named_states,
    _load_parent_states,
)
from build_lns_diverse_hardware_frontier import build_lns_diverse_frontier  # noqa: E402
from build_partial_order_hardware_frontier import build_frontier  # noqa: E402
from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)


def _session(*, frontier_stable: bool, sentinel_stable: bool) -> dict[str, object]:
    def row(role: str, stable: bool, candidate: str) -> dict[str, object]:
        return {
            "candidate_state_hash": candidate,
            "anchor_state_hash": "anchor",
            "role": role,
            "start": "greedy",
            "operator": "domain_local_lane_split",
            "moved_experts": [1],
            "robust_gain_pct": 1.0,
            "partial_order": {"relation": "incomparable" if "frontier" in role else "candidate_worse"},
            "paired_gain_pct": {"median": 3.0, "p10": 1.0, "p90": 4.0, "runs": 31},
            "session_stable_over_2pct": stable,
        }

    return {
        "kind": "partial_order_hardware_frontier_session",
        "identity": {"frontier_sha256": "frontier", "extension_sha256": "extension"},
        "method": {"runs": 31},
        "plan_stats": {
            "anchor": {"median_ms": 10.0},
            "candidate": {"median_ms": 9.0},
            "sentinel": {"median_ms": 11.0},
        },
        "comparisons": [
            row("incomparable_frontier", frontier_stable, "candidate"),
            row("candidate_worse_sentinel", sentinel_stable, "sentinel"),
        ],
    }


def test_cross_session_frontier_selects_hardware_assisted_beam() -> None:
    result = analyze_sessions(
        _session(frontier_stable=True, sentinel_stable=False),
        _session(frontier_stable=True, sentinel_stable=False),
    )

    assert result["decision"] == "use_hardware_assisted_beam_search"
    assert len(result["stable_frontier_candidates"]) == 1
    assert result["false_pruning_sentinels"] == []


def test_cross_session_sentinel_failure_overrides_frontier_winner() -> None:
    result = analyze_sessions(
        _session(frontier_stable=True, sentinel_stable=True),
        _session(frontier_stable=True, sentinel_stable=True),
    )

    assert result["decision"] == "partial_order_pruning_gate_failed"
    assert len(result["false_pruning_sentinels"]) == 1


def test_cross_session_failed_candidate_better_disables_automatic_acceptance() -> None:
    sessions = [
        _session(frontier_stable=False, sentinel_stable=False),
        _session(frontier_stable=False, sentinel_stable=False),
    ]
    for session in sessions:
        candidate = session["comparisons"][0]
        candidate["role"] = "candidate_better_frontier"
        candidate["partial_order"] = {"relation": "candidate_better"}

    result = analyze_sessions(*sessions)

    assert result["decision"] == "partial_order_acceptance_gate_failed"
    assert len(result["false_acceptance_candidates"]) == 1
    assert result["stable_frontier_candidates"] == []


def test_cross_session_stable_candidate_better_remains_in_frontier() -> None:
    sessions = [
        _session(frontier_stable=True, sentinel_stable=False),
        _session(frontier_stable=True, sentinel_stable=False),
    ]
    for session in sessions:
        candidate = session["comparisons"][0]
        candidate["role"] = "candidate_better_frontier"
        candidate["partial_order"] = {"relation": "candidate_better"}

    result = analyze_sessions(*sessions)

    assert result["decision"] == "use_hardware_assisted_beam_search"
    assert len(result["stable_frontier_candidates"]) == 1
    assert result["false_acceptance_candidates"] == []


def test_cross_session_model_worse_spectrum_is_not_counted_as_false_pruning() -> None:
    sessions = [
        _session(frontier_stable=True, sentinel_stable=False),
        _session(frontier_stable=True, sentinel_stable=False),
    ]
    for session in sessions:
        candidate = session["comparisons"][0]
        candidate["role"] = "model_worse_spectrum"
        candidate["partial_order"] = {"relation": "candidate_worse"}

    result = analyze_sessions(*sessions)

    assert result["decision"] == "use_hardware_assisted_beam_search"
    assert len(result["stable_frontier_candidates"]) == 1
    assert result["false_pruning_sentinels"] == []


def test_template_lns_parent_loader_requires_self_consistent_executable_states() -> None:
    state = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(
            ExecutableLane(
                0,
                2,
                (ExecutableExpertTask(3, 7),),
            ),
        ),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )
    state_hash = state.canonical_hash()
    frontier = {
        "kind": "partial_order_hardware_frontier",
        "plans": {
            state_hash: {
                "canonical_state": state.canonical_payload(),
                "plan_v2_bridge": state.to_bridge(),
            }
        },
    }

    loaded = _load_parent_states(frontier, [state_hash])

    assert loaded[state_hash] == state
    with pytest.raises(ValueError, match="unique"):
        _load_parent_states(frontier, [state_hash, state_hash])
    with pytest.raises(ValueError, match="absent"):
        _load_parent_states(frontier, ["missing"])
    frontier["plans"][state_hash]["plan_v2_bridge"] = {"plan_version": -1}
    with pytest.raises(ValueError, match="failed reconstruction"):
        _load_parent_states(frontier, [state_hash])


def test_template_lns_control_states_are_deduplicated_with_all_roles() -> None:
    state = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 2, (ExecutableExpertTask(3, 7),)),),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )

    states, roles = _deduplicate_named_states(
        {
            "full": state,
            "one_step": state,
            "fixed_width": state,
        }
    )

    assert states == {state.canonical_hash(): state}
    assert roles == {state.canonical_hash(): ["full", "one_step", "fixed_width"]}


def test_template_lns_analysis_adopts_stable_frontier_without_false_pruning() -> None:
    first = _session(frontier_stable=True, sentinel_stable=False)
    second = _session(frontier_stable=True, sentinel_stable=False)
    frontier = {
        "kind": "partial_order_hardware_frontier",
        "unique_plans": 3,
        "records": {
            "anchor": {"roles": ["anchor"]},
            "candidate": {"roles": ["incomparable_frontier"]},
            "sentinel": {"roles": ["candidate_worse_sentinel"]},
        },
        "plans": {
            state_hash: {
                "canonical_state": {
                    "lanes": [
                        {
                            "core_begin": 0,
                            "threads": 2,
                            "tasks": [{"expert_id": 0, "routes": 7}],
                        }
                    ]
                }
            }
            for state_hash in ("anchor", "candidate", "sentinel")
        },
    }
    model = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "summary": {"event_calls": 10, "search_wall_s": 2.0},
    }

    result = analyze_template_lns(frontier, model, first, second)

    assert result["decision"] == "adopt_template_lns_and_expand_consensus_elite"
    assert result["partial_order"]["stable_unique_candidates"] == 1
    assert result["partial_order"]["false_pruning_sentinels"] == 0
    assert result["winner"]["state_hash"] == "candidate"


def test_diagnostic_lns_frontier_roles_do_not_claim_acceptance_or_pruning(tmp_path) -> None:
    task = ExecutableExpertTask
    states = {
        "anchor": ExecutablePlanState(
            num_threads=2,
            thread_cpu_ids=(10, 11),
            lanes=(ExecutableLane(0, 2, (task(0, 7), task(1, 5), task(2, 3))),),
            llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
        ),
        "better": ExecutablePlanState(
            num_threads=2,
            thread_cpu_ids=(10, 11),
            lanes=(ExecutableLane(0, 2, (task(1, 5), task(0, 7), task(2, 3))),),
            llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
        ),
        "incomparable": ExecutablePlanState(
            num_threads=2,
            thread_cpu_ids=(10, 11),
            lanes=(ExecutableLane(0, 2, (task(0, 7), task(2, 3), task(1, 5))),),
            llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
        ),
        "worse": ExecutablePlanState(
            num_threads=2,
            thread_cpu_ids=(10, 11),
            lanes=(ExecutableLane(0, 2, (task(2, 3), task(1, 5), task(0, 7))),),
            llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
        ),
    }

    def candidate(name: str, relation: str) -> dict[str, object]:
        state = states[name]
        return {
            "state_hash": state.canonical_hash(),
            "operator": "critical_window_template_repartition_domain_local_d4_b16",
            "moved_experts": [0, 1, 2],
            "event_gain_pct": 1.0,
            "robust_gain_pct": 1.0,
            "partial_order": {"relation": relation},
            "canonical_state": state.canonical_payload(),
            "plan_v2_bridge": state.to_bridge(),
        }

    payload = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "route": {},
        "shape": {},
        "identity": {},
        "method": {"start_names": ["start"], "partial_order_shortlist_budget": 2},
        "runs": {
            "start": {
                "initial_state_hash": states["anchor"].canonical_hash(),
                "initial_canonical_state": states["anchor"].canonical_payload(),
                "initial_plan_v2_bridge": states["anchor"].to_bridge(),
                "iterations": [
                    {
                        "partial_order": {
                            "automatic_acceptance_enabled": False,
                            "dominance_pruning_enabled": False,
                        },
                        "selected_frontier": [
                            candidate("better", "candidate_better"),
                            candidate("incomparable", "incomparable"),
                        ],
                        "worse_sentinels": [candidate("worse", "candidate_worse")],
                    }
                ],
            }
        },
    }
    source = tmp_path / "model.json"
    source.write_text("{}", encoding="utf-8")

    frontier = build_frontier(payload, source_path=source)

    assert "model_better_frontier" in frontier["records"][states["better"].canonical_hash()]["roles"]
    assert "model_worse_spectrum" in frontier["records"][states["worse"].canonical_hash()]["roles"]
    assert frontier["role_counts"]["candidate_better_frontier"] == 0
    assert frontier["role_counts"]["candidate_worse_sentinel"] == 0


def test_template_lns_suite_separates_neighborhood_and_comparator_gates() -> None:
    def analysis(*, false_pruning: int = 0, false_acceptance: int = 0, decision: str = "adopt"):
        return {
            "kind": "template_lns_cross_session_analysis",
            "decision": decision,
            "winner": {
                "state_hash": "winner",
                "absolute_median_ms": [9.0, 9.1],
                "absolute_median_gain_vs_anchor_pct": {"anchor": [3.0, 2.5]},
                "session_absolute_best_state_hashes": ["winner", "winner"],
                "session_best_gap_pct": [0.0, 0.0],
            },
            "partial_order": {
                "stable_unique_candidates": 2,
                "false_pruning_sentinels": false_pruning,
                "false_acceptance_candidates": false_acceptance,
                "model_hardware_spearman": [0.5, 0.4],
            },
            "budget_comparison": {
                "template_lns": {
                    "event_calls": 10,
                    "model_wall_s": 2.0,
                    "hardware_plans": 5,
                }
            },
        }

    result = analyze_suite(
        analysis(),
        analysis(decision="retain_lns_frontier_without_automatic_expansion"),
        analysis(false_pruning=1),
        analysis(false_acceptance=1),
    )

    assert result["decision"] == "adopt_lns_neighborhood_disable_partial_order_decisions"
    assert result["gates"] == {
        "winner_over_strongest_anchor_both_sessions": True,
        "zero_false_pruning": False,
        "zero_false_acceptance": False,
        "high_skew_depth2_requires_another_expansion": False,
    }
    assert result["aggregate"]["event_calls"] == 40


def test_lns_diverse_builder_uses_nested_roles_and_rejects_sentinels(tmp_path: Path) -> None:
    task = ExecutableExpertTask
    anchor = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 2, (task(0, 7), task(1, 5))),),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )
    top16 = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 2, (task(1, 5), task(0, 7))),),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )
    audit_only = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 1, (task(0, 7),)), ExecutableLane(1, 1, (task(1, 5),))),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )

    def row(state: ExecutablePlanState) -> dict[str, object]:
        return {
            "state_hash": state.canonical_hash(),
            "operator": "critical_window_template_repartition_domain_local_d4_b16",
            "moved_experts": [0, 1],
            "event_gain_pct": 1.0,
            "robust_gain_pct": 1.0,
            "partial_order": {"relation": "candidate_worse"},
            "canonical_state": state.canonical_payload(),
            "plan_v2_bridge": state.to_bridge(),
        }

    payload = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "route": {},
        "shape": {},
        "identity": {},
        "method": {
            "start_names": ["start"],
            "shortlist_budget": 1,
            "audit_budget": 2,
        },
        "runs": {
            "start": {
                "initial_state_hash": anchor.canonical_hash(),
                "initial_canonical_state": anchor.canonical_payload(),
                "initial_plan_v2_bridge": anchor.to_bridge(),
                "iterations": [
                    {
                        "partial_order": {
                            "automatic_acceptance_enabled": False,
                            "dominance_pruning_enabled": False,
                        },
                        "lns_diverse_shortlist": {
                            "schema_version": 1,
                            "policy": "relation_agnostic_categorical_farthest_first_v1",
                            "shortlist_budget": 1,
                            "audit_budget": 2,
                            "ranked_keys": [top16.canonical_hash(), audit_only.canonical_hash()],
                            "selected_keys": [top16.canonical_hash()],
                            "audit_keys": [top16.canonical_hash(), audit_only.canonical_hash()],
                            "budget_deferred_keys": [],
                            "coverage": {},
                        },
                        "selected_frontier": [row(top16)],
                        "audit_frontier": [row(top16), row(audit_only)],
                        "worse_sentinels": [],
                    }
                ],
            }
        },
    }
    source = tmp_path / "model.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="diverse shortlists"):
        build_frontier(payload, source_path=source)

    frontier = build_lns_diverse_frontier(payload, source_path=source)
    assert "lns_diverse_top16" in frontier["records"][top16.canonical_hash()]["roles"]
    assert "lns_diverse_audit_top32" in frontier["records"][audit_only.canonical_hash()]["roles"]
    assert "anchor" in frontier["records"][anchor.canonical_hash()]["roles"]
    assert frontier["role_counts"]["lns_diverse_top16"] == 1
    assert frontier["lns_diverse_shortlist"]["selected_keys"] == [top16.canonical_hash()]


def test_lns_diverse_analyzer_requires_nested_top16_to_keep_measured_best() -> None:
    frontier = {
        "kind": "partial_order_hardware_frontier",
        "plans": {"anchor": {}, "best": {}, "other": {}},
        "records": {
            "anchor": {"roles": ["anchor"], "comparisons": []},
            "best": {
                "roles": ["lns_diverse_top16"],
                "comparisons": [{"role": "lns_diverse_top16", "partial_order": {"relation": "candidate_worse"}}],
            },
            "other": {
                "roles": ["lns_diverse_audit_top32"],
                "comparisons": [{"role": "lns_diverse_audit_top32", "partial_order": {}}],
            },
        },
        "identity": {},
        "method": {"policy": "relation_agnostic_categorical_farthest_first_v1"},
    }

    def session(best_ms: float, other_ms: float) -> dict[str, object]:
        return {
            "kind": "partial_order_hardware_frontier_session",
            "identity": {"frontier_sha256": "frontier"},
            "plan_stats": {
                "anchor": {"median_ms": 10.0},
                "best": {"median_ms": best_ms},
                "other": {"median_ms": other_ms},
            },
        }

    result = analyze_lns_diverse_sessions(frontier, session(8.0, 9.0), session(8.1, 9.1))
    assert result["gate"]["pass"] is True
    assert result["decision"] == "adopt_selector_v1_offline_lns_shortlist"

    frontier["records"]["best"]["roles"] = ["lns_diverse_audit_top32"]
    missed = analyze_lns_diverse_sessions(frontier, session(8.0, 9.0), session(8.1, 9.1))
    assert missed["gate"]["pass"] is False
