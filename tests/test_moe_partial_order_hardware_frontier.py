from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from analyze_lns_diverse_hardware_frontier import analyze_lns_diverse_sessions  # noqa: E402
from analyze_lns_second_proposal_seed import analyze_second_proposal_seed  # noqa: E402
from analyze_partial_order_hardware_frontier import analyze_sessions  # noqa: E402
from analyze_template_lns_frontier import analyze_template_lns  # noqa: E402
from analyze_template_lns_suite import analyze_suite  # noqa: E402
from bench_template_lns_layer import (  # noqa: E402
    MODEL_STAGE_TIMER_FIELDS,
    _deduplicate_named_states,
    _load_parent_states,
    assemble_template_lns_post_search,
    reconcile_model_stage_timing,
    write_template_lns_model_artifact,
)
from build_lns_diverse_hardware_frontier import build_lns_diverse_frontier  # noqa: E402
from build_partial_order_hardware_frontier import build_frontier  # noqa: E402
from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)
from lns_diverse_shortlist import (  # noqa: E402
    build_candidate_feature,
    select_lns_diverse_shortlist,
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


def test_known_elite_is_an_extra_control_without_dropping_lns_roles(tmp_path: Path) -> None:
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
    elite = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 1, (task(1, 5),)), ExecutableLane(1, 1, (task(0, 7),))),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )

    def row(state: ExecutablePlanState) -> dict[str, object]:
        return {
            "state_hash": state.canonical_hash(),
            "operator": "critical_window_template_repartition_cross_domain_d4_b16",
            "moved_experts": [1],
            "event_gain_pct": 1.0,
            "robust_gain_pct": 1.0,
            "canonical_state": state.canonical_payload(),
            "plan_v2_bridge": state.to_bridge(),
        }

    payload = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "route": {},
        "shape": {},
        "identity": {"parent_source": {"named_state_hashes": {"full": anchor.canonical_hash()}}},
        "method": {
            "start_names": ["lns_00_r00"],
            "shortlist_budget": 1,
            "audit_budget": 2,
        },
        "parents": {"lns_00_r00": {"state_hash": anchor.canonical_hash(), "roles": ["full"]}},
        "runs": {
            "lns_00_r00": {
                "start": "lns_00_r00",
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
                            "budget_deferred_keys": [audit_only.canonical_hash()],
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
    source.write_text("{}\n", encoding="utf-8")
    elite_plan = {
        "state_hash": elite.canonical_hash(),
        "canonical_state": elite.canonical_payload(),
        "plan_v2_bridge": elite.to_bridge(),
    }
    with_audit = build_lns_diverse_frontier(payload, source_path=source, known_elite=elite_plan)
    assert with_audit["unique_plans"] == 4
    assert "known_hardware_elite" in with_audit["records"][elite.canonical_hash()]["roles"]
    assert "lns_diverse_top16" in with_audit["records"][top16.canonical_hash()]["roles"]
    no_audit = build_lns_diverse_frontier(
        payload,
        source_path=source,
        known_elite=elite_plan,
        include_audit=False,
    )
    assert audit_only.canonical_hash() not in no_audit["plans"]
    assert elite.canonical_hash() in no_audit["plans"]
    assert no_audit["unique_plans"] == 3


def test_reference_plan_is_extra_control_without_dropping_lns_roles(tmp_path: Path) -> None:
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
    reference = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 1, (task(0, 7),)), ExecutableLane(1, 1, (task(1, 5),))),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )
    elite = ExecutablePlanState(
        num_threads=2,
        thread_cpu_ids=(10, 11),
        lanes=(ExecutableLane(0, 1, (task(1, 5),)), ExecutableLane(1, 1, (task(0, 7),))),
        llc_domains=(ExecutableLlcDomain("llc", 0, 2),),
    )

    def row(state: ExecutablePlanState) -> dict[str, object]:
        return {
            "state_hash": state.canonical_hash(),
            "operator": "critical_window_template_repartition_cross_domain_d4_b16",
            "moved_experts": [1],
            "event_gain_pct": 1.0,
            "robust_gain_pct": 1.0,
            "canonical_state": state.canonical_payload(),
            "plan_v2_bridge": state.to_bridge(),
        }

    payload = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "route": {},
        "shape": {},
        "identity": {"parent_source": {"named_state_hashes": {"full": anchor.canonical_hash()}}},
        "method": {
            "start_names": ["lns_00_r00"],
            "shortlist_budget": 1,
            "audit_budget": 1,
        },
        "parents": {"lns_00_r00": {"state_hash": anchor.canonical_hash(), "roles": ["full"]}},
        "runs": {
            "lns_00_r00": {
                "start": "lns_00_r00",
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
                            "audit_budget": 1,
                            "ranked_keys": [top16.canonical_hash()],
                            "selected_keys": [top16.canonical_hash()],
                            "audit_keys": [top16.canonical_hash()],
                            "budget_deferred_keys": [],
                            "coverage": {},
                        },
                        "selected_frontier": [row(top16)],
                        "audit_frontier": [row(top16)],
                        "worse_sentinels": [],
                    }
                ],
            }
        },
    }
    source = tmp_path / "model.json"
    source.write_text("{}\n", encoding="utf-8")
    elite_plan = {
        "state_hash": elite.canonical_hash(),
        "canonical_state": elite.canonical_payload(),
        "plan_v2_bridge": elite.to_bridge(),
    }
    reference_plan = {
        "state_hash": reference.canonical_hash(),
        "canonical_state": reference.canonical_payload(),
        "plan_v2_bridge": reference.to_bridge(),
    }
    frontier = build_lns_diverse_frontier(
        payload,
        source_path=source,
        known_elite=elite_plan,
        reference_plan=reference_plan,
        include_audit=False,
    )
    assert frontier["unique_plans"] == 4
    assert "known_hardware_elite" in frontier["records"][elite.canonical_hash()]["roles"]
    assert "previous_seed_selected" in frontier["records"][reference.canonical_hash()]["roles"]
    assert "lns_diverse_top16" in frontier["records"][top16.canonical_hash()]["roles"]
    assert frontier["method"]["reference_state_hash"] == reference.canonical_hash()


def test_second_seed_analyzer_reports_overlap_and_reference_gain() -> None:
    current_frontier = {
        "plans": {"anchor": {}, "new": {}, "shared": {}, "elite": {}, "old": {}, "outside": {}},
        "records": {
            "anchor": {"roles": ["anchor"]},
            "new": {"roles": ["lns_diverse_top16"]},
            "shared": {"roles": ["lns_diverse_top16"]},
            "elite": {"roles": ["known_hardware_elite"]},
            "old": {"roles": ["previous_seed_selected"]},
            "outside": {"roles": ["lns_stratified_outside_top32"]},
        },
        "role_counts": {
            "anchor": 1,
            "lns_diverse_top16": 64,
            "lns_diverse_audit_top32": 0,
            "lns_stratified_outside_top32": 16,
            "known_hardware_elite": 1,
            "previous_seed_selected": 1,
        },
    }
    previous_frontier = {
        "records": {
            "shared": {"roles": ["lns_diverse_top16"]},
            "old": {"roles": ["lns_diverse_top16"]},
        }
    }
    current_model = {
        "summary": {
            "starts": 8,
            "unique_candidates": 2300,
            "event_calls": 2380,
            "search_wall_s": 500.0,
        },
        "method": {
            "restarts_per_parent": 2,
            "neighbors_per_operator": 25,
            "maximum_exact_candidate_budget": 2400,
            "seed": 20261011,
        },
    }
    previous_model = {
        "summary": {
            "starts": 8,
            "unique_candidates": 2368,
            "event_calls": 2384,
            "search_wall_s": 558.71,
        },
        "method": {
            "restarts_per_parent": 2,
            "neighbors_per_operator": 25,
            "maximum_exact_candidate_budget": 2400,
            "seed": 20261010,
        },
    }

    def session(*, new_ms: float, old_ms: float, elite_ms: float) -> dict[str, object]:
        return {
            "summary": {"measurement_wall_s": 100.0},
            "plan_stats": {
                "anchor": {"median_ms": 32.0},
                "new": {"median_ms": new_ms},
                "shared": {"median_ms": 31.4},
                "elite": {"median_ms": elite_ms},
                "old": {"median_ms": old_ms},
                "outside": {"median_ms": 31.8},
            },
        }

    result = analyze_second_proposal_seed(
        current_frontier,
        session(new_ms=31.0, old_ms=31.2, elite_ms=31.3),
        session(new_ms=31.05, old_ms=31.25, elite_ms=31.35),
        current_model,
        previous_frontier,
        previous_model,
    )
    assert result["protocol_ok"] is True
    assert result["selected_overlap"]["overlap_count"] == 1
    assert result["selected_overlap"]["overlap_keys"] == ["shared"]
    assert result["current"]["two_session_selected_gain_vs_anchor_over_2pct"] is True
    assert result["current"]["two_session_selected_beats_elite"] is True
    assert result["reference"]["two_session_selected_beats_reference"] is True
    assert result["current"]["session_1"]["selected_best_state_hash"] == "new"

    current_frontier["role_counts"]["lns_diverse_top16"] = 63
    broken = analyze_second_proposal_seed(
        current_frontier,
        session(new_ms=31.0, old_ms=31.2, elite_ms=31.3),
        session(new_ms=31.05, old_ms=31.25, elite_ms=31.35),
        current_model,
        previous_frontier,
        previous_model,
    )
    assert broken["protocol_ok"] is False


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


class _FrontierState:
    def __init__(self, key: str) -> None:
        self.key = key
        self.shape = (8, 8)

    def canonical_payload(self) -> dict[str, object]:
        return {"key": self.key}

    def to_bridge(self) -> dict[str, object]:
        return {"bridge": self.key}


def test_template_lns_layer_records_nested_model_stage_timers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = "critical_window_template_repartition_cross_domain_d4_b16"
    task = ExecutableExpertTask
    anchor = ExecutablePlanState(
        num_threads=8,
        thread_cpu_ids=tuple(range(8)),
        lanes=(
            ExecutableLane(0, 4, (task(0, 20), task(1, 10))),
            ExecutableLane(4, 4, (task(2, 12), task(3, 8), task(4, 6))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 4),
            ExecutableLlcDomain("right", 4, 4),
        ),
        early_merge=False,
    )
    local = ExecutablePlanState(
        num_threads=8,
        thread_cpu_ids=tuple(range(8)),
        lanes=(
            ExecutableLane(0, 2, (task(0, 20),)),
            ExecutableLane(2, 2, (task(1, 10),)),
            ExecutableLane(4, 4, (task(2, 12), task(3, 8), task(4, 6))),
        ),
        llc_domains=anchor.llc_domains,
        early_merge=False,
    )
    first = [
        build_candidate_feature(
            local,
            anchor,
            operator=operator,
            moved_experts=[0, 1, 2, 3],
            predicted_gain_pct=1.5,
            start="lns_00_r00",
            strategy="critical",
        )
    ]
    extra = [
        build_candidate_feature(
            local,
            anchor,
            operator=operator,
            moved_experts=[0, 1],
            predicted_gain_pct=3.0,
            start="lns_00_r01",
            strategy="random",
        )
    ]
    start_a = select_lns_diverse_shortlist(first, shortlist_budget=2, audit_budget=3)
    start_b = select_lns_diverse_shortlist(extra, shortlist_budget=2, audit_budget=3)
    parent = "parent-hash"

    def run_payload(shortlist, features) -> dict[str, object]:
        keys = list(shortlist.ranked_keys)
        rows = {}
        states = {}
        for key in keys:
            states[key] = _FrontierState(key)
            rows[key] = (
                "critical",
                {
                    "operator": operator,
                    "moved_experts": [1],
                    "event_ns": 1.0,
                    "robust_ns": 1.0,
                    "event_gain_pct": 0.0,
                    "robust_gain_pct": 0.0,
                },
            )
        return {
            "iterations": [
                {
                    "unique_candidates": len(keys),
                    "lns_diverse_shortlist": shortlist.to_dict(),
                    "candidate_features": [item.to_dict() for item in features],
                    "partial_order": {
                        "relation_counts": {
                            "candidate_better": 0,
                            "candidate_worse": 0,
                            "incomparable": len(keys),
                        }
                    },
                    "search_breakdown": {
                        "shortlist_s": 0.5,
                        "shortlist_components": {
                            "feature_construction_s": 0.1,
                            "nesting": "nested_in_shortlist_s",
                            "partial_order_evidence_selection_s": 0.2,
                            "per_start_diverse_ranking_s": 0.15,
                            "quantile_dedup_s": 0.04,
                            "unaccounted_s": 0.01,
                        },
                    },
                }
            ],
            "exact_event_calls": 1,
            "search_wall_s": 1.25,
            "_states": states,
            "_rows": rows,
        }

    run_a = run_payload(start_a, first)
    run_b = run_payload(start_b, extra)
    libraries = {
        "lns_00_r00": {"states": run_a.pop("_states"), "rows": run_a.pop("_rows")},
        "lns_00_r01": {"states": run_b.pop("_states"), "rows": run_b.pop("_rows")},
    }
    parents = {
        "lns_00_r00": {"state_hash": parent, "roles": ["full"]},
        "lns_00_r01": {"state_hash": parent, "roles": ["full"]},
    }
    assembled = assemble_template_lns_post_search(
        runs={"lns_00_r00": run_a, "lns_00_r01": run_b},
        parents=parents,
        libraries=libraries,
        shortlist_budget=2,
        audit_budget=3,
        pool_restarts_by_parent=True,
        stratified_outside_audit=1,
        stratified_seed=20261013,
    )
    for field in (
        "restart_pooling_s",
        "parent_pooled_ranking_s",
        "outside_audit_sampling_s",
        "frontier_construction_s",
        "global_shortlist_s",
    ):
        assert field in assembled
        assert float(assembled[field]) >= 0.0
    assert assembled["global_shortlist"]["selected_keys"]
    assert assembled["pooled_shortlists"][parent].ranked_keys
    assert "shortlist_s" not in assembled
    components = {
        "control_construction_s": 0.2,
        "search_wall_s": 2.5,
        "restart_pooling_s": float(assembled["restart_pooling_s"]),
        "parent_pooled_ranking_s": float(assembled["parent_pooled_ranking_s"]),
        "outside_audit_sampling_s": float(assembled["outside_audit_sampling_s"]),
        "frontier_construction_s": float(assembled["frontier_construction_s"]),
        "global_shortlist_s": float(assembled["global_shortlist_s"]),
        "serialization_s": 0.0,
    }
    assert tuple(components) == MODEL_STAGE_TIMER_FIELDS
    output = tmp_path / "model.json"
    result = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "summary": dict(components),
        "lns_diverse_shortlist": assembled["global_shortlist"],
    }
    write_delay_s = 0.05
    timing_bearing_finish_ns: dict[str, int] = {}
    original_write_text = Path.write_text

    def delayed_write(self: Path, data: str, *args: object, **kwargs: object) -> None:
        if self.resolve() == output.resolve():
            payload = json.loads(data)
            summary = payload.get("summary") or {}
            timing_bearing = (
                float(summary.get("serialization_s") or 0.0) != 0.0
                or float(summary.get("model_command_elapsed_s") or 0.0) != 0.0
            )
            if timing_bearing:
                time.sleep(write_delay_s)
            original_write_text(self, data, *args, **kwargs)
            if timing_bearing:
                timing_bearing_finish_ns["model"] = time.perf_counter_ns()
            return
        original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", delayed_write)
    begin = time.perf_counter_ns()
    timings = write_template_lns_model_artifact(output, result, command_begin_ns=begin)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["global_shortlist_s"] == components["global_shortlist_s"]
    assert payload["summary"]["search_wall_s"] == 2.5
    recon = payload["summary"]["timing_reconciliation"]
    assert "unaccounted_s" in recon
    assert recon["nested"]["shortlist_s"] == "nested_in_search_wall_s; not ranking-only"
    assert "model" in timing_bearing_finish_ns
    assert timings["serialization_s"] >= write_delay_s
    assert timings["model_command_elapsed_s"] >= timings["serialization_s"]
    clock_stop_ns = begin + int(timings["model_command_elapsed_s"] * 1_000_000_000)
    assert clock_stop_ns + 1_000_000 >= timing_bearing_finish_ns["model"]
    sidecar = json.loads((tmp_path / "model.json.wall.json").read_text(encoding="utf-8"))
    assert sidecar["model_command_elapsed_s"] == timings["model_command_elapsed_s"]
    assert sidecar["serialization_s"] == timings["serialization_s"]
    direct = reconcile_model_stage_timing(elapsed_s=3.0, components=components)
    assert abs(direct["accounted_s"] + direct["unaccounted_s"] - 3.0) < 1.0e-9
