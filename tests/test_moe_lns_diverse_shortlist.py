from __future__ import annotations

import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
PLANNER_DIR = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(BENCHMARK_DIR), str(PLANNER_DIR)]

from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)
from compare_lns_frozen_ranking import compare_artifacts  # noqa: E402
from replay_lns_frozen_ranking import replay_frozen_ranking, replayed_model_payload  # noqa: E402
from lns_diverse_shortlist import (  # noqa: E402
    DEFAULT_AUDIT_BUDGET,
    DEFAULT_SHORTLIST_BUDGET,
    POLICY_NAME,
    RANKING_FILL_NAME,
    RANKING_FILL_REFERENCE_NAME,
    AnchorRecoveryError,
    LnsCandidateFeature,
    assign_model_score_quantiles,
    build_candidate_feature,
    compact_structure,
    merge_duplicate_features,
    policy_document,
    policy_sha256,
    rank_lns_diverse_candidates,
    recover_anchor_domain_signature,
    recover_run_features,
    select_lns_diverse_shortlist,
    select_lns_global_shortlist,
    select_lns_parent_pooled_shortlists,
    state_from_canonical_payload,
    stratified_keys_outside_audit,
    token_reuse,
    _farthest_first_fill,
    _farthest_first_fill_reference,
)


OPERATORS = (
    "critical_window_template_repartition_cross_domain_d16_b64",
    "critical_window_template_repartition_cross_domain_d4_b16",
    "critical_window_template_repartition_cross_domain_d8_b32",
    "critical_window_template_repartition_domain_local_d16_b64",
    "critical_window_template_repartition_domain_local_d4_b16",
    "critical_window_template_repartition_domain_local_d8_b32",
)


def _anchor() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
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


def _split_local() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=8,
        thread_cpu_ids=tuple(range(8)),
        lanes=(
            ExecutableLane(0, 2, (task(0, 20),)),
            ExecutableLane(2, 2, (task(1, 10),)),
            ExecutableLane(4, 4, (task(2, 12), task(3, 8), task(4, 6))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 4),
            ExecutableLlcDomain("right", 4, 4),
        ),
        early_merge=False,
    )


def _split_cross() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=8,
        thread_cpu_ids=tuple(range(8)),
        lanes=(
            ExecutableLane(0, 4, (task(0, 20), task(1, 10))),
            ExecutableLane(4, 2, (task(2, 12), task(3, 8))),
            ExecutableLane(6, 2, (task(4, 6),)),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 4),
            ExecutableLlcDomain("right", 4, 4),
        ),
        early_merge=False,
    )


def _wide_cross() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=8,
        thread_cpu_ids=tuple(range(8)),
        lanes=(
            ExecutableLane(0, 8, (task(0, 20), task(1, 10), task(2, 12), task(3, 8), task(4, 6))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 4),
            ExecutableLlcDomain("right", 4, 4),
        ),
        early_merge=False,
    )


def _salted(state: ExecutablePlanState, salt: int) -> ExecutablePlanState:
    lanes = []
    for lane_index, lane in enumerate(state.lanes):
        tasks = []
        for task_index, task in enumerate(lane.tasks):
            w13 = salt if lane_index == 0 and task_index == 0 else task.w13_window_tiles
            tasks.append(
                ExecutableExpertTask(task.expert_id, task.routes, w13, task.w2_window_tiles)
            )
        lanes.append(ExecutableLane(lane.core_begin, lane.threads, tuple(tasks)))
    return ExecutablePlanState(
        num_threads=state.num_threads,
        thread_cpu_ids=state.thread_cpu_ids,
        lanes=tuple(lanes),
        llc_domains=state.llc_domains,
        early_merge=state.early_merge,
    )


def _reordered() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=8,
        thread_cpu_ids=tuple(range(8)),
        lanes=(
            ExecutableLane(0, 4, (task(1, 10), task(0, 20))),
            ExecutableLane(4, 4, (task(4, 6), task(3, 8), task(2, 12))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 4),
            ExecutableLlcDomain("right", 4, 4),
        ),
        early_merge=False,
    )


def _feature(
    candidate: ExecutablePlanState,
    *,
    operator: str,
    moved: SequenceLike,
    gain: float,
    start: str = "lns_00_r00",
    strategy: str = "critical",
    anchor: ExecutablePlanState | None = None,
) -> LnsCandidateFeature:
    parent = anchor or _anchor()
    return build_candidate_feature(
        candidate,
        parent,
        operator=operator,
        moved_experts=list(moved),
        predicted_gain_pct=gain,
        start=start,
        strategy=strategy,
    )


SequenceLike = list[int] | tuple[int, ...]


def _pool() -> list[object]:
    states = {
        "local": _split_local(),
        "cross": _split_cross(),
        "wide": _wide_cross(),
        "order": _reordered(),
        "anchor": _anchor(),
    }
    moved_sizes = {
        OPERATORS[0]: list(range(16)),
        OPERATORS[1]: list(range(91)),
        OPERATORS[2]: list(range(8)),
        OPERATORS[3]: list(range(16)),
        OPERATORS[4]: list(range(4)),
        OPERATORS[5]: list(range(24)),
    }
    state_cycle = ("local", "cross", "wide", "order", "local", "cross")
    features = []
    salt = 1
    for index, operator in enumerate(OPERATORS):
        for copy in range(3):
            gain = -4.0 + index + copy * 1.7
            strategy = "critical" if copy % 2 == 0 else "random"
            moved = (
                moved_sizes[operator]
                if copy != 1
                else moved_sizes[operator][: max(1, len(moved_sizes[operator]) // 2)]
            )
            features.append(
                _feature(
                    _salted(states[state_cycle[(index + copy) % 4]], salt),
                    operator=operator,
                    moved=moved,
                    gain=gain,
                    start="lns_00_r00",
                    strategy=strategy,
                )
            )
            salt += 1
    return features


def test_selection_is_deterministic_under_shuffled_input() -> None:
    features = _pool()
    baseline = select_lns_diverse_shortlist(features)
    hashes = []
    for seed in (1, 2, 3):
        shuffled = list(features)
        random.Random(seed).shuffle(shuffled)
        selected = select_lns_diverse_shortlist(shuffled)
        hashes.append(tuple(selected.selected_keys))
        assert selected.selected_keys == baseline.selected_keys
        assert selected.audit_keys == baseline.audit_keys
    assert len(set(hashes)) == 1


def test_canonical_duplicates_merge_and_global_backfill() -> None:
    parent = _anchor()
    candidate = _split_local()
    first = _feature(candidate, operator=OPERATORS[1], moved=list(range(4)), gain=-3.0, strategy="random")
    second = _feature(
        candidate,
        operator=OPERATORS[1],
        moved=list(range(4)),
        gain=-3.0,
        strategy="critical",
        start="lns_00_r00",
    )
    merged = merge_duplicate_features([first, second])
    assert len(merged) == 1
    assert merged[0].strategy == "critical"
    assert set(merged[0].provenance) == {
        ("lns_00_r00", "random", OPERATORS[1]),
        ("lns_00_r00", "critical", OPERATORS[1]),
    }

    other = _feature(
        _split_cross(),
        operator=OPERATORS[2],
        moved=list(range(8)),
        gain=1.0,
        start="lns_01_r00",
        anchor=parent,
    )
    first_start = select_lns_diverse_shortlist([first, second], shortlist_budget=1, audit_budget=2)
    second_start = select_lns_diverse_shortlist([other, first], shortlist_budget=1, audit_budget=2)
    global_result = select_lns_global_shortlist(
        {"lns_00_r00": first_start, "lns_01_r00": second_start},
        shortlist_budget=1,
        audit_budget=2,
    )
    assert global_result["selected_keys"][0] == candidate.canonical_hash()
    assert other.state_hash in global_result["selected_keys"]
    assert global_result["selected_duplicate_counts"]["lns_01_r00"]["duplicate"] == 1
    assert global_result["selected_duplicate_counts"]["lns_01_r00"]["exhausted"] is False
    assert other.state_hash in global_result["audit_keys"]


def test_every_available_operator_is_represented_when_budget_allows() -> None:
    selected = select_lns_diverse_shortlist(_pool(), shortlist_budget=16, audit_budget=32)
    assert set(selected.coverage["operators"]) == set(OPERATORS)
    assert set(selected.coverage["scopes"]) == {"cross_domain", "domain_local"}
    assert set(selected.coverage["target_destroy_sizes"]) == {4, 8, 16}


def test_quantile_zero_and_four_are_represented_when_available() -> None:
    selected = select_lns_diverse_shortlist(_pool(), shortlist_budget=16, audit_budget=32)
    assert 0 in selected.coverage["score_quantiles"]
    assert 4 in selected.coverage["score_quantiles"]


def test_actual_closure_and_target_destroy_stay_distinct() -> None:
    d4 = _feature(_split_cross(), operator=OPERATORS[1], moved=list(range(91)), gain=-3.7)
    d16 = _feature(_split_local(), operator=OPERATORS[0], moved=list(range(16)), gain=1.2)
    selected = select_lns_diverse_shortlist([d4, d16], shortlist_budget=2, audit_budget=2)
    features = {item.state_hash: item for item in [d4, d16]}
    assert {features[key].target_destroy_size for key in selected.selected_keys} == {4, 16}
    assert {features[key].actual_closure_bin for key in selected.selected_keys} == {"16_31", "64_127"}
    assert d4.actual_closure_size == 91
    assert d4.target_destroy_size == 4


def test_top16_is_an_exact_prefix_of_top32() -> None:
    selected = select_lns_diverse_shortlist(_pool(), shortlist_budget=16, audit_budget=32)
    assert selected.selected_keys == selected.audit_keys[:16]
    assert list(selected.selected_keys) == list(selected.ranked_keys[: DEFAULT_SHORTLIST_BUDGET])
    assert list(selected.audit_keys) == list(selected.ranked_keys[: DEFAULT_AUDIT_BUDGET])
    smaller = select_lns_diverse_shortlist(_pool(), shortlist_budget=8, audit_budget=12)
    assert smaller.selected_keys == selected.ranked_keys[:8]
    assert smaller.audit_keys == selected.ranked_keys[:12]


def test_width_and_domain_signatures_track_executable_structure_only() -> None:
    parent = _anchor()
    local = compact_structure(_split_local())
    cross = compact_structure(_split_cross())
    wide = compact_structure(_wide_cross())
    order = compact_structure(_reordered())
    assert local["domain_assignment_signature"] != wide["domain_assignment_signature"]
    assert local["candidate_width_histogram"] == cross["candidate_width_histogram"]
    assert local["domain_assignment_signature"] != cross["domain_assignment_signature"]
    assert order["candidate_width_histogram"] == compact_structure(parent)["candidate_width_histogram"]
    assert order["domain_assignment_signature"] == compact_structure(parent)["domain_assignment_signature"]
    metadata_changed = build_candidate_feature(
        _split_local(),
        parent,
        operator=OPERATORS[4],
        moved_experts=[0, 1],
        predicted_gain_pct=0.5,
        start="lns_00_r00",
        strategy="critical",
    )
    same_structure_other_meta = build_candidate_feature(
        _split_local(),
        parent,
        operator=OPERATORS[4],
        moved_experts=[0, 1],
        predicted_gain_pct=9.5,
        start="lns_00_r01",
        strategy="random",
    )
    assert metadata_changed.domain_assignment_signature == same_structure_other_meta.domain_assignment_signature
    assert metadata_changed.candidate_width_histogram == same_structure_other_meta.candidate_width_histogram
    assert "relation" not in metadata_changed.to_dict()
    assert "residual_radius_pct" not in metadata_changed.to_dict()


def test_relation_values_do_not_change_selected_or_deferred_semantics() -> None:
    features = _pool()
    baseline = select_lns_diverse_shortlist(features)
    labeled = []
    for index, feature in enumerate(features):
        labeled.append(feature)
        feature.to_dict()["relation"] = "candidate_worse" if index % 2 else "candidate_better"
    relabeled = select_lns_diverse_shortlist(labeled)
    assert relabeled.selected_keys == baseline.selected_keys
    assert relabeled.budget_deferred_keys == baseline.budget_deferred_keys
    assert "dominated_keys" not in relabeled.to_dict()
    assert relabeled.policy == POLICY_NAME


def test_shortlist_never_labels_dominated_or_automatically_accepted() -> None:
    selected = select_lns_diverse_shortlist(_pool())
    payload = selected.to_dict()
    assert payload["policy"] == POLICY_NAME
    assert "dominated_keys" not in payload
    assert "accepted_key" not in payload
    assert payload["selected_keys"]
    assert set(payload["budget_deferred_keys"]).isdisjoint(payload["selected_keys"])


def test_audit_plan_state_and_bridge_round_trip() -> None:
    state = _wide_cross()
    payload = state.canonical_payload()
    bridge = state.to_bridge()
    restored = state_from_canonical_payload(payload)
    assert restored.canonical_hash() == state.canonical_hash()
    assert restored.canonical_payload() == payload
    assert restored.to_bridge() == bridge
    assert restored.lane_domain_ids(0) == ("left", "right")


def test_frozen_policy_rule_hash_matches_selector() -> None:
    path = (
        ROOT
        / "optimizations"
        / "fused_moe_sve"
        / "benchmarks"
        / "profiles"
        / "lns_diverse_shortlist_relation_agnostic_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["policy"] == POLICY_NAME
    assert payload["shortlist_budget"] == DEFAULT_SHORTLIST_BUDGET
    assert payload["audit_budget"] == DEFAULT_AUDIT_BUDGET
    assert payload["rule_sha256"] == policy_sha256(policy_document())
    assert payload["hardware_weights"] is None
    assert payload["relation_agnostic"] is True


def _ranking_artifact(gain: float = 1.25) -> dict[str, object]:
    feature = {
        "actual_closure_bin": "16_31",
        "actual_closure_size": 18,
        "anchor_state_hash": "anchor",
        "candidate_width_histogram": [[8, 4]],
        "changed_core_begin": 0,
        "changed_core_end": 32,
        "cross_domain_lane_count": 0,
        "domain_assignment": {"left": 4},
        "domain_assignment_signature": "left:4",
        "model_score_quantile": 4,
        "operator": OPERATORS[0],
        "predicted_gain_pct": gain,
        "restart": 0,
        "scope": "cross_domain",
        "start": "lns_00_r00",
        "state_hash": "cand",
        "strategy": "critical",
        "target_destroy_size": 16,
        "width_histogram_delta": [[8, 1]],
    }
    frontier = {
        "state_hash": "cand",
        "event_ns": 1.0,
        "robust_ns": 1.1,
        "event_gain_pct": gain,
        "robust_gain_pct": gain,
        "operator": OPERATORS[0],
        "strategy": "critical",
    }
    return {
        "identity": {
            "calibration_sha256": "cal",
            "pairwise_calibration_sha256": "pair",
            "extension_sha256": "ext",
            "parent_source": {"named_state_hashes": {"full": "anchor"}},
        },
        "method": {
            "seed": 20261010,
            "restarts_per_parent": 1,
            "lns_shortlist_policy_sha256": "policy",
        },
        "summary": {
            "unique_candidates": 1,
            "event_calls": 2,
            "better": 0,
            "worse": 0,
            "incomparable": 1,
            "search_wall_s": 12.0,
        },
        "lns_diverse_shortlist": {
            "selected_keys": ["cand"],
            "audit_keys": ["cand"],
        },
        "runs": {
            "lns_00_r00": {
                "iterations": [
                    {
                        "unique_candidates": 1,
                        "operators": {OPERATORS[0]: {"proposed": 2, "unique": 1, "sampled": 1}},
                        "critical_expert_ids": [1],
                        "random_expert_ids": [2],
                        "lns_diverse_shortlist": {
                            "selected_keys": ["cand"],
                            "audit_keys": ["cand"],
                            "ranked_keys": ["cand"],
                        },
                        "candidate_features": [feature],
                        "selected_frontier": [frontier],
                        "audit_frontier": [frontier],
                    }
                ]
            }
        },
    }


def test_parent_pooled_shortlist_matches_union_selector() -> None:
    first = _pool()
    extra = _feature(
        _salted(_split_local(), 99),
        operator=OPERATORS[0],
        moved=list(range(16)),
        gain=12.0,
        start="lns_00_r01",
    )
    parent = "parent-hash"
    pooled = select_lns_parent_pooled_shortlists(
        {"lns_00_r00": first, "lns_00_r01": [extra]},
        {"lns_00_r00": parent, "lns_00_r01": parent},
        shortlist_budget=16,
        audit_budget=32,
    )
    union = select_lns_diverse_shortlist([*first, extra], shortlist_budget=16, audit_budget=32)
    assert pooled[parent].selected_keys == union.selected_keys
    assert pooled[parent].audit_keys == union.audit_keys
    assert extra.state_hash in pooled[parent].ranked_keys


def test_one_restart_pool_matches_per_start_shortlist() -> None:
    features = _pool()
    parent = "parent-hash"
    pooled = select_lns_parent_pooled_shortlists(
        {"lns_00_r00": features},
        {"lns_00_r00": parent},
    )
    per_start = select_lns_diverse_shortlist(features)
    assert pooled[parent].selected_keys == per_start.selected_keys
    assert pooled[parent].audit_keys == per_start.audit_keys


def test_stratified_outside_audit_is_disjoint_nested_and_deterministic() -> None:
    features = _pool()
    selected = select_lns_diverse_shortlist(features, shortlist_budget=4, audit_budget=8)
    first = stratified_keys_outside_audit(
        features,
        selected.ranked_keys,
        selected.audit_keys,
        sample_size=4,
        seed=20261013,
    )
    second = stratified_keys_outside_audit(
        [replace(item, predicted_gain_pct=item.predicted_gain_pct) for item in features],
        selected.ranked_keys,
        selected.audit_keys,
        sample_size=4,
        seed=20261013,
    )
    assert len(first) == 4
    assert first == second
    assert set(first).isdisjoint(selected.audit_keys)
    assert all(key in selected.ranked_keys for key in first)


def test_feature_from_dict_round_trips_selector_fields() -> None:
    feature = _pool()[0]
    restored = LnsCandidateFeature.from_dict(feature.to_dict())
    assert restored.state_hash == feature.state_hash
    assert restored.operator == feature.operator
    assert restored.predicted_gain_pct == feature.predicted_gain_pct
    assert restored.candidate_width_histogram == feature.candidate_width_histogram
    assert restored.anchor_domain_assignment_signature == feature.anchor_domain_assignment_signature
    assert "anchor_domain_assignment_signature" in feature.to_dict()


def test_frozen_ranking_compare_ignores_search_wall_and_detects_score_drift() -> None:
    baseline = _ranking_artifact()
    faster = _ranking_artifact()
    faster["summary"]["search_wall_s"] = 4.0
    faster["summary"]["search_breakdown"] = {"beam_s": 1.0}
    faster["method"]["lns_ranking_fill"] = RANKING_FILL_NAME
    assert compare_artifacts(baseline, faster)["equal"] is True

    drifted = _ranking_artifact(gain=0.5)
    result = compare_artifacts(baseline, drifted)
    assert result["equal"] is False
    assert any("predicted_gain_pct" in item["path"] for item in result["mismatches"])


def _ranked_keys(features, fill: str) -> list[str]:
    return [item.state_hash for item in rank_lns_diverse_candidates(features, fill=fill)]


def test_incremental_fill_matches_reference_on_duplicates_ties_missing_and_shuffle() -> None:
    features = _pool()
    unique = assign_model_score_quantiles(merge_duplicate_features(features))
    assert [item.state_hash for item in _farthest_first_fill(unique)] == [
        item.state_hash for item in _farthest_first_fill_reference(unique)
    ]
    assert _ranked_keys(features, RANKING_FILL_NAME) == _ranked_keys(
        features, RANKING_FILL_REFERENCE_NAME
    )
    for seed in (1, 2, 3, 11):
        shuffled = list(features)
        random.Random(seed).shuffle(shuffled)
        assert _ranked_keys(shuffled, RANKING_FILL_NAME) == _ranked_keys(
            shuffled, RANKING_FILL_REFERENCE_NAME
        )
        assert _ranked_keys(shuffled, RANKING_FILL_NAME) == _ranked_keys(
            features, RANKING_FILL_NAME
        )

    duplicate = features + [features[0]]
    assert _ranked_keys(duplicate, RANKING_FILL_NAME) == _ranked_keys(
        duplicate, RANKING_FILL_REFERENCE_NAME
    )

    tied = [
        _feature(
            _salted(_split_local(), 1),
            operator=OPERATORS[4],
            moved=list(range(4)),
            gain=1.25,
            start="lns_00_r00",
        ),
        _feature(
            _salted(_split_local(), 2),
            operator=OPERATORS[4],
            moved=list(range(4)),
            gain=1.25,
            start="lns_00_r00",
        ),
    ]
    assert tied[0].state_hash != tied[1].state_hash
    assert _ranked_keys(tied, RANKING_FILL_NAME) == _ranked_keys(tied, RANKING_FILL_REFERENCE_NAME)

    missing = [
        _feature(_split_local(), operator=OPERATORS[4], moved=list(range(4)), gain=-1.0),
        _feature(_split_cross(), operator=OPERATORS[4], moved=list(range(8)), gain=2.0),
        _feature(_wide_cross(), operator=OPERATORS[4], moved=list(range(16)), gain=4.0),
    ]
    ranked_missing = select_lns_diverse_shortlist(missing, shortlist_budget=2, audit_budget=3)
    assert set(ranked_missing.coverage["operators"]) == {OPERATORS[4]}
    assert _ranked_keys(missing, RANKING_FILL_NAME) == _ranked_keys(
        missing, RANKING_FILL_REFERENCE_NAME
    )


def test_incremental_token_reuse_matches_reference_definition() -> None:
    unique = assign_model_score_quantiles(merge_duplicate_features(_pool()))
    ranked = _farthest_first_fill(unique)
    reference = _farthest_first_fill_reference(unique)
    assert [item.state_hash for item in ranked] == [item.state_hash for item in reference]
    for index in range(1, len(ranked)):
        selected = ranked[:index]
        nxt = ranked[index]
        tokens = nxt.categories()
        expected = 0
        for item in selected:
            expected += sum(left == right for left, right in zip(tokens, item.categories(), strict=True))
        assert token_reuse(nxt, selected) == expected


def test_pooled_restart_ranking_matches_reference_fill() -> None:
    first = _pool()
    extra = _feature(
        _salted(_split_local(), 99),
        operator=OPERATORS[0],
        moved=list(range(16)),
        gain=12.0,
        start="lns_00_r01",
    )
    parent = "parent-hash"
    incremental = select_lns_parent_pooled_shortlists(
        {"lns_00_r00": first, "lns_00_r01": [extra]},
        {"lns_00_r00": parent, "lns_00_r01": parent},
        fill=RANKING_FILL_NAME,
    )
    reference = select_lns_parent_pooled_shortlists(
        {"lns_00_r00": first, "lns_00_r01": [extra]},
        {"lns_00_r00": parent, "lns_00_r01": parent},
        fill=RANKING_FILL_REFERENCE_NAME,
    )
    assert incremental[parent].ranked_keys == reference[parent].ranked_keys
    assert incremental[parent].selected_keys == reference[parent].selected_keys


def test_frozen_ranking_compare_checks_pooled_keys_and_plan_payloads() -> None:
    baseline = _ranking_artifact()
    baseline["parent_pooled_shortlists"] = {
        "parent": {
            "starts": ["lns_00_r00"],
            "shortlist": {
                "ranked_keys": ["cand"],
                "selected_keys": ["cand"],
                "audit_keys": ["cand"],
            },
            "stratified_keys": ["outside"],
        }
    }
    baseline["pooled_frontier_rows"] = {
        "cand": {
            "state_hash": "cand",
            "canonical_state": {"key": "cand"},
            "plan_v2_bridge": {"bridge": "cand"},
        }
    }
    baseline["runs"]["lns_00_r00"]["iterations"][0]["selected_frontier"][0]["canonical_state"] = {
        "key": "cand"
    }
    baseline["runs"]["lns_00_r00"]["iterations"][0]["selected_frontier"][0]["plan_v2_bridge"] = {
        "bridge": "cand"
    }
    matching = json.loads(json.dumps(baseline))
    matching["summary"]["global_shortlist_s"] = 0.001
    matching["method"]["lns_ranking_fill"] = RANKING_FILL_NAME
    assert compare_artifacts(baseline, matching)["equal"] is True

    pooled_drift = json.loads(json.dumps(baseline))
    pooled_drift["parent_pooled_shortlists"]["parent"]["shortlist"]["ranked_keys"] = ["other"]
    pooled = compare_artifacts(baseline, pooled_drift)
    assert pooled["equal"] is False
    assert any("ranked_keys" in item["path"] for item in pooled["mismatches"])

    outside_drift = json.loads(json.dumps(baseline))
    outside_drift["parent_pooled_shortlists"]["parent"]["stratified_keys"] = ["changed"]
    outside = compare_artifacts(baseline, outside_drift)
    assert outside["equal"] is False
    assert any("stratified_keys" in item["path"] for item in outside["mismatches"])

    plan_drift = json.loads(json.dumps(baseline))
    plan_drift["pooled_frontier_rows"]["cand"]["plan_v2_bridge"] = {"bridge": "other"}
    plan = compare_artifacts(baseline, plan_drift)
    assert plan["equal"] is False
    assert any("plan_v2_bridge" in item["path"] for item in plan["mismatches"])


def test_replay_frozen_feature_pools_match_reference_fill() -> None:
    parent = _anchor()
    features = _pool()
    unique = assign_model_score_quantiles(merge_duplicate_features(features))
    shortlist = select_lns_diverse_shortlist(unique)
    extra = _feature(
        _salted(_split_local(), 99),
        operator=OPERATORS[0],
        moved=list(range(16)),
        gain=12.0,
        start="lns_00_r01",
    )
    extra_shortlist = select_lns_diverse_shortlist([extra])
    model = _ranking_artifact()
    model["method"]["shortlist_budget"] = 16
    model["method"]["audit_budget"] = 32
    model["method"]["stratified_outside_audit"] = 0
    model["parents"] = {
        "lns_00_r00": {"state_hash": "parent"},
        "lns_00_r01": {"state_hash": "parent"},
    }

    def run_payload(shortlist_obj, rows):
        return {
            "initial_canonical_state": parent.canonical_payload(),
            "initial_state_hash": parent.canonical_hash(),
            "iterations": [
                {
                    "unique_candidates": len(rows),
                    "operators": {},
                    "critical_expert_ids": [1],
                    "random_expert_ids": [2],
                    "lns_diverse_shortlist": shortlist_obj.to_dict(),
                    "candidate_features": [item.to_dict() for item in rows],
                    "selected_frontier": [{"state_hash": key} for key in shortlist_obj.selected_keys],
                    "audit_frontier": [{"state_hash": key} for key in shortlist_obj.audit_keys],
                }
            ],
        }

    model["runs"] = {
        "lns_00_r00": run_payload(shortlist, unique),
        "lns_00_r01": run_payload(extra_shortlist, [extra]),
    }
    replay = replay_frozen_ranking(model, repeats=2)
    assert replay["repeat_agreement"] is True
    assert replay["fills_equal"] is True
    assert replay["matches_original_ranked_keys"] is True
    assert replay["equal"] is True
    assert replay["before"]["repeat_agreement"] is True
    assert replay["after"]["repeat_agreement"] is True


def test_empty_anchor_signature_diverges_like_lns_01_r01_and_recovery_restores() -> None:
    parent = _anchor()
    template = _feature(_reordered(), operator=OPERATORS[4], moved=[0, 1], gain=1.0)
    parent_sig = template.anchor_domain_assignment_signature
    match = replace(
        template,
        state_hash="0" * 64,
        domain_assignment_signature=parent_sig,
        width_histogram_delta=(),
        model_score_quantile=2,
    )
    mismatch = replace(
        template,
        state_hash="f" * 64,
        domain_assignment_signature="other-domain",
        width_histogram_delta=(),
        model_score_quantile=2,
    )
    live_keys = _ranked_keys([match, mismatch], RANKING_FILL_NAME)
    empty_keys = _ranked_keys(
        [
            replace(match, anchor_domain_assignment_signature=""),
            replace(mismatch, anchor_domain_assignment_signature=""),
        ],
        RANKING_FILL_NAME,
    )
    assert empty_keys != live_keys
    live = _feature(_split_local(), operator=OPERATORS[4], moved=[0, 1], gain=1.0)
    row = live.to_dict()
    row.pop("anchor_domain_assignment_signature", None)
    recovered = recover_anchor_domain_signature(
        row,
        anchor_canonical_state=parent.canonical_payload(),
        anchor_canonical_hash=parent.canonical_hash(),
    )
    assert recovered.anchor_domain_assignment_signature == live.anchor_domain_assignment_signature
    assert _ranked_keys([recovered], RANKING_FILL_NAME) == _ranked_keys([live], RANKING_FILL_NAME)


def test_mismatched_or_missing_anchor_fails_recovery() -> None:
    parent = _anchor()
    feature = _feature(_split_local(), operator=OPERATORS[4], moved=[0, 1], gain=1.0)
    row = feature.to_dict()
    row.pop("anchor_domain_assignment_signature", None)
    with pytest.raises(AnchorRecoveryError, match="does not match iteration canonical hash"):
        recover_anchor_domain_signature(
            row,
            anchor_canonical_state=parent.canonical_payload(),
            anchor_canonical_hash="not-the-anchor",
        )
    with pytest.raises(AnchorRecoveryError, match="missing iteration canonical state"):
        recover_run_features({"iterations": [{"candidate_features": [row]}]})


def test_replayed_model_payload_fails_when_selected_state_is_unavailable() -> None:
    parent = _anchor()
    candidate = _split_local()
    feature = _feature(candidate, operator=OPERATORS[4], moved=[0, 1], gain=1.0)
    shortlist = select_lns_diverse_shortlist([feature], shortlist_budget=1, audit_budget=1)
    model = _ranking_artifact()
    model["method"]["shortlist_budget"] = 1
    model["method"]["audit_budget"] = 1
    model["method"]["stratified_outside_audit"] = 0
    model["parents"] = {"lns_00_r00": {"state_hash": "parent"}}
    model["runs"] = {
        "lns_00_r00": {
            "initial_canonical_state": parent.canonical_payload(),
            "initial_state_hash": parent.canonical_hash(),
            "iterations": [
                {
                    "unique_candidates": 1,
                    "operators": {},
                    "critical_expert_ids": [1],
                    "random_expert_ids": [2],
                    "lns_diverse_shortlist": shortlist.to_dict(),
                    "candidate_features": [feature.to_dict()],
                    "selected_frontier": [{"state_hash": feature.state_hash}],
                    "audit_frontier": [{"state_hash": feature.state_hash}],
                }
            ],
        }
    }
    with pytest.raises(KeyError, match="unavailable selected state"):
        replayed_model_payload(model)


def test_replayed_model_payload_recomputes_outside_keys_and_round_trips_plans() -> None:
    parent = _anchor()
    candidate = _split_local()
    feature = _feature(candidate, operator=OPERATORS[4], moved=[0, 1], gain=1.0)
    shortlist = select_lns_diverse_shortlist([feature], shortlist_budget=1, audit_budget=1)
    plan_row = {
        "state_hash": feature.state_hash,
        "event_ns": 1.0,
        "robust_ns": 1.1,
        "event_gain_pct": 1.0,
        "robust_gain_pct": 1.0,
        "operator": OPERATORS[4],
        "strategy": "critical",
        "canonical_state": candidate.canonical_payload(),
        "plan_v2_bridge": candidate.to_bridge(),
    }
    model = _ranking_artifact()
    model["method"]["shortlist_budget"] = 1
    model["method"]["audit_budget"] = 1
    model["method"]["stratified_outside_audit"] = 0
    model["parents"] = {"lns_00_r00": {"state_hash": parent.canonical_hash()}}
    model["runs"] = {
        "lns_00_r00": {
            "initial_canonical_state": parent.canonical_payload(),
            "initial_state_hash": parent.canonical_hash(),
            "iterations": [
                {
                    "unique_candidates": 1,
                    "operators": {},
                    "critical_expert_ids": [1],
                    "random_expert_ids": [2],
                    "lns_diverse_shortlist": shortlist.to_dict(),
                    "candidate_features": [feature.to_dict()],
                    "selected_frontier": [plan_row],
                    "audit_frontier": [plan_row],
                }
            ],
        }
    }
    replayed = replayed_model_payload(model)
    restored = state_from_canonical_payload(replayed["runs"]["lns_00_r00"]["iterations"][0]["selected_frontier"][0]["canonical_state"])
    assert restored.canonical_hash() == feature.state_hash
    assert restored.to_bridge() == candidate.to_bridge()
    assert replayed["runs"]["lns_00_r00"]["iterations"][0]["anchor_recovery"]["recovery_source"] == (
        "iteration_canonical_state"
    )
    assert "anchor_domain_assignment_signature" in replayed["runs"]["lns_00_r00"]["iterations"][0]["candidate_features"][0]
