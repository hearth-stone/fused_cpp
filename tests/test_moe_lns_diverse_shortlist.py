from __future__ import annotations

import json
import random
import sys
from pathlib import Path


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
from lns_diverse_shortlist import (  # noqa: E402
    DEFAULT_AUDIT_BUDGET,
    DEFAULT_SHORTLIST_BUDGET,
    POLICY_NAME,
    LnsCandidateFeature,
    build_candidate_feature,
    compact_structure,
    merge_duplicate_features,
    policy_document,
    policy_sha256,
    select_lns_diverse_shortlist,
    select_lns_global_shortlist,
    state_from_canonical_payload,
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
