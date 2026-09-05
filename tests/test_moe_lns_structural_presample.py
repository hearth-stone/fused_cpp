from __future__ import annotations

import random

from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import CLOSURE_BINS
from optimizations.fused_moe_sve.benchmarks.lns_structural_presample import (
    CLOSURE_WIDTH_POLICY_NAME,
    POLICY_NAME,
    StructuralCandidate,
    bin_occupancy,
    closure_width_policy_descriptor,
    closure_width_policy_sha256,
    policy_descriptor,
    policy_sha256,
    sample_structural_coverage_closure_width,
    sample_structural_coverage_then_random,
)


def _items(*sizes: int) -> list[StructuralCandidate]:
    return [
        StructuralCandidate(state_hash=f"h{index:04d}_{size}", actual_closure_size=size)
        for index, size in enumerate(sizes)
    ]


def test_policy_is_timing_free_and_versioned() -> None:
    descriptor = policy_descriptor()
    assert descriptor["name"] == POLICY_NAME
    assert descriptor["features"] == ["actual_closure_size"]
    assert "predicted_gain" in descriptor["forbidden_features"]
    assert "state_hash_allowlist" in descriptor["forbidden_features"]
    assert len(policy_sha256()) == 64
    assert policy_sha256() == policy_sha256()


def test_coverage_keeps_one_member_of_each_nonempty_bin() -> None:
    items = _items(8, 8, 24, 48, 48, 48, 80, 80, 200)
    sampled = sample_structural_coverage_then_random(
        {"op": items},
        operators=("op",),
        per_operator=5,
        seed=11,
    )["op"]
    chosen = [item for item in items if item.state_hash in sampled]
    occupancy = bin_occupancy(chosen)
    assert occupancy["1_15"] == 1
    assert occupancy["16_31"] == 1
    assert occupancy["32_63"] == 1
    assert occupancy["64_127"] == 1
    assert occupancy["128_plus"] == 1
    assert len(sampled) == 5
    assert len(set(sampled)) == 5


def test_single_bin_matches_shuffle_truncate() -> None:
    items = _items(8, 9, 10, 11, 12, 13, 14, 15)
    sampled = sample_structural_coverage_then_random(
        {"op": items},
        operators=("op",),
        per_operator=4,
        seed=3,
    )["op"]
    rng_items = list(items)
    random.Random(3).shuffle(rng_items)
    assert sampled == [item.state_hash for item in rng_items[:4]]
    assert bin_occupancy([item for item in items if item.state_hash in sampled])["1_15"] == 4


def test_sampling_is_seed_deterministic_and_ignores_hash_identity() -> None:
    items = _items(24, 24, 24, 24, 48, 48, 48, 48, 80, 80, 8, 8)
    first = sample_structural_coverage_then_random(
        {"op": items},
        operators=("op",),
        per_operator=6,
        seed=29,
    )["op"]
    second = sample_structural_coverage_then_random(
        {"op": items},
        operators=("op",),
        per_operator=6,
        seed=29,
    )["op"]
    other = sample_structural_coverage_then_random(
        {"op": items},
        operators=("op",),
        per_operator=6,
        seed=31,
    )["op"]
    assert first == second
    assert first != other
    assert set(CLOSURE_BINS) == {
        "1_15",
        "16_31",
        "32_63",
        "64_127",
        "128_plus",
    }


def _same_bin_two_widths() -> list[StructuralCandidate]:
    hist_a = ((8, 4), (16, 2))
    hist_b = ((4, 8), (8, 4), (16, 1))
    items = [
        StructuralCandidate(state_hash=f"a{index:02d}", actual_closure_size=80, width_histogram=hist_a)
        for index in range(8)
    ]
    items.extend(
        StructuralCandidate(state_hash=f"b{index:02d}", actual_closure_size=80, width_histogram=hist_b)
        for index in range(8)
    )
    return items


def test_closure_width_policy_is_distinct_from_v1_and_timing_free() -> None:
    descriptor = closure_width_policy_descriptor()
    assert descriptor["name"] == CLOSURE_WIDTH_POLICY_NAME
    assert descriptor["name"] != POLICY_NAME
    assert "candidate_width_histogram" in descriptor["features"]
    assert "predicted_gain" in descriptor["forbidden_features"]
    assert "state_hash_allowlist" in descriptor["forbidden_features"]
    assert closure_width_policy_sha256() != policy_sha256()
    assert len(closure_width_policy_sha256()) == 64


def test_width_histogram_changes_selection_when_closure_bins_match() -> None:
    items = _same_bin_two_widths()
    operators = ("op",)
    v2 = None
    v1 = None
    for seed in range(256):
        v1 = sample_structural_coverage_then_random(
            {"op": items},
            operators=operators,
            per_operator=2,
            seed=seed,
        )["op"]
        v2 = sample_structural_coverage_closure_width(
            {"op": items},
            operators=operators,
            per_operator=2,
            seed=seed,
        )["op"]
        v1_labels = {key[0] for key in v1}
        if v1_labels == {"a"} or v1_labels == {"b"}:
            break
    else:
        raise AssertionError("did not find a seed where closure-only coverage stays in one width")
    assert v2 is not None and v1 is not None
    assert {key[0] for key in v2} == {"a", "b"}
    assert v1 != v2
    assert len(v2) == 2
    assert sample_structural_coverage_closure_width(
        {"op": items},
        operators=operators,
        per_operator=2,
        seed=seed,
    )["op"] == v2
