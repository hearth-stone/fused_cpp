"""Timing-free structural presampler candidate for template-LNS N=25.

This is a Lab proposal-retention policy. It does not replace
``_sample_neighborhood`` and must not read event scores, quantiles, hardware
timings, or specific state hashes.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import CLOSURE_BINS, closure_bin

POLICY_NAME = "structural_coverage_then_random_v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StructuralCandidate:
    state_hash: str
    actual_closure_size: int
    width_histogram: tuple[tuple[int, int], ...] = ()


def policy_descriptor() -> dict[str, object]:
    return {
        "bins": list(CLOSURE_BINS),
        "coverage": "first_representative_of_each_nonempty_bin_in_shuffle_order",
        "features": ["actual_closure_size"],
        "forbidden_features": [
            "predicted_gain",
            "model_score_quantile",
            "hardware_timing",
            "state_hash_allowlist",
            "previous_winner",
        ],
        "name": POLICY_NAME,
        "remainder": "same_shuffle_skipping_coverage_picks",
        "rng": "random.Random(seed)_one_shuffle_per_operator_matching_baseline",
        "schema_version": SCHEMA_VERSION,
    }


def policy_sha256() -> str:
    encoded = json.dumps(policy_descriptor(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_structural_coverage_then_random(
    unique_by_operator: Mapping[str, Sequence[StructuralCandidate]],
    *,
    operators: Sequence[str],
    per_operator: int,
    seed: int,
) -> dict[str, list[str]]:
    return _sample_policies(
        unique_by_operator,
        operators=operators,
        per_operator=per_operator,
        seed=seed,
        key_fn=_closure_key,
    )


CLOSURE_WIDTH_POLICY_NAME = "structural_coverage_closure_width_v1"
CLOSURE_WIDTH_SCHEMA_VERSION = 1


def closure_width_policy_descriptor() -> dict[str, object]:
    return {
        "bins": list(CLOSURE_BINS),
        "coverage": "first_representative_of_each_nonempty_closure_bin_and_width_histogram_in_shuffle_order",
        "features": ["actual_closure_size", "candidate_width_histogram"],
        "forbidden_features": [
            "predicted_gain",
            "model_score_quantile",
            "hardware_timing",
            "state_hash_allowlist",
            "previous_winner",
        ],
        "name": CLOSURE_WIDTH_POLICY_NAME,
        "remainder": "same_shuffle_skipping_coverage_picks",
        "rng": "random.Random(seed)_one_shuffle_per_operator_matching_baseline",
        "schema_version": CLOSURE_WIDTH_SCHEMA_VERSION,
        "width_histogram": "tuple_of_sorted_(width,count)_pairs_from_width_histogram(state)",
    }


def closure_width_policy_sha256() -> str:
    encoded = json.dumps(closure_width_policy_descriptor(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_structural_coverage_closure_width(
    unique_by_operator: Mapping[str, Sequence[StructuralCandidate]],
    *,
    operators: Sequence[str],
    per_operator: int,
    seed: int,
) -> dict[str, list[str]]:
    return _sample_policies(
        unique_by_operator,
        operators=operators,
        per_operator=per_operator,
        seed=seed,
        key_fn=_closure_width_key,
    )


def _closure_key(item: StructuralCandidate) -> object:
    return closure_bin(item.actual_closure_size)


def _closure_width_key(item: StructuralCandidate) -> object:
    return (closure_bin(item.actual_closure_size), tuple(item.width_histogram))


def _sample_policies(
    unique_by_operator: Mapping[str, Sequence[StructuralCandidate]],
    *,
    operators: Sequence[str],
    per_operator: int,
    seed: int,
    key_fn: Callable[[StructuralCandidate], object],
) -> dict[str, list[str]]:
    if per_operator <= 0:
        raise ValueError("per_operator must be positive")
    rng = random.Random(int(seed))
    sampled: dict[str, list[str]] = {}
    for operator in operators:
        items = list(unique_by_operator.get(operator, ()))
        sampled[operator] = _sample_operator(items, per_operator=per_operator, rng=rng, key_fn=key_fn)
    return sampled


def _sample_operator(
    items: Sequence[StructuralCandidate],
    *,
    per_operator: int,
    rng: random.Random,
    key_fn: Callable[[StructuralCandidate], object] = _closure_key,
) -> list[str]:
    if not items:
        return []
    shuffled = list(items)
    rng.shuffle(shuffled)
    chosen: list[StructuralCandidate] = []
    chosen_ids: set[str] = set()
    seen_keys: set[object] = set()
    for item in shuffled:
        key = key_fn(item)
        if key in seen_keys:
            continue
        chosen.append(item)
        chosen_ids.add(item.state_hash)
        seen_keys.add(key)
        if len(chosen) >= per_operator:
            return [row.state_hash for row in chosen]
    for item in shuffled:
        if item.state_hash in chosen_ids:
            continue
        chosen.append(item)
        chosen_ids.add(item.state_hash)
        if len(chosen) >= per_operator:
            break
    return [row.state_hash for row in chosen]


def bin_occupancy(items: Sequence[StructuralCandidate]) -> dict[str, int]:
    counts = {bin_name: 0 for bin_name in CLOSURE_BINS}
    for item in items:
        counts[closure_bin(item.actual_closure_size)] += 1
    return counts
