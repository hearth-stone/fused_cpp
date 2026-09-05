#!/usr/bin/env python3
"""Compare two template-LNS model artifacts for ranking invariance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lns_diverse_shortlist import (  # noqa: E402
    AnchorRecoveryError,
    domain_assignment_payload_from_mapping,
    domain_assignment_signature,
    iteration_anchor_from_run,
)


RANKING_FEATURE_FIELDS = (
    "actual_closure_bin",
    "actual_closure_size",
    "anchor_domain_assignment_signature",
    "anchor_state_hash",
    "candidate_width_histogram",
    "changed_core_begin",
    "changed_core_end",
    "cross_domain_lane_count",
    "domain_assignment",
    "domain_assignment_signature",
    "model_score_quantile",
    "operator",
    "predicted_gain_pct",
    "restart",
    "scope",
    "start",
    "state_hash",
    "strategy",
    "target_destroy_size",
    "width_histogram_delta",
)
FRONTIER_SCORE_FIELDS = (
    "state_hash",
    "event_ns",
    "robust_ns",
    "event_gain_pct",
    "robust_gain_pct",
    "operator",
    "strategy",
)
FRONTIER_PLAN_FIELDS = (
    "canonical_state",
    "plan_v2_bridge",
)
POOLED_SHORTLIST_KEY_FIELDS = (
    "ranked_keys",
    "selected_keys",
    "audit_keys",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact is not an object: {path}")
    return payload


def _mismatch(path: str, left: object, right: object) -> dict[str, object]:
    return {"path": path, "baseline": left, "candidate": right}


def _recovered_anchor_signature(run: dict[str, object]) -> str | None:
    try:
        _hashed, payload = iteration_anchor_from_run(run, iteration_index=0)
    except (AnchorRecoveryError, KeyError, TypeError, ValueError):
        return None
    return domain_assignment_signature(domain_assignment_payload_from_mapping(payload))


def _feature_field(
    field: str,
    left_row: dict[str, object],
    right_row: dict[str, object],
    *,
    expected_anchor_signature: str | None,
) -> tuple[object, object]:
    left = left_row.get(field)
    right = right_row.get(field)
    if field != "anchor_domain_assignment_signature":
        return left, right
    if left in (None, "") and right in (None, "") and expected_anchor_signature is None:
        return left, right
    if left in (None, ""):
        return expected_anchor_signature, right
    return left, right


def _compare_features(
    start: str,
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    expected_anchor_signature: str | None = None,
) -> list[dict[str, object]]:
    mismatches = []
    if len(baseline) != len(candidate):
        return [
            _mismatch(
                f"runs.{start}.candidate_features.length",
                len(baseline),
                len(candidate),
            )
        ]
    left = sorted(baseline, key=lambda row: str(row["state_hash"]))
    right = sorted(candidate, key=lambda row: str(row["state_hash"]))
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        for field in RANKING_FEATURE_FIELDS:
            left_value, right_value = _feature_field(
                field,
                left_row,
                right_row,
                expected_anchor_signature=expected_anchor_signature,
            )
            if left_value != right_value:
                mismatches.append(
                    _mismatch(
                        f"runs.{start}.candidate_features[{index}].{field}",
                        left_value,
                        right_value,
                    )
                )
                if len(mismatches) >= 20:
                    return mismatches
    return mismatches


def _compare_frontier(
    start: str,
    name: str,
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> list[dict[str, object]]:
    mismatches = []
    if len(baseline) != len(candidate):
        return [_mismatch(f"runs.{start}.{name}.length", len(baseline), len(candidate))]
    for index, (left_row, right_row) in enumerate(zip(baseline, candidate, strict=True)):
        for field in (*FRONTIER_SCORE_FIELDS, *FRONTIER_PLAN_FIELDS):
            if left_row.get(field) != right_row.get(field):
                mismatches.append(
                    _mismatch(
                        f"runs.{start}.{name}[{index}].{field}",
                        left_row.get(field),
                        right_row.get(field),
                    )
                )
                if len(mismatches) >= 20:
                    return mismatches
    return mismatches


def _compare_parent_pooled(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    left = baseline.get("parent_pooled_shortlists") or {}
    right = candidate.get("parent_pooled_shortlists") or {}
    if not isinstance(left, dict) or not isinstance(right, dict):
        return [_mismatch("parent_pooled_shortlists.type", type(left).__name__, type(right).__name__)]
    if sorted(left) != sorted(right):
        return [_mismatch("parent_pooled_shortlists.keys", sorted(left), sorted(right))]
    for parent in sorted(left):
        left_row = left[parent]
        right_row = right[parent]
        left_shortlist = (left_row or {}).get("shortlist") or {}
        right_shortlist = (right_row or {}).get("shortlist") or {}
        for field in POOLED_SHORTLIST_KEY_FIELDS:
            if left_shortlist.get(field) != right_shortlist.get(field):
                mismatches.append(
                    _mismatch(
                        f"parent_pooled_shortlists.{parent}.shortlist.{field}",
                        left_shortlist.get(field),
                        right_shortlist.get(field),
                    )
                )
        if (left_row or {}).get("stratified_keys") != (right_row or {}).get("stratified_keys"):
            mismatches.append(
                _mismatch(
                    f"parent_pooled_shortlists.{parent}.stratified_keys",
                    (left_row or {}).get("stratified_keys"),
                    (right_row or {}).get("stratified_keys"),
                )
            )
        if len(mismatches) >= 20:
            return mismatches
    return mismatches


def _compare_pooled_frontier_rows(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    left = baseline.get("pooled_frontier_rows") or {}
    right = candidate.get("pooled_frontier_rows") or {}
    if not isinstance(left, dict) or not isinstance(right, dict):
        return [_mismatch("pooled_frontier_rows.type", type(left).__name__, type(right).__name__)]
    if sorted(left) != sorted(right):
        return [_mismatch("pooled_frontier_rows.keys", sorted(left), sorted(right))]
    for state_hash in sorted(left):
        left_row = left[state_hash] or {}
        right_row = right[state_hash] or {}
        for field in ("state_hash", *FRONTIER_PLAN_FIELDS):
            if left_row.get(field) != right_row.get(field):
                mismatches.append(
                    _mismatch(
                        f"pooled_frontier_rows.{state_hash}.{field}",
                        left_row.get(field),
                        right_row.get(field),
                    )
                )
                if len(mismatches) >= 20:
                    return mismatches
    return mismatches


def compare_artifacts(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    for path, left, right in (
        ("method.seed", baseline["method"]["seed"], candidate["method"]["seed"]),
        (
            "method.restarts_per_parent",
            baseline["method"]["restarts_per_parent"],
            candidate["method"]["restarts_per_parent"],
        ),
        (
            "method.lns_shortlist_policy_sha256",
            baseline["method"]["lns_shortlist_policy_sha256"],
            candidate["method"]["lns_shortlist_policy_sha256"],
        ),
        (
            "identity.calibration_sha256",
            baseline["identity"]["calibration_sha256"],
            candidate["identity"]["calibration_sha256"],
        ),
        (
            "identity.pairwise_calibration_sha256",
            baseline["identity"]["pairwise_calibration_sha256"],
            candidate["identity"]["pairwise_calibration_sha256"],
        ),
        (
            "identity.extension_sha256",
            baseline["identity"]["extension_sha256"],
            candidate["identity"]["extension_sha256"],
        ),
        (
            "identity.parent_source.named_state_hashes",
            baseline["identity"]["parent_source"]["named_state_hashes"],
            candidate["identity"]["parent_source"]["named_state_hashes"],
        ),
        (
            "summary.unique_candidates",
            baseline["summary"]["unique_candidates"],
            candidate["summary"]["unique_candidates"],
        ),
        (
            "summary.event_calls",
            baseline["summary"]["event_calls"],
            candidate["summary"]["event_calls"],
        ),
        ("summary.better", baseline["summary"]["better"], candidate["summary"]["better"]),
        ("summary.worse", baseline["summary"]["worse"], candidate["summary"]["worse"]),
        (
            "summary.incomparable",
            baseline["summary"]["incomparable"],
            candidate["summary"]["incomparable"],
        ),
        (
            "lns_diverse_shortlist.selected_keys",
            (baseline.get("lns_diverse_shortlist") or {}).get("selected_keys"),
            (candidate.get("lns_diverse_shortlist") or {}).get("selected_keys"),
        ),
        (
            "lns_diverse_shortlist.audit_keys",
            (baseline.get("lns_diverse_shortlist") or {}).get("audit_keys"),
            (candidate.get("lns_diverse_shortlist") or {}).get("audit_keys"),
        ),
    ):
        if left != right:
            mismatches.append(_mismatch(path, left, right))

    baseline_runs = baseline["runs"]
    candidate_runs = candidate["runs"]
    if list(baseline_runs) != list(candidate_runs):
        mismatches.append(_mismatch("runs.keys", list(baseline_runs), list(candidate_runs)))
        return {
            "equal": False,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }

    for start, baseline_run in baseline_runs.items():
        candidate_run = candidate_runs[start]
        baseline_iter = baseline_run["iterations"][0]
        candidate_iter = candidate_run["iterations"][0]
        for path, left, right in (
            (
                f"runs.{start}.unique_candidates",
                baseline_iter["unique_candidates"],
                candidate_iter["unique_candidates"],
            ),
            (
                f"runs.{start}.operators",
                baseline_iter["operators"],
                candidate_iter["operators"],
            ),
            (
                f"runs.{start}.critical_expert_ids",
                baseline_iter["critical_expert_ids"],
                candidate_iter["critical_expert_ids"],
            ),
            (
                f"runs.{start}.random_expert_ids",
                baseline_iter["random_expert_ids"],
                candidate_iter["random_expert_ids"],
            ),
            (
                f"runs.{start}.lns_diverse_shortlist.selected_keys",
                baseline_iter["lns_diverse_shortlist"]["selected_keys"],
                candidate_iter["lns_diverse_shortlist"]["selected_keys"],
            ),
            (
                f"runs.{start}.lns_diverse_shortlist.audit_keys",
                baseline_iter["lns_diverse_shortlist"]["audit_keys"],
                candidate_iter["lns_diverse_shortlist"]["audit_keys"],
            ),
            (
                f"runs.{start}.lns_diverse_shortlist.ranked_keys",
                baseline_iter["lns_diverse_shortlist"]["ranked_keys"],
                candidate_iter["lns_diverse_shortlist"]["ranked_keys"],
            ),
        ):
            if left != right:
                mismatches.append(_mismatch(path, left, right))
        mismatches.extend(
            _compare_features(
                start,
                baseline_iter.get("candidate_features") or [],
                candidate_iter.get("candidate_features") or [],
                expected_anchor_signature=_recovered_anchor_signature(baseline_run),
            )
        )
        mismatches.extend(
            _compare_frontier(
                start,
                "selected_frontier",
                baseline_iter.get("selected_frontier") or [],
                candidate_iter.get("selected_frontier") or [],
            )
        )
        mismatches.extend(
            _compare_frontier(
                start,
                "audit_frontier",
                baseline_iter.get("audit_frontier") or [],
                candidate_iter.get("audit_frontier") or [],
            )
        )
        if len(mismatches) >= 40:
            break

    mismatches.extend(_compare_parent_pooled(baseline, candidate))
    mismatches.extend(_compare_pooled_frontier_rows(baseline, candidate))

    return {
        "equal": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "baseline_search_wall_s": baseline["summary"].get("search_wall_s"),
        "candidate_search_wall_s": candidate["summary"].get("search_wall_s"),
        "baseline_search_breakdown": baseline["summary"].get("search_breakdown"),
        "candidate_search_breakdown": candidate["summary"].get("search_breakdown"),
        "candidate_ru_maxrss_kb": candidate["summary"].get("ru_maxrss_kb"),
        "candidate_global_shortlist_s": candidate["summary"].get("global_shortlist_s"),
    }


def main() -> int:
    args = parse_args()
    result = compare_artifacts(_load(args.baseline), _load(args.candidate))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
