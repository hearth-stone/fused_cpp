#!/usr/bin/env python3
"""Compare a second proposal seed against the frozen 2-restart median protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizations.fused_moe_sve.benchmarks.analyze_lns_restart_budget import (  # noqa: E402
    _fastest,
    _gain_pct,
    _role_set,
    _sha256,
    analyze_arm,
)

FROZEN_EXACT_CAP = 2400
FROZEN_RESTARTS = 2
FROZEN_NEIGHBORS = 25
FROZEN_STARTS = 8
FROZEN_TOP16 = 64
FROZEN_STRATIFIED = 16
PREVIOUS_SELECTED = "0418b88445c2f88488ca10a1eeae3aeb7b6080e806e241eacfed17ec10904bf6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--session", type=Path, nargs=2, required=True)
    parser.add_argument("--previous-frontier", type=Path, required=True)
    parser.add_argument("--previous-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reference_metrics(
    frontier: dict[str, object],
    first: dict[str, object],
    second: dict[str, object],
) -> dict[str, object]:
    reference = _role_set(frontier, "previous_seed_selected")
    if len(reference) > 1:
        raise ValueError("expected at most one previous-seed selected control")
    reference_hash = next(iter(reference), None)
    selected = _role_set(frontier, "lns_diverse_top16")
    payload: dict[str, object] = {
        "reference_state_hash": reference_hash,
        "reference_in_selected_top16": bool(reference_hash is not None and reference_hash in selected),
    }
    if reference_hash is None:
        return payload

    def session_metrics(stats: dict[str, dict[str, object]]) -> dict[str, object]:
        selected_best = _fastest(stats, selected)
        selected_ms = float(stats[selected_best]["median_ms"])
        reference_ms = float(stats[reference_hash]["median_ms"])
        return {
            "reference_median_ms": reference_ms,
            "selected_gain_vs_reference_pct": _gain_pct(reference_ms, selected_ms),
            "selected_regret_vs_reference_pct": _gain_pct(selected_ms, reference_ms),
        }

    first_metrics = session_metrics(first["plan_stats"])
    second_metrics = session_metrics(second["plan_stats"])
    payload["session_1"] = first_metrics
    payload["session_2"] = second_metrics
    payload["two_session_selected_beats_reference"] = bool(
        first_metrics["selected_gain_vs_reference_pct"] > 0.0
        and second_metrics["selected_gain_vs_reference_pct"] > 0.0
    )
    return payload


def analyze_second_proposal_seed(
    frontier: dict[str, object],
    first: dict[str, object],
    second: dict[str, object],
    model: dict[str, object],
    previous_frontier: dict[str, object],
    previous_model: dict[str, object],
) -> dict[str, object]:
    current = analyze_arm(frontier, first, second, model)
    previous_selected = _role_set(previous_frontier, "lns_diverse_top16")
    current_selected = _role_set(frontier, "lns_diverse_top16")
    overlap = current_selected & previous_selected
    search = current["search"]
    role_counts = current["role_counts"] or {}
    protocol = {
        "exact_candidate_budget": search.get("maximum_exact_candidate_budget"),
        "exact_candidate_budget_frozen": search.get("maximum_exact_candidate_budget") == FROZEN_EXACT_CAP,
        "restarts_per_parent": search.get("restarts_per_parent"),
        "restarts_frozen": search.get("restarts_per_parent") == FROZEN_RESTARTS,
        "neighbors_per_operator": search.get("neighbors_per_operator"),
        "neighbors_frozen": search.get("neighbors_per_operator") == FROZEN_NEIGHBORS,
        "starts": search.get("starts"),
        "starts_frozen": search.get("starts") == FROZEN_STARTS,
        "selected_top16": role_counts.get("lns_diverse_top16"),
        "selected_top16_frozen": role_counts.get("lns_diverse_top16") == FROZEN_TOP16,
        "stratified": role_counts.get("lns_stratified_outside_top32"),
        "stratified_frozen": role_counts.get("lns_stratified_outside_top32") == FROZEN_STRATIFIED,
        "seed": model.get("method", {}).get("seed"),
        "previous_seed": previous_model.get("method", {}).get("seed"),
    }
    protocol_ok = all(
        protocol[key]
        for key in (
            "exact_candidate_budget_frozen",
            "restarts_frozen",
            "neighbors_frozen",
            "starts_frozen",
            "selected_top16_frozen",
            "stratified_frozen",
        )
    )
    return {
        "kind": "lns_second_proposal_seed_comparison",
        "protocol": protocol,
        "protocol_ok": protocol_ok,
        "current": current,
        "previous_search": {
            "starts": previous_model.get("summary", {}).get("starts"),
            "restarts_per_parent": previous_model.get("method", {}).get("restarts_per_parent"),
            "neighbors_per_operator": previous_model.get("method", {}).get("neighbors_per_operator"),
            "maximum_exact_candidate_budget": previous_model.get("method", {}).get(
                "maximum_exact_candidate_budget"
            ),
            "unique_candidates": previous_model.get("summary", {}).get("unique_candidates"),
            "event_calls": previous_model.get("summary", {}).get("event_calls"),
            "search_wall_s": previous_model.get("summary", {}).get("search_wall_s"),
            "seed": previous_model.get("method", {}).get("seed"),
        },
        "selected_overlap": {
            "current_top16": sorted(current_selected),
            "previous_top16": sorted(previous_selected),
            "overlap_keys": sorted(overlap),
            "overlap_count": len(overlap),
            "current_count": len(current_selected),
            "previous_count": len(previous_selected),
            "previous_selected_best_in_current_top16": PREVIOUS_SELECTED in current_selected,
        },
        "reference": _reference_metrics(frontier, first, second),
    }


def main() -> int:
    args = parse_args()
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    model = json.loads(args.model.read_text(encoding="utf-8"))
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in args.session]
    previous_frontier = json.loads(args.previous_frontier.read_text(encoding="utf-8"))
    previous_model = json.loads(args.previous_model.read_text(encoding="utf-8"))
    result = analyze_second_proposal_seed(
        frontier,
        sessions[0],
        sessions[1],
        model,
        previous_frontier,
        previous_model,
    )
    result["identity"] = {
        "frontier_sha256": _sha256(args.frontier),
        "model_sha256": _sha256(args.model),
        "previous_frontier_sha256": _sha256(args.previous_frontier),
        "previous_model_sha256": _sha256(args.previous_model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current = result["current"]
    print(
        json.dumps(
            {
                "protocol_ok": result["protocol_ok"],
                "unique_plans": current["unique_plans"],
                "seed": result["protocol"]["seed"],
                "previous_seed": result["protocol"]["previous_seed"],
                "selected_overlap_count": result["selected_overlap"]["overlap_count"],
                "selected_gain_vs_anchor": [
                    current["session_1"]["selected_gain_vs_strongest_anchor_pct"],
                    current["session_2"]["selected_gain_vs_strongest_anchor_pct"],
                ],
                "two_session_selected_gain_vs_anchor_over_2pct": current[
                    "two_session_selected_gain_vs_anchor_over_2pct"
                ],
                "two_session_selected_beats_elite": current["two_session_selected_beats_elite"],
                "two_session_selected_beats_reference": result["reference"].get(
                    "two_session_selected_beats_reference"
                ),
                "selected_best": [
                    current["session_1"]["selected_best_state_hash"],
                    current["session_2"]["selected_best_state_hash"],
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["protocol_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
