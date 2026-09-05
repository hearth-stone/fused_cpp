#!/usr/bin/env python3
"""Compare two independent partial-order hardware frontier sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row["candidate_state_hash"]),
        str(row["anchor_state_hash"]),
        str(row["role"]),
        str(row["start"]),
        str(row.get("operator", "")),
    )


def analyze_sessions(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    expected_kind = "partial_order_hardware_frontier_session"
    if first.get("kind") != expected_kind or second.get("kind") != expected_kind:
        raise ValueError("both inputs must be partial-order hardware frontier sessions")
    first_identity = first["identity"]
    second_identity = second["identity"]
    for field in ("frontier_sha256", "extension_sha256"):
        if first_identity[field] != second_identity[field]:
            raise ValueError(f"session identity mismatch: {field}")
    first_rows = {_key(row): row for row in first["comparisons"]}
    second_rows = {_key(row): row for row in second["comparisons"]}
    if len(first_rows) != len(first["comparisons"]) or len(second_rows) != len(second["comparisons"]):
        raise ValueError("session contains duplicate comparison keys")
    if set(first_rows) != set(second_rows):
        raise ValueError("sessions do not contain the same comparisons")
    first_stats = first.get("plan_stats", {})
    second_stats = second.get("plan_stats", {})
    if set(first_stats) != set(second_stats) or not first_stats:
        raise ValueError("sessions do not contain the same non-empty plan set")
    session_best = [
        min(stats, key=lambda key: (float(stats[key]["median_ms"]), key))
        for stats in (first_stats, second_stats)
    ]
    normalized_scores = {
        state_hash: sum(
            float(stats[state_hash]["median_ms"])
            / min(float(row["median_ms"]) for row in stats.values())
            for stats in (first_stats, second_stats)
        )
        for state_hash in first_stats
    }
    global_best = min(normalized_scores, key=lambda key: (normalized_scores[key], key))
    global_elites = sorted(normalized_scores, key=lambda key: (normalized_scores[key], key))

    stable = []
    false_pruning = []
    false_acceptance = []
    for key in sorted(first_rows):
        first_row = first_rows[key]
        second_row = second_rows[key]
        stable_both = bool(first_row["session_stable_over_2pct"]) and bool(
            second_row["session_stable_over_2pct"]
        )
        item = {
            "candidate_state_hash": key[0],
            "anchor_state_hash": key[1],
            "role": key[2],
            "start": key[3],
            "operator": key[4],
            "moved_experts": first_row["moved_experts"],
            "partial_order": first_row["partial_order"],
            "session_1_paired_gain_pct": first_row["paired_gain_pct"],
            "session_2_paired_gain_pct": second_row["paired_gain_pct"],
        }
        if stable_both:
            stable.append(item)
            if key[2] == "candidate_worse_sentinel":
                false_pruning.append(item)
        elif key[2] == "candidate_better_frontier":
            false_acceptance.append(item)
    stable_frontier = [
        item
        for item in stable
        if item["role"]
        in {
            "candidate_better_frontier",
            "model_better_frontier",
            "incomparable_frontier",
            "model_worse_spectrum",
        }
    ]
    if false_pruning:
        decision = "partial_order_pruning_gate_failed"
    elif false_acceptance:
        decision = "partial_order_acceptance_gate_failed"
    elif stable_frontier:
        decision = "use_hardware_assisted_beam_search"
    else:
        decision = "move_to_template_level_lns"
    return {
        "kind": "partial_order_hardware_frontier_cross_session_analysis",
        "identity": {
            "frontier_sha256": first_identity["frontier_sha256"],
            "extension_sha256": first_identity["extension_sha256"],
        },
        "sessions": [first["method"], second["method"]],
        "comparisons": len(first_rows),
        "stable_over_2pct_both_sessions": stable,
        "stable_frontier_candidates": stable_frontier,
        "false_pruning_sentinels": false_pruning,
        "false_acceptance_candidates": false_acceptance,
        "session_absolute_best_state_hashes": session_best,
        "global_measured_best_state_hash": global_best,
        "global_measured_elite_state_hashes": global_elites,
        "global_measured_best_is_session_best_twice": session_best == [global_best, global_best],
        "decision": decision,
    }


def main() -> int:
    args = parse_args()
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in args.session]
    result = analyze_sessions(*sessions)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "comparisons": result["comparisons"],
                "stable_frontier_candidates": len(result["stable_frontier_candidates"]),
                "false_pruning_sentinels": len(result["false_pruning_sentinels"]),
                "false_acceptance_candidates": len(result["false_acceptance_candidates"]),
                "decision": result["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
