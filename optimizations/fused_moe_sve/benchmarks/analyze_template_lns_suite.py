#!/usr/bin/env python3
"""Aggregate high-skew, median, and uniformish template-LNS decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-skew-depth1", type=Path, required=True)
    parser.add_argument("--high-skew-depth2", type=Path, required=True)
    parser.add_argument("--median", type=Path, required=True)
    parser.add_argument("--uniformish", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_summary(analysis: dict[str, object]) -> dict[str, object]:
    if analysis.get("kind") != "template_lns_cross_session_analysis":
        raise ValueError("input has an unexpected template-LNS analysis kind")
    gains = analysis["winner"]["absolute_median_gain_vs_anchor_pct"]
    strongest_anchor_gain = [
        min(float(values[session]) for values in gains.values())
        for session in range(2)
    ]
    return {
        "decision": analysis["decision"],
        "winner_state_hash": analysis["winner"]["state_hash"],
        "winner_absolute_median_ms": analysis["winner"]["absolute_median_ms"],
        "winner_gain_vs_strongest_anchor_pct": strongest_anchor_gain,
        "session_absolute_best_state_hashes": analysis["winner"][
            "session_absolute_best_state_hashes"
        ],
        "session_best_gap_pct": analysis["winner"]["session_best_gap_pct"],
        "stable_unique_candidates": analysis["partial_order"]["stable_unique_candidates"],
        "false_pruning_sentinels": analysis["partial_order"]["false_pruning_sentinels"],
        "false_acceptance_candidates": analysis["partial_order"].get(
            "false_acceptance_candidates",
            0,
        ),
        "model_hardware_spearman": analysis["partial_order"]["model_hardware_spearman"],
        "event_calls": analysis["budget_comparison"]["template_lns"]["event_calls"],
        "model_wall_s": analysis["budget_comparison"]["template_lns"]["model_wall_s"],
        "hardware_plans": analysis["budget_comparison"]["template_lns"]["hardware_plans"],
    }


def analyze_suite(
    high_skew_depth1: dict[str, object],
    high_skew_depth2: dict[str, object],
    median: dict[str, object],
    uniformish: dict[str, object],
) -> dict[str, object]:
    traces = {
        "high_skew": _trace_summary(high_skew_depth1),
        "median": _trace_summary(median),
        "uniformish": _trace_summary(uniformish),
    }
    high_depth2 = _trace_summary(high_skew_depth2)
    improvement_gate = all(
        min(row["winner_gain_vs_strongest_anchor_pct"]) > 2.0
        for row in traces.values()
    )
    pruning_gate = all(row["false_pruning_sentinels"] == 0 for row in traces.values())
    acceptance_gate = all(
        row["false_acceptance_candidates"] == 0 for row in traces.values()
    )
    if not improvement_gate:
        decision = "reject_fixed_template_lns_mixture"
    elif not pruning_gate or not acceptance_gate:
        decision = "adopt_lns_neighborhood_disable_partial_order_decisions"
    else:
        decision = "adopt_lns_with_partial_order_gate"
    return {
        "kind": "template_lns_suite_analysis",
        "decision": decision,
        "gates": {
            "winner_over_strongest_anchor_both_sessions": improvement_gate,
            "zero_false_pruning": pruning_gate,
            "zero_false_acceptance": acceptance_gate,
            "high_skew_depth2_requires_another_expansion": high_depth2["decision"]
            == "adopt_template_lns_and_expand_consensus_elite",
        },
        "traces": traces,
        "high_skew_depth2": high_depth2,
        "aggregate": {
            "event_calls": sum(row["event_calls"] for row in traces.values())
            + high_depth2["event_calls"],
            "model_wall_s": sum(row["model_wall_s"] for row in traces.values())
            + high_depth2["model_wall_s"],
            "hardware_frontier_plans": sum(row["hardware_plans"] for row in traces.values())
            + high_depth2["hardware_plans"],
            "hardware_sessions": 8,
        },
        "policy": {
            "template_lns_partial_order_mode": "diagnostic_only",
            "automatic_acceptance": False,
            "dominance_pruning": False,
            "hardware_selection": "top16_plus_model_score_spectrum",
            "incumbent_policy": "return_cross_session_hardware_consensus_elite",
        },
    }


def main() -> int:
    args = parse_args()
    paths = {
        "high_skew_depth1": args.high_skew_depth1,
        "high_skew_depth2": args.high_skew_depth2,
        "median": args.median,
        "uniformish": args.uniformish,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    result = analyze_suite(**payloads)
    result["identity"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], **result["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
