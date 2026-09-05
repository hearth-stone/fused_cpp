#!/usr/bin/env python3
"""Turn two template-LNS hardware sessions into an adoption decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from optimizations.fused_moe_sve.benchmarks.analyze_partial_order_hardware_frontier import (
    analyze_sessions,
)
from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import _spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--session", type=Path, nargs=2, required=True)
    parser.add_argument("--beam-model-artifact", type=Path, action="append", default=[])
    parser.add_argument("--beam-frontier", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _destroy_size(operator: str) -> str:
    match = re.search(r"_d(\d+)_b\d+$", operator)
    return match.group(1) if match else "unknown"


def analyze_template_lns(
    frontier: dict[str, object],
    model: dict[str, object],
    first: dict[str, object],
    second: dict[str, object],
    *,
    comparison_models: list[dict[str, object]] | None = None,
    comparison_frontiers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if frontier.get("kind") != "partial_order_hardware_frontier":
        raise ValueError("frontier has an unexpected kind")
    if model.get("kind") != "executable_partial_order_template_lns_model_replay":
        raise ValueError("model artifact has an unexpected kind")
    if set(frontier["plans"]) != set(first["plan_stats"]) or set(frontier["plans"]) != set(
        second["plan_stats"]
    ):
        raise ValueError("frontier and hardware sessions do not contain the same plan set")

    base = analyze_sessions(first, second)
    stable = list(base["stable_frontier_candidates"])
    stable_unique = {str(row["candidate_state_hash"]) for row in stable}
    false_pruning = list(base["false_pruning_sentinels"])
    false_acceptance = list(base.get("false_acceptance_candidates", []))
    global_best = str(base["global_measured_best_state_hash"])
    global_best_stable = [row for row in stable if row["candidate_state_hash"] == global_best]
    if false_pruning:
        decision = "reject_partial_order_pruning_before_lns_expansion"
    elif false_acceptance:
        decision = "disable_automatic_acceptance_use_hardware_rerank"
    elif not stable:
        decision = "reject_current_template_lns"
    elif global_best_stable:
        decision = "adopt_template_lns_and_expand_consensus_elite"
    else:
        decision = "retain_lns_frontier_without_automatic_expansion"

    anchor_hashes = sorted(
        state_hash
        for state_hash, record in frontier["records"].items()
        if "anchor" in record["roles"]
    )
    absolute_medians = {
        state_hash: [
            float(first["plan_stats"][state_hash]["median_ms"]),
            float(second["plan_stats"][state_hash]["median_ms"]),
        ]
        for state_hash in dict.fromkeys(
            [*anchor_hashes, global_best, *base["session_absolute_best_state_hashes"]]
        )
    }
    absolute_gain_vs_anchor = {
        anchor_hash: [
            100.0 * (absolute_medians[anchor_hash][index] / absolute_medians[global_best][index] - 1.0)
            for index in range(2)
        ]
        for anchor_hash in anchor_hashes
    }
    session_best_gap = [
        100.0
        * (
            float(session["plan_stats"][global_best]["median_ms"])
            / min(float(row["median_ms"]) for row in session["plan_stats"].values())
            - 1.0
        )
        for session in (first, second)
    ]
    winner_state = frontier["plans"][global_best]["canonical_state"]
    winner_shape = tuple(int(lane["threads"]) for lane in winner_state["lanes"])
    winner_lanes = [
        {
            "core_begin": int(lane["core_begin"]),
            "threads": int(lane["threads"]),
            "task_count": len(lane["tasks"]),
            "route_count": sum(int(task["routes"]) for task in lane["tasks"]),
        }
        for lane in winner_state["lanes"]
    ]
    measured_rows = [
        [row for row in session["comparisons"] if row["role"] == "incomparable_frontier"]
        for session in (first, second)
    ]
    model_hardware_spearman = [
        (
            _spearman(
                [float(row["robust_gain_pct"]) for row in rows],
                [float(row["paired_gain_pct"]["median"]) for row in rows],
            )
            if len(rows) >= 2
            else None
        )
        for rows in measured_rows
    ]
    comparison_models = comparison_models or []
    comparison_frontiers = comparison_frontiers or []
    return {
        "kind": "template_lns_cross_session_analysis",
        "decision": decision,
        "partial_order": {
            "stable_comparisons": len(stable),
            "stable_unique_candidates": len(stable_unique),
            "false_pruning_sentinels": len(false_pruning),
            "false_acceptance_candidates": len(false_acceptance),
            "stable_by_operator": dict(sorted(Counter(row["operator"] for row in stable).items())),
            "stable_by_destroy_size": dict(
                sorted(Counter(_destroy_size(str(row["operator"])) for row in stable).items())
            ),
            "stable_by_parent": dict(
                sorted(Counter(row["anchor_state_hash"] for row in stable).items())
            ),
            "model_hardware_spearman": model_hardware_spearman,
        },
        "winner": {
            "state_hash": global_best,
            "session_absolute_best_state_hashes": base["session_absolute_best_state_hashes"],
            "session_best_gap_pct": session_best_gap,
            "stable_relative_comparisons": global_best_stable,
            "absolute_median_ms": absolute_medians[global_best],
            "absolute_median_gain_vs_anchor_pct": absolute_gain_vs_anchor,
            "shape": list(winner_shape),
            "width_histogram": dict(sorted(Counter(winner_shape).items())),
            "lanes": winner_lanes,
        },
        "budget_comparison": {
            "template_lns": {
                "event_calls": int(model["summary"]["event_calls"]),
                "model_wall_s": float(model["summary"]["search_wall_s"]),
                "hardware_plans": int(frontier["unique_plans"]),
            },
            "closed_local_beam": {
                "event_calls": sum(int(item["summary"]["event_calls"]) for item in comparison_models),
                "model_wall_s": sum(float(item["summary"]["search_wall_s"]) for item in comparison_models),
                "hardware_plans": sum(int(item["unique_plans"]) for item in comparison_frontiers),
            },
        },
        "absolute_median_ms": absolute_medians,
        "base_analysis": base,
    }


def main() -> int:
    args = parse_args()
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    model = json.loads(args.model_artifact.read_text(encoding="utf-8"))
    frontier_sha256 = _sha256(args.frontier)
    if frontier["source"]["sha256"] != _sha256(args.model_artifact):
        raise ValueError("frontier does not identify the supplied model artifact")
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in args.session]
    if any(session["identity"]["frontier_sha256"] != frontier_sha256 for session in sessions):
        raise ValueError("hardware session does not identify the supplied frontier")
    if any(
        session["identity"]["extension_sha256"] != frontier["identity"]["extension_sha256"]
        for session in sessions
    ):
        raise ValueError("hardware session extension does not match the supplied frontier")
    result = analyze_template_lns(
        frontier,
        model,
        *sessions,
        comparison_models=[
            json.loads(path.read_text(encoding="utf-8")) for path in args.beam_model_artifact
        ],
        comparison_frontiers=[
            json.loads(path.read_text(encoding="utf-8")) for path in args.beam_frontier
        ],
    )
    result["identity"] = {
        "frontier_sha256": frontier_sha256,
        "model_artifact_sha256": _sha256(args.model_artifact),
        "session_sha256": [_sha256(path) for path in args.session],
        "comparison_model_sha256": [_sha256(path) for path in args.beam_model_artifact],
        "comparison_frontier_sha256": [_sha256(path) for path in args.beam_frontier],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "stable_unique_candidates": result["partial_order"]["stable_unique_candidates"],
                "false_pruning_sentinels": result["partial_order"]["false_pruning_sentinels"],
                "winner_state_hash": result["winner"]["state_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
