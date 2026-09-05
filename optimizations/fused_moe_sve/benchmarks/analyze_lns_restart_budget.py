#!/usr/bin/env python3
"""Compare equal-budget 1-restart vs multi-restart LNS hardware sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-restart-frontier", type=Path, required=True)
    parser.add_argument("--one-restart-model", type=Path, required=True)
    parser.add_argument("--one-restart-session", type=Path, nargs=2, required=True)
    parser.add_argument("--two-restart-frontier", type=Path, required=True)
    parser.add_argument("--two-restart-model", type=Path, required=True)
    parser.add_argument("--two-restart-session", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fastest(plan_stats: dict[str, dict[str, object]], keys: set[str]) -> str | None:
    present = [key for key in keys if key in plan_stats]
    if not present:
        return None
    return min(present, key=lambda key: (float(plan_stats[key]["median_ms"]), key))


def _gain_pct(reference_ms: float, candidate_ms: float) -> float:
    return 100.0 * (reference_ms / candidate_ms - 1.0)


def _role_set(frontier: dict[str, object], role: str) -> set[str]:
    return {
        state_hash
        for state_hash, record in frontier["records"].items()
        if role in record["roles"]
    }


def analyze_arm(
    frontier: dict[str, object],
    first: dict[str, object],
    second: dict[str, object],
    model: dict[str, object],
) -> dict[str, object]:
    if set(frontier["plans"]) != set(first["plan_stats"]) or set(frontier["plans"]) != set(
        second["plan_stats"]
    ):
        raise ValueError("frontier and hardware sessions do not contain the same plan set")
    anchors = _role_set(frontier, "anchor")
    elite = _role_set(frontier, "known_hardware_elite")
    selected = _role_set(frontier, "lns_diverse_top16")
    stratified = _role_set(frontier, "lns_stratified_outside_top32")
    if len(elite) > 1:
        raise ValueError("expected at most one known hardware elite")
    elite_hash = next(iter(elite), None)

    def session_metrics(stats: dict[str, dict[str, object]]) -> dict[str, object]:
        strongest_anchor = _fastest(stats, anchors)
        selected_best = _fastest(stats, selected)
        stratified_best = _fastest(stats, stratified)
        measured_best = _fastest(stats, set(stats))
        strongest_ms = float(stats[strongest_anchor]["median_ms"])
        selected_ms = float(stats[selected_best]["median_ms"])
        elite_ms = None if elite_hash is None else float(stats[elite_hash]["median_ms"])
        payload = {
            "strongest_anchor_state_hash": strongest_anchor,
            "strongest_anchor_median_ms": strongest_ms,
            "selected_best_state_hash": selected_best,
            "selected_best_median_ms": selected_ms,
            "measured_best_state_hash": measured_best,
            "measured_best_median_ms": float(stats[measured_best]["median_ms"]),
            "selected_gain_vs_strongest_anchor_pct": _gain_pct(strongest_ms, selected_ms),
            "stratified_best_state_hash": stratified_best,
            "stratified_beats_selected": bool(
                stratified_best is not None
                and float(stats[stratified_best]["median_ms"]) < selected_ms
            ),
        }
        if elite_ms is not None:
            payload["elite_state_hash"] = elite_hash
            payload["elite_median_ms"] = elite_ms
            payload["elite_gain_vs_strongest_anchor_pct"] = _gain_pct(strongest_ms, elite_ms)
            payload["selected_gain_vs_elite_pct"] = _gain_pct(elite_ms, selected_ms)
            payload["selected_regret_vs_elite_pct"] = _gain_pct(selected_ms, elite_ms)
        return payload

    first_metrics = session_metrics(first["plan_stats"])
    second_metrics = session_metrics(second["plan_stats"])
    stable_vs_anchor = bool(
        first_metrics["selected_gain_vs_strongest_anchor_pct"] > 2.0
        and second_metrics["selected_gain_vs_strongest_anchor_pct"] > 2.0
    )
    beats_elite_both = bool(
        first_metrics.get("selected_gain_vs_elite_pct", -1.0) > 0.0
        and second_metrics.get("selected_gain_vs_elite_pct", -1.0) > 0.0
    )
    return {
        "unique_plans": len(frontier["plans"]),
        "role_counts": frontier.get("role_counts"),
        "search": {
            "starts": model["summary"].get("starts"),
            "restarts_per_parent": model["method"].get("restarts_per_parent"),
            "neighbors_per_operator": model["method"].get("neighbors_per_operator"),
            "maximum_exact_candidate_budget": model["method"].get("maximum_exact_candidate_budget"),
            "unique_candidates": model["summary"].get("unique_candidates"),
            "event_calls": model["summary"].get("event_calls"),
            "search_wall_s": model["summary"].get("search_wall_s"),
        },
        "session_1": first_metrics,
        "session_2": second_metrics,
        "two_session_selected_gain_vs_anchor_over_2pct": stable_vs_anchor,
        "two_session_selected_beats_elite": beats_elite_both,
        "hardware_wall_s": [
            (first.get("summary") or {}).get("measurement_wall_s", first.get("measurement_wall_s")),
            (second.get("summary") or {}).get("measurement_wall_s", second.get("measurement_wall_s")),
        ],
    }


def main() -> int:
    args = parse_args()
    one_frontier = json.loads(args.one_restart_frontier.read_text(encoding="utf-8"))
    two_frontier = json.loads(args.two_restart_frontier.read_text(encoding="utf-8"))
    one_model = json.loads(args.one_restart_model.read_text(encoding="utf-8"))
    two_model = json.loads(args.two_restart_model.read_text(encoding="utf-8"))
    one_sessions = [json.loads(path.read_text(encoding="utf-8")) for path in args.one_restart_session]
    two_sessions = [json.loads(path.read_text(encoding="utf-8")) for path in args.two_restart_session]
    one = analyze_arm(one_frontier, *one_sessions, one_model)
    two = analyze_arm(two_frontier, *two_sessions, two_model)
    exact_budget_equal = (
        one["search"]["maximum_exact_candidate_budget"]
        == two["search"]["maximum_exact_candidate_budget"]
    )
    hardware_equal = one["unique_plans"] == two["unique_plans"]
    result = {
        "kind": "lns_restart_budget_comparison",
        "exact_candidate_budget_equal": exact_budget_equal,
        "hardware_plan_count_equal": hardware_equal,
        "one_restart": one,
        "two_restart": two,
        "identity": {
            "one_restart_frontier_sha256": _sha256(args.one_restart_frontier),
            "two_restart_frontier_sha256": _sha256(args.two_restart_frontier),
            "one_restart_model_sha256": _sha256(args.one_restart_model),
            "two_restart_model_sha256": _sha256(args.two_restart_model),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "exact_candidate_budget_equal": exact_budget_equal,
                "hardware_plan_count_equal": hardware_equal,
                "one_restart_unique_plans": one["unique_plans"],
                "two_restart_unique_plans": two["unique_plans"],
                "one_restart_selected_gain_vs_anchor": [
                    one["session_1"]["selected_gain_vs_strongest_anchor_pct"],
                    one["session_2"]["selected_gain_vs_strongest_anchor_pct"],
                ],
                "two_restart_selected_gain_vs_anchor": [
                    two["session_1"]["selected_gain_vs_strongest_anchor_pct"],
                    two["session_2"]["selected_gain_vs_strongest_anchor_pct"],
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if exact_budget_equal and hardware_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
