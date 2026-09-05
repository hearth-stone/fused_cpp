#!/usr/bin/env python3
"""Compare nested LNS top-16 selection against a measured top-32 audit frontier."""

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
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path)
    parser.add_argument("--session", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fastest(plan_stats: dict[str, dict[str, object]], keys: set[str] | None = None) -> str:
    universe = keys if keys is not None else set(plan_stats)
    return min(
        (key for key in universe if key in plan_stats),
        key=lambda key: (float(plan_stats[key]["median_ms"]), key),
    )


def _subset_metrics(
    plan_stats: dict[str, dict[str, object]],
    selected: set[str],
    anchors: set[str],
) -> dict[str, object]:
    retained = selected | anchors
    absolute_best = _fastest(plan_stats)
    selected_best = _fastest(plan_stats, retained)
    best_ms = float(plan_stats[absolute_best]["median_ms"])
    selected_ms = float(plan_stats[selected_best]["median_ms"])
    return {
        "absolute_best_state_hash": absolute_best,
        "absolute_best_retained": absolute_best in retained,
        "selected_best_state_hash": selected_best,
        "selected_best_regret_pct": 100.0 * (selected_ms / best_ms - 1.0),
        "absolute_median_ms": {
            "best": best_ms,
            "selected_best": selected_ms,
        },
    }


def analyze_lns_diverse_sessions(
    frontier: dict[str, object],
    first: dict[str, object],
    second: dict[str, object],
    model: dict[str, object] | None = None,
) -> dict[str, object]:
    if frontier.get("kind") != "partial_order_hardware_frontier":
        raise ValueError("frontier has an unexpected kind")
    if set(frontier["plans"]) != set(first["plan_stats"]) or set(frontier["plans"]) != set(
        second["plan_stats"]
    ):
        raise ValueError("frontier and hardware sessions do not contain the same plan set")
    for session in (first, second):
        if session.get("kind") != "partial_order_hardware_frontier_session":
            raise ValueError("hardware session has an unexpected kind")
        if session["identity"]["frontier_sha256"] is None:
            raise ValueError("hardware session is missing frontier identity")

    anchors = {
        state_hash
        for state_hash, record in frontier["records"].items()
        if "anchor" in record["roles"]
    }
    top16 = {
        state_hash
        for state_hash, record in frontier["records"].items()
        if "lns_diverse_top16" in record["roles"]
    }
    audit = {
        state_hash
        for state_hash, record in frontier["records"].items()
        if "lns_diverse_top16" in record["roles"] or "lns_diverse_audit_top32" in record["roles"]
    }
    if not top16 <= audit:
        raise ValueError("top-16 is not nested inside the audit top-32")
    selected_and_anchors = top16 | anchors
    first_gate = _subset_metrics(first["plan_stats"], top16, anchors)
    second_gate = _subset_metrics(second["plan_stats"], top16, anchors)
    first_gate["absolute_best_in_top16"] = first_gate["absolute_best_retained"]
    second_gate["absolute_best_in_top16"] = second_gate["absolute_best_retained"]
    normalized = {
        state_hash: sum(
            float(stats[state_hash]["median_ms"])
            / min(float(row["median_ms"]) for row in stats.values())
            for stats in (first["plan_stats"], second["plan_stats"])
        )
        for state_hash in first["plan_stats"]
    }
    consensus = min(normalized, key=lambda key: (normalized[key], key))
    illegal_roles = {
        role
        for record in frontier["records"].values()
        for role in record["roles"]
        if role
        in {
            "candidate_better_frontier",
            "candidate_worse_sentinel",
            "dominated",
        }
    }
    strongest_anchor = {
        "session_1": _fastest(first["plan_stats"], anchors),
        "session_2": _fastest(second["plan_stats"], anchors),
    }
    strongest_anchor_ms = [
        float(first["plan_stats"][strongest_anchor["session_1"]]["median_ms"]),
        float(second["plan_stats"][strongest_anchor["session_2"]]["median_ms"]),
    ]
    audit_gain_vs_anchor = [
        100.0
        * (
            strongest_anchor_ms[index]
            / min(float(stats[key]["median_ms"]) for key in audit | anchors)
            - 1.0
        )
        for index, stats in enumerate((first["plan_stats"], second["plan_stats"]))
    ]
    neighborhood_found_stable_improvement = all(gain > 2.0 for gain in audit_gain_vs_anchor)
    identity_match = (
        first["identity"].get("frontier_sha256") == second["identity"].get("frontier_sha256")
        and first["identity"].get("extension_sha256") == second["identity"].get("extension_sha256")
    )
    gate = {
        "top16_contains_top32_absolute_best_both_sessions": bool(
            first_gate["absolute_best_retained"] and second_gate["absolute_best_retained"]
        ),
        "top16_contains_consensus_winner": consensus in selected_and_anchors,
        "zero_regret_both_sessions": bool(
            first_gate["selected_best_regret_pct"] == 0.0
            and second_gate["selected_best_regret_pct"] == 0.0
        ),
        "anchors_retained": bool(anchors) and anchors <= set(frontier["plans"]),
        "automatic_acceptance_disabled": True,
        "dominance_pruning_disabled": not bool(illegal_roles),
        "nested_top16_prefix": top16 <= audit,
        "session_identities_match": identity_match,
    }
    gate["pass"] = all(gate.values())
    if not neighborhood_found_stable_improvement:
        decision = "neighborhood_or_proposal_failure_for_independent_seed"
    elif not (first_gate["absolute_best_retained"] or second_gate["absolute_best_retained"]):
        decision = "reject_selector_v1_require_new_independent_frontier"
    elif not gate["pass"]:
        decision = "keep_v1_experimental_report_budget_tradeoff"
    else:
        decision = "adopt_selector_v1_offline_lns_shortlist"
    sensitivity = {}
    if model is not None:
        from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import (
            LnsDiverseShortlist,
            select_lns_global_shortlist,
        )

        per_start = {
            start: LnsDiverseShortlist.from_dict(run["iterations"][0]["lns_diverse_shortlist"])
            for start, run in model["runs"].items()
            if run.get("iterations") and run["iterations"][0].get("lns_diverse_shortlist")
        }
        for budget in (8, 12, 16, 24, 32):
            sliced = {}
            for start, shortlist in per_start.items():
                selected_keys = shortlist.ranked_keys[:budget]
                sliced[start] = LnsDiverseShortlist(
                    ranked_keys=shortlist.ranked_keys,
                    selected_keys=selected_keys,
                    audit_keys=shortlist.ranked_keys[: max(budget, 32)],
                    budget_deferred_keys=shortlist.ranked_keys[budget:],
                    coverage=shortlist.coverage,
                    shortlist_budget=budget,
                    audit_budget=max(budget, 32),
                )
            global_result = select_lns_global_shortlist(
                sliced,
                shortlist_budget=budget,
                audit_budget=max(budget, 32),
            )
            selected = set(global_result["selected_keys"])
            sensitivity[str(budget)] = {
                "session_1": _subset_metrics(first["plan_stats"], selected, anchors),
                "session_2": _subset_metrics(second["plan_stats"], selected, anchors),
                "selected_count": len(selected),
            }
    return {
        "kind": "lns_diverse_hardware_frontier_cross_session_analysis",
        "decision": decision,
        "gate": gate,
        "session_1": first_gate,
        "session_2": second_gate,
        "consensus_winner_state_hash": consensus,
        "strongest_anchor_state_hash": strongest_anchor,
        "audit_gain_vs_strongest_anchor_pct": audit_gain_vs_anchor,
        "neighborhood_found_stable_improvement": neighborhood_found_stable_improvement,
        "budget_sensitivity": sensitivity,
        "role_counts": {
            "anchor": len(anchors),
            "lns_diverse_top16": len(top16),
            "lns_diverse_audit_top32": len(audit - top16),
        },
        "unique_plans": len(frontier["plans"]),
    }


def main() -> int:
    args = parse_args()
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    sessions = [json.loads(path.read_text(encoding="utf-8")) for path in args.session]
    frontier_sha256 = _sha256(args.frontier)
    for session, path in zip(sessions, args.session, strict=True):
        if session["identity"]["frontier_sha256"] != frontier_sha256:
            raise ValueError(f"hardware session does not identify the supplied frontier: {path}")
    model = (
        json.loads(args.model_artifact.read_text(encoding="utf-8"))
        if args.model_artifact is not None
        else None
    )
    if model is not None and frontier["source"]["sha256"] != _sha256(args.model_artifact):
        raise ValueError("frontier does not identify the supplied model artifact")
    result = analyze_lns_diverse_sessions(frontier, *sessions, model=model)
    result["identity"] = {
        "frontier_sha256": frontier_sha256,
        "model_artifact_sha256": None if args.model_artifact is None else _sha256(args.model_artifact),
        "session_sha256": [_sha256(path) for path in args.session],
        "extension_sha256": frontier["identity"].get("extension_sha256"),
        "calibration_sha256": frontier["identity"].get("calibration_sha256"),
        "pairwise_calibration_sha256": frontier["identity"].get("pairwise_calibration_sha256"),
        "policy": frontier.get("method", {}).get("policy"),
        "policy_sha256": frontier.get("method", {}).get("policy_sha256")
        or (model or {}).get("method", {}).get("lns_shortlist_policy_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "pass": result["gate"]["pass"],
                "consensus_winner_state_hash": result["consensus_winner_state_hash"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
