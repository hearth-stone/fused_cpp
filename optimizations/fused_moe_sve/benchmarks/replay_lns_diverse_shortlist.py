#!/usr/bin/env python3
"""Replay the relation-agnostic LNS shortlist on the frozen measured suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import (  # noqa: E402
    POLICY_NAME,
    LnsDiverseShortlist,
    assign_model_score_quantiles,
    features_from_hardware_frontier,
    merge_duplicate_features,
    policy_document,
    policy_sha256,
    select_lns_diverse_shortlist,
    select_lns_global_shortlist,
)


FROZEN_SUITE_DIR = Path("tmp/moe_partial_order_vnd_20260904")
FROZEN_HASHES = {
    "suite_analysis": "8e59f93276e31d183fea6db2741af0807cbe6662c77216ac2ecc7204ae8890bb",
    "pairwise": "a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a",
}
CASES = {
    "high_skew_l1": {
        "frontier": "high_skew_template_lns_frontier.json",
        "frontier_sha256": "ed446199e673fa9da58c7024e3e06ec70d25361caf1a279c6e8be53df9a5d947",
        "model": "high_skew_template_lns_model.json",
        "model_sha256": "32c5a530c4c75462fd06c20f9340ec6e9977eb406d05e88bc3d29b651643d784",
        "session1": "high_skew_lns_session1.json",
        "session1_sha256": "642196316e2384210e9e2ce03ed42f66cc8fe19fe5fb0ea36e5da11551b76dde",
        "session2": "high_skew_lns_session2.json",
        "session2_sha256": "ae226a3a6e20b5e85237d8fa2ae437d9455f776fee5f28d69494c8ae907744de",
        "analysis": "high_skew_template_lns_formal_analysis.json",
        "analysis_sha256": "a2ad7b24dc1445c8e5e3af8fab26706ba14980b0a6f6ee985cdf9170d02d3ad1",
    },
    "high_skew_l2": {
        "frontier": "high_skew_template_lns_depth2_frontier.json",
        "frontier_sha256": "5f928bcc75006eff41a67ca24f66524ec594cb4b237650a98ca00e0ae5ea404d",
        "model": "high_skew_template_lns_depth2_model.json",
        "model_sha256": "b2e01faf7ca3c8e400d133d2413ec04ac5903a53e226b144b0c4bb18274fdc56",
        "session1": "high_skew_lns_depth2_session1.json",
        "session1_sha256": "0b30d9377d8c29f45a3d86a724e297986f0a94ceae287738d93b806cf5ea676b",
        "session2": "high_skew_lns_depth2_session2.json",
        "session2_sha256": "fb49eb367d8b12f61c9d29c797746f75b18d017d49cf8e0b77db7b918439daa2",
        "analysis": "high_skew_template_lns_depth2_analysis.json",
        "analysis_sha256": "aaefffcd9f6197092e5eb5d4aad63871665682273222b17c60abc398b4859718",
    },
    "median": {
        "frontier": "median_template_lns_frontier.json",
        "frontier_sha256": "443c3d3bdde7c7475c04848da745ae056f388d77417be41c37ddb34afa98b416",
        "model": "median_template_lns_model.json",
        "model_sha256": "fb60dfa0f1232a94ac9372c49af310e251048aea7fecc953326537184e561c1d",
        "session1": "median_lns_session1.json",
        "session1_sha256": "27b0b6b549d6c3fb8fa6805245c0fd89fb53cc2dd37068a293e36b97feb553ca",
        "session2": "median_lns_session2.json",
        "session2_sha256": "8bc15fb05f12689d2c2a92a6c618b8d05ebe8d606f62ae7d033aa8e43b4f6434",
        "analysis": "median_template_lns_analysis.json",
        "analysis_sha256": "79f5058a9e612e9e51702f5c471b6de2c11395e1a213df41a9b91414af3cd03f",
    },
    "uniformish": {
        "frontier": "uniformish_template_lns_frontier.json",
        "frontier_sha256": "f53d40a1ac1622059fe7d77e4a25084cb1bf64c48db9105e2b43be250d5df6c5",
        "model": "uniformish_template_lns_model.json",
        "model_sha256": "851905912a823341239988742e31c1b71bbe42beca2a47194b91d453645e0144",
        "session1": "uniformish_lns_session1.json",
        "session1_sha256": "e86795d5485c6eb47d66e2b87f0ba989878b03e9036eddb45c186efff5a70c93",
        "session2": "uniformish_lns_session2.json",
        "session2_sha256": "cea46c04a08965bd68019f23931b61aaef21c5a263477414f4f2eb13b86f2d62",
        "analysis": "uniformish_template_lns_analysis.json",
        "analysis_sha256": "088d296484034772c4a970cc821cd3fb9eb00274227df7e5d908b1c96275af7c",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=FROZEN_SUITE_DIR)
    parser.add_argument("--budgets", default="8,12,16,24,32")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checked(path: Path, expected: str) -> dict[str, Any]:
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"frozen hash mismatch for {path}: {digest} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def _anchors(frontier: Mapping[str, Any]) -> set[str]:
    return {
        state_hash
        for state_hash, record in frontier["records"].items()
        if "anchor" in record["roles"]
    }


def _fastest(plan_stats: Mapping[str, Mapping[str, Any]], keys: Iterable[str] | None = None) -> str:
    universe = list(keys) if keys is not None else list(plan_stats)
    return min(universe, key=lambda key: (float(plan_stats[key]["median_ms"]), key))


def _session_metrics(
    *,
    selected: Sequence[str],
    anchors: set[str],
    plan_stats: Mapping[str, Mapping[str, Any]],
    winner: str,
    stable_hashes: set[str],
    parent_best: Mapping[str, str],
) -> dict[str, Any]:
    retained = set(selected) | set(anchors)
    absolute_best = _fastest(plan_stats)
    selected_best = _fastest(plan_stats, [key for key in retained if key in plan_stats])
    universe_best_ms = float(plan_stats[absolute_best]["median_ms"])
    selected_best_ms = float(plan_stats[selected_best]["median_ms"])
    parent_recall = {
        parent: parent_best[parent] in retained
        for parent in parent_best
    }
    return {
        "absolute_best_state_hash": absolute_best,
        "absolute_best_retained": absolute_best in retained,
        "consensus_winner_retained": winner in retained,
        "selected_best_state_hash": selected_best,
        "selected_best_regret_pct": 100.0 * (selected_best_ms / universe_best_ms - 1.0),
        "strict_stable_recall": (
            None
            if not stable_hashes
            else sum(key in retained for key in stable_hashes) / len(stable_hashes)
        ),
        "per_parent_best_recall": parent_recall,
        "per_parent_best_retained": all(parent_recall.values()),
    }


def replay_case(
    frontier: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    analysis: dict[str, Any],
    *,
    budgets: Sequence[int],
) -> dict[str, Any]:
    by_start = features_from_hardware_frontier(frontier)
    anchors = _anchors(frontier)
    winner = str(analysis["winner"]["state_hash"])
    stable_hashes = {
        str(row["candidate_state_hash"])
        for row in analysis.get("base_analysis", {}).get("stable_frontier_candidates", [])
    }
    parent_candidates: dict[str, set[str]] = {anchor: {anchor} for anchor in anchors}
    relation_counts: Counter[str] = Counter()
    for record in frontier["records"].values():
        for comparison in record["comparisons"]:
            if comparison.get("role") in {"anchor", "carried_anchor"}:
                continue
            parent_candidates.setdefault(str(comparison["anchor_state_hash"]), set()).add(
                str(record["state_hash"])
            )
            relation = str((comparison.get("partial_order") or {}).get("relation", "unknown"))
            relation_counts[relation] += 1

    ranked = {
        start: select_lns_diverse_shortlist(features, shortlist_budget=max(budgets), audit_budget=max(budgets))
        for start, features in by_start.items()
    }
    shuffle_hashes = []
    for seed in (1, 2, 3):
        shuffled = {}
        for start, features in by_start.items():
            order = list(features)
            random.Random(seed).shuffle(order)
            shuffled[start] = select_lns_diverse_shortlist(
                order,
                shortlist_budget=16,
                audit_budget=32,
            ).selected_keys
        shuffle_hashes.append(tuple((start, shuffled[start]) for start in sorted(shuffled)))
    deterministic = len(set(shuffle_hashes)) == 1

    by_budget = {}
    for budget in budgets:
        per_start = {}
        for start, shortlist in ranked.items():
            selected_keys = shortlist.ranked_keys[:budget]
            audit_keys = shortlist.ranked_keys[: max(budget, 32)] if budget <= 32 else selected_keys
            per_start[start] = LnsDiverseShortlist(
                ranked_keys=shortlist.ranked_keys,
                selected_keys=selected_keys,
                audit_keys=audit_keys,
                budget_deferred_keys=shortlist.ranked_keys[budget:],
                coverage=shortlist.coverage,
                shortlist_budget=budget,
                audit_budget=max(budget, 32) if budget <= 32 else budget,
            )
        global_result = select_lns_global_shortlist(
            per_start,
            shortlist_budget=budget,
            audit_budget=max(budget, 32) if budget <= 32 else budget,
        )
        selected = tuple(global_result["selected_keys"])
        coverage_features = []
        for start, features in by_start.items():
            keep = set(per_start[start].selected_keys)
            assigned = assign_model_score_quantiles(merge_duplicate_features(features))
            coverage_features.extend(item for item in assigned if item.state_hash in keep)
        parent_best = {
            parent: _fastest(first["plan_stats"], [key for key in keys if key in first["plan_stats"]])
            for parent, keys in parent_candidates.items()
            if any(key in first["plan_stats"] for key in keys)
        }
        by_budget[str(budget)] = {
            "global": global_result,
            "session_1": _session_metrics(
                selected=selected,
                anchors=anchors,
                plan_stats=first["plan_stats"],
                winner=winner,
                stable_hashes=stable_hashes,
                parent_best=parent_best,
            ),
            "session_2": _session_metrics(
                selected=selected,
                anchors=anchors,
                plan_stats=second["plan_stats"],
                winner=winner,
                stable_hashes=stable_hashes,
                parent_best={
                    parent: _fastest(
                        second["plan_stats"],
                        [key for key in keys if key in second["plan_stats"]],
                    )
                    for parent, keys in parent_candidates.items()
                    if any(key in second["plan_stats"] for key in keys)
                },
            ),
            "coverage": {
                "operators": sorted({item.operator for item in coverage_features}),
                "scopes": sorted({item.scope for item in coverage_features}),
                "target_destroy_sizes": sorted({item.target_destroy_size for item in coverage_features}),
                "actual_closure_bins": sorted({item.actual_closure_bin for item in coverage_features}),
                "width_histograms": len({item.candidate_width_histogram for item in coverage_features}),
                "domain_assignment_signatures": len(
                    {item.domain_assignment_signature for item in coverage_features}
                ),
                "score_quantiles": sorted({item.model_score_quantile for item in coverage_features}),
            },
            "relation_composition": dict(sorted(relation_counts.items())),
            "no_dominance_pruning": True,
            "no_automatic_acceptance": True,
        }
    k16 = by_budget["16"]
    gate = {
        "absolute_best_both_sessions": bool(
            k16["session_1"]["absolute_best_retained"] and k16["session_2"]["absolute_best_retained"]
        ),
        "consensus_winner_both_sessions": bool(
            k16["session_1"]["consensus_winner_retained"]
            and k16["session_2"]["consensus_winner_retained"]
        ),
        "zero_regret_both_sessions": bool(
            k16["session_1"]["selected_best_regret_pct"] == 0.0
            and k16["session_2"]["selected_best_regret_pct"] == 0.0
        ),
        "no_dominance_or_acceptance": True,
        "deterministic_under_shuffle": deterministic,
    }
    gate["pass"] = all(gate.values())
    return {
        "starts": sorted(by_start),
        "measured_candidates": sum(len(items) for items in by_start.values()),
        "unique_measured_states": len(frontier["plans"]) - len(anchors),
        "anchors": sorted(anchors),
        "winner_state_hash": winner,
        "deterministic_under_shuffle": deterministic,
        "budgets": by_budget,
        "gate": gate,
    }


def replay_suite(suite_dir: Path, budgets: Sequence[int]) -> dict[str, Any]:
    cases = {}
    for name, spec in CASES.items():
        frontier = _load_checked(suite_dir / spec["frontier"], spec["frontier_sha256"])
        first = _load_checked(suite_dir / spec["session1"], spec["session1_sha256"])
        second = _load_checked(suite_dir / spec["session2"], spec["session2_sha256"])
        analysis = _load_checked(suite_dir / spec["analysis"], spec["analysis_sha256"])
        cases[name] = replay_case(frontier, first, second, analysis, budgets=budgets)
    k16_pass = all(case["gate"]["pass"] for case in cases.values())
    return {
        "kind": "lns_diverse_shortlist_measured_suite_replay",
        "policy": POLICY_NAME,
        "policy_document": policy_document(),
        "policy_sha256": policy_sha256(),
        "budgets": list(budgets),
        "cases": cases,
        "gate": {
            "k16_all_cases": k16_pass,
            "decision": "pass_design_replay" if k16_pass else "fix_structural_selection_before_holdout",
        },
    }


def main() -> int:
    args = parse_args()
    budgets = tuple(int(item) for item in args.budgets.split(",") if item)
    suite_analysis = args.suite_dir / "template_lns_suite_analysis.json"
    if _sha256(suite_analysis) != FROZEN_HASHES["suite_analysis"]:
        raise ValueError("frozen suite analysis hash mismatch")
    result = replay_suite(args.suite_dir, budgets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["gate"]["decision"],
                "k16_all_cases": result["gate"]["k16_all_cases"],
                "cases": {
                    name: case["gate"]
                    for name, case in result["cases"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["gate"]["k16_all_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
