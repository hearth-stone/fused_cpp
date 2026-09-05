#!/usr/bin/env python3
"""Locate measured fast plans in frozen template-LNS enumeration and sampling.

This is a Lab diagnostic. It drives the shipped
``enumerate_template_lns_neighbors``, ``_sample_neighborhood``, and
``sample_template_lns_neighborhood`` entry points on saved start inputs. It does
not change the sampler, selector, K, or production planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from executable_plan_neighborhood import (  # noqa: E402
    TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
    TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
    TEMPLATE_LNS_SCOPES,
    _sample_neighborhood,
    _template_lns_operator,
    enumerate_template_lns_neighbors,
    sample_template_lns_neighborhood,
)
from interval_planner import IntervalPlanner  # noqa: E402

from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import (  # noqa: E402
    closure_bin,
    state_from_canonical_payload,
    width_histogram,
)
from optimizations.fused_moe_sve.benchmarks.lns_structural_presample import (  # noqa: E402
    StructuralCandidate,
    bin_occupancy,
    closure_width_policy_descriptor,
    closure_width_policy_sha256,
    policy_descriptor,
    policy_sha256,
    sample_structural_coverage_closure_width,
    sample_structural_coverage_then_random,
)

DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "bench_assets/moe_paper/arm_codex_numa3_80c_temporal"
    / "analytic_machine_numa3_80c_narrow_merge_v8_20260903.json"
)
FEATURE_FIELDS = (
    "actual_closure_bin",
    "actual_closure_size",
    "candidate_width_histogram",
    "changed_core_begin",
    "changed_core_end",
    "cross_domain_lane_count",
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
MEASURED_FAST_PLANS = {
    "0418b88445c2f88488ca10a1eeae3aeb7b6080e806e241eacfed17ec10904bf6": {
        "label": "previous_selected_and_two_restart_winner",
        "reason": ">=2% vs reconstructed full in every frozen-protocol median session",
    },
    "2ab43572d14e11060e4fc07b3aa7e0c1780553233dfda61abe74f08abc51e973": {
        "label": "known_median_elite",
        "reason": "preserved hardware elite; >=2% vs full in every frozen-protocol median session",
    },
    "7cac2afd50154d1f053e46998aa6452d8a3efa2f5f9278a2783a11fc15f02db8": {
        "label": "independent_median_consensus_and_one_restart_selected",
        "reason": ">=2% vs full in 1-restart sessions; independent-median session-2 absolute best",
    },
    "1faf090a718ee781f860c2d06554162cacd23d7fbf51e321ac5b58b74d8e2c0e": {
        "label": "independent_median_session1_ge2_audit",
        "reason": "independent-median session-1 absolute median >=2% vs full; audit-only",
    },
    "07355dde6a1770dc5844a11f9941df95e47ab7a4d928918a308d80bbfa4a7227": {
        "label": "independent_median_session1_ge2_audit",
        "reason": "independent-median session-1 absolute median >=2% vs full; audit-only",
    },
}
FULL_PARENT = "98a32da5c3ee195802fbc4291ecf0227f57af8f94dcd4330653f3b6244c2d10e"
DEFAULT_STARTS = ("lns_00_r00", "lns_00_r01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--proposal-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--start", action="append", dest="starts")
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument(
        "--skip-enumeration",
        action="store_true",
        help="Record frozen-model membership only; do not replay enumeration.",
    )
    parser.add_argument("--feature-replay-output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_sample_seed(start_name: str, proposal_seed: int, iteration: int, strategy: str) -> int:
    start_seed = sum((index + 1) * byte for index, byte in enumerate(start_name.encode("utf-8")))
    delta = 0xC17 if strategy == "critical" else 0xA11
    return int(proposal_seed) ^ start_seed ^ (iteration * 0x9E3779B1) ^ delta


def template_operators() -> tuple[str, ...]:
    return tuple(
        _template_lns_operator(scope, destroy_size, beam_width)
        for scope in TEMPLATE_LNS_SCOPES
        for destroy_size, beam_width in zip(
            TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
            TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
            strict=True,
        )
    )


def compact_feature(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in FEATURE_FIELDS}


def hashes_by_operator(sampled: Any, operators: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    cursor = 0
    neighbors = sampled.neighbors
    for operator in operators:
        count = int(sampled.sampled_by_operator[operator])
        grouped[operator] = [neighbor.state.canonical_hash() for neighbor in neighbors[cursor : cursor + count]]
        cursor += count
    if cursor != len(neighbors):
        raise RuntimeError("sampled neighborhood is not concatenated in operator order")
    return grouped


def membership_for_target(model: Mapping[str, Any], target: str) -> dict[str, Any]:
    per_start = []
    for start, run in model["runs"].items():
        iteration = run["iterations"][0]
        shortlist = iteration["lns_diverse_shortlist"]
        ranked = list(shortlist["ranked_keys"])
        selected = list(shortlist["selected_keys"])
        audit = list(shortlist.get("audit_keys") or [])
        feature = next(
            (row for row in (iteration.get("candidate_features") or []) if row["state_hash"] == target),
            None,
        )
        if feature is None and target not in ranked and target not in selected and target not in audit:
            continue
        per_start.append(
            {
                "audit": target in audit,
                "audit_index": audit.index(target) if target in audit else None,
                "feature": None if feature is None else compact_feature(feature),
                "in_candidate_features": feature is not None,
                "n_ranked": len(ranked),
                "parent_state_hash": run.get("initial_state_hash"),
                "ranked_index": ranked.index(target) if target in ranked else None,
                "selected": target in selected,
                "start": start,
            }
        )
    pooled_hits = []
    for parent, record in (model.get("parent_pooled_shortlists") or {}).items():
        shortlist = record.get("shortlist") if isinstance(record, Mapping) else None
        if not isinstance(shortlist, Mapping):
            continue
        ranked = list(shortlist.get("ranked_keys") or [])
        selected = list(shortlist.get("selected_keys") or [])
        audit = list(shortlist.get("audit_keys") or [])
        stratified = list(record.get("stratified_keys") or [])
        if target not in ranked and target not in selected and target not in audit and target not in stratified:
            continue
        pooled_hits.append(
            {
                "audit": target in audit,
                "audit_index": audit.index(target) if target in audit else None,
                "n_ranked": len(ranked),
                "parent_state_hash": parent,
                "ranked_index": ranked.index(target) if target in ranked else None,
                "selected": target in selected,
                "stratified": target in stratified,
            }
        )
    return {"per_start": per_start, "parent_pooled": pooled_hits}


def window_selector_for_model(model: Mapping[str, Any], calibration: Path):
    analytic = AnalyticMoeCostModel(
        calibration,
        hidden_size=int(model["shape"]["hidden"]),
        intermediate_size=int(model["shape"]["intermediate"]),
        global_experts=int(model["shape"]["experts"]),
        local_experts=int(model["shape"]["experts"]),
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    interval = IntervalPlanner(
        analytic,
        num_cores=int(model["shape"]["threads"]),
        cpu_ids=list(range(int(model["shape"]["threads"]))),
    )
    policy = interval._stage_window_policy()

    def window_selector(routes: int, threads: int) -> tuple[int, int]:
        return (0, 0) if policy is None else policy.select(routes, threads)

    return analytic, interval, window_selector


def enumerate_start(
    model: Mapping[str, Any],
    start_name: str,
    proposal_seed: int,
    targets: Sequence[str],
    analytic: Any,
    interval: Any,
    window_selector: Any,
) -> dict[str, Any]:
    run = model["runs"][start_name]
    state = state_from_canonical_payload(run["initial_canonical_state"])
    iteration = run["iterations"][0]
    expert_ids = {
        "critical": list(iteration["critical_expert_ids"]),
        "random": list(iteration["random_expert_ids"]),
    }
    per_operator = int(model["method"]["neighbors_per_operator"])
    operators = template_operators()
    reports: dict[str, Any] = {}
    target_set = set(targets)
    for strategy, ids in expert_ids.items():
        seed = start_sample_seed(start_name, proposal_seed, 0, strategy)
        print(f"enumerate {start_name} {strategy} seed={seed}", flush=True)
        neighbors = list(
            enumerate_template_lns_neighbors(
                state,
                allowed_widths=interval.widths,
                isolated_cost=analytic.T_iso,
                window_selector=window_selector,
                critical_expert_ids=ids,
                destroy_sizes=TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
                repair_beam_widths=TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
                templates_per_block=int(model["method"]["templates_per_block"]),
            )
        )
        emitted_records: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
        unique_owner: dict[str, str] = {}
        seen = {state.canonical_hash()}
        unique_items: dict[str, list[StructuralCandidate]] = {operator: [] for operator in operators}
        for neighbor in neighbors:
            hashed = neighbor.state.canonical_hash()
            closure_size = len(neighbor.moved_experts)
            if hashed in target_set:
                emitted_records[hashed].append(
                    {
                        "moved_expert_count": closure_size,
                        "operator": neighbor.operator,
                    }
                )
            if hashed in seen:
                continue
            seen.add(hashed)
            unique_items[neighbor.operator].append(
                StructuralCandidate(
                    state_hash=hashed,
                    actual_closure_size=closure_size,
                    width_histogram=width_histogram(neighbor.state),
                )
            )
            if hashed in target_set:
                unique_owner[hashed] = neighbor.operator
        sampled = _sample_neighborhood(
            state,
            neighbors,
            operators=operators,
            per_operator=per_operator,
            seed=seed,
        )
        unique_cap = max(int(count) for count in sampled.unique_by_operator.values())
        shuffled = _sample_neighborhood(
            state,
            neighbors,
            operators=operators,
            per_operator=max(unique_cap, 1),
            seed=seed,
        )
        shipped = sample_template_lns_neighborhood(
            state,
            allowed_widths=interval.widths,
            isolated_cost=analytic.T_iso,
            window_selector=window_selector,
            critical_expert_ids=ids,
            destroy_sizes=TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
            repair_beam_widths=TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
            templates_per_block=int(model["method"]["templates_per_block"]),
            per_operator=per_operator,
            seed=seed,
        )
        sampled_order = hashes_by_operator(sampled, operators)
        shuffled_order = hashes_by_operator(shuffled, operators)
        shipped_hashes = [neighbor.state.canonical_hash() for neighbor in shipped.neighbors]
        sampled_hashes = [neighbor.state.canonical_hash() for neighbor in sampled.neighbors]
        prefix_ok = all(
            shuffled_order[operator][: len(sampled_order[operator])] == sampled_order[operator]
            for operator in operators
        )
        target_reports = {}
        for target in targets:
            owner = unique_owner.get(target)
            shuffle_index = None
            unique_count = None
            if owner is not None:
                unique_count = len(shuffled_order[owner])
                if target in shuffled_order[owner]:
                    shuffle_index = shuffled_order[owner].index(target)
            width = None
            if owner is not None:
                for item in unique_items[owner]:
                    if item.state_hash == target:
                        width = [list(pair) for pair in item.width_histogram]
                        break
            target_reports[target] = {
                "sampled": target in sampled_hashes,
                "shipped_sample": target in shipped_hashes,
                "shuffle_index": shuffle_index,
                "unique": owner is not None,
                "unique_count_for_owner": unique_count,
                "unique_operator": owner,
                "width_histogram": width,
                "would_sample_n25": shuffle_index is not None and shuffle_index < 25,
                "would_sample_n50": shuffle_index is not None and shuffle_index < 50,
                "emitted": bool(emitted_records[target]),
                "emitted_records": emitted_records[target],
            }
        candidate_v1 = sample_structural_coverage_then_random(
            unique_items,
            operators=operators,
            per_operator=per_operator,
            seed=seed,
        )
        candidate_v2 = sample_structural_coverage_closure_width(
            unique_items,
            operators=operators,
            per_operator=per_operator,
            seed=seed,
        )
        candidate_hashes_v1 = [key for operator in operators for key in candidate_v1[operator]]
        candidate_hashes_v2 = [key for operator in operators for key in candidate_v2[operator]]
        reports[strategy] = {
            "emitted": len(neighbors),
            "prefix_matches_full_shuffle": prefix_ok,
            "presampler": {
                "policy": policy_descriptor(),
                "policy_sha256": policy_sha256(),
                "targets_sampled": {
                    target: target in candidate_hashes_v1 for target in targets
                },
                "bin_occupancy_by_operator": {
                    operator: {
                        "unique": bin_occupancy(unique_items[operator]),
                        "baseline": bin_occupancy(
                            [
                                item
                                for item in unique_items[operator]
                                if item.state_hash in set(sampled_order[operator])
                            ]
                        ),
                        "candidate": bin_occupancy(
                            [
                                item
                                for item in unique_items[operator]
                                if item.state_hash in set(candidate_v1[operator])
                            ]
                        ),
                    }
                    for operator in operators
                    if unique_items[operator]
                },
            },
            "presampler_closure_width": {
                "policy": closure_width_policy_descriptor(),
                "policy_sha256": closure_width_policy_sha256(),
                "targets_sampled": {
                    target: target in candidate_hashes_v2 for target in targets
                },
                "coverage_keys_by_operator": {
                    operator: _coverage_key_counts(
                        unique_items[operator],
                        sampled_order[operator],
                        candidate_v1[operator],
                        candidate_v2[operator],
                    )
                    for operator in operators
                    if unique_items[operator]
                },
            },
            "sampled": len(sampled_hashes),
            "sampled_matches_shipped": sampled_hashes == shipped_hashes,
            "seed": seed,
            "targets": target_reports,
            "unique": int(sampled.unique),
            "unique_by_operator": dict(sampled.unique_by_operator),
            "unique_hashes_for_tracked_owners": {
                operator: [item.state_hash for item in unique_items[operator]]
                for operator in sorted({owner for owner in unique_owner.values()})
            },
        }
    return {
        "parent_state_hash": run.get("initial_state_hash"),
        "start": start_name,
        "strategies": reports,
    }


def _coverage_key_counts(
    items: Sequence[StructuralCandidate],
    baseline_hashes: Sequence[str],
    v1_hashes: Sequence[str],
    v2_hashes: Sequence[str],
) -> dict[str, Any]:
    def keys_for(selected: set[str]) -> list[list[object]]:
        seen: list[list[object]] = []
        for item in items:
            if item.state_hash not in selected:
                continue
            key = [closure_bin(item.actual_closure_size), [list(pair) for pair in item.width_histogram]]
            if key not in seen:
                seen.append(key)
        return seen

    unique_keys = keys_for({item.state_hash for item in items})
    return {
        "baseline_keys": keys_for(set(baseline_hashes)),
        "unique_keys": unique_keys,
        "unique_key_count": len(unique_keys),
        "v1_keys": keys_for(set(v1_hashes)),
        "v2_keys": keys_for(set(v2_hashes)),
    }


def compact_feature_replay(payload: Mapping[str, Any]) -> dict[str, Any]:
    targets = tuple(payload["targets"])
    seed_row: dict[str, Any] = {}
    for target in targets:
        base_any = False
        v1_any = False
        v2_any = False
        rows = []
        for start in payload.get("enumerations") or []:
            for strategy, report in start["strategies"].items():
                row = report["targets"][target]
                v1 = bool(report["presampler"]["targets_sampled"][target])
                v2 = bool(report["presampler_closure_width"]["targets_sampled"][target])
                if not (row["emitted"] or row["unique"] or row["sampled"] or v1 or v2):
                    continue
                base_any = base_any or bool(row["sampled"])
                v1_any = v1_any or v1
                v2_any = v2_any or v2
                rows.append(
                    {
                        "emitted": row["emitted"],
                        "sampled_baseline": row["sampled"],
                        "sampled_v1": v1,
                        "sampled_v2": v2,
                        "shuffle_index": row["shuffle_index"],
                        "start": start["start"],
                        "strategy": strategy,
                        "unique": row["unique"],
                        "unique_count": row["unique_count_for_owner"],
                        "unique_operator": row["unique_operator"],
                        "width_histogram": row.get("width_histogram"),
                    }
                )
        seed_row[target] = {
            "baseline_sampled": base_any,
            "enumeration": rows,
            "v1_sampled": v1_any,
            "v2_sampled": v2_any,
        }
    coverage = []
    for start in payload.get("enumerations") or []:
        for strategy, report in start["strategies"].items():
            for operator, counts in report["presampler_closure_width"]["coverage_keys_by_operator"].items():
                coverage.append(
                    {
                        "baseline_key_count": len(counts["baseline_keys"]),
                        "operator": operator,
                        "start": start["start"],
                        "strategy": strategy,
                        "unique_key_count": counts["unique_key_count"],
                        "v1_key_count": len(counts["v1_keys"]),
                        "v2_key_count": len(counts["v2_keys"]),
                    }
                )
    return {
        "kind": "lns_task4_closure_width_replay",
        "model_sha256": payload["model_sha256"],
        "prefix_stable": all(
            report["prefix_matches_full_shuffle"]
            for start in payload.get("enumerations") or []
            for report in start["strategies"].values()
        ),
        "proposal_seed": payload["proposal_seed"],
        "protocol": payload["protocol"],
        "starts_enumerated": payload.get("starts_enumerated"),
        "targets": seed_row,
        "v1_policy_sha256": policy_sha256(),
        "v2_policy": closure_width_policy_descriptor(),
        "v2_policy_sha256": closure_width_policy_sha256(),
        "coverage": coverage,
    }


def classify_loss(target: str, membership: Mapping[str, Any], enumerations: Sequence[Mapping[str, Any]]) -> str:
    pooled = membership.get("parent_pooled") or []
    if any(row.get("selected") for row in pooled) or any(row.get("selected") for row in membership["per_start"]):
        return "not_lost_selected"
    if any(row.get("audit") for row in pooled) or any(row.get("audit") for row in membership["per_start"]):
        return "parent_top32_not_hardware_top16"
    if membership["per_start"]:
        return "event_scored_outside_parent_top32"
    emitted = False
    unique = False
    sampled = False
    shuffle_indexes = []
    for start in enumerations:
        for strategy in start["strategies"].values():
            row = strategy["targets"][target]
            emitted = emitted or bool(row["emitted"])
            unique = unique or bool(row["unique"])
            sampled = sampled or bool(row["sampled"])
            if row["shuffle_index"] is not None:
                shuffle_indexes.append(row["shuffle_index"])
    if sampled:
        return "sampled_but_absent_from_candidate_features"
    if unique and shuffle_indexes and min(shuffle_indexes) >= 25:
        return "sampling_shuffle_truncate_n25"
    if unique:
        return "unique_not_sampled"
    if emitted:
        return "emitted_only_as_global_duplicate"
    if enumerations:
        return "outside_enumerated_parent_neighborhood"
    return "absent_from_candidate_features_enumeration_not_run"


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    targets = tuple(args.targets) if args.targets else tuple(MEASURED_FAST_PLANS)
    starts = tuple(args.starts) if args.starts else DEFAULT_STARTS
    print("load", model_path, flush=True)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    membership = {target: membership_for_target(model, target) for target in targets}
    enumerations: list[dict[str, Any]] = []
    if not args.skip_enumeration:
        analytic, interval, window_selector = window_selector_for_model(model, args.calibration)
        for start_name in starts:
            enumerations.append(
                enumerate_start(
                    model,
                    start_name,
                    args.proposal_seed,
                    targets,
                    analytic,
                    interval,
                    window_selector,
                )
            )
            print(json.dumps({"start": start_name, "parent": enumerations[-1]["parent_state_hash"]}, sort_keys=True), flush=True)
    loss_stages = {
        target: classify_loss(target, membership[target], enumerations) for target in targets
    }
    payload = {
        "kind": "lns_fast_plan_loss_audit",
        "loss_stages": loss_stages,
        "membership": membership,
        "model_sha256": sha256_file(model_path),
        "notes": [
            "Absence from candidate_features is not proof of pre-sampling absence.",
            "Window selector used IntervalPlanner._stage_window_policy().select, not (0,0).",
            "Shuffle index comes from shipped _sample_neighborhood with per_operator equal to the unique count.",
            "Enumeration is limited to the requested starts; other parents are membership-only.",
        ],
        "proposal_seed": args.proposal_seed,
        "protocol": {
            "neighbors_per_operator": model["method"].get("neighbors_per_operator"),
            "restarts_per_parent": model["method"].get("restarts_per_parent"),
            "seed": model["method"].get("seed"),
            "shortlist_budget": model["method"].get("shortlist_budget"),
            "templates_per_block": model["method"].get("templates_per_block"),
        },
        "set_definition": (
            "Plans with >=2% absolute-median gain versus reconstructed full in any "
            "frozen-protocol median hardware session, plus the preserved elite."
        ),
        "starts_enumerated": list(starts) if not args.skip_enumeration else [],
        "enumerations": enumerations,
        "targets": {target: MEASURED_FAST_PLANS.get(target, {"label": "cli"}) for target in targets},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.output, sha256_file(args.output), flush=True)
    if args.feature_replay_output is not None:
        replay = compact_feature_replay(payload)
        args.feature_replay_output.parent.mkdir(parents=True, exist_ok=True)
        args.feature_replay_output.write_text(
            json.dumps(replay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("wrote", args.feature_replay_output, sha256_file(args.feature_replay_output), flush=True)
    print(json.dumps(loss_stages, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
