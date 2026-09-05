#!/usr/bin/env python3
"""Expand preserved hardware incumbents with one offline template-LNS layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from executable_plan_neighborhood import TEMPLATE_LNS_SCOPES  # noqa: E402
from executable_plan_state import ExecutablePlanState  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402

from optimizations.fused_moe_sve.benchmarks.bench_executable_neighborhood_audit import (  # noqa: E402
    _build_initial_results,
    _full_strict_result,
    _load_pairwise_comparator,
    _parse_positive_int_list,
    _run_partial_order_vnd_start,
)
from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import (  # noqa: E402
    load_route_layer,
)
from optimizations.fused_moe_sve.benchmarks.bench_partial_order_beam_layer import (  # noqa: E402
    _state_from_payload,
)
from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import (  # noqa: E402
    LnsCandidateFeature,
    LnsDiverseShortlist,
    policy_sha256,
    select_lns_global_shortlist,
    select_lns_parent_pooled_shortlists,
    stratified_keys_outside_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parent_source = parser.add_mutually_exclusive_group(required=True)
    parent_source.add_argument("--parent-frontier", type=Path)
    parent_source.add_argument("--initial-controls", action="store_true")
    parser.add_argument("--parent-state-hash", action="append", default=[])
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--route-layer", type=int, required=True)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--pairwise-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--critical-experts", type=int, default=32)
    parser.add_argument("--neighbors-per-operator", type=int, default=50)
    parser.add_argument("--shortlist-budget", type=int, default=16)
    parser.add_argument("--audit-budget", type=int, default=32)
    parser.add_argument("--partial-order-shortlist-budget", type=int, default=16)
    parser.add_argument("--minimum-actionable-gain-pct", type=float, default=2.0)
    parser.add_argument("--lns-destroy-sizes", default="4,8,16")
    parser.add_argument("--lns-repair-beam-widths", default="16,32,64")
    parser.add_argument("--lns-templates-per-block", type=int, default=4)
    parser.add_argument("--restarts-per-parent", type=int, default=2)
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--pool-restarts-by-parent", action="store_true")
    parser.add_argument("--stratified-outside-audit", type=int, default=0)
    parser.add_argument("--stratified-seed", type=int, default=20261013)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_kb() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(round(usage / 1024.0))
    return int(usage)


def _sum_search_breakdowns(runs: dict[str, dict[str, object]]) -> dict[str, float | int] | None:
    merged: dict[str, float | int] = {}
    for run in runs.values():
        iterations = run.get("iterations") or []
        if not iterations:
            continue
        item = iterations[0].get("search_breakdown")
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            merged[key] = merged.get(key, 0) + value
    return merged or None


def _frontier_row_from_library(
    state_hash: str,
    *,
    start_name: str,
    state: ExecutablePlanState,
    strategy: str,
    row: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "state_hash": state_hash,
        "strategy": strategy,
        "operator": row["operator"],
        "moved_experts": row["moved_experts"],
        "shape": list(state.shape),
        "event_ns": row["event_ns"],
        "robust_ns": row["robust_ns"],
        "event_gain_pct": row["event_gain_pct"],
        "robust_gain_pct": row["robust_gain_pct"],
        "start": start_name,
        "canonical_state": state.canonical_payload(),
        "plan_v2_bridge": state.to_bridge(),
    }
    if "structure" in row:
        payload["structure"] = row["structure"]
    if "partial_order" in row:
        payload["partial_order"] = row["partial_order"]
    return payload


def _library_lookup(
    state_hash: str,
    start_names: Sequence[str],
    libraries: Mapping[str, Mapping[str, object]],
) -> tuple[str, ExecutablePlanState, str, dict[str, object]]:
    for start_name in start_names:
        rows = libraries[start_name]["rows"]
        states = libraries[start_name]["states"]
        if state_hash in rows and state_hash in states:
            strategy, row = rows[state_hash]
            return start_name, states[state_hash], strategy, row
    raise KeyError(f"pooled candidate is missing from restart libraries: {state_hash}")


def _load_parent_states(
    frontier: dict[str, object],
    state_hashes: list[str],
) -> dict[str, ExecutablePlanState]:
    if frontier.get("kind") != "partial_order_hardware_frontier":
        raise ValueError("parent frontier has an unexpected kind")
    if not state_hashes or len(set(state_hashes)) != len(state_hashes):
        raise ValueError("parent state hashes must be non-empty and unique")
    plans = frontier.get("plans")
    if not isinstance(plans, dict):
        raise ValueError("parent frontier has no plan mapping")
    states = {}
    for state_hash in state_hashes:
        plan = plans.get(state_hash)
        if not isinstance(plan, dict):
            raise ValueError(f"parent state is absent from frontier: {state_hash}")
        canonical_state = plan.get("canonical_state")
        if not isinstance(canonical_state, dict):
            raise ValueError(f"parent state has no canonical payload: {state_hash}")
        state = _state_from_payload(canonical_state)
        if state.canonical_hash() != state_hash or state.to_bridge() != plan.get("plan_v2_bridge"):
            raise ValueError(f"parent state/bridge failed reconstruction: {state_hash}")
        states[state_hash] = state
    return states


def _deduplicate_named_states(
    named_states: dict[str, ExecutablePlanState],
) -> tuple[dict[str, ExecutablePlanState], dict[str, list[str]]]:
    states = {}
    roles: dict[str, list[str]] = {}
    for name, state in named_states.items():
        state_hash = state.canonical_hash()
        existing = states.setdefault(state_hash, state)
        if existing.canonical_payload() != state.canonical_payload():
            raise RuntimeError(f"canonical state hash collision for control {name}")
        roles.setdefault(state_hash, []).append(name)
    return states, roles


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    destroy_sizes = _parse_positive_int_list(args.lns_destroy_sizes, "lns-destroy-sizes")
    repair_beam_widths = _parse_positive_int_list(
        args.lns_repair_beam_widths,
        "lns-repair-beam-widths",
    )
    if len(destroy_sizes) != len(repair_beam_widths):
        raise ValueError("lns destroy sizes and repair beam widths must have the same length")
    if len(set(destroy_sizes)) != len(destroy_sizes):
        raise ValueError("lns destroy sizes must be unique")
    if (
        min(
            args.hidden,
            args.intermediate,
            args.experts,
            args.threads,
            args.critical_experts,
            args.neighbors_per_operator,
            args.shortlist_budget,
            args.audit_budget,
            args.partial_order_shortlist_budget,
            args.lns_templates_per_block,
            args.restarts_per_parent,
        )
        <= 0
    ):
        raise ValueError("dimensions and search budgets must be positive")
    if not 0.0 <= args.minimum_actionable_gain_pct < 100.0:
        raise ValueError("minimum-actionable-gain-pct must be in [0, 100)")
    if args.audit_budget < args.shortlist_budget:
        raise ValueError("audit-budget must be at least shortlist-budget")

    actual_extension = _sha256(Path(_moe_C.__file__))
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    torch.set_num_threads(1)
    topk_ids, route_metadata = load_route_layer(args.route_file, args.route_layer, args.experts)
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=args.experts,
        local_experts=args.experts,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    planned = PlannedMoE(
        model,
        num_cores=args.threads,
        cpu_ids=cpu_ids,
        search_mode="full",
        cache_plans=False,
    )
    calibration_sha256 = _sha256(args.analytic_calibration)
    pairwise_sha256 = _sha256(args.pairwise_calibration)
    parent_roles: dict[str, list[str]]
    parent_source: dict[str, object]
    if args.parent_frontier is not None:
        if not args.parent_state_hash:
            raise ValueError("--parent-frontier requires at least one --parent-state-hash")
        frontier = json.loads(args.parent_frontier.read_text(encoding="utf-8"))
        parent_states = _load_parent_states(frontier, args.parent_state_hash)
        if str(frontier["identity"]["extension_sha256"]) != actual_extension:
            raise ValueError("parent frontier extension identity does not match the loaded extension")
        if str(frontier["identity"]["calibration_sha256"]) != calibration_sha256:
            raise ValueError("parent frontier calibration identity does not match --analytic-calibration")
        if str(frontier["identity"]["pairwise_calibration_sha256"]) != pairwise_sha256:
            raise ValueError("parent frontier pairwise identity does not match --pairwise-calibration")
        for field in ("hidden", "intermediate", "experts", "threads"):
            if int(frontier["shape"][field]) != int(getattr(args, field)):
                raise ValueError(f"parent frontier shape does not match --{field}")
        if route_metadata["sha256"] != frontier["route"]["sha256"]:
            raise ValueError("parent frontier route does not match --route-file")
        shape = frontier["shape"]
        parent_roles = {
            state_hash: list(frontier["records"][state_hash]["roles"])
            for state_hash in parent_states
        }
        parent_source = {
            "kind": "hardware_frontier",
            "path": str(args.parent_frontier),
            "sha256": _sha256(args.parent_frontier),
        }
    else:
        if args.parent_state_hash:
            raise ValueError("--parent-state-hash cannot be used with --initial-controls")
        route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
        counts = [
            (expert, int(routes))
            for expert, routes in enumerate(route_counts.tolist())
            if routes
        ]
        full_result, strict_candidates, full_plan_ms = _full_strict_result(
            planned,
            counts,
            topk_ids,
        )
        initial_results, initial_metadata = _build_initial_results(
            planned.interval_planners[0],
            counts,
            topk_ids,
            full_result,
            strict_candidates,
        )
        llc_domains = tuple(
            (domain.domain_id, domain.cpu_ids) for domain in model.calibration.llc_domains
        )
        named_states = {
            name: ExecutablePlanState.from_planner_result(
                result,
                llc_domains=llc_domains,
            )
            for name, result in initial_results.items()
        }
        parent_states, parent_roles = _deduplicate_named_states(named_states)
        shape = {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "tokens": int(topk_ids.shape[0]),
            "top_k": int(topk_ids.shape[1]),
            "threads": args.threads,
            "backend_n_tile": int(model.policy.backend_n_tile),
        }
        parent_source = {
            "kind": "reconstructed_initial_controls",
            "full_plan_wall_ms": full_plan_ms,
            "metadata": initial_metadata,
            "named_state_hashes": {
                name: state.canonical_hash() for name, state in named_states.items()
            },
        }
    comparator = _load_pairwise_comparator(
        args.pairwise_calibration,
        args.minimum_actionable_gain_pct,
        args.partial_order_shortlist_budget,
        calibration_sha256=calibration_sha256,
        extension_sha256=actual_extension,
    )
    search_args = SimpleNamespace(
        partial_order_max_iterations=1,
        critical_experts=args.critical_experts,
        seed=args.seed,
        neighborhood_mode="lns",
        neighbors_per_operator=args.neighbors_per_operator,
        event_top_per_strategy=4,
        shortlist_budget=args.shortlist_budget,
        audit_budget=args.audit_budget,
        partial_order_shortlist_budget=args.partial_order_shortlist_budget,
        partial_order_decision_mode="diagnostic_only",
        lns_destroy_sizes=destroy_sizes,
        lns_repair_beam_widths=repair_beam_widths,
        lns_templates_per_block=args.lns_templates_per_block,
    )
    runs = {}
    parents = {}
    libraries: dict[str, dict[str, object]] = {}
    parent_items = list(parent_states.items())
    if args.max_parents:
        if args.max_parents < 0:
            raise ValueError("max-parents must be non-negative")
        parent_items = parent_items[: args.max_parents]
    if args.stratified_outside_audit < 0:
        raise ValueError("stratified-outside-audit must be non-negative")
    for parent_index, (state_hash, state) in enumerate(parent_items):
        for restart in range(args.restarts_per_parent):
            start_name = f"lns_{parent_index:02d}_r{restart:02d}"
            run = _run_partial_order_vnd_start(
                search_args,
                start_name=start_name,
                initial_state=state,
                interval=planned.interval_planners[0],
                proposal_model=model,
                score_model=model,
                comparator=comparator,
                screen_budgets=(args.shortlist_budget,),
            )
            libraries[start_name] = {
                "states": run.pop("_candidate_states"),
                "rows": run.pop("_candidate_rows"),
            }
            runs[start_name] = run
            parents[start_name] = {
                "state_hash": state_hash,
                "parent_index": parent_index,
                "restart": restart,
                "roles": parent_roles[state_hash],
            }

    operator_count = len(TEMPLATE_LNS_SCOPES) * len(destroy_sizes)
    maximum_exact_candidates = (
        len(runs) * 2 * operator_count * args.neighbors_per_operator
    )
    per_start_shortlists = {
        start_name: LnsDiverseShortlist.from_dict(run["iterations"][0]["lns_diverse_shortlist"])
        for start_name, run in runs.items()
        if run.get("iterations") and run["iterations"][0].get("lns_diverse_shortlist")
    }
    parent_of_start = {start_name: item["state_hash"] for start_name, item in parents.items()}
    pooled_shortlists: dict[str, LnsDiverseShortlist] = {}
    stratified_by_parent: dict[str, tuple[str, ...]] = {}
    pooled_frontier_rows: dict[str, dict[str, object]] = {}
    if args.pool_restarts_by_parent and per_start_shortlists:
        per_start_features = {
            start_name: [
                LnsCandidateFeature.from_dict(row)
                for row in run["iterations"][0].get("candidate_features") or []
            ]
            for start_name, run in runs.items()
            if run.get("iterations")
        }
        pooled_shortlists = select_lns_parent_pooled_shortlists(
            per_start_features,
            parent_of_start,
            shortlist_budget=args.shortlist_budget,
            audit_budget=args.audit_budget,
        )
        starts_by_parent: dict[str, list[str]] = {}
        for start_name, parent_hash in parent_of_start.items():
            starts_by_parent.setdefault(parent_hash, []).append(start_name)
        all_features = [feature for rows in per_start_features.values() for feature in rows]
        audit_union = {key for shortlist in pooled_shortlists.values() for key in shortlist.audit_keys}
        ranked_outside: list[str] = []
        seen_outside: set[str] = set()
        for shortlist in pooled_shortlists.values():
            for key in shortlist.ranked_keys:
                if key in audit_union or key in seen_outside:
                    continue
                ranked_outside.append(key)
                seen_outside.add(key)
        global_stratified = stratified_keys_outside_audit(
            all_features,
            ranked_outside,
            (),
            sample_size=args.stratified_outside_audit,
            seed=args.stratified_seed,
        )
        for parent_hash in pooled_shortlists:
            stratified_by_parent[parent_hash] = tuple(
                key
                for key in global_stratified
                if any(key in libraries[start]["rows"] for start in starts_by_parent[parent_hash])
            )
        needed = [
            *{key for shortlist in pooled_shortlists.values() for key in shortlist.audit_keys},
            *global_stratified,
        ]
        for state_hash in needed:
            start_name, state, strategy, row = _library_lookup(
                state_hash,
                sorted(libraries),
                libraries,
            )
            pooled_frontier_rows[state_hash] = _frontier_row_from_library(
                state_hash,
                start_name=start_name,
                state=state,
                strategy=strategy,
                row=row,
            )
    global_shortlist_begin = time.perf_counter_ns()
    shortlist_source = pooled_shortlists if pooled_shortlists else per_start_shortlists
    global_shortlist = (
        select_lns_global_shortlist(
            shortlist_source,
            shortlist_budget=args.shortlist_budget,
            audit_budget=args.audit_budget,
        )
        if shortlist_source
        else None
    )
    global_shortlist_s = (time.perf_counter_ns() - global_shortlist_begin) / 1.0e9
    measured_lns_keys = 0
    if pooled_shortlists:
        measured_lns_keys = len({*global_shortlist["selected_keys"], *sum(stratified_by_parent.values(), ())})
    else:
        measured_lns_keys = len(runs) * args.audit_budget
    maximum_hardware_frontier_plans = len(parent_items) + measured_lns_keys
    result = {
        "kind": "executable_partial_order_template_lns_model_replay",
        "route": route_metadata,
        "shape": shape,
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": calibration_sha256,
            "pairwise_calibration": str(args.pairwise_calibration),
            "pairwise_calibration_sha256": pairwise_sha256,
            "extension": str(_moe_C.__file__),
            "extension_sha256": actual_extension,
            "parent_source": parent_source,
        },
        "method": {
            "start_names": list(runs),
            "parent_state_hashes": list(parent_states),
            "restarts_per_parent": args.restarts_per_parent,
            "neighborhood_mode": "lns",
            "critical_experts": args.critical_experts,
            "destroy_sizes": list(destroy_sizes),
            "repair_beam_widths": list(repair_beam_widths),
            "templates_per_block": args.lns_templates_per_block,
            "scopes": list(TEMPLATE_LNS_SCOPES),
            "neighbors_per_operator": args.neighbors_per_operator,
            "shortlist_budget": args.shortlist_budget,
            "audit_budget": args.audit_budget,
            "partial_order_shortlist_budget": args.partial_order_shortlist_budget,
            "minimum_actionable_gain_pct": args.minimum_actionable_gain_pct,
            "partial_order_decision_mode": "diagnostic_only",
            "lns_shortlist_policy": "relation_agnostic_categorical_farthest_first_v1",
            "lns_shortlist_policy_sha256": policy_sha256(),
            "maximum_exact_candidate_budget": maximum_exact_candidates,
            "maximum_hardware_frontier_plan_budget": maximum_hardware_frontier_plans,
            "pool_restarts_by_parent": bool(args.pool_restarts_by_parent),
            "stratified_outside_audit": int(args.stratified_outside_audit),
            "stratified_seed": int(args.stratified_seed),
            "seed": args.seed,
        },
        "parents": parents,
        "runs": runs,
        "parent_pooled_shortlists": {
            parent_hash: {
                "starts": sorted(start for start, item in parents.items() if item["state_hash"] == parent_hash),
                "shortlist": shortlist.to_dict(),
                "stratified_keys": list(stratified_by_parent.get(parent_hash, ())),
            }
            for parent_hash, shortlist in pooled_shortlists.items()
        }
        or None,
        "pooled_frontier_rows": pooled_frontier_rows or None,
        "lns_diverse_shortlist": global_shortlist,
        "summary": {
            "parents": len(parent_states),
            "starts": len(runs),
            "unique_candidates": sum(run["iterations"][0]["unique_candidates"] for run in runs.values()),
            "better": sum(
                run["iterations"][0]["partial_order"]["relation_counts"]["candidate_better"]
                for run in runs.values()
            ),
            "worse": sum(
                run["iterations"][0]["partial_order"]["relation_counts"]["candidate_worse"]
                for run in runs.values()
            ),
            "incomparable": sum(
                run["iterations"][0]["partial_order"]["relation_counts"]["incomparable"]
                for run in runs.values()
            ),
            "event_calls": sum(run["exact_event_calls"] for run in runs.values()),
            "search_wall_s": sum(run["search_wall_s"] for run in runs.values()),
            "search_breakdown": _sum_search_breakdowns(runs),
            "global_shortlist_s": global_shortlist_s,
            "ru_maxrss_kb": _peak_rss_kb(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
