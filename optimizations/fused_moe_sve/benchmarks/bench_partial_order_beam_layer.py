#!/usr/bin/env python3
"""Expand independently validated hardware winners into one offline beam layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)
from fused_cpp import _moe_C  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402

from optimizations.fused_moe_sve.benchmarks.bench_executable_neighborhood_audit import (  # noqa: E402
    _load_pairwise_comparator,
    _run_partial_order_vnd_start,
)
from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import (  # noqa: E402
    load_route_layer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-frontier", type=Path, required=True)
    parser.add_argument("--parent-analysis", type=Path, required=True)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--route-layer", type=int, required=True)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--pairwise-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--critical-experts", type=int, default=32)
    parser.add_argument("--neighbors-per-operator", type=int, default=64)
    parser.add_argument("--partial-order-shortlist-budget", type=int, default=16)
    parser.add_argument("--minimum-actionable-gain-pct", type=float, default=2.0)
    parser.add_argument(
        "--parent-policy",
        choices=("stable_plus_global", "global_elite"),
        default="stable_plus_global",
    )
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--beam-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_from_payload(payload: dict[str, object]) -> ExecutablePlanState:
    state = ExecutablePlanState(
        num_threads=int(payload["num_threads"]),
        thread_cpu_ids=tuple(int(cpu) for cpu in payload["thread_cpu_ids"]),
        llc_domains=tuple(
            ExecutableLlcDomain(
                str(domain["id"]),
                int(domain["core_begin"]),
                int(domain["core_count"]),
            )
            for domain in payload["llc_domains"]
        ),
        lanes=tuple(
            ExecutableLane(
                int(lane["core_begin"]),
                int(lane["threads"]),
                tuple(
                    ExecutableExpertTask(
                        int(task["expert_id"]),
                        int(task["routes"]),
                        int(task["w13_window_tiles"]),
                        int(task["w2_window_tiles"]),
                    )
                    for task in lane["tasks"]
                ),
            )
            for lane in payload["lanes"]
        ),
        early_merge=payload.get("early_merge"),
    )
    return state


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if min(
        args.hidden,
        args.intermediate,
        args.experts,
        args.threads,
        args.critical_experts,
        args.neighbors_per_operator,
        args.partial_order_shortlist_budget,
        args.beam_width,
        args.beam_depth,
    ) <= 0:
        raise ValueError("dimensions and search budgets must be positive")
    frontier = json.loads(args.parent_frontier.read_text(encoding="utf-8"))
    analysis = json.loads(args.parent_analysis.read_text(encoding="utf-8"))
    if frontier.get("kind") != "partial_order_hardware_frontier":
        raise ValueError("parent frontier has an unexpected kind")
    if analysis.get("decision") != "use_hardware_assisted_beam_search":
        raise ValueError("parent analysis did not approve hardware-assisted beam search")
    if analysis["identity"]["frontier_sha256"] != _sha256(args.parent_frontier):
        raise ValueError("parent analysis does not identify the supplied frontier")
    winners = list(analysis.get("stable_frontier_candidates", []))
    if not winners:
        raise ValueError("parent analysis contains no stable frontier winner")
    global_best_hash = str(analysis.get("global_measured_best_state_hash", ""))
    if not global_best_hash:
        raise ValueError("parent analysis has no global measured-best state")
    if args.parent_policy == "global_elite":
        elite_hashes = analysis.get("global_measured_elite_state_hashes", [])
        if len(elite_hashes) < args.beam_width:
            raise ValueError("parent analysis does not contain enough global elites")
        parent_candidates = [
            {
                "candidate_state_hash": str(state_hash),
                "role": "global_measured_elite",
            }
            for state_hash in elite_hashes[: args.beam_width]
        ]
    else:
        parent_candidates = list(winners)
        if global_best_hash not in {str(item["candidate_state_hash"]) for item in parent_candidates}:
            parent_candidates.append(
                {
                    "candidate_state_hash": global_best_hash,
                    "role": "global_measured_best_incumbent",
                }
            )

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    torch.set_num_threads(1)
    topk_ids, route_metadata = load_route_layer(args.route_file, args.route_layer, args.experts)
    if route_metadata["sha256"] != frontier["route"]["sha256"]:
        raise ValueError("parent frontier route does not match --route-file")
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
    comparator = _load_pairwise_comparator(
        args.pairwise_calibration,
        args.minimum_actionable_gain_pct,
        args.partial_order_shortlist_budget,
        calibration_sha256=_sha256(args.analytic_calibration),
        extension_sha256=_sha256(Path(_moe_C.__file__)),
    )
    search_args = SimpleNamespace(
        partial_order_max_iterations=1,
        critical_experts=args.critical_experts,
        seed=args.seed,
        neighborhood_mode="combined",
        neighbors_per_operator=args.neighbors_per_operator,
        event_top_per_strategy=4,
        partial_order_shortlist_budget=args.partial_order_shortlist_budget,
    )
    runs = {}
    parent_rows = {}
    for index, winner in enumerate(parent_candidates):
        state_hash = str(winner["candidate_state_hash"])
        plan = frontier["plans"].get(state_hash)
        if plan is None:
            raise ValueError(f"winner state is absent from parent frontier: {state_hash}")
        state = _state_from_payload(plan["canonical_state"])
        if state.canonical_hash() != state_hash or state.to_bridge() != plan["plan_v2_bridge"]:
            raise ValueError(f"winner state/bridge failed reconstruction: {state_hash}")
        start_name = f"beam_{index:02d}"
        runs[start_name] = _run_partial_order_vnd_start(
            search_args,
            start_name=start_name,
            initial_state=state,
            interval=planned.interval_planners[0],
            proposal_model=model,
            score_model=model,
            comparator=comparator,
            screen_budgets=(args.partial_order_shortlist_budget,),
        )
        parent_rows[start_name] = winner

    result = {
        "kind": "executable_partial_order_beam_layer_model_replay",
        "route": route_metadata,
        "shape": frontier["shape"],
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "pairwise_calibration": str(args.pairwise_calibration),
            "pairwise_calibration_sha256": _sha256(args.pairwise_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
            "parent_frontier": str(args.parent_frontier),
            "parent_frontier_sha256": _sha256(args.parent_frontier),
            "parent_analysis": str(args.parent_analysis),
            "parent_analysis_sha256": _sha256(args.parent_analysis),
        },
        "method": {
            "start_names": list(runs),
            "beam_depth": args.beam_depth,
            "parent_policy": args.parent_policy,
            "beam_width": len(runs),
            "neighborhood_mode": "combined",
            "critical_experts": args.critical_experts,
            "neighbors_per_operator": args.neighbors_per_operator,
            "partial_order_shortlist_budget": args.partial_order_shortlist_budget,
            "minimum_actionable_gain_pct": args.minimum_actionable_gain_pct,
            "seed": args.seed,
        },
        "parents": parent_rows,
        "runs": runs,
        "summary": {
            "parents": len(runs),
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
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
