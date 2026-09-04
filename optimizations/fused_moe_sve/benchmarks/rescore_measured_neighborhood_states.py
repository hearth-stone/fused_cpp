#!/usr/bin/env python3
"""Regenerate a frozen exhaustive neighborhood and rescore only measured state hashes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from executable_plan_neighborhood import (  # noqa: E402
    ExecutablePlanEvaluator,
    sample_combined_neighborhood,
)
from executable_plan_state import ExecutablePlanState  # noqa: E402
from optimizations.fused_moe_sve.benchmarks.bench_executable_neighborhood_audit import (  # noqa: E402
    _full_strict_result,
)
from optimizations.fused_moe_sve.benchmarks.bench_high_skew_planner_closure import (  # noqa: E402
    load_route_layer,
)
from planned_moe import PlannedMoE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-file", type=Path, required=True)
    parser.add_argument("--route-layer", type=int, required=True)
    parser.add_argument("--proposal-calibration", type=Path, required=True)
    parser.add_argument("--score-calibration", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _model(path: Path, experts: int) -> AnalyticMoeCostModel:
    return AnalyticMoeCostModel(
        path,
        hidden_size=4096,
        intermediate_size=512,
        global_experts=experts,
        local_experts=experts,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    torch.set_num_threads(1)
    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    target_hashes = {
        str(item["state_hash"])
        for item in measurement["hardware_shortlist"]["measurement"]["candidates"]
    }
    baseline_hash = str(measurement["baseline"]["state_hash"])
    proposal_model = _model(args.proposal_calibration, args.experts)
    score_model = _model(args.score_calibration, args.experts)
    topk_ids, _ = load_route_layer(args.route_file, args.route_layer, args.experts)
    route_counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=args.experts)
    counts = [(expert, int(routes)) for expert, routes in enumerate(route_counts.tolist()) if routes]
    planned = PlannedMoE(
        proposal_model,
        num_cores=args.threads,
        cpu_ids=cpu_ids,
        search_mode="full",
        cache_plans=False,
    )
    interval = planned.interval_planners[0]
    full_result, _ = _full_strict_result(planned, counts, topk_ids)
    llc_domains = tuple(
        (domain.domain_id, domain.cpu_ids)
        for domain in proposal_model.calibration.llc_domains
    )
    baseline = ExecutablePlanState.from_planner_result(full_result, llc_domains=llc_domains)
    if baseline.canonical_hash() != baseline_hash:
        raise RuntimeError(
            f"regenerated baseline hash differs: {baseline.canonical_hash()} != {baseline_hash}"
        )
    expert_ids = sorted(
        {
            int(expert)
            for strategy in measurement["strategies"].values()
            for expert in strategy["selected_expert_ids"]
        }
    )
    policy = interval._stage_window_policy()

    def window_selector(routes: int, threads: int) -> tuple[int, int]:
        return (0, 0) if policy is None else policy.select(routes, threads)

    sampled = sample_combined_neighborhood(
        baseline,
        allowed_widths=interval.widths,
        isolated_cost=proposal_model.T_iso,
        window_selector=window_selector,
        expert_filter=expert_ids,
        per_operator=10_000,
        seed=args.seed,
    )
    state_by_hash = {
        neighbor.state.canonical_hash(): neighbor.state
        for neighbor in sampled.neighbors
        if neighbor.state.canonical_hash() in target_hashes
    }
    scorer = ExecutablePlanEvaluator(score_model)
    rows = [
        {
            "state_hash": state_hash,
            "event_ns": scorer.exact(state).event_ns,
        }
        for state_hash, state in sorted(state_by_hash.items())
    ]
    report = {
        "kind": "moe_measured_state_rescore",
        "proposal_calibration": str(args.proposal_calibration),
        "score_calibration": str(args.score_calibration),
        "measurement": str(args.measurement),
        "baseline": {
            "state_hash": baseline_hash,
            "event_ms": scorer.exact(baseline).event_ns / 1.0e6,
        },
        "strategies": {"replay": {"scored": rows}},
        "enumeration": {
            "expert_ids": expert_ids,
            "proposed": sampled.proposed,
            "unique": sampled.unique,
            "target_hashes": len(target_hashes),
            "matched_hashes": len(state_by_hash),
            "missing_hashes": sorted(target_hashes - state_by_hash.keys()),
            "proposed_by_operator": dict(sampled.proposed_by_operator),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
