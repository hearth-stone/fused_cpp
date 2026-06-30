#!/usr/bin/env python3
"""Run a synthetic workload sweep for CPU MoE scheduling plans."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PLANNERS_DIR = ROOT / "planners"
sys.path.insert(0, str(PLANNERS_DIR))

from offline_simulator import (  # noqa: E402
    ExpertCostModel,
    Plan,
    PlanKind,
    active_experts_from_routes,
    build_planner_cost_model,
    build_plans,
    format_ns,
    generate_routes,
    select_best_plan,
    select_auto_planners,
    workload_stats,
)


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    distribution: str
    params: Dict[str, object] = field(default_factory=dict)
    num_experts: Optional[int] = None
    tokens: Optional[int] = None
    top_k: Optional[int] = None
    cores: Optional[int] = None


def default_cases(case_set: str) -> List[SyntheticCase]:
    smoke = [
        SyntheticCase("uniform", "uniform"),
        SyntheticCase(
            "dsv4_sparse_topk",
            "active_subset",
            {"active_experts": 6},
            tokens=2100,
            top_k=6,
        ),
        SyntheticCase(
            "dsv4_sparse_topk_8c",
            "active_subset",
            {"active_experts": 6},
            tokens=2100,
            top_k=6,
            cores=8,
        ),
        SyntheticCase(
            "dsv4_broad_heavytail",
            "lognormal",
            {"lognormal_sigma": 2.0, "seed": 2},
            tokens=2048,
            top_k=6,
        ),
        SyntheticCase(
            "dsv4_broad_heavytail_8c",
            "lognormal",
            {"lognormal_sigma": 2.0, "seed": 2},
            tokens=2048,
            top_k=6,
            cores=8,
        ),
        SyntheticCase("active_subset_8", "active_subset", {"active_experts": 8}),
        SyntheticCase("hotspot_4x75", "hotspot", {"hot_experts": 4, "hot_fraction": 0.75}),
        SyntheticCase("zipf_1p15", "zipf", {"zipf_alpha": 1.15}),
        SyntheticCase("dirichlet_0p15", "dirichlet", {"dirichlet_alpha": 0.15, "seed": 1}),
        SyntheticCase(
            "domain_cluster_32_bg10",
            "domain_cluster",
            {"cluster_experts": 32, "background_fraction": 0.10, "seed": 3},
        ),
        SyntheticCase("one_hot", "one_hot"),
    ]
    if case_set == "smoke":
        return smoke

    return smoke + [
        SyntheticCase("random_balanced", "random_balanced", {"seed": 0}),
        SyntheticCase("active_subset_32", "active_subset", {"active_experts": 32}),
        SyntheticCase("active_subset_64", "active_subset", {"active_experts": 64}),
        SyntheticCase("hotspot_2x80", "hotspot", {"hot_experts": 2, "hot_fraction": 0.80}),
        SyntheticCase("hotspot_8x60", "hotspot", {"hot_experts": 8, "hot_fraction": 0.60}),
        SyntheticCase(
            "heavy_light_8x80",
            "heavy_light",
            {"heavy_experts": 8, "heavy_fraction": 0.80},
        ),
        SyntheticCase(
            "heavy_light_16x85",
            "heavy_light",
            {"heavy_experts": 16, "heavy_fraction": 0.85},
        ),
        SyntheticCase("zipf_0p80", "zipf", {"zipf_alpha": 0.80}),
        SyntheticCase("zipf_1p50", "zipf", {"zipf_alpha": 1.50}),
        SyntheticCase("dirichlet_0p50", "dirichlet", {"dirichlet_alpha": 0.50, "seed": 2}),
        SyntheticCase("dirichlet_1p00", "dirichlet", {"dirichlet_alpha": 1.00, "seed": 3}),
        SyntheticCase("lognormal_1p0", "lognormal", {"lognormal_sigma": 1.0, "seed": 1}),
        SyntheticCase("lognormal_2p0", "lognormal", {"lognormal_sigma": 2.0, "seed": 2}),
        SyntheticCase(
            "domain_cluster_16_bg05",
            "domain_cluster",
            {"cluster_experts": 16, "background_fraction": 0.05, "seed": 4},
        ),
    ]


def case_routes(
    case: SyntheticCase,
    num_experts: int,
    tokens: int,
    top_k: int,
) -> List[int]:
    case_num_experts = case.num_experts or num_experts
    case_tokens = case.tokens or tokens
    case_top_k = case.top_k or top_k
    defaults: Dict[str, object] = {
        "seed": 0,
        "zipf_alpha": 1.15,
        "active_experts": 32,
        "hot_experts": 4,
        "hot_fraction": 0.70,
        "heavy_experts": 8,
        "heavy_fraction": 0.80,
        "dirichlet_alpha": 0.25,
        "lognormal_sigma": 1.50,
        "cluster_experts": 32,
        "background_fraction": 0.10,
    }
    defaults.update(case.params)
    return generate_routes(
        distribution=case.distribution,
        num_experts=case_num_experts,
        tokens=case_tokens,
        top_k=case_top_k,
        seed=int(defaults["seed"]),
        zipf_alpha=float(defaults["zipf_alpha"]),
        active_experts=int(defaults["active_experts"]),
        hot_experts=int(defaults["hot_experts"]),
        hot_fraction=float(defaults["hot_fraction"]),
        heavy_experts=int(defaults["heavy_experts"]),
        heavy_fraction=float(defaults["heavy_fraction"]),
        dirichlet_alpha=float(defaults["dirichlet_alpha"]),
        lognormal_sigma=float(defaults["lognormal_sigma"]),
        cluster_experts=int(defaults["cluster_experts"]),
        background_fraction=float(defaults["background_fraction"]),
    )


def plan_by_kind(plans: Sequence[Plan], kind: PlanKind) -> Optional[Plan]:
    for plan in plans:
        if plan.kind == kind:
            return plan
    return None


def run_case(
    case: SyntheticCase,
    num_experts: int,
    tokens: int,
    top_k: int,
    cores: int,
    cost_model: ExpertCostModel,
    planner_cost_model,
) -> Dict[str, object]:
    case_num_experts = case.num_experts or num_experts
    case_tokens = case.tokens or tokens
    case_top_k = case.top_k or top_k
    case_cores = case.cores or cores
    routes = case_routes(case, num_experts, tokens, top_k)
    active = active_experts_from_routes(routes)
    plans = build_plans(active, case_cores, cost_model, planner_cost_model, ["all"])
    best_total = select_best_plan(plans)
    best_execute = min(
        plans,
        key=lambda plan: (plan.estimated_execute_cost_ns, plan.kind.value),
    )
    fixed = plan_by_kind(plans, PlanKind.FIXED_GLOBAL_THREADS)
    if fixed is None:
        raise RuntimeError("fixed plan missing from planner set")
    auto_planner_names, auto_reason, auto_stats = select_auto_planners(
        routes,
        case_cores,
    )
    auto_plans = build_plans(
        active,
        case_cores,
        cost_model,
        planner_cost_model,
        auto_planner_names,
    )
    auto_best = select_best_plan(auto_plans)

    stats = workload_stats(routes)
    total_speedup = fixed.estimated_total_cost_ns / best_total.estimated_total_cost_ns
    exec_speedup = fixed.estimated_execute_cost_ns / best_execute.estimated_execute_cost_ns
    auto_regret = auto_best.estimated_total_cost_ns / best_total.estimated_total_cost_ns

    return {
        "case": case.name,
        "distribution": case.distribution,
        "num_experts": case_num_experts,
        "tokens": case_tokens,
        "top_k": case_top_k,
        "cores": case_cores,
        "params": case.params,
        "stats": stats,
        "best_total_kind": best_total.kind.value,
        "best_execute_kind": best_execute.kind.value,
        "auto_best_kind": auto_best.kind.value,
        "auto_total_ns": auto_best.estimated_total_cost_ns,
        "auto_regret_vs_best": auto_regret,
        "auto_selector": {
            "enabled_planners": auto_planner_names,
            "reason": auto_reason,
            "stats": auto_stats,
        },
        "best_total_ns": best_total.estimated_total_cost_ns,
        "best_execute_ns": best_execute.estimated_execute_cost_ns,
        "fixed_total_ns": fixed.estimated_total_cost_ns,
        "fixed_execute_ns": fixed.estimated_execute_cost_ns,
        "total_speedup_vs_fixed": total_speedup,
        "execute_speedup_vs_fixed": exec_speedup,
        "plans": [plan.to_dict() for plan in plans],
        "auto_plans": [plan.to_dict() for plan in auto_plans],
    }


def print_summary(rows: Sequence[Dict[str, object]]) -> None:
    header = (
        f"{'case':<27} {'tok':>5} {'core':>4} {'act':>4} {'gini':>5} {'maxvio':>7} "
        f"{'best_total':<24} {'total':>10} {'vs_fixed':>8} "
        f"{'auto':<24} {'regret':>7} "
        f"{'exec_best':<24} {'exec':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        stats = row["stats"]
        assert isinstance(stats, dict)
        print(
            f"{str(row['case']):<27} "
            f"{int(row['tokens']):>5} "
            f"{int(row['cores']):>4} "
            f"{int(stats['active_experts']):>4} "
            f"{float(stats['gini']):>5.3f} "
            f"{float(stats['maxvio']):>7.2f} "
            f"{str(row['best_total_kind']):<24} "
            f"{format_ns(int(row['best_total_ns'])):>10} "
            f"{float(row['total_speedup_vs_fixed']):>8.2f} "
            f"{str(row['auto_best_kind']):<24} "
            f"{float(row['auto_regret_vs_best']):>7.3f} "
            f"{str(row['best_execute_kind']):<24} "
            f"{format_ns(int(row['best_execute_ns'])):>10}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic benchmark sweep for CPU MoE schedule plans.",
    )
    parser.add_argument("--case-set", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument(
        "--plan-cost-source",
        choices=["complexity", "model", "measured"],
        default="complexity",
    )
    parser.add_argument(
        "--planner-cost",
        action="append",
        default=None,
        help="Override planner cost, for example fixed=1us or greedy=20us.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print full JSON with per-plan details.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.cores <= 0:
        raise ValueError("--cores must be positive")

    cost_model = ExpertCostModel.synthetic()
    planner_cost_model = build_planner_cost_model(
        source=args.plan_cost_source,
        overrides=args.planner_cost,
    )
    rows = [
        run_case(
            case=case,
            num_experts=args.num_experts,
            tokens=args.tokens,
            top_k=args.top_k,
            cores=args.cores,
            cost_model=cost_model,
            planner_cost_model=planner_cost_model,
        )
        for case in default_cases(args.case_set)
    ]

    if args.dump_json:
        payload = {
            "input": {
                "case_set": args.case_set,
                "num_experts": args.num_experts,
                "tokens": args.tokens,
                "top_k": args.top_k,
                "cores": args.cores,
                "plan_cost_source": args.plan_cost_source,
            },
            "results": rows,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_summary(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
