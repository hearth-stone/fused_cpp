#!/usr/bin/env python3
"""Stress test the AUTO planner selector across synthetic workload grids."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
PLANNERS_DIR = ROOT / "planners"
BENCHMARKS_DIR = ROOT / "benchmarks"
sys.path.insert(0, str(PLANNERS_DIR))
sys.path.insert(0, str(BENCHMARKS_DIR))

from offline_simulator import (  # noqa: E402
    ExpertCostModel,
    PlanKind,
    active_experts_from_routes,
    build_planner_cost_model,
    build_plans,
    format_ns,
    select_auto_planners,
    select_best_plan,
    workload_stats,
)
from synthetic_sweep import SyntheticCase, case_routes  # noqa: E402


@dataclass(frozen=True)
class StressResult:
    case: str
    distribution: str
    cores: int
    tokens: int
    top_k: int
    params: Dict[str, object]
    active_experts: int
    gini: float
    maxvio: float
    oracle_kind: str
    auto_kind: str
    auto_reason: str
    auto_enabled: Sequence[str]
    oracle_total_ns: int
    auto_total_ns: int
    regret: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "case": self.case,
            "distribution": self.distribution,
            "cores": self.cores,
            "tokens": self.tokens,
            "top_k": self.top_k,
            "params": self.params,
            "active_experts": self.active_experts,
            "gini": self.gini,
            "maxvio": self.maxvio,
            "oracle_kind": self.oracle_kind,
            "auto_kind": self.auto_kind,
            "auto_reason": self.auto_reason,
            "auto_enabled": list(self.auto_enabled),
            "oracle_total_ns": self.oracle_total_ns,
            "auto_total_ns": self.auto_total_ns,
            "regret": self.regret,
        }


def build_stress_cases(case_set: str) -> List[SyntheticCase]:
    cases: List[SyntheticCase] = []

    def add(name: str, distribution: str, params: Dict[str, object]) -> None:
        cases.append(SyntheticCase(name, distribution, params))

    for active in [1, 2, 4, 6, 8, 16, 32, 64, 128, 256]:
        add(f"active_subset_{active}", "active_subset", {"active_experts": active})

    for hot_experts in [1, 2, 4, 8, 16]:
        for hot_fraction in [0.20, 0.40, 0.60, 0.75, 0.90, 0.95]:
            add(
                f"hotspot_{hot_experts}x{int(hot_fraction * 100):02d}",
                "hotspot",
                {"hot_experts": hot_experts, "hot_fraction": hot_fraction},
            )

    for alpha in [0.50, 0.80, 1.00, 1.15, 1.35, 1.50, 1.75, 2.00]:
        add(f"zipf_{alpha:.2f}", "zipf", {"zipf_alpha": alpha})

    for alpha in [0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]:
        for seed in [0, 1, 2]:
            add(
                f"dirichlet_{alpha:.2f}_s{seed}",
                "dirichlet",
                {"dirichlet_alpha": alpha, "seed": seed},
            )

    for sigma in [0.50, 1.00, 1.50, 2.00, 2.50, 3.00]:
        for seed in [0, 1, 2]:
            add(
                f"lognormal_{sigma:.2f}_s{seed}",
                "lognormal",
                {"lognormal_sigma": sigma, "seed": seed},
            )

    for heavy_experts in [4, 8, 16, 32]:
        for heavy_fraction in [0.60, 0.75, 0.85, 0.95]:
            add(
                f"heavy_light_{heavy_experts}x{int(heavy_fraction * 100):02d}",
                "heavy_light",
                {
                    "heavy_experts": heavy_experts,
                    "heavy_fraction": heavy_fraction,
                },
            )

    for cluster_experts in [8, 16, 32, 64]:
        for background_fraction in [0.01, 0.05, 0.10, 0.25]:
            add(
                f"domain_cluster_{cluster_experts}_bg{int(background_fraction * 100):02d}",
                "domain_cluster",
                {
                    "cluster_experts": cluster_experts,
                    "background_fraction": background_fraction,
                    "seed": 3,
                },
            )

    if case_set == "quick":
        return cases[:30]
    return cases


def run_stress_case(
    case: SyntheticCase,
    cores: int,
    num_experts: int,
    tokens: int,
    top_k: int,
    cost_model: ExpertCostModel,
    planner_cost_model,
) -> StressResult:
    routes = case_routes(case, num_experts, tokens, top_k)
    active = active_experts_from_routes(routes)
    all_plans = build_plans(active, cores, cost_model, planner_cost_model, ["all"])
    oracle = select_best_plan(all_plans)

    auto_names, auto_reason, _ = select_auto_planners(routes, cores)
    auto_plans = build_plans(active, cores, cost_model, planner_cost_model, auto_names)
    auto = select_best_plan(auto_plans)
    stats = workload_stats(routes)

    return StressResult(
        case=case.name,
        distribution=case.distribution,
        cores=cores,
        tokens=tokens,
        top_k=top_k,
        params=case.params,
        active_experts=int(stats["active_experts"]),
        gini=float(stats["gini"]),
        maxvio=float(stats["maxvio"]),
        oracle_kind=oracle.kind.value,
        auto_kind=auto.kind.value,
        auto_reason=auto_reason,
        auto_enabled=auto_names,
        oracle_total_ns=oracle.estimated_total_cost_ns,
        auto_total_ns=auto.estimated_total_cost_ns,
        regret=auto.estimated_total_cost_ns / oracle.estimated_total_cost_ns,
    )


def summarize(results: Sequence[StressResult], top_n: int, regret_threshold: float) -> None:
    worst = sorted(results, key=lambda row: row.regret, reverse=True)
    misses = [row for row in results if row.regret > regret_threshold]
    reason_counts = Counter(row.auto_reason for row in results)
    oracle_counts = Counter(row.oracle_kind for row in results)
    auto_counts = Counter(row.auto_kind for row in results)

    print(f"cases={len(results)}")
    print(f"max_regret={worst[0].regret:.4f}")
    print(f"avg_regret={sum(row.regret for row in results) / len(results):.4f}")
    print(f"misses>{regret_threshold:.3f}={len(misses)}")
    print(f"oracle_counts={dict(oracle_counts)}")
    print(f"auto_counts={dict(auto_counts)}")
    print(f"auto_reason_counts={dict(reason_counts)}")
    print()

    header = (
        f"{'case':<32} {'c':>2} {'act':>4} {'gini':>5} {'maxvio':>7} "
        f"{'oracle':<24} {'auto':<24} {'regret':>7} {'reason':<28} "
        f"{'oracle_total':>12} {'auto_total':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in worst[:top_n]:
        print(
            f"{row.case:<32} "
            f"{row.cores:>2} "
            f"{row.active_experts:>4} "
            f"{row.gini:>5.3f} "
            f"{row.maxvio:>7.2f} "
            f"{row.oracle_kind:<24} "
            f"{row.auto_kind:<24} "
            f"{row.regret:>7.3f} "
            f"{row.auto_reason:<28} "
            f"{format_ns(row.oracle_total_ns):>12} "
            f"{format_ns(row.auto_total_ns):>12}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress test AUTO planner selector regret.",
    )
    parser.add_argument("--case-set", choices=["quick", "full"], default="full")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--cores",
        type=int,
        action="append",
        default=None,
        help="Core count to test. Can be passed multiple times.",
    )
    parser.add_argument(
        "--plan-cost-source",
        choices=["complexity", "model", "measured", "native_table", "profile"],
        default="complexity",
    )
    parser.add_argument(
        "--planner-cost-profile",
        type=Path,
        default=None,
        help=(
            "Native C++ planner cost table. Use with "
            "--plan-cost-source native_table/profile."
        ),
    )
    parser.add_argument(
        "--planner-cost",
        action="append",
        default=None,
        help="Override planner cost, for example fixed=1us or greedy=20us.",
    )
    parser.add_argument(
        "--cost-table",
        type=Path,
        default=None,
        help="Optional JSON expert cost table generated by profile_expert_cost.py.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--regret-threshold", type=float, default=1.03)
    parser.add_argument("--dump-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    cores_list = args.cores or [4, 8, 16, 32]
    if any(cores <= 0 for cores in cores_list):
        raise ValueError("--cores values must be positive")

    cost_model = (
        ExpertCostModel.from_json(args.cost_table)
        if args.cost_table is not None
        else ExpertCostModel.synthetic()
    )
    planner_cost_model = build_planner_cost_model(
        source=args.plan_cost_source,
        overrides=args.planner_cost,
        profile_path=args.planner_cost_profile,
    )
    cases = build_stress_cases(args.case_set)
    results = [
        run_stress_case(
            case=case,
            cores=cores,
            num_experts=args.num_experts,
            tokens=args.tokens,
            top_k=args.top_k,
            cost_model=cost_model,
            planner_cost_model=planner_cost_model,
        )
        for cores in cores_list
        for case in cases
    ]

    if args.dump_json:
        print(json.dumps([row.to_dict() for row in results], indent=2, sort_keys=True))
    else:
        summarize(results, top_n=args.top_n, regret_threshold=args.regret_threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
