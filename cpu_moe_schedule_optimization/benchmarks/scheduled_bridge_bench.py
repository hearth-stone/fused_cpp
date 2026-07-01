#!/usr/bin/env python3
"""Run real BF16 tiled MoE timings from offline simulator schedules."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PLANNERS_DIR = ROOT / "planners"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(PLANNERS_DIR))
sys.path.insert(0, str(SRC_DIR))

from offline_simulator import (  # noqa: E402
    ExpertCostModel,
    Plan,
    PlanKind,
    active_experts_from_routes,
    build_planner_cost_model,
    build_plans,
    format_ns,
    generate_routes,
    planner_cost_summary,
    select_auto_planners,
    select_best_plan,
    workload_stats,
)

from fused_cpp.moe import (  # noqa: E402
    _HAS_BF16_TILED_FUSED_MOE,
    fused_moe_bf16_tiled,
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


DSV4_HIDDEN_SIZE = 4096
DSV4_MOE_INTERMEDIATE_SIZE = 2048
DSV4_TP_SIZE = 4
DSV4_FFN_PER_RANK = DSV4_MOE_INTERMEDIATE_SIZE // DSV4_TP_SIZE
DSV4_EXPERTS = 256
DSV4_TOP_K = 6


def bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def topk_from_routes(
    routes: Sequence[int],
    *,
    tokens: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_routes = tokens * top_k
    if sum(routes) != total_routes:
        raise ValueError(
            f"route sum must equal tokens * top_k: {sum(routes)} vs {total_routes}"
        )

    topk_ids = torch.empty((tokens, top_k), dtype=torch.int32)
    cursor = 0
    for expert, count in enumerate(routes):
        for offset in range(int(count)):
            flat = cursor + offset
            token = flat % tokens
            slot = flat // tokens
            if slot >= top_k:
                raise ValueError("route counts overfilled topk slots")
            topk_ids[token, slot] = expert
        cursor += int(count)
    topk_weights = torch.full(
        (tokens, top_k),
        1.0 / float(top_k),
        dtype=torch.float32,
    )
    return topk_weights, topk_ids


def bridge_tensors(plan: Plan) -> Dict[str, torch.Tensor | int]:
    bridge = plan.to_scheduled_bridge()
    return {
        "num_threads": int(bridge["num_threads"]),
        "thread_cpu_ids": torch.tensor(
            bridge.get(
                "thread_cpu_ids",
                list(range(int(bridge["num_threads"]))),
            ),
            dtype=torch.int32,
        ),
        "wave_offsets": torch.tensor(bridge["wave_offsets"], dtype=torch.int32),
        "team_expert_ids": torch.tensor(
            bridge["team_expert_ids"],
            dtype=torch.int32,
        ),
        "team_threads": torch.tensor(bridge["team_threads"], dtype=torch.int32),
    }


def bench_call(
    run,
    *,
    warmup: int,
    runs: int,
) -> tuple[torch.Tensor, Dict[str, object]]:
    out = None
    for _ in range(warmup):
        out = run()
        _ = float(out.flatten()[0])
    gc.collect()

    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = run()
        _ = float(out.flatten()[0])
        times.append(time.perf_counter() - t0)

    if out is None:
        out = run()
    return out, {
        "median_s": statistics.median(times),
        "best_s": min(times),
        "mean_s": statistics.mean(times),
        "times_s": times,
    }


def timing_to_dict(result: Dict[str, object]) -> Dict[str, object]:
    times_s = [float(value) for value in result["times_s"]]
    median_s = float(result["median_s"])
    best_s = float(result["best_s"])
    mean_s = float(result["mean_s"])
    return {
        "median_s": median_s,
        "best_s": best_s,
        "mean_s": mean_s,
        "times_s": times_s,
        "median_ms": median_s * 1e3,
        "best_ms": best_s * 1e3,
        "mean_ms": mean_s * 1e3,
        "times_ms": [value * 1e3 for value in times_s],
        "median_ns": int(round(median_s * 1e9)),
        "best_ns": int(round(best_s * 1e9)),
        "mean_ns": int(round(mean_s * 1e9)),
    }


def plan_shape_features(plan: Plan) -> Dict[str, object]:
    teams = [team for wave in plan.waves for team in wave.teams]
    multithread_teams = [team for team in teams if team.threads > 1]
    thread_histogram: Dict[str, int] = {}
    for team in teams:
        key = str(team.threads)
        thread_histogram[key] = thread_histogram.get(key, 0) + 1
    return {
        "num_waves": len(plan.waves),
        "num_teams": len(teams),
        "num_multithread_teams": len(multithread_teams),
        "total_team_threads": sum(team.threads for team in teams),
        "max_team_threads": max((team.threads for team in teams), default=0),
        "max_teams_per_wave": max((len(wave.teams) for wave in plan.waves), default=0),
        "thread_histogram": thread_histogram,
    }


def serialize_args(args: argparse.Namespace) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, list):
            data[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            data[key] = value
    return data


def measured_plan_row(
    plan: Plan,
    result: Dict[str, object],
    *,
    baseline_median_s: float | None,
    best_measured_median_s: float,
) -> Dict[str, object]:
    timing = timing_to_dict(result)
    shape = plan_shape_features(plan)
    median_s = float(timing["median_s"])
    speedup_vs_balanced = (
        baseline_median_s / median_s if baseline_median_s is not None else None
    )
    return {
        "kind": plan.kind.value,
        "exactness_scope": plan.exactness_scope,
        "prediction": {
            "estimated_plan_cost_ns": plan.estimated_plan_cost_ns,
            "measured_plan_cost_ns": plan.measured_plan_cost_ns,
            "estimated_execute_cost_ns": plan.estimated_execute_cost_ns,
            "estimated_total_cost_ns": plan.estimated_total_cost_ns,
        },
        "measured": timing,
        "measured_speedup_vs_balanced": speedup_vs_balanced,
        "measured_regret_vs_best": median_s / best_measured_median_s,
        "shape": shape,
        "metadata": plan.metadata,
        "plan": plan.to_dict(),
    }


def build_result_payload(
    *,
    args: argparse.Namespace,
    routes: Sequence[int],
    stats: Dict[str, float],
    cost_model: ExpertCostModel,
    planner_cost_model_summary: str,
    planner_names: Sequence[str],
    best: Plan,
    default_result: Dict[str, object] | None,
    measured_results: Sequence[tuple[Plan, Dict[str, object]]],
) -> Dict[str, object]:
    baseline_median_s = None
    for plan, result in measured_results:
        if plan.kind == PlanKind.SORTED_TOKEN_BALANCED_1T:
            baseline_median_s = float(result["median_s"])
            break
    best_measured_median_s = min(
        (float(result["median_s"]) for _, result in measured_results),
        default=None,
    )
    if best_measured_median_s is None:
        best_measured_median_s = float("nan")

    active_routes = [
        {"expert_id": expert_id, "routes": int(route_count)}
        for expert_id, route_count in enumerate(routes)
        if route_count > 0
    ]
    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "argv": sys.argv,
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "args": serialize_args(args),
        "workload": {
            "distribution": args.distribution,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "num_experts": args.num_experts,
            "cores": args.cores,
            "routes": [int(value) for value in routes],
            "active_routes": active_routes,
            "stats": stats,
        },
        "shape": {
            "hidden_size": args.hidden_size,
            "ffn_hidden_size": args.ffn_hidden_size,
            "activation": args.activation,
        },
        "cost_model": {
            "source": cost_model.source,
            "metric": cost_model.metric,
            "cost_table": str(args.cost_table) if args.cost_table is not None else None,
        },
        "planner_cost_model": planner_cost_model_summary,
        "planner_names": list(planner_names),
        "best_estimated_total": best.kind.value,
        "default_kernel": (
            {"kind": "EXISTING_DEFAULT_KERNEL", "measured": timing_to_dict(default_result)}
            if default_result is not None
            else None
        ),
        "plans": [
            measured_plan_row(
                plan,
                result,
                baseline_median_s=baseline_median_s,
                best_measured_median_s=best_measured_median_s,
            )
            for plan, result in measured_results
        ],
    }
    return payload


def csv_rows_from_payload(payload: Dict[str, object]) -> List[Dict[str, object]]:
    workload = payload["workload"]
    shape = payload["shape"]
    rows: List[Dict[str, object]] = []
    for plan in payload["plans"]:
        prediction = plan["prediction"]
        measured = plan["measured"]
        plan_shape = plan["shape"]
        rows.append(
            {
                "distribution": workload["distribution"],
                "tokens": workload["tokens"],
                "top_k": workload["top_k"],
                "num_experts": workload["num_experts"],
                "cores": workload["cores"],
                "active_experts": workload["stats"]["active_experts"],
                "gini": workload["stats"]["gini"],
                "maxvio": workload["stats"]["maxvio"],
                "hidden_size": shape["hidden_size"],
                "ffn_hidden_size": shape["ffn_hidden_size"],
                "activation": shape["activation"],
                "planner": plan["kind"],
                "estimated_plan_cost_ns": prediction["estimated_plan_cost_ns"],
                "measured_plan_cost_ns": prediction["measured_plan_cost_ns"],
                "estimated_execute_cost_ns": prediction["estimated_execute_cost_ns"],
                "estimated_total_cost_ns": prediction["estimated_total_cost_ns"],
                "median_ns": measured["median_ns"],
                "best_ns": measured["best_ns"],
                "mean_ns": measured["mean_ns"],
                "median_ms": measured["median_ms"],
                "best_ms": measured["best_ms"],
                "mean_ms": measured["mean_ms"],
                "measured_speedup_vs_balanced": plan["measured_speedup_vs_balanced"],
                "measured_regret_vs_best": plan["measured_regret_vs_best"],
                "num_waves": plan_shape["num_waves"],
                "num_teams": plan_shape["num_teams"],
                "num_multithread_teams": plan_shape["num_multithread_teams"],
                "total_team_threads": plan_shape["total_team_threads"],
                "max_team_threads": plan_shape["max_team_threads"],
                "max_teams_per_wave": plan_shape["max_teams_per_wave"],
                "selected_core_group_shape": plan["metadata"].get(
                    "selected_core_group_shape",
                    "",
                ),
            }
        )
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no benchmark rows to write")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_result(
    plan: Plan,
    result: Dict[str, object],
    baseline_median_s: float | None = None,
) -> None:
    bridge = plan.to_scheduled_bridge()
    selected_shape = plan.metadata.get("selected_core_group_shape", "")
    median_s = float(result["median_s"])
    best_s = float(result["best_s"])
    mean_s = float(result["mean_s"])
    times = [round(float(value) * 1e3, 3) for value in result["times_s"]]
    baseline_text = ""
    if baseline_median_s is not None:
        baseline_text = f"vs_balanced={baseline_median_s / median_s:>7.3f} "
    print(
        f"{plan.kind.value:<24} "
        f"waves={len(plan.waves):<3} "
        f"teams={len(bridge['team_expert_ids']):<4} "
        f"shape={selected_shape!s:<18} "
        f"est_execute={format_ns(plan.estimated_execute_cost_ns):>10} "
        f"est_total={format_ns(plan.estimated_total_cost_ns):>10} "
        f"median_ms={median_s * 1e3:>9.3f} "
        f"best_ms={best_s * 1e3:>9.3f} "
        f"mean_ms={mean_s * 1e3:>9.3f} "
        f"{baseline_text}"
        f"times_ms={times}"
    )


def print_default_result(result: Dict[str, object]) -> None:
    median_s = float(result["median_s"])
    best_s = float(result["best_s"])
    mean_s = float(result["mean_s"])
    times = [round(float(value) * 1e3, 3) for value in result["times_s"]]
    print(
        f"{'EXISTING_DEFAULT_KERNEL':<24} "
        f"waves={'n/a':<3} "
        f"teams={'n/a':<4} "
        f"shape={'default':<18} "
        f"est_execute={'n/a':>10} "
        f"est_total={'n/a':>10} "
        f"median_ms={median_s * 1e3:>9.3f} "
        f"best_ms={best_s * 1e3:>9.3f} "
        f"mean_ms={mean_s * 1e3:>9.3f} "
        f"times_ms={times}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure fused_moe_bf16_tiled_scheduled with simulator plans.",
    )
    parser.add_argument("--num-experts", type=int, default=DSV4_EXPERTS)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=DSV4_TOP_K)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=DSV4_HIDDEN_SIZE)
    parser.add_argument("--ffn-hidden-size", type=int, default=DSV4_FFN_PER_RANK)
    parser.add_argument(
        "--distribution",
        choices=[
            "uniform",
            "random_balanced",
            "active_subset",
            "hotspot",
            "heavy_light",
            "zipf",
            "dirichlet",
            "lognormal",
            "domain_cluster",
            "one_hot",
        ],
        default="active_subset",
    )
    parser.add_argument("--zipf-alpha", type=float, default=1.15)
    parser.add_argument("--active-experts", type=int, default=6)
    parser.add_argument("--hot-experts", type=int, default=4)
    parser.add_argument("--hot-fraction", type=float, default=0.70)
    parser.add_argument("--heavy-experts", type=int, default=8)
    parser.add_argument("--heavy-fraction", type=float, default=0.80)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.25)
    parser.add_argument("--lognormal-sigma", type=float, default=1.50)
    parser.add_argument("--cluster-experts", type=int, default=32)
    parser.add_argument("--background-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--planner",
        action="append",
        choices=[
            "all",
            "auto",
            "fixed",
            "balanced",
            "uniform",
            "greedy",
            "groups",
        ],
        default=None,
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
    parser.add_argument("--planner-cost", action="append", default=None)
    parser.add_argument(
        "--cost-table",
        type=Path,
        default=None,
        help="Optional JSON expert cost table generated by profile_expert_cost.py.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--activation", choices=["silu", "swigluoai"], default="silu")
    parser.add_argument(
        "--include-default",
        action="store_true",
        help="Also time the existing non-scheduled fused_moe_bf16_tiled path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write structured benchmark results, including full plans, to JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write one row per measured planner to CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError("BF16 tiled fused MoE backend is unavailable")
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.top_k <= 0 or args.top_k > args.num_experts:
        raise ValueError("--top-k must be in [1, num_experts]")
    if args.cores <= 0:
        raise ValueError("--cores must be positive")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    routes = generate_routes(
        distribution=args.distribution,
        num_experts=args.num_experts,
        tokens=args.tokens,
        top_k=args.top_k,
        seed=args.seed,
        zipf_alpha=args.zipf_alpha,
        active_experts=args.active_experts,
        hot_experts=args.hot_experts,
        hot_fraction=args.hot_fraction,
        heavy_experts=args.heavy_experts,
        heavy_fraction=args.heavy_fraction,
        dirichlet_alpha=args.dirichlet_alpha,
        lognormal_sigma=args.lognormal_sigma,
        cluster_experts=args.cluster_experts,
        background_fraction=args.background_fraction,
    )
    active = active_experts_from_routes(routes)
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
    planner_cost_model_text = planner_cost_summary(planner_cost_model)
    planner_names = args.planner or ["balanced"]
    if "auto" in planner_names:
        if len(planner_names) > 1:
            raise ValueError("--planner auto cannot be combined with other planners")
        planner_names, auto_reason, _ = select_auto_planners(routes, args.cores)
        print(f"auto_selector: reason={auto_reason} enabled={','.join(planner_names)}")
    plans = build_plans(
        active,
        args.cores,
        cost_model,
        planner_cost_model,
        planner_names,
    )
    best = select_best_plan(plans)

    hidden_states = bf16_normal(
        (args.tokens, args.hidden_size),
        generator=generator,
        std=args.std,
    )
    w13_weight = bf16_normal(
        (args.num_experts, 2 * args.ffn_hidden_size, args.hidden_size),
        generator=generator,
        std=args.std,
    )
    w2_weight = bf16_normal(
        (args.num_experts, args.hidden_size, args.ffn_hidden_size),
        generator=generator,
        std=args.std,
    )
    topk_weights, topk_ids = topk_from_routes(
        routes,
        tokens=args.tokens,
        top_k=args.top_k,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    stats = workload_stats(routes)
    print(
        "workload: "
        f"distribution={args.distribution} tokens={args.tokens} top_k={args.top_k} "
        f"experts={args.num_experts} cores={args.cores} "
        f"active={int(stats['active_experts'])} "
        f"gini={float(stats['gini']):.3f} maxvio={float(stats['maxvio']):.2f}"
    )
    print(
        f"shape: H={args.hidden_size} F={args.ffn_hidden_size} "
        f"activation={args.activation}"
    )
    print(f"best_estimated_total={best.kind.value}")

    default_result = None
    if args.include_default:
        _, default_result = bench_call(
            lambda: fused_moe_bf16_tiled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                num_threads=args.cores,
                activation=args.activation,
            ),
            warmup=args.warmup,
            runs=args.runs,
        )
        print_default_result(default_result)

    measured_results: List[tuple[Plan, Dict[str, object]]] = []
    for plan in sorted(plans, key=lambda item: item.estimated_total_cost_ns):
        bridge = bridge_tensors(plan)
        _, result = bench_call(
            lambda bridge=bridge: fused_moe_bf16_tiled_scheduled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                bridge["wave_offsets"],
                bridge["team_expert_ids"],
                bridge["team_threads"],
                thread_cpu_ids=bridge["thread_cpu_ids"],
                num_threads=int(bridge["num_threads"]),
                activation=args.activation,
            ),
            warmup=args.warmup,
            runs=args.runs,
        )
        measured_results.append((plan, result))

    baseline_median_s = None
    for plan, result in measured_results:
        if plan.kind == PlanKind.SORTED_TOKEN_BALANCED_1T:
            baseline_median_s = float(result["median_s"])
            break

    for plan, result in measured_results:
        print_result(plan, result, baseline_median_s=baseline_median_s)

    if args.output_json is not None or args.output_csv is not None:
        payload = build_result_payload(
            args=args,
            routes=routes,
            stats=stats,
            cost_model=cost_model,
            planner_cost_model_summary=planner_cost_model_text,
            planner_names=planner_names,
            best=best,
            default_result=default_result,
            measured_results=measured_results,
        )
        if args.output_json is not None:
            write_json(args.output_json, payload)
            print(f"wrote_json={args.output_json}")
        if args.output_csv is not None:
            write_csv(args.output_csv, csv_rows_from_payload(payload))
            print(f"wrote_csv={args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
