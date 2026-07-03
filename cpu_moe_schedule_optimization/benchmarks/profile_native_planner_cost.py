#!/usr/bin/env python3
# ⚠ DEPRECATED (wave, 后续不考虑) — see cpu_moe_schedule_optimization/DEPRECATED_WAVE.md
"""Profile native C++ MoE planner latency and emit a planner-cost profile."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PLANNERS_DIR = ROOT / "planners"
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(PLANNERS_DIR))
sys.path.insert(0, str(SRC_DIR))

from offline_simulator import (  # noqa: E402
    COMPLEXITY_COST_COEFFICIENTS_NS,
    PlanKind,
    build_planner_cost_model,
    count_core_group_shapes,
    normalize_planner_kind,
    workload_stats,
)
from synthetic_sweep import SyntheticCase, case_routes, default_cases  # noqa: E402

try:
    from fused_cpp import _C  # type: ignore[attr-defined]  # noqa: E402
except ImportError as exc:  # pragma: no cover - depends on extension build.
    raise RuntimeError(
        "failed to import fused_cpp._C; rebuild the extension before profiling"
    ) from exc


DEFAULT_PLANNERS = (
    "fixed",
    "balanced",
    "uniform",
    "groups",
    "greedy",
)
DEFAULT_ACTIVE_COUNTS = "1,2,4,6,8,16,32,64,128,192,256"
COEFFICIENT_NAMES = [
    "fixed_base",
    "balanced_base",
    "uniform_base",
    "greedy_base",
    "groups_base",
    "cost_lookup",
    "linear_scan",
    "wave_pack",
    "gain_scan",
    "sort_compare",
    "budget_eval",
]
BASE_COEFFICIENT_BY_KIND = {
    PlanKind.FIXED_GLOBAL_THREADS: "fixed_base",
    PlanKind.SORTED_TOKEN_BALANCED_1T: "balanced_base",
    PlanKind.UNIFORM_WAVES: "uniform_base",
    PlanKind.GREEDY_MARGINAL_GAIN: "greedy_base",
    PlanKind.ENUMERATE_CORE_GROUPS: "groups_base",
}
OP_FEATURES = {
    "cost_lookup": "cost_lookup_ops",
    "linear_scan": "linear_scan_ops",
    "wave_pack": "wave_pack_ops",
    "gain_scan": "gain_scan_ops",
    "sort_compare": "sort_compare_ops",
    "budget_eval": "budget_eval_ops",
}


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def parse_planners(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("planner list must not be empty")
    if values == ["all"]:
        return list(DEFAULT_PLANNERS)
    for value in values:
        normalize_planner_kind(value)
    return values


def load_expert_cost_table(
    path: Path | None,
) -> Tuple[List[int] | None, List[int] | None, List[int] | None, Dict[str, object]]:
    if path is None:
        return None, None, None, {"source": "synthetic"}

    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = str(payload.get("metric", "median_ns"))
    route_buckets = [int(value) for value in payload["route_buckets"]]
    thread_buckets = [int(value) for value in payload["thread_buckets"]]
    table = {
        (int(entry["routes"]), int(entry["threads"])): int(entry[metric])
        for entry in payload["entries"]
    }
    values = [
        table[(routes, threads)]
        for routes in route_buckets
        for threads in thread_buckets
    ]
    return route_buckets, thread_buckets, values, {
        "source": str(path),
        "metric": metric,
        "route_buckets": route_buckets,
        "thread_buckets": thread_buckets,
    }


def active_grid_cases(active_counts: Sequence[int], *, cores: int) -> List[SyntheticCase]:
    cases = []
    for active_count in active_counts:
        cases.append(
            SyntheticCase(
                name=f"active_grid_{active_count}",
                distribution="active_subset",
                params={"active_experts": active_count},
                cores=cores,
            )
        )
    return cases


def feature_vector(kind: PlanKind, active_count: int, cores: int) -> List[float]:
    model = build_planner_cost_model("complexity", None)
    features = model.complexity_features(
        kind=kind,
        active_count=active_count,
        num_cores=cores,
        extra_wave_window=8,
        core_group_shape_count=(
            count_core_group_shapes(cores)
            if kind == PlanKind.ENUMERATE_CORE_GROUPS
            else None
        ),
    )
    row = {name: 0.0 for name in COEFFICIENT_NAMES}
    row[BASE_COEFFICIENT_BY_KIND[kind]] = 1.0
    for coeff_name, feature_name in OP_FEATURES.items():
        row[coeff_name] = float(features[feature_name])
    return [row[name] for name in COEFFICIENT_NAMES]


def solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-18:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        divisor = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= divisor
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fit_coefficients(
    rows: Sequence[Dict[str, object]],
    *,
    metric: str,
    ridge: float,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    samples = [
        row
        for row in rows
        if normalize_planner_kind(str(row["planner"])) in BASE_COEFFICIENT_BY_KIND
    ]
    if len(samples) < len(COEFFICIENT_NAMES):
        return dict(COMPLEXITY_COST_COEFFICIENTS_NS), {
            "num_samples": float(len(samples)),
            "status": "not_enough_samples",
        }

    x_rows = [
        feature_vector(
            normalize_planner_kind(str(row["planner"])),
            int(row["active_experts"]),
            int(row["cores"]),
        )
        for row in samples
    ]
    y = [float(row[metric]) for row in samples]
    scales = [1.0] * len(COEFFICIENT_NAMES)
    for col in range(len(COEFFICIENT_NAMES)):
        scales[col] = max(1.0, max(abs(row[col]) for row in x_rows))
    x_scaled = [[value / scales[col] for col, value in enumerate(row)] for row in x_rows]

    n = len(COEFFICIENT_NAMES)
    xtx = [[0.0 for _ in range(n)] for _ in range(n)]
    xty = [0.0 for _ in range(n)]
    for row, target in zip(x_scaled, y):
        for i in range(n):
            xty[i] += row[i] * target
            for j in range(n):
                xtx[i][j] += row[i] * row[j]
    for i in range(n):
        xtx[i][i] += ridge

    beta_scaled = solve_linear_system(xtx, xty)
    beta = [beta_scaled[i] / scales[i] for i in range(n)]
    coefficients = {
        name: max(0, int(round(value)))
        for name, value in zip(COEFFICIENT_NAMES, beta)
    }

    predictions = []
    errors = []
    for row, target in zip(x_rows, y):
        predicted = sum(
            row[i] * coefficients[name] for i, name in enumerate(COEFFICIENT_NAMES)
        )
        predictions.append(predicted)
        if target > 0.0:
            errors.append(abs(predicted - target) / target)
    diagnostics = {
        "num_samples": float(len(samples)),
        "ridge": ridge,
        "mean_absolute_percentage_error": (
            sum(errors) / len(errors) if errors else 0.0
        ),
        "max_absolute_percentage_error": max(errors) if errors else 0.0,
        "min_prediction_ns": min(predictions) if predictions else 0.0,
        "max_prediction_ns": max(predictions) if predictions else 0.0,
    }
    return coefficients, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile native C++ MoE planner latency."
    )
    parser.add_argument("--case-set", choices=["smoke", "full"], default="full")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--planners", default="all")
    parser.add_argument(
        "--cost-table",
        type=Path,
        default=None,
        help="Expert cost table passed to the native planner.",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--include-active-grid",
        action="store_true",
        help="Add active_subset cases for --active-counts.",
    )
    parser.add_argument("--active-counts", default=DEFAULT_ACTIVE_COUNTS)
    parser.add_argument("--fit-metric", default="total_native_median_ns")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
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
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.ridge < 0.0:
        raise ValueError("--ridge must be non-negative")

    planners = parse_planners(args.planners)
    route_buckets, thread_buckets, table_values, cost_table_meta = (
        load_expert_cost_table(args.cost_table)
    )

    cases = default_cases(args.case_set)
    if args.include_active_grid:
        cases = cases + active_grid_cases(
            parse_int_list(args.active_counts),
            cores=args.cores,
        )

    rows: List[Dict[str, object]] = []
    print(
        "case planner active cores prepare_us plan_us total_us waves execute_ms",
        flush=True,
    )
    for case in cases:
        routes = case_routes(case, args.num_experts, args.tokens, args.top_k)
        case_cores = case.cores or args.cores
        stats = workload_stats(routes)
        for planner in planners:
            timed = _C.moe_schedule_plan_timed(
                [int(value) for value in routes],
                int(case_cores),
                planner,
                int(args.warmup),
                int(args.iters),
                route_buckets,
                thread_buckets,
                table_values,
            )
            row = {
                "case": case.name,
                "distribution": case.distribution,
                "params": case.params,
                "num_experts": case.num_experts or args.num_experts,
                "tokens": case.tokens or args.tokens,
                "top_k": case.top_k or args.top_k,
                "cores": case_cores,
                "active_experts": int(stats["active_experts"]),
                "gini": float(stats["gini"]),
                "maxvio": float(stats["maxvio"]),
                "planner": str(timed["kind"]),
                "warmup": int(timed["warmup"]),
                "iters": int(timed["iters"]),
                "prepare_median_ns": int(timed["prepare_median_ns"]),
                "prepare_mean_ns": int(timed["prepare_mean_ns"]),
                "prepare_p90_ns": int(timed["prepare_p90_ns"]),
                "prepare_p99_ns": int(timed["prepare_p99_ns"]),
                "plan_median_ns": int(timed["plan_median_ns"]),
                "plan_mean_ns": int(timed["plan_mean_ns"]),
                "plan_p90_ns": int(timed["plan_p90_ns"]),
                "plan_p99_ns": int(timed["plan_p99_ns"]),
                "total_native_median_ns": int(timed["total_native_median_ns"]),
                "total_native_mean_ns": int(timed["total_native_mean_ns"]),
                "total_native_p90_ns": int(timed["total_native_p90_ns"]),
                "total_native_p99_ns": int(timed["total_native_p99_ns"]),
                "estimated_execute_cost_ns": int(timed["estimated_execute_cost_ns"]),
                "num_waves": int(timed["num_waves"]),
                "num_teams": int(timed["num_teams"]),
                "selected_threads_per_expert": int(
                    timed["selected_threads_per_expert"]
                ),
                "selected_wave_budget": int(timed["selected_wave_budget"]),
                "selected_core_group_shape": list(timed["selected_core_group_shape"]),
            }
            rows.append(row)
            print(
                f"{case.name:<27} {row['planner']:<24} "
                f"{row['active_experts']:>4} {case_cores:>4} "
                f"{row['prepare_median_ns'] / 1e3:>9.2f} "
                f"{row['plan_median_ns'] / 1e3:>8.2f} "
                f"{row['total_native_median_ns'] / 1e3:>8.2f} "
                f"{row['num_waves']:>5} "
                f"{row['estimated_execute_cost_ns'] / 1e6:>10.3f}",
                flush=True,
            )

    coefficients, fit_diagnostics = fit_coefficients(
        rows,
        metric=args.fit_metric,
        ridge=args.ridge,
    )
    payload = {
        "schema_version": 1,
        "created_unix_s": time.time(),
        "target": {
            "machine": platform.machine(),
            "cpu": platform.processor() or platform.machine(),
            "os": platform.platform(),
        },
        "input": {
            "case_set": args.case_set,
            "num_experts": args.num_experts,
            "tokens": args.tokens,
            "top_k": args.top_k,
            "cores": args.cores,
            "planners": planners,
            "warmup": args.warmup,
            "iters": args.iters,
            "include_active_grid": args.include_active_grid,
            "active_counts": parse_int_list(args.active_counts),
            "cost_table": cost_table_meta,
            "fit_metric": args.fit_metric,
        },
        "metric": "total_native_median_ns",
        "calibrated_coefficients_ns": coefficients,
        "fit_diagnostics": fit_diagnostics,
        "entries": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote_json={args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                key
                for key in rows[0].keys()
                if key not in {"params", "selected_core_group_shape"}
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"wrote_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
