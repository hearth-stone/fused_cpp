"""Benchmark uncached Python and native IntervalPlanner searches.

The profile and native planner objects are constructed once. Each timed call
performs the full candidate search and Python bridge materialization without
using PlannedMoE's plan cache.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"))

from interval_planner import IntervalPlanner  # noqa: E402
from phase_model import ContentionCostModel  # noqa: E402
from workload_catalog import default_offline_workloads  # noqa: E402


DEFAULT_CASES = ("moe256-long-short-bimodal", "dsv4-real-2048-seq70")


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = round(quantile * (len(ordered) - 1))
    return ordered[index]


def _assert_plan_equivalent(case: str, backend: str, expected: dict, actual: dict) -> None:
    exact_fields = (
        "shape",
        "assignment_order",
        "execution_mode",
        "tail_pool_threads",
        "tail_pool_max_routes",
        "tail_pool_tasks",
        "active_working_set_bytes",
        "resource_groups",
        "window_bytes_per_worker",
        "tasks",
        "bridge",
        "ranking",
    )
    for field in exact_fields:
        if actual[field] != expected[field]:
            raise AssertionError(f"{case}: {backend} differs in {field}")
    for field in ("makespan_ns", "uncertainty_ns"):
        if not math.isclose(actual[field], expected[field], rel_tol=1e-12):
            raise AssertionError(
                f"{case}: {backend} differs in {field}: "
                f"{actual[field]} != {expected[field]}"
            )


def _measure(planner: IntervalPlanner, experts, warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        planner.plan(experts)
    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(runs):
            begin = time.perf_counter_ns()
            planner.plan(experts)
            samples.append((time.perf_counter_ns() - begin) / 1e6)
    finally:
        gc.enable()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=96)
    parser.add_argument("--workers", default="1,2,4,8,16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    args = parser.parse_args()

    if args.cores <= 0 or args.warmup < 0 or args.runs <= 0:
        raise ValueError("cores and runs must be positive; warmup must be non-negative")
    workers = tuple(int(value) for value in args.workers.split(",") if value)
    if not workers or any(value <= 0 for value in workers):
        raise ValueError("workers must contain positive integers")

    model = ContentionCostModel(args.profile)
    workloads = default_offline_workloads()
    cases = tuple(value for value in args.cases.split(",") if value)
    missing = [name for name in cases if name not in workloads]
    if missing:
        raise KeyError(f"unknown workloads: {missing}")

    planners = [("python", IntervalPlanner(model, args.cores, native_cold_planner=False))]
    planners.extend(
        (
            f"cpp-{worker}T",
            IntervalPlanner(
                model,
                args.cores,
                native_cold_planner=True,
                planner_threads=worker,
            ),
        )
        for worker in workers
    )

    print(
        "case backend workers strict dynamic median_ms p10_ms p90_ms speedup "
        "mode shape"
    )
    for case in cases:
        experts = workloads[case].experts
        expected = planners[0][1].plan(experts)
        reference_samples = _measure(
            planners[0][1],
            experts,
            args.warmup,
            args.runs,
        )
        reference_median = statistics.median(reference_samples)
        for index, (backend, planner) in enumerate(planners):
            actual = planner.plan(experts)
            _assert_plan_equivalent(case, backend, expected, actual)
            samples = (
                reference_samples
                if index == 0
                else _measure(planner, experts, args.warmup, args.runs)
            )
            median = statistics.median(samples)
            print(
                case,
                backend,
                actual["planner_workers"],
                actual["strict_candidates"],
                actual["dynamic_candidates"],
                f"{median:.3f}",
                f"{_percentile(samples, 0.10):.3f}",
                f"{_percentile(samples, 0.90):.3f}",
                f"{reference_median / median:.2f}x",
                actual["execution_mode"],
                "x".join(str(value) for value in actual["shape"]),
            )


if __name__ == "__main__":
    main()
