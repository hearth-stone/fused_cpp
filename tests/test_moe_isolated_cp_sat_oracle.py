from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNERS))

from isolated_cp_sat_oracle import (  # noqa: E402
    IsolatedJob,
    IsolatedMode,
    build_isolated_jobs,
    compare_candidate_to_oracle,
    evaluate_isolated_dag,
    main,
    prune_dominated_modes,
    solve_isolated_cp_sat,
)

PROFILE = (
    ROOT
    / "cpu_moe_schedule_optimization"
    / "cost_model"
    / "profiles"
    / "contention_async_amazon_ecs_8c_standalone_sve_F512_E8_splitw13_xbyak_exactm_v2_20260720.json"
)


def test_prune_dominated_modes() -> None:
    modes = (
        IsolatedMode(threads=4, duration_ns=9.0),
        IsolatedMode(threads=1, duration_ns=10.0),
        IsolatedMode(threads=3, duration_ns=8.0),
        IsolatedMode(threads=2, duration_ns=10.0),
    )

    assert prune_dominated_modes(modes) == (
        IsolatedMode(threads=1, duration_ns=10.0),
        IsolatedMode(threads=3, duration_ns=8.0),
    )


def test_build_jobs_filters_inactive_experts_and_validates_modes() -> None:
    jobs = build_isolated_jobs(
        [(0, 12), (1, 0), (2, 24)],
        [1, 2, 4],
        lambda routes, threads: routes * 100.0 / threads,
        num_cores=4,
    )

    assert [job.expert_id for job in jobs] == [0, 2]
    assert jobs[0].modes[-1] == IsolatedMode(threads=4, duration_ns=300.0)

    with pytest.raises(ValueError, match="duplicate expert_id"):
        build_isolated_jobs(
            [(0, 12), (0, 24)],
            [1],
            lambda routes, threads: routes + threads,
            num_cores=4,
        )


def test_isolated_cp_sat_finds_global_width_allocation() -> None:
    pytest.importorskip("ortools")
    jobs = tuple(
        IsolatedJob(
            expert_id=expert_id,
            routes=12,
            modes=(
                IsolatedMode(threads=1, duration_ns=10.0),
                IsolatedMode(threads=2, duration_ns=6.0),
                IsolatedMode(threads=4, duration_ns=5.0),
            ),
        )
        for expert_id in range(2)
    )

    result = solve_isolated_cp_sat(
        jobs,
        num_cores=4,
        workers=1,
        max_time_s=5.0,
    )

    assert result.status == "OPTIMAL"
    assert result.optimal
    assert result.objective_ns == 6
    assert result.best_bound_ns == pytest.approx(6.0)
    assert result.relative_gap == pytest.approx(0.0)
    assert sorted(assignment.threads for assignment in result.assignments) == [2, 2]
    assert result.input_modes == 6
    assert result.retained_modes == 6
    for timestamp in range(result.objective_ns + 1):
        active_threads = sum(
            assignment.threads
            for assignment in result.assignments
            if assignment.start_ns <= timestamp < assignment.end_ns
        )
        assert active_threads <= 4


def test_dominance_pruning_preserves_optimum() -> None:
    pytest.importorskip("ortools")
    jobs = (
        IsolatedJob(
            expert_id=0,
            routes=12,
            modes=(
                IsolatedMode(threads=1, duration_ns=10.0),
                IsolatedMode(threads=2, duration_ns=11.0),
                IsolatedMode(threads=4, duration_ns=5.0),
            ),
        ),
        IsolatedJob(
            expert_id=1,
            routes=12,
            modes=(
                IsolatedMode(threads=1, duration_ns=10.0),
                IsolatedMode(threads=2, duration_ns=11.0),
                IsolatedMode(threads=4, duration_ns=5.0),
            ),
        ),
    )

    pruned = solve_isolated_cp_sat(jobs, num_cores=4, workers=1, max_time_s=5.0)
    full = solve_isolated_cp_sat(
        jobs,
        num_cores=4,
        workers=1,
        max_time_s=5.0,
        prune_dominated=False,
    )

    assert pruned.objective_ns == full.objective_ns == 10
    assert pruned.retained_modes == 4
    assert full.retained_modes == 6


def test_evaluate_dag_and_oracle_comparison() -> None:
    pytest.importorskip("ortools")

    def isolated(routes: int, threads: int) -> float:
        assert routes == 12
        return {1: 10.0, 2: 6.0, 4: 5.0}[threads]

    tasks = [
        (0, 12, 0, 2, []),
        (1, 12, 0, 2, [0]),
        (2, 12, 2, 2, []),
    ]
    candidate_ns = evaluate_isolated_dag(tasks, isolated)
    jobs = build_isolated_jobs(
        [(0, 12), (1, 12), (2, 12)],
        [1, 2, 4],
        isolated,
        num_cores=4,
    )
    oracle = solve_isolated_cp_sat(jobs, num_cores=4, workers=1, max_time_s=5.0)
    comparison = compare_candidate_to_oracle(candidate_ns, oracle)

    assert candidate_ns == 12
    assert oracle.objective_ns == 10
    assert comparison.exact_regret == pytest.approx(0.2)
    assert comparison.regret_lower_bound == pytest.approx(0.2)
    assert comparison.regret_upper_bound == pytest.approx(0.2)
    assert comparison.bound_efficiency == pytest.approx(10.0 / 12.0)


def test_evaluate_dag_rejects_cycles() -> None:
    tasks = [
        (0, 12, 0, 1, [1]),
        (1, 12, 1, 1, [0]),
    ]

    with pytest.raises(ValueError, match="cycle"):
        evaluate_isolated_dag(tasks, lambda routes, threads: 10.0)


def test_cli_compares_strict_planner_with_profile(tmp_path: Path) -> None:
    pytest.importorskip("ortools")
    output = tmp_path / "oracle.json"

    assert (
        main(
            [
                str(PROFILE),
                "--routes",
                "2040,768,192,12",
                "--num-cores",
                "8",
                "--workers",
                "1",
                "--max-time-s",
                "5",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["model"] == "isolated_cp_sat_v1"
    assert report["oracle"]["status"] in {"OPTIMAL", "FEASIBLE"}
    assert report["oracle"]["objective_ns"] > 0
    assert report["oracle"]["time_quantum_ns"] == 1000
    assert report["planner"]["isolated_makespan_ns"] > 0
    assert report["planner"]["comparison"]["regret_upper_bound"] >= 0.0
