from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNERS))

from cold_phase_cp_sat_oracle import (  # noqa: E402
    ColdPhase,
    ColdPhaseJob,
    ColdPhaseMode,
    build_cold_phase_jobs,
    compare_cold_phase_oracles,
    main,
    solve_cold_phase_cp_sat,
)


PROFILE = (
    ROOT
    / "cpu_moe_schedule_optimization"
    / "cost_model"
    / "profiles"
    / "contention_async_amazon_ecs_8c_standalone_sve_F512_E8_splitw13_xbyak_exactm_v2_20260727.json"
)


class _FakePhaseModel:
    def T_iso(self, routes: int, threads: int) -> float:
        del threads
        return 30.0 if routes <= 12 else 90.0

    def task_stage_phases(self, stage: str, routes: int, threads: int):
        del routes, threads
        if stage == "w13":
            return ((2.0, 4), (2.0, 4))
        if stage == "w2":
            return ((2.0, 4),)
        raise ValueError(stage)


def _mixed_jobs() -> tuple[ColdPhaseJob, ...]:
    big = ColdPhaseJob(
        expert_id=0,
        routes=120,
        modes=(
            ColdPhaseMode(
                threads=2,
                phases=(
                    ColdPhase("big:cold", 2.0, 2),
                    ColdPhase("big:steady", 12.0),
                ),
            ),
            ColdPhaseMode(
                threads=4,
                phases=(
                    ColdPhase("big:cold", 2.0, 2),
                    ColdPhase("big:steady", 8.0),
                ),
            ),
        ),
    )
    shorts = tuple(
        ColdPhaseJob(
            expert_id=expert_id,
            routes=1,
            modes=(
                ColdPhaseMode(
                    threads=1,
                    phases=(ColdPhase("short:cold", 2.0, 2),),
                ),
            ),
        )
        for expert_id in range(1, 5)
    )
    return (big, *shorts)


def test_build_cold_phase_jobs_preserves_isolated_time_and_weight_bytes() -> None:
    jobs = build_cold_phase_jobs(
        [(3, 24)],
        [1],
        _FakePhaseModel(),
        num_cores=4,
        phase_granularity="range",
    )

    mode = jobs[0].modes[0]
    assert mode.duration_ns == pytest.approx(90.0)
    assert sum(phase.duration_ns for phase in mode.phases if phase.is_cold) == pytest.approx(30.0)
    assert sum(phase.cold_weight_bytes for phase in mode.phases) == 12
    assert [phase.name for phase in mode.phases] == [
        "w13:0:cold",
        "w13:0:steady",
        "w13:1:cold",
        "w13:1:steady",
        "w2:0:cold",
        "w2:0:steady",
    ]

    aggregate = build_cold_phase_jobs(
        [(3, 24)],
        [1],
        _FakePhaseModel(),
        num_cores=4,
        phase_granularity="expert",
    )[0].modes[0]
    assert aggregate.duration_ns == pytest.approx(90.0)
    assert aggregate.phases == (
        ColdPhase("expert:cold", 30.0, 12),
        ColdPhase("expert:steady", 60.0),
    )


def test_cold_phase_oracle_uses_narrow_big_expert_to_overlap_short_jobs() -> None:
    pytest.importorskip("ortools")

    result = solve_cold_phase_cp_sat(
        _mixed_jobs(),
        num_cores=4,
        dram_bandwidth_gbps=1.0,
        bandwidth_quantum_gbps=1.0,
        time_quantum_ns=1,
        workers=1,
        max_time_s=5.0,
    )

    assert result.status == "OPTIMAL"
    assert result.objective_ns == 14
    assert result.mode_histogram() == {1: 4, 2: 1}
    for timestamp in range(result.objective_ns):
        active_threads = sum(
            assignment.threads
            for assignment in result.assignments
            if assignment.start_ns <= timestamp < assignment.end_ns
        )
        reserved_bandwidth = sum(
            phase.reserved_dram_gbps
            for assignment in result.assignments
            for phase in assignment.phases
            if phase.start_ns <= timestamp < phase.end_ns
        )
        assert active_threads <= 4
        assert reserved_bandwidth <= 1.0


def test_cold_phase_oracle_comparison_bounds_exact_gain() -> None:
    pytest.importorskip("ortools")
    mixed_jobs = _mixed_jobs()
    fixed_jobs = (
        ColdPhaseJob(
            expert_id=mixed_jobs[0].expert_id,
            routes=mixed_jobs[0].routes,
            modes=(mixed_jobs[0].modes[1],),
        ),
        *mixed_jobs[1:],
    )
    kwargs = {
        "num_cores": 4,
        "dram_bandwidth_gbps": 1.0,
        "bandwidth_quantum_gbps": 1.0,
        "time_quantum_ns": 1,
        "workers": 1,
        "max_time_s": 5.0,
    }

    fixed = solve_cold_phase_cp_sat(fixed_jobs, **kwargs)
    mixed = solve_cold_phase_cp_sat(
        mixed_jobs,
        initial_assignments=fixed.assignments,
        **kwargs,
    )
    comparison = compare_cold_phase_oracles(fixed, mixed)

    assert fixed.objective_ns == 18
    assert mixed.objective_ns == 14
    assert comparison.exact_gain == pytest.approx(18.0 / 14.0 - 1.0)
    assert comparison.gain_lower_bound == pytest.approx(comparison.exact_gain)
    assert comparison.gain_upper_bound == pytest.approx(comparison.exact_gain)


def test_cold_phase_oracle_applies_single_phase_bandwidth_floor() -> None:
    pytest.importorskip("ortools")
    jobs = (
        ColdPhaseJob(
            expert_id=0,
            routes=1,
            modes=(
                ColdPhaseMode(
                    threads=1,
                    phases=(ColdPhase("cold", 1.0, 4),),
                ),
            ),
        ),
    )

    result = solve_cold_phase_cp_sat(
        jobs,
        num_cores=1,
        dram_bandwidth_gbps=1.0,
        bandwidth_quantum_gbps=1.0,
        time_quantum_ns=1,
        workers=1,
        max_time_s=5.0,
    )

    assert result.objective_ns == 4
    assert result.bandwidth_floor_added_ns == 3


def test_cli_compares_fixed_and_mixed_oracles_with_profile(tmp_path: Path) -> None:
    pytest.importorskip("ortools")
    output = tmp_path / "cold-phase-oracle.json"

    assert (
        main(
            [
                str(PROFILE),
                "--routes",
                "12,12",
                "--num-cores",
                "8",
                "--widths",
                "1,2,4",
                "--baseline-width",
                "4",
                "--dram-bandwidth-gbps",
                "100",
                "--max-time-s",
                "5",
                "--workers",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["model"] == "cold_phase_cp_sat_v1"
    assert report["current_strict_plan"]["baseline_shape"] == [4, 4]
    assert report["fixed_baseline_oracle"]["objective_ns"] > 0
    assert report["mixed_oracle"]["objective_ns"] > 0
    assert report["comparison"]["gain_upper_bound"] is not None
