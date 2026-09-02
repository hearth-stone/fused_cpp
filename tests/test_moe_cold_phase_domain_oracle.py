from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNERS))

from cold_phase_cp_sat_oracle import (  # noqa: E402
    ColdPhase,
    ColdPhaseAssignment,
    ColdPhaseJob,
    ColdPhaseMode,
    PhaseAssignment,
)
from cold_phase_domain_oracle import (  # noqa: E402
    DomainAssignment,
    DomainCandidate,
    LlcDomain,
    compare_strict_greedy_union,
    materialize_domain_candidate,
    solve_fixed_strict_greedy_cp_sat,
    solve_cold_phase_domain_shortlist,
)


def _jobs() -> tuple[ColdPhaseJob, ...]:
    return tuple(
        ColdPhaseJob(
            expert_id,
            1,
            (
                ColdPhaseMode(1, (ColdPhase("cold", 2.0, 1),)),
                ColdPhaseMode(2, (ColdPhase("cold", 1.0, 1),)),
            ),
        )
        for expert_id in range(4)
    )


def _solve(**overrides):
    kwargs = {
        "num_cores": 4,
        "domains": (LlcDomain("left", 0, 2), LlcDomain("right", 2, 2)),
        "dram_bandwidth_gbps": 4.0,
        "bandwidth_quantum_gbps": 1.0,
        "time_quantum_ns": 1,
        "workers": 1,
        "max_time_s": 5.0,
        "subsequent_time_s": 1.0,
        "solution_limit": 5,
    }
    kwargs.update(overrides)
    return solve_cold_phase_domain_shortlist(_jobs(), **kwargs)


def test_domain_shortlist_reports_root_bound_and_diverse_assignments() -> None:
    pytest.importorskip("ortools")

    result = _solve()

    assert result.status == "OPTIMAL"
    assert result.optimal
    assert result.root_objective_ns == 2
    assert result.root_best_bound_ns == 2
    assert result.root_relative_gap == 0
    assert len(result.candidates) == 5
    signatures = [candidate.schedule_signature() for candidate in result.candidates]
    assert len(signatures) == len(set(signatures))
    assert all(sum(candidate.domain_histogram().values()) == 4 for candidate in result.candidates)


def test_domain_shortlist_uses_complete_incumbent_hint() -> None:
    pytest.importorskip("ortools")
    hints = tuple(
        DomainAssignment(
            "left" if expert_id < 2 else "right",
            ColdPhaseAssignment(
                expert_id,
                1,
                2,
                expert_id % 2,
                1,
                0,
                expert_id % 2 + 1,
                (
                    PhaseAssignment(
                        "cold",
                        expert_id % 2,
                        1,
                        expert_id % 2 + 1,
                        1,
                        1.0,
                    ),
                ),
            ),
        )
        for expert_id in range(4)
    )

    result = _solve(initial_assignments=hints, solution_limit=1)

    assert result.root_objective_ns == 2
    assert result.candidates[0].mode_histogram() == {2: 4}


def test_domain_shortlist_rejects_nonpartitioned_domains() -> None:
    with pytest.raises(ValueError, match="contiguous partition"):
        _solve(domains=(LlcDomain("left", 0, 2), LlcDomain("right", 3, 1)))


def test_domain_shortlist_ignores_widths_that_do_not_fit_any_domain_for_horizon() -> None:
    pytest.importorskip("ortools")
    jobs = (
        ColdPhaseJob(
            0,
            1,
            (
                ColdPhaseMode(2, (ColdPhase("fits", 10.0),)),
                ColdPhaseMode(4, (ColdPhase("too-wide", 1.0),)),
            ),
        ),
    )

    result = solve_cold_phase_domain_shortlist(
        jobs,
        num_cores=4,
        domains=(LlcDomain("left", 0, 2), LlcDomain("right", 2, 2)),
        dram_bandwidth_gbps=1.0,
        bandwidth_quantum_gbps=1.0,
        time_quantum_ns=1,
        workers=1,
        max_time_s=5.0,
        subsequent_time_s=1.0,
        solution_limit=1,
    )

    assert result.root_objective_ns == 10
    assert result.candidates[0].mode_histogram() == {2: 1}


def test_domain_candidate_lowering_keeps_teams_inside_domains() -> None:
    pytest.importorskip("ortools")
    result = _solve(solution_limit=1)

    placement = materialize_domain_candidate(
        result.candidates[0],
        domains=result.domains,
        workers=1,
    )

    assert placement.status == "OPTIMAL"
    assert len(placement.tasks) == 4
    for task in placement.tasks:
        assert (0 <= task.core_begin < task.core_begin + task.threads <= 2) or (
            2 <= task.core_begin < task.core_begin + task.threads <= 4
        )
        assert all(dependency < placement.tasks.index(task) for dependency in task.dependencies)


def test_domain_candidate_lowering_may_delay_fluid_starts_to_make_contiguous_teams() -> None:
    pytest.importorskip("ortools")
    candidate = DomainCandidate(
        10,
        tuple(
            DomainAssignment(
                "only",
                ColdPhaseAssignment(
                    expert_id,
                    1,
                    2,
                    0,
                    10,
                    0,
                    10,
                    (),
                ),
            )
            for expert_id in range(2)
        ),
    )

    placement = materialize_domain_candidate(
        candidate,
        domains=(LlcDomain("only", 0, 2),),
        workers=1,
    )

    assert placement.status == "OPTIMAL_DELAYED"
    assert [task.modeled_end_ns for task in placement.tasks] == [10, 20]
    assert placement.tasks[1].dependencies == (0,)


def test_fixed_strict_greedy_branch_preserves_width_order_and_union_bound() -> None:
    pytest.importorskip("ortools")
    tasks = (
        (0, 1, 0, 2, ()),
        (2, 1, 2, 2, ()),
        (1, 1, 0, 2, (0,)),
        (3, 1, 2, 2, (1,)),
    )
    greedy = solve_fixed_strict_greedy_cp_sat(
        _jobs(),
        tasks,
        num_cores=4,
        dram_bandwidth_gbps=4.0,
        bandwidth_quantum_gbps=1.0,
        time_quantum_ns=1,
        workers=1,
        max_time_s=5.0,
    )
    sat = _solve(solution_limit=1)
    union = compare_strict_greedy_union(greedy, sat)

    assert greedy.status == "OPTIMAL"
    assert greedy.objective_ns == 2
    assert greedy.mode_histogram() == {2: 4}
    assert union.selected_branch == "domain_sat"
    assert union.objective_ns == 2
    assert union.best_bound_ns == 2
    assert union.relative_gap == 0


def test_fixed_strict_greedy_branch_rejects_unordered_core_overlap() -> None:
    tasks = (
        (0, 1, 0, 2, ()),
        (1, 1, 1, 2, ()),
        (2, 1, 2, 1, ()),
        (3, 1, 3, 1, ()),
    )

    with pytest.raises(ValueError, match="overlapping strict greedy placements"):
        solve_fixed_strict_greedy_cp_sat(
            _jobs(),
            tasks,
            num_cores=4,
            dram_bandwidth_gbps=4.0,
        )
