from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))
sys.path.insert(0, str(PLANNERS))

from resource_lower_bound import (  # noqa: E402
    CriticalChain,
    LowerBoundProblem,
    MoldableStage,
    ResourceCapacity,
    build_lower_bound_certificate,
    compute_lb0,
    main,
    mode,
    solve_mode_relaxed_lp,
)
from sve_fused_expert_lower_bound import (  # noqa: E402
    ServiceUpperBound,
    SveFusedExpertHardwareEnvelope,
    build_sve_fused_expert_lower_bound_problem,
)


def _tradeoff_problem(stage_count: int = 1) -> LowerBoundProblem:
    stages = tuple(
        MoldableStage(
            stage_id=f"stage-{stage_index}",
            modes=(
                mode("compute-light", 1, {"compute": 1.0, "memory": 9.0}, 2.0),
                mode("memory-light", 2, {"compute": 9.0, "memory": 1.0}, 1.0),
            ),
        )
        for stage_index in range(stage_count)
    )
    return LowerBoundProblem(
        resources=(
            ResourceCapacity("compute", 1.0),
            ResourceCapacity("memory", 1.0),
        ),
        stages=stages,
    )


def _integer_mode_optimum(problem: LowerBoundProblem) -> float:
    capacities = {resource.name: resource.units_per_second for resource in problem.resources}
    best = float("inf")
    for selected_modes in itertools.product(*(stage.modes for stage in problem.stages)):
        resource_loads = {
            resource: sum(
                next((demand.amount for demand in selected.demands if demand.resource == resource), 0.0)
                for selected in selected_modes
            )
            / capacity
            for resource, capacity in capacities.items()
        }
        chain_loads = {
            chain.name: sum(
                selected_modes[[stage.stage_id for stage in problem.stages].index(stage_id)].duration_lower_bound_s
                for stage_id in chain.stage_ids
            )
            for chain in problem.critical_chains
        }
        best = min(best, max((*resource_loads.values(), *chain_loads.values()), default=0.0))
    return best


def test_lb0_allows_independent_modes_but_lp_couples_them() -> None:
    problem = _tradeoff_problem()

    certificate = build_lower_bound_certificate(
        problem,
        max_iterations=40_000,
        relative_tolerance=2e-4,
    )

    assert certificate.lb0.lower_bound_s == pytest.approx(1.0)
    assert certificate.lb0.active_constraints == ("resource:compute", "resource:memory")
    assert certificate.mode_relaxed_lp.lower_bound_s == pytest.approx(5.0, rel=3e-4)
    assert certificate.mode_relaxed_lp.primal_upper_bound_s == pytest.approx(5.0, rel=3e-4)
    assert certificate.mode_relaxed_lp.converged
    assert certificate.mode_relaxed_lp.lower_bound_s <= _integer_mode_optimum(problem)
    assert _integer_mode_optimum(problem) == pytest.approx(9.0)

    mixture = certificate.mode_relaxed_lp.stage_mixtures[0]
    fractions = {entry.mode: entry.fraction for entry in mixture.modes}
    assert fractions == pytest.approx({"compute-light": 0.5, "memory-light": 0.5}, abs=2e-3)


def test_lp_certificate_remains_valid_when_iteration_budget_is_tiny() -> None:
    problem = _tradeoff_problem(stage_count=3)

    certificate = solve_mode_relaxed_lp(
        problem,
        max_iterations=1,
        minimum_iterations=1,
        relative_tolerance=0.0,
        solver="mirror",
    )

    assert not certificate.converged
    assert certificate.lower_bound_s <= certificate.primal_upper_bound_s
    assert certificate.lower_bound_s <= _integer_mode_optimum(problem)
    assert certificate.absolute_gap_s == pytest.approx(
        certificate.primal_upper_bound_s - certificate.lower_bound_s
    )


def test_critical_chain_is_a_first_class_constraint() -> None:
    problem = LowerBoundProblem(
        resources=(ResourceCapacity("compute", 100.0),),
        stages=(
            MoldableStage("expert-0:w13", (mode("1t", 1, {"compute": 10.0}, 3.0),)),
            MoldableStage("expert-0:w2", (mode("1t", 1, {"compute": 10.0}, 4.0),)),
            MoldableStage("expert-1:w13", (mode("1t", 1, {"compute": 10.0}, 2.0),)),
            MoldableStage("expert-1:w2", (mode("1t", 1, {"compute": 10.0}, 2.0),)),
        ),
        critical_chains=(
            CriticalChain("expert-0", ("expert-0:w13", "expert-0:w2")),
            CriticalChain("expert-1", ("expert-1:w13", "expert-1:w2")),
        ),
    )

    lb0 = compute_lb0(problem)
    terms = {term.constraint: term.lower_bound_s for term in lb0.terms}

    assert terms == pytest.approx(
        {
            "resource:compute": 0.4,
            "chain:expert-0": 7.0,
            "chain:expert-1": 4.0,
        }
    )
    assert lb0.lower_bound_s == pytest.approx(7.0)
    assert lb0.active_constraints == ("chain:expert-0",)


def test_empty_workload_has_zero_bound_and_a_certificate() -> None:
    problem = LowerBoundProblem(resources=(ResourceCapacity("compute", 1.0),), stages=())

    certificate = build_lower_bound_certificate(problem)

    assert certificate.lower_bound_s == 0.0
    assert certificate.mode_relaxed_lp.converged
    assert certificate.mode_relaxed_lp.relative_gap == 0.0
    assert certificate.to_dict()["lower_bound_s"] == 0.0


def test_problem_rejects_unknown_resources_and_chain_stages() -> None:
    stage = MoldableStage("w13", (mode("1t", 1, {"missing": 1.0}, 1.0),))
    with pytest.raises(ValueError, match="unknown resources"):
        LowerBoundProblem(resources=(ResourceCapacity("compute", 1.0),), stages=(stage,))

    valid_stage = MoldableStage("w13", (mode("1t", 1, {"compute": 1.0}, 1.0),))
    with pytest.raises(ValueError, match="unknown stages"):
        LowerBoundProblem(
            resources=(ResourceCapacity("compute", 1.0),),
            stages=(valid_stage,),
            critical_chains=(CriticalChain("expert", ("w13", "w2")),),
        )


def test_cli_writes_inspectable_certificate(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.json"
    output_path = tmp_path / "certificate.json"
    problem_path.write_text(
        json.dumps(
            {
                "resources": [
                    {"name": "compute", "units_per_second": 1.0},
                    {"name": "memory", "units_per_second": 1.0},
                ],
                "stages": [
                    {
                        "stage_id": "w13",
                        "modes": [
                            {
                                "name": "compute-light",
                                "threads": 1,
                                "demands": {"compute": 1.0, "memory": 9.0},
                                "duration_lower_bound_s": 2.0,
                            },
                            {
                                "name": "memory-light",
                                "threads": 2,
                                "demands": {"compute": 9.0, "memory": 1.0},
                                "duration_lower_bound_s": 1.0,
                            },
                        ],
                    }
                ],
                "critical_chains": [{"name": "expert-0", "stage_ids": ["w13"]}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(problem_path),
                "--output",
                str(output_path),
                "--max-iterations",
                "40000",
                "--relative-tolerance",
                "0.0002",
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["model"] == "moe_resource_lb0_mode_relaxed_lp_v1"
    assert report["problem"] == {"critical_chains": 1, "modes": 2, "resources": 2, "stages": 1}
    assert report["certificate"]["lb0"]["lower_bound_s"] == pytest.approx(1.0)
    assert report["certificate"]["mode_relaxed_lp"]["lower_bound_s"] == pytest.approx(5.0, rel=3e-4)
    weights = report["certificate"]["mode_relaxed_lp"]["constraint_weights"]
    assert sum(entry["numerator"] for entry in weights) == weights[0]["denominator"]


def test_sve_builder_emits_w13_w2_chains_and_exact_m_modes() -> None:
    envelope = SveFusedExpertHardwareEnvelope(
        num_cores=4,
        bfmmla=ServiceUpperBound("bfmmla_flops", 4e12, 1e12),
        key_instructions=ServiceUpperBound("key_instructions", 4e12, 1e12),
        l1_load_bytes=ServiceUpperBound("l1_load_bytes", 4e12, 1e12),
        epilogue_elements=ServiceUpperBound("epilogue_elements", 4e12, 1e12),
        dram_bytes=ServiceUpperBound("dram_bytes", 100e9),
    )

    problem = build_sve_fused_expert_lower_bound_problem(
        {3: 1, 7: 12, 9: 0},
        hidden_size=64,
        intermediate_size=32,
        widths=(1, 2, 4),
        n_tile=8,
        envelope=envelope,
        include_weight_dram=True,
    )

    assert [stage.stage_id for stage in problem.stages] == [
        "expert-3:w13",
        "expert-3:w2",
        "expert-7:w13",
        "expert-7:w2",
    ]
    assert [entry.name for entry in problem.stages[0].modes] == [
        "1t-minimum-work",
        "2t-minimum-work",
        "4t-minimum-work",
    ]
    assert problem.critical_chains == (
        CriticalChain("expert-3", ("expert-3:w13", "expert-3:w2")),
        CriticalChain("expert-7", ("expert-7:w13", "expert-7:w2")),
    )
    assert [resource.name for resource in problem.resources] == [
        "bfmmla_flops",
        "key_instructions",
        "l1_load_bytes",
        "epilogue_elements",
        "dram_bytes",
        "core_seconds",
    ]

    m1_w13 = problem.stages[0].modes[0]
    demands = {demand.resource: demand.amount for demand in m1_w13.demands}
    logical_useful_flops = 4 * 1 * 64 * 32
    assert demands["bfmmla_flops"] == 2 * logical_useful_flops
    assert demands["dram_bytes"] == 4 * 64 * 32
    assert demands["core_seconds"] > 0.0

    certificate = build_lower_bound_certificate(problem, max_iterations=5_000, relative_tolerance=2e-3)
    assert certificate.lower_bound_s > 0.0
    assert certificate.mode_relaxed_lp.lower_bound_s <= certificate.mode_relaxed_lp.primal_upper_bound_s


def test_sve_core_time_does_not_assume_all_n_lanes_match_the_busiest_lane() -> None:
    per_core_rate = 1e12
    envelope = SveFusedExpertHardwareEnvelope(
        num_cores=4,
        bfmmla=ServiceUpperBound("bfmmla_flops", 4e12, per_core_rate),
        key_instructions=ServiceUpperBound("key_instructions", 4e12, per_core_rate),
        l1_load_bytes=ServiceUpperBound("l1_load_bytes", 4e12, per_core_rate),
        epilogue_elements=ServiceUpperBound("epilogue_elements", 4e12, per_core_rate),
    )
    problem = build_sve_fused_expert_lower_bound_problem(
        [12],
        hidden_size=64,
        intermediate_size=24,
        widths=(4,),
        n_tile=8,
        envelope=envelope,
    )

    w13 = problem.stages[0].modes[0]
    demands = {demand.resource: demand.amount for demand in w13.demands}
    expected_core_seconds = max(
        demands[resource] / per_core_rate
        for resource in ("bfmmla_flops", "key_instructions", "l1_load_bytes", "epilogue_elements")
    )

    assert demands["core_seconds"] == pytest.approx(expected_core_seconds)
    assert demands["core_seconds"] < w13.threads * w13.duration_lower_bound_s
