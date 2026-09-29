from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(COST_MODEL), str(PLANNERS)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from interval_planner import IntervalPlanner  # noqa: E402

ANALYTIC_PROFILE = (
    COST_MODEL
    / "profiles"
    / "analytic_machine_amazon_ecs_v1_8c_sve_jit_hot_gemm_20260809.json"
)


def _native_quick_extension():
    try:
        return importlib.import_module("fused_cpp._moe_C")
    except ImportError:
        pytest.skip("fused_cpp._moe_C is not built")


def _assert_plan_equivalent(reference: dict, actual: dict) -> None:
    exact_fields = (
        "shape",
        "assignment_order",
        "execution_mode",
        "tail_pool_threads",
        "tail_pool_max_routes",
        "tail_pool_tasks",
        "tail_repartition_width",
        "tail_repartition_tasks",
        "tail_repartition_route_slices",
        "active_working_set_bytes",
        "resource_groups",
        "window_bytes_per_worker",
        "tasks",
        "bridge",
        "strict_candidates",
        "dynamic_candidates",
        "tail_repartition_candidates",
        "ranking",
    )
    for field in exact_fields:
        assert actual[field] == reference[field], field
    assert actual["makespan_ns"] == pytest.approx(reference["makespan_ns"], rel=1e-13)
    assert actual["uncertainty_ns"] == pytest.approx(
        reference["uncertainty_ns"],
        rel=1e-13,
    )


def test_native_analytical_quick_search_matches_python() -> None:
    _native_quick_extension()
    model = AnalyticMoeCostModel(
        ANALYTIC_PROFILE,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
    )
    reference = IntervalPlanner(model, num_cores=8, native_cold_planner=False)
    reference._native_quick_planner = None
    experts = [
        (0, 2040),
        (1, 768),
        (2, 192),
        (3, 48),
        (4, 12),
        (5, 8),
        (6, 4),
        (7, 1),
    ]

    expected = reference.plan_quick(experts)
    default = IntervalPlanner(model, num_cores=8, native_cold_planner=False)
    default_result = default.plan_quick(experts)
    _assert_plan_equivalent(expected, default_result)
    assert default_result["planner_workers"] == 1
    for workers in (1, 2, 4):
        actual = IntervalPlanner(
            model,
            num_cores=8,
            native_cold_planner=False,
            planner_threads=workers,
        )
        assert actual._native_quick_planner is not None
        result = actual.plan_quick(experts)

        _assert_plan_equivalent(expected, result)
        assert result["planner_backend"] == "cpp_quick"
        assert result["planner_workers"] == workers


def test_native_quick_assignment_preserves_rounded_score_tie_break() -> None:
    extension = _native_quick_extension()
    planner = extension.NativeQuickPlanner(
        2,
        [[1, 1]],
        1,
        [(1, 1)],
        0.0,
        1,
    )

    result = planner.assign([0, 1, 2], [4, 3, 2], [1, 1], [2.0, 1.0, 1.0e20])

    assert [task[0] for task in result["tasks"]] == [0, 2, 1]


def test_native_shared_quick_search_matches_python() -> None:
    extension = _native_quick_extension()
    if not hasattr(extension.NativeQuickPlanner, "plan_shared"):
        pytest.skip("fused_cpp._moe_C was built without shared quick planning")
    model = AnalyticMoeCostModel(
        ANALYTIC_PROFILE,
        hidden_size=64,
        intermediate_size=32,
        global_experts=9,
        local_experts=9,
    )
    experts = [(expert, routes) for expert, routes in enumerate((12, 11, 10, 9, 8, 7, 6, 5, 128))]
    reference = IntervalPlanner(model, num_cores=8, native_cold_planner=False)
    reference._native_quick_planner = None
    expected = reference.plan_quick_with_shared(experts, shared_expert_id=8)
    actual = IntervalPlanner(model, num_cores=8, native_cold_planner=False).plan_quick_with_shared(
        experts,
        shared_expert_id=8,
    )

    for field in ("shape", "tasks", "bridge", "shared_expert_id", "shared_width", "routed_width"):
        assert actual[field] == expected[field], field
    assert actual["makespan_ns"] == pytest.approx(expected["makespan_ns"], rel=1e-13)
    assert actual["planner_backend"] == "cpp_shared_quick"
