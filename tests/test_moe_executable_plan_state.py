from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(ROOT / "src"), str(COST_MODEL), str(PLANNERS)]

from analytic_model import (  # noqa: E402
    AnalyticMachineCalibration,
    AnalyticMoeCostModel,
    CacheCalibration,
    LlcDomainCalibration,
    SaturatingServiceCurve,
)
from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutablePlanState,
)
from fused_cpp.moe import AsyncMoEPlanV2  # noqa: E402
from interval_planner import IntervalPlanner  # noqa: E402


def _curve(single: float, saturated: float, threads: int) -> SaturatingServiceCurve:
    return SaturatingServiceCurve(single, saturated, threads)


def _model() -> AnalyticMoeCostModel:
    calibration = AnalyticMachineCalibration(
        machine_id="executable-state-test",
        cores_per_rank=4,
        caches=CacheCalibration(
            l1d_bytes_per_core=64 * 1024,
            l2_bytes_per_core=256 * 1024,
            llc_bytes_per_rank=2 * 1024 * 1024,
        ),
        matrix_flops=_curve(100e9, 400e9, 4),
        gemm_core_flops=_curve(80e9, 320e9, 4),
        frontend_instructions=_curve(20e9, 80e9, 4),
        l1_bytes=_curve(100e9, 400e9, 4),
        l2_bytes=_curve(50e9, 200e9, 4),
        llc_bytes=_curve(25e9, 100e9, 4),
        dram_bytes=_curve(10e9, 40e9, 4),
        epilogue_elements=_curve(1e9, 4e9, 4),
        supported_widths=(1, 2, 4),
        rank_cpu_ids=(10, 11, 20, 21),
    )
    domain_curve = _curve(25e9, 50e9, 2)
    calibration = replace(
        calibration,
        llc_domains=(
            LlcDomainCalibration("left", (10, 11), 1024 * 1024, domain_curve),
            LlcDomainCalibration("right", (20, 21), 1024 * 1024, domain_curve),
        ),
    )
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=64,
        intermediate_size=32,
        global_experts=8,
        local_experts=8,
    )


def _planner_result() -> dict[str, object]:
    tasks = [
        (3, 24, 0, 2, []),
        (1, 8, 0, 2, [0]),
        (7, 12, 2, 1, []),
    ]
    bridge = {
        "plan_version": 2,
        "execution_mode": "strict",
        "num_threads": 4,
        "thread_cpu_ids": [10, 11, 20, 21],
        "task_expert_ids": [3, 1, 7],
        "task_core_begins": [0, 0, 2],
        "task_threads": [2, 2, 1],
        "task_dep_offsets": [0, 0, 1, 1],
        "task_deps": [0],
        "task_preferred_threads": [2, 2, 1],
        "task_min_threads": [2, 2, 1],
        "task_max_threads": [2, 2, 1],
        "task_allowed_thread_offsets": [0, 1, 2, 3],
        "task_allowed_threads": [2, 2, 1],
        "task_placement_modes": [0, 0, 0],
        "task_numa_nodes": [-1, -1, -1],
        "task_stage_ids": [0, 0, 0],
        "task_resize_points": [0, 0, 0],
        "task_range_granularities": [0, 0, 0],
        "task_w13_window_tiles": [4, 2, 1],
        "task_w2_window_tiles": [8, 4, 2],
        "early_merge": True,
    }
    return {
        "shape": (2, 1, 1),
        "execution_mode": "strict",
        "tail_pool_threads": None,
        "tasks": tasks,
        "bridge": bridge,
    }


def _event_score(model: AnalyticMoeCostModel, state: ExecutablePlanState) -> float:
    return model.dag_makespan_placed(
        [
            (
                routes,
                threads,
                state.thread_cpu_ids[core_begin : core_begin + threads],
                dependencies,
            )
            for _, routes, core_begin, threads, dependencies in state.to_planner_tasks()
        ]
    )


def test_planner_result_round_trip_preserves_tasks_bridge_hash_and_event_score() -> None:
    result = _planner_result()
    domains = (("left", (10, 11)), ("right", (20, 21)))

    state = ExecutablePlanState.from_planner_result(result, llc_domains=domains)
    repeated = ExecutablePlanState.from_planner_result(result, llc_domains=domains)

    assert state.shape == (2, 1, 1)
    assert state.task_count == 3
    assert state.lanes[-1].tasks == ()
    assert state.to_planner_tasks() == result["tasks"]
    assert state.to_bridge() == result["bridge"]
    assert state.canonical_hash() == repeated.canonical_hash()
    assert state.canonical_hash() != replace(state, early_merge=False).canonical_hash()
    swapped_lane = replace(
        state.lanes[0],
        tasks=tuple(reversed(state.lanes[0].tasks)),
    )
    assert state.canonical_hash() != replace(
        state,
        lanes=(swapped_lane, *state.lanes[1:]),
    ).canonical_hash()
    AsyncMoEPlanV2.from_dict(state.to_bridge())
    model = _model()
    source_score = model.dag_makespan_placed(
        [
            (
                routes,
                threads,
                state.thread_cpu_ids[core_begin : core_begin + threads],
                dependencies,
            )
            for _, routes, core_begin, threads, dependencies in result["tasks"]
        ]
    )
    assert _event_score(model, state) == pytest.approx(source_score)


def test_current_interval_planner_result_round_trips_without_score_change() -> None:
    model = _model()
    planner = IntervalPlanner(
        model,
        num_cores=4,
        cpu_ids=(10, 11, 20, 21),
        shapes=((2, 1, 1),),
        native_cold_planner=False,
    )
    result = planner.plan([(3, 24), (1, 8), (7, 12)], dynamic_tail_pool=False)

    state = ExecutablePlanState.from_planner_result(
        result,
        llc_domains=(("left", (10, 11)), ("right", (20, 21))),
    )

    assert state.to_planner_tasks() == result["tasks"]
    assert state.to_bridge() == result["bridge"]
    assert planner._score(state.to_planner_tasks()) == pytest.approx(
        planner._score(result["tasks"])
    )


def test_lane_domain_ids_record_existing_cross_domain_lane_without_rewriting_it() -> None:
    result = _planner_result()
    result["shape"] = (3, 1)
    bridge = dict(result["bridge"])
    bridge.update(
        {
            "task_core_begins": [0, 0, 3],
            "task_threads": [3, 3, 1],
            "task_preferred_threads": [3, 3, 1],
            "task_min_threads": [3, 3, 1],
            "task_max_threads": [3, 3, 1],
            "task_allowed_threads": [3, 3, 1],
        }
    )
    result["tasks"] = [
        (3, 24, 0, 3, []),
        (1, 8, 0, 3, [0]),
        (7, 12, 3, 1, []),
    ]
    result["bridge"] = bridge

    state = ExecutablePlanState.from_planner_result(
        result,
        llc_domains=(("left", (10, 11)), ("right", (20, 21))),
    )

    assert state.lane_domain_ids(0) == ("left", "right")
    assert state.to_bridge() == bridge


def test_search_state_rejects_duplicate_experts_and_non_partitioned_lanes() -> None:
    task = ExecutableExpertTask(1, 8)

    with pytest.raises(ValueError, match="one task per active expert"):
        ExecutablePlanState(
            2,
            (0, 1),
            (ExecutableLane(0, 2, (task, task)),),
        )
    with pytest.raises(ValueError, match="gap-free partition"):
        ExecutablePlanState(
            2,
            (0, 1),
            (ExecutableLane(1, 1, (task,)), ExecutableLane(0, 1)),
        )


def test_planner_result_rejects_noncanonical_lane_dependency_and_width_metadata() -> None:
    result = _planner_result()
    result["tasks"] = [
        (3, 24, 0, 2, []),
        (1, 8, 0, 2, []),
        (7, 12, 2, 1, []),
    ]
    bridge = dict(result["bridge"])
    bridge["task_dep_offsets"] = [0, 0, 0, 0]
    bridge["task_deps"] = []
    result["bridge"] = bridge

    with pytest.raises(ValueError, match="previous lane task"):
        ExecutablePlanState.from_planner_result(result)

    result = _planner_result()
    bridge = dict(result["bridge"])
    bridge["task_allowed_threads"] = [1, 2, 1]
    result["bridge"] = bridge
    with pytest.raises(ValueError, match="task_allowed_threads"):
        ExecutablePlanState.from_planner_result(result)


def test_planner_result_rejects_tail_pool_and_noncontiguous_llc_mapping() -> None:
    result = _planner_result()
    result["execution_mode"] = "tail_pool"
    result["tail_pool_threads"] = 1
    with pytest.raises(ValueError, match="only strict"):
        ExecutablePlanState.from_planner_result(result)

    result = _planner_result()
    with pytest.raises(ValueError, match="not contiguous"):
        ExecutablePlanState.from_planner_result(
            result,
            llc_domains=(("striped", (10, 20)), ("other", (11, 21))),
        )
