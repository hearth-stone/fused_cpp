from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNERS))

from executable_plan_neighborhood import (  # noqa: E402
    ADJACENT_SWAP,
    CROSS_LANE_RELOCATION,
    CROSS_LANE_SWAP,
    CROSS_LLC_RELOCATION,
    SAME_LANE_INSERTION,
    critical_expert_scores,
    enumerate_order_only_neighbors,
    placed_tasks,
    sample_order_only_neighborhood,
)
from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)


def _state() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=5,
        thread_cpu_ids=(10, 11, 20, 21, 22),
        lanes=(
            ExecutableLane(0, 1, (task(0, 20), task(1, 10), task(2, 5))),
            ExecutableLane(1, 1, (task(3, 15), task(4, 8))),
            ExecutableLane(2, 1, (task(5, 12), task(6, 6))),
            ExecutableLane(3, 2, (task(7, 9), task(8, 4))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 2),
            ExecutableLlcDomain("right", 2, 3),
        ),
        early_merge=False,
    )


def test_order_only_neighbors_preserve_experts_widths_and_execution_contract() -> None:
    state = _state()
    neighbors = list(enumerate_order_only_neighbors(state))

    assert {neighbor.operator for neighbor in neighbors} == {
        ADJACENT_SWAP,
        SAME_LANE_INSERTION,
        CROSS_LANE_RELOCATION,
        CROSS_LANE_SWAP,
        CROSS_LLC_RELOCATION,
    }
    expected_experts = sorted(task.expert_id for lane in state.lanes for task in lane.tasks)
    for neighbor in neighbors:
        assert neighbor.state.shape == state.shape
        assert neighbor.state.thread_cpu_ids == state.thread_cpu_ids
        assert neighbor.state.llc_domains == state.llc_domains
        assert neighbor.state.early_merge is False
        assert sorted(task.expert_id for lane in neighbor.state.lanes for task in lane.tasks) == expected_experts
        bridge = neighbor.state.to_bridge()
        assert bridge["execution_mode"] == "strict"
        assert bridge["task_threads"] == [task[3] for task in neighbor.state.to_planner_tasks()]


def test_expert_filter_limits_every_move_to_a_selected_expert() -> None:
    neighbors = list(enumerate_order_only_neighbors(_state(), expert_filter={4}))

    assert neighbors
    assert all(4 in neighbor.moved_experts for neighbor in neighbors)


def test_sampled_neighborhood_is_deterministic_and_globally_deduplicated() -> None:
    state = _state()
    first = sample_order_only_neighborhood(
        state,
        expert_filter={0, 1, 3, 4},
        per_operator=3,
        seed=17,
    )
    second = sample_order_only_neighborhood(
        state,
        expert_filter={0, 1, 3, 4},
        per_operator=3,
        seed=17,
    )

    first_hashes = [neighbor.state.canonical_hash() for neighbor in first.neighbors]
    second_hashes = [neighbor.state.canonical_hash() for neighbor in second.neighbors]
    assert first_hashes == second_hashes
    assert len(first_hashes) == len(set(first_hashes))
    assert state.canonical_hash() not in first_hashes
    assert first.proposed == first.unique + first.duplicates


def test_placed_tasks_preserve_lane_dependencies_and_cpu_placement() -> None:
    tasks = placed_tasks(_state())

    assert tasks[0] == (20, 1, (10,), [])
    assert tasks[1] == (10, 1, (10,), [0])
    assert tasks[3] == (15, 1, (11,), [])
    assert tasks[5] == (12, 1, (20,), [])
    assert tasks[7] == (9, 2, (21, 22), [])


def test_critical_scores_are_tail_and_dilation_weighted() -> None:
    explanation = {
        "makespan_ns": 100.0,
        "events": [
            {
                "start_ns": 0.0,
                "duration_ns": 20.0,
                "active_tasks": [0, 3],
                "phase_dilation": {"0": 1.0, "3": 2.0},
            },
            {
                "start_ns": 80.0,
                "duration_ns": 20.0,
                "active_tasks": [0],
                "phase_dilation": {"0": 1.0},
            },
        ],
    }

    scores = critical_expert_scores(_state(), explanation)

    assert scores[0] == 20.0 * 1.0 * 1.2 + 20.0 * 1.0 * 2.0
    assert scores[3] == 20.0 * 2.0 * 1.2
    assert scores[1] == 0.0
