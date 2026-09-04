from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNERS))

from executable_plan_neighborhood import (  # noqa: E402
    ADJACENT_WIDTH_MIGRATION,
    ADJACENT_SWAP,
    CROSS_LANE_RELOCATION,
    CROSS_LANE_SWAP,
    CROSS_LLC_RELOCATION,
    LANE_MERGE,
    LANE_SPLIT,
    ORDER_ONLY_OPERATORS,
    SAME_LANE_INSERTION,
    WIDTH_ONLY_OPERATORS,
    ExecutablePlanEvaluator,
    ExecutablePlanScore,
    critical_expert_scores,
    enumerate_order_only_neighbors,
    enumerate_width_only_neighbors,
    is_resolvable_improvement,
    placed_tasks,
    sample_combined_neighborhood,
    sample_order_only_neighborhood,
    sample_width_only_neighborhood,
    score_executable_plan,
    summarize_executable_plan_pair,
    summarize_placed_event_context,
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


def test_plan_pair_summary_serializes_affected_lane_sequences_and_loads() -> None:
    before = _state()
    task = ExecutableExpertTask
    after = ExecutablePlanState(
        num_threads=before.num_threads,
        thread_cpu_ids=before.thread_cpu_ids,
        lanes=(
            ExecutableLane(0, 1, (task(1, 10), task(2, 5))),
            ExecutableLane(1, 1, (task(3, 15), task(4, 8), task(0, 20))),
            before.lanes[2],
            before.lanes[3],
        ),
        llc_domains=before.llc_domains,
        early_merge=before.early_merge,
    )

    summary = summarize_executable_plan_pair(_ScreenModel(), before, after)

    assert summary["changed_expert_ids"] == [0, 1, 2]
    assert summary["affected_expert_ids"] == [0, 1, 2, 3, 4]
    assert summary["affected_route_counts"] == [
        {"expert_id": 0, "routes": 20},
        {"expert_id": 1, "routes": 10},
        {"expert_id": 2, "routes": 5},
        {"expert_id": 3, "routes": 15},
        {"expert_id": 4, "routes": 8},
    ]
    assert [lane["lane_index"] for lane in summary["before_affected_lanes"]] == [0, 1]
    assert [lane["lane_index"] for lane in summary["after_affected_lanes"]] == [0, 1]
    assert [
        task_row["expert_id"]
        for task_row in summary["before_affected_lanes"][0]["tasks"]
    ] == [0, 1, 2]
    assert [
        task_row["expert_id"]
        for task_row in summary["after_affected_lanes"][1]["tasks"]
    ] == [3, 4, 0]
    assert summary["before_affected_lanes"][0]["isolated_load_ns"] == 350.0
    assert summary["after_affected_lanes"][1]["isolated_load_ns"] == 430.0
    assert summary["critical_isolated_lane_switched"] is True


def test_placed_event_context_summarizes_head_tail_pressure_and_transitions() -> None:
    explanation = {
        "makespan_ns": 100.0,
        "task_finish_ns": [10.0, 20.0, 100.0, 15.0, 30.0, 40.0, 50.0, 70.0, 90.0],
        "events": [
            {
                "start_ns": 0.0,
                "duration_ns": 10.0,
                "active_tasks": [2, 3],
                "phase_kinds": {"2": "cold_b", "3": "hot"},
                "phase_dilation": {"2": 2.0, "3": 1.0},
                "team_pressure_dilation": {"2": 1.5, "3": 1.0},
                "resources": {"dram": {"dilation": 2.0}},
            },
            {
                "start_ns": 10.0,
                "duration_ns": 15.0,
                "active_tasks": [2],
                "phase_kinds": {"2": "cold_b"},
                "phase_dilation": {"2": 1.2},
                "team_pressure_dilation": {"2": 1.1},
                "resources": {},
            },
            {
                "start_ns": 80.0,
                "duration_ns": 20.0,
                "active_tasks": [2, 7],
                "phase_kinds": {"2": "hot", "7": "cold_a"},
                "phase_dilation": {"2": 1.4, "7": 1.1},
                "team_pressure_dilation": {"2": 1.3, "7": 1.0},
                "resources": {"llc": {"dilation": 1.4}},
            },
        ],
    }

    summary = summarize_placed_event_context(
        _state(),
        explanation,
        affected_expert_ids={2},
    )

    assert summary["affected_task_count"] == 1
    assert summary["affected_active_ns"] == 45.0
    assert summary["affected_solo_ns"] == 15.0
    assert summary["affected_head_ns"] == 20.0
    assert summary["affected_tail_ns"] == 20.0
    assert summary["mean_peer_threads_while_affected"] == 50.0 / 45.0
    assert summary["cohort_transition_count"] == 3
    assert summary["cohort_transition_ns"] == {
        "1x1T->1x1T+1x2T": 20.0,
        "2x1T->1x1T": 15.0,
        "idle->2x1T": 10.0,
    }
    assert summary["critical_lane"]["lane_index"] == 0
    assert summary["critical_task"]["expert_id"] == 2
    assert summary["critical_task"]["affected"] is True


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


def test_robust_score_exposes_heavy_serial_lane() -> None:
    class _ScoreModel:
        relative_error = 0.15

        @staticmethod
        def T_iso(routes: int, threads: int) -> float:
            return routes * 10.0 / threads

        @staticmethod
        def dag_makespan_placed(tasks) -> float:
            assert tasks
            return 300.0

    score = score_executable_plan(_ScoreModel(), _state())

    assert score.event_ns == 300.0
    assert score.lane_guard_ns == (20 + 10 + 5) * 10.0 * 1.15
    assert score.robust_ns == score.lane_guard_ns


def test_resolvable_improvement_requires_robust_margin() -> None:
    incumbent = ExecutablePlanScore(event_ns=100.0, lane_guard_ns=90.0, robust_ns=100.0)

    assert not is_resolvable_improvement(
        incumbent,
        ExecutablePlanScore(99.0, 90.0, 99.0),
        minimum_gain_fraction=0.02,
    )
    assert is_resolvable_improvement(
        incumbent,
        ExecutablePlanScore(95.0, 90.0, 95.0),
        minimum_gain_fraction=0.02,
    )


def _isolated_cost(routes: int, threads: int) -> float:
    return routes * 100.0 / threads


def _windows(routes: int, threads: int) -> tuple[int, int]:
    return (routes + threads, routes + 2 * threads)


def test_width_neighbors_preserve_domain_topology_and_refresh_windows() -> None:
    state = _state()
    neighbors = list(
        enumerate_width_only_neighbors(
            state,
            allowed_widths=(1, 2, 4),
            isolated_cost=_isolated_cost,
            window_selector=_windows,
        )
    )

    assert {neighbor.operator for neighbor in neighbors} == set(WIDTH_ONLY_OPERATORS)
    expected_experts = sorted(task.expert_id for lane in state.lanes for task in lane.tasks)
    for neighbor in neighbors:
        assert neighbor.state.thread_cpu_ids == state.thread_cpu_ids
        assert neighbor.state.llc_domains == state.llc_domains
        assert neighbor.state.early_merge is False
        assert sum(neighbor.state.shape) == state.num_threads
        assert sorted(task.expert_id for lane in neighbor.state.lanes for task in lane.tasks) == expected_experts
        assert all(len(neighbor.state.lane_domain_ids(index)) == 1 for index in range(len(neighbor.state.lanes)))
        for lane in neighbor.state.lanes:
            for task in lane.tasks:
                if task.expert_id in neighbor.moved_experts:
                    assert (task.w13_window_tiles, task.w2_window_tiles) == _windows(task.routes, lane.threads)
        bridge = neighbor.state.to_bridge()
        assert bridge["execution_mode"] == "strict"
        assert bridge["task_threads"] == [task[3] for task in neighbor.state.to_planner_tasks()]


def test_width_neighbors_have_expected_split_merge_and_migration_shapes() -> None:
    neighbors = list(
        enumerate_width_only_neighbors(
            _state(),
            allowed_widths=(1, 2, 4),
            isolated_cost=_isolated_cost,
            window_selector=_windows,
        )
    )

    split_shapes = {neighbor.state.shape for neighbor in neighbors if neighbor.operator == LANE_SPLIT}
    merge_shapes = {neighbor.state.shape for neighbor in neighbors if neighbor.operator == LANE_MERGE}
    migrations = [neighbor for neighbor in neighbors if neighbor.operator == ADJACENT_WIDTH_MIGRATION]
    assert split_shapes == {(1, 1, 1, 1, 1)}
    assert merge_shapes == {(2, 1, 2)}
    assert migrations
    assert all(neighbor.state.shape == _state().shape for neighbor in migrations)


def test_width_expert_filter_and_sampling_are_deterministic() -> None:
    state = _state()
    kwargs = {
        "allowed_widths": (1, 2, 4),
        "isolated_cost": _isolated_cost,
        "window_selector": _windows,
        "expert_filter": {7},
        "per_operator": 3,
        "seed": 19,
    }
    first = sample_width_only_neighborhood(state, **kwargs)
    second = sample_width_only_neighborhood(state, **kwargs)

    first_hashes = [neighbor.state.canonical_hash() for neighbor in first.neighbors]
    assert first_hashes == [neighbor.state.canonical_hash() for neighbor in second.neighbors]
    assert len(first_hashes) == len(set(first_hashes))
    assert all(7 in neighbor.moved_experts for neighbor in first.neighbors)
    assert first.proposed == first.unique + first.duplicates


def test_combined_sampling_contains_both_operator_families() -> None:
    sampled = sample_combined_neighborhood(
        _state(),
        allowed_widths=(1, 2, 4),
        isolated_cost=_isolated_cost,
        window_selector=_windows,
        expert_filter=None,
        per_operator=2,
        seed=23,
    )

    sampled_operators = {neighbor.operator for neighbor in sampled.neighbors}
    assert sampled_operators & set(ORDER_ONLY_OPERATORS)
    assert sampled_operators & set(WIDTH_ONLY_OPERATORS)
    hashes = [neighbor.state.canonical_hash() for neighbor in sampled.neighbors]
    assert len(hashes) == len(set(hashes))


def test_width_neighbors_preserve_but_never_modify_cross_domain_lanes() -> None:
    task = ExecutableExpertTask
    state = ExecutablePlanState(
        num_threads=4,
        thread_cpu_ids=(10, 11, 20, 21),
        lanes=(
            ExecutableLane(0, 1, (task(0, 8),)),
            ExecutableLane(1, 2, (task(1, 16), task(2, 4))),
            ExecutableLane(3, 1, (task(3, 8),)),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 2),
            ExecutableLlcDomain("right", 2, 2),
        ),
    )

    neighbors = list(
        enumerate_width_only_neighbors(
            state,
            allowed_widths=(1, 2, 4),
            isolated_cost=_isolated_cost,
            window_selector=_windows,
        )
    )

    assert not neighbors


class _ScreenPressure:
    @staticmethod
    def isolated_scale(threads: int) -> float:
        del threads
        return 1.0

    @staticmethod
    def full_cohort_scale(threads: int) -> float:
        del threads
        return 1.0


class _ScreenCorrection:
    @staticmethod
    def scale(threads: int, peer_fraction: float) -> float:
        del threads, peer_fraction
        return 1.0


class _ScreenModel:
    relative_error = 0.1
    call_setup_ns = 5.0
    calibration = SimpleNamespace(
        wide_team_pressure=_ScreenPressure(),
        narrow_team_contention_correction=_ScreenCorrection(),
    )

    @staticmethod
    def T_iso(routes: int, threads: int) -> float:
        return routes * 10.0 / threads

    @staticmethod
    def predict_expert(routes: int, threads: int):
        return SimpleNamespace(
            phases=(SimpleNamespace(base_ns=routes * 10.0 / threads, kind="cold_b"),)
        )

    @staticmethod
    def dag_makespan_placed(tasks) -> float:
        return 100.0 + len(tasks)


def test_two_level_evaluator_caches_exact_states_and_unchanged_lane_loads() -> None:
    state = _state()
    evaluator = ExecutablePlanEvaluator(_ScreenModel())

    first_exact = evaluator.exact(state)
    second_exact = evaluator.exact(state)
    first_screen = evaluator.screen(state)
    lane_misses = evaluator.lane_cache_misses
    neighbor = next(iter(enumerate_order_only_neighbors(state)))
    neighbor_screen = evaluator.screen(neighbor.state)
    repeated_screen = evaluator.screen(neighbor.state)

    assert first_exact == second_exact
    assert evaluator.exact_calls == 1
    assert evaluator.exact_cache_hits == 1
    assert first_screen.lower_bound_ns <= first_screen.priority_ns
    assert neighbor_screen == repeated_screen
    assert evaluator.screen_calls == 2
    assert evaluator.screen_cache_hits == 1
    assert evaluator.lane_cache_misses == lane_misses
    assert evaluator.lane_cache_hits >= len(state.lanes)


def test_screening_bounds_include_lane_domain_and_rank_capacity() -> None:
    evaluator = ExecutablePlanEvaluator(_ScreenModel())

    score = evaluator.screen(_state())

    assert score.lane_bound_ns == 350.0
    assert score.rank_bound_ns == 178.0
    assert score.domain_bound_ns == 290.0
    assert score.lower_bound_ns == 350.0
