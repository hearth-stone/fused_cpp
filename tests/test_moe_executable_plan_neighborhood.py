from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    TEMPLATE_LNS_REPARTITION,
    WIDTH_ONLY_OPERATORS,
    ExecutablePlanEvaluator,
    ExecutablePlanScore,
    _beam_assign_tasks,
    _retarget_task,
    critical_expert_scores,
    enumerate_order_only_neighbors,
    enumerate_template_lns_neighbors,
    enumerate_width_only_neighbors,
    is_resolvable_improvement,
    placed_tasks,
    sample_combined_neighborhood,
    sample_order_only_neighborhood,
    sample_template_lns_neighborhood,
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


def _sampled_hashes_by_operator(sampled, operators: tuple[str, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    cursor = 0
    for operator in operators:
        count = int(sampled.sampled_by_operator[operator])
        grouped[operator] = [
            neighbor.state.canonical_hash() for neighbor in sampled.neighbors[cursor : cursor + count]
        ]
        cursor += count
    assert cursor == len(sampled.neighbors)
    return grouped


def test_per_operator_sample_is_prefix_of_larger_budget() -> None:
    state = _state()
    kwargs = {
        "allowed_widths": (1, 2, 4),
        "isolated_cost": _isolated_cost,
        "window_selector": _windows,
        "expert_filter": None,
        "seed": 41,
    }
    small = sample_combined_neighborhood(state, per_operator=1, **kwargs)
    large = sample_combined_neighborhood(state, per_operator=4, **kwargs)
    operators = (*ORDER_ONLY_OPERATORS, *WIDTH_ONLY_OPERATORS)
    small_by_operator = _sampled_hashes_by_operator(small, operators)
    large_by_operator = _sampled_hashes_by_operator(large, operators)
    for operator in operators:
        small_keys = small_by_operator[operator]
        assert large_by_operator[operator][: len(small_keys)] == small_keys


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


def _template_lns_state() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=12,
        thread_cpu_ids=tuple(range(100, 112)),
        lanes=(
            ExecutableLane(0, 4, (task(0, 80), task(1, 32), task(2, 8))),
            ExecutableLane(4, 4, (task(3, 64), task(4, 24), task(5, 12), task(6, 4))),
            ExecutableLane(8, 2, (task(7, 20), task(8, 6))),
            ExecutableLane(10, 2, (task(9, 18), task(10, 5))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 6),
            ExecutableLlcDomain("right", 6, 6),
        ),
        early_merge=True,
    )


def _reference_beam_assign_tasks(tasks, lane_widths, isolated_cost, window_selector, critical_expert_ids, *, repair_beam_width):
    if repair_beam_width <= 0:
        raise ValueError("repair_beam_width must be positive")
    if len(lane_widths) > len(tasks):
        return ()
    order_policies = 4
    placement_beam_width = max(1, (repair_beam_width + order_policies - 1) // order_policies)
    task_costs = {
        (task.expert_id, width): float(isolated_cost(task.routes, width))
        for task in tasks
        for width in lane_widths
    }
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            -max(task_costs[(task.expert_id, width)] for width in lane_widths),
            task.expert_id,
        ),
    )
    empty_assignment = tuple(() for _ in lane_widths)
    beam = [(empty_assignment, tuple(0.0 for _ in lane_widths))]
    for task_index, task in enumerate(ordered_tasks):
        remaining_tasks = len(ordered_tasks) - task_index - 1
        next_beam = {}
        for lane_tasks, lane_loads in beam:
            for lane_index, width in enumerate(lane_widths):
                empty_after = sum(
                    not tasks_for_lane
                    for index, tasks_for_lane in enumerate(lane_tasks)
                    if index != lane_index
                )
                if empty_after > remaining_tasks:
                    continue
                updated_lanes = list(lane_tasks)
                updated_lanes[lane_index] = (
                    *updated_lanes[lane_index],
                    _retarget_task(task, width, window_selector),
                )
                updated_loads = list(lane_loads)
                updated_loads[lane_index] += task_costs[(task.expert_id, width)]
                assignment = tuple(updated_lanes)
                signature = tuple(
                    tuple(item.expert_id for item in tasks_for_lane)
                    for tasks_for_lane in assignment
                )
                next_beam[signature] = (assignment, tuple(updated_loads))

        def placement_priority(item):
            signature, (_, lane_loads) = item
            return (
                max(lane_loads),
                sum(load * load for load in lane_loads),
                max(lane_loads) - min(lane_loads),
                signature,
            )

        beam = [value for _, value in sorted(next_beam.items(), key=placement_priority)[:placement_beam_width]]
        if not beam:
            return ()
    repaired = {}
    for lane_tasks, _ in beam:
        variants = (
            lane_tasks,
            tuple(tuple(reversed(row)) for row in lane_tasks),
            tuple(
                tuple(sorted(row, key=lambda task: (task.expert_id not in critical_expert_ids, -task.routes, task.expert_id)))
                for row in lane_tasks
            ),
            tuple(
                tuple(sorted(row, key=lambda task: (task.expert_id in critical_expert_ids, -task.routes, task.expert_id)))
                for row in lane_tasks
            ),
        )
        for variant in variants:
            signature = tuple(tuple(task.expert_id for task in row) for row in variant)
            repaired.setdefault(signature, variant)
            if len(repaired) == repair_beam_width:
                return tuple(repaired.values())
    return tuple(repaired.values())


def test_beam_assign_matches_reference_repair_signatures() -> None:
    state = _template_lns_state()
    tasks = tuple(task for lane in state.lanes for task in lane.tasks)
    kwargs = {
        "isolated_cost": _isolated_cost,
        "window_selector": _windows,
        "critical_expert_ids": frozenset({3, 9}),
        "repair_beam_width": 8,
    }
    for template in ((4, 4, 2, 2), (8, 4), (2, 2, 2, 2, 4)):
        got = _beam_assign_tasks(tasks, template, **kwargs)
        expected = _reference_beam_assign_tasks(tasks, template, **kwargs)
        got_ids = tuple(tuple(tuple(task.expert_id for task in row) for row in assignment) for assignment in got)
        expected_ids = tuple(tuple(tuple(task.expert_id for task in row) for row in assignment) for assignment in expected)
        assert got_ids == expected_ids


def test_template_lns_repairs_lane_atomic_local_and_cross_domain_windows() -> None:
    state = _template_lns_state()
    neighbors = list(
        enumerate_template_lns_neighbors(
            state,
            allowed_widths=(1, 2, 4, 6),
            isolated_cost=_isolated_cost,
            window_selector=_windows,
            critical_expert_ids=(3, 9),
            destroy_sizes=(4,),
            repair_beam_widths=(4,),
            templates_per_block=4,
        )
    )

    assert neighbors
    assert {"domain_local", "cross_domain"} == {
        "cross_domain" if "_cross_domain_" in neighbor.operator else "domain_local"
        for neighbor in neighbors
    }
    expected_experts = sorted(task.expert_id for lane in state.lanes for task in lane.tasks)
    for neighbor in neighbors:
        assert neighbor.operator.startswith(TEMPLATE_LNS_REPARTITION)
        assert neighbor.operator.endswith("_d4_b4")
        assert len(neighbor.moved_experts) >= 4
        assert neighbor.state.thread_cpu_ids == state.thread_cpu_ids
        assert neighbor.state.llc_domains == state.llc_domains
        assert neighbor.state.early_merge is True
        assert sum(neighbor.state.shape) == state.num_threads
        assert sorted(
            task.expert_id for lane in neighbor.state.lanes for task in lane.tasks
        ) == expected_experts
        assert all(lane.tasks for lane in neighbor.state.lanes)
        for original_lane in state.lanes:
            original_experts = {task.expert_id for task in original_lane.tasks}
            if original_experts.isdisjoint(neighbor.moved_experts):
                assert original_lane in neighbor.state.lanes
        for lane in neighbor.state.lanes:
            if any(task.expert_id in neighbor.moved_experts for task in lane.tasks):
                for task in lane.tasks:
                    assert (task.w13_window_tiles, task.w2_window_tiles) == _windows(
                        task.routes,
                        lane.threads,
                    )
        bridge = neighbor.state.to_bridge()
        assert bridge["execution_mode"] == "strict"
        assert bridge["task_threads"] == [task[3] for task in neighbor.state.to_planner_tasks()]

    cross_neighbors = [neighbor for neighbor in neighbors if "_cross_domain_" in neighbor.operator]
    assert cross_neighbors
    for neighbor in cross_neighbors:
        repaired_lanes = [
            lane
            for lane in neighbor.state.lanes
            if any(task.expert_id in neighbor.moved_experts for task in lane.tasks)
        ]
        assert all(
            any(
                domain.core_begin <= lane.core_begin and lane.core_end <= domain.core_end
                for domain in neighbor.state.llc_domains
            )
            for lane in repaired_lanes
        )


def test_template_lns_sampling_is_deterministic_and_validates_budgets() -> None:
    state = _template_lns_state()
    kwargs = {
        "allowed_widths": (1, 2, 4, 6),
        "isolated_cost": _isolated_cost,
        "window_selector": _windows,
        "critical_expert_ids": (3, 9),
        "destroy_sizes": (4,),
        "repair_beam_widths": (4,),
        "templates_per_block": 3,
        "per_operator": 3,
        "seed": 29,
    }
    first = sample_template_lns_neighborhood(state, **kwargs)
    second = sample_template_lns_neighborhood(state, **kwargs)

    first_hashes = [neighbor.state.canonical_hash() for neighbor in first.neighbors]
    assert first_hashes == [neighbor.state.canonical_hash() for neighbor in second.neighbors]
    assert first.breakdown is not None
    assert first.breakdown["unique_blocks"] > 0
    assert first.breakdown["hashed_states"] == first.unique + first.duplicates
    assert first.breakdown["unique_hashed_states"] == first.unique
    assert first.breakdown["duplicate_hashed_states"] == first.duplicates
    for key in ("assemble_s", "beam_s", "closure_s", "hash_s", "template_s"):
        assert first.breakdown[key] >= 0.0
    assert first.breakdown["beam_calls"] >= 1
    assert len(first_hashes) == len(set(first_hashes))
    assert state.canonical_hash() not in first_hashes
    assert first.proposed == first.unique + first.duplicates
    assert all(count <= 3 for count in first.sampled_by_operator.values())

    one_template_neighbors = list(
        enumerate_template_lns_neighbors(
            state,
            allowed_widths=(1, 2, 4, 6),
            isolated_cost=_isolated_cost,
            window_selector=_windows,
            critical_expert_ids=(3, 9),
            destroy_sizes=(4,),
            repair_beam_widths=(4,),
            templates_per_block=1,
        )
    )
    assert len(one_template_neighbors) <= 2 * 2 * 4

    with pytest.raises(ValueError, match="match repair_beam_widths"):
        list(
            enumerate_template_lns_neighbors(
                state,
                allowed_widths=(1, 2, 4),
                isolated_cost=_isolated_cost,
                window_selector=_windows,
                critical_expert_ids=(3,),
                destroy_sizes=(4, 8),
                repair_beam_widths=(16,),
            )
        )
    with pytest.raises(ValueError, match="critical_expert_ids"):
        list(
            enumerate_template_lns_neighbors(
                state,
                allowed_widths=(1, 2, 4),
                isolated_cost=_isolated_cost,
                window_selector=_windows,
                critical_expert_ids=(),
            )
        )


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
