"""Order-only executable neighborhoods for the Step-1 MoE planner audit."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from executable_plan_state import ExecutableLane, ExecutablePlanState


ADJACENT_SWAP = "same_lane_adjacent_swap"
SAME_LANE_INSERTION = "same_lane_insertion"
CROSS_LANE_RELOCATION = "same_width_cross_lane_relocation"
CROSS_LANE_SWAP = "same_width_cross_lane_swap"
CROSS_LLC_RELOCATION = "same_width_cross_llc_relocation"
ORDER_ONLY_OPERATORS = (
    ADJACENT_SWAP,
    SAME_LANE_INSERTION,
    CROSS_LANE_RELOCATION,
    CROSS_LANE_SWAP,
    CROSS_LLC_RELOCATION,
)


@dataclass(frozen=True)
class ExecutablePlanNeighbor:
    """One legal order-only move and its resulting executable state."""

    operator: str
    moved_experts: tuple[int, ...]
    state: ExecutablePlanState


@dataclass(frozen=True)
class SampledNeighborhood:
    """Deduplicated, per-operator sampled neighbors plus audit counters."""

    neighbors: tuple[ExecutablePlanNeighbor, ...]
    proposed_by_operator: Mapping[str, int]
    unique_by_operator: Mapping[str, int]
    duplicate_by_operator: Mapping[str, int]
    sampled_by_operator: Mapping[str, int]

    @property
    def proposed(self) -> int:
        return sum(self.proposed_by_operator.values())

    @property
    def unique(self) -> int:
        return sum(self.unique_by_operator.values())

    @property
    def duplicates(self) -> int:
        return sum(self.duplicate_by_operator.values())


def _with_lane_tasks(
    state: ExecutablePlanState,
    replacements: Mapping[int, Sequence],
) -> ExecutablePlanState:
    lanes = tuple(
        ExecutableLane(lane.core_begin, lane.threads, tuple(replacements.get(index, lane.tasks)))
        for index, lane in enumerate(state.lanes)
    )
    return ExecutablePlanState(
        num_threads=state.num_threads,
        thread_cpu_ids=state.thread_cpu_ids,
        lanes=lanes,
        llc_domains=state.llc_domains,
        early_merge=state.early_merge,
    )


def _selected(expert_ids: Iterable[int], expert_filter: frozenset[int] | None) -> bool:
    return expert_filter is None or any(expert_id in expert_filter for expert_id in expert_ids)


def _is_cross_llc(state: ExecutablePlanState, source_lane: int, target_lane: int) -> bool:
    if not state.llc_domains:
        return False
    return state.lane_domain_ids(source_lane) != state.lane_domain_ids(target_lane)


def enumerate_order_only_neighbors(
    state: ExecutablePlanState,
    *,
    expert_filter: Iterable[int] | None = None,
) -> Iterable[ExecutablePlanNeighbor]:
    """Yield legal order-only moves, including intentional cross-operator duplicates."""

    selected_experts = None if expert_filter is None else frozenset(int(value) for value in expert_filter)

    for lane_index, lane in enumerate(state.lanes):
        tasks = lane.tasks
        for left in range(len(tasks) - 1):
            moved = (tasks[left].expert_id, tasks[left + 1].expert_id)
            if not _selected(moved, selected_experts):
                continue
            updated = list(tasks)
            updated[left], updated[left + 1] = updated[left + 1], updated[left]
            yield ExecutablePlanNeighbor(
                ADJACENT_SWAP,
                moved,
                _with_lane_tasks(state, {lane_index: updated}),
            )

        for source_position, task in enumerate(tasks):
            if not _selected((task.expert_id,), selected_experts):
                continue
            without = list(tasks)
            without.pop(source_position)
            for target_position in range(len(without) + 1):
                if target_position == source_position:
                    continue
                updated = list(without)
                updated.insert(target_position, task)
                yield ExecutablePlanNeighbor(
                    SAME_LANE_INSERTION,
                    (task.expert_id,),
                    _with_lane_tasks(state, {lane_index: updated}),
                )

    lanes_by_width: dict[int, list[int]] = defaultdict(list)
    for lane_index, lane in enumerate(state.lanes):
        lanes_by_width[lane.threads].append(lane_index)

    for lane_indices in lanes_by_width.values():
        for source_lane in lane_indices:
            source_tasks = state.lanes[source_lane].tasks
            for target_lane in lane_indices:
                if source_lane == target_lane:
                    continue
                target_tasks = state.lanes[target_lane].tasks
                cross_llc = _is_cross_llc(state, source_lane, target_lane)
                relocation_operator = CROSS_LLC_RELOCATION if cross_llc else CROSS_LANE_RELOCATION
                for source_position, task in enumerate(source_tasks):
                    if not _selected((task.expert_id,), selected_experts):
                        continue
                    new_source = list(source_tasks)
                    new_source.pop(source_position)
                    for target_position in range(len(target_tasks) + 1):
                        new_target = list(target_tasks)
                        new_target.insert(target_position, task)
                        yield ExecutablePlanNeighbor(
                            relocation_operator,
                            (task.expert_id,),
                            _with_lane_tasks(
                                state,
                                {source_lane: new_source, target_lane: new_target},
                            ),
                        )

                if target_lane <= source_lane:
                    continue
                for source_position, source_task in enumerate(source_tasks):
                    for target_position, target_task in enumerate(target_tasks):
                        moved = (source_task.expert_id, target_task.expert_id)
                        if not _selected(moved, selected_experts):
                            continue
                        new_source = list(source_tasks)
                        new_target = list(target_tasks)
                        new_source[source_position] = target_task
                        new_target[target_position] = source_task
                        yield ExecutablePlanNeighbor(
                            CROSS_LANE_SWAP,
                            moved,
                            _with_lane_tasks(
                                state,
                                {source_lane: new_source, target_lane: new_target},
                            ),
                        )


def sample_order_only_neighborhood(
    state: ExecutablePlanState,
    *,
    expert_filter: Iterable[int] | None,
    per_operator: int,
    seed: int,
) -> SampledNeighborhood:
    """Deduplicate all eligible moves and sample equal budgets per operator."""

    if per_operator <= 0:
        raise ValueError("per_operator must be positive")
    baseline_hash = state.canonical_hash()
    proposed = {operator: 0 for operator in ORDER_ONLY_OPERATORS}
    duplicates = {operator: 0 for operator in ORDER_ONLY_OPERATORS}
    unique: dict[str, dict[str, ExecutablePlanNeighbor]] = {operator: {} for operator in ORDER_ONLY_OPERATORS}
    seen_global = {baseline_hash}
    for neighbor in enumerate_order_only_neighbors(state, expert_filter=expert_filter):
        proposed[neighbor.operator] += 1
        state_hash = neighbor.state.canonical_hash()
        if state_hash in seen_global:
            duplicates[neighbor.operator] += 1
            continue
        seen_global.add(state_hash)
        unique[neighbor.operator][state_hash] = neighbor

    rng = random.Random(seed)
    sampled = []
    sampled_counts = {}
    unique_counts = {}
    for operator in ORDER_ONLY_OPERATORS:
        candidates = list(unique[operator].values())
        unique_counts[operator] = len(candidates)
        rng.shuffle(candidates)
        chosen = candidates[:per_operator]
        sampled_counts[operator] = len(chosen)
        sampled.extend(chosen)
    return SampledNeighborhood(
        neighbors=tuple(sampled),
        proposed_by_operator=proposed,
        unique_by_operator=unique_counts,
        duplicate_by_operator=duplicates,
        sampled_by_operator=sampled_counts,
    )


def placed_tasks(state: ExecutablePlanState) -> list[tuple[int, int, tuple[int, ...], list[int]]]:
    """Lower a state to the placement-aware analytical scorer input."""

    return [
        (
            routes,
            threads,
            state.thread_cpu_ids[core_begin : core_begin + threads],
            dependencies,
        )
        for _, routes, core_begin, threads, dependencies in state.to_planner_tasks()
    ]


def critical_expert_scores(
    state: ExecutablePlanState,
    explanation: Mapping[str, object],
) -> dict[int, float]:
    """Rank experts by tail-weighted time exposed to their event dilation."""

    tasks = [task for lane in state.lanes for task in lane.tasks]
    makespan_ns = float(explanation["makespan_ns"])
    scores = {task.expert_id: 0.0 for task in tasks}
    raw_events = explanation.get("events")
    if not isinstance(raw_events, Sequence):
        raise TypeError("explanation events must be a sequence")
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise TypeError("every explanation event must be a mapping")
        start_ns = float(event["start_ns"])
        duration_ns = float(event["duration_ns"])
        active_tasks = tuple(int(value) for value in event["active_tasks"])
        dilations = event["phase_dilation"]
        if not isinstance(dilations, Mapping):
            raise TypeError("event phase_dilation must be a mapping")
        tail_weight = 1.0 + (start_ns + duration_ns) / max(makespan_ns, 1.0)
        for task_index in active_tasks:
            scores[tasks[task_index].expert_id] += duration_ns * float(dilations[str(task_index)]) * tail_weight
    return scores


__all__ = [
    "ADJACENT_SWAP",
    "CROSS_LANE_RELOCATION",
    "CROSS_LANE_SWAP",
    "CROSS_LLC_RELOCATION",
    "ORDER_ONLY_OPERATORS",
    "SAME_LANE_INSERTION",
    "ExecutablePlanNeighbor",
    "SampledNeighborhood",
    "critical_expert_scores",
    "enumerate_order_only_neighbors",
    "placed_tasks",
    "sample_order_only_neighborhood",
]
