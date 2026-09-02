"""Executable order and topology-preserving width neighborhoods for MoE audits."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain

from executable_plan_state import ExecutableExpertTask, ExecutableLane, ExecutablePlanState


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
LANE_SPLIT = "domain_local_lane_split"
LANE_MERGE = "domain_local_lane_merge"
ADJACENT_WIDTH_MIGRATION = "domain_local_adjacent_width_migration"
WIDTH_ONLY_OPERATORS = (
    LANE_SPLIT,
    LANE_MERGE,
    ADJACENT_WIDTH_MIGRATION,
)


@dataclass(frozen=True)
class ExecutablePlanNeighbor:
    """One legal local move and its resulting executable state."""

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


@dataclass(frozen=True)
class ExecutablePlanScore:
    """Event estimate plus a serial-lane uncertainty guard."""

    event_ns: float
    lane_guard_ns: float
    robust_ns: float


def score_executable_plan(
    model,
    state: ExecutablePlanState,
    *,
    relative_uncertainty: float | None = None,
) -> ExecutablePlanScore:
    """Score one state without hiding a newly heavy lane below the event path."""

    uncertainty = (
        float(getattr(model, "relative_error", 0.0)) if relative_uncertainty is None else float(relative_uncertainty)
    )
    if not 0.0 <= uncertainty < 1.0:
        raise ValueError("relative_uncertainty must be in [0, 1)")
    event_ns = float(model.dag_makespan_placed(placed_tasks(state)))
    lane_guard_ns = max(
        sum(float(model.T_iso(task.routes, lane.threads)) for task in lane.tasks) for lane in state.lanes
    ) * (1.0 + uncertainty)
    return ExecutablePlanScore(
        event_ns=event_ns,
        lane_guard_ns=lane_guard_ns,
        robust_ns=max(event_ns, lane_guard_ns),
    )


def is_resolvable_improvement(
    incumbent: ExecutablePlanScore,
    candidate: ExecutablePlanScore,
    *,
    minimum_gain_fraction: float,
) -> bool:
    """Require a positive robust gain larger than the action-resolution margin."""

    if not 0.0 <= minimum_gain_fraction < 1.0:
        raise ValueError("minimum_gain_fraction must be in [0, 1)")
    return candidate.robust_ns <= incumbent.robust_ns * (1.0 - minimum_gain_fraction)


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


def _containing_domain_id(state: ExecutablePlanState, lane_index: int) -> str | None:
    """Return the sole LLC domain containing a lane, rejecting cross-domain lanes."""

    domain_ids = state.lane_domain_ids(lane_index)
    return domain_ids[0] if len(domain_ids) == 1 else None


def _retarget_task(
    task: ExecutableExpertTask,
    threads: int,
    window_selector: Callable[[int, int], tuple[int, int]],
) -> ExecutableExpertTask:
    w13_window_tiles, w2_window_tiles = window_selector(task.routes, threads)
    return ExecutableExpertTask(
        expert_id=task.expert_id,
        routes=task.routes,
        w13_window_tiles=int(w13_window_tiles),
        w2_window_tiles=int(w2_window_tiles),
    )


def _lpt_assign_tasks(
    tasks: Sequence[ExecutableExpertTask],
    lane_widths: Sequence[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
) -> tuple[tuple[ExecutableExpertTask, ...], ...]:
    lane_tasks: list[list[ExecutableExpertTask]] = [[] for _ in lane_widths]
    lane_loads = [0.0 for _ in lane_widths]
    ordered = sorted(
        tasks,
        key=lambda task: (
            -max(float(isolated_cost(task.routes, width)) for width in lane_widths),
            task.expert_id,
        ),
    )
    for task in ordered:
        lane_index = min(
            range(len(lane_widths)),
            key=lambda index: (
                lane_loads[index] + float(isolated_cost(task.routes, lane_widths[index])),
                lane_loads[index],
                index,
            ),
        )
        width = lane_widths[lane_index]
        lane_tasks[lane_index].append(_retarget_task(task, width, window_selector))
        lane_loads[lane_index] += float(isolated_cost(task.routes, width))
    return tuple(tuple(tasks_for_lane) for tasks_for_lane in lane_tasks)


def _with_replaced_lanes(
    state: ExecutablePlanState,
    first_lane: int,
    remove_count: int,
    replacement: Sequence[ExecutableLane],
) -> ExecutablePlanState:
    lanes = state.lanes[:first_lane] + tuple(replacement) + state.lanes[first_lane + remove_count :]
    return ExecutablePlanState(
        num_threads=state.num_threads,
        thread_cpu_ids=state.thread_cpu_ids,
        lanes=lanes,
        llc_domains=state.llc_domains,
        early_merge=state.early_merge,
    )


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


def enumerate_width_only_neighbors(
    state: ExecutablePlanState,
    *,
    allowed_widths: Iterable[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
    expert_filter: Iterable[int] | None = None,
) -> Iterable[ExecutablePlanNeighbor]:
    """Yield domain-local split, merge, and adjacent-width migration moves.

    Split and merge deterministically reassign only the tasks on the replaced
    lanes with isolated-time LPT. Migration preserves the relative order of all
    existing tasks and enumerates every insertion point on the target lane.
    Every task whose width changes receives the deterministic stage windows for
    its new ``(routes, threads)`` pair.
    """

    widths = tuple(sorted({int(width) for width in allowed_widths}))
    if not widths or widths[0] <= 0:
        raise ValueError("allowed_widths must contain positive widths")
    selected_experts = None if expert_filter is None else frozenset(int(value) for value in expert_filter)

    for lane_index, lane in enumerate(state.lanes):
        domain_id = _containing_domain_id(state, lane_index)
        half_width = lane.threads // 2
        if (
            domain_id is not None
            and lane.tasks
            and lane.threads % 2 == 0
            and half_width in widths
            and _selected((task.expert_id for task in lane.tasks), selected_experts)
        ):
            assignments = _lpt_assign_tasks(
                lane.tasks,
                (half_width, half_width),
                isolated_cost,
                window_selector,
            )
            replacement = (
                ExecutableLane(lane.core_begin, half_width, assignments[0]),
                ExecutableLane(lane.core_begin + half_width, half_width, assignments[1]),
            )
            yield ExecutablePlanNeighbor(
                LANE_SPLIT,
                tuple(task.expert_id for task in lane.tasks),
                _with_replaced_lanes(state, lane_index, 1, replacement),
            )

        for target_lane_index, target_lane in enumerate(state.lanes):
            if target_lane_index == lane_index:
                continue
            target_domain_id = _containing_domain_id(state, target_lane_index)
            if domain_id is None or target_domain_id != domain_id:
                continue
            if lane.threads not in widths or target_lane.threads not in widths:
                continue
            width_distance = abs(widths.index(lane.threads) - widths.index(target_lane.threads))
            if width_distance != 1:
                continue
            for source_position, task in enumerate(lane.tasks):
                if not _selected((task.expert_id,), selected_experts):
                    continue
                new_source = list(lane.tasks)
                new_source.pop(source_position)
                moved_task = _retarget_task(task, target_lane.threads, window_selector)
                for target_position in range(len(target_lane.tasks) + 1):
                    new_target = list(target_lane.tasks)
                    new_target.insert(target_position, moved_task)
                    yield ExecutablePlanNeighbor(
                        ADJACENT_WIDTH_MIGRATION,
                        (task.expert_id,),
                        _with_lane_tasks(
                            state,
                            {
                                lane_index: new_source,
                                target_lane_index: new_target,
                            },
                        ),
                    )

    for lane_index in range(len(state.lanes) - 1):
        left = state.lanes[lane_index]
        right = state.lanes[lane_index + 1]
        domain_id = _containing_domain_id(state, lane_index)
        merged_width = left.threads + right.threads
        moved = tuple(task.expert_id for task in (*left.tasks, *right.tasks))
        if (
            domain_id is None
            or _containing_domain_id(state, lane_index + 1) != domain_id
            or left.threads != right.threads
            or merged_width not in widths
            or not _selected(moved, selected_experts)
        ):
            continue
        (merged_tasks,) = _lpt_assign_tasks(
            (*left.tasks, *right.tasks),
            (merged_width,),
            isolated_cost,
            window_selector,
        )
        yield ExecutablePlanNeighbor(
            LANE_MERGE,
            moved,
            _with_replaced_lanes(
                state,
                lane_index,
                2,
                (ExecutableLane(left.core_begin, merged_width, merged_tasks),),
            ),
        )


def _sample_neighborhood(
    state: ExecutablePlanState,
    neighbors: Iterable[ExecutablePlanNeighbor],
    *,
    operators: Sequence[str],
    per_operator: int,
    seed: int,
) -> SampledNeighborhood:
    if per_operator <= 0:
        raise ValueError("per_operator must be positive")
    baseline_hash = state.canonical_hash()
    proposed = {operator: 0 for operator in operators}
    duplicates = {operator: 0 for operator in operators}
    unique: dict[str, dict[str, ExecutablePlanNeighbor]] = {operator: {} for operator in operators}
    seen_global = {baseline_hash}
    for neighbor in neighbors:
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
    for operator in operators:
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


def sample_order_only_neighborhood(
    state: ExecutablePlanState,
    *,
    expert_filter: Iterable[int] | None,
    per_operator: int,
    seed: int,
) -> SampledNeighborhood:
    """Deduplicate all eligible moves and sample equal budgets per operator."""
    return _sample_neighborhood(
        state,
        enumerate_order_only_neighbors(state, expert_filter=expert_filter),
        operators=ORDER_ONLY_OPERATORS,
        per_operator=per_operator,
        seed=seed,
    )


def sample_width_only_neighborhood(
    state: ExecutablePlanState,
    *,
    allowed_widths: Iterable[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
    expert_filter: Iterable[int] | None,
    per_operator: int,
    seed: int,
) -> SampledNeighborhood:
    """Deduplicate and sample equal budgets from the legal width operators."""

    return _sample_neighborhood(
        state,
        enumerate_width_only_neighbors(
            state,
            allowed_widths=allowed_widths,
            isolated_cost=isolated_cost,
            window_selector=window_selector,
            expert_filter=expert_filter,
        ),
        operators=WIDTH_ONLY_OPERATORS,
        per_operator=per_operator,
        seed=seed,
    )


def sample_combined_neighborhood(
    state: ExecutablePlanState,
    *,
    allowed_widths: Iterable[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
    expert_filter: Iterable[int] | None,
    per_operator: int,
    seed: int,
) -> SampledNeighborhood:
    """Sample the union of order and width moves with global deduplication."""

    return _sample_neighborhood(
        state,
        chain(
            enumerate_order_only_neighbors(state, expert_filter=expert_filter),
            enumerate_width_only_neighbors(
                state,
                allowed_widths=allowed_widths,
                isolated_cost=isolated_cost,
                window_selector=window_selector,
                expert_filter=expert_filter,
            ),
        ),
        operators=(*ORDER_ONLY_OPERATORS, *WIDTH_ONLY_OPERATORS),
        per_operator=per_operator,
        seed=seed,
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
    "ADJACENT_WIDTH_MIGRATION",
    "ADJACENT_SWAP",
    "CROSS_LANE_RELOCATION",
    "CROSS_LANE_SWAP",
    "CROSS_LLC_RELOCATION",
    "LANE_MERGE",
    "LANE_SPLIT",
    "ORDER_ONLY_OPERATORS",
    "SAME_LANE_INSERTION",
    "WIDTH_ONLY_OPERATORS",
    "ExecutablePlanNeighbor",
    "ExecutablePlanScore",
    "SampledNeighborhood",
    "critical_expert_scores",
    "enumerate_order_only_neighbors",
    "enumerate_width_only_neighbors",
    "placed_tasks",
    "is_resolvable_improvement",
    "sample_order_only_neighborhood",
    "sample_combined_neighborhood",
    "sample_width_only_neighborhood",
    "score_executable_plan",
]
