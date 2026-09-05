"""Executable order and topology-preserving width neighborhoods for MoE audits."""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
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
TEMPLATE_LNS_REPARTITION = "critical_window_template_repartition"
TEMPLATE_LNS_SCOPES = ("domain_local", "cross_domain")
TEMPLATE_LNS_DEFAULT_DESTROY_SIZES = (4, 8, 16)
TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS = (16, 32, 64)


def _template_lns_operator(scope: str, destroy_size: int, repair_beam_width: int) -> str:
    return f"{TEMPLATE_LNS_REPARTITION}_{scope}_d{destroy_size}_b{repair_beam_width}"


TEMPLATE_LNS_OPERATORS = tuple(
    _template_lns_operator(scope, destroy_size, beam_width)
    for scope in TEMPLATE_LNS_SCOPES
    for destroy_size, beam_width in zip(
        TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
        TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
        strict=True,
    )
)


@dataclass(frozen=True)
class ExecutablePlanNeighbor:
    """One legal local move and its resulting executable state."""

    operator: str
    moved_experts: tuple[int, ...]
    state: ExecutablePlanState


@dataclass
class LnsSearchBreakdown:
    """Wall-time and counter split for one template-LNS enumeration/sample."""

    closure_ns: int = 0
    template_ns: int = 0
    beam_ns: int = 0
    assemble_ns: int = 0
    hash_ns: int = 0
    unique_blocks: int = 0
    template_enumerations: int = 0
    templates_selected: int = 0
    beam_calls: int = 0
    neighbors_emitted: int = 0
    hashed_states: int = 0
    unique_hashed_states: int = 0
    duplicate_hashed_states: int = 0

    def to_dict(self) -> dict[str, object]:
        scale = 1.0e-9
        return {
            "assemble_s": self.assemble_ns * scale,
            "beam_calls": self.beam_calls,
            "beam_s": self.beam_ns * scale,
            "closure_s": self.closure_ns * scale,
            "duplicate_hashed_states": self.duplicate_hashed_states,
            "hash_s": self.hash_ns * scale,
            "hashed_states": self.hashed_states,
            "neighbors_emitted": self.neighbors_emitted,
            "template_enumerations": self.template_enumerations,
            "template_s": self.template_ns * scale,
            "templates_selected": self.templates_selected,
            "unique_blocks": self.unique_blocks,
            "unique_hashed_states": self.unique_hashed_states,
        }


@dataclass(frozen=True)
class SampledNeighborhood:
    """Deduplicated, per-operator sampled neighbors plus audit counters."""

    neighbors: tuple[ExecutablePlanNeighbor, ...]
    proposed_by_operator: Mapping[str, int]
    unique_by_operator: Mapping[str, int]
    duplicate_by_operator: Mapping[str, int]
    sampled_by_operator: Mapping[str, int]
    breakdown: Mapping[str, object] | None = None

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


@dataclass(frozen=True)
class ExecutablePlanScreenScore:
    """Cheap phase surrogate plus valid isolated resource lower bounds."""

    phase_surrogate_ns: float
    lane_bound_ns: float
    domain_bound_ns: float
    rank_bound_ns: float
    lower_bound_ns: float
    priority_ns: float


class ExecutablePlanEvaluator:
    """Two-level executable-plan evaluator with immutable phase reuse.

    The screen is intentionally not an exact objective. It combines a cheap
    phase-only event simulation with isolated lane/domain/rank lower bounds.
    Exact scores remain authoritative and are cached only by canonical state
    hash, so cache hits cannot change planner semantics.
    """

    def __init__(self, model) -> None:
        self.model = model
        self._exact_cache: dict[str, ExecutablePlanScore] = {}
        self._screen_cache: dict[str, ExecutablePlanScreenScore] = {}
        self._lane_load_cache: dict[tuple[int, tuple[tuple[int, int], ...]], float] = {}
        self._lane_phase_cache: dict[
            tuple[int, tuple[int, ...]],
            tuple[tuple[float, bool], ...],
        ] = {}
        self._team_scale_cache: dict[tuple[int, int, int], float] = {}
        self.exact_calls = 0
        self.exact_cache_hits = 0
        self.screen_calls = 0
        self.screen_cache_hits = 0
        self.lane_cache_hits = 0
        self.lane_cache_misses = 0
        self.lane_phase_cache_hits = 0
        self.lane_phase_cache_misses = 0

    def exact(self, state: ExecutablePlanState) -> ExecutablePlanScore:
        state_hash = state.canonical_hash()
        cached = self._exact_cache.get(state_hash)
        if cached is not None:
            self.exact_cache_hits += 1
            return cached
        score = score_executable_plan(self.model, state)
        self._exact_cache[state_hash] = score
        self.exact_calls += 1
        return score

    def _lane_load(self, lane: ExecutableLane) -> float:
        key = (
            lane.threads,
            tuple(sorted((task.expert_id, task.routes) for task in lane.tasks)),
        )
        cached = self._lane_load_cache.get(key)
        if cached is not None:
            self.lane_cache_hits += 1
            return cached
        load = sum(float(self.model.T_iso(task.routes, lane.threads)) for task in lane.tasks)
        self._lane_load_cache[key] = load
        self.lane_cache_misses += 1
        return load

    def _lane_phases(self, lane: ExecutableLane) -> tuple[tuple[float, bool], ...]:
        key = (lane.threads, tuple(task.routes for task in lane.tasks))
        cached = self._lane_phase_cache.get(key)
        if cached is not None:
            self.lane_phase_cache_hits += 1
            return cached
        phases = tuple(
            (float(phase.base_ns), phase.kind in {"cold_b", "steady_b"})
            for task in lane.tasks
            for phase in self.model.predict_expert(task.routes, lane.threads).phases
        )
        self._lane_phase_cache[key] = phases
        self.lane_phase_cache_misses += 1
        return phases

    def _team_scale(self, width: int, occupied_threads: int, total_threads: int) -> float:
        key = (width, occupied_threads, total_threads)
        cached = self._team_scale_cache.get(key)
        if cached is not None:
            return cached
        available_peer_threads = total_threads - width
        peer_threads = max(occupied_threads - width, 0)
        peer_fraction = (
            min(peer_threads / available_peer_threads, 1.0)
            if available_peer_threads > 0
            else 0.0
        )
        pressure = self.model.calibration.wide_team_pressure
        isolated = pressure.isolated_scale(width)
        full = pressure.full_cohort_scale(width)
        wide_scale = isolated + (full - isolated) * peer_fraction
        narrow_scale = self.model.calibration.narrow_team_contention_correction.scale(
            width,
            peer_fraction,
        )
        scale = max(1.0, wide_scale * narrow_scale)
        self._team_scale_cache[key] = scale
        return scale

    def _phase_surrogate(self, state: ExecutablePlanState) -> float:
        lanes = [lane for lane in state.lanes if lane.tasks]
        lane_phases = [self._lane_phases(lane) for lane in lanes]
        phase_indices = [0] * len(lanes)
        remaining = [phases[0][0] for phases in lane_phases]
        wall_ns = float(getattr(self.model, "call_setup_ns", 0.0))
        unfinished = set(range(len(lanes)))
        max_events = sum(map(len, lane_phases))
        events = 0
        while unfinished:
            events += 1
            if events > max_events + len(lanes):
                raise RuntimeError("phase-only screening simulator did not converge")
            occupied_threads = min(
                sum(lanes[index].threads for index in unfinished),
                state.num_threads,
            )
            multipliers = {}
            for index in unfinished:
                lane = lanes[index]
                _, is_compute = lane_phases[index][phase_indices[index]]
                if is_compute:
                    multipliers[index] = self._team_scale(
                        lane.threads,
                        occupied_threads,
                        state.num_threads,
                    )
                else:
                    multipliers[index] = 1.0
            elapsed = min(remaining[index] * multipliers[index] for index in unfinished)
            wall_ns += elapsed
            completed = []
            for index in unfinished:
                remaining[index] -= elapsed / multipliers[index]
                if remaining[index] <= 1e-6:
                    completed.append(index)
            for index in completed:
                phase_indices[index] += 1
                if phase_indices[index] >= len(lane_phases[index]):
                    unfinished.remove(index)
                    continue
                remaining[index] = lane_phases[index][phase_indices[index]][0]
        return wall_ns

    def screen(self, state: ExecutablePlanState) -> ExecutablePlanScreenScore:
        state_hash = state.canonical_hash()
        cached = self._screen_cache.get(state_hash)
        if cached is not None:
            self.screen_cache_hits += 1
            return cached
        lane_loads = [self._lane_load(lane) for lane in state.lanes]
        lane_bound = max(lane_loads, default=0.0)
        total_core_ns = sum(load * lane.threads for load, lane in zip(lane_loads, state.lanes, strict=True))
        rank_bound = total_core_ns / state.num_threads
        domain_work = {domain.domain_id: 0.0 for domain in state.llc_domains}
        for load, lane in zip(lane_loads, state.lanes, strict=True):
            for domain in state.llc_domains:
                overlap = max(
                    min(lane.core_end, domain.core_end) - max(lane.core_begin, domain.core_begin),
                    0,
                )
                domain_work[domain.domain_id] += load * overlap
        domain_bound = max(
            (
                domain_work[domain.domain_id] / domain.core_count
                for domain in state.llc_domains
            ),
            default=rank_bound,
        )
        lower_bound = max(lane_bound, domain_bound, rank_bound)
        phase_surrogate = self._phase_surrogate(state)
        score = ExecutablePlanScreenScore(
            phase_surrogate_ns=phase_surrogate,
            lane_bound_ns=lane_bound,
            domain_bound_ns=domain_bound,
            rank_bound_ns=rank_bound,
            lower_bound_ns=lower_bound,
            priority_ns=max(lower_bound, phase_surrogate),
        )
        self._screen_cache[state_hash] = score
        self.screen_calls += 1
        return score


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


def _region_width_templates(
    total_threads: int,
    allowed_widths: Sequence[int],
    *,
    maximum_lanes: int,
) -> tuple[tuple[int, ...], ...]:
    templates: set[tuple[int, ...]] = set()

    def visit(remaining: int, maximum_width: int, prefix: tuple[int, ...]) -> None:
        if remaining == 0:
            templates.add(prefix)
            templates.add(tuple(reversed(prefix)))
            return
        if len(prefix) >= maximum_lanes:
            return
        for width in reversed(allowed_widths):
            if width <= min(remaining, maximum_width):
                visit(remaining - width, width, (*prefix, width))

    visit(total_threads, total_threads, ())
    return tuple(sorted(templates, key=lambda item: (len(item), item)))


def _block_domain_ids(
    state: ExecutablePlanState,
    first_lane: int,
    end_lane: int,
) -> frozenset[str]:
    return frozenset(
        domain_id
        for lane_index in range(first_lane, end_lane)
        for domain_id in state.lane_domain_ids(lane_index)
    )


def _minimum_critical_block(
    state: ExecutablePlanState,
    center_lane: int,
    destroy_size: int,
    *,
    scope: str,
) -> tuple[int, int, int] | None:
    center_domain = _containing_domain_id(state, center_lane)
    candidates = []
    for first_lane in range(center_lane + 1):
        for end_lane in range(center_lane + 1, len(state.lanes) + 1):
            domain_ids = _block_domain_ids(state, first_lane, end_lane)
            if scope == "domain_local":
                if center_domain is None or domain_ids != {center_domain}:
                    continue
                if any(
                    _containing_domain_id(state, lane_index) != center_domain
                    for lane_index in range(first_lane, end_lane)
                ):
                    continue
            elif scope == "cross_domain":
                if len(domain_ids) < 2:
                    continue
            else:
                raise ValueError(f"unknown template LNS scope: {scope}")
            task_count = sum(len(state.lanes[index].tasks) for index in range(first_lane, end_lane))
            if task_count < destroy_size:
                continue
            core_span = state.lanes[end_lane - 1].core_end - state.lanes[first_lane].core_begin
            candidates.append(
                (
                    task_count - destroy_size,
                    end_lane - first_lane,
                    core_span,
                    first_lane,
                    end_lane,
                    task_count,
                )
            )
    if not candidates:
        return None
    _, _, _, first_lane, end_lane, task_count = min(candidates)
    return first_lane, end_lane, task_count


def _template_is_domain_local(
    state: ExecutablePlanState,
    core_begin: int,
    lane_widths: Sequence[int],
) -> bool:
    cursor = core_begin
    for width in lane_widths:
        core_end = cursor + width
        if not any(
            domain.core_begin <= cursor and core_end <= domain.core_end
            for domain in state.llc_domains
        ):
            return False
        cursor = core_end
    return True


def _width_histogram_distance(left: Sequence[int], right: Sequence[int]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    return sum(abs(left_counts[width] - right_counts[width]) for width in left_counts | right_counts)


def _select_region_width_templates(
    templates: Sequence[tuple[int, ...]],
    original_widths: tuple[int, ...],
    *,
    maximum_templates: int,
) -> tuple[tuple[int, ...], ...]:
    if maximum_templates <= 0:
        raise ValueError("maximum_templates must be positive")
    candidates = tuple(dict.fromkeys(templates))
    if len(candidates) <= maximum_templates:
        return candidates

    def distance(template: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        return (
            _width_histogram_distance(original_widths, template),
            abs(len(original_widths) - len(template)),
            template,
        )

    near_budget = (maximum_templates + 1) // 2
    far_budget = maximum_templates - near_budget
    selected = list(sorted(candidates, key=distance)[:near_budget])
    for template in sorted(candidates, key=distance, reverse=True):
        if len(selected) >= near_budget + far_budget:
            break
        if template not in selected:
            selected.append(template)
    return tuple(selected)


def _beam_assign_tasks(
    tasks: Sequence[ExecutableExpertTask],
    lane_widths: Sequence[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
    critical_expert_ids: frozenset[int],
    *,
    repair_beam_width: int,
) -> tuple[tuple[tuple[ExecutableExpertTask, ...], ...], ...]:
    """Repair one width template with a bounded placement beam and four order policies."""

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
    retargeted = {
        (task.expert_id, width): _retarget_task(task, width, window_selector)
        for task in ordered_tasks
        for width in lane_widths
    }
    empty_assignment = tuple(() for _ in lane_widths)
    empty_signature = empty_assignment
    beam = [(empty_assignment, tuple(0.0 for _ in lane_widths), empty_signature)]
    for task_index, task in enumerate(ordered_tasks):
        remaining_tasks = len(ordered_tasks) - task_index - 1
        next_beam: dict[
            tuple[tuple[int, ...], ...],
            tuple[tuple[tuple[ExecutableExpertTask, ...], ...], tuple[float, ...], tuple[tuple[int, ...], ...]],
        ] = {}
        expert_id = task.expert_id
        for lane_tasks, lane_loads, signature in beam:
            empty_count = sum(not row for row in lane_tasks)
            for lane_index, width in enumerate(lane_widths):
                empty_after = empty_count if lane_tasks[lane_index] else empty_count - 1
                if empty_after > remaining_tasks:
                    continue
                updated_row = (*lane_tasks[lane_index], retargeted[(expert_id, width)])
                assignment = (
                    *lane_tasks[:lane_index],
                    updated_row,
                    *lane_tasks[lane_index + 1 :],
                )
                updated_loads = list(lane_loads)
                updated_loads[lane_index] += task_costs[(expert_id, width)]
                new_signature = (
                    *signature[:lane_index],
                    (*signature[lane_index], expert_id),
                    *signature[lane_index + 1 :],
                )
                next_beam[new_signature] = (assignment, tuple(updated_loads), new_signature)

        def placement_priority(item):
            signature, (_, lane_loads, _) = item
            return (
                max(lane_loads),
                sum(load * load for load in lane_loads),
                max(lane_loads) - min(lane_loads),
                signature,
            )

        beam = [value for _, value in sorted(next_beam.items(), key=placement_priority)[:placement_beam_width]]
        if not beam:
            return ()

    repaired: dict[
        tuple[tuple[int, ...], ...],
        tuple[tuple[ExecutableExpertTask, ...], ...],
    ] = {}
    for lane_tasks, _, _ in beam:
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


def enumerate_template_lns_neighbors(
    state: ExecutablePlanState,
    *,
    allowed_widths: Iterable[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
    critical_expert_ids: Iterable[int],
    destroy_sizes: Iterable[int] = TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
    repair_beam_widths: Iterable[int] = TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
    templates_per_block: int = 4,
    breakdown: LnsSearchBreakdown | None = None,
) -> Iterable[ExecutablePlanNeighbor]:
    """Repartition lane-atomic critical windows into executable templates.

    ``destroy_sizes`` are target expert counts. The actual moved set is the
    smallest whole-lane closure that reaches the target and contains the
    selected critical expert. Local proposals stay inside one LLC domain;
    cross-domain proposals must repair the window into domain-contained lanes.
    """

    widths = tuple(sorted({int(width) for width in allowed_widths}))
    size_values = tuple(int(size) for size in destroy_sizes)
    beam_widths = tuple(int(width) for width in repair_beam_widths)
    if len(size_values) != len(beam_widths) or len(set(size_values)) != len(size_values):
        raise ValueError("destroy_sizes must be unique and match repair_beam_widths")
    size_beam_pairs = tuple(sorted(zip(size_values, beam_widths, strict=True)))
    sizes = tuple(size for size, _ in size_beam_pairs)
    beam_widths = tuple(beam for _, beam in size_beam_pairs)
    critical = frozenset(int(expert) for expert in critical_expert_ids)
    if not widths or widths[0] <= 0:
        raise ValueError("allowed_widths must contain positive widths")
    if not sizes or sizes[0] <= 0:
        raise ValueError("destroy_sizes must contain positive integers")
    if not beam_widths or min(beam_widths) <= 0:
        raise ValueError("repair_beam_widths must be positive and match destroy_sizes")
    if templates_per_block <= 0:
        raise ValueError("templates_per_block must be positive")
    if not critical:
        raise ValueError("critical_expert_ids must be non-empty")

    lane_by_expert = {
        task.expert_id: lane_index
        for lane_index, lane in enumerate(state.lanes)
        for task in lane.tasks
    }
    blocks: set[tuple[str, int, int, int, int, int]] = set()
    closure_begin = time.perf_counter_ns()
    for expert_id in sorted(critical):
        center = lane_by_expert.get(expert_id)
        if center is None:
            continue
        for destroy_size, repair_beam_width in zip(sizes, beam_widths, strict=True):
            for scope in TEMPLATE_LNS_SCOPES:
                block = _minimum_critical_block(
                    state,
                    center,
                    destroy_size,
                    scope=scope,
                )
                if block is not None:
                    first, end, task_count = block
                    blocks.add((scope, first, end, destroy_size, task_count, repair_beam_width))
    if breakdown is not None:
        breakdown.closure_ns += time.perf_counter_ns() - closure_begin
        breakdown.unique_blocks += len(blocks)

    for scope, first, end, destroy_size, task_count, repair_beam_width in sorted(blocks):
        original_lanes = state.lanes[first:end]
        original_widths = tuple(lane.threads for lane in original_lanes)
        total_threads = sum(original_widths)
        tasks = tuple(task for lane in original_lanes for task in lane.tasks)
        if len(tasks) != task_count:
            raise RuntimeError("template LNS block task count changed during enumeration")
        template_begin = time.perf_counter_ns()
        templates = tuple(
            template
            for template in _region_width_templates(
                total_threads,
                widths,
                maximum_lanes=min(len(tasks), max(len(original_lanes) + 4, 8)),
            )
            if len(template) <= len(tasks)
            and (
                scope == "domain_local"
                or _template_is_domain_local(state, original_lanes[0].core_begin, template)
            )
        )
        templates = _select_region_width_templates(
            templates,
            original_widths,
            maximum_templates=templates_per_block,
        )
        if breakdown is not None:
            breakdown.template_ns += time.perf_counter_ns() - template_begin
            breakdown.template_enumerations += 1
            breakdown.templates_selected += len(templates)
        for template in templates:
            beam_begin = time.perf_counter_ns()
            repairs = _beam_assign_tasks(
                tasks,
                template,
                isolated_cost,
                window_selector,
                critical,
                repair_beam_width=repair_beam_width,
            )
            if breakdown is not None:
                breakdown.beam_ns += time.perf_counter_ns() - beam_begin
                breakdown.beam_calls += 1
            for ordering in repairs:
                assemble_begin = time.perf_counter_ns()
                cursor = original_lanes[0].core_begin
                replacement = []
                for width, lane_tasks in zip(template, ordering, strict=True):
                    replacement.append(ExecutableLane(cursor, width, lane_tasks))
                    cursor += width
                candidate = _with_replaced_lanes(
                    state,
                    first,
                    end - first,
                    replacement,
                )
                if breakdown is not None:
                    breakdown.assemble_ns += time.perf_counter_ns() - assemble_begin
                    breakdown.neighbors_emitted += 1
                yield ExecutablePlanNeighbor(
                    _template_lns_operator(scope, destroy_size, repair_beam_width),
                    tuple(task.expert_id for task in tasks),
                    candidate,
                )


def _sample_neighborhood(
    state: ExecutablePlanState,
    neighbors: Iterable[ExecutablePlanNeighbor],
    *,
    operators: Sequence[str],
    per_operator: int,
    seed: int,
    breakdown: LnsSearchBreakdown | None = None,
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
        hash_begin = time.perf_counter_ns()
        state_hash = neighbor.state.canonical_hash()
        if breakdown is not None:
            breakdown.hash_ns += time.perf_counter_ns() - hash_begin
            breakdown.hashed_states += 1
        if state_hash in seen_global:
            duplicates[neighbor.operator] += 1
            if breakdown is not None:
                breakdown.duplicate_hashed_states += 1
            continue
        seen_global.add(state_hash)
        unique[neighbor.operator][state_hash] = neighbor
        if breakdown is not None:
            breakdown.unique_hashed_states += 1

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
        breakdown=None if breakdown is None else breakdown.to_dict(),
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


def sample_template_lns_neighborhood(
    state: ExecutablePlanState,
    *,
    allowed_widths: Iterable[int],
    isolated_cost: Callable[[int, int], float],
    window_selector: Callable[[int, int], tuple[int, int]],
    critical_expert_ids: Iterable[int],
    destroy_sizes: Iterable[int] = TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
    repair_beam_widths: Iterable[int] = TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
    templates_per_block: int = 4,
    per_operator: int,
    seed: int,
) -> SampledNeighborhood:
    """Sample executable multi-lane template repairs around critical experts."""

    sizes = tuple(int(size) for size in destroy_sizes)
    beams = tuple(int(width) for width in repair_beam_widths)
    if len(sizes) != len(beams) or len(set(sizes)) != len(sizes):
        raise ValueError("destroy_sizes must be unique and match repair_beam_widths")
    pairs = tuple(sorted(zip(sizes, beams, strict=True)))
    sizes = tuple(size for size, _ in pairs)
    beams = tuple(beam for _, beam in pairs)
    operators = tuple(
        _template_lns_operator(scope, destroy_size, beam_width)
        for scope in TEMPLATE_LNS_SCOPES
        for destroy_size, beam_width in zip(sizes, beams, strict=True)
    )
    breakdown = LnsSearchBreakdown()
    return _sample_neighborhood(
        state,
        enumerate_template_lns_neighbors(
            state,
            allowed_widths=allowed_widths,
            isolated_cost=isolated_cost,
            window_selector=window_selector,
            critical_expert_ids=critical_expert_ids,
            destroy_sizes=sizes,
            repair_beam_widths=beams,
            templates_per_block=templates_per_block,
            breakdown=breakdown,
        ),
        operators=operators,
        per_operator=per_operator,
        seed=seed,
        breakdown=breakdown,
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


def _lane_signature(lane: ExecutableLane) -> tuple[object, ...]:
    return (
        lane.core_begin,
        lane.threads,
        tuple(
            (
                task.expert_id,
                task.routes,
                task.w13_window_tiles,
                task.w2_window_tiles,
            )
            for task in lane.tasks
        ),
    )


def _lane_isolated_load_ns(model, lane: ExecutableLane) -> float:
    return sum(float(model.T_iso(task.routes, lane.threads)) for task in lane.tasks)


def _lane_summary(model, state: ExecutablePlanState, lane_index: int) -> dict[str, object]:
    lane = state.lanes[lane_index]
    return {
        "lane_index": lane_index,
        "core_begin": lane.core_begin,
        "core_end": lane.core_end,
        "threads": lane.threads,
        "physical_cpu_ids": list(state.thread_cpu_ids[lane.core_begin : lane.core_end]),
        "llc_domain_ids": list(state.lane_domain_ids(lane_index)),
        "isolated_load_ns": _lane_isolated_load_ns(model, lane),
        "task_count": len(lane.tasks),
        "tasks": [
            {
                "position": position,
                "expert_id": task.expert_id,
                "routes": task.routes,
                "isolated_ns": float(model.T_iso(task.routes, lane.threads)),
                "w13_window_tiles": task.w13_window_tiles,
                "w2_window_tiles": task.w2_window_tiles,
            }
            for position, task in enumerate(lane.tasks)
        ],
    }


def _critical_isolated_lane(model, state: ExecutablePlanState) -> dict[str, object]:
    lane_index, lane = max(
        enumerate(state.lanes),
        key=lambda item: (_lane_isolated_load_ns(model, item[1]), -item[0]),
    )
    return {
        "lane_index": lane_index,
        "core_begin": lane.core_begin,
        "threads": lane.threads,
        "isolated_load_ns": _lane_isolated_load_ns(model, lane),
        "tail_expert_id": lane.tasks[-1].expert_id if lane.tasks else None,
    }


def summarize_executable_plan_pair(
    model,
    before: ExecutablePlanState,
    after: ExecutablePlanState,
) -> dict[str, object]:
    """Serialize exactly which lanes and ordered tasks differ between two states."""

    if before.num_threads != after.num_threads or before.thread_cpu_ids != after.thread_cpu_ids:
        raise ValueError("plan-pair context requires the same rank threads and physical CPU mapping")
    before_by_interval = {
        (lane.core_begin, lane.threads): _lane_signature(lane)
        for lane in before.lanes
    }
    after_by_interval = {
        (lane.core_begin, lane.threads): _lane_signature(lane)
        for lane in after.lanes
    }
    before_affected = tuple(
        index
        for index, lane in enumerate(before.lanes)
        if after_by_interval.get((lane.core_begin, lane.threads)) != _lane_signature(lane)
    )
    after_affected = tuple(
        index
        for index, lane in enumerate(after.lanes)
        if before_by_interval.get((lane.core_begin, lane.threads)) != _lane_signature(lane)
    )

    def locations(state: ExecutablePlanState) -> dict[int, tuple[int, int, int, int, int, int]]:
        return {
            task.expert_id: (
                lane.core_begin,
                lane.threads,
                position,
                task.routes,
                task.w13_window_tiles,
                task.w2_window_tiles,
            )
            for lane in state.lanes
            for position, task in enumerate(lane.tasks)
        }

    before_locations = locations(before)
    after_locations = locations(after)
    if before_locations.keys() != after_locations.keys():
        raise ValueError("plan-pair context requires the same active experts")
    changed_experts = tuple(
        expert
        for expert in sorted(before_locations)
        if before_locations[expert] != after_locations[expert]
    )
    affected_experts = tuple(
        sorted(
            {
                task.expert_id
                for index in before_affected
                for task in before.lanes[index].tasks
            }
            | {
                task.expert_id
                for index in after_affected
                for task in after.lanes[index].tasks
            }
        )
    )
    before_critical = _critical_isolated_lane(model, before)
    after_critical = _critical_isolated_lane(model, after)
    affected_intervals = sorted(
        {
            (lane.core_begin, lane.core_end)
            for index, lane in enumerate(before.lanes)
            if index in before_affected
        }
        | {
            (lane.core_begin, lane.core_end)
            for index, lane in enumerate(after.lanes)
            if index in after_affected
        }
    )
    return {
        "changed_expert_ids": list(changed_experts),
        "affected_expert_ids": list(affected_experts),
        "affected_route_counts": [
            {"expert_id": expert, "routes": before_locations[expert][3]}
            for expert in affected_experts
        ],
        "affected_core_intervals": [list(interval) for interval in affected_intervals],
        "before_shape": list(before.shape),
        "after_shape": list(after.shape),
        "before_affected_lanes": [
            _lane_summary(model, before, index)
            for index in before_affected
        ],
        "after_affected_lanes": [
            _lane_summary(model, after, index)
            for index in after_affected
        ],
        "before_critical_isolated_lane": before_critical,
        "after_critical_isolated_lane": after_critical,
        "critical_isolated_lane_switched": (
            before_critical["core_begin"],
            before_critical["threads"],
        )
        != (
            after_critical["core_begin"],
            after_critical["threads"],
        ),
    }


def _width_histogram_key(widths: Sequence[int]) -> str:
    counts = Counter(int(width) for width in widths)
    return "+".join(f"{count}x{width}T" for width, count in sorted(counts.items())) or "idle"


def summarize_placed_event_context(
    state: ExecutablePlanState,
    explanation: Mapping[str, object],
    *,
    affected_expert_ids: Iterable[int],
) -> dict[str, object]:
    """Compress placed event logs around affected experts without storing raw events."""

    focus_experts = frozenset(int(expert) for expert in affected_expert_ids)
    if not focus_experts:
        raise ValueError("placed event context requires at least one affected expert")
    task_layout = [
        (lane_index, lane.threads, task)
        for lane_index, lane in enumerate(state.lanes)
        for task in lane.tasks
    ]
    focus_tasks = {
        index
        for index, (_, _, task) in enumerate(task_layout)
        if task.expert_id in focus_experts
    }
    if not focus_tasks:
        raise ValueError("affected experts are not present in the executable state")
    finish_times = tuple(float(value) for value in explanation["task_finish_ns"])
    if len(finish_times) != len(task_layout):
        raise ValueError("task_finish_ns does not match executable task count")
    raw_events = explanation.get("events")
    if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
        raise TypeError("explanation events must be a sequence")

    makespan_ns = float(explanation["makespan_ns"])
    event_origin_ns = float(raw_events[0]["start_ns"]) if raw_events else 0.0
    modeled_span_ns = max(makespan_ns - event_origin_ns, 0.0)
    head_end_ns = event_origin_ns + 0.2 * modeled_span_ns
    tail_begin_ns = event_origin_ns + 0.8 * modeled_span_ns
    affected_active_ns = 0.0
    affected_solo_ns = 0.0
    affected_head_ns = 0.0
    affected_tail_ns = 0.0
    affected_dilation_time = 0.0
    affected_team_pressure_time = 0.0
    peer_thread_time = 0.0
    max_affected_dilation = 1.0
    phase_overlap_ns: dict[str, float] = defaultdict(float)
    dominant_resource_ns: dict[str, float] = defaultdict(float)
    cohort_transition_ns: dict[str, float] = defaultdict(float)
    cohort_transition_count = 0
    previous_cohort = "idle"
    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            raise TypeError("every explanation event must be a mapping")
        start_ns = float(raw_event["start_ns"])
        duration_ns = float(raw_event["duration_ns"])
        end_ns = start_ns + duration_ns
        active_tasks = tuple(int(value) for value in raw_event["active_tasks"])
        active_focus = tuple(index for index in active_tasks if index in focus_tasks)
        cohort = _width_histogram_key([task_layout[index][1] for index in active_tasks])
        if not active_focus:
            previous_cohort = cohort
            continue
        affected_active_ns += duration_ns
        peer_tasks = tuple(index for index in active_tasks if index not in focus_tasks)
        if not peer_tasks:
            affected_solo_ns += duration_ns
        affected_head_ns += max(min(end_ns, head_end_ns) - start_ns, 0.0)
        affected_tail_ns += max(end_ns - max(start_ns, tail_begin_ns), 0.0)
        peer_thread_time += duration_ns * sum(task_layout[index][1] for index in peer_tasks)

        phase_kinds = raw_event["phase_kinds"]
        dilations = raw_event["phase_dilation"]
        if not isinstance(phase_kinds, Mapping) or not isinstance(dilations, Mapping):
            raise TypeError("event phase kinds and dilations must be mappings")
        focus_kinds = sorted({str(phase_kinds[str(index)]) for index in active_focus})
        peer_kinds = sorted({str(phase_kinds[str(index)]) for index in peer_tasks})
        overlap_key = f"{'+'.join(focus_kinds)}|{'+'.join(peer_kinds) if peer_kinds else 'none'}"
        phase_overlap_ns[overlap_key] += duration_ns
        focus_dilations = [float(dilations[str(index)]) for index in active_focus]
        mean_dilation = sum(focus_dilations) / len(focus_dilations)
        affected_dilation_time += duration_ns * mean_dilation
        max_affected_dilation = max(max_affected_dilation, *focus_dilations)

        team_pressure = raw_event.get("team_pressure_dilation", {})
        if isinstance(team_pressure, Mapping):
            values = [float(team_pressure.get(str(index), 1.0)) for index in active_focus]
            affected_team_pressure_time += duration_ns * sum(values) / len(values)

        resources = raw_event.get("resources", {})
        if isinstance(resources, Mapping) and resources:
            resource, _ = max(
                resources.items(),
                key=lambda item: float(item[1].get("dilation", 1.0)),
            )
            dominant_resource_ns[str(resource)] += duration_ns
        else:
            dominant_resource_ns["none"] += duration_ns

        if cohort != previous_cohort:
            cohort_transition_count += 1
            cohort_transition_ns[f"{previous_cohort}->{cohort}"] += duration_ns
        previous_cohort = cohort

    lane_finish = []
    cursor = 0
    for lane_index, lane in enumerate(state.lanes):
        task_indices = list(range(cursor, cursor + len(lane.tasks)))
        cursor += len(lane.tasks)
        if not task_indices:
            continue
        lane_finish.append(
            {
                "lane_index": lane_index,
                "core_begin": lane.core_begin,
                "threads": lane.threads,
                "finish_ns": finish_times[task_indices[-1]],
                "tail_expert_id": lane.tasks[-1].expert_id,
                "affected": any(index in focus_tasks for index in task_indices),
            }
        )
    critical_lane = max(lane_finish, key=lambda row: (float(row["finish_ns"]), -int(row["lane_index"])))
    critical_task = max(range(len(finish_times)), key=lambda index: (finish_times[index], -index))
    critical_task_record = task_layout[critical_task]
    return {
        "makespan_ns": makespan_ns,
        "event_count": len(raw_events),
        "affected_task_count": len(focus_tasks),
        "affected_active_ns": affected_active_ns,
        "affected_solo_ns": affected_solo_ns,
        "affected_head_ns": affected_head_ns,
        "affected_tail_ns": affected_tail_ns,
        "mean_affected_phase_dilation": (
            affected_dilation_time / affected_active_ns if affected_active_ns else 1.0
        ),
        "mean_affected_team_pressure_dilation": (
            affected_team_pressure_time / affected_active_ns if affected_active_ns else 1.0
        ),
        "max_affected_phase_dilation": max_affected_dilation,
        "mean_peer_threads_while_affected": (
            peer_thread_time / affected_active_ns if affected_active_ns else 0.0
        ),
        "phase_overlap_ns": dict(sorted(phase_overlap_ns.items())),
        "dominant_resource_ns": dict(sorted(dominant_resource_ns.items())),
        "cohort_transition_count": cohort_transition_count,
        "cohort_transition_ns": dict(sorted(cohort_transition_ns.items())),
        "critical_lane": critical_lane,
        "critical_task": {
            "task_index": critical_task,
            "lane_index": critical_task_record[0],
            "threads": critical_task_record[1],
            "expert_id": critical_task_record[2].expert_id,
            "routes": critical_task_record[2].routes,
            "finish_ns": finish_times[critical_task],
            "affected": critical_task in focus_tasks,
        },
        "affected_lane_finishes": [row for row in lane_finish if row["affected"]],
    }


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
    "TEMPLATE_LNS_DEFAULT_DESTROY_SIZES",
    "TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS",
    "TEMPLATE_LNS_OPERATORS",
    "TEMPLATE_LNS_REPARTITION",
    "TEMPLATE_LNS_SCOPES",
    "WIDTH_ONLY_OPERATORS",
    "ExecutablePlanNeighbor",
    "ExecutablePlanEvaluator",
    "ExecutablePlanScore",
    "ExecutablePlanScreenScore",
    "SampledNeighborhood",
    "critical_expert_scores",
    "enumerate_order_only_neighbors",
    "enumerate_template_lns_neighbors",
    "enumerate_width_only_neighbors",
    "placed_tasks",
    "is_resolvable_improvement",
    "sample_order_only_neighborhood",
    "sample_combined_neighborhood",
    "sample_template_lns_neighborhood",
    "sample_width_only_neighborhood",
    "score_executable_plan",
    "summarize_executable_plan_pair",
    "summarize_placed_event_context",
]
