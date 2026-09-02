"""LLC-domain shortlist extension for the offline cold-phase CP-SAT oracle.

The master searches only whole-expert width, LLC domain, and start/order.  It
keeps the cold-phase oracle's fixed-rate CPU and DRAM surrogate; stage windows
remain outside this model and are assigned by the existing deterministic
window policy after a candidate is lowered to an executable task DAG.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from cold_phase_cp_sat_oracle import (
    ColdPhaseAssignment,
    ColdPhaseJob,
    ColdPhaseOracleResult,
    ColdPhaseRuntimePlacement,
    ColdPhaseRuntimeTask,
    PhaseAssignment,
    _cp_model_module,
    _critical_path_bound,
    _normalize_dependencies,
    _normalize_jobs,
    _quantize_jobs,
    solve_cold_phase_cp_sat,
)


@dataclass(frozen=True)
class LlcDomain:
    domain_id: str
    core_begin: int
    core_count: int


@dataclass(frozen=True)
class DomainAssignment:
    domain_id: str
    assignment: ColdPhaseAssignment

    @property
    def expert_id(self) -> int:
        return self.assignment.expert_id


@dataclass(frozen=True)
class DomainCandidate:
    objective_ns: int
    assignments: tuple[DomainAssignment, ...]

    def signature(self) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            sorted(
                (item.expert_id, item.assignment.threads, item.domain_id)
                for item in self.assignments
            )
        )

    def schedule_signature(self) -> tuple[tuple[int, int, str, int], ...]:
        return tuple(
            sorted(
                (
                    item.expert_id,
                    item.assignment.threads,
                    item.domain_id,
                    item.assignment.start_ns,
                )
                for item in self.assignments
            )
        )

    def mode_histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(item.assignment.threads for item in self.assignments).items()))

    def domain_histogram(self) -> dict[str, int]:
        return dict(sorted(Counter(item.domain_id for item in self.assignments).items()))

    def to_dict(self, *, include_phases: bool = False) -> dict[str, object]:
        rows = []
        for item in self.assignments:
            row = asdict(item.assignment)
            row["domain_id"] = item.domain_id
            if not include_phases:
                row.pop("phases")
            rows.append(row)
        return {
            "objective_ns": self.objective_ns,
            "mode_histogram": self.mode_histogram(),
            "domain_histogram": self.domain_histogram(),
            "assignments": rows,
        }


@dataclass(frozen=True)
class DomainShortlistResult:
    status: str
    optimal: bool
    root_objective_ns: int | None
    root_best_bound_ns: float | None
    root_relative_gap: float | None
    planning_wall_time_s: float
    solver_wall_time_s: float
    requested_solutions: int
    domains: tuple[LlcDomain, ...]
    candidates: tuple[DomainCandidate, ...]

    def to_dict(self, *, include_phases: bool = False) -> dict[str, object]:
        payload = asdict(self)
        payload["candidates"] = [
            candidate.to_dict(include_phases=include_phases) for candidate in self.candidates
        ]
        return payload


@dataclass(frozen=True)
class StrictGreedyUnionResult:
    selected_branch: str | None
    objective_ns: int | None
    best_bound_ns: float | None
    relative_gap: float | None
    greedy_objective_ns: int | None
    greedy_best_bound_ns: float | None
    sat_objective_ns: int | None
    sat_best_bound_ns: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fixed_strict_jobs(
    jobs: Sequence[ColdPhaseJob],
    tasks: Sequence[tuple[int, int, int, int, Sequence[int]]],
    *,
    num_cores: int,
) -> tuple[tuple[ColdPhaseJob, ...], tuple[tuple[int, ...], ...]]:
    if not tasks:
        raise ValueError("strict greedy plan must contain at least one task")
    by_expert = {job.expert_id: job for job in jobs}
    if len(by_expert) != len(jobs):
        raise ValueError("cold-phase jobs contain duplicate expert ids")
    if len({int(task[0]) for task in tasks}) != len(tasks):
        raise ValueError("strict greedy plan requires one task per active expert")
    if {int(task[0]) for task in tasks} != set(by_expert):
        raise ValueError("strict greedy plan must cover exactly the cold-phase jobs")

    fixed_jobs = []
    dependencies = []
    core_intervals = []
    for task_index, raw_task in enumerate(tasks):
        expert_id, routes, core_begin, threads, raw_dependencies = raw_task
        expert_id = int(expert_id)
        routes = int(routes)
        core_begin = int(core_begin)
        threads = int(threads)
        if routes != by_expert[expert_id].routes:
            raise ValueError(f"strict greedy routes mismatch for expert_id={expert_id}")
        if threads <= 0 or core_begin < 0 or core_begin + threads > num_cores:
            raise ValueError(f"strict greedy placement is out of range for expert_id={expert_id}")
        modes = tuple(mode for mode in by_expert[expert_id].modes if mode.threads == threads)
        if len(modes) != 1:
            raise ValueError(f"strict greedy width={threads} is unavailable for expert_id={expert_id}")
        row = tuple(sorted({int(dependency) for dependency in raw_dependencies}))
        if row and (row[0] < 0 or row[-1] >= task_index):
            raise ValueError("strict greedy dependencies must refer to earlier tasks")
        fixed_jobs.append(ColdPhaseJob(expert_id, routes, modes))
        dependencies.append(row)
        core_intervals.append((core_begin, core_begin + threads))

    ancestors: list[set[int]] = []
    for task_index, row in enumerate(dependencies):
        reachable = set(row)
        for dependency in row:
            reachable.update(ancestors[dependency])
        ancestors.append(reachable)
        for earlier in range(task_index):
            left_begin, left_end = core_intervals[earlier]
            right_begin, right_end = core_intervals[task_index]
            overlaps = left_begin < right_end and right_begin < left_end
            if overlaps and earlier not in reachable:
                raise ValueError(
                    "overlapping strict greedy placements must be ordered by dependencies: "
                    f"earlier_task={earlier}, task={task_index}"
                )
    return tuple(fixed_jobs), tuple(dependencies)


def solve_fixed_strict_greedy_cp_sat(
    jobs: Sequence[ColdPhaseJob],
    tasks: Sequence[tuple[int, int, int, int, Sequence[int]]],
    *,
    num_cores: int,
    dram_bandwidth_gbps: float,
    max_time_s: float = 60.0,
    workers: int = 1,
    relative_gap_limit: float = 0.0,
    time_quantum_ns: int = 1000,
    bandwidth_quantum_gbps: float = 0.25,
    cold_phase_slots: int | None = None,
    random_seed: int = 0,
) -> ColdPhaseOracleResult:
    """Optimize timing for one exact strict greedy width/placement/order DAG."""

    fixed_jobs, dependencies = _fixed_strict_jobs(jobs, tasks, num_cores=num_cores)
    return solve_cold_phase_cp_sat(
        fixed_jobs,
        num_cores=num_cores,
        dram_bandwidth_gbps=dram_bandwidth_gbps,
        dependencies=dependencies,
        max_time_s=max_time_s,
        workers=workers,
        relative_gap_limit=relative_gap_limit,
        time_quantum_ns=time_quantum_ns,
        bandwidth_quantum_gbps=bandwidth_quantum_gbps,
        cold_phase_slots=cold_phase_slots,
        random_seed=random_seed,
    )


def compare_strict_greedy_union(
    greedy,
    sat: DomainShortlistResult,
) -> StrictGreedyUnionResult:
    """Combine disjoint fixed-greedy and domain-SAT branches exactly."""

    upper_rows = [
        ("greedy_strict", greedy.objective_ns),
        ("domain_sat", sat.root_objective_ns),
    ]
    feasible = [(name, value) for name, value in upper_rows if value is not None]
    selected_branch = min(feasible, key=lambda row: (row[1], row[0]))[0] if feasible else None
    objective_ns = min((value for _, value in feasible), default=None)
    bounds = [value for value in (greedy.best_bound_ns, sat.root_best_bound_ns) if value is not None]
    best_bound_ns = min(bounds) if len(bounds) == 2 else None
    relative_gap = None
    if objective_ns is not None and best_bound_ns is not None:
        relative_gap = max((objective_ns - best_bound_ns) / max(abs(objective_ns), 1.0), 0.0)
    return StrictGreedyUnionResult(
        selected_branch=selected_branch,
        objective_ns=objective_ns,
        best_bound_ns=best_bound_ns,
        relative_gap=relative_gap,
        greedy_objective_ns=greedy.objective_ns,
        greedy_best_bound_ns=greedy.best_bound_ns,
        sat_objective_ns=sat.root_objective_ns,
        sat_best_bound_ns=sat.root_best_bound_ns,
    )


def _normalize_domains(domains: Sequence[LlcDomain], *, num_cores: int) -> tuple[LlcDomain, ...]:
    if not domains:
        raise ValueError("at least one LLC domain is required")
    normalized = tuple(sorted(domains, key=lambda item: (item.core_begin, item.domain_id)))
    if len({item.domain_id for item in normalized}) != len(normalized):
        raise ValueError("LLC domain ids must be unique")
    cursor = 0
    for domain in normalized:
        if not domain.domain_id:
            raise ValueError("LLC domain id must be non-empty")
        if domain.core_count <= 0:
            raise ValueError("LLC domain core_count must be positive")
        if domain.core_begin != cursor:
            raise ValueError("LLC domains must form a contiguous partition starting at core zero")
        cursor += domain.core_count
    if cursor != num_cores:
        raise ValueError(f"LLC domains cover {cursor} cores, expected {num_cores}")
    return normalized


def solve_cold_phase_domain_shortlist(
    jobs: Sequence[ColdPhaseJob],
    *,
    num_cores: int,
    domains: Sequence[LlcDomain],
    dram_bandwidth_gbps: float,
    dependencies: Sequence[Sequence[int]] | None = None,
    initial_assignments: Sequence[DomainAssignment] | None = None,
    solution_limit: int = 32,
    max_signature_changes: int = 8,
    shortlist_objective_slack: float = 0.10,
    max_time_s: float = 60.0,
    subsequent_time_s: float = 5.0,
    workers: int = 1,
    relative_gap_limit: float = 0.01,
    time_quantum_ns: int = 1000,
    bandwidth_quantum_gbps: float = 0.25,
    cold_phase_slots: int | None = None,
    random_seed: int = 0,
) -> DomainShortlistResult:
    """Return diverse near-optimal width/domain assignments from one master.

    The root solve owns the reported optimality bound.  Later solves add one
    no-good cut per selected width/domain vector and are constrained to remain
    within ``shortlist_objective_slack`` of the root incumbent.
    """

    if solution_limit <= 0:
        raise ValueError("solution_limit must be positive")
    if max_signature_changes <= 0:
        raise ValueError("max_signature_changes must be positive")
    if not 0.0 <= shortlist_objective_slack < 1.0:
        raise ValueError("shortlist_objective_slack must be in [0, 1)")
    if max_time_s <= 0.0 or subsequent_time_s <= 0.0:
        raise ValueError("solver time limits must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 0.0 <= relative_gap_limit < 1.0:
        raise ValueError("relative_gap_limit must be in [0, 1)")
    if dram_bandwidth_gbps <= 0.0 or bandwidth_quantum_gbps <= 0.0:
        raise ValueError("bandwidth values must be positive")
    if cold_phase_slots is not None and cold_phase_slots <= 0:
        raise ValueError("cold_phase_slots must be positive when set")

    normalized_domains = _normalize_domains(domains, num_cores=num_cores)
    normalized_jobs, _ = _normalize_jobs(jobs, num_cores=num_cores)
    normalized_dependencies, topological_order = _normalize_dependencies(
        len(normalized_jobs), dependencies
    )
    capacity_units = int(math.floor(dram_bandwidth_gbps / bandwidth_quantum_gbps + 1e-12))
    if capacity_units <= 0:
        raise ValueError("dram bandwidth is smaller than one bandwidth quantum")
    effective_bandwidth = capacity_units * bandwidth_quantum_gbps
    quantized = _quantize_jobs(
        normalized_jobs,
        time_quantum_ns=time_quantum_ns,
        bandwidth_quantum_gbps=bandwidth_quantum_gbps,
        bandwidth_capacity_units=capacity_units,
    )
    largest_domain = max(domain.core_count for domain in normalized_domains)
    eligible_modes = [
        tuple(mode for mode in modes if mode.source.threads <= largest_domain)
        for modes in quantized
    ]
    for job, modes in zip(normalized_jobs, eligible_modes, strict=True):
        if not modes:
            raise ValueError(f"expert_id={job.expert_id} has no width fitting any LLC domain")
    minimum_durations = [min(mode.service_ticks for mode in modes) for modes in eligible_modes]
    horizon = sum(minimum_durations)
    if horizon <= 0 or horizon >= (1 << 62):
        raise ValueError(f"invalid CP-SAT horizon={horizon} ticks")

    cp_model = _cp_model_module()
    model = cp_model.CpModel()
    domain_intervals: dict[str, list[object]] = {domain.domain_id: [] for domain in normalized_domains}
    domain_demands: dict[str, list[int]] = {domain.domain_id: [] for domain in normalized_domains}
    cold_intervals = []
    cold_demands = []
    cold_slot_intervals = []
    variables = []
    job_starts = []
    job_ends = []
    min_core_area = 0
    min_cold_bytes = 0

    for job_index, (job, modes) in enumerate(zip(normalized_jobs, quantized, strict=True)):
        choices = []
        job_start = model.new_int_var(0, horizon, f"job_start_e{job.expert_id}_j{job_index}")
        job_end = model.new_int_var(0, horizon, f"job_end_e{job.expert_id}_j{job_index}")
        feasible_modes = [mode for mode in modes if mode.source.threads <= largest_domain]
        min_core_area += min(mode.source.threads * mode.service_ticks for mode in feasible_modes)
        min_cold_bytes += min(
            sum(phase.source.cold_weight_bytes for phase in mode.phases) for mode in feasible_modes
        )
        for mode_index, mode in enumerate(modes):
            for domain_index, domain in enumerate(normalized_domains):
                threads = mode.source.threads
                if threads > domain.core_count:
                    continue
                suffix = f"e{job.expert_id}_j{job_index}_m{mode_index}_d{domain_index}"
                presence = model.new_bool_var(f"select_{suffix}")
                start = model.new_int_var(0, horizon, f"start_{suffix}")
                end = model.new_int_var(0, horizon, f"end_{suffix}")
                size = model.new_int_var(0, horizon, f"residence_{suffix}")
                master = model.new_optional_interval_var(start, size, end, presence, f"run_{suffix}")
                model.add(start == 0).only_enforce_if(presence.Not())
                model.add(end == 0).only_enforce_if(presence.Not())
                model.add(size == 0).only_enforce_if(presence.Not())
                model.add(size >= mode.service_ticks).only_enforce_if(presence)
                model.add(job_start == start).only_enforce_if(presence)
                model.add(job_end == end).only_enforce_if(presence)
                domain_intervals[domain.domain_id].append(master)
                domain_demands[domain.domain_id].append(threads)

                phase_variables = []
                previous_end = None
                for phase_index, phase in enumerate(mode.phases):
                    phase_start = model.new_int_var(0, horizon, f"phase_start_{suffix}_p{phase_index}")
                    phase_end = model.new_int_var(0, horizon, f"phase_end_{suffix}_p{phase_index}")
                    interval = model.new_optional_interval_var(
                        phase_start,
                        phase.duration_ticks,
                        phase_end,
                        presence,
                        f"phase_{suffix}_p{phase_index}",
                    )
                    model.add(phase_start == 0).only_enforce_if(presence.Not())
                    model.add(phase_end == 0).only_enforce_if(presence.Not())
                    if previous_end is None:
                        model.add(phase_start == start).only_enforce_if(presence)
                    else:
                        model.add(phase_start >= previous_end).only_enforce_if(presence)
                    previous_end = phase_end
                    if phase.bandwidth_units > 0:
                        cold_intervals.append(interval)
                        cold_demands.append(phase.bandwidth_units)
                        cold_slot_intervals.append(interval)
                    phase_variables.append((phase, phase_start, phase_end))
                model.add(end == previous_end).only_enforce_if(presence)
                choices.append((mode_index, domain, mode, presence, start, size, end, phase_variables))
        if not choices:
            raise ValueError(f"expert_id={job.expert_id} has no width fitting any LLC domain")
        model.add_exactly_one([choice[3] for choice in choices])
        variables.append(choices)
        job_starts.append(job_start)
        job_ends.append(job_end)

    for domain in normalized_domains:
        model.add_cumulative(
            domain_intervals[domain.domain_id],
            domain_demands[domain.domain_id],
            domain.core_count,
        )
    model.add_cumulative(cold_intervals, cold_demands, capacity_units)
    if cold_phase_slots is not None:
        model.add_cumulative(cold_slot_intervals, [1] * len(cold_slot_intervals), cold_phase_slots)
    for job_index, row in enumerate(normalized_dependencies):
        for dependency in row:
            model.add(job_starts[job_index] >= job_ends[dependency])

    model.add_min_equality(model.new_int_var(0, 0, "first_start"), job_starts)
    area_bound = math.ceil(min_core_area / num_cores)
    bandwidth_bound = math.ceil(min_cold_bytes / (effective_bandwidth * time_quantum_ns))
    critical_path_bound = _critical_path_bound(
        minimum_durations, normalized_dependencies, topological_order
    )
    makespan = model.new_int_var(max(area_bound, bandwidth_bound, critical_path_bound), horizon, "makespan")
    model.add_max_equality(makespan, job_ends)
    model.minimize(makespan)

    hint_by_expert = None
    if initial_assignments is not None:
        hint_by_expert = {item.expert_id: item for item in initial_assignments}
        if len(hint_by_expert) != len(initial_assignments):
            raise ValueError("initial_assignments contains duplicate expert ids")
        if set(hint_by_expert) != {job.expert_id for job in normalized_jobs}:
            raise ValueError("initial_assignments must cover exactly the oracle jobs")
    cursor = 0
    for job_index in topological_order:
        job = normalized_jobs[job_index]
        choices = variables[job_index]
        if hint_by_expert is None:
            selected = min(choices, key=lambda item: (item[2].service_ticks, item[2].source.threads))
            start_hint = cursor
            phase_rows = []
            for phase in selected[2].phases:
                phase_rows.append((cursor, cursor + phase.duration_ticks))
                cursor += phase.duration_ticks
            end_hint = cursor
        else:
            hinted = hint_by_expert[job.expert_id]
            matching = [
                choice
                for choice in choices
                if choice[1].domain_id == hinted.domain_id
                and choice[2].source.threads == hinted.assignment.threads
            ]
            if len(matching) != 1:
                raise ValueError(f"initial width/domain is unavailable for expert_id={job.expert_id}")
            selected = matching[0]
            start_hint = hinted.assignment.start_ns // time_quantum_ns
            end_hint = hinted.assignment.end_ns // time_quantum_ns
            phase_rows = [
                (phase.start_ns // time_quantum_ns, phase.end_ns // time_quantum_ns)
                for phase in hinted.assignment.phases
            ]
            if len(phase_rows) != len(selected[7]):
                raise ValueError(f"initial phase count mismatch for expert_id={job.expert_id}")
            cursor = max(cursor, end_hint)
        for choice in choices:
            is_selected = choice is selected
            _, _, _, presence, start, size, end, phase_variables = choice
            model.add_hint(presence, int(is_selected))
            model.add_hint(start, start_hint if is_selected else 0)
            model.add_hint(size, end_hint - start_hint if is_selected else 0)
            model.add_hint(end, end_hint if is_selected else 0)
            for phase_index, (_, phase_start, phase_end) in enumerate(phase_variables):
                row = phase_rows[phase_index] if is_selected else (0, 0)
                model.add_hint(phase_start, row[0])
                model.add_hint(phase_end, row[1])
        model.add_hint(job_starts[job_index], start_hint)
        model.add_hint(job_ends[job_index], end_hint)
    model.add_hint(makespan, cursor)
    if hint_by_expert is not None:
        model.add(makespan <= cursor)

    def extract(solver) -> DomainCandidate:
        assignments = []
        for job, choices in zip(normalized_jobs, variables, strict=True):
            selected = [choice for choice in choices if solver.boolean_value(choice[3])]
            if len(selected) != 1:
                raise RuntimeError(f"CP-SAT selected {len(selected)} choices for expert_id={job.expert_id}")
            _, domain, mode, _, start, _, end, phase_variables = selected[0]
            phases = []
            service_ticks = 0
            for phase, phase_start, phase_end in phase_variables:
                service_ticks += phase.duration_ticks
                phases.append(
                    PhaseAssignment(
                        name=phase.source.name,
                        start_ns=int(solver.value(phase_start) * time_quantum_ns),
                        duration_ns=phase.duration_ticks * time_quantum_ns,
                        end_ns=int(solver.value(phase_end) * time_quantum_ns),
                        cold_weight_bytes=phase.source.cold_weight_bytes,
                        reserved_dram_gbps=phase.bandwidth_units * bandwidth_quantum_gbps,
                    )
                )
            start_ns = int(solver.value(start) * time_quantum_ns)
            end_ns = int(solver.value(end) * time_quantum_ns)
            service_ns = service_ticks * time_quantum_ns
            assignments.append(
                DomainAssignment(
                    domain.domain_id,
                    ColdPhaseAssignment(
                        job.expert_id,
                        job.routes,
                        mode.source.threads,
                        start_ns,
                        service_ns,
                        max(end_ns - start_ns - service_ns, 0),
                        end_ns,
                        tuple(phases),
                    ),
                )
            )
        assignments.sort(key=lambda item: (item.assignment.start_ns, item.assignment.end_ns, item.expert_id))
        return DomainCandidate(int(solver.value(makespan) * time_quantum_ns), tuple(assignments))

    begin = time.perf_counter()
    candidates = []
    signatures = set()
    root_status = "UNKNOWN"
    root_objective_ns = None
    root_best_bound_ns = None
    root_relative_gap = None
    root_optimal = False
    solver_wall_time_s = 0.0
    root_limit_ticks = None

    def signature_literals(candidate: DomainCandidate) -> list[object]:
        selected_by_expert = set(candidate.signature())
        selected_literals = []
        for job, choices in zip(normalized_jobs, variables, strict=True):
            for _, domain, mode, presence, *_ in choices:
                if (job.expert_id, mode.source.threads, domain.domain_id) in selected_by_expert:
                    selected_literals.append(presence)
                    break
        return selected_literals

    def add_schedule_cut(candidate: DomainCandidate) -> None:
        by_expert = {item.expert_id: item.assignment for item in candidate.assignments}
        model.add_forbidden_assignments(
            job_starts,
            [[by_expert[job.expert_id].start_ns // time_quantum_ns for job in normalized_jobs]],
        )

    def fix_repair_neighborhood(candidate: DomainCandidate) -> None:
        mutable_experts = {
            job.expert_id
            for job in sorted(
                normalized_jobs,
                key=lambda item: (-item.routes, item.expert_id),
            )[: min(max_signature_changes, len(normalized_jobs))]
        }
        selected_by_expert = set(candidate.signature())
        assignment_by_expert = {item.expert_id: item for item in candidate.assignments}
        for job_index, (job, choices) in enumerate(zip(normalized_jobs, variables, strict=True)):
            fixed = assignment_by_expert[job.expert_id].assignment
            for _, domain, mode, presence, start, size, end, phase_variables in choices:
                selected = (job.expert_id, mode.source.threads, domain.domain_id) in selected_by_expert
                model.add_hint(presence, int(selected))
                model.add_hint(start, fixed.start_ns // time_quantum_ns if selected else 0)
                model.add_hint(
                    size,
                    (fixed.end_ns - fixed.start_ns) // time_quantum_ns if selected else 0,
                )
                model.add_hint(end, fixed.end_ns // time_quantum_ns if selected else 0)
                if selected and job.expert_id not in mutable_experts:
                    model.add(presence == 1)
                    model.add(start == fixed.start_ns // time_quantum_ns)
                    model.add(end == fixed.end_ns // time_quantum_ns)
                if selected and len(phase_variables) != len(fixed.phases):
                    raise ValueError(f"repair phase count mismatch for expert_id={job.expert_id}")
                for phase_index, (_, phase_start, phase_end) in enumerate(phase_variables):
                    fixed_phase = fixed.phases[phase_index] if selected else None
                    model.add_hint(
                        phase_start,
                        fixed_phase.start_ns // time_quantum_ns if fixed_phase is not None else 0,
                    )
                    model.add_hint(
                        phase_end,
                        fixed_phase.end_ns // time_quantum_ns if fixed_phase is not None else 0,
                    )
                    if fixed_phase is not None and job.expert_id not in mutable_experts:
                        model.add(phase_start == fixed_phase.start_ns // time_quantum_ns)
                        model.add(phase_end == fixed_phase.end_ns // time_quantum_ns)
            model.add_hint(job_starts[job_index], fixed.start_ns // time_quantum_ns)
            model.add_hint(job_ends[job_index], fixed.end_ns // time_quantum_ns)
        model.add_hint(makespan, candidate.objective_ns // time_quantum_ns)

    for solution_index in range(solution_limit):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(max_time_s if solution_index == 0 else subsequent_time_s)
        solver.parameters.num_search_workers = int(workers if solution_index == 0 else 1)
        solver.parameters.relative_gap_limit = float(relative_gap_limit if solution_index == 0 else 0.0)
        solver.parameters.random_seed = int(random_seed + solution_index)
        if solution_index == 0 and initial_assignments is not None:
            solver.parameters.hint_conflict_limit = 100_000
        elif solution_index > 0:
            solver.parameters.repair_hint = True
            solver.parameters.hint_conflict_limit = 100_000
        status_code = solver.solve(model)
        solver_wall_time_s += float(solver.wall_time)
        status_names = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.OPTIMAL: "OPTIMAL",
        }
        status = status_names.get(status_code, f"STATUS_{status_code}")
        if solution_index == 0:
            root_status = status
            if status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                root_objective_ns = int(solver.value(makespan) * time_quantum_ns)
                root_best_bound_ns = float(solver.best_objective_bound) * time_quantum_ns
                root_relative_gap = max(
                    (root_objective_ns - root_best_bound_ns) / max(abs(root_objective_ns), 1.0),
                    0.0,
                )
                root_optimal = status_code == cp_model.OPTIMAL and root_relative_gap < 0.5e-12
                root_limit_ticks = math.ceil(
                    solver.value(makespan) * (1.0 + shortlist_objective_slack)
                )
                model.add(makespan <= root_limit_ticks)
            elif initial_assignments is not None and status_code == cp_model.UNKNOWN:
                incumbent = DomainCandidate(
                    objective_ns=max(item.assignment.end_ns for item in initial_assignments),
                    assignments=tuple(initial_assignments),
                )
                root_status = "UNKNOWN_WITH_INCUMBENT"
                root_objective_ns = incumbent.objective_ns
                root_best_bound_ns = float(solver.best_objective_bound) * time_quantum_ns
                root_relative_gap = max(
                    (root_objective_ns - root_best_bound_ns) / max(abs(root_objective_ns), 1.0),
                    0.0,
                )
                root_limit_ticks = math.ceil(
                    root_objective_ns / time_quantum_ns * (1.0 + shortlist_objective_slack)
                )
                model.add(makespan <= root_limit_ticks)
                signatures.add(incumbent.schedule_signature())
                candidates.append(incumbent)
                model.clear_hints()
                fix_repair_neighborhood(incumbent)
                add_schedule_cut(incumbent)
                continue
        if status_code not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            break
        candidate = extract(solver)
        signature = candidate.schedule_signature()
        if signature in signatures:
            raise RuntimeError("shortlist solver returned a duplicate schedule signature")
        signatures.add(signature)
        candidates.append(candidate)
        if solution_index == 0:
            model.clear_hints()
            fix_repair_neighborhood(candidate)
        add_schedule_cut(candidate)

    return DomainShortlistResult(
        status=root_status,
        optimal=root_optimal,
        root_objective_ns=root_objective_ns,
        root_best_bound_ns=root_best_bound_ns,
        root_relative_gap=root_relative_gap,
        planning_wall_time_s=time.perf_counter() - begin,
        solver_wall_time_s=solver_wall_time_s,
        requested_solutions=solution_limit,
        domains=normalized_domains,
        candidates=tuple(candidates),
    )


def materialize_domain_candidate(
    candidate: DomainCandidate,
    *,
    domains: Sequence[LlcDomain],
    max_time_s: float = 30.0,
    workers: int = 1,
    random_seed: int = 0,
) -> ColdPhaseRuntimePlacement:
    """Lower one fluid domain schedule to contiguous teams and dependency edges."""

    if not candidate.assignments:
        raise ValueError("candidate must contain at least one assignment")
    normalized_domains = _normalize_domains(
        domains,
        num_cores=sum(domain.core_count for domain in domains),
    )
    by_id = {domain.domain_id: domain for domain in normalized_domains}
    cp_model = _cp_model_module()
    model = cp_model.CpModel()
    variables = []
    for task_id, item in enumerate(candidate.assignments):
        domain = by_id.get(item.domain_id)
        if domain is None:
            raise ValueError(f"unknown LLC domain {item.domain_id!r}")
        assignment = item.assignment
        if assignment.threads > domain.core_count:
            raise ValueError(f"expert_id={item.expert_id} width exceeds its LLC domain")
        core_begin = model.new_int_var(
            domain.core_begin,
            domain.core_begin + domain.core_count - assignment.threads,
            f"core_begin_t{task_id}_e{item.expert_id}",
        )
        time_interval = model.new_fixed_size_interval_var(
            assignment.start_ns,
            assignment.end_ns - assignment.start_ns,
            f"time_t{task_id}_e{item.expert_id}",
        )
        core_interval = model.new_fixed_size_interval_var(
            core_begin,
            assignment.threads,
            f"cores_t{task_id}_e{item.expert_id}",
        )
        variables.append((item, core_begin, time_interval, core_interval))
    model.add_no_overlap_2d(
        [row[2] for row in variables],
        [row[3] for row in variables],
    )
    model.add_decision_strategy(
        [row[1] for row in variables],
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MIN_VALUE,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_s)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    status_code = solver.solve(model)
    status_names = {
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.OPTIMAL: "OPTIMAL",
    }
    status = status_names.get(status_code, f"STATUS_{status_code}")
    num_cores = sum(domain.core_count for domain in normalized_domains)
    if status_code not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return ColdPhaseRuntimePlacement(status, float(solver.wall_time), num_cores, ())

    ordered = sorted(
        ((row[0], int(solver.value(row[1]))) for row in variables),
        key=lambda row: (row[0].assignment.start_ns, row[0].assignment.end_ns, row[0].expert_id),
    )
    previous_by_core = [-1] * num_cores
    tasks = []
    origin = min(item.assignment.start_ns for item, _ in ordered)
    for task_id, (item, core_begin) in enumerate(ordered):
        assignment = item.assignment
        core_end = core_begin + assignment.threads
        dependencies = tuple(sorted({value for value in previous_by_core[core_begin:core_end] if value >= 0}))
        tasks.append(
            ColdPhaseRuntimeTask(
                assignment.expert_id,
                assignment.routes,
                assignment.threads,
                core_begin,
                0,
                assignment.end_ns - origin,
                dependencies,
            )
        )
        previous_by_core[core_begin:core_end] = [task_id] * assignment.threads
    return ColdPhaseRuntimePlacement(status, float(solver.wall_time), num_cores, tuple(tasks))
