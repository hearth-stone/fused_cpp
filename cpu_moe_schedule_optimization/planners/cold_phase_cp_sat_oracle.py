"""Offline cold-weight CP-SAT oracle for CPU MoE expert scheduling.

This module refines the fixed-duration isolated oracle by expanding each
expert into ordered cold/steady packed-B phases. The expert keeps its selected
team for its entire lifetime, including resource waits, while cold phases
reserve rank-local DRAM bandwidth. It remains an offline surrogate: A traffic,
stores, shared-cache refills, merge work, and contention-dependent compute
rates are intentionally outside the first model.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence


class CpSatUnavailableError(RuntimeError):
    """Raised when the optional OR-Tools dependency is unavailable."""


class StagePhaseModel(Protocol):
    def T_iso(self, routes: int, threads: int) -> float: ...

    def task_stage_phases(
        self,
        stage: str,
        routes: int,
        threads: int,
    ) -> Sequence[tuple[float, int]]: ...


@dataclass(frozen=True)
class ColdPhase:
    name: str
    duration_ns: float
    cold_weight_bytes: int = 0

    @property
    def is_cold(self) -> bool:
        return self.cold_weight_bytes > 0


@dataclass(frozen=True)
class ColdPhaseMode:
    threads: int
    phases: tuple[ColdPhase, ...]

    @property
    def duration_ns(self) -> float:
        return sum(phase.duration_ns for phase in self.phases)


@dataclass(frozen=True)
class ColdPhaseJob:
    expert_id: int
    routes: int
    modes: tuple[ColdPhaseMode, ...]


@dataclass(frozen=True)
class PhaseAssignment:
    name: str
    start_ns: int
    duration_ns: int
    end_ns: int
    cold_weight_bytes: int
    reserved_dram_gbps: float


@dataclass(frozen=True)
class ColdPhaseAssignment:
    expert_id: int
    routes: int
    threads: int
    start_ns: int
    service_ns: int
    wait_ns: int
    end_ns: int
    phases: tuple[PhaseAssignment, ...]


@dataclass(frozen=True)
class ColdPhaseOracleResult:
    status: str
    optimal: bool
    objective_ns: int | None
    best_bound_ns: float | None
    relative_gap: float | None
    wall_time_s: float
    conflicts: int
    branches: int
    num_jobs: int
    input_modes: int
    retained_modes: int
    num_phases: int
    time_quantum_ns: int
    duration_quantization_error_bound_ns: float
    dram_bandwidth_gbps: float
    bandwidth_quantum_gbps: float
    bandwidth_capacity_units: int
    effective_dram_bandwidth_gbps: float
    cold_phase_slots: int | None
    bandwidth_floor_added_ns: int
    assignments: tuple[ColdPhaseAssignment, ...]

    def mode_histogram(self) -> dict[int, int]:
        return dict(sorted(Counter(assignment.threads for assignment in self.assignments).items()))

    def to_dict(self, *, include_phases: bool = True) -> dict[str, object]:
        payload = asdict(self)
        rows = []
        for assignment in self.assignments:
            row = asdict(assignment)
            if not include_phases:
                row.pop("phases")
            rows.append(row)
        payload["assignments"] = rows
        payload["mode_histogram"] = self.mode_histogram()
        return payload


@dataclass(frozen=True)
class ColdPhaseRuntimeTask:
    expert_id: int
    routes: int
    threads: int
    core_begin: int
    release_ns: int
    modeled_end_ns: int
    dependencies: tuple[int, ...]

    def planner_tuple(self) -> tuple[int, int, int, int, list[int]]:
        return (
            self.expert_id,
            self.routes,
            self.core_begin,
            self.threads,
            list(self.dependencies),
        )


@dataclass(frozen=True)
class ColdPhaseRuntimePlacement:
    status: str
    wall_time_s: float
    num_cores: int
    tasks: tuple[ColdPhaseRuntimeTask, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ColdPhaseOracleComparison:
    fixed_objective_ns: int | None
    fixed_best_bound_ns: float | None
    mixed_objective_ns: int | None
    mixed_best_bound_ns: float | None
    incumbent_gain: float | None
    exact_gain: float | None
    gain_lower_bound: float | None
    gain_upper_bound: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _QuantizedPhase:
    source: ColdPhase
    duration_ticks: int
    base_duration_ticks: int
    bandwidth_units: int


@dataclass(frozen=True)
class _QuantizedMode:
    source: ColdPhaseMode
    phases: tuple[_QuantizedPhase, ...]

    @property
    def service_ticks(self) -> int:
        return sum(phase.duration_ticks for phase in self.phases)


def _cp_model_module():
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise CpSatUnavailableError(
            "the cold-phase CP-SAT oracle requires the optional 'oracle' dependency; "
            "install it with 'uv sync --extra oracle'"
        ) from error
    return cp_model


def _quantize_duration(duration_ns: float, time_quantum_ns: int) -> int:
    if not math.isfinite(duration_ns) or duration_ns <= 0.0:
        raise ValueError(f"phase duration must be finite and positive, got {duration_ns!r}")
    if time_quantum_ns <= 0:
        raise ValueError(f"time_quantum_ns must be positive, got {time_quantum_ns}")
    return max(1, int(math.floor(duration_ns / time_quantum_ns + 0.5)))


def _build_profile_mode(
    model: StagePhaseModel,
    routes: int,
    threads: int,
    *,
    cold_panel_rows: int,
    phase_granularity: str,
) -> ColdPhaseMode:
    isolated_ns = float(model.T_iso(routes, threads))
    if not math.isfinite(isolated_ns) or isolated_ns <= 0.0:
        raise ValueError(f"invalid isolated duration for routes={routes}, threads={threads}: {isolated_ns!r}")

    stages: list[tuple[str, float, int]] = []
    for stage in ("w13", "w2"):
        for duration_ns, weight_bytes in model.task_stage_phases(stage, routes, threads):
            duration_ns = float(duration_ns)
            weight_bytes = int(weight_bytes)
            if duration_ns <= 0.0 or not math.isfinite(duration_ns):
                raise ValueError(
                    f"invalid {stage} phase duration for routes={routes}, threads={threads}: {duration_ns!r}"
                )
            if weight_bytes < 0:
                raise ValueError(f"invalid {stage} packed-B bytes: {weight_bytes}")
            if weight_bytes == 0:
                continue
            stages.append((stage, duration_ns, weight_bytes))
    if not stages:
        raise ValueError(f"no packed-B phases for routes={routes}, threads={threads}")

    total_weight_bytes = sum(weight_bytes for _, _, weight_bytes in stages)
    raw_weight_ns = sum(duration_ns for _, duration_ns, _ in stages)
    reference_routes = min(routes, cold_panel_rows)
    cold_ns = min(float(model.T_iso(reference_routes, threads)), isolated_ns)
    steady_ns = max(isolated_ns - cold_ns, 0.0)

    if phase_granularity == "expert":
        phases = [
            ColdPhase(
                name="expert:cold",
                duration_ns=cold_ns,
                cold_weight_bytes=total_weight_bytes,
            )
        ]
        if steady_ns > 0.0:
            phases.append(ColdPhase(name="expert:steady", duration_ns=steady_ns))
        return ColdPhaseMode(threads=threads, phases=tuple(phases))
    if phase_granularity != "stage":
        raise ValueError(f"phase_granularity must be 'expert' or 'stage', got {phase_granularity!r}")

    phases: list[ColdPhase] = []
    for name, raw_duration_ns, weight_bytes in stages:
        stage_cold_ns = cold_ns * weight_bytes / total_weight_bytes
        phases.append(
            ColdPhase(
                name=f"{name}:cold",
                duration_ns=stage_cold_ns,
                cold_weight_bytes=weight_bytes,
            )
        )
        stage_steady_ns = steady_ns * raw_duration_ns / raw_weight_ns
        if stage_steady_ns > 0.0:
            phases.append(
                ColdPhase(
                    name=f"{name}:steady",
                    duration_ns=stage_steady_ns,
                )
            )

    return ColdPhaseMode(threads=threads, phases=tuple(phases))


def build_cold_phase_jobs(
    experts: Iterable[tuple[int, int]],
    widths: Iterable[int],
    model: StagePhaseModel,
    *,
    num_cores: int,
    cold_panel_rows: int = 12,
    phase_granularity: str = "expert",
) -> tuple[ColdPhaseJob, ...]:
    """Build profile-backed cold/steady modes for each active expert."""

    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    if cold_panel_rows <= 0:
        raise ValueError(f"cold_panel_rows must be positive, got {cold_panel_rows}")
    if phase_granularity not in {"expert", "stage"}:
        raise ValueError(f"phase_granularity must be 'expert' or 'stage', got {phase_granularity!r}")
    normalized_widths = tuple(sorted({int(width) for width in widths}))
    if not normalized_widths:
        raise ValueError("at least one thread width is required")
    if normalized_widths[0] <= 0 or normalized_widths[-1] > num_cores:
        raise ValueError(f"thread widths must be within [1, {num_cores}], got {normalized_widths}")

    jobs: list[ColdPhaseJob] = []
    seen_experts: set[int] = set()
    for raw_expert_id, raw_routes in experts:
        expert_id = int(raw_expert_id)
        routes = int(raw_routes)
        if expert_id in seen_experts:
            raise ValueError(f"duplicate expert_id={expert_id}")
        seen_experts.add(expert_id)
        if routes < 0:
            raise ValueError(f"routes must be non-negative, got expert_id={expert_id}, routes={routes}")
        if routes == 0:
            continue
        modes = tuple(
            _build_profile_mode(
                model,
                routes,
                threads,
                cold_panel_rows=cold_panel_rows,
                phase_granularity=phase_granularity,
            )
            for threads in normalized_widths
        )
        jobs.append(ColdPhaseJob(expert_id=expert_id, routes=routes, modes=modes))
    if not jobs:
        raise ValueError("at least one active expert is required")
    return tuple(jobs)


def _normalize_jobs(
    jobs: Sequence[ColdPhaseJob],
    *,
    num_cores: int,
) -> tuple[tuple[ColdPhaseJob, ...], int]:
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    if not jobs:
        raise ValueError("at least one active expert is required")

    normalized: list[ColdPhaseJob] = []
    seen_experts: set[int] = set()
    input_modes = 0
    for job in jobs:
        if job.expert_id in seen_experts:
            raise ValueError(f"duplicate expert_id={job.expert_id}")
        seen_experts.add(job.expert_id)
        if job.routes <= 0:
            raise ValueError(f"job routes must be positive, got expert_id={job.expert_id}, routes={job.routes}")
        input_modes += len(job.modes)
        modes: list[ColdPhaseMode] = []
        seen_threads: set[int] = set()
        for mode in sorted(job.modes, key=lambda item: item.threads):
            if mode.threads <= 0:
                raise ValueError(f"mode threads must be positive, got {mode.threads}")
            if mode.threads > num_cores:
                continue
            if mode.threads in seen_threads:
                raise ValueError(f"duplicate mode threads={mode.threads} for expert_id={job.expert_id}")
            seen_threads.add(mode.threads)
            if not mode.phases:
                raise ValueError(f"mode has no phases for expert_id={job.expert_id}, threads={mode.threads}")
            for phase in mode.phases:
                if not phase.name:
                    raise ValueError("phase name must be non-empty")
                if not math.isfinite(phase.duration_ns) or phase.duration_ns <= 0.0:
                    raise ValueError(f"phase duration must be finite and positive, got {phase.duration_ns!r}")
                if phase.cold_weight_bytes < 0:
                    raise ValueError(f"cold_weight_bytes must be non-negative, got {phase.cold_weight_bytes}")
            modes.append(mode)
        if not modes:
            raise ValueError(f"expert_id={job.expert_id} has no mode within the {num_cores}-core capacity")
        normalized.append(ColdPhaseJob(job.expert_id, job.routes, tuple(modes)))
    return tuple(normalized), input_modes


def _normalize_dependencies(
    num_jobs: int,
    dependencies: Sequence[Sequence[int]] | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    if dependencies is None:
        normalized = tuple(() for _ in range(num_jobs))
    else:
        if len(dependencies) != num_jobs:
            raise ValueError(f"expected {num_jobs} dependency rows, got {len(dependencies)}")
        rows = []
        for job_index, raw_dependencies in enumerate(dependencies):
            row = tuple(sorted({int(dependency) for dependency in raw_dependencies}))
            if job_index in row:
                raise ValueError(f"job {job_index} depends on itself")
            if row and (row[0] < 0 or row[-1] >= num_jobs):
                raise ValueError(f"job {job_index} has an out-of-range dependency")
            rows.append(row)
        normalized = tuple(rows)

    children: list[list[int]] = [[] for _ in range(num_jobs)]
    indegree = [0] * num_jobs
    for job_index, row in enumerate(normalized):
        indegree[job_index] = len(row)
        for dependency in row:
            children[dependency].append(job_index)
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        job_index = heapq.heappop(ready)
        order.append(job_index)
        for child in children[job_index]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != num_jobs:
        raise ValueError("job dependencies contain a cycle")
    return normalized, tuple(order)


def _quantize_jobs(
    jobs: Sequence[ColdPhaseJob],
    *,
    time_quantum_ns: int,
    bandwidth_quantum_gbps: float,
    bandwidth_capacity_units: int,
) -> tuple[tuple[_QuantizedMode, ...], ...]:
    effective_bandwidth = bandwidth_quantum_gbps * bandwidth_capacity_units
    quantized_jobs = []
    for job in jobs:
        quantized_modes = []
        for mode in job.modes:
            quantized_phases = []
            for phase in mode.phases:
                base_ticks = _quantize_duration(phase.duration_ns, time_quantum_ns)
                duration_ticks = base_ticks
                bandwidth_units = 0
                if phase.is_cold:
                    transfer_ticks = math.ceil(phase.cold_weight_bytes / (effective_bandwidth * time_quantum_ns))
                    duration_ticks = max(duration_ticks, transfer_ticks, 1)
                    rate_gbps = phase.cold_weight_bytes / (duration_ticks * time_quantum_ns)
                    bandwidth_units = max(
                        1,
                        int(math.ceil(rate_gbps / bandwidth_quantum_gbps - 1e-12)),
                    )
                    while bandwidth_units > bandwidth_capacity_units:
                        duration_ticks += 1
                        rate_gbps = phase.cold_weight_bytes / (duration_ticks * time_quantum_ns)
                        bandwidth_units = max(
                            1,
                            int(math.ceil(rate_gbps / bandwidth_quantum_gbps - 1e-12)),
                        )
                quantized_phases.append(
                    _QuantizedPhase(
                        source=phase,
                        duration_ticks=duration_ticks,
                        base_duration_ticks=base_ticks,
                        bandwidth_units=bandwidth_units,
                    )
                )
            quantized_modes.append(_QuantizedMode(source=mode, phases=tuple(quantized_phases)))
        quantized_jobs.append(tuple(quantized_modes))
    return tuple(quantized_jobs)


def _critical_path_bound(
    minimum_durations: Sequence[int],
    dependencies: Sequence[Sequence[int]],
    topological_order: Sequence[int],
) -> int:
    finish = [0] * len(minimum_durations)
    for job_index in topological_order:
        ready = max((finish[dependency] for dependency in dependencies[job_index]), default=0)
        finish[job_index] = ready + minimum_durations[job_index]
    return max(finish, default=0)


def solve_cold_phase_cp_sat(
    jobs: Sequence[ColdPhaseJob],
    *,
    num_cores: int,
    dram_bandwidth_gbps: float,
    dependencies: Sequence[Sequence[int]] | None = None,
    initial_assignments: Sequence[ColdPhaseAssignment] | None = None,
    max_time_s: float = 60.0,
    workers: int = 1,
    relative_gap_limit: float = 0.0,
    time_quantum_ns: int = 1000,
    bandwidth_quantum_gbps: float = 0.25,
    cold_phase_slots: int | None = None,
    random_seed: int = 0,
    log_search_progress: bool = False,
) -> ColdPhaseOracleResult:
    """Solve the fixed-rate cold-phase moldable scheduling surrogate."""

    if max_time_s <= 0.0 or not math.isfinite(max_time_s):
        raise ValueError(f"max_time_s must be finite and positive, got {max_time_s!r}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if relative_gap_limit < 0.0 or relative_gap_limit >= 1.0:
        raise ValueError(f"relative_gap_limit must be in [0, 1), got {relative_gap_limit}")
    if not math.isfinite(dram_bandwidth_gbps) or dram_bandwidth_gbps <= 0.0:
        raise ValueError(f"dram_bandwidth_gbps must be finite and positive, got {dram_bandwidth_gbps!r}")
    if not math.isfinite(bandwidth_quantum_gbps) or bandwidth_quantum_gbps <= 0.0:
        raise ValueError(f"bandwidth_quantum_gbps must be finite and positive, got {bandwidth_quantum_gbps!r}")
    if cold_phase_slots is not None and cold_phase_slots <= 0:
        raise ValueError(f"cold_phase_slots must be positive when set, got {cold_phase_slots}")

    normalized, input_modes = _normalize_jobs(jobs, num_cores=num_cores)
    normalized_dependencies, topological_order = _normalize_dependencies(len(normalized), dependencies)
    bandwidth_capacity_units = int(math.floor(dram_bandwidth_gbps / bandwidth_quantum_gbps + 1e-12))
    if bandwidth_capacity_units <= 0:
        raise ValueError("dram bandwidth is smaller than one bandwidth quantum")
    effective_bandwidth = bandwidth_capacity_units * bandwidth_quantum_gbps
    quantized = _quantize_jobs(
        normalized,
        time_quantum_ns=time_quantum_ns,
        bandwidth_quantum_gbps=bandwidth_quantum_gbps,
        bandwidth_capacity_units=bandwidth_capacity_units,
    )

    minimum_durations = [min(mode.service_ticks for mode in modes) for modes in quantized]
    horizon = sum(minimum_durations)
    if horizon <= 0 or horizon >= (1 << 62):
        raise ValueError(f"invalid CP-SAT horizon={horizon} ticks")

    cp_model = _cp_model_module()
    model = cp_model.CpModel()
    master_intervals = []
    master_demands = []
    cold_intervals = []
    cold_demands = []
    cold_slot_intervals = []
    variables = []
    job_starts = []
    job_ends = []
    job_mode_indices = []

    min_core_area = 0
    min_cold_bytes = 0
    for job_index, (job, modes) in enumerate(zip(normalized, quantized)):
        presences = []
        mode_variables = []
        job_start = model.new_int_var(0, horizon, f"job_start_e{job.expert_id}_j{job_index}")
        job_end = model.new_int_var(0, horizon, f"job_end_e{job.expert_id}_j{job_index}")
        job_mode_index = model.new_int_var(0, len(modes) - 1, f"job_mode_e{job.expert_id}_j{job_index}")
        min_core_area += min(mode.source.threads * mode.service_ticks for mode in modes)
        min_cold_bytes += min(sum(phase.source.cold_weight_bytes for phase in mode.phases) for mode in modes)
        for mode_index, quantized_mode in enumerate(modes):
            threads = quantized_mode.source.threads
            suffix = f"e{job.expert_id}_j{job_index}_m{mode_index}_t{threads}"
            presence = model.new_bool_var(f"select_{suffix}")
            start = model.new_int_var(0, horizon, f"start_{suffix}")
            end = model.new_int_var(0, horizon, f"end_{suffix}")
            size = model.new_int_var(0, horizon, f"residence_{suffix}")
            master = model.new_optional_interval_var(start, size, end, presence, f"run_{suffix}")
            model.add(start == 0).only_enforce_if(presence.Not())
            model.add(end == 0).only_enforce_if(presence.Not())
            model.add(size == 0).only_enforce_if(presence.Not())
            model.add(size >= quantized_mode.service_ticks).only_enforce_if(presence)
            model.add(job_start == start).only_enforce_if(presence)
            model.add(job_end == end).only_enforce_if(presence)
            model.add(job_mode_index == mode_index).only_enforce_if(presence)
            master_intervals.append(master)
            master_demands.append(threads)

            phase_variables = []
            previous_end = None
            for phase_index, phase in enumerate(quantized_mode.phases):
                phase_suffix = f"{suffix}_p{phase_index}"
                phase_start = model.new_int_var(0, horizon, f"phase_start_{phase_suffix}")
                phase_end = model.new_int_var(0, horizon, f"phase_end_{phase_suffix}")
                interval = model.new_optional_interval_var(
                    phase_start,
                    phase.duration_ticks,
                    phase_end,
                    presence,
                    f"phase_{phase_suffix}",
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
            assert previous_end is not None
            model.add(end == previous_end).only_enforce_if(presence)
            presences.append(presence)
            mode_variables.append((quantized_mode, presence, start, size, end, phase_variables))
        model.add_exactly_one(presences)
        variables.append(mode_variables)
        job_starts.append(job_start)
        job_ends.append(job_end)
        job_mode_indices.append(job_mode_index)

    model.add_cumulative(master_intervals, master_demands, num_cores)
    model.add_cumulative(cold_intervals, cold_demands, bandwidth_capacity_units)
    if cold_phase_slots is not None:
        model.add_cumulative(cold_slot_intervals, [1] * len(cold_slot_intervals), cold_phase_slots)
    for job_index, row in enumerate(normalized_dependencies):
        for dependency in row:
            model.add(job_starts[job_index] >= job_ends[dependency])

    if not any(normalized_dependencies) and initial_assignments is None:
        identical_groups: dict[tuple[int, tuple[ColdPhaseMode, ...]], list[int]] = defaultdict(list)
        for job_index, job in enumerate(normalized):
            identical_groups[(job.routes, job.modes)].append(job_index)
        for indices in identical_groups.values():
            for left, right in zip(indices, indices[1:]):
                model.add(job_starts[left] <= job_starts[right])

    first_start = model.new_int_var(0, 0, "first_start")
    model.add_min_equality(first_start, job_starts)
    area_bound = math.ceil(min_core_area / num_cores)
    bandwidth_bound = math.ceil(min_cold_bytes / (effective_bandwidth * time_quantum_ns))
    critical_path_bound = _critical_path_bound(
        minimum_durations,
        normalized_dependencies,
        topological_order,
    )
    model_bound = max(area_bound, bandwidth_bound, critical_path_bound)
    makespan = model.new_int_var(model_bound, horizon, "makespan")
    model.add_max_equality(makespan, job_ends)
    model.minimize(makespan)

    schedule_hints = [None] * len(normalized)
    if initial_assignments is None:
        cursor = 0
        for job_index in topological_order:
            mode_index, quantized_mode = min(
                enumerate(quantized[job_index]),
                key=lambda item: (item[1].service_ticks, item[1].source.threads),
            )
            phase_rows = []
            phase_cursor = cursor
            for phase in quantized_mode.phases:
                phase_rows.append((phase_cursor, phase_cursor + phase.duration_ticks))
                phase_cursor += phase.duration_ticks
            schedule_hints[job_index] = (mode_index, cursor, phase_cursor, phase_rows)
            cursor = phase_cursor
    else:
        by_expert = {assignment.expert_id: assignment for assignment in initial_assignments}
        if len(by_expert) != len(initial_assignments):
            raise ValueError("initial_assignments contains duplicate expert ids")
        if set(by_expert) != {job.expert_id for job in normalized}:
            raise ValueError("initial_assignments must cover exactly the oracle jobs")
        cursor = 0
        for job_index, (job, modes) in enumerate(zip(normalized, quantized)):
            assignment = by_expert[job.expert_id]
            matching_modes = [
                (mode_index, mode) for mode_index, mode in enumerate(modes) if mode.source.threads == assignment.threads
            ]
            if len(matching_modes) != 1:
                raise ValueError(
                    f"initial assignment width {assignment.threads} is unavailable for expert_id={job.expert_id}"
                )
            mode_index, quantized_mode = matching_modes[0]
            if len(quantized_mode.phases) != len(assignment.phases):
                raise ValueError(f"initial assignment phase count mismatch for expert_id={job.expert_id}")
            phase_rows = []
            for phase, assigned_phase in zip(quantized_mode.phases, assignment.phases):
                if phase.source.name != assigned_phase.name:
                    raise ValueError(f"initial assignment phase mismatch for expert_id={job.expert_id}")
                if assigned_phase.duration_ns != phase.duration_ticks * time_quantum_ns:
                    raise ValueError(f"initial assignment duration mismatch for expert_id={job.expert_id}")
                phase_rows.append(
                    (
                        assigned_phase.start_ns // time_quantum_ns,
                        assigned_phase.end_ns // time_quantum_ns,
                    )
                )
            start_hint = assignment.start_ns // time_quantum_ns
            end_hint = assignment.end_ns // time_quantum_ns
            schedule_hints[job_index] = (mode_index, start_hint, end_hint, phase_rows)
            cursor = max(cursor, end_hint)
        model.add(makespan <= cursor)

    for job_index, hint in enumerate(schedule_hints):
        assert hint is not None
        selected_mode, start_hint, end_hint, phase_hints = hint
        for mode_index, (_, presence, start, size, end, phase_variables) in enumerate(variables[job_index]):
            selected = mode_index == selected_mode
            model.add_hint(presence, int(selected))
            model.add_hint(start, start_hint if selected else 0)
            model.add_hint(size, end_hint - start_hint if selected else 0)
            model.add_hint(end, end_hint if selected else 0)
            for phase_index, (_, phase_start, phase_end) in enumerate(phase_variables):
                if selected:
                    phase_start_hint, phase_end_hint = phase_hints[phase_index]
                else:
                    phase_start_hint = phase_end_hint = 0
                model.add_hint(phase_start, phase_start_hint)
                model.add_hint(phase_end, phase_end_hint)
        model.add_hint(job_starts[job_index], start_hint)
        model.add_hint(job_ends[job_index], end_hint)
        model.add_hint(job_mode_indices[job_index], selected_mode)
    model.add_hint(makespan, cursor)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_s)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.relative_gap_limit = float(relative_gap_limit)
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.log_search_progress = bool(log_search_progress)
    status_code = solver.solve(model)
    status_names = {
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.OPTIMAL: "OPTIMAL",
    }
    status = status_names.get(status_code, f"STATUS_{status_code}")
    has_solution = status_code in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    objective_ns = int(solver.value(makespan) * time_quantum_ns) if has_solution else None
    best_bound_ns = (
        float(solver.best_objective_bound) * time_quantum_ns
        if status_code in (cp_model.UNKNOWN, cp_model.FEASIBLE, cp_model.OPTIMAL)
        else None
    )
    relative_gap = None
    if objective_ns is not None and best_bound_ns is not None:
        relative_gap = max((objective_ns - best_bound_ns) / max(abs(objective_ns), 1.0), 0.0)

    assignments: list[ColdPhaseAssignment] = []
    selected_phases = 0
    bandwidth_floor_added_ns = 0
    if has_solution:
        for job, mode_variables in zip(normalized, variables):
            selected = [row for row in mode_variables if solver.boolean_value(row[1])]
            if len(selected) != 1:
                raise RuntimeError(f"CP-SAT selected {len(selected)} modes for expert_id={job.expert_id}")
            quantized_mode, _, start, _, end, phase_variables = selected[0]
            phase_assignments = []
            service_ticks = 0
            for phase, phase_start, phase_end in phase_variables:
                phase_start_ns = int(solver.value(phase_start) * time_quantum_ns)
                phase_end_ns = int(solver.value(phase_end) * time_quantum_ns)
                duration_ns = phase.duration_ticks * time_quantum_ns
                service_ticks += phase.duration_ticks
                selected_phases += 1
                bandwidth_floor_added_ns += (phase.duration_ticks - phase.base_duration_ticks) * time_quantum_ns
                phase_assignments.append(
                    PhaseAssignment(
                        name=phase.source.name,
                        start_ns=phase_start_ns,
                        duration_ns=duration_ns,
                        end_ns=phase_end_ns,
                        cold_weight_bytes=phase.source.cold_weight_bytes,
                        reserved_dram_gbps=phase.bandwidth_units * bandwidth_quantum_gbps,
                    )
                )
            start_ns = int(solver.value(start) * time_quantum_ns)
            end_ns = int(solver.value(end) * time_quantum_ns)
            service_ns = service_ticks * time_quantum_ns
            assignments.append(
                ColdPhaseAssignment(
                    expert_id=job.expert_id,
                    routes=job.routes,
                    threads=quantized_mode.source.threads,
                    start_ns=start_ns,
                    service_ns=service_ns,
                    wait_ns=max(end_ns - start_ns - service_ns, 0),
                    end_ns=end_ns,
                    phases=tuple(phase_assignments),
                )
            )
        assignments.sort(key=lambda assignment: (assignment.start_ns, assignment.end_ns, assignment.expert_id))

    proven_optimal = (
        status_code == cp_model.OPTIMAL
        and objective_ns is not None
        and best_bound_ns is not None
        and abs(objective_ns - best_bound_ns) < 0.5 * time_quantum_ns
    )
    return ColdPhaseOracleResult(
        status=status,
        optimal=proven_optimal,
        objective_ns=objective_ns,
        best_bound_ns=best_bound_ns,
        relative_gap=relative_gap,
        wall_time_s=float(solver.wall_time),
        conflicts=int(solver.num_conflicts),
        branches=int(solver.num_branches),
        num_jobs=len(normalized),
        input_modes=input_modes,
        retained_modes=sum(len(job.modes) for job in normalized),
        num_phases=selected_phases,
        time_quantum_ns=time_quantum_ns,
        duration_quantization_error_bound_ns=0.5 * selected_phases * time_quantum_ns,
        dram_bandwidth_gbps=dram_bandwidth_gbps,
        bandwidth_quantum_gbps=bandwidth_quantum_gbps,
        bandwidth_capacity_units=bandwidth_capacity_units,
        effective_dram_bandwidth_gbps=effective_bandwidth,
        cold_phase_slots=cold_phase_slots,
        bandwidth_floor_added_ns=bandwidth_floor_added_ns,
        assignments=tuple(assignments),
    )


def compare_cold_phase_oracles(
    fixed: ColdPhaseOracleResult,
    mixed: ColdPhaseOracleResult,
) -> ColdPhaseOracleComparison:
    incumbent_gain = None
    if fixed.objective_ns is not None and mixed.objective_ns is not None:
        incumbent_gain = fixed.objective_ns / mixed.objective_ns - 1.0
    exact_gain = None
    if fixed.optimal and mixed.optimal and fixed.objective_ns is not None and mixed.objective_ns is not None:
        exact_gain = fixed.objective_ns / mixed.objective_ns - 1.0
    gain_lower_bound = None
    if fixed.best_bound_ns is not None and mixed.objective_ns is not None and mixed.objective_ns > 0:
        gain_lower_bound = max(fixed.best_bound_ns / mixed.objective_ns - 1.0, 0.0)
    gain_upper_bound = None
    if fixed.objective_ns is not None and mixed.best_bound_ns is not None and mixed.best_bound_ns > 0.0:
        gain_upper_bound = max(fixed.objective_ns / mixed.best_bound_ns - 1.0, 0.0)
    return ColdPhaseOracleComparison(
        fixed_objective_ns=fixed.objective_ns,
        fixed_best_bound_ns=fixed.best_bound_ns,
        mixed_objective_ns=mixed.objective_ns,
        mixed_best_bound_ns=mixed.best_bound_ns,
        incumbent_gain=incumbent_gain,
        exact_gain=exact_gain,
        gain_lower_bound=gain_lower_bound,
        gain_upper_bound=gain_upper_bound,
    )


def materialize_cold_phase_runtime_placement(
    assignments: Sequence[ColdPhaseAssignment],
    *,
    num_cores: int,
    max_time_s: float = 30.0,
    workers: int = 1,
    random_seed: int = 0,
) -> ColdPhaseRuntimePlacement:
    """Map a fluid CPU-capacity schedule to contiguous runtime core teams.

    The cold-phase oracle constrains only aggregate CPU capacity. The native
    runtime needs one contiguous logical-core interval per task, so this second
    CP-SAT model keeps every oracle time interval fixed and solves only the
    spatial placement. Dependencies serialize consecutive users of each core;
    release times preserve deliberate oracle idling when predecessors finish
    early on hardware.
    """

    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    if max_time_s <= 0.0 or not math.isfinite(max_time_s):
        raise ValueError(f"max_time_s must be finite and positive, got {max_time_s!r}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if not assignments:
        raise ValueError("at least one cold-phase assignment is required")

    ordered = tuple(sorted(assignments, key=lambda item: (item.start_ns, item.end_ns, item.expert_id)))
    if len({assignment.expert_id for assignment in ordered}) != len(ordered):
        raise ValueError("runtime materialization requires one whole task per expert")
    for assignment in ordered:
        if assignment.threads <= 0 or assignment.threads > num_cores:
            raise ValueError(
                f"assignment width must be within [1, {num_cores}], got {assignment.threads}"
            )
        if assignment.start_ns < 0 or assignment.end_ns <= assignment.start_ns:
            raise ValueError(f"assignment has an invalid time interval: {assignment}")
        if assignment.wait_ns != 0 or assignment.end_ns - assignment.start_ns != assignment.service_ns:
            raise ValueError(
                "runtime materialization currently requires contiguous phase service with no internal waits: "
                f"expert_id={assignment.expert_id}, wait_ns={assignment.wait_ns}"
            )

    cp_model = _cp_model_module()
    model = cp_model.CpModel()
    time_intervals = []
    core_intervals = []
    core_begins = []
    for task_id, assignment in enumerate(ordered):
        core_begin = model.new_int_var(
            0,
            num_cores - assignment.threads,
            f"core_begin_t{task_id}_e{assignment.expert_id}",
        )
        time_intervals.append(
            model.new_fixed_size_interval_var(
                assignment.start_ns,
                assignment.end_ns - assignment.start_ns,
                f"time_t{task_id}_e{assignment.expert_id}",
            )
        )
        core_intervals.append(
            model.new_fixed_size_interval_var(
                core_begin,
                assignment.threads,
                f"cores_t{task_id}_e{assignment.expert_id}",
            )
        )
        core_begins.append(core_begin)
    model.add_no_overlap_2d(time_intervals, core_intervals)
    model.add_decision_strategy(
        core_begins,
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
    if status_code not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return ColdPhaseRuntimePlacement(
            status=status,
            wall_time_s=float(solver.wall_time),
            num_cores=num_cores,
            tasks=(),
        )

    solved_core_begins = [int(solver.value(core_begin)) for core_begin in core_begins]
    previous_by_core = [-1] * num_cores
    runtime_tasks = []
    schedule_origin_ns = min(assignment.start_ns for assignment in ordered)
    for task_id, (assignment, core_begin) in enumerate(zip(ordered, solved_core_begins, strict=True)):
        core_end = core_begin + assignment.threads
        dependencies = tuple(sorted({task for task in previous_by_core[core_begin:core_end] if task >= 0}))
        runtime_tasks.append(
            ColdPhaseRuntimeTask(
                expert_id=assignment.expert_id,
                routes=assignment.routes,
                threads=assignment.threads,
                core_begin=core_begin,
                release_ns=assignment.start_ns - schedule_origin_ns,
                modeled_end_ns=assignment.end_ns - schedule_origin_ns,
                dependencies=dependencies,
            )
        )
        previous_by_core[core_begin:core_end] = [task_id] * assignment.threads

    return ColdPhaseRuntimePlacement(
        status=status,
        wall_time_s=float(solver.wall_time),
        num_cores=num_cores,
        tasks=tuple(runtime_tasks),
    )


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _load_modules(profile: Path):
    root = Path(__file__).resolve().parents[1]
    cost_model_dir = root / "cost_model"
    planners_dir = root / "planners"
    sys.path[:0] = [str(cost_model_dir), str(planners_dir)]
    from interval_planner import IntervalPlanner
    from phase_model import ContentionCostModel
    from workload_catalog import default_offline_workloads

    return (
        ContentionCostModel(profile),
        IntervalPlanner,
        default_offline_workloads,
    )


def _model_widths(model, num_cores: int) -> tuple[int, ...]:
    declared = getattr(model, "supported_widths", None)
    if declared is None:
        declared = {int(width) for shape in getattr(model, "supported_shapes", ()) for width in shape}
    widths = tuple(sorted({int(width) for width in declared if 0 < int(width) <= num_cores}))
    if not widths:
        raise ValueError("cost model does not expose any thread width within the requested core capacity")
    return widths


def _run_cli(args: argparse.Namespace) -> dict[str, object]:
    raw_model, interval_planner_type, workload_loader = _load_modules(args.profile)
    if args.routes:
        routes = _parse_int_list(args.routes, name="routes")
        experts = tuple((expert_id, count) for expert_id, count in enumerate(routes) if count > 0)
        workload_name = "explicit"
        workload_metadata = None
    else:
        workloads = workload_loader()
        if args.workload not in workloads:
            raise KeyError(f"unknown workload {args.workload!r}; choices={sorted(workloads)}")
        workload = workloads[args.workload]
        experts = tuple(workload.experts)
        routes = tuple(workload.histogram)
        workload_name = workload.name
        workload_metadata = {
            "tokens": workload.tokens,
            "top_k": workload.top_k,
            "num_experts": workload.num_experts,
            "active_experts": workload.observed_active_experts,
            "routes_std": workload.observed_routes_std,
            "source": workload.source,
        }

    widths = (
        _parse_int_list(args.widths, name="widths")
        if args.widths
        else tuple(width for width in _model_widths(raw_model, args.num_cores) if width <= 16)
    )
    if args.baseline_width <= 0 or args.num_cores % args.baseline_width:
        raise ValueError("baseline_width must be a positive divisor of num_cores")
    if args.baseline_width not in _model_widths(raw_model, args.num_cores):
        raise ValueError(f"baseline_width={args.baseline_width} is not supported by the profile")
    if args.baseline_width not in widths:
        raise ValueError("mixed widths must include baseline_width so the mixed domain contains the fixed baseline")

    planner = interval_planner_type(
        raw_model,
        args.num_cores,
        widths=_model_widths(raw_model, args.num_cores),
        native_cold_planner=False,
    )
    phase_model = planner.model
    baseline_shape = (args.baseline_width,) * (args.num_cores // args.baseline_width)
    baseline_contention_ns, baseline_tasks = planner.score_shape(list(experts), baseline_shape)
    current_plan = planner.plan(
        list(experts),
        dynamic_tail_pool=False,
        bounded_tail_repartition=False,
    )

    baseline_jobs = build_cold_phase_jobs(
        ((expert_id, count) for expert_id, count, _, _, _ in baseline_tasks),
        (args.baseline_width,),
        phase_model,
        num_cores=args.num_cores,
        cold_panel_rows=args.cold_panel_rows,
        phase_granularity=args.phase_granularity,
    )
    baseline_dependencies = tuple(tuple(int(value) for value in task[4]) for task in baseline_tasks)
    mixed_jobs = build_cold_phase_jobs(
        experts,
        widths,
        phase_model,
        num_cores=args.num_cores,
        cold_panel_rows=args.cold_panel_rows,
        phase_granularity=args.phase_granularity,
    )

    solve_kwargs = {
        "num_cores": args.num_cores,
        "dram_bandwidth_gbps": args.dram_bandwidth_gbps,
        "max_time_s": args.max_time_s,
        "workers": args.workers,
        "relative_gap_limit": args.relative_gap_limit,
        "time_quantum_ns": args.time_quantum_ns,
        "bandwidth_quantum_gbps": args.bandwidth_quantum_gbps,
        "cold_phase_slots": args.cold_phase_slots,
        "random_seed": args.random_seed,
        "log_search_progress": args.log_search_progress,
    }
    fixed = solve_cold_phase_cp_sat(
        baseline_jobs,
        dependencies=baseline_dependencies,
        **solve_kwargs,
    )
    mixed = solve_cold_phase_cp_sat(
        mixed_jobs,
        initial_assignments=fixed.assignments if fixed.objective_ns is not None else None,
        **solve_kwargs,
    )

    from isolated_cp_sat_oracle import evaluate_isolated_dag

    baseline_isolated_ns = evaluate_isolated_dag(
        baseline_tasks,
        phase_model.T_iso,
        time_quantum_ns=args.time_quantum_ns,
    )
    return {
        "model": "cold_phase_cp_sat_v1",
        "profile": str(args.profile),
        "workload": workload_name,
        "workload_metadata": workload_metadata,
        "routes": list(routes),
        "num_cores": args.num_cores,
        "mixed_widths": list(widths),
        "cold_panel_rows": args.cold_panel_rows,
        "phase_granularity": args.phase_granularity,
        "dram_bandwidth_gbps": args.dram_bandwidth_gbps,
        "bandwidth_quantum_gbps": args.bandwidth_quantum_gbps,
        "cold_phase_slots": args.cold_phase_slots,
        "assumptions": {
            "cold_phase": (
                "first min(routes, cold_panel_rows) rows; W13/W2 are fluid-aggregated"
                if args.phase_granularity == "expert"
                else "first min(routes, cold_panel_rows) rows, split across full-N W13/W2 stages by bytes"
            ),
            "steady_phase": (
                "remaining isolated time as one aggregate phase"
                if args.phase_granularity == "expert"
                else "remaining isolated time, split across full-N W13/W2 stages"
            ),
            "cpu": "selected team is retained from expert start through all phase waits",
            "mode": "whole-expert fixed width; no preemption or W13-to-W2 resizing",
            "modeled_shared_resource": "cold packed-B DRAM bandwidth only",
            "omitted": [
                "packed-A traffic",
                "output stores",
                "LLC-to-L2 refill capacity",
                "compute/frequency contention",
                "route merge and communication",
            ],
        },
        "current_strict_plan": {
            "selected_shape": list(current_plan["shape"]),
            "selected_contention_model_makespan_ns": current_plan["makespan_ns"],
            "baseline_shape": list(baseline_shape),
            "baseline_contention_model_makespan_ns": baseline_contention_ns,
            "baseline_isolated_dag_makespan_ns": baseline_isolated_ns,
        },
        "fixed_baseline_oracle": fixed.to_dict(include_phases=args.include_phases),
        "mixed_oracle": mixed.to_dict(include_phases=args.include_phases),
        "comparison": compare_cold_phase_oracles(fixed, mixed).to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="schema-v2 empirical cost profile")
    workload = parser.add_mutually_exclusive_group(required=True)
    workload.add_argument("--workload", help="name from workload_catalog.default_offline_workloads()")
    workload.add_argument("--routes", help="comma-separated expert route histogram")
    parser.add_argument("--num-cores", type=int, default=96)
    parser.add_argument(
        "--widths",
        default="1,2,4,8,16",
        help="mixed-oracle widths; default: 1,2,4,8,16",
    )
    parser.add_argument("--baseline-width", type=int, default=8)
    parser.add_argument("--cold-panel-rows", type=int, default=12)
    parser.add_argument("--phase-granularity", choices=("expert", "stage"), default="expert")
    parser.add_argument("--dram-bandwidth-gbps", type=float, default=336.4)
    parser.add_argument("--bandwidth-quantum-gbps", type=float, default=0.25)
    parser.add_argument("--cold-phase-slots", type=int)
    parser.add_argument("--max-time-s", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--relative-gap-limit", type=float, default=0.0)
    parser.add_argument("--time-quantum-ns", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--log-search-progress", action="store_true")
    parser.add_argument("--include-phases", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _run_cli(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
