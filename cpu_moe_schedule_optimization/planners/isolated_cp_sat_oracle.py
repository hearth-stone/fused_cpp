"""Offline no-contention CP-SAT oracle for CPU MoE expert scheduling.

The oracle solves the fixed-duration moldable-job relaxation:

* every active expert selects exactly one thread width;
* the selected expert executes non-preemptively for ``T_iso(routes, width)``;
* concurrent thread demand never exceeds the rank-local core count;
* experts do not slow each other down.

This is deliberately not a production planner. Under the model assumption that
contention never speeds up an expert, its optimum is an optimistic lower bound
on the contention-aware makespan for the same mode set.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


IsolatedTimeFn = Callable[[int, int], float]


class CpSatUnavailableError(RuntimeError):
    """Raised when the optional OR-Tools dependency is unavailable."""


@dataclass(frozen=True)
class IsolatedMode:
    threads: int
    duration_ns: float


@dataclass(frozen=True)
class IsolatedJob:
    expert_id: int
    routes: int
    modes: tuple[IsolatedMode, ...]


@dataclass(frozen=True)
class OracleAssignment:
    expert_id: int
    routes: int
    threads: int
    start_ns: int
    duration_ns: int
    end_ns: int


@dataclass(frozen=True)
class IsolatedOracleResult:
    status: str
    optimal: bool
    objective_ns: int | None
    best_bound_ns: float | None
    relative_gap: float | None
    bound_gap: float | None
    wall_time_s: float
    conflicts: int
    branches: int
    num_jobs: int
    input_modes: int
    retained_modes: int
    time_quantum_ns: int
    quantization_error_bound_ns: float
    assignments: tuple[OracleAssignment, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["assignments"] = [asdict(assignment) for assignment in self.assignments]
        return payload


@dataclass(frozen=True)
class OracleComparison:
    candidate_ns: float
    oracle_objective_ns: int | None
    oracle_best_bound_ns: float | None
    exact_regret: float | None
    regret_lower_bound: float | None
    regret_upper_bound: float | None
    bound_efficiency: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cp_model_module():
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise CpSatUnavailableError(
            "the isolated CP-SAT oracle requires the optional 'oracle' dependency; "
            "install it with 'uv sync --extra oracle'"
        ) from error
    return cp_model


def _quantize_duration(duration_ns: float, time_quantum_ns: int) -> int:
    if not math.isfinite(duration_ns) or duration_ns <= 0.0:
        raise ValueError(f"isolated duration must be finite and positive, got {duration_ns!r}")
    if time_quantum_ns <= 0:
        raise ValueError(f"time_quantum_ns must be positive, got {time_quantum_ns}")
    return max(1, int(math.floor(duration_ns / time_quantum_ns + 0.5)))


def prune_dominated_modes(modes: Iterable[IsolatedMode]) -> tuple[IsolatedMode, ...]:
    """Remove a mode when another mode is no slower and uses no more threads."""

    ordered = sorted(modes, key=lambda mode: (mode.threads, mode.duration_ns))
    retained: list[IsolatedMode] = []
    best_duration = math.inf
    seen_threads: set[int] = set()
    for mode in ordered:
        if mode.threads <= 0:
            raise ValueError(f"mode threads must be positive, got {mode.threads}")
        if not math.isfinite(mode.duration_ns) or mode.duration_ns <= 0.0:
            raise ValueError(f"mode duration must be finite and positive, got {mode.duration_ns!r}")
        if mode.threads in seen_threads:
            continue
        seen_threads.add(mode.threads)
        if mode.duration_ns >= best_duration:
            continue
        retained.append(mode)
        best_duration = mode.duration_ns
    return tuple(retained)


def build_isolated_jobs(
    experts: Iterable[tuple[int, int]],
    widths: Iterable[int],
    isolated_time: IsolatedTimeFn,
    *,
    num_cores: int,
) -> tuple[IsolatedJob, ...]:
    """Build one moldable job per positive-route expert."""

    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    normalized_widths = tuple(sorted({int(width) for width in widths}))
    if not normalized_widths:
        raise ValueError("at least one thread width is required")
    if normalized_widths[0] <= 0 or normalized_widths[-1] > num_cores:
        raise ValueError(f"thread widths must be within [1, {num_cores}], got {normalized_widths}")

    jobs: list[IsolatedJob] = []
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
        modes = []
        for threads in normalized_widths:
            try:
                duration_ns = float(isolated_time(routes, threads))
            except Exception as error:
                raise ValueError(
                    f"failed to evaluate isolated time for expert_id={expert_id}, routes={routes}, threads={threads}"
                ) from error
            modes.append(IsolatedMode(threads=threads, duration_ns=duration_ns))
        jobs.append(IsolatedJob(expert_id=expert_id, routes=routes, modes=tuple(modes)))
    if not jobs:
        raise ValueError("at least one active expert is required")
    return tuple(jobs)


def _normalize_jobs(
    jobs: Sequence[IsolatedJob],
    *,
    num_cores: int,
    prune_dominated: bool,
) -> tuple[tuple[IsolatedJob, ...], int]:
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    if not jobs:
        raise ValueError("at least one active expert is required")

    normalized: list[IsolatedJob] = []
    seen_experts: set[int] = set()
    input_modes = 0
    for job in jobs:
        if job.expert_id in seen_experts:
            raise ValueError(f"duplicate expert_id={job.expert_id}")
        seen_experts.add(job.expert_id)
        if job.routes <= 0:
            raise ValueError(f"job routes must be positive, got expert_id={job.expert_id}, routes={job.routes}")
        input_modes += len(job.modes)
        by_threads: dict[int, IsolatedMode] = {}
        for mode in job.modes:
            if mode.threads <= 0:
                raise ValueError(f"mode threads must be positive, got {mode.threads}")
            if not math.isfinite(mode.duration_ns) or mode.duration_ns <= 0.0:
                raise ValueError(f"mode duration must be finite and positive, got {mode.duration_ns!r}")
            if mode.threads > num_cores:
                continue
            previous = by_threads.get(mode.threads)
            if previous is None or mode.duration_ns < previous.duration_ns:
                by_threads[mode.threads] = mode
        modes = tuple(sorted(by_threads.values(), key=lambda mode: mode.threads))
        if not modes:
            raise ValueError(f"expert_id={job.expert_id} has no mode within the {num_cores}-core capacity")
        modes = prune_dominated_modes(modes) if prune_dominated else modes
        if not modes:
            raise ValueError(f"expert_id={job.expert_id} has no retained execution mode")
        normalized.append(IsolatedJob(job.expert_id, job.routes, modes))
    return tuple(normalized), input_modes


def _homogeneous_incumbent(
    quantized: Sequence[Sequence[tuple[IsolatedMode, int]]],
    *,
    num_cores: int,
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    """Build a deterministic feasible hint from the best common-width LPT schedule."""

    by_width = [
        {mode.threads: (mode_index, duration) for mode_index, (mode, duration) in enumerate(modes)}
        for modes in quantized
    ]
    common_widths = set(by_width[0])
    for modes in by_width[1:]:
        common_widths.intersection_update(modes)

    candidates: list[tuple[int, tuple[tuple[int, int, int], ...]]] = []
    for width in sorted(common_widths):
        lanes = num_cores // width
        if lanes <= 0:
            continue
        lane_heap = [(0, lane) for lane in range(lanes)]
        heapq.heapify(lane_heap)
        schedule: list[tuple[int, int, int] | None] = [None] * len(quantized)
        order = sorted(
            range(len(quantized)),
            key=lambda job_index: (-by_width[job_index][width][1], job_index),
        )
        for job_index in order:
            available, lane = heapq.heappop(lane_heap)
            mode_index, duration = by_width[job_index][width]
            end = available + duration
            schedule[job_index] = (mode_index, available, end)
            heapq.heappush(lane_heap, (end, lane))
        concrete = tuple(item for item in schedule if item is not None)
        candidates.append((max(end for _, _, end in concrete), concrete))

    serial: list[tuple[int, int, int]] = []
    cursor = 0
    for modes in quantized:
        mode_index, (_, duration) = min(enumerate(modes), key=lambda item: (item[1][1], item[1][0].threads))
        serial.append((mode_index, cursor, cursor + duration))
        cursor += duration
    candidates.append((cursor, tuple(serial)))
    objective, schedule = min(candidates, key=lambda candidate: candidate[0])
    return schedule, objective


def solve_isolated_cp_sat(
    jobs: Sequence[IsolatedJob],
    *,
    num_cores: int,
    max_time_s: float = 60.0,
    workers: int = 1,
    relative_gap_limit: float = 0.0,
    time_quantum_ns: int = 1,
    prune_dominated: bool = True,
    random_seed: int = 0,
    log_search_progress: bool = False,
) -> IsolatedOracleResult:
    """Solve the no-contention moldable-job makespan problem."""

    if max_time_s <= 0.0 or not math.isfinite(max_time_s):
        raise ValueError(f"max_time_s must be finite and positive, got {max_time_s!r}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if relative_gap_limit < 0.0 or relative_gap_limit >= 1.0:
        raise ValueError(f"relative_gap_limit must be in [0, 1), got {relative_gap_limit}")
    normalized, input_modes = _normalize_jobs(
        jobs,
        num_cores=num_cores,
        prune_dominated=prune_dominated,
    )
    cp_model = _cp_model_module()
    model = cp_model.CpModel()

    quantized: list[list[tuple[IsolatedMode, int]]] = [
        [(mode, _quantize_duration(mode.duration_ns, time_quantum_ns)) for mode in job.modes] for job in normalized
    ]
    horizon = sum(min(duration for _, duration in modes) for modes in quantized)
    if horizon <= 0 or horizon >= (1 << 62):
        raise ValueError(f"invalid CP-SAT horizon={horizon} ticks")

    intervals = []
    demands = []
    variables = []
    job_starts = []
    job_ends = []
    job_mode_indices = []
    min_area = 0
    longest_job = 0
    for job_index, (job, modes) in enumerate(zip(normalized, quantized)):
        presences = []
        job_variables = []
        job_start = model.new_int_var(0, horizon, f"job_start_e{job.expert_id}_j{job_index}")
        job_end = model.new_int_var(0, horizon, f"job_end_e{job.expert_id}_j{job_index}")
        job_mode_index = model.new_int_var(0, len(modes) - 1, f"job_mode_e{job.expert_id}_j{job_index}")
        min_area += min(mode.threads * duration for mode, duration in modes)
        longest_job = max(longest_job, min(duration for _, duration in modes))
        for mode_index, (mode, duration) in enumerate(modes):
            suffix = f"e{job.expert_id}_j{job_index}_m{mode_index}_t{mode.threads}"
            presence = model.new_bool_var(f"select_{suffix}")
            start = model.new_int_var(0, horizon, f"start_{suffix}")
            end = model.new_int_var(0, horizon, f"end_{suffix}")
            interval = model.new_optional_interval_var(start, duration, end, presence, f"run_{suffix}")
            model.add(start == 0).only_enforce_if(presence.Not())
            model.add(end == 0).only_enforce_if(presence.Not())
            model.add(job_start == start).only_enforce_if(presence)
            model.add(job_end == end).only_enforce_if(presence)
            model.add(job_mode_index == mode_index).only_enforce_if(presence)
            presences.append(presence)
            intervals.append(interval)
            demands.append(mode.threads)
            job_variables.append((mode, duration, presence, start, end))
        model.add_exactly_one(presences)
        variables.append(job_variables)
        job_starts.append(job_start)
        job_ends.append(job_end)
        job_mode_indices.append(job_mode_index)

    model.add_cumulative(intervals, demands, num_cores)
    first_start = model.new_int_var(0, 0, "first_start")
    model.add_min_equality(first_start, job_starts)
    area_bound = (min_area + num_cores - 1) // num_cores
    model_bound = max(longest_job, area_bound)
    makespan = model.new_int_var(model_bound, horizon, "makespan")
    model.add_max_equality(makespan, job_ends)
    model.minimize(makespan)
    hint, hint_makespan = _homogeneous_incumbent(quantized, num_cores=num_cores)
    for job_index, (mode_index, start_hint, end_hint) in enumerate(hint):
        for candidate_index, (_, _, presence, start, end) in enumerate(variables[job_index]):
            selected = candidate_index == mode_index
            model.add_hint(presence, int(selected))
            model.add_hint(start, start_hint if selected else 0)
            model.add_hint(end, end_hint if selected else 0)
        model.add_hint(job_starts[job_index], start_hint)
        model.add_hint(job_ends[job_index], end_hint)
        model.add_hint(job_mode_indices[job_index], mode_index)
    model.add_hint(makespan, hint_makespan)

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
    bound_gap = None
    if objective_ns is not None and best_bound_ns is not None:
        relative_gap = max((objective_ns - best_bound_ns) / max(abs(objective_ns), 1.0), 0.0)
        if best_bound_ns > 0.0:
            bound_gap = max((objective_ns - best_bound_ns) / best_bound_ns, 0.0)

    assignments: list[OracleAssignment] = []
    if has_solution:
        for job, job_variables in zip(normalized, variables):
            selected = [
                (mode, duration, start, end)
                for mode, duration, presence, start, end in job_variables
                if solver.boolean_value(presence)
            ]
            if len(selected) != 1:
                raise RuntimeError(f"CP-SAT selected {len(selected)} modes for expert_id={job.expert_id}")
            mode, duration, start, end = selected[0]
            assignments.append(
                OracleAssignment(
                    expert_id=job.expert_id,
                    routes=job.routes,
                    threads=mode.threads,
                    start_ns=int(solver.value(start) * time_quantum_ns),
                    duration_ns=int(duration * time_quantum_ns),
                    end_ns=int(solver.value(end) * time_quantum_ns),
                )
            )
        assignments.sort(key=lambda assignment: (assignment.start_ns, assignment.end_ns, assignment.expert_id))

    proven_optimal = (
        status_code == cp_model.OPTIMAL
        and objective_ns is not None
        and best_bound_ns is not None
        and abs(objective_ns - best_bound_ns) < 0.5 * time_quantum_ns
    )
    return IsolatedOracleResult(
        status=status,
        optimal=proven_optimal,
        objective_ns=objective_ns,
        best_bound_ns=best_bound_ns,
        relative_gap=relative_gap,
        bound_gap=bound_gap,
        wall_time_s=float(solver.wall_time),
        conflicts=int(solver.num_conflicts),
        branches=int(solver.num_branches),
        num_jobs=len(normalized),
        input_modes=input_modes,
        retained_modes=sum(len(job.modes) for job in normalized),
        time_quantum_ns=time_quantum_ns,
        quantization_error_bound_ns=0.5 * len(normalized) * time_quantum_ns,
        assignments=tuple(assignments),
    )


def evaluate_isolated_dag(
    tasks: Sequence[tuple[int, int, int, int, Sequence[int]]],
    isolated_time: IsolatedTimeFn,
    *,
    time_quantum_ns: int = 1,
) -> int:
    """Evaluate a fixed interval-planner DAG with the oracle's duration grid."""

    if not tasks:
        return 0
    children: list[list[int]] = [[] for _ in tasks]
    indegree = [0] * len(tasks)
    ready_time = [0] * len(tasks)
    durations = []
    for index, (_, routes, _, threads, dependencies) in enumerate(tasks):
        durations.append(_quantize_duration(float(isolated_time(int(routes), int(threads))), time_quantum_ns))
        unique_dependencies = set(int(dependency) for dependency in dependencies)
        if index in unique_dependencies:
            raise ValueError(f"task {index} depends on itself")
        for dependency in unique_dependencies:
            if dependency < 0 or dependency >= len(tasks):
                raise ValueError(f"task {index} has out-of-range dependency {dependency}")
            children[dependency].append(index)
            indegree[index] += 1

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    finish = [0] * len(tasks)
    completed = 0
    while ready:
        index = ready.pop()
        finish[index] = ready_time[index] + durations[index]
        completed += 1
        for child in children[index]:
            ready_time[child] = max(ready_time[child], finish[index])
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if completed != len(tasks):
        raise ValueError("task dependencies contain a cycle")
    return max(finish) * time_quantum_ns


def compare_candidate_to_oracle(
    candidate_ns: float,
    oracle: IsolatedOracleResult,
) -> OracleComparison:
    if not math.isfinite(candidate_ns) or candidate_ns <= 0.0:
        raise ValueError(f"candidate_ns must be finite and positive, got {candidate_ns!r}")
    exact_regret = None
    if oracle.optimal and oracle.objective_ns is not None:
        exact_regret = candidate_ns / oracle.objective_ns - 1.0
    regret_lower_bound = None
    if oracle.objective_ns is not None and oracle.objective_ns > 0:
        regret_lower_bound = max(candidate_ns / oracle.objective_ns - 1.0, 0.0)
    regret_upper_bound = None
    bound_efficiency = None
    if oracle.best_bound_ns is not None and oracle.best_bound_ns > 0.0:
        regret_upper_bound = candidate_ns / oracle.best_bound_ns - 1.0
        bound_efficiency = oracle.best_bound_ns / candidate_ns
    return OracleComparison(
        candidate_ns=float(candidate_ns),
        oracle_objective_ns=oracle.objective_ns,
        oracle_best_bound_ns=oracle.best_bound_ns,
        exact_regret=exact_regret,
        regret_lower_bound=regret_lower_bound,
        regret_upper_bound=regret_upper_bound,
        bound_efficiency=bound_efficiency,
    )


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _load_profile_model(profile: Path):
    root = Path(__file__).resolve().parents[1]
    cost_model_dir = root / "cost_model"
    planners_dir = root / "planners"
    sys.path[:0] = [str(cost_model_dir), str(planners_dir)]
    from phase_model import ContentionCostModel

    return ContentionCostModel(profile)


def _model_widths(model, num_cores: int) -> tuple[int, ...]:
    declared = getattr(model, "supported_widths", None)
    if declared is None:
        declared = {int(width) for shape in getattr(model, "supported_shapes", ()) for width in shape}
    widths = tuple(sorted({int(width) for width in declared if 0 < int(width) <= num_cores}))
    if not widths:
        raise ValueError("cost model does not expose any thread width within the requested core capacity")
    return widths


def _run_cli(args: argparse.Namespace) -> dict[str, object]:
    model = _load_profile_model(args.profile)
    routes = _parse_int_list(args.routes, name="routes")
    experts = tuple((expert_id, count) for expert_id, count in enumerate(routes) if count > 0)
    if args.widths:
        widths = _parse_int_list(args.widths, name="widths")
    else:
        widths = _model_widths(model, args.num_cores)
    jobs = build_isolated_jobs(experts, widths, model.T_iso, num_cores=args.num_cores)
    oracle = solve_isolated_cp_sat(
        jobs,
        num_cores=args.num_cores,
        max_time_s=args.max_time_s,
        workers=args.workers,
        relative_gap_limit=args.relative_gap_limit,
        time_quantum_ns=args.time_quantum_ns,
        prune_dominated=not args.keep_dominated_modes,
        random_seed=args.random_seed,
        log_search_progress=args.log_search_progress,
    )
    report: dict[str, object] = {
        "model": "isolated_cp_sat_v1",
        "profile": str(args.profile),
        "num_cores": args.num_cores,
        "routes": list(routes),
        "widths": list(widths),
        "oracle": oracle.to_dict(),
    }
    if not args.no_planner_comparison:
        from interval_planner import IntervalPlanner

        planner = IntervalPlanner(
            model,
            num_cores=args.num_cores,
            widths=widths,
            native_cold_planner=False,
        )
        selected = planner.plan(list(experts), dynamic_tail_pool=False)
        planner_isolated_ns = evaluate_isolated_dag(
            selected["tasks"],
            model.T_iso,
            time_quantum_ns=args.time_quantum_ns,
        )
        report["planner"] = {
            "shape": list(selected["shape"]),
            "contention_model_makespan_ns": selected["makespan_ns"],
            "isolated_makespan_ns": planner_isolated_ns,
            "comparison": compare_candidate_to_oracle(planner_isolated_ns, oracle).to_dict(),
        }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="schema-v2 empirical cost profile")
    parser.add_argument("--routes", required=True, help="comma-separated expert route histogram")
    parser.add_argument("--num-cores", type=int, required=True)
    parser.add_argument("--widths", default="", help="comma-separated widths; default: profile-supported widths")
    parser.add_argument("--max-time-s", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--relative-gap-limit", type=float, default=0.0)
    parser.add_argument(
        "--time-quantum-ns",
        type=int,
        default=1000,
        help="duration discretization; 1000 ns is usually much faster, use 1 ns for maximum fidelity",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--keep-dominated-modes", action="store_true")
    parser.add_argument("--no-planner-comparison", action="store_true")
    parser.add_argument("--log-search-progress", action="store_true")
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
