"""Certified resource lower bounds for phase-moldable MoE schedules.

This module deliberately does not construct an executable schedule.  It relaxes
each stage to a convex combination of legal execution modes and lower-bounds the
makespan with resource conservation and explicitly declared critical chains.

The mode-relaxed LP is solved through its zero-sum dual.  Every dual iterate is
a valid lower-bound certificate, even when the iterative solver stops before
closing its primal-dual gap.  No measured task duration or CP-SAT interval is
required by the solver itself.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ResourceCapacity:
    """Aggregate capacity of one conserved resource.

    ``units_per_second`` must be an upper bound on service available to the
    complete scheduling domain when the result is presented as a physical
    lower-bound certificate.
    """

    name: str
    units_per_second: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resource name must be non-empty")
        if not math.isfinite(self.units_per_second) or self.units_per_second <= 0.0:
            raise ValueError(f"resource {self.name!r} capacity must be finite and positive")


@dataclass(frozen=True)
class ResourceDemand:
    """Unavoidable amount of one resource consumed by a stage mode."""

    resource: str
    amount: float

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("demand resource name must be non-empty")
        if not math.isfinite(self.amount) or self.amount < 0.0:
            raise ValueError(f"resource demand for {self.resource!r} must be finite and non-negative")


@dataclass(frozen=True)
class StageMode:
    """One legal implementation mode for a logical stage.

    Modes of the same stage must perform the same logical operation.  Resource
    demands may differ because thread width, padding, windows, and physical
    kernel work differ.  ``duration_lower_bound_s`` is a mode-local wall-time
    floor used only by critical-chain constraints.
    """

    name: str
    threads: int
    demands: tuple[ResourceDemand, ...]
    duration_lower_bound_s: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mode name must be non-empty")
        if self.threads <= 0:
            raise ValueError(f"mode {self.name!r} threads must be positive")
        if not math.isfinite(self.duration_lower_bound_s) or self.duration_lower_bound_s < 0.0:
            raise ValueError(f"mode {self.name!r} duration lower bound must be finite and non-negative")
        names = [demand.resource for demand in self.demands]
        if len(names) != len(set(names)):
            raise ValueError(f"mode {self.name!r} contains duplicate resource demands")

    @classmethod
    def from_mapping(
        cls,
        name: str,
        threads: int,
        demands: Mapping[str, float],
        duration_lower_bound_s: float,
    ) -> StageMode:
        return cls(
            name=name,
            threads=threads,
            demands=tuple(
                ResourceDemand(resource=resource, amount=float(amount))
                for resource, amount in sorted(demands.items())
            ),
            duration_lower_bound_s=float(duration_lower_bound_s),
        )


@dataclass(frozen=True)
class MoldableStage:
    """A logical stage and all legal modes admitted by the relaxation."""

    stage_id: str
    modes: tuple[StageMode, ...]

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must be non-empty")
        if not self.modes:
            raise ValueError(f"stage {self.stage_id!r} must have at least one mode")
        names = [mode.name for mode in self.modes]
        if len(names) != len(set(names)):
            raise ValueError(f"stage {self.stage_id!r} contains duplicate mode names")


@dataclass(frozen=True)
class CriticalChain:
    """A serial path whose stage-local lower bounds cannot overlap."""

    name: str
    stage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("critical-chain name must be non-empty")
        if not self.stage_ids:
            raise ValueError(f"critical chain {self.name!r} must contain at least one stage")
        if len(self.stage_ids) != len(set(self.stage_ids)):
            raise ValueError(f"critical chain {self.name!r} contains a stage more than once")


@dataclass(frozen=True)
class LowerBoundProblem:
    """Hardware envelope and relaxed MoE stage graph."""

    resources: tuple[ResourceCapacity, ...]
    stages: tuple[MoldableStage, ...]
    critical_chains: tuple[CriticalChain, ...] = ()

    def __post_init__(self) -> None:
        resource_names = [resource.name for resource in self.resources]
        if not resource_names:
            raise ValueError("at least one resource capacity is required")
        if len(resource_names) != len(set(resource_names)):
            raise ValueError("resource capacity names must be unique")

        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("stage identifiers must be unique")
        known_resources = set(resource_names)
        for stage in self.stages:
            for mode in stage.modes:
                unknown = {demand.resource for demand in mode.demands} - known_resources
                if unknown:
                    raise ValueError(
                        f"stage {stage.stage_id!r} mode {mode.name!r} uses unknown resources {sorted(unknown)}"
                    )

        chain_names = [chain.name for chain in self.critical_chains]
        if len(chain_names) != len(set(chain_names)):
            raise ValueError("critical-chain names must be unique")
        known_stages = set(stage_ids)
        for chain in self.critical_chains:
            unknown = set(chain.stage_ids) - known_stages
            if unknown:
                raise ValueError(f"critical chain {chain.name!r} uses unknown stages {sorted(unknown)}")


@dataclass(frozen=True)
class BoundTerm:
    constraint: str
    lower_bound_s: float


@dataclass(frozen=True)
class Lb0Certificate:
    """Independent per-constraint resource/chain lower bound."""

    lower_bound_s: float
    terms: tuple[BoundTerm, ...]
    active_constraints: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintWeight:
    constraint: str
    numerator: int
    denominator: int
    weight: float


@dataclass(frozen=True)
class ModeFraction:
    mode: str
    numerator: int
    denominator: int
    fraction: float


@dataclass(frozen=True)
class StageMixture:
    stage_id: str
    modes: tuple[ModeFraction, ...]


@dataclass(frozen=True)
class ModeRelaxedLpCertificate:
    """Primal/dual certificate for the mode-relaxed minimax LP.

    ``lower_bound_s`` is dual feasible and therefore certified even when
    ``converged`` is false.  ``primal_upper_bound_s`` is the objective of a
    feasible convex mode mixture, so the LP optimum lies between both values.
    """

    lower_bound_s: float
    raw_dual_lower_bound_s: float
    primal_upper_bound_s: float
    absolute_gap_s: float
    relative_gap: float
    numerical_safety_s: float
    dual_weight_denominator: int
    solver: str
    solver_status: str
    converged: bool
    iterations: int
    constraint_weights: tuple[ConstraintWeight, ...]
    stage_mixtures: tuple[StageMixture, ...]


@dataclass(frozen=True)
class LowerBoundCertificate:
    """Complete first-version lower-bound report."""

    lb0: Lb0Certificate
    mode_relaxed_lp: ModeRelaxedLpCertificate

    @property
    def lower_bound_s(self) -> float:
        return max(self.lb0.lower_bound_s, self.mode_relaxed_lp.lower_bound_s)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["lower_bound_s"] = self.lower_bound_s
        return payload


@dataclass(frozen=True)
class _IndexedMode:
    mode: StageMode
    coefficients_s: tuple[float, ...]
    sparse_coefficients_s: tuple[tuple[int, float], ...]
    exact_coefficients_s: tuple[Fraction, ...]
    sparse_exact_coefficients_s: tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True)
class _IndexedProblem:
    constraint_names: tuple[str, ...]
    stages: tuple[tuple[_IndexedMode, ...], ...]


def _constraint_names(problem: LowerBoundProblem) -> tuple[str, ...]:
    return (
        *(f"resource:{resource.name}" for resource in problem.resources),
        *(f"chain:{chain.name}" for chain in problem.critical_chains),
    )


def _index_problem(problem: LowerBoundProblem) -> _IndexedProblem:
    capacities = {resource.name: resource.units_per_second for resource in problem.resources}
    resource_indices = {resource.name: index for index, resource in enumerate(problem.resources)}
    chain_offset = len(problem.resources)
    chain_membership: dict[str, list[int]] = {stage.stage_id: [] for stage in problem.stages}
    for chain_index, chain in enumerate(problem.critical_chains):
        for stage_id in chain.stage_ids:
            chain_membership[stage_id].append(chain_offset + chain_index)

    indexed_stages = []
    constraint_count = len(problem.resources) + len(problem.critical_chains)
    for stage in problem.stages:
        indexed_modes = []
        for mode in stage.modes:
            coefficients = [0.0] * constraint_count
            exact_coefficients = [Fraction(0)] * constraint_count
            for demand in mode.demands:
                resource_index = resource_indices[demand.resource]
                exact_coefficient = Fraction.from_float(demand.amount) / Fraction.from_float(
                    capacities[demand.resource]
                )
                exact_coefficients[resource_index] = exact_coefficient
                coefficients[resource_index] = float(exact_coefficient)
            for chain_index in chain_membership[stage.stage_id]:
                coefficients[chain_index] = mode.duration_lower_bound_s
                exact_coefficients[chain_index] = Fraction.from_float(mode.duration_lower_bound_s)
            indexed_modes.append(
                _IndexedMode(
                    mode=mode,
                    coefficients_s=tuple(coefficients),
                    sparse_coefficients_s=tuple(
                        (index, coefficient) for index, coefficient in enumerate(coefficients) if coefficient > 0.0
                    ),
                    exact_coefficients_s=tuple(exact_coefficients),
                    sparse_exact_coefficients_s=tuple(
                        (index, coefficient)
                        for index, coefficient in enumerate(exact_coefficients)
                        if coefficient > 0
                    ),
                )
            )
        indexed_stages.append(tuple(indexed_modes))
    return _IndexedProblem(
        constraint_names=_constraint_names(problem),
        stages=tuple(indexed_stages),
    )


def compute_lb0(problem: LowerBoundProblem) -> Lb0Certificate:
    """Compute the independent-resource and independent-chain lower bound.

    Every term minimizes its own constraint over modes.  Different terms may
    therefore select incompatible modes; taking their maximum is deliberately
    optimistic and remains a valid lower bound.
    """

    indexed = _index_problem(problem)
    terms = []
    for constraint_index, constraint_name in enumerate(indexed.constraint_names):
        exact_value = sum(
            min(mode.exact_coefficients_s[constraint_index] for mode in modes)
            for modes in indexed.stages
        )
        value = _round_fraction_down(exact_value)
        terms.append(BoundTerm(constraint=constraint_name, lower_bound_s=value))
    lower_bound = max((term.lower_bound_s for term in terms), default=0.0)
    tolerance = max(1e-15, abs(lower_bound) * 1e-12)
    active = tuple(term.constraint for term in terms if lower_bound - term.lower_bound_s <= tolerance)
    return Lb0Certificate(lower_bound_s=lower_bound, terms=tuple(terms), active_constraints=active)


def _best_response(indexed: _IndexedProblem, weights: Sequence[float]) -> tuple[list[int], list[float]]:
    selected_modes = []
    loads = [0.0] * len(indexed.constraint_names)
    for modes in indexed.stages:
        selected_index = min(
            range(len(modes)),
            key=lambda mode_index: (
                sum(weights[index] * value for index, value in modes[mode_index].sparse_coefficients_s),
                mode_index,
            ),
        )
        selected_modes.append(selected_index)
        for constraint_index, value in modes[selected_index].sparse_coefficients_s:
            loads[constraint_index] += value
    return selected_modes, loads


def _normalized_weights(log_weights: Sequence[float]) -> list[float]:
    maximum = max(log_weights)
    exponentials = [math.exp(value - maximum) for value in log_weights]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _quantize_simplex(weights: Sequence[float], denominator: int) -> tuple[int, ...]:
    """Map floating weights to an exact integer simplex deterministically."""

    if denominator <= 0:
        raise ValueError("dual weight denominator must be positive")
    if not weights or any(not math.isfinite(weight) for weight in weights):
        raise ValueError("simplex weights must be non-empty and finite")
    clipped = [Fraction.from_float(max(0.0, weight)) for weight in weights]
    total = sum(clipped)
    if total <= 0:
        raise ValueError("simplex weights must contain a positive entry")
    scaled = [weight * denominator / total for weight in clipped]
    numerators = [value.numerator // value.denominator for value in scaled]
    missing = denominator - sum(numerators)
    order = sorted(
        range(len(weights)),
        key=lambda index: (scaled[index] - numerators[index], -index),
        reverse=True,
    )
    for index in order[:missing]:
        numerators[index] += 1
    if min(numerators) < 0 or sum(numerators) != denominator:
        raise AssertionError("quantized dual weights must form an exact simplex")
    return tuple(numerators)


def _exact_dual_value(indexed: _IndexedProblem, numerators: Sequence[int], denominator: int) -> Fraction:
    weights = tuple(Fraction(numerator, denominator) for numerator in numerators)
    return sum(
        min(
            sum(weights[index] * value for index, value in mode.sparse_exact_coefficients_s)
            for mode in modes
        )
        for modes in indexed.stages
    )


def _exact_primal_value(
    indexed: _IndexedProblem,
    mode_numerators: Sequence[Sequence[int]],
    denominator: int,
) -> Fraction:
    loads = [Fraction(0) for _ in indexed.constraint_names]
    for modes, numerators in zip(indexed.stages, mode_numerators):
        for mode, numerator in zip(modes, numerators):
            if numerator == 0:
                continue
            fraction = Fraction(numerator, denominator)
            for constraint_index, coefficient in mode.sparse_exact_coefficients_s:
                loads[constraint_index] += fraction * coefficient
    return max(loads, default=Fraction(0))


def _round_fraction_down(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def _round_fraction_up(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _certify_best_dual(
    indexed: _IndexedProblem,
    candidate_weights: Iterable[Sequence[float]],
    denominator: int,
) -> tuple[Fraction, tuple[int, ...]]:
    best_value = Fraction(-1)
    best_numerators: tuple[int, ...] = ()
    for weights in candidate_weights:
        numerators = _quantize_simplex(weights, denominator)
        value = _exact_dual_value(indexed, numerators, denominator)
        if value > best_value:
            best_value = value
            best_numerators = numerators
    return best_value, best_numerators


def _best_one_hot_weights(indexed: _IndexedProblem) -> tuple[float, ...]:
    active_index = max(
        range(len(indexed.constraint_names)),
        key=lambda index: sum(
            min(mode.exact_coefficients_s[index] for mode in modes) for modes in indexed.stages
        ),
    )
    return tuple(float(index == active_index) for index in range(len(indexed.constraint_names)))


def _solve_glop(
    indexed: _IndexedProblem,
) -> tuple[list[float], list[list[float]], float, int, str] | None:
    try:
        from ortools.linear_solver import pywraplp
    except ImportError:
        return None

    infinity = pywraplp.Solver.infinity()
    primal = pywraplp.Solver.CreateSolver("GLOP")
    dual = pywraplp.Solver.CreateSolver("GLOP")
    if primal is None or dual is None:
        return None

    primal_t = primal.NumVar(0.0, infinity, "T")
    primal_modes = []
    for stage_index, modes in enumerate(indexed.stages):
        variables = [primal.NumVar(0.0, 1.0, f"x_{stage_index}_{mode_index}") for mode_index in range(len(modes))]
        simplex = primal.Constraint(1.0, 1.0)
        for variable in variables:
            simplex.SetCoefficient(variable, 1.0)
        primal_modes.append(variables)
    capacities = [primal.Constraint(-infinity, 0.0) for _ in indexed.constraint_names]
    for capacity in capacities:
        capacity.SetCoefficient(primal_t, -1.0)
    for modes, variables in zip(indexed.stages, primal_modes):
        for mode, variable in zip(modes, variables):
            for constraint_index, coefficient in mode.sparse_coefficients_s:
                capacities[constraint_index].SetCoefficient(variable, coefficient)
    primal.Objective().SetCoefficient(primal_t, 1.0)
    primal.Objective().SetMinimization()

    dual_weights = [dual.NumVar(0.0, 1.0, f"lambda_{index}") for index in range(len(indexed.constraint_names))]
    dual_simplex = dual.Constraint(1.0, 1.0)
    for variable in dual_weights:
        dual_simplex.SetCoefficient(variable, 1.0)
    dual_stage_values = []
    for stage_index, modes in enumerate(indexed.stages):
        stage_value = dual.NumVar(-infinity, infinity, f"z_{stage_index}")
        dual_stage_values.append(stage_value)
        for mode in modes:
            upper = dual.Constraint(-infinity, 0.0)
            upper.SetCoefficient(stage_value, 1.0)
            for constraint_index, coefficient in mode.sparse_coefficients_s:
                upper.SetCoefficient(dual_weights[constraint_index], -coefficient)
    for stage_value in dual_stage_values:
        dual.Objective().SetCoefficient(stage_value, 1.0)
    dual.Objective().SetMaximization()

    primal_status = primal.Solve()
    dual_status = dual.Solve()
    optimal = pywraplp.Solver.OPTIMAL
    if primal_status != optimal or dual_status != optimal:
        raise RuntimeError(f"GLOP failed to solve lower-bound LP: primal={primal_status}, dual={dual_status}")
    weights = [max(0.0, variable.solution_value()) for variable in dual_weights]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise RuntimeError("GLOP returned an empty dual simplex")
    weights = [weight / weight_sum for weight in weights]
    mixtures = []
    for variables in primal_modes:
        values = [max(0.0, variable.solution_value()) for variable in variables]
        total = sum(values)
        if total <= 0.0:
            raise RuntimeError("GLOP returned an empty primal stage simplex")
        mixtures.append([value / total for value in values])
    iterations = int(primal.iterations()) + int(dual.iterations())
    return weights, mixtures, float(dual.Objective().Value()), iterations, "OPTIMAL"


def _solve_mirror(
    indexed: _IndexedProblem,
    *,
    max_iterations: int,
    relative_tolerance: float,
    minimum_iterations: int,
) -> tuple[list[float], list[list[int]], float, int, str]:
    constraint_count = len(indexed.constraint_names)
    max_constraint_load = max(
        sum(max(mode.coefficients_s[index] for mode in modes) for modes in indexed.stages)
        for index in range(constraint_count)
    )
    log_weights = [0.0] * constraint_count
    learning_rate = math.sqrt(2.0 * math.log(max(2, constraint_count)) / max_iterations)
    cumulative_loads = [0.0] * constraint_count
    mode_counts = [[0] * len(modes) for modes in indexed.stages]
    best_dual = -math.inf
    best_weights = [1.0 / constraint_count] * constraint_count

    for iteration in range(1, max_iterations + 1):
        weights = _normalized_weights(log_weights)
        selected_modes, loads = _best_response(indexed, weights)
        dual_value = sum(weights[index] * loads[index] for index in range(constraint_count))
        if dual_value > best_dual:
            best_dual = dual_value
            best_weights = weights.copy()
        for stage_index, mode_index in enumerate(selected_modes):
            mode_counts[stage_index][mode_index] += 1
        for index, load in enumerate(loads):
            cumulative_loads[index] += load
        primal_upper = max(load / iteration for load in cumulative_loads)
        if iteration >= minimum_iterations:
            gap = max(0.0, primal_upper - best_dual)
            if gap <= relative_tolerance * max(primal_upper, 1e-30):
                return best_weights, mode_counts, best_dual, iteration, "GAP_REACHED"
        for index, load in enumerate(loads):
            log_weights[index] += learning_rate * load / max_constraint_load
    return best_weights, mode_counts, best_dual, max_iterations, "ITERATION_LIMIT"


def solve_mode_relaxed_lp(
    problem: LowerBoundProblem,
    *,
    max_iterations: int = 20_000,
    relative_tolerance: float = 1e-4,
    minimum_iterations: int = 100,
    dual_weight_denominator: int = 1 << 40,
    solver: str = "auto",
) -> ModeRelaxedLpCertificate:
    """Solve the mode-relaxed LP with a certified dual lower bound.

    The primal LP is

    ``min T`` subject to ``sum(j,m) c[a,j,m] x[j,m] <= T`` for every
    resource/chain constraint, and ``sum(m) x[j,m] = 1`` for every stage.

    Its dual maximizes ``sum(j) min(m) dot(lambda, c[j,m])`` over the
    constraint simplex.  Entropic mirror ascent supplies dual-feasible iterates;
    the average best responses supply a primal-feasible mixture.  The returned
    lower bound therefore remains valid if the requested gap is not reached.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if minimum_iterations <= 0 or minimum_iterations > max_iterations:
        raise ValueError("minimum_iterations must be in [1, max_iterations]")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    if dual_weight_denominator <= 0:
        raise ValueError("dual_weight_denominator must be positive")
    if solver not in {"auto", "glop", "mirror"}:
        raise ValueError("solver must be 'auto', 'glop', or 'mirror'")

    indexed = _index_problem(problem)
    constraint_count = len(indexed.constraint_names)
    stage_count = len(indexed.stages)
    if stage_count == 0:
        weights = tuple(
            ConstraintWeight(constraint=name, numerator=1, denominator=constraint_count, weight=1.0 / constraint_count)
            for name in indexed.constraint_names
        )
        return ModeRelaxedLpCertificate(
            lower_bound_s=0.0,
            raw_dual_lower_bound_s=0.0,
            primal_upper_bound_s=0.0,
            absolute_gap_s=0.0,
            relative_gap=0.0,
            numerical_safety_s=0.0,
            dual_weight_denominator=constraint_count,
            solver="trivial",
            solver_status="EMPTY_WORKLOAD",
            converged=True,
            iterations=0,
            constraint_weights=weights,
            stage_mixtures=(),
        )

    max_constraint_load = max(
        sum(max(mode.coefficients_s[index] for mode in modes) for modes in indexed.stages)
        for index in range(constraint_count)
    )
    if max_constraint_load == 0.0:
        mixtures = tuple(
            StageMixture(
                stage_id=stage.stage_id,
                modes=(ModeFraction(stage.modes[0].name, numerator=1, denominator=1, fraction=1.0),),
            )
            for stage in problem.stages
        )
        weights = tuple(
            ConstraintWeight(constraint=name, numerator=1, denominator=constraint_count, weight=1.0 / constraint_count)
            for name in indexed.constraint_names
        )
        return ModeRelaxedLpCertificate(
            lower_bound_s=0.0,
            raw_dual_lower_bound_s=0.0,
            primal_upper_bound_s=0.0,
            absolute_gap_s=0.0,
            relative_gap=0.0,
            numerical_safety_s=0.0,
            dual_weight_denominator=constraint_count,
            solver="trivial",
            solver_status="ZERO_DEMAND",
            converged=True,
            iterations=0,
            constraint_weights=weights,
            stage_mixtures=mixtures,
        )

    glop_result = _solve_glop(indexed) if solver in {"auto", "glop"} else None
    if glop_result is None:
        if solver == "glop":
            raise RuntimeError("GLOP is unavailable; install the optional 'oracle' dependency")
        best_weights, mode_counts, raw_dual, iterations, solver_status = _solve_mirror(
            indexed,
            max_iterations=max_iterations,
            relative_tolerance=relative_tolerance,
            minimum_iterations=minimum_iterations,
        )
        solver_name = "mirror"
        primal_denominator = iterations
        primal_numerators = mode_counts
    else:
        best_weights, primal_mixtures, raw_dual, iterations, solver_status = glop_result
        solver_name = "glop"
        primal_denominator = dual_weight_denominator
        primal_numerators = [
            list(_quantize_simplex(mixture, primal_denominator)) for mixture in primal_mixtures
        ]

    exact_dual, dual_numerators = _certify_best_dual(
        indexed,
        (_best_one_hot_weights(indexed), best_weights),
        dual_weight_denominator,
    )
    exact_primal = _exact_primal_value(indexed, primal_numerators, primal_denominator)
    certified_dual = max(0.0, _round_fraction_down(exact_dual))
    primal_upper = _round_fraction_up(exact_primal)
    numerical_safety = max(0.0, raw_dual - certified_dual)
    absolute_gap = max(0.0, primal_upper - certified_dual)
    relative_gap = absolute_gap / primal_upper if primal_upper > 0.0 else 0.0
    converged = relative_gap <= relative_tolerance

    constraint_weights = tuple(
        ConstraintWeight(
            constraint=name,
            numerator=dual_numerators[index],
            denominator=dual_weight_denominator,
            weight=dual_numerators[index] / dual_weight_denominator,
        )
        for index, name in enumerate(indexed.constraint_names)
        if dual_numerators[index] > 0
    )
    stage_mixtures = []
    for stage, numerators in zip(problem.stages, primal_numerators):
        fractions = tuple(
            ModeFraction(
                mode=mode.name,
                numerator=numerator,
                denominator=primal_denominator,
                fraction=numerator / primal_denominator,
            )
            for mode, numerator in zip(stage.modes, numerators)
            if numerator > 0
        )
        stage_mixtures.append(StageMixture(stage_id=stage.stage_id, modes=fractions))

    return ModeRelaxedLpCertificate(
        lower_bound_s=certified_dual,
        raw_dual_lower_bound_s=raw_dual,
        primal_upper_bound_s=primal_upper,
        absolute_gap_s=absolute_gap,
        relative_gap=relative_gap,
        numerical_safety_s=numerical_safety,
        dual_weight_denominator=dual_weight_denominator,
        solver=solver_name,
        solver_status=solver_status,
        converged=converged,
        iterations=iterations,
        constraint_weights=constraint_weights,
        stage_mixtures=tuple(stage_mixtures),
    )


def build_lower_bound_certificate(
    problem: LowerBoundProblem,
    *,
    max_iterations: int = 20_000,
    relative_tolerance: float = 1e-4,
    minimum_iterations: int = 100,
    dual_weight_denominator: int = 1 << 40,
    solver: str = "auto",
) -> LowerBoundCertificate:
    """Compute LB0 and the coupled mode-relaxed LP certificate."""

    return LowerBoundCertificate(
        lb0=compute_lb0(problem),
        mode_relaxed_lp=solve_mode_relaxed_lp(
            problem,
            max_iterations=max_iterations,
            relative_tolerance=relative_tolerance,
            minimum_iterations=minimum_iterations,
            dual_weight_denominator=dual_weight_denominator,
            solver=solver,
        ),
    )


def mode(
    name: str,
    threads: int,
    demands: Mapping[str, float] | Iterable[tuple[str, float]],
    duration_lower_bound_s: float,
) -> StageMode:
    """Concise constructor for callers that already have resource accounting."""

    demand_mapping = dict(demands)
    return StageMode.from_mapping(name, threads, demand_mapping, duration_lower_bound_s)


def problem_from_dict(payload: Mapping[str, object]) -> LowerBoundProblem:
    """Parse the stable, implementation-neutral lower-bound problem schema."""

    raw_resources = payload.get("resources")
    raw_stages = payload.get("stages")
    raw_chains = payload.get("critical_chains", ())
    if not isinstance(raw_resources, list) or not isinstance(raw_stages, list):
        raise ValueError("problem JSON must contain list-valued resources and stages")
    if not isinstance(raw_chains, list) and not isinstance(raw_chains, tuple):
        raise ValueError("problem JSON critical_chains must be a list when present")

    resources = []
    for entry in raw_resources:
        if not isinstance(entry, Mapping):
            raise ValueError("each resource entry must be an object")
        resources.append(
            ResourceCapacity(
                name=str(entry["name"]),
                units_per_second=float(entry["units_per_second"]),
            )
        )

    stages = []
    for entry in raw_stages:
        if not isinstance(entry, Mapping):
            raise ValueError("each stage entry must be an object")
        raw_modes = entry.get("modes")
        if not isinstance(raw_modes, list):
            raise ValueError("each stage must contain a list-valued modes field")
        modes = []
        for raw_mode in raw_modes:
            if not isinstance(raw_mode, Mapping):
                raise ValueError("each mode entry must be an object")
            raw_demands = raw_mode.get("demands")
            if not isinstance(raw_demands, Mapping):
                raise ValueError("each mode must contain an object-valued demands field")
            modes.append(
                StageMode.from_mapping(
                    name=str(raw_mode["name"]),
                    threads=int(raw_mode["threads"]),
                    demands={str(name): float(amount) for name, amount in raw_demands.items()},
                    duration_lower_bound_s=float(raw_mode["duration_lower_bound_s"]),
                )
            )
        stages.append(MoldableStage(stage_id=str(entry["stage_id"]), modes=tuple(modes)))

    chains = []
    for entry in raw_chains:
        if not isinstance(entry, Mapping):
            raise ValueError("each critical-chain entry must be an object")
        raw_stage_ids = entry.get("stage_ids")
        if not isinstance(raw_stage_ids, list):
            raise ValueError("each critical chain must contain a list-valued stage_ids field")
        chains.append(
            CriticalChain(
                name=str(entry["name"]),
                stage_ids=tuple(str(stage_id) for stage_id in raw_stage_ids),
            )
        )
    return LowerBoundProblem(
        resources=tuple(resources),
        stages=tuple(stages),
        critical_chains=tuple(chains),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path, help="lower-bound problem JSON")
    parser.add_argument("--output", type=Path, help="write the certificate as JSON")
    parser.add_argument("--max-iterations", type=int, default=20_000)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--minimum-iterations", type=int, default=100)
    parser.add_argument("--dual-weight-denominator", type=int, default=1 << 40)
    parser.add_argument("--solver", choices=("auto", "glop", "mirror"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    problem = problem_from_dict(json.loads(args.problem.read_text(encoding="utf-8")))
    certificate = build_lower_bound_certificate(
        problem,
        max_iterations=args.max_iterations,
        relative_tolerance=args.relative_tolerance,
        minimum_iterations=args.minimum_iterations,
        dual_weight_denominator=args.dual_weight_denominator,
        solver=args.solver,
    )
    report = {
        "schema_version": 1,
        "model": "moe_resource_lb0_mode_relaxed_lp_v1",
        "problem": {
            "resources": len(problem.resources),
            "stages": len(problem.stages),
            "modes": sum(len(stage.modes) for stage in problem.stages),
            "critical_chains": len(problem.critical_chains),
        },
        "certificate": certificate.to_dict(),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
