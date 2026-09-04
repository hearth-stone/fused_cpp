"""Anchor-relative partial ordering for offline executable-plan search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Iterable, Mapping


CANDIDATE_BETTER = "candidate_better"
CANDIDATE_WORSE = "candidate_worse"
INCOMPARABLE = "incomparable"
ORDER_RELATIONS = (CANDIDATE_BETTER, CANDIDATE_WORSE, INCOMPARABLE)


def _finite(value: float, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _nearest_rank(values: Iterable[float], coverage: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calibrate an empty residual set")
    index = max(0, min(math.ceil(coverage * len(ordered)) - 1, len(ordered) - 1))
    return ordered[index]


@dataclass(frozen=True)
class PairwiseObservation:
    """One measured candidate-versus-anchor ordering observation."""

    family: str
    context: str
    predicted_gain_pct: float
    measured_gain_pct: float

    def __post_init__(self) -> None:
        if not self.family or not self.context:
            raise ValueError("pairwise family and context must be non-empty")
        object.__setattr__(
            self,
            "predicted_gain_pct",
            _finite(self.predicted_gain_pct, "predicted_gain_pct"),
        )
        object.__setattr__(
            self,
            "measured_gain_pct",
            _finite(self.measured_gain_pct, "measured_gain_pct"),
        )

    @property
    def residual_pct(self) -> float:
        return self.measured_gain_pct - self.predicted_gain_pct


@dataclass(frozen=True)
class PairwiseResidualCalibration:
    """Conservative absolute delta-error radii with hierarchical fallback."""

    coverage: float
    minimum_context_samples: int
    global_radius_pct: float
    family_radii_pct: tuple[tuple[str, float], ...] = ()
    context_radii_pct: tuple[tuple[str, float], ...] = ()
    family_sample_counts: tuple[tuple[str, int], ...] = ()
    context_sample_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        if self.minimum_context_samples <= 0:
            raise ValueError("minimum_context_samples must be positive")
        if _finite(self.global_radius_pct, "global_radius_pct") < 0.0:
            raise ValueError("global_radius_pct must be non-negative")
        for name, points in (
            ("family_radii_pct", self.family_radii_pct),
            ("context_radii_pct", self.context_radii_pct),
        ):
            keys = [key for key, _ in points]
            if any(not key for key in keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} keys must be non-empty and unique")
            if any(_finite(radius, name) < 0.0 for _, radius in points):
                raise ValueError(f"{name} values must be non-negative")

    @classmethod
    def fit(
        cls,
        observations: Iterable[PairwiseObservation],
        *,
        coverage: float = 0.90,
        minimum_context_samples: int = 4,
    ) -> "PairwiseResidualCalibration":
        rows = tuple(observations)
        if not rows:
            raise ValueError("pairwise calibration requires at least one observation")
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        if minimum_context_samples <= 0:
            raise ValueError("minimum_context_samples must be positive")

        family_residuals: dict[str, list[float]] = {}
        context_residuals: dict[str, list[float]] = {}
        all_residuals = []
        for row in rows:
            residual = abs(row.residual_pct)
            all_residuals.append(residual)
            family_residuals.setdefault(row.family, []).append(residual)
            context_residuals.setdefault(row.context, []).append(residual)

        family_radii = tuple(
            sorted(
                (family, _nearest_rank(values, coverage))
                for family, values in family_residuals.items()
                if len(values) >= minimum_context_samples
            )
        )
        context_radii = tuple(
            sorted(
                (context, _nearest_rank(values, coverage))
                for context, values in context_residuals.items()
                if len(values) >= minimum_context_samples
            )
        )
        return cls(
            coverage=coverage,
            minimum_context_samples=minimum_context_samples,
            global_radius_pct=_nearest_rank(all_residuals, coverage),
            family_radii_pct=family_radii,
            context_radii_pct=context_radii,
            family_sample_counts=tuple(sorted((key, len(values)) for key, values in family_residuals.items())),
            context_sample_counts=tuple(sorted((key, len(values)) for key, values in context_residuals.items())),
        )

    def radius_pct(self, *, family: str, context: str) -> tuple[float, str]:
        context_radii = dict(self.context_radii_pct)
        if context in context_radii:
            return context_radii[context], "context"
        family_radii = dict(self.family_radii_pct)
        if family in family_radii:
            return family_radii[family], "family"
        return self.global_radius_pct, "global"

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "minimum_context_samples": self.minimum_context_samples,
            "global_radius_pct": self.global_radius_pct,
            "family_radii_pct": dict(self.family_radii_pct),
            "context_radii_pct": dict(self.context_radii_pct),
            "family_sample_counts": dict(self.family_sample_counts),
            "context_sample_counts": dict(self.context_sample_counts),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PairwiseResidualCalibration":
        def float_points(name: str) -> tuple[tuple[str, float], ...]:
            values = payload.get(name, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            return tuple(sorted((str(key), float(value)) for key, value in values.items()))

        def int_points(name: str) -> tuple[tuple[str, int], ...]:
            values = payload.get(name, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"{name} must be a mapping")
            return tuple(sorted((str(key), int(value)) for key, value in values.items()))

        return cls(
            coverage=float(payload["coverage"]),
            minimum_context_samples=int(payload["minimum_context_samples"]),
            global_radius_pct=float(payload["global_radius_pct"]),
            family_radii_pct=float_points("family_radii_pct"),
            context_radii_pct=float_points("context_radii_pct"),
            family_sample_counts=int_points("family_sample_counts"),
            context_sample_counts=int_points("context_sample_counts"),
        )


@dataclass(frozen=True)
class AnchorRelativeOrderEvidence:
    """Calibrated interval and partial-order decision for one candidate."""

    family: str
    context: str
    predicted_gain_pct: float
    residual_radius_pct: float
    residual_scope: str
    lower_gain_pct: float
    upper_gain_pct: float
    minimum_gain_pct: float
    relation: str


class AnchorRelativePartialOrder:
    """Compare candidates only with a named anchor, avoiding cyclic total ranks."""

    def __init__(
        self,
        calibration: PairwiseResidualCalibration,
        *,
        minimum_gain_pct: float = 0.0,
    ) -> None:
        self.calibration = calibration
        self.minimum_gain_pct = _finite(minimum_gain_pct, "minimum_gain_pct")
        if self.minimum_gain_pct < 0.0:
            raise ValueError("minimum_gain_pct must be non-negative")

    def compare(
        self,
        predicted_gain_pct: float,
        *,
        family: str,
        context: str,
    ) -> AnchorRelativeOrderEvidence:
        predicted = _finite(predicted_gain_pct, "predicted_gain_pct")
        radius, scope = self.calibration.radius_pct(family=family, context=context)
        lower = predicted - radius
        upper = predicted + radius
        if lower > self.minimum_gain_pct:
            relation = CANDIDATE_BETTER
        elif upper < -self.minimum_gain_pct:
            relation = CANDIDATE_WORSE
        else:
            relation = INCOMPARABLE
        return AnchorRelativeOrderEvidence(
            family=family,
            context=context,
            predicted_gain_pct=predicted,
            residual_radius_pct=radius,
            residual_scope=scope,
            lower_gain_pct=lower,
            upper_gain_pct=upper,
            minimum_gain_pct=self.minimum_gain_pct,
            relation=relation,
        )


@dataclass(frozen=True)
class PartialOrderCandidate:
    """Plan-visible candidate information used by offline search policy."""

    key: str
    family: str
    context: str
    predicted_gain_pct: float

    def __post_init__(self) -> None:
        if not self.key or not self.family or not self.context:
            raise ValueError("candidate key, family, and context must be non-empty")
        object.__setattr__(
            self,
            "predicted_gain_pct",
            _finite(self.predicted_gain_pct, "predicted_gain_pct"),
        )


@dataclass(frozen=True)
class PartialOrderShortlist:
    """Safe pruning result with budget deferral kept separate from dominance."""

    selected_keys: tuple[str, ...]
    dominated_keys: tuple[str, ...]
    budget_deferred_keys: tuple[str, ...]
    evidence: tuple[tuple[str, AnchorRelativeOrderEvidence], ...]

    @property
    def accepted_key(self) -> str | None:
        evidence = dict(self.evidence)
        for key in self.selected_keys:
            if evidence[key].relation == CANDIDATE_BETTER:
                return key
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_keys": list(self.selected_keys),
            "accepted_key": self.accepted_key,
            "dominated_keys": list(self.dominated_keys),
            "budget_deferred_keys": list(self.budget_deferred_keys),
            "relation_counts": {
                relation: sum(item.relation == relation for _, item in self.evidence)
                for relation in ORDER_RELATIONS
            },
            "evidence": {
                key: {
                    "family": item.family,
                    "context": item.context,
                    "predicted_gain_pct": item.predicted_gain_pct,
                    "residual_radius_pct": item.residual_radius_pct,
                    "residual_scope": item.residual_scope,
                    "lower_gain_pct": item.lower_gain_pct,
                    "upper_gain_pct": item.upper_gain_pct,
                    "minimum_gain_pct": item.minimum_gain_pct,
                    "relation": item.relation,
                }
                for key, item in self.evidence
            },
        }


def neighborhood_context(operator: str, features: Mapping[str, object] | None = None) -> str:
    """Encode the plan-visible context shared by calibration and search."""

    if not operator:
        raise ValueError("operator must be non-empty")
    if not features:
        return operator
    required = (
        "placed_critical_lane_switched",
        "placed_critical_expert_switched",
        "affected_head_direction",
        "affected_tail_direction",
        "cohort_transition_direction",
    )
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"context features are missing: {', '.join(missing)}")
    return "|".join(
        (
            operator,
            f"critical_lane_switch={int(bool(features['placed_critical_lane_switched']))}",
            f"critical_expert_switch={int(bool(features['placed_critical_expert_switched']))}",
            f"head={features['affected_head_direction']}",
            f"tail={features['affected_tail_direction']}",
            f"cohort={features['cohort_transition_direction']}",
        )
    )


def select_partial_order_shortlist(
    comparator: AnchorRelativePartialOrder,
    candidates: Iterable[PartialOrderCandidate],
    *,
    budget: int,
) -> PartialOrderShortlist:
    """Build a diverse shortlist without confusing budget limits with pruning.

    Only confidently worse candidates are dominance-pruned. Incomparable
    candidates beyond ``budget`` are reported as deferred, so a caller cannot
    claim that the partial order proved them worse.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    rows = tuple(candidates)
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate keys must be unique")
    evidence = tuple(
        (
            row.key,
            comparator.compare(
                row.predicted_gain_pct,
                family=row.family,
                context=row.context,
            ),
        )
        for row in rows
    )
    evidence_by_key = dict(evidence)
    retained = [row for row in rows if evidence_by_key[row.key].relation != CANDIDATE_WORSE]

    def priority(row: PartialOrderCandidate) -> tuple[int, float, float, str]:
        item = evidence_by_key[row.key]
        relation_rank = 0 if item.relation == CANDIDATE_BETTER else 1
        return relation_rank, -item.lower_gain_pct, -row.predicted_gain_pct, row.key

    by_context: dict[str, list[PartialOrderCandidate]] = defaultdict(list)
    for row in retained:
        by_context[row.context].append(row)
    for values in by_context.values():
        values.sort(key=priority)
    leaders = sorted((values[0] for values in by_context.values()), key=priority)
    ordered = leaders[:budget]
    selected = {row.key for row in ordered}
    remainder = sorted((row for row in retained if row.key not in selected), key=priority)
    ordered.extend(remainder[: max(budget - len(ordered), 0)])
    selected_keys = tuple(row.key for row in ordered)
    selected = set(selected_keys)
    return PartialOrderShortlist(
        selected_keys=selected_keys,
        dominated_keys=tuple(
            sorted(key for key, item in evidence if item.relation == CANDIDATE_WORSE)
        ),
        budget_deferred_keys=tuple(sorted(row.key for row in retained if row.key not in selected)),
        evidence=tuple(sorted(evidence, key=lambda item: item[0])),
    )


@dataclass(frozen=True)
class PartialOrderSearchIteration:
    """One anchor-relative search decision."""

    anchor_key: str
    shortlist: PartialOrderShortlist


@dataclass(frozen=True)
class PartialOrderSearchResult:
    """Deterministic descent trace that never accepts an unresolved move."""

    initial_anchor_key: str
    incumbent_key: str
    iterations: tuple[PartialOrderSearchIteration, ...]
    stop_reason: str


def run_partial_order_search(
    comparator: AnchorRelativePartialOrder,
    initial_anchor_key: str,
    propose: Callable[[str], Iterable[PartialOrderCandidate]],
    *,
    shortlist_budget: int,
    maximum_iterations: int,
) -> PartialOrderSearchResult:
    """Run an offline best-improvement loop under partial-order acceptance.

    ``propose`` must express every gain relative to the supplied current
    anchor. The loop retains incomparable candidates in each iteration's
    shortlist, but only a confidently better candidate can replace the
    incumbent. Revisited incumbents terminate the loop instead of forming a
    cycle.
    """

    if not initial_anchor_key:
        raise ValueError("initial_anchor_key must be non-empty")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    incumbent = initial_anchor_key
    visited = {incumbent}
    iterations = []
    stop_reason = "iteration_limit"
    for _ in range(maximum_iterations):
        shortlist = select_partial_order_shortlist(
            comparator,
            propose(incumbent),
            budget=shortlist_budget,
        )
        iterations.append(PartialOrderSearchIteration(incumbent, shortlist))
        accepted = shortlist.accepted_key
        if accepted is None:
            stop_reason = "no_confident_improvement"
            break
        if accepted in visited:
            stop_reason = "revisited_incumbent"
            break
        incumbent = accepted
        visited.add(incumbent)
    return PartialOrderSearchResult(
        initial_anchor_key=initial_anchor_key,
        incumbent_key=incumbent,
        iterations=tuple(iterations),
        stop_reason=stop_reason,
    )


def comparator_from_validated_report(
    payload: Mapping[str, object],
    *,
    shortlist_budget: int,
    minimum_gain_pct: float,
) -> AnchorRelativePartialOrder:
    """Load a comparator only after its replay safety gates have passed."""

    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("pairwise report has no replay gates")
    if gates.get("zero_false_pruning") is not True:
        raise ValueError("pairwise report failed the zero-false-pruning gate")
    recall = gates.get("measured_best_retained")
    if not isinstance(recall, Mapping) or recall.get(str(shortlist_budget)) is not True:
        raise ValueError(
            f"pairwise report did not pass measured-best recall at top-{shortlist_budget}"
        )
    calibration = payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("pairwise report has no calibration object")
    return AnchorRelativePartialOrder(
        PairwiseResidualCalibration.from_dict(calibration),
        minimum_gain_pct=minimum_gain_pct,
    )


__all__ = [
    "CANDIDATE_BETTER",
    "CANDIDATE_WORSE",
    "INCOMPARABLE",
    "ORDER_RELATIONS",
    "AnchorRelativeOrderEvidence",
    "AnchorRelativePartialOrder",
    "PartialOrderCandidate",
    "PartialOrderSearchIteration",
    "PartialOrderSearchResult",
    "PartialOrderShortlist",
    "PairwiseObservation",
    "PairwiseResidualCalibration",
    "comparator_from_validated_report",
    "neighborhood_context",
    "run_partial_order_search",
    "select_partial_order_shortlist",
]
