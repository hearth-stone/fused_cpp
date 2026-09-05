"""Relation-agnostic diverse shortlist for template-level LNS hardware budgets.

The selector never reads partial-order relations, residual radii, hardware
timings, or previous winner status. Quantiles are diversity labels only.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_PLANNER_DIR = Path(__file__).resolve().parents[3] / "cpu_moe_schedule_optimization" / "planners"
if str(_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(_PLANNER_DIR))

from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)


POLICY_NAME = "relation_agnostic_categorical_farthest_first_v1"
RANKING_FILL_NAME = "incremental_min_distance_v1"
RANKING_FILL_REFERENCE_NAME = "naive_selected_scan_v1"
SCHEMA_VERSION = 1
DEFAULT_SHORTLIST_BUDGET = 16
DEFAULT_AUDIT_BUDGET = 32
SCORE_QUANTILE_COUNT = 5
CLOSURE_BINS = ("1_15", "16_31", "32_63", "64_127", "128_plus")
QUANTILE_FILL_ORDER = (0, 4, 1, 2, 3)
CATEGORY_NAMES = (
    "operator",
    "target_destroy_size",
    "actual_closure_bin",
    "width_histogram",
    "domain_assignment_signature",
    "score_quantile",
)
_OPERATOR_RE = re.compile(
    r"^critical_window_template_repartition_(domain_local|cross_domain)_d(\d+)_b(\d+)$"
)
_RESTART_RE = re.compile(r"_r(\d+)$")


def parse_restart(start_name: str) -> int:
    match = _RESTART_RE.search(str(start_name))
    return int(match.group(1)) if match else 0


def parse_operator(operator: str) -> tuple[str, int, int]:
    match = _OPERATOR_RE.fullmatch(str(operator))
    if match is None:
        raise ValueError(f"unsupported template-LNS operator: {operator}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def closure_bin(size: int) -> str:
    count = int(size)
    if count <= 0:
        raise ValueError("actual_closure_size must be positive")
    if count <= 15:
        return "1_15"
    if count <= 31:
        return "16_31"
    if count <= 63:
        return "32_63"
    if count <= 127:
        return "64_127"
    return "128_plus"


def width_histogram(state: ExecutablePlanState) -> tuple[tuple[int, int], ...]:
    counts = Counter(int(lane.threads) for lane in state.lanes)
    return tuple(sorted((width, int(count)) for width, count in counts.items()))


def width_histogram_from_payload(payload: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    lanes = payload.get("lanes")
    if not isinstance(lanes, Sequence) or isinstance(lanes, (str, bytes)):
        return ((1, 1),)
    counts = Counter(int(lane["threads"]) for lane in lanes)
    return tuple(sorted((width, int(count)) for width, count in counts.items()))


def histogram_delta(
    candidate: Sequence[tuple[int, int]],
    anchor: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    left = Counter(dict(candidate))
    right = Counter(dict(anchor))
    return tuple(
        (width, int(left[width] - right[width]))
        for width in sorted(set(left) | set(right))
        if left[width] - right[width] != 0
    )


def domain_assignment_payload(state: ExecutablePlanState) -> dict[str, Any]:
    contained = {domain.domain_id: [] for domain in state.llc_domains}
    crossing: list[list[Any]] = []
    for lane_index, lane in enumerate(state.lanes):
        domain_ids = list(state.lane_domain_ids(lane_index))
        if len(domain_ids) == 1:
            contained[domain_ids[0]].append(int(lane.threads))
        else:
            crossing.append([int(lane.core_begin), int(lane.threads), domain_ids])
    return {"contained": contained, "crossing": crossing}


def domain_assignment_payload_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_domains = payload.get("llc_domains") or ()
    raw_lanes = payload.get("lanes") or ()
    domains = [
        (str(domain["id"]), int(domain["core_begin"]), int(domain["core_count"]))
        for domain in raw_domains
    ]
    contained = {domain_id: [] for domain_id, _, _ in domains}
    crossing: list[list[Any]] = []
    for lane in raw_lanes:
        core_begin = int(lane["core_begin"])
        core_end = core_begin + int(lane["threads"])
        domain_ids = [
            domain_id
            for domain_id, domain_begin, domain_count in domains
            if core_begin < domain_begin + domain_count and domain_begin < core_end
        ]
        if len(domain_ids) == 1:
            contained[domain_ids[0]].append(int(lane["threads"]))
        else:
            crossing.append([core_begin, int(lane["threads"]), domain_ids])
    return {"contained": contained, "crossing": crossing}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def domain_assignment_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def core_span_for_experts(state: ExecutablePlanState, expert_ids: Sequence[int]) -> tuple[int, int]:
    wanted = {int(expert) for expert in expert_ids}
    begins: list[int] = []
    ends: list[int] = []
    for lane in state.lanes:
        if any(task.expert_id in wanted for task in lane.tasks):
            begins.append(int(lane.core_begin))
            ends.append(int(lane.core_end))
    if not begins:
        return 0, 0
    return min(begins), max(ends)


def core_span_from_payload(payload: Mapping[str, Any], expert_ids: Sequence[int]) -> tuple[int, int]:
    wanted = {int(expert) for expert in expert_ids}
    begins: list[int] = []
    ends: list[int] = []
    for lane in payload.get("lanes") or ():
        if any(int(task["expert_id"]) in wanted for task in lane.get("tasks") or ()):
            core_begin = int(lane["core_begin"])
            begins.append(core_begin)
            ends.append(core_begin + int(lane["threads"]))
    if not begins:
        return 0, 0
    return min(begins), max(ends)


def state_from_canonical_payload(payload: Mapping[str, Any]) -> ExecutablePlanState:
    return ExecutablePlanState(
        num_threads=int(payload["num_threads"]),
        thread_cpu_ids=tuple(int(cpu) for cpu in payload["thread_cpu_ids"]),
        llc_domains=tuple(
            ExecutableLlcDomain(
                str(domain["id"]),
                int(domain["core_begin"]),
                int(domain["core_count"]),
            )
            for domain in payload.get("llc_domains") or ()
        ),
        lanes=tuple(
            ExecutableLane(
                int(lane["core_begin"]),
                int(lane["threads"]),
                tuple(
                    ExecutableExpertTask(
                        int(task["expert_id"]),
                        int(task["routes"]),
                        int(task.get("w13_window_tiles", 0)),
                        int(task.get("w2_window_tiles", 0)),
                    )
                    for task in lane.get("tasks") or ()
                ),
            )
            for lane in payload["lanes"]
        ),
        early_merge=payload.get("early_merge"),
    )


def policy_document(*, source_commit: str | None = None) -> dict[str, Any]:
    document = {
        "audit_budget": DEFAULT_AUDIT_BUDGET,
        "categories": list(CATEGORY_NAMES),
        "closure_bins": list(CLOSURE_BINS),
        "hardware_weights": None,
        "operator_order": "lexicographic_name",
        "policy": POLICY_NAME,
        "quantile_fill_order": list(QUANTILE_FILL_ORDER),
        "relation_agnostic": True,
        "schema_version": SCHEMA_VERSION,
        "score_quantile_count": SCORE_QUANTILE_COUNT,
        "shortlist_budget": DEFAULT_SHORTLIST_BUDGET,
        "tie_break": ["token_reuse_ascending", "canonical_state_hash"],
    }
    if source_commit is not None:
        document["source_commit"] = str(source_commit)
    return document


def policy_sha256(document: Mapping[str, Any] | None = None) -> str:
    payload = dict(document or policy_document())
    encoded = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LnsCandidateFeature:
    """Compact, plan-visible candidate record used by the shortlist selector."""

    state_hash: str
    anchor_state_hash: str
    start: str
    restart: int
    strategy: str
    operator: str
    scope: str
    target_destroy_size: int
    actual_closure_size: int
    actual_closure_bin: str
    changed_core_begin: int
    changed_core_end: int
    candidate_width_histogram: tuple[tuple[int, int], ...]
    width_histogram_delta: tuple[tuple[int, int], ...]
    domain_assignment: str
    domain_assignment_signature: str
    anchor_domain_assignment_signature: str
    cross_domain_lane_count: int
    predicted_gain_pct: float
    model_score_quantile: int | None = None
    provenance: tuple[tuple[str, str, str], ...] = ()
    schema_version: int = SCHEMA_VERSION

    def categories(self) -> tuple[Any, ...]:
        if self.model_score_quantile is None:
            raise ValueError(f"score quantile is unassigned for {self.state_hash}")
        return (
            self.operator,
            self.target_destroy_size,
            self.actual_closure_bin,
            self.candidate_width_histogram,
            self.domain_assignment_signature,
            self.model_score_quantile,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "actual_closure_bin": self.actual_closure_bin,
            "actual_closure_size": self.actual_closure_size,
            "anchor_domain_assignment_signature": self.anchor_domain_assignment_signature,
            "anchor_state_hash": self.anchor_state_hash,
            "candidate_width_histogram": [list(item) for item in self.candidate_width_histogram],
            "changed_core_begin": self.changed_core_begin,
            "changed_core_end": self.changed_core_end,
            "cross_domain_lane_count": self.cross_domain_lane_count,
            "domain_assignment": json.loads(self.domain_assignment),
            "domain_assignment_signature": self.domain_assignment_signature,
            "model_score_quantile": self.model_score_quantile,
            "operator": self.operator,
            "predicted_gain_pct": self.predicted_gain_pct,
            "restart": self.restart,
            "schema_version": self.schema_version,
            "scope": self.scope,
            "start": self.start,
            "state_hash": self.state_hash,
            "strategy": self.strategy,
            "target_destroy_size": self.target_destroy_size,
            "width_histogram_delta": [list(item) for item in self.width_histogram_delta],
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LnsCandidateFeature":
        histogram = tuple(tuple(int(part) for part in item) for item in payload["candidate_width_histogram"])
        delta = tuple(tuple(int(part) for part in item) for item in payload["width_histogram_delta"])
        domain = payload["domain_assignment"]
        domain_json = domain if isinstance(domain, str) else canonical_json(domain)
        quantile = payload.get("model_score_quantile")
        return cls(
            state_hash=str(payload["state_hash"]),
            anchor_state_hash=str(payload["anchor_state_hash"]),
            start=str(payload["start"]),
            restart=int(payload["restart"]),
            strategy=str(payload["strategy"]),
            operator=str(payload["operator"]),
            scope=str(payload["scope"]),
            target_destroy_size=int(payload["target_destroy_size"]),
            actual_closure_size=int(payload["actual_closure_size"]),
            actual_closure_bin=str(payload["actual_closure_bin"]),
            changed_core_begin=int(payload["changed_core_begin"]),
            changed_core_end=int(payload["changed_core_end"]),
            candidate_width_histogram=histogram,
            width_histogram_delta=delta,
            domain_assignment=domain_json,
            domain_assignment_signature=str(payload["domain_assignment_signature"]),
            anchor_domain_assignment_signature=str(
                payload.get("anchor_domain_assignment_signature", "")
            ),
            cross_domain_lane_count=int(payload["cross_domain_lane_count"]),
            predicted_gain_pct=float(payload["predicted_gain_pct"]),
            model_score_quantile=None if quantile is None else int(quantile),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )


class AnchorRecoveryError(ValueError):
    """A frozen feature cannot be bound to its iteration anchor."""


def iteration_anchor_from_run(
    run: Mapping[str, Any],
    *,
    iteration_index: int = 0,
) -> tuple[str, Mapping[str, Any]]:
    """Return the bound iteration anchor hash and canonical payload.

    Single-iteration LNS uses ``initial_canonical_state``. Later iterations must
    supply that iteration's actual incumbent; this helper never assumes the
    initial state once ``iteration_index`` is positive.
    """

    if iteration_index < 0:
        raise AnchorRecoveryError("iteration_index must be non-negative")
    payload: Mapping[str, Any] | None
    stated: object
    if iteration_index == 0:
        payload = run.get("initial_canonical_state")
        stated = run.get("initial_state_hash")
    else:
        iterations = run.get("iterations") or []
        if iteration_index > len(iterations):
            raise AnchorRecoveryError(f"iteration {iteration_index} is missing")
        previous = iterations[iteration_index - 1]
        accepted = previous.get("accepted_move") if isinstance(previous, Mapping) else None
        payload = None
        if isinstance(previous, Mapping):
            payload = previous.get("accepted_canonical_state") or previous.get("canonical_state")
        stated = None
        if isinstance(accepted, Mapping):
            stated = accepted.get("state_hash")
        if payload is None:
            raise AnchorRecoveryError(
                f"iteration {iteration_index} has no bound canonical incumbent to recover from"
            )
    if not isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        raise AnchorRecoveryError("missing iteration canonical state for anchor recovery")
    restored = state_from_canonical_payload(payload)
    hashed = restored.canonical_hash()
    if stated is not None and str(stated) != hashed:
        raise AnchorRecoveryError(
            f"stated iteration hash {stated} does not match canonical hash {hashed}"
        )
    return hashed, payload


def recover_anchor_domain_signature(
    feature: Mapping[str, Any] | LnsCandidateFeature,
    *,
    anchor_canonical_state: Mapping[str, Any],
    anchor_canonical_hash: str,
) -> LnsCandidateFeature:
    """Restore omitted ``anchor_domain_assignment_signature`` from the bound anchor.

    ``anchor_state_hash`` must equal the saved iteration canonical hash. A
    serialized signature that disagrees with the recovered value is an error.
    """

    row = feature.to_dict() if isinstance(feature, LnsCandidateFeature) else dict(feature)
    if "state_hash" not in row or "anchor_state_hash" not in row:
        raise AnchorRecoveryError("feature is missing state_hash or anchor_state_hash")
    stored_anchor = str(row["anchor_state_hash"])
    if stored_anchor != str(anchor_canonical_hash):
        raise AnchorRecoveryError(
            f"anchor_state_hash {stored_anchor} does not match iteration canonical hash "
            f"{anchor_canonical_hash} for {row['state_hash']}"
        )
    recovered = domain_assignment_signature(
        domain_assignment_payload_from_mapping(anchor_canonical_state)
    )
    stored_sig = row.get("anchor_domain_assignment_signature")
    if stored_sig not in (None, "", recovered):
        raise AnchorRecoveryError(
            f"serialized anchor_domain_assignment_signature disagrees with recovered "
            f"value for {row['state_hash']}"
        )
    return LnsCandidateFeature.from_dict({**row, "anchor_domain_assignment_signature": recovered})


def recover_run_features(
    run: Mapping[str, Any],
    *,
    iteration_index: int = 0,
) -> tuple[list[LnsCandidateFeature], dict[str, Any]]:
    """Recover ranking inputs for one start and record provenance."""

    hashed, payload = iteration_anchor_from_run(run, iteration_index=iteration_index)
    iterations = run.get("iterations") or []
    if iteration_index >= len(iterations):
        raise AnchorRecoveryError(f"iteration {iteration_index} is missing")
    iteration = iterations[iteration_index]
    rows = iteration.get("candidate_features") or []
    recovered = [
        recover_anchor_domain_signature(
            row,
            anchor_canonical_state=payload,
            anchor_canonical_hash=hashed,
        )
        for row in rows
    ]
    provenance = {
        "anchor_canonical_hash": hashed,
        "iteration_index": int(iteration_index),
        "recovered_fields": ["anchor_domain_assignment_signature"],
        "recovery_source": "iteration_canonical_state",
        "row_count": len(recovered),
    }
    return recovered, provenance


@dataclass(frozen=True)
class LnsDiverseShortlist:
    """Prefix-nested per-start ranking plus the declared nested budgets."""

    ranked_keys: tuple[str, ...]
    selected_keys: tuple[str, ...]
    audit_keys: tuple[str, ...]
    budget_deferred_keys: tuple[str, ...]
    coverage: dict[str, Any]
    shortlist_budget: int
    audit_budget: int
    policy: str = POLICY_NAME
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_budget": self.audit_budget,
            "audit_keys": list(self.audit_keys),
            "budget_deferred_keys": list(self.budget_deferred_keys),
            "coverage": self.coverage,
            "policy": self.policy,
            "ranked_keys": list(self.ranked_keys),
            "schema_version": self.schema_version,
            "selected_keys": list(self.selected_keys),
            "shortlist_budget": self.shortlist_budget,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LnsDiverseShortlist":
        ranked = tuple(str(key) for key in payload.get("ranked_keys", payload.get("audit_keys", ())))
        selected = tuple(str(key) for key in payload["selected_keys"])
        audit = tuple(str(key) for key in payload["audit_keys"])
        if selected != audit[: len(selected)]:
            raise ValueError("selected_keys must be an ordered prefix of audit_keys")
        return cls(
            ranked_keys=ranked,
            selected_keys=selected,
            audit_keys=audit,
            budget_deferred_keys=tuple(str(key) for key in payload.get("budget_deferred_keys", ())),
            coverage=dict(payload.get("coverage", {})),
            shortlist_budget=int(payload["shortlist_budget"]),
            audit_budget=int(payload["audit_budget"]),
            policy=str(payload.get("policy", POLICY_NAME)),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )


def compact_structure(state: Any) -> dict[str, Any]:
    """Extract width/domain structure before a candidate state leaves memory."""

    payload = state.canonical_payload() if hasattr(state, "canonical_payload") else dict(state)
    if isinstance(payload, Mapping) and payload.get("lanes"):
        assignment = (
            domain_assignment_payload(state)
            if isinstance(state, ExecutablePlanState)
            else domain_assignment_payload_from_mapping(payload)
        )
        histogram = (
            width_histogram(state)
            if isinstance(state, ExecutablePlanState)
            else width_histogram_from_payload(payload)
        )
        return {
            "candidate_width_histogram": histogram,
            "cross_domain_lane_count": len(assignment["crossing"]),
            "domain_assignment": assignment,
            "domain_assignment_signature": domain_assignment_signature(assignment),
        }
    state_hash = state.canonical_hash() if hasattr(state, "canonical_hash") else str(payload)
    return {
        "candidate_width_histogram": ((1, 1),),
        "cross_domain_lane_count": 0,
        "domain_assignment": {"contained": {}, "crossing": []},
        "domain_assignment_signature": hashlib.sha256(str(state_hash).encode("utf-8")).hexdigest(),
    }


def build_candidate_feature(
    candidate: Any,
    anchor: Any,
    *,
    operator: str,
    moved_experts: Sequence[int],
    predicted_gain_pct: float,
    start: str,
    strategy: str,
    restart: int | None = None,
    state_hash: str | None = None,
    anchor_state_hash: str | None = None,
) -> LnsCandidateFeature:
    scope, destroy_size, _beam = parse_operator(operator)
    closure_size = len(tuple(int(expert) for expert in moved_experts))
    candidate_hash = state_hash or candidate.canonical_hash()
    anchor_hash = anchor_state_hash or anchor.canonical_hash()
    candidate_structure = compact_structure(candidate)
    anchor_structure = compact_structure(anchor)
    candidate_payload = candidate.canonical_payload() if hasattr(candidate, "canonical_payload") else {}
    anchor_payload = anchor.canonical_payload() if hasattr(anchor, "canonical_payload") else {}
    if isinstance(candidate, ExecutablePlanState) and isinstance(anchor, ExecutablePlanState):
        candidate_span = core_span_for_experts(candidate, moved_experts)
        anchor_span = core_span_for_experts(anchor, moved_experts)
    else:
        candidate_span = core_span_from_payload(candidate_payload, moved_experts)
        anchor_span = core_span_from_payload(anchor_payload, moved_experts)
    spans = [span for span in (candidate_span, anchor_span) if span != (0, 0)]
    changed_begin = min(span[0] for span in spans) if spans else 0
    changed_end = max(span[1] for span in spans) if spans else 0
    resolved_restart = parse_restart(start) if restart is None else int(restart)
    assignment_json = canonical_json(candidate_structure["domain_assignment"])
    return LnsCandidateFeature(
        state_hash=str(candidate_hash),
        anchor_state_hash=str(anchor_hash),
        start=str(start),
        restart=resolved_restart,
        strategy=str(strategy),
        operator=str(operator),
        scope=scope,
        target_destroy_size=destroy_size,
        actual_closure_size=closure_size,
        actual_closure_bin=closure_bin(closure_size),
        changed_core_begin=int(changed_begin),
        changed_core_end=int(changed_end),
        candidate_width_histogram=tuple(candidate_structure["candidate_width_histogram"]),
        width_histogram_delta=histogram_delta(
            candidate_structure["candidate_width_histogram"],
            anchor_structure["candidate_width_histogram"],
        ),
        domain_assignment=assignment_json,
        domain_assignment_signature=str(candidate_structure["domain_assignment_signature"]),
        anchor_domain_assignment_signature=str(anchor_structure["domain_assignment_signature"]),
        cross_domain_lane_count=int(candidate_structure["cross_domain_lane_count"]),
        predicted_gain_pct=float(predicted_gain_pct),
        provenance=((str(start), str(strategy), str(operator)),),
    )


def merge_duplicate_features(features: Iterable[LnsCandidateFeature]) -> list[LnsCandidateFeature]:
    merged: dict[str, LnsCandidateFeature] = {}
    for feature in features:
        existing = merged.get(feature.state_hash)
        if existing is None:
            merged[feature.state_hash] = feature
            continue
        if (
            existing.candidate_width_histogram != feature.candidate_width_histogram
            or existing.domain_assignment_signature != feature.domain_assignment_signature
        ):
            raise ValueError(f"canonical hash collision for {feature.state_hash}")
        provenances = tuple(sorted(set(existing.provenance + feature.provenance)))
        primary = min(
            (existing, feature),
            key=lambda item: (item.operator, item.strategy, item.start, item.state_hash),
        )
        merged[feature.state_hash] = replace(primary, provenance=provenances)
    return [merged[key] for key in sorted(merged)]


def assign_model_score_quantiles(features: Sequence[LnsCandidateFeature]) -> list[LnsCandidateFeature]:
    if not features:
        return []
    ordered = sorted(features, key=lambda item: (item.predicted_gain_pct, item.state_hash))
    count = len(ordered)
    assigned = []
    for index, feature in enumerate(ordered):
        quantile = min(SCORE_QUANTILE_COUNT - 1, (index * SCORE_QUANTILE_COUNT) // count)
        assigned.append(replace(feature, model_score_quantile=int(quantile)))
    by_hash = {feature.state_hash: feature for feature in assigned}
    return [by_hash[key] for key in sorted(by_hash)]


def categorical_distance(left: LnsCandidateFeature, right: LnsCandidateFeature) -> int:
    return sum(first != second for first, second in zip(left.categories(), right.categories(), strict=True))


def distance_from_anchor(feature: LnsCandidateFeature) -> int:
    distance = 4
    if feature.width_histogram_delta:
        distance += 1
    if feature.domain_assignment_signature != feature.anchor_domain_assignment_signature:
        distance += 1
    return distance


def token_reuse(feature: LnsCandidateFeature, selected: Sequence[LnsCandidateFeature]) -> int:
    tokens = feature.categories()
    reuse = 0
    for item in selected:
        reuse += sum(left == right for left, right in zip(tokens, item.categories(), strict=True))
    return reuse


def _coverage(features: Sequence[LnsCandidateFeature]) -> dict[str, Any]:
    return {
        "actual_closure_bins": sorted({item.actual_closure_bin for item in features}),
        "domain_assignment_signatures": len({item.domain_assignment_signature for item in features}),
        "operators": sorted({item.operator for item in features}),
        "scopes": sorted({item.scope for item in features}),
        "score_quantiles": sorted({item.model_score_quantile for item in features}),
        "target_destroy_sizes": sorted({item.target_destroy_size for item in features}),
        "width_histograms": len({item.candidate_width_histogram for item in features}),
    }


def _novelty_key(
    feature: LnsCandidateFeature,
    selected: Sequence[LnsCandidateFeature],
) -> tuple[int, int, int, int, int, str]:
    quantiles = {item.model_score_quantile for item in selected}
    closures = {item.actual_closure_bin for item in selected}
    domains = {item.domain_assignment_signature for item in selected}
    widths = {item.candidate_width_histogram for item in selected}
    return (
        0 if feature.model_score_quantile not in quantiles else 1,
        0 if feature.actual_closure_bin not in closures else 1,
        0 if feature.domain_assignment_signature not in domains else 1,
        0 if feature.candidate_width_histogram not in widths else 1,
        -distance_from_anchor(feature),
        feature.state_hash,
    )


def _category_hamming(left: Sequence[Any], right: Sequence[Any]) -> int:
    return sum(first != second for first, second in zip(left, right, strict=True))


def _seed_operator_quantile_coverage(
    unique: Sequence[LnsCandidateFeature],
) -> tuple[dict[str, LnsCandidateFeature], list[LnsCandidateFeature]]:
    remaining = {item.state_hash: item for item in unique}
    selected: list[LnsCandidateFeature] = []

    def take(feature: LnsCandidateFeature) -> None:
        selected.append(feature)
        remaining.pop(feature.state_hash, None)

    for operator in sorted({item.operator for item in unique}):
        pool = [item for item in remaining.values() if item.operator == operator]
        if not pool:
            continue
        take(min(pool, key=lambda item: _novelty_key(item, selected)))

    represented = {item.model_score_quantile for item in selected}
    for quantile in QUANTILE_FILL_ORDER:
        if quantile in represented:
            continue
        pool = [item for item in remaining.values() if item.model_score_quantile == quantile]
        if not pool:
            continue
        take(min(pool, key=lambda item: _novelty_key(item, selected)))
        represented.add(quantile)
    return remaining, selected


def _farthest_first_fill_reference(
    unique: Sequence[LnsCandidateFeature],
) -> list[LnsCandidateFeature]:
    """Naive selected-set scan. Test reference only; ranking order is the contract."""

    remaining, selected = _seed_operator_quantile_coverage(unique)
    while remaining:
        chosen = min(
            remaining.values(),
            key=lambda item: (
                -min(categorical_distance(item, chosen) for chosen in selected) if selected else 0,
                token_reuse(item, selected),
                item.state_hash,
            ),
        )
        selected.append(chosen)
        remaining.pop(chosen.state_hash, None)
    return selected


def _farthest_first_fill(unique: Sequence[LnsCandidateFeature]) -> list[LnsCandidateFeature]:
    """Incremental min-distance / token-reuse fill. Exact versus the reference."""

    remaining, selected = _seed_operator_quantile_coverage(unique)
    if not remaining:
        return selected
    categories = {item.state_hash: item.categories() for item in unique}
    n_dims = len(next(iter(categories.values())))
    min_dist: dict[str, int] = {}
    reuse: dict[str, int] = {}
    selected_tokens = [categories[item.state_hash] for item in selected]
    for item in remaining.values():
        tokens = categories[item.state_hash]
        if selected_tokens:
            distances = [_category_hamming(tokens, chosen) for chosen in selected_tokens]
            min_dist[item.state_hash] = min(distances)
            reuse[item.state_hash] = sum(n_dims - distance for distance in distances)
        else:
            min_dist[item.state_hash] = 0
            reuse[item.state_hash] = 0
    initialized = bool(selected_tokens)
    while remaining:
        chosen = min(
            remaining.values(),
            key=lambda item: (
                -min_dist[item.state_hash],
                reuse[item.state_hash],
                item.state_hash,
            ),
        )
        selected.append(chosen)
        remaining.pop(chosen.state_hash, None)
        min_dist.pop(chosen.state_hash, None)
        reuse.pop(chosen.state_hash, None)
        chosen_tokens = categories[chosen.state_hash]
        for item in remaining.values():
            state_hash = item.state_hash
            distance = _category_hamming(categories[state_hash], chosen_tokens)
            matches = n_dims - distance
            if initialized:
                if distance < min_dist[state_hash]:
                    min_dist[state_hash] = distance
                reuse[state_hash] += matches
            else:
                min_dist[state_hash] = distance
                reuse[state_hash] = matches
        initialized = True
    return selected


def rank_lns_diverse_candidates(
    features: Sequence[LnsCandidateFeature],
    *,
    timing: dict[str, float] | None = None,
    fill: str = RANKING_FILL_NAME,
) -> list[LnsCandidateFeature]:
    prepare_begin = time.perf_counter_ns()
    unique = assign_model_score_quantiles(merge_duplicate_features(features))
    prepare_s = (time.perf_counter_ns() - prepare_begin) / 1.0e9
    if fill == RANKING_FILL_REFERENCE_NAME:
        fill_fn = _farthest_first_fill_reference
    elif fill == RANKING_FILL_NAME:
        fill_fn = _farthest_first_fill
    else:
        raise ValueError(f"unknown ranking fill: {fill}")
    fill_begin = time.perf_counter_ns()
    ranked = fill_fn(unique)
    fill_s = (time.perf_counter_ns() - fill_begin) / 1.0e9
    if timing is not None:
        timing["quantile_dedup_s"] = prepare_s
        timing["per_start_diverse_ranking_s"] = fill_s
    return ranked


def _shortlist_from_ranked(
    ranked: Sequence[LnsCandidateFeature],
    *,
    shortlist_budget: int,
    audit_budget: int,
) -> LnsDiverseShortlist:
    ranked_keys = tuple(item.state_hash for item in ranked)
    selected_keys = ranked_keys[:shortlist_budget]
    audit_keys = ranked_keys[:audit_budget]
    selected_features = [item for item in ranked if item.state_hash in set(selected_keys)]
    return LnsDiverseShortlist(
        ranked_keys=ranked_keys,
        selected_keys=selected_keys,
        audit_keys=audit_keys,
        budget_deferred_keys=ranked_keys[shortlist_budget:],
        coverage=_coverage(selected_features),
        shortlist_budget=int(shortlist_budget),
        audit_budget=int(audit_budget),
    )


def select_lns_diverse_shortlist(
    features: Sequence[LnsCandidateFeature],
    *,
    shortlist_budget: int = DEFAULT_SHORTLIST_BUDGET,
    audit_budget: int = DEFAULT_AUDIT_BUDGET,
    timing: dict[str, float] | None = None,
    fill: str = RANKING_FILL_NAME,
) -> LnsDiverseShortlist:
    if shortlist_budget <= 0 or audit_budget < shortlist_budget:
        raise ValueError("audit budget must be at least the shortlist budget, and both must be positive")
    ranked = rank_lns_diverse_candidates(features, timing=timing, fill=fill)
    return _shortlist_from_ranked(
        ranked,
        shortlist_budget=shortlist_budget,
        audit_budget=audit_budget,
    )


def _fill_unique(
    per_start_rankings: Mapping[str, Sequence[str]],
    *,
    budget: int,
    start_order: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    selected: list[str] = []
    selected_set: set[str] = set()
    counts = {
        start: {"requested": int(budget), "selected": 0, "duplicate": 0, "exhausted": False}
        for start in start_order
    }
    for start in start_order:
        taken = 0
        for key in per_start_rankings[start]:
            if taken >= budget:
                break
            if key in selected_set:
                counts[start]["duplicate"] += 1
                continue
            selected.append(key)
            selected_set.add(key)
            taken += 1
            counts[start]["selected"] += 1
    for start in start_order:
        if counts[start]["selected"] >= budget:
            continue
        for key in per_start_rankings[start]:
            if key in selected_set:
                continue
            selected.append(key)
            selected_set.add(key)
            counts[start]["selected"] += 1
            if counts[start]["selected"] >= budget:
                break
        if counts[start]["selected"] < budget:
            counts[start]["exhausted"] = True
    return tuple(selected), counts


def select_lns_global_shortlist(
    per_start: Mapping[str, LnsDiverseShortlist],
    *,
    shortlist_budget: int = DEFAULT_SHORTLIST_BUDGET,
    audit_budget: int = DEFAULT_AUDIT_BUDGET,
) -> dict[str, Any]:
    start_order = tuple(sorted(per_start))
    rankings = {start: shortlist.ranked_keys for start, shortlist in per_start.items()}
    selected_keys, selected_counts = _fill_unique(
        rankings,
        budget=shortlist_budget,
        start_order=start_order,
    )
    audit_keys, audit_counts = _fill_unique(
        rankings,
        budget=audit_budget,
        start_order=start_order,
    )
    return {
        "audit_budget": int(audit_budget),
        "audit_duplicate_counts": audit_counts,
        "audit_keys": list(audit_keys),
        "policy": POLICY_NAME,
        "schema_version": SCHEMA_VERSION,
        "selected_duplicate_counts": selected_counts,
        "selected_keys": list(selected_keys),
        "shortlist_budget": int(shortlist_budget),
        "starts": list(start_order),
    }


def select_lns_parent_pooled_shortlists(
    per_start_features: Mapping[str, Sequence[LnsCandidateFeature | Mapping[str, Any]]],
    parent_of_start: Mapping[str, str],
    *,
    shortlist_budget: int = DEFAULT_SHORTLIST_BUDGET,
    audit_budget: int = DEFAULT_AUDIT_BUDGET,
    fill: str = RANKING_FILL_NAME,
) -> dict[str, LnsDiverseShortlist]:
    """Run selector v1 once per unique parent over pooled restart features."""

    grouped: dict[str, list[LnsCandidateFeature]] = {}
    for start, rows in per_start_features.items():
        parent = parent_of_start[start]
        for row in rows:
            feature = row if isinstance(row, LnsCandidateFeature) else LnsCandidateFeature.from_dict(row)
            grouped.setdefault(parent, []).append(feature)
    return {
        parent: select_lns_diverse_shortlist(
            features,
            shortlist_budget=shortlist_budget,
            audit_budget=audit_budget,
            fill=fill,
        )
        for parent, features in grouped.items()
    }


def stratified_keys_outside_audit(
    features: Sequence[LnsCandidateFeature | Mapping[str, Any]],
    ranked_keys: Sequence[str],
    audit_keys: Sequence[str],
    *,
    sample_size: int,
    seed: int,
) -> tuple[str, ...]:
    """Deterministic round-robin sample from ranked keys beyond the audit prefix."""

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size == 0:
        return ()
    parsed = [
        row if isinstance(row, LnsCandidateFeature) else LnsCandidateFeature.from_dict(row)
        for row in features
    ]
    unique = assign_model_score_quantiles(merge_duplicate_features(parsed))
    by_hash = {item.state_hash: item for item in unique}
    audit = {str(key) for key in audit_keys}
    outside = [str(key) for key in ranked_keys if str(key) not in audit and str(key) in by_hash]
    buckets: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for key in outside:
        feature = by_hash[key]
        quantile = 0 if feature.model_score_quantile is None else int(feature.model_score_quantile)
        buckets[(feature.operator, feature.actual_closure_bin, quantile)].append(key)
    rng = random.Random(int(seed))
    for bucket_key in buckets:
        items = list(buckets[bucket_key])
        rng.shuffle(items)
        buckets[bucket_key] = items
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < sample_size and any(buckets.values()):
        for bucket_key in sorted(buckets):
            if not buckets[bucket_key]:
                continue
            key = buckets[bucket_key].pop(0)
            if key in selected_set:
                continue
            selected.append(key)
            selected_set.add(key)
            if len(selected) == sample_size:
                break
    return tuple(selected)


def features_from_hardware_frontier(
    frontier: Mapping[str, Any],
    *,
    plans: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[LnsCandidateFeature]]:
    if frontier.get("kind") != "partial_order_hardware_frontier":
        raise ValueError("frontier has an unexpected kind")
    plan_map = plans or frontier["plans"]
    by_start: dict[str, list[LnsCandidateFeature]] = {}
    for state_hash, record in frontier["records"].items():
        for comparison in record["comparisons"]:
            role = str(comparison.get("role", ""))
            if role in {"anchor", "carried_anchor"}:
                continue
            start = str(comparison["start"])
            anchor_hash = str(comparison["anchor_state_hash"])
            candidate_payload = plan_map[str(state_hash)]["canonical_state"]
            anchor_payload = plan_map[anchor_hash]["canonical_state"]
            candidate_state = state_from_canonical_payload(candidate_payload)
            anchor_state = state_from_canonical_payload(anchor_payload)
            if candidate_state.canonical_hash() != str(state_hash):
                raise ValueError(f"frontier canonical hash mismatch: {state_hash}")
            feature = build_candidate_feature(
                candidate_state,
                anchor_state,
                operator=str(comparison["operator"]),
                moved_experts=comparison["moved_experts"],
                predicted_gain_pct=float(comparison["robust_gain_pct"]),
                start=start,
                strategy="unknown",
                state_hash=str(state_hash),
                anchor_state_hash=anchor_hash,
            )
            by_start.setdefault(start, []).append(feature)
    return {start: by_start[start] for start in sorted(by_start)}
