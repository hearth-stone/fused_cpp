#!/usr/bin/env python3
"""Build an anchor-relative ordering report from measured planner artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNER_DIR))

from pairwise_plan_ordering import (  # noqa: E402
    CANDIDATE_BETTER,
    CANDIDATE_WORSE,
    INCOMPARABLE,
    AnchorRelativePartialOrder,
    PartialOrderCandidate,
    PairwiseObservation,
    PairwiseResidualCalibration,
    neighborhood_context,
    select_partial_order_shortlist,
)


_ORDER_OPERATORS = {
    "same_lane_adjacent_swap",
    "same_lane_insertion",
    "same_width_cross_lane_relocation",
    "same_width_cross_lane_swap",
    "same_width_cross_llc_relocation",
}
_WIDTH_OPERATORS = {
    "adjacent_width_expert_migration",
    "domain_local_adjacent_width_migration",
    "domain_local_lane_merge",
    "domain_local_lane_split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, nargs="+", required=True, help="Measured artifacts used to fit residual radii")
    parser.add_argument(
        "--evaluate",
        type=Path,
        nargs="+",
        help="Independent measured artifacts; omitted for an explicitly labelled development replay",
    )
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--minimum-context-samples", type=int, default=4)
    parser.add_argument("--minimum-actionable-gain-pct", type=float, default=2.0)
    parser.add_argument("--dominance-margin-pct", type=float, default=0.0)
    parser.add_argument("--allow-identity-mismatch", action="store_true")
    parser.add_argument("--top-k", default="8,16,32")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _paired_stats(anchor: list[float], candidate: list[float]) -> dict[str, float | int]:
    if not anchor or len(anchor) != len(candidate):
        raise ValueError("paired plan samples must be non-empty and have equal length")
    gains = [
        100.0 * (float(anchor_value) / float(candidate_value) - 1.0)
        for anchor_value, candidate_value in zip(anchor, candidate, strict=True)
    ]
    return {
        "median": statistics.median(gains),
        "p10": _percentile(gains, 0.10),
        "p90": _percentile(gains, 0.90),
        "wins": sum(value > 0.0 for value in gains),
        "runs": len(gains),
    }


def _operator_family(operator: str) -> str:
    if operator in _ORDER_OPERATORS or operator == "unknown_order_neighbor":
        return "order"
    if operator in _WIDTH_OPERATORS or operator == "unknown_width_neighbor":
        return "width"
    return "global"


def _width_histogram(plan: dict[str, object] | None) -> tuple[tuple[int, int], ...]:
    if not plan:
        return ()
    raw = plan.get("width_histogram", {})
    if not isinstance(raw, dict):
        return ()
    return tuple(sorted((int(width), int(count)) for width, count in raw.items()))


def _strict_context(
    name: str,
    anchor: dict[str, object],
    candidate: dict[str, object],
) -> tuple[str, str]:
    if name.startswith("cp_sat_"):
        return "global", "cp_sat_width_order_domain"
    anchor_histogram = _width_histogram(anchor.get("plan"))
    candidate_histogram = _width_histogram(candidate.get("plan"))
    if anchor_histogram and anchor_histogram == candidate_histogram:
        return "order", "template_temporal_order"
    if len(candidate_histogram) == 1:
        return "width", "homogeneous_width_change"
    if name == "conservative_selected":
        return "width", "one_step_width_change"
    if name == "greedy_strict":
        return "global", "greedy_family_change"
    return "global", "mixed_width_family_change"


def _prediction_by_hash(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for strategy in payload.get("strategies", {}).values():
        for row in strategy.get("scored", []):
            result[str(row["state_hash"])] = row
    decisions = [payload.get("uncertainty_aware_decision", {})]
    decisions.extend(payload.get("ablation_decisions", {}).values())
    for decision in decisions:
        state_hash = decision.get("candidate_state_hash")
        gain = decision.get("candidate_event_gain_pct")
        if state_hash is None or gain is None or str(state_hash) in result:
            continue
        result[str(state_hash)] = {
            "state_hash": str(state_hash),
            "event_gain_pct": float(gain),
            "robust_gain_pct": decision.get("candidate_robust_gain_pct"),
        }
    return result


def _inferred_neighborhood_operator(payload: dict[str, object], state_hash: str) -> str:
    sources = payload.get("hardware_shortlist", {}).get("sources", {}).get(state_hash, [])
    has_width = any("width_only" in str(source) for source in sources)
    has_order = any("order_only" in str(source) for source in sources)
    if has_width and not has_order:
        return "unknown_width_neighbor"
    if has_order and not has_width:
        return "unknown_order_neighbor"
    return "unknown_neighborhood"


def _delta_direction(value: object, *, tolerance: float = 1.0e-9) -> str:
    numeric = float(value)
    if numeric > tolerance:
        return "increase"
    if numeric < -tolerance:
        return "decrease"
    return "unchanged"


def _neighborhood_event_features(candidate: dict[str, object]) -> dict[str, object]:
    pair = candidate.get("affected_lanes")
    placed = candidate.get("placed_event_context")
    if not isinstance(pair, dict) or not isinstance(placed, dict):
        return {}
    delta = placed.get("delta")
    before = placed.get("before")
    after = placed.get("after")
    if not isinstance(delta, dict) or not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    before_lane = before.get("critical_lane", {})
    after_lane = after.get("critical_lane", {})
    before_task = before.get("critical_task", {})
    after_task = after.get("critical_task", {})
    return {
        "isolated_critical_lane_switched": bool(pair.get("critical_isolated_lane_switched", False)),
        "placed_critical_lane_switched": bool(delta.get("critical_lane_switched", False)),
        "placed_critical_expert_switched": bool(delta.get("critical_expert_switched", False)),
        "affected_head_direction": _delta_direction(delta.get("affected_head_ns", 0.0)),
        "affected_tail_direction": _delta_direction(delta.get("affected_tail_ns", 0.0)),
        "cohort_transition_direction": _delta_direction(delta.get("cohort_transition_count", 0)),
        "before_critical_lane": before_lane.get("lane_index"),
        "after_critical_lane": after_lane.get("lane_index"),
        "before_critical_expert": before_task.get("expert_id"),
        "after_critical_expert": after_task.get("expert_id"),
        "affected_tail_delta_ns": float(delta.get("affected_tail_ns", 0.0)),
        "affected_head_delta_ns": float(delta.get("affected_head_ns", 0.0)),
        "affected_phase_dilation_delta": float(
            delta.get("mean_affected_phase_dilation", 0.0)
        ),
        "affected_team_pressure_delta": float(
            delta.get("mean_affected_team_pressure_dilation", 0.0)
        ),
        "cohort_transition_count_delta": int(delta.get("cohort_transition_count", 0)),
    }


def _artifact_identity(path: Path, payload: dict[str, object]) -> dict[str, object]:
    route = payload.get("route", {})
    identity = payload.get("identity", {})
    route_payload = route if isinstance(route, dict) else {}
    return {
        "artifact": str(path),
        "artifact_kind": str(payload.get("kind", "unknown")),
        "calibration_sha256": identity.get("calibration_sha256", payload.get("profile_sha256")),
        "extension_sha256": identity.get("extension_sha256", payload.get("extension_sha256")),
        "route_sha256": route_payload.get("sha256", payload.get("route_sha256")),
        "route_layer": route_payload.get("layer_index", route_payload.get("layer_id")),
    }


def _extract_neighborhood_rows(path: Path, payload: dict[str, object]) -> list[dict[str, object]]:
    hardware = payload.get("hardware_shortlist", {}).get("measurement")
    if not hardware:
        return []
    baseline = hardware["baseline"]
    baseline_samples = [float(value) for value in baseline["samples_ms"]]
    predictions = _prediction_by_hash(payload)
    identity = _artifact_identity(path, payload)
    rows = []
    for candidate in hardware.get("candidates", []):
        state_hash = str(candidate["state_hash"])
        prediction = predictions.get(state_hash, {})
        operator = str(
            candidate.get(
                "operator",
                prediction.get("operator", _inferred_neighborhood_operator(payload, state_hash)),
            )
        )
        event_gain = candidate.get("event_gain_pct", prediction.get("event_gain_pct"))
        robust_gain = candidate.get("robust_gain_pct", prediction.get("robust_gain_pct"))
        predicted_gain = robust_gain if robust_gain is not None else event_gain
        if predicted_gain is None or event_gain is None:
            raise ValueError(f"{path}: no event gain for neighborhood state {state_hash}")
        paired = candidate.get("paired_speedup_pct")
        if not paired:
            paired = _paired_stats(
                baseline_samples,
                [float(value) for value in candidate["stats"]["samples_ms"]],
            )
        family = _operator_family(operator)
        event_features = _neighborhood_event_features(candidate)
        rows.append(
            {
                **identity,
                "anchor": "baseline",
                "anchor_state_hash": payload.get("baseline", {}).get("state_hash"),
                "candidate": str(candidate["name"]),
                "candidate_state_hash": state_hash,
                "family": family,
                "context": neighborhood_context(operator, event_features),
                "operator": operator,
                "event_context_features": event_features,
                "affected_lanes": candidate.get("affected_lanes"),
                "placed_event_context": candidate.get("placed_event_context"),
                "event_gain_pct": float(event_gain),
                "prediction_basis": "robust_gain" if robust_gain is not None else "event_gain",
                "predicted_gain_pct": float(predicted_gain),
                "measured_gain_pct": float(paired["median"]),
                "measured_p10_gain_pct": float(paired["p10"]),
                "measured_p90_gain_pct": float(paired["p90"]),
                "wins": int(paired["wins"]),
                "runs": int(paired["runs"]),
                "residual_pct": float(paired["median"]) - float(predicted_gain),
            }
        )
    return rows


def _predicted_ms(candidate: dict[str, object]) -> float | None:
    for field in ("event_model_ms", "predicted_ms"):
        if candidate.get(field) is not None:
            return float(candidate[field])
    return None


def _extract_strict_rows(path: Path, payload: dict[str, object]) -> list[dict[str, object]]:
    candidates = payload.get("candidates", {})
    stats = payload.get("stats", {})
    anchor_name = str(payload.get("cp_sat_measured", {}).get("event_selected", "full_selected"))
    anchor = candidates.get(anchor_name)
    anchor_stats = stats.get(anchor_name)
    if not anchor or not anchor_stats:
        return []
    anchor_predicted = _predicted_ms(anchor)
    if anchor_predicted is None:
        raise ValueError(f"{path}: anchor {anchor_name} has no event prediction")
    anchor_samples = [float(value) for value in anchor_stats["samples_ms"]]
    identity = _artifact_identity(path, payload)
    rows = []
    for name, candidate_stats in stats.items():
        if name == anchor_name or name not in candidates:
            continue
        candidate = candidates[name]
        candidate_predicted = _predicted_ms(candidate)
        if candidate_predicted is None:
            continue
        paired = _paired_stats(
            anchor_samples,
            [float(value) for value in candidate_stats["samples_ms"]],
        )
        family, context = _strict_context(str(name), anchor, candidate)
        predicted_gain = 100.0 * (anchor_predicted / candidate_predicted - 1.0)
        rows.append(
            {
                **identity,
                "anchor": anchor_name,
                "anchor_state_hash": None,
                "candidate": str(name),
                "candidate_state_hash": None,
                "family": family,
                "context": context,
                "event_gain_pct": predicted_gain,
                "prediction_basis": "event_gain",
                "predicted_gain_pct": predicted_gain,
                "measured_gain_pct": float(paired["median"]),
                "measured_p10_gain_pct": float(paired["p10"]),
                "measured_p90_gain_pct": float(paired["p90"]),
                "wins": int(paired["wins"]),
                "runs": int(paired["runs"]),
                "residual_pct": float(paired["median"]) - predicted_gain,
            }
        )
    return rows


def _extract_width_order_rows(path: Path, payload: dict[str, object]) -> list[dict[str, object]]:
    metadata = payload.get("metadata", {})
    stats = payload.get("stats", {})
    anchor_name = "full_selected"
    anchor = metadata.get(anchor_name)
    anchor_stats = stats.get(anchor_name)
    if not anchor or not anchor_stats:
        return []
    anchor_predicted = float(anchor["predicted_full_ms"])
    anchor_samples = [float(value) for value in anchor_stats["samples_ms"]]
    identity = _artifact_identity(path, payload)
    rows = []
    for name, candidate_stats in stats.items():
        if name == anchor_name or name not in metadata:
            continue
        candidate = metadata[name]
        candidate_predicted = float(candidate["predicted_full_ms"])
        paired = _paired_stats(
            anchor_samples,
            [float(value) for value in candidate_stats["samples_ms"]],
        )
        family, context = _strict_context(str(name), anchor, candidate)
        predicted_gain = 100.0 * (anchor_predicted / candidate_predicted - 1.0)
        rows.append(
            {
                **identity,
                "anchor": anchor_name,
                "anchor_state_hash": None,
                "candidate": str(name),
                "candidate_state_hash": None,
                "family": family,
                "context": context,
                "event_gain_pct": predicted_gain,
                "prediction_basis": "event_gain",
                "predicted_gain_pct": predicted_gain,
                "measured_gain_pct": float(paired["median"]),
                "measured_p10_gain_pct": float(paired["p10"]),
                "measured_p90_gain_pct": float(paired["p90"]),
                "wins": int(paired["wins"]),
                "runs": int(paired["runs"]),
                "residual_pct": float(paired["median"]) - predicted_gain,
            }
        )
    return rows


def extract_pairwise_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    kind = str(payload.get("kind", ""))
    if kind == "executable_neighborhood_audit":
        return _extract_neighborhood_rows(path, payload)
    if kind == "high_skew_planner_closure":
        return _extract_strict_rows(path, payload)
    if kind == "high_skew_width_order_measured_oracle":
        return _extract_width_order_rows(path, payload)
    raise ValueError(f"{path}: unsupported pairwise artifact kind {kind!r}")


def _measured_relation(row: dict[str, object], minimum_gain_pct: float) -> str:
    median = float(row["measured_gain_pct"])
    if median > minimum_gain_pct and float(row["measured_p10_gain_pct"]) > 0.0:
        return CANDIDATE_BETTER
    if median < -minimum_gain_pct and float(row["measured_p90_gain_pct"]) < 0.0:
        return CANDIDATE_WORSE
    return INCOMPARABLE


def _group_summary(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = {}
    for name, items in sorted(groups.items()):
        residuals = [float(row["residual_pct"]) for row in items]
        absolute = [abs(value) for value in residuals]
        result[name] = {
            "pairs": len(items),
            "median_residual_pct": statistics.median(residuals),
            "mean_absolute_residual_pct": statistics.fmean(absolute),
            "p90_absolute_residual_pct": _percentile(absolute, 0.90),
            "max_absolute_residual_pct": max(absolute),
        }
    return result


def _diverse_top_k(
    rows: list[dict[str, object]],
    top_k: int,
    comparator: AnchorRelativePartialOrder,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows_by_key = {str(row["candidate"]): row for row in rows}
    shortlist = select_partial_order_shortlist(
        comparator,
        (
            PartialOrderCandidate(
                key=str(row["candidate"]),
                family=str(row["family"]),
                context=str(row["context"]),
                predicted_gain_pct=float(row["predicted_gain_pct"]),
            )
            for row in rows
        ),
        budget=top_k,
    )
    return [rows_by_key[key] for key in shortlist.selected_keys], shortlist.to_dict()


def build_report(
    fit_rows: list[dict[str, object]],
    evaluation_rows: list[dict[str, object]],
    *,
    coverage: float,
    minimum_context_samples: int,
    minimum_gain_pct: float,
    dominance_margin_pct: float,
    top_ks: tuple[int, ...],
    evaluation_role: str,
) -> dict[str, object]:
    observations = [
        PairwiseObservation(
            family=str(row["family"]),
            context=str(row["context"]),
            predicted_gain_pct=float(row["predicted_gain_pct"]),
            measured_gain_pct=float(row["measured_gain_pct"]),
        )
        for row in fit_rows
    ]
    calibration = PairwiseResidualCalibration.fit(
        observations,
        coverage=coverage,
        minimum_context_samples=minimum_context_samples,
    )
    comparator = AnchorRelativePartialOrder(calibration, minimum_gain_pct=dominance_margin_pct)
    evaluated = []
    for raw in evaluation_rows:
        row = dict(raw)
        evidence = comparator.compare(
            float(row["predicted_gain_pct"]),
            family=str(row["family"]),
            context=str(row["context"]),
        )
        measured_relation = _measured_relation(row, minimum_gain_pct)
        row["measured_relation"] = measured_relation
        row["partial_order"] = {
            "relation": evidence.relation,
            "lower_gain_pct": evidence.lower_gain_pct,
            "upper_gain_pct": evidence.upper_gain_pct,
            "residual_radius_pct": evidence.residual_radius_pct,
            "residual_scope": evidence.residual_scope,
        }
        row["false_dominance"] = (
            evidence.relation == CANDIDATE_BETTER and measured_relation == CANDIDATE_WORSE
        ) or (
            evidence.relation == CANDIDATE_WORSE and measured_relation == CANDIDATE_BETTER
        )
        evaluated.append(row)

    resolvable = [row for row in evaluated if row["measured_relation"] != INCOMPARABLE]
    decided_resolvable = [row for row in resolvable if row["partial_order"]["relation"] != INCOMPARABLE]
    artifacts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in evaluated:
        artifacts[str(row["artifact"])].append(row)
    top_k_report = {}
    for artifact, rows in sorted(artifacts.items()):
        measured_best = max(rows, key=lambda row: float(row["measured_gain_pct"]))
        best_gain = float(measured_best["measured_gain_pct"])
        anchor_is_best = best_gain <= 0.0
        budgets = {}
        for top_k in top_ks:
            selected, selection = _diverse_top_k(rows, top_k, comparator)
            selected_names = {str(row["candidate"]) for row in selected}
            retained = anchor_is_best or str(measured_best["candidate"]) in selected_names
            selected_best_gain = max(
                [0.0, *(float(row["measured_gain_pct"]) for row in selected)],
            )
            best_time_ratio = 1.0 / (1.0 + max(best_gain, 0.0) / 100.0)
            selected_time_ratio = 1.0 / (1.0 + selected_best_gain / 100.0)
            budgets[str(top_k)] = {
                "selected_candidates": [str(row["candidate"]) for row in selected],
                "measured_best_retained": retained,
                "shortlist_regret_pct": 100.0 * (selected_time_ratio / best_time_ratio - 1.0),
                "dominated_candidates": selection["dominated_keys"],
                "budget_deferred_candidates": selection["budget_deferred_keys"],
            }
        top_k_report[artifact] = {
            "anchor": str(rows[0]["anchor"]),
            "measured_best": "anchor" if anchor_is_best else str(measured_best["candidate"]),
            "measured_best_gain_pct": max(best_gain, 0.0),
            "budgets": budgets,
        }

    false_dominance = [row for row in evaluated if row["false_dominance"]]
    false_pruning = [
        row
        for row in evaluated
        if row["partial_order"]["relation"] == CANDIDATE_WORSE
        and row["measured_relation"] == CANDIDATE_BETTER
    ]
    relation_counts = {
        relation: sum(row["partial_order"]["relation"] == relation for row in evaluated)
        for relation in (CANDIDATE_BETTER, CANDIDATE_WORSE, INCOMPARABLE)
    }
    return {
        "kind": "plan_pairwise_ordering_validation",
        "method": {
            "evaluation_role": evaluation_role,
            "gain_convention": "positive_candidate_speedup_over_anchor",
            "measured_relation": "paired_p10/p90_clear_minimum_actionable_gain",
            "partial_order": "symmetric_empirical_absolute_delta_residual_interval",
            "minimum_actionable_gain_pct": minimum_gain_pct,
            "dominance_margin_pct": dominance_margin_pct,
            "top_k": list(top_ks),
        },
        "calibration": calibration.to_dict(),
        "fit": {
            "pairs": len(fit_rows),
            "artifacts": sorted({str(row["artifact"]) for row in fit_rows}),
            "model_identities": [
                {"calibration_sha256": calibration_sha, "extension_sha256": extension_sha}
                for calibration_sha, extension_sha in sorted(
                    _model_identities(fit_rows),
                    key=lambda item: (str(item[0]), str(item[1])),
                )
            ],
            "by_family": _group_summary(fit_rows, "family"),
            "by_context": _group_summary(fit_rows, "context"),
        },
        "evaluation": {
            "pairs": len(evaluated),
            "model_identities": [
                {"calibration_sha256": calibration_sha, "extension_sha256": extension_sha}
                for calibration_sha, extension_sha in sorted(
                    _model_identities(evaluated),
                    key=lambda item: (str(item[0]), str(item[1])),
                )
            ],
            "resolvable_pairs": len(resolvable),
            "decided_resolvable_pairs": len(decided_resolvable),
            "resolvable_decision_coverage": (
                len(decided_resolvable) / len(resolvable) if resolvable else 1.0
            ),
            "false_dominance_pairs": len(false_dominance),
            "false_pruning_pairs": len(false_pruning),
            "relation_counts": relation_counts,
            "by_family": _group_summary(evaluated, "family"),
            "by_context": _group_summary(evaluated, "context"),
            "largest_absolute_residuals": sorted(
                evaluated,
                key=lambda row: -abs(float(row["residual_pct"])),
            )[:16],
            "top_k_recall": top_k_report,
            "rows": evaluated,
        },
        "gates": {
            "zero_false_pruning": not false_pruning,
            "measured_best_retained": {
                str(top_k): all(
                    artifact["budgets"][str(top_k)]["measured_best_retained"]
                    for artifact in top_k_report.values()
                )
                for top_k in top_ks
            },
        },
    }


def _parse_top_ks(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise ValueError("top-k must be a comma-separated integer list") from error
    if not result or result[0] <= 0:
        raise ValueError("top-k must contain positive integers")
    return result


def _load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        rows.extend(extract_pairwise_rows(path))
    if not rows:
        raise ValueError("no measured pairwise rows were found")
    return rows


def _model_identities(rows: list[dict[str, object]]) -> set[tuple[object, object]]:
    return {
        (row.get("calibration_sha256"), row.get("extension_sha256"))
        for row in rows
    }


def main() -> int:
    args = parse_args()
    fit_rows = _load_rows(args.fit)
    evaluation_paths = args.evaluate or args.fit
    evaluation_rows = _load_rows(evaluation_paths)
    evaluation_role = "development_replay_not_holdout"
    if args.evaluate:
        overlapping = {path.resolve() for path in args.fit} & {path.resolve() for path in args.evaluate}
        if overlapping:
            raise ValueError(f"fit and evaluation artifacts overlap: {sorted(map(str, overlapping))}")
        identities_match = _model_identities(fit_rows) == _model_identities(evaluation_rows)
        if not identities_match and not args.allow_identity_mismatch:
            raise ValueError(
                "fit and evaluation calibration/extension identities differ; "
                "use --allow-identity-mismatch only for a diagnostic replay"
            )
        evaluation_role = "holdout" if identities_match else "cross_identity_diagnostic_not_holdout"
    report = build_report(
        fit_rows,
        evaluation_rows,
        coverage=args.coverage,
        minimum_context_samples=args.minimum_context_samples,
        minimum_gain_pct=args.minimum_actionable_gain_pct,
        dominance_margin_pct=args.dominance_margin_pct,
        top_ks=_parse_top_ks(args.top_k),
        evaluation_role=evaluation_role,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
