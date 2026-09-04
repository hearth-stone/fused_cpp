#!/usr/bin/env python3
"""Compare DDR-queue and victim-LLC PMU features with anchored LOCO fits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


COUNT_MODES = {
    0: "isolated_head",
    1: "wide16_same_head",
    2: "two_8t_same_head",
    4: "four_4t_same_head",
    8: "eight_2t_same_head",
    16: "many16_1t_same_head",
}
LAYOUT_MODE = "four_1t_same_head"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--layout-repeat", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _perf_counts(path: Path) -> dict[str, tuple[int, int, float]]:
    result: dict[str, tuple[int, int, float]] = {}
    with path.open(encoding="utf-8") as stream:
        rows = csv.reader(line for line in stream if line.strip() and not line.startswith("#"))
        for row in rows:
            if len(row) < 5:
                continue
            try:
                count = int(row[0])
                enabled_ns = int(float(row[3]))
                running_percent = float(row[4])
            except ValueError:
                continue
            result[row[2]] = count, enabled_ns, running_percent
    if not result:
        raise ValueError(f"no perf counts in {path}")
    return result


def _load_raw(session: Path, mode: str) -> dict[str, float]:
    payload = json.loads((session / f"{mode}.json").read_text(encoding="utf-8"))
    core = _perf_counts(session / f"core_{mode}.csv")
    uncore = _perf_counts(session / f"uncore_{mode}.csv")
    if any(value[2] != 100.0 for value in (*core.values(), *uncore.values())):
        raise ValueError(f"{mode} contains multiplexed or incomplete counters")
    read_cmd = sum(value[0] for event, value in uncore.items() if "/read_cmd/" in event)
    occupancy = sum(value[0] for event, value in uncore.items() if "/read_cmd_occupancy/" in event)
    if read_cmd <= 0:
        raise ValueError(f"{mode} has no DDR read commands")
    ll_read = core["ll_cache_rd"][0]
    ll_miss = core["ll_cache_miss_rd"][0]
    if ll_read <= 0:
        raise ValueError(f"{mode} has no victim LLC reads")
    return {
        "span_ms": float(payload["trace"]["target_span"]["median_ms"]),
        "queue_latency_cycles": occupancy / read_cmd,
        "llc_miss_ratio": ll_miss / ll_read,
    }


def load_points(session: Path, modes: dict[int, str]) -> list[dict[str, float]]:
    raw = {count: _load_raw(session, mode) for count, mode in modes.items()}
    baseline = raw[0]
    return [
        {
            "count": float(count),
            "response_ms": values["span_ms"] - baseline["span_ms"],
            "queue_pressure": values["queue_latency_cycles"] - baseline["queue_latency_cycles"],
            "llc_miss_pressure": values["llc_miss_ratio"] - baseline["llc_miss_ratio"],
            **values,
        }
        for count, values in sorted(raw.items())
    ]


def _slope(points: list[dict[str, float]], feature: str) -> float:
    numerator = sum(point[feature] * point["response_ms"] for point in points)
    denominator = sum(point[feature] ** 2 for point in points)
    if denominator <= 0:
        raise ValueError(f"feature {feature} has no positive fit energy")
    return max(0.0, numerator / denominator)


def fit_loco(points: list[dict[str, float]], feature: str) -> dict[str, object]:
    positive = [point for point in points if point["count"] > 0]
    folds = []
    for held_out in positive:
        training = [point for point in positive if point is not held_out]
        slope = _slope(training, feature)
        prediction = slope * held_out[feature]
        folds.append(
            {
                "held_out_count": int(held_out["count"]),
                "slope_ms_per_unit": slope,
                "measured_ms": held_out["response_ms"],
                "predicted_ms": prediction,
                "error_ms": prediction - held_out["response_ms"],
            }
        )
    errors = [float(fold["error_ms"]) for fold in folds]
    false_nonpositive = sum(float(fold["predicted_ms"]) <= 0 for fold in folds)
    feature_nonnegative = all(point[feature] >= 0 for point in positive)
    return {
        "feature": feature,
        "anchored_at_isolated": True,
        "full_slope_ms_per_unit": _slope(positive, feature),
        "loco_mae_ms": sum(abs(error) for error in errors) / len(errors),
        "loco_rmse_ms": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "loco_max_abs_error_ms": max(abs(error) for error in errors),
        "feature_nonnegative": feature_nonnegative,
        "loco_false_nonpositive": false_nonpositive,
        "eligible_for_monotone_pressure": feature_nonnegative and false_nonpositive == 0,
        "folds": folds,
    }


def _affine_coefficients(points: list[dict[str, float]], feature: str) -> tuple[float, float]:
    mean_x = sum(point[feature] for point in points) / len(points)
    mean_y = sum(point["span_ms"] for point in points) / len(points)
    denominator = sum((point[feature] - mean_x) ** 2 for point in points)
    if denominator <= 0:
        raise ValueError(f"feature {feature} has no affine fit energy")
    slope = max(
        0.0,
        sum((point[feature] - mean_x) * (point["span_ms"] - mean_y) for point in points) / denominator,
    )
    return mean_y - slope * mean_x, slope


def fit_loco_affine(points: list[dict[str, float]], feature: str) -> dict[str, object]:
    folds = []
    for held_out in (point for point in points if point["count"] > 0):
        intercept, slope = _affine_coefficients([point for point in points if point is not held_out], feature)
        prediction = intercept + slope * held_out[feature]
        folds.append(
            {
                "held_out_count": int(held_out["count"]),
                "intercept_ms": intercept,
                "slope_ms_per_unit": slope,
                "measured_ms": held_out["span_ms"],
                "predicted_ms": prediction,
                "error_ms": prediction - held_out["span_ms"],
            }
        )
    errors = [float(fold["error_ms"]) for fold in folds]
    intercept, slope = _affine_coefficients(points, feature)
    return {
        "feature": feature,
        "full_intercept_ms": intercept,
        "full_slope_ms_per_unit": slope,
        "loco_mae_ms": sum(abs(error) for error in errors) / len(errors),
        "loco_rmse_ms": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "loco_max_abs_error_ms": max(abs(error) for error in errors),
        "folds": folds,
    }


def layout_holdout(session: Path, models: dict[str, dict[str, object]]) -> dict[str, object]:
    points = load_points(session, {0: COUNT_MODES[0], 4: LAYOUT_MODE})
    point = points[1]
    predictions = {}
    for name, model in models.items():
        feature = str(model["feature"])
        predicted = float(model["full_slope_ms_per_unit"]) * point[feature]
        predictions[name] = {
            "predicted_ms": predicted,
            "error_ms": predicted - point["response_ms"],
        }
    return {"session": str(session), "point": point, "predictions": predictions}


def main() -> int:
    args = parse_args()
    points = load_points(args.session, COUNT_MODES)
    models = {
        "ddrc_queue_latency": fit_loco(points, "queue_pressure"),
        "victim_llc_miss_ratio": fit_loco(points, "llc_miss_pressure"),
    }
    affine_sensitivity = {
        "ddrc_queue_latency": fit_loco_affine(points, "queue_latency_cycles"),
        "victim_llc_miss_ratio": fit_loco_affine(points, "llc_miss_ratio"),
    }
    eligible = [name for name, model in models.items() if bool(model["eligible_for_monotone_pressure"])]
    result = {
        "kind": "moe_stream_pressure_pmu_loco",
        "schema_version": 1,
        "session": str(args.session),
        "points": points,
        "models": models,
        "affine_sensitivity": affine_sensitivity,
        "lowest_loco_mae": min(models, key=lambda name: float(models[name]["loco_mae_ms"])),
        "preferred_for_followup": (
            min(eligible, key=lambda name: float(models[name]["loco_mae_ms"])) if eligible else None
        ),
        "adopt_model": False,
        "decision_reason": (
            "LOCO errors remain large relative to the measured deltas; use physical-direction validity only to "
            "select a follow-up feature, not to freeze a model."
        ),
        "layout_holdouts": [layout_holdout(path, models) for path in args.layout_repeat],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
