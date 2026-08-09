"""Validate an analytical machine calibration against an empirical profile.

The empirical profile is a holdout oracle only. It is never loaded by the
analytical predictor and therefore cannot become a route/thread lookup table.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from analytic_model import ANALYTIC_MODEL_NAME, AnalyticMachineCalibration, AnalyticMoeCostModel
except ImportError:  # pragma: no cover - package-style import
    from .analytic_model import ANALYTIC_MODEL_NAME, AnalyticMachineCalibration, AnalyticMoeCostModel


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def _error_summary(rows: list[dict]) -> dict:
    absolute_relative_errors = [abs(float(row["relative_error"])) for row in rows]
    signed_relative_errors = [float(row["relative_error"]) for row in rows]
    return {
        "points": len(rows),
        "mape": statistics.fmean(absolute_relative_errors) if rows else 0.0,
        "p90_absolute_relative_error": _percentile(absolute_relative_errors, 0.90),
        "max_absolute_relative_error": max(absolute_relative_errors, default=0.0),
        "mean_signed_relative_error": statistics.fmean(signed_relative_errors) if rows else 0.0,
    }


def _model_from_profile(
    calibration: AnalyticMachineCalibration,
    profile: dict,
    *,
    down_output_element_bytes: int,
) -> AnalyticMoeCostModel:
    shape = profile["expert_shape"]
    kernel = profile["kernel"]
    parallelism = profile["parallelism"]
    exact_m = str(kernel.get("m_tail_policy", "")).startswith("xbyak_exact")
    return AnalyticMoeCostModel(
        calibration,
        hidden_size=int(shape["hidden_size"]),
        intermediate_size=int(shape["intermediate_size"]),
        global_experts=int(parallelism["global_experts"]),
        local_experts=int(parallelism["local_experts"]),
        mode=str(parallelism["mode"]),
        degree=int(parallelism["degree"]),
        concurrent_ranks=int(profile["target"]["concurrent_ranks"]),
        backend_n_tile=int(kernel["backend_n_tile"]),
        activation=str(shape["activation"]),
        dtype=str(shape["dtype"]),
        w13_ranges=int(kernel["w13_window_ranges"]),
        w2_ranges=int(kernel["w2_window_ranges"]),
        exact_m=exact_m,
        down_output_element_bytes=down_output_element_bytes,
    )


def _isolated_rows(model: AnalyticMoeCostModel, profile: dict) -> list[dict]:
    rows = []
    for observation in profile["isolated"]:
        routes = int(observation["routes"])
        threads = int(observation["threads"])
        if threads not in model.supported_widths:
            continue
        measured_ns = float(observation["median_ns"])
        predicted_ns = model.T_iso(routes, threads)
        rows.append(
            {
                "routes": routes,
                "threads": threads,
                "measured_ns": measured_ns,
                "predicted_ns": predicted_ns,
                "relative_error": predicted_ns / measured_ns - 1.0,
            }
        )
    return rows


def _profile_tasks(entry: dict) -> list[tuple[int, int, list[int]]]:
    routes = int(entry["routes"])
    shape = tuple(int(value) for value in entry["shape"])
    lane_counts = tuple(int(value) for value in entry["lane_task_counts"])
    if len(shape) != len(lane_counts):
        raise ValueError(
            f"shape/lane count mismatch for routes={routes}: shape={shape}, lane_task_counts={lane_counts}"
        )
    tasks: list[tuple[int, int, list[int]]] = []
    for width, count in zip(shape, lane_counts):
        previous: int | None = None
        for _ in range(count):
            dependencies = [] if previous is None else [previous]
            tasks.append((routes, width, dependencies))
            previous = len(tasks) - 1
    return tasks


def _contention_rows(model: AnalyticMoeCostModel, profile: dict) -> list[dict]:
    rows = []
    for entry in profile["entries"]:
        shape = tuple(int(value) for value in entry["shape"])
        if any(width not in model.supported_widths for width in shape):
            continue
        measured_ns = float(entry["full_call_median_ns"])
        predicted_ns = model.dag_makespan(_profile_tasks(entry))
        rows.append(
            {
                "routes": int(entry["routes"]),
                "shape": list(shape),
                "measured_ns": measured_ns,
                "predicted_ns": predicted_ns,
                "relative_error": predicted_ns / measured_ns - 1.0,
            }
        )
    return rows


def _ranking_summary(rows: list[dict]) -> dict:
    route_summaries = []
    for routes in sorted({int(row["routes"]) for row in rows}):
        candidates = [row for row in rows if int(row["routes"]) == routes]
        measured_best = min(candidates, key=lambda row: float(row["measured_ns"]))
        predicted_best = min(candidates, key=lambda row: float(row["predicted_ns"]))
        regret = float(predicted_best["measured_ns"]) / float(measured_best["measured_ns"]) - 1.0
        route_summaries.append(
            {
                "routes": routes,
                "predicted_shape": predicted_best["shape"],
                "measured_best_shape": measured_best["shape"],
                "measured_regret": regret,
            }
        )
    regrets = [float(row["measured_regret"]) for row in route_summaries]
    return {
        "routes": route_summaries,
        "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
        "max_regret": max(regrets, default=0.0),
    }


def build_validation_report(
    calibration: AnalyticMachineCalibration,
    profile: dict,
    *,
    down_output_element_bytes: int = 2,
    isolated_training_points: set[tuple[int, int]] | None = None,
) -> dict:
    if int(profile.get("schema_version", 0)) < 2:
        raise ValueError("analytical validation requires an empirical schema-v2 profile")
    profile_cores = int(profile["target"]["cores_per_rank"])
    if calibration.cores_per_rank != profile_cores:
        raise ValueError(
            f"calibration/profile cores_per_rank mismatch: {calibration.cores_per_rank} != {profile_cores}"
        )
    model = _model_from_profile(
        calibration,
        profile,
        down_output_element_bytes=down_output_element_bytes,
    )
    isolated = _isolated_rows(model, profile)
    training_points = isolated_training_points or set()
    isolated_holdout = [
        row for row in isolated if (int(row["routes"]), int(row["threads"])) not in training_points
    ]
    contention = _contention_rows(model, profile)
    return {
        "kind": "moe_analytic_validation",
        "analytic_model_schema_version": model.schema_version,
        "analytic_model": ANALYTIC_MODEL_NAME,
        "machine_id": calibration.machine_id,
        "holdout_profile": profile.get("measurement", {}).get("profile_id"),
        "model_policy": {
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "w13_window_ranges": model.w13_window_ranges,
            "w2_window_ranges": model.w2_window_ranges,
            "supported_widths": list(model.supported_widths),
        },
        "isolated": {
            "coverage": {
                "profile_points": len(profile["isolated"]),
                "evaluated_points": len(isolated),
                "skipped_points": len(profile["isolated"]) - len(isolated),
            },
            "summary": _error_summary(isolated),
            "holdout_summary": _error_summary(isolated_holdout),
            "training_points": [
                {"routes": routes, "threads": threads} for routes, threads in sorted(training_points)
            ],
            "rows": isolated,
        },
        "contention": {
            "coverage": {
                "profile_points": len(profile["entries"]),
                "evaluated_points": len(contention),
                "skipped_points": len(profile["entries"]) - len(contention),
            },
            "summary": _error_summary(contention),
            "ranking": _ranking_summary(contention),
            "rows": contention,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path, help="analytic machine calibration JSON")
    parser.add_argument("profile", type=Path, help="empirical schema-v2 holdout profile")
    parser.add_argument(
        "--down-output-element-bytes",
        type=int,
        choices=(2, 4),
        default=2,
        help="physical W2 route-store element size",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration_payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = AnalyticMachineCalibration.from_dict(calibration_payload)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    residual_training = calibration_payload.get("provenance", {}).get("isolated_residual_training", {})
    isolated_training_points = {
        (int(routes), int(threads))
        for routes in residual_training.get("routes", ())
        for threads in residual_training.get("threads", ())
    }
    report = build_validation_report(
        calibration,
        profile,
        down_output_element_bytes=args.down_output_element_bytes,
        isolated_training_points=isolated_training_points,
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(f"{output}\n", encoding="utf-8")

    isolated = report["isolated"]["summary"]
    contention = report["contention"]["summary"]
    ranking = report["contention"]["ranking"]
    if not all(
        math.isfinite(float(value))
        for value in (
            isolated["mape"],
            contention["mape"],
            ranking["max_regret"],
        )
    ):
        raise RuntimeError("analytical validation produced a non-finite metric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
