#!/usr/bin/env python3
"""Fit count 4/8 injection proxies and validate the locked count-6 grid."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_pmu_paired import _paired, mode_series


COUNTS = (4, 6, 8)
WIDTHS = (1, 2)
TRAIN_COUNTS = (4, 8)
HOLDOUT_COUNT = 6
QUEUE_ABS_GATE_CYCLES = 3.0
QUEUE_REL_GATE = 0.15
SLOWDOWN_ABS_GATE_MS = 0.02
PARAMETER_DRIFT_GATE = 0.20
PROXIES = (
    "distinct_b_count",
    "active_requester_threads",
    "count_x_requesters",
    "count_x_w13_overlap_core_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mode_name(count: int, width: int) -> str:
    return f"grid{count}_{width}t_same_head"


def _slope(points: list[dict[str, float]], x: str, y: str) -> float:
    denominator = sum(point[x] ** 2 for point in points)
    if denominator <= 0:
        raise ValueError(f"proxy {x} has no fit energy")
    return max(0.0, sum(point[x] * point[y] for point in points) / denominator)


def _relative_drift(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return abs(left - right) / scale


def _same_nonzero_direction(left: float, right: float) -> bool:
    return left * right > 0.0


def session_points(payload: dict[str, object], source: str) -> dict[str, object]:
    isolated = mode_series(payload, "isolated_head")
    points = []
    minimum_running = 1.0
    for count in COUNTS:
        for width in WIDTHS:
            name = mode_name(count, width)
            raw = payload["modes"][name]
            values = mode_series(payload, name)
            for rows in raw["pmu"].values():
                minimum_running = min(minimum_running, *(float(row["running_ratio"]) for row in rows))
            w13_overlap = float(raw["trace"]["peer_overlap_core_ms.w13_fused_silu_packc"]["median_ms"])
            points.append(
                {
                    "count": float(count),
                    "width": float(width),
                    "distinct_b_count": float(count),
                    "active_requester_threads": float(count * width),
                    "count_x_requesters": float(count * count * width),
                    "count_x_w13_overlap_core_ms": float(count) * w13_overlap,
                    "w13_overlap_core_ms": w13_overlap,
                    "queue_pressure": statistics.median(
                        _paired(values["queue_latency_cycles"], isolated["queue_latency_cycles"])
                    ),
                    "slowdown_ms": statistics.median(_paired(values["span_ms"], isolated["span_ms"])),
                }
            )
    return {"source": source, "minimum_running_ratio": minimum_running, "points": points}


def fit_session(session: dict[str, object]) -> dict[str, object]:
    points = session["points"]
    training = [point for point in points if int(point["count"]) in TRAIN_COUNTS]
    holdout = [point for point in points if int(point["count"]) == HOLDOUT_COUNT]
    queue_to_slowdown = _slope(training, "queue_pressure", "slowdown_ms")
    proxy_results = {}
    for proxy in PROXIES:
        proxy_to_queue = _slope(training, proxy, "queue_pressure")
        predictions = []
        for point in holdout:
            predicted_queue = proxy_to_queue * point[proxy]
            predicted_slowdown = queue_to_slowdown * predicted_queue
            queue_error = predicted_queue - point["queue_pressure"]
            slowdown_error = predicted_slowdown - point["slowdown_ms"]
            predictions.append(
                {
                    "width": int(point["width"]),
                    "measured_queue": point["queue_pressure"],
                    "predicted_queue": predicted_queue,
                    "queue_error": queue_error,
                    "queue_gate": max(QUEUE_ABS_GATE_CYCLES, QUEUE_REL_GATE * abs(point["queue_pressure"])),
                    "measured_slowdown_ms": point["slowdown_ms"],
                    "predicted_slowdown_ms": predicted_slowdown,
                    "slowdown_error_ms": slowdown_error,
                    "queue_pass": abs(queue_error)
                    <= max(QUEUE_ABS_GATE_CYCLES, QUEUE_REL_GATE * abs(point["queue_pressure"])),
                    "slowdown_pass": abs(slowdown_error) <= SLOWDOWN_ABS_GATE_MS,
                }
            )
        measured_direction = holdout[1]["slowdown_ms"] - holdout[0]["slowdown_ms"]
        predicted_direction = predictions[1]["predicted_slowdown_ms"] - predictions[0]["predicted_slowdown_ms"]
        proxy_results[proxy] = {
            "proxy_to_queue_slope": proxy_to_queue,
            "queue_to_slowdown_slope": queue_to_slowdown,
            "count6_predictions": predictions,
            "width_direction_correct": _same_nonzero_direction(predicted_direction, measured_direction),
            "session_pass": all(row["queue_pass"] and row["slowdown_pass"] for row in predictions)
            and _same_nonzero_direction(predicted_direction, measured_direction),
        }
    return {"source": session["source"], "proxy_results": proxy_results}


def decide(fits: list[dict[str, object]]) -> dict[str, object]:
    decisions = {}
    for proxy in PROXIES:
        rows = [fit["proxy_results"][proxy] for fit in fits]
        proxy_drift = _relative_drift(
            float(rows[0]["proxy_to_queue_slope"]), float(rows[1]["proxy_to_queue_slope"])
        )
        response_drift = _relative_drift(
            float(rows[0]["queue_to_slowdown_slope"]), float(rows[1]["queue_to_slowdown_slope"])
        )
        decisions[proxy] = {
            "all_session_gates_pass": all(bool(row["session_pass"]) for row in rows),
            "proxy_parameter_drift": proxy_drift,
            "response_parameter_drift": response_drift,
            "parameter_stability_pass": proxy_drift <= PARAMETER_DRIFT_GATE
            and response_drift <= PARAMETER_DRIFT_GATE,
            "accept": all(bool(row["session_pass"]) for row in rows)
            and proxy_drift <= PARAMETER_DRIFT_GATE
            and response_drift <= PARAMETER_DRIFT_GATE,
        }
    accepted = [proxy for proxy, result in decisions.items() if result["accept"]]
    return {
        "gates": {
            "queue_abs_cycles": QUEUE_ABS_GATE_CYCLES,
            "queue_relative": QUEUE_REL_GATE,
            "slowdown_abs_ms": SLOWDOWN_ABS_GATE_MS,
            "parameter_relative_drift": PARAMETER_DRIFT_GATE,
        },
        "proxies": decisions,
        "accepted": accepted,
        "stop_absolute_model_expansion": not accepted,
    }


def main() -> int:
    args = parse_args()
    if len(args.session) != 2:
        raise ValueError("proxy grid requires exactly two independent sessions")
    sessions = [
        session_points(json.loads(path.read_text(encoding="utf-8")), str(path)) for path in args.session
    ]
    fits = [fit_session(session) for session in sessions]
    result = {
        "kind": "moe_stream_pressure_proxy_grid",
        "schema_version": 1,
        "fit_counts": list(TRAIN_COUNTS),
        "holdout_count": HOLDOUT_COUNT,
        "sessions": sessions,
        "fits": fits,
        "decision": decide(fits),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
