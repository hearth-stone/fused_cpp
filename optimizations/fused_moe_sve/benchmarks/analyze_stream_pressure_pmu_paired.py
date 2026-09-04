#!/usr/bin/env python3
"""Analyze same-process per-round stream-pressure PMU measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_pmu import (
    COUNT_MODES,
    LAYOUT_MODE,
    fit_loco,
    fit_loco_affine,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--layout-repeat", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _event_counts(mode: dict[str, object], event: str) -> list[float]:
    rows = mode["pmu"][event]
    if any(abs(float(row["running_ratio"]) - 1.0) > 1e-9 for row in rows):
        raise ValueError(f"event {event} contains incomplete counter intervals")
    return [float(row["count"]) for row in rows]


def mode_series(payload: dict[str, object], mode_name: str) -> dict[str, list[float]]:
    mode = payload["modes"][mode_name]
    span = [float(value) for value in mode["trace"]["target_span"]["samples_ms"]]
    read_cmd = [0.0] * len(span)
    occupancy = [0.0] * len(span)
    l3_ref = [0.0] * len(span)
    l3_hit = [0.0] * len(span)
    for event in mode["pmu"]:
        values = _event_counts(mode, event)
        if len(values) != len(span):
            raise ValueError(f"event {event} has {len(values)} samples, expected {len(span)}")
        if event.startswith("ddrc") and event.endswith(".read_cmd"):
            read_cmd = [left + right for left, right in zip(read_cmd, values, strict=True)]
        elif event.startswith("ddrc") and event.endswith(".read_cmd_occupancy"):
            occupancy = [left + right for left, right in zip(occupancy, values, strict=True)]
        elif event.startswith("l3c") and event.endswith(".l3c_ref"):
            l3_ref = [left + right for left, right in zip(l3_ref, values, strict=True)]
        elif event.startswith("l3c") and event.endswith(".l3c_hit"):
            l3_hit = [left + right for left, right in zip(l3_hit, values, strict=True)]
    ll_read = _event_counts(mode, "core.ll_cache_rd")
    ll_miss = _event_counts(mode, "core.ll_cache_miss_rd")
    cycles = _event_counts(mode, "core.cycles")
    stalls = _event_counts(mode, "core.stall_backend")
    if any(value <= 0 for value in read_cmd + ll_read + l3_ref + cycles):
        raise ValueError(f"mode {mode_name} contains a zero PMU denominator")
    return {
        "span_ms": span,
        "queue_latency_cycles": [left / right for left, right in zip(occupancy, read_cmd, strict=True)],
        "llc_miss_ratio": [left / right for left, right in zip(ll_miss, ll_read, strict=True)],
        "backend_stall_ratio": [left / right for left, right in zip(stalls, cycles, strict=True)],
        "l3_hit_ratio": [left / right for left, right in zip(l3_hit, l3_ref, strict=True)],
    }


def _paired(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("paired series lengths differ")
    return [value - baseline for value, baseline in zip(left, right, strict=True)]


def count_points(payload: dict[str, object]) -> list[dict[str, float]]:
    baseline = mode_series(payload, COUNT_MODES[0])
    points = []
    for count, mode in sorted(COUNT_MODES.items()):
        values = mode_series(payload, mode)
        points.append(
            {
                "count": float(count),
                "span_ms": statistics.median(values["span_ms"]),
                "response_ms": statistics.median(_paired(values["span_ms"], baseline["span_ms"])),
                "queue_latency_cycles": statistics.median(values["queue_latency_cycles"]),
                "queue_pressure": statistics.median(
                    _paired(values["queue_latency_cycles"], baseline["queue_latency_cycles"])
                ),
                "llc_miss_ratio": statistics.median(values["llc_miss_ratio"]),
                "llc_miss_pressure": statistics.median(
                    _paired(values["llc_miss_ratio"], baseline["llc_miss_ratio"])
                ),
                "l3_hit_ratio": statistics.median(values["l3_hit_ratio"]),
            }
        )
    return points


def layout_holdout(
    payload: dict[str, object],
    source: str,
    models: dict[str, dict[str, object]],
) -> dict[str, object]:
    baseline = mode_series(payload, COUNT_MODES[0])
    layout = mode_series(payload, LAYOUT_MODE)
    point = {
        "response_ms": statistics.median(_paired(layout["span_ms"], baseline["span_ms"])),
        "queue_pressure": statistics.median(
            _paired(layout["queue_latency_cycles"], baseline["queue_latency_cycles"])
        ),
        "llc_miss_pressure": statistics.median(
            _paired(layout["llc_miss_ratio"], baseline["llc_miss_ratio"])
        ),
    }
    predictions = {}
    for name, model in models.items():
        feature = str(model["feature"])
        prediction = float(model["full_slope_ms_per_unit"]) * point[feature]
        predictions[name] = {"predicted_ms": prediction, "error_ms": prediction - point["response_ms"]}
    return {"source": source, "point": point, "predictions": predictions}


def data_quality(payload: dict[str, object]) -> dict[str, object]:
    runs = int(payload["method"]["runs"])
    expected = {mode: count for count, mode in COUNT_MODES.items()}
    expected[LAYOUT_MODE] = 4
    overlap = {}
    minimum_running = 1.0
    for mode, count in expected.items():
        values = payload["modes"][mode]
        target = values["trace"]["target_span"]["samples_ms"]
        if len(target) != runs:
            raise ValueError(f"mode {mode} has {len(target)} trace samples, expected {runs}")
        actual = 0.0
        if count:
            actual = float(values["trace"]["peer_overlap_experts"]["median_ms"])
        overlap[mode] = actual
        if actual != float(count):
            raise ValueError(f"mode {mode} overlap {actual} does not match {count}")
        for rows in values["pmu"].values():
            if len(rows) != runs:
                raise ValueError(f"mode {mode} has an incomplete PMU series")
            minimum_running = min(minimum_running, *(float(row["running_ratio"]) for row in rows))
    return {"runs": runs, "overlap_experts": overlap, "minimum_running_ratio": minimum_running}


def main() -> int:
    args = parse_args()
    payload = json.loads(args.session.read_text(encoding="utf-8"))
    quality = data_quality(payload)
    points = count_points(payload)
    models = {
        "ddrc_queue_latency": fit_loco(points, "queue_pressure"),
        "victim_llc_miss_ratio": fit_loco(points, "llc_miss_pressure"),
    }
    repeats = []
    for path in args.layout_repeat:
        repeat = json.loads(path.read_text(encoding="utf-8"))
        repeats.append(layout_holdout(repeat, str(path), models))
    result = {
        "kind": "moe_stream_pressure_pmu_paired_loco",
        "schema_version": 1,
        "session": str(args.session),
        "data_quality": quality,
        "points": points,
        "models": models,
        "affine_sensitivity": {
            "ddrc_queue_latency": fit_loco_affine(points, "queue_latency_cycles"),
            "victim_llc_miss_ratio": fit_loco_affine(points, "llc_miss_ratio"),
        },
        "layout_holdouts": [layout_holdout(payload, str(args.session), models), *repeats],
        "lowest_loco_mae": min(models, key=lambda name: float(models[name]["loco_mae_ms"])),
        "adopt_model": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
