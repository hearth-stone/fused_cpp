#!/usr/bin/env python3
"""Compare request-arrival shapes at fixed distinct packed-B counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimizations.fused_moe_sve.benchmarks.analyze_stream_pressure_pmu_paired import _paired, mode_series
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import _stats


SHAPE_GROUPS = {
    4: ("four_4t_same_head", "four_2t_same_head", "four_1t_same_head"),
    8: ("eight_2t_same_head", "eight_1t_same_head"),
}
PAIRWISE = (
    (4, "four_2t_same_head", "four_4t_same_head"),
    (4, "four_1t_same_head", "four_2t_same_head"),
    (4, "four_1t_same_head", "four_4t_same_head"),
    (8, "eight_1t_same_head", "eight_2t_same_head"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    stats = _stats(values)
    return {
        "median": float(stats["median_ms"]),
        "p10": float(stats["p10_ms"]),
        "p90": float(stats["p90_ms"]),
    }


def analyze_session(payload: dict[str, object], source: str) -> dict[str, object]:
    runs = int(payload["method"]["runs"])
    required = {"isolated_head", *(mode for modes in SHAPE_GROUPS.values() for mode in modes)}
    missing = required.difference(payload["modes"])
    if missing:
        raise ValueError(f"request-shape session is missing modes: {sorted(missing)}")
    series = {mode: mode_series(payload, mode) for mode in required}
    isolated = series["isolated_head"]
    modes = {}
    minimum_running = 1.0
    for mode in required:
        raw_mode = payload["modes"][mode]
        if len(raw_mode["trace"]["target_span"]["samples_ms"]) != runs:
            raise ValueError(f"mode {mode} has incomplete trace samples")
        for rows in raw_mode["pmu"].values():
            if len(rows) != runs:
                raise ValueError(f"mode {mode} has incomplete PMU samples")
            minimum_running = min(minimum_running, *(float(row["running_ratio"]) for row in rows))
        values = series[mode]
        modes[mode] = {
            "live_streams": int(raw_mode["live_streams"]),
            "live_threads": int(raw_mode["live_threads"]),
            "span_ms": _summary(values["span_ms"]),
            "vs_isolated": {
                metric: _summary(_paired(values[metric], isolated[metric]))
                for metric in (
                    "span_ms",
                    "queue_latency_cycles",
                    "llc_miss_ratio",
                    "backend_stall_ratio",
                )
            },
        }
    comparisons = []
    for count, narrower, wider in PAIRWISE:
        narrow_values = series[narrower]
        wide_values = series[wider]
        metrics = {
            metric: _summary(_paired(narrow_values[metric], wide_values[metric]))
            for metric in (
                "span_ms",
                "queue_latency_cycles",
                "llc_miss_ratio",
                "backend_stall_ratio",
            )
        }
        comparisons.append(
            {
                "packed_b_count": count,
                "narrower": narrower,
                "wider": wider,
                "narrower_minus_wider": metrics,
                "stable_narrower_slowdown": metrics["span_ms"]["p10"] > 0.0,
                "stable_narrower_speedup": metrics["span_ms"]["p90"] < 0.0,
                "stable_lower_aggregate_queue": metrics["queue_latency_cycles"]["p90"] < 0.0,
            }
        )
    return {
        "source": source,
        "runs": runs,
        "minimum_running_ratio": minimum_running,
        "modes": modes,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    sessions = [
        analyze_session(json.loads(path.read_text(encoding="utf-8")), str(path)) for path in args.session
    ]
    stable_asymmetric = []
    for pair_index, pair in enumerate(PAIRWISE):
        rows = [session["comparisons"][pair_index] for session in sessions]
        stable_asymmetric.append(
            {
                "packed_b_count": pair[0],
                "narrower": pair[1],
                "wider": pair[2],
                "narrower_slower_with_lower_queue_in_all_sessions": all(
                    row["stable_narrower_slowdown"] and row["stable_lower_aggregate_queue"] for row in rows
                ),
                "narrower_faster_with_lower_queue_in_all_sessions": all(
                    row["stable_narrower_speedup"] and row["stable_lower_aggregate_queue"] for row in rows
                ),
            }
        )
    result = {
        "kind": "moe_stream_pressure_request_shape",
        "schema_version": 1,
        "sessions": sessions,
        "cross_session_signatures": stable_asymmetric,
        "add_model": False,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
