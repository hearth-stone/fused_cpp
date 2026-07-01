#!/usr/bin/env python3
"""Build a compact native planner cost table from a native planner profile."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_PLANNERS = [
    "FIXED_GLOBAL_THREADS",
    "SORTED_TOKEN_BALANCED_1T",
    "UNIFORM_WAVES",
    "ENUMERATE_CORE_GROUPS",
    "GREEDY_MARGINAL_GAIN",
]


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def nearest(value: int, buckets: List[int]) -> int:
    return min(buckets, key=lambda bucket: (abs(bucket - value), bucket))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert profile_native_planner_cost.py output into a lightweight "
            "lookup table keyed by planner, core bucket, and active-expert bucket."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric to keep. Defaults to input metric or total_native_median_ns.",
    )
    parser.add_argument(
        "--active-buckets",
        default=None,
        help=(
            "Comma-separated active-expert buckets to emit. Defaults to the "
            "active_grid_* buckets in the input profile, or all observed active counts."
        ),
    )
    parser.add_argument(
        "--core-buckets",
        default=None,
        help="Comma-separated core buckets to emit. Defaults to all observed cores.",
    )
    parser.add_argument(
        "--planner",
        action="append",
        default=None,
        help=(
            "Planner kind to keep. Can be passed multiple times. Defaults to "
            "the five scheduler planners."
        ),
    )
    parser.add_argument(
        "--prefer-active-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer rows whose case name is active_grid_<N>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    metric = args.metric or str(payload.get("metric", "total_native_median_ns"))
    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"input profile has no entries: {args.input_json}")

    planners = set(args.planner or DEFAULT_PLANNERS)
    observed_core_buckets = sorted({int(entry["cores"]) for entry in raw_entries})
    core_buckets = (
        parse_int_list(args.core_buckets)
        if args.core_buckets is not None
        else observed_core_buckets
    )

    active_grid_counts = sorted(
        {
            int(str(entry["case"]).split("active_grid_", 1)[1])
            for entry in raw_entries
            if str(entry.get("case", "")).startswith("active_grid_")
        }
    )
    observed_active_counts = sorted(
        {int(entry["active_experts"]) for entry in raw_entries}
    )
    active_buckets = (
        parse_int_list(args.active_buckets)
        if args.active_buckets is not None
        else active_grid_counts or observed_active_counts
    )

    grouped: Dict[Tuple[int, str, int], List[int]] = {}
    preferred: Dict[Tuple[int, str, int], List[int]] = {}
    for entry in raw_entries:
        planner = str(entry["planner"])
        if planner not in planners:
            continue
        cores = int(entry["cores"])
        active = int(entry["active_experts"])
        if metric not in entry:
            raise ValueError(f"entry missing metric {metric!r}: {entry}")
        key = (cores, planner, active)
        grouped.setdefault(key, []).append(int(entry[metric]))
        if str(entry.get("case", "")) == f"active_grid_{active}":
            preferred.setdefault(key, []).append(int(entry[metric]))

    table: Dict[str, Dict[str, Dict[str, int]]] = {}
    for core_bucket in core_buckets:
        table[str(core_bucket)] = {}
        source_core = nearest(core_bucket, observed_core_buckets)
        for planner in sorted(planners):
            by_active: Dict[str, int] = {}
            available_active = sorted(
                {
                    active
                    for cores, planner_name, active in grouped
                    if cores == source_core and planner_name == planner
                }
            )
            if not available_active:
                continue
            for active_bucket in active_buckets:
                source_active = nearest(active_bucket, available_active)
                key = (source_core, planner, source_active)
                values = (
                    preferred.get(key)
                    if args.prefer_active_grid and preferred.get(key)
                    else grouped[key]
                )
                by_active[str(active_bucket)] = int(statistics.median(values))
            table[str(core_bucket)][planner] = by_active

    output = {
        "schema_version": 1,
        "kind": "native_planner_cost_table",
        "source_profile": str(args.input_json),
        "metric": metric,
        "lookup": "nearest_core_then_nearest_active",
        "core_buckets": core_buckets,
        "active_buckets": active_buckets,
        "planners": sorted(planners),
        "table": table,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
