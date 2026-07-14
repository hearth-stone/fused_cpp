#!/usr/bin/env python3
"""Validate sparse ``T_iso`` calibration against held-out measurements.

The input profile may contain a dense isolated table for evaluation.  Only the
requested calibration routes are passed to the fitter; every other measured
route is a true holdout for this report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from bisect import bisect_left
from pathlib import Path

try:
    from iso_formula import IsoFormula, fit_from_measurements
except ImportError:  # pragma: no cover - package-style import
    from .iso_formula import IsoFormula, fit_from_measurements


DEFAULT_CALIBRATION_ROUTES = "1,2,4,8,12,24,48,192,768,2040"


def parse_ints(text: str) -> list[int]:
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    absolute = [abs(row["error_pct"]) for row in rows]
    signed = [row["error_pct"] for row in rows]
    return {
        "count": len(rows),
        "median_abs_error_pct": statistics.median(absolute),
        "p90_abs_error_pct": percentile(absolute, 0.90),
        "max_abs_error_pct": max(absolute),
        "median_signed_error_pct": statistics.median(signed),
    }


def predict_m12(
    formula: IsoFormula,
    routes: int,
    threads: int,
    tail_lookup: dict[tuple[int, int], float],
) -> float:
    blocks, remainder = divmod(routes, 12)
    if remainder <= 0:
        tail = 0
    elif remainder <= 2:
        tail = remainder
    elif remainder <= 4:
        tail = 4
    elif remainder <= 8:
        tail = 8
    else:
        tail = 12
    overhead = formula.O(threads)
    if blocks == 0:
        return tail_lookup[(tail, threads)]
    bulk = (
        tail_lookup[(blocks * 12, threads)]
        if blocks <= 2
        else formula.T_iso(blocks * 12, threads)
    )
    if tail == 0:
        return bulk
    return (
        overhead
        + max(bulk - overhead, 0.0)
        + max(tail_lookup[(tail, threads)] - overhead, 0.0)
    )


def table_interpolate(curve: dict[int, float], routes: int) -> float:
    anchors = sorted(curve)
    if routes <= anchors[0]:
        return curve[anchors[0]] * routes / anchors[0]
    if routes >= anchors[-1]:
        return curve[anchors[-1]] * routes / anchors[-1]
    index = bisect_left(anchors, routes)
    r0, r1 = anchors[index - 1], anchors[index]
    return curve[r0] + (curve[r1] - curve[r0]) * (routes - r0) / (r1 - r0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--calibration-routes",
        default=DEFAULT_CALIBRATION_ROUTES,
        help="Routes admitted to the fitter; all remaining routes are holdout.",
    )
    parser.add_argument(
        "--calibration-threads",
        default=None,
        help="Thread counts admitted to the fitter; defaults to every measured team.",
    )
    parser.add_argument("--phi-route-min", type=int, default=192)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--fail-holdout-median-pct",
        type=float,
        default=None,
        help="Return non-zero when holdout median absolute error exceeds this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    measured = [
        (int(entry["routes"]), int(entry["threads"]), float(entry["median_ns"]))
        for entry in profile["isolated"]
    ]
    measured_routes = sorted({route for route, _, _ in measured})
    measured_threads = sorted({threads for _, threads, _ in measured})
    measured_lookup = {
        (routes, threads): value for routes, threads, value in measured
    }
    calibration_routes = [
        route
        for route in parse_ints(args.calibration_routes)
        if route in measured_routes
    ]
    calibration_threads = (
        measured_threads
        if args.calibration_threads is None
        else [
            team
            for team in parse_ints(args.calibration_threads)
            if team in measured_threads
        ]
    )
    calibration_route_set = set(calibration_routes)
    calibration_thread_set = set(calibration_threads)
    fit_points = [
        point
        for point in measured
        if point[0] in calibration_route_set and point[1] in calibration_thread_set
    ]
    formula = fit_from_measurements(fit_points, phi_route_min=args.phi_route_min)

    use_m12 = int(profile.get("schema_version", 1)) >= 2
    table_curves = {
        team: {
            route: value
            for route, measured_team, value in fit_points
            if measured_team == team
        }
        for team in calibration_threads
    }
    rows: list[dict] = []
    for routes, threads, measured_ns in measured:
        if threads < formula.min_threads or threads > formula.max_threads:
            continue
        predicted_ns = (
            predict_m12(formula, routes, threads, measured_lookup)
            if use_m12
            else formula.T_iso(routes, threads)
        )
        table_ns = (
            table_interpolate(table_curves[threads], routes)
            if threads in table_curves
            else None
        )
        rows.append(
            {
                "routes": routes,
                "threads": threads,
                "kind": (
                    "calibration"
                    if routes in calibration_route_set
                    and threads in calibration_thread_set
                    else "holdout"
                ),
                "measured_ns": measured_ns,
                "predicted_ns": predicted_ns,
                "error_pct": (predicted_ns / measured_ns - 1.0) * 100.0,
                "sparse_table_error_pct": (
                    None
                    if table_ns is None
                    else (table_ns / measured_ns - 1.0) * 100.0
                ),
            }
        )

    calibration_rows = [row for row in rows if row["kind"] == "calibration"]
    holdout_rows = [row for row in rows if row["kind"] == "holdout"]
    summary = {
        "calibration": summarize(calibration_rows),
        "holdout": summarize(holdout_rows),
        "holdout_by_thread": {
            str(team): summarize(
                [row for row in holdout_rows if row["threads"] == team]
            )
            for team in measured_threads
        },
    }
    table_holdout = [
        {
            **row,
            "error_pct": row["sparse_table_error_pct"],
        }
        for row in holdout_rows
        if row["sparse_table_error_pct"] is not None
    ]
    summary["sparse_table_holdout"] = summarize(table_holdout)

    print(f"profile: {args.profile}")
    print(f"calibration routes: {calibration_routes}")
    print(f"holdout routes: {sorted(set(measured_routes) - calibration_route_set)}")
    print(f"calibration threads: {calibration_threads}")
    print(
        "formula: O(t)=%.6f%+.6f/t ms, alpha=%.7f beta=%.8f"
        % (formula.o0 / 1e6, formula.o1 / 1e6, formula.alpha, formula.beta)
    )
    for name in ("calibration", "holdout", "sparse_table_holdout"):
        result = summary[name]
        if result["count"] == 0:
            print(f"{name:22s}: no points")
            continue
        print(
            f"{name:22s}: n={result['count']:3d} "
            f"median={result['median_abs_error_pct']:6.2f}% "
            f"p90={result['p90_abs_error_pct']:6.2f}% "
            f"max={result['max_abs_error_pct']:6.2f}% "
            f"bias={result['median_signed_error_pct']:+6.2f}%"
        )
    if args.details:
        print("\n  routes threads kind          measured_ms predicted_ms error%")
        for row in rows:
            print(
                "  %6d %7d %-12s %11.4f %12.4f %+7.2f"
                % (
                    row["routes"],
                    row["threads"],
                    row["kind"],
                    row["measured_ns"] / 1e6,
                    row["predicted_ns"] / 1e6,
                    row["error_pct"],
                )
            )

    payload = {
        "profile": str(args.profile),
        "calibration_routes": calibration_routes,
        "calibration_threads": calibration_threads,
        "formula": formula.to_dict(),
        "summary": summary,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    if (
        args.fail_holdout_median_pct is not None
        and summary["holdout"].get("median_abs_error_pct", 0.0)
        > args.fail_holdout_median_pct
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
