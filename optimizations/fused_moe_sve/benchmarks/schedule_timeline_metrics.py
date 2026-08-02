"""Derive utilization metrics from a captured per-core MoE timeline."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_GFLOPS_COLOR_MAX = 400.0
_GEMM_STAGE_KINDS = {"w13", "w2"}
_COMPUTE_STAGE_KINDS = {"gather", *_GEMM_STAGE_KINDS}


def stage_kind(stage: str) -> str:
    if "gather" in stage:
        return "gather"
    if "w13" in stage:
        return "w13"
    if "w2" in stage:
        return "w2"
    if stage == "merge_ready_token":
        return "merge"
    if "merge" in stage:
        return "cleanup"
    return "overhead"


def _split_evenly(units: int, parts: int, index: int) -> tuple[int, int]:
    if units <= 0 or parts <= 0 or index < 0 or index >= parts:
        return 0, 0
    units_per_part, extra_units = divmod(units, parts)
    if index < extra_units:
        return index * (units_per_part + 1), units_per_part + 1
    return (
        extra_units * (units_per_part + 1) + (index - extra_units) * units_per_part,
        units_per_part,
    )


def _owned_n_columns(
    *,
    n: int,
    k: int,
    n_tile: int,
    threads: int,
    local_tid: int,
    weight_window_bytes: int,
    fallback_ranges: int,
) -> int:
    if min(n, k, n_tile, threads) <= 0:
        raise ValueError("GEMM dimensions, N tile, and thread count must be positive")
    if n % n_tile != 0:
        raise ValueError(f"GEMM N={n} is not aligned to N tile {n_tile}")
    if weight_window_bytes < 0:
        raise ValueError(f"resolved weight window must be non-negative, got {weight_window_bytes}")

    total_tiles = n // n_tile
    ranges = max(1, min(fallback_ranges, total_tiles))
    if weight_window_bytes > 0:
        bytes_per_tile = k * n_tile * 2
        max_tiles_per_window = max(1, weight_window_bytes // bytes_per_tile)
        ranges = math.ceil(total_tiles / max_tiles_per_window)

    owned_tiles = 0
    for range_index in range(ranges):
        _, window_tiles = _split_evenly(total_tiles, ranges, range_index)
        _, local_tiles = _split_evenly(window_tiles, threads, local_tid)
        owned_tiles += local_tiles
    return owned_tiles * n_tile


def annotate_gemm_throughput(
    actual: dict[str, Any],
    *,
    tasks: list[dict[str, Any]],
    hidden_size: int,
    intermediate_size: int,
    n_tile: int,
    w13_split: bool,
    inherited_weight_window_bytes: int,
    color_max_gflops: float,
) -> dict[str, Any]:
    """Attach logical per-core GEMM work and measured throughput to each segment."""
    if color_max_gflops <= 0:
        raise ValueError(f"color_max_gflops must be positive, got {color_max_gflops}")
    task_by_id = {int(task["task"]): task for task in tasks}
    observed_team_sizes: dict[tuple[int, str], int] = {}
    for segments in actual["cores"].values():
        for segment in segments:
            kind = stage_kind(str(segment["stage"]))
            if kind not in _GEMM_STAGE_KINDS:
                continue
            task_id = int(segment["task"])
            task = task_by_id.get(task_id)
            if task is None:
                continue
            local_tid = int(segment["local_tid"])
            key = task_id, kind
            observed_team_sizes[key] = max(
                observed_team_sizes.get(key, int(task["threads"])),
                local_tid + 1,
            )

    observed_gflops: list[float] = []
    for segments in actual["cores"].values():
        for segment in segments:
            kind = stage_kind(str(segment["stage"]))
            if kind not in _GEMM_STAGE_KINDS:
                continue
            task_id = int(segment["task"])
            task = task_by_id.get(task_id)
            if task is None:
                continue
            if kind == "w13":
                k = hidden_size
                n = 2 * intermediate_size
                fallback_ranges = 2 if w13_split else 1
            else:
                k = intermediate_size
                n = hidden_size
                fallback_ranges = 1
            configured_window = int(task.get(f"{kind}_window_bytes", -1))
            resolved_window = inherited_weight_window_bytes if configured_window < 0 else configured_window
            threads = observed_team_sizes[task_id, kind]
            local_tid = int(segment["local_tid"])
            n_columns = _owned_n_columns(
                n=n,
                k=k,
                n_tile=n_tile,
                threads=threads,
                local_tid=local_tid,
                weight_window_bytes=resolved_window,
                fallback_ranges=fallback_ranges,
            )
            rows = int(task["routes"])
            logical_flops = 2 * rows * k * n_columns
            duration_ms = float(segment["end_ms"]) - float(segment["start_ms"])
            gflops = logical_flops / (duration_ms * 1.0e6) if duration_ms > 0 else 0.0
            segment["n_columns"] = n_columns
            segment["logical_flops"] = logical_flops
            segment["gflops"] = gflops
            observed_gflops.append(gflops)

    summary = {
        "definition": "logical GEMM FLOPs assigned to one core divided by traced stage duration",
        "color_min_gflops": 0.0,
        "color_max_gflops": float(color_max_gflops),
        "observed_min_gflops": min(observed_gflops, default=0.0),
        "observed_max_gflops": max(observed_gflops, default=0.0),
        "segments": len(observed_gflops),
    }
    actual["gemm_throughput"] = summary
    return summary


def _merged_intervals(
    segments: list[dict[str, Any]],
    start_ms: float,
    end_ms: float,
) -> list[tuple[float, float]]:
    clipped = sorted(
        (
            max(float(segment["start_ms"]), start_ms),
            min(float(segment["end_ms"]), end_ms),
        )
        for segment in segments
        if float(segment["end_ms"]) > start_ms and float(segment["start_ms"]) < end_ms
    )
    merged: list[tuple[float, float]] = []
    for interval_start, interval_end in clipped:
        if interval_end <= interval_start:
            continue
        if merged and interval_start <= merged[-1][1] + 1.0e-9:
            merged[-1] = merged[-1][0], max(merged[-1][1], interval_end)
        else:
            merged.append((interval_start, interval_end))
    return merged


def _idle_gaps(
    segments: list[dict[str, Any]],
    start_ms: float,
    end_ms: float,
) -> list[dict[str, float]]:
    if end_ms <= start_ms:
        return []
    gaps: list[dict[str, float]] = []
    cursor = start_ms
    for interval_start, interval_end in _merged_intervals(segments, start_ms, end_ms):
        if interval_start > cursor + 1.0e-9:
            gaps.append(
                {
                    "start_ms": cursor,
                    "end_ms": interval_start,
                    "duration_ms": interval_start - cursor,
                }
            )
        cursor = max(cursor, interval_end)
    if cursor < end_ms - 1.0e-9:
        gaps.append(
            {
                "start_ms": cursor,
                "end_ms": end_ms,
                "duration_ms": end_ms - cursor,
            }
        )
    return gaps


def compute_idle_metrics(actual: dict[str, Any], *, cores: int) -> dict[str, Any]:
    """Split visible per-core idle time into internal gaps and end-of-compute tail."""
    if cores <= 0:
        raise ValueError(f"cores must be positive, got {cores}")
    host_compute = [segment for segment in actual.get("host_segments", []) if segment["stage"] == "scheduled_compute"]
    compute_segments = [
        segment
        for segments in actual["cores"].values()
        for segment in segments
        if stage_kind(str(segment["stage"])) in _COMPUTE_STAGE_KINDS
    ]
    if not compute_segments:
        raise ValueError("actual timeline has no per-core expert-compute segments")
    earliest_compute_ms = min(float(segment["start_ms"]) for segment in compute_segments)
    window_start_ms = (
        min(float(host_compute[-1]["start_ms"]), earliest_compute_ms) if host_compute else earliest_compute_ms
    )
    window_end_ms = max(float(segment["end_ms"]) for segment in compute_segments)
    window_ms = max(window_end_ms - window_start_ms, 0.0)

    per_core: dict[str, dict[str, Any]] = {}
    internal_idle_core_ms = 0.0
    tail_idle_core_ms = 0.0
    tail_wait_core_ms = 0.0
    for core in range(cores):
        segments = actual["cores"].get(str(core), [])
        core_compute = [segment for segment in segments if stage_kind(str(segment["stage"])) in _COMPUTE_STAGE_KINDS]
        compute_end_ms = max(
            (float(segment["end_ms"]) for segment in core_compute),
            default=window_start_ms,
        )
        internal_gaps = _idle_gaps(segments, window_start_ms, compute_end_ms)
        tail_gaps = _idle_gaps(segments, compute_end_ms, window_end_ms)
        internal_idle_ms = sum(gap["duration_ms"] for gap in internal_gaps)
        tail_idle_ms = sum(gap["duration_ms"] for gap in tail_gaps)
        tail_wait_ms = max(window_end_ms - compute_end_ms, 0.0)
        internal_idle_core_ms += internal_idle_ms
        tail_idle_core_ms += tail_idle_ms
        tail_wait_core_ms += tail_wait_ms
        per_core[str(core)] = {
            "compute_end_ms": compute_end_ms,
            "internal_idle_ms": internal_idle_ms,
            "tail_wait_ms": tail_wait_ms,
            "tail_idle_ms": tail_idle_ms,
            "tail_active_ms": max(tail_wait_ms - tail_idle_ms, 0.0),
            "internal_gaps": internal_gaps,
            "tail_gaps": tail_gaps,
        }

    capacity_core_ms = cores * window_ms
    metrics = {
        "definition": {
            "internal": "untraced gaps before each core's final expert-compute segment ends",
            "tail": "untraced gaps after that core ends expert compute and before the last core ends",
            "tail_wait": "raw completion skew; traced merge work inside this span is not idle",
        },
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "window_ms": window_ms,
        "capacity_core_ms": capacity_core_ms,
        "internal_idle_core_ms": internal_idle_core_ms,
        "internal_idle_mean_ms": internal_idle_core_ms / cores,
        "internal_idle_pct": 100.0 * internal_idle_core_ms / capacity_core_ms if capacity_core_ms else 0.0,
        "tail_wait_core_ms": tail_wait_core_ms,
        "tail_idle_core_ms": tail_idle_core_ms,
        "tail_idle_mean_ms": tail_idle_core_ms / cores,
        "tail_idle_pct": 100.0 * tail_idle_core_ms / capacity_core_ms if capacity_core_ms else 0.0,
        "per_core": per_core,
    }
    actual["idle_metrics"] = metrics
    return metrics


def enrich_actual_timeline(
    payload: dict[str, Any],
    *,
    gflops_color_max: float | None = None,
) -> None:
    actual = payload.get("actual")
    if actual is None:
        return
    case = payload["case"]
    plan = payload["plan"]
    previous_scale = actual.get("gemm_throughput", {}).get("color_max_gflops")
    color_max = (
        float(gflops_color_max) if gflops_color_max is not None else float(previous_scale or DEFAULT_GFLOPS_COLOR_MAX)
    )
    annotate_gemm_throughput(
        actual,
        tasks=plan["tasks"],
        hidden_size=int(case["hidden_size"]),
        intermediate_size=int(case["intermediate_size"]),
        n_tile=int(case.get("backend_n_tile", 8)),
        w13_split=bool(plan["w13_split"]),
        inherited_weight_window_bytes=int(plan["weight_window_bytes"]),
        color_max_gflops=color_max,
    )
    compute_idle_metrics(actual, cores=len(case["cpu_ids"]))
