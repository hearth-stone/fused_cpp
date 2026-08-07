"""Packed-B weight-window geometry shared by profiling and planning."""

from __future__ import annotations

from dataclasses import dataclass


MIN_SVE_N_TILE = 8
INHERIT_WEIGHT_WINDOW = -1


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class WeightWindowGeometry:
    """Tile-aligned geometry of one sequential packed-B GEMM stage."""

    target_bytes: int
    n_tile: int
    total_tiles: int
    ranges: int
    max_range_tiles: int
    max_range_bytes: int

    def bytes_per_worker(self, threads: int) -> int:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        bytes_per_tile = self.max_range_bytes // self.max_range_tiles
        return _ceil_div(self.max_range_tiles, threads) * bytes_per_tile

    def active_threads(self, threads: int) -> int:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        return min(threads, self.max_range_tiles)


def stage_weight_window_geometry(
    *,
    k: int,
    n: int,
    n_tile: int,
    target_bytes: int,
    fallback_ranges: int = 1,
) -> WeightWindowGeometry:
    """Mirror native ``make_weight_window_plan`` without executing the kernel."""
    tile = max(int(n_tile), MIN_SVE_N_TILE)
    if k <= 0 or n <= 0 or tile <= 0:
        raise ValueError(f"weight-window GEMM dimensions must be positive: K={k}, N={n}, tile={tile}")
    if n % tile:
        raise ValueError(f"weight-window N must be tile aligned: N={n}, tile={tile}")
    if target_bytes < 0:
        raise ValueError(f"target_bytes must be non-negative, got {target_bytes}")

    total_tiles = n // tile
    ranges = max(1, min(int(fallback_ranges), total_tiles))
    bytes_per_tile = k * tile * 2
    if target_bytes > 0:
        max_tiles_per_window = max(1, target_bytes // bytes_per_tile)
        ranges = _ceil_div(total_tiles, max_tiles_per_window)
    max_range_tiles = _ceil_div(total_tiles, ranges)
    return WeightWindowGeometry(
        target_bytes=int(target_bytes),
        n_tile=tile,
        total_tiles=total_tiles,
        ranges=ranges,
        max_range_tiles=max_range_tiles,
        max_range_bytes=max_range_tiles * bytes_per_tile,
    )


def achievable_worker_windows(
    *,
    k: int,
    n: int,
    n_tile: int,
    threads: int,
) -> dict[int, tuple[int, int]]:
    """Return ``{bytes_per_worker: (ranges, max_range_bytes)}`` for one stage.

    Enumerating the range count covers every geometry the tile-aligned splitter
    can produce, so the keys are exactly the per-worker windows a policy may
    request. Each key keeps the entry with the fewest ranges, which is the
    largest range budget that reaches that window. ``ranges`` is normalised to
    the value ``stage_weight_window_geometry`` derives from ``max_range_bytes``,
    so feeding the returned budget back round-trips exactly.
    """
    tile = max(int(n_tile), MIN_SVE_N_TILE)
    if k <= 0 or n <= 0 or tile <= 0:
        raise ValueError(f"weight-window GEMM dimensions must be positive: K={k}, N={n}, tile={tile}")
    if n % tile:
        raise ValueError(f"weight-window N must be tile aligned: N={n}, tile={tile}")
    if threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}")

    total_tiles = n // tile
    bytes_per_tile = k * tile * 2
    windows: dict[int, tuple[int, int]] = {}
    for ranges in range(1, total_tiles + 1):
        max_range_tiles = _ceil_div(total_tiles, ranges)
        worker_bytes = _ceil_div(max_range_tiles, threads) * bytes_per_tile
        windows.setdefault(
            worker_bytes,
            (_ceil_div(total_tiles, max_range_tiles), max_range_tiles * bytes_per_tile),
        )
    return windows


def range_bytes_for_worker_window(
    *,
    k: int,
    n: int,
    n_tile: int,
    threads: int,
    target_worker_bytes: int,
) -> int:
    """Return the largest tile-aligned range budget that fits the per-worker window.

    The per-thread window is the measured invariant of packed-B reuse, while the
    kernel and the plan speak per-range bytes. This inverts
    ``WeightWindowGeometry.bytes_per_worker`` so a policy can be expressed in the
    invariant and lowered once, where the team width is known.
    """
    if target_worker_bytes <= 0:
        raise ValueError(f"target_worker_bytes must be positive, got {target_worker_bytes}")
    windows = achievable_worker_windows(k=k, n=n, n_tile=n_tile, threads=threads)
    fitting = [worker_bytes for worker_bytes in windows if worker_bytes <= target_worker_bytes]
    if not fitting:
        raise ValueError(
            f"no tile-aligned window fits {target_worker_bytes} bytes per worker for "
            f"K={k}, N={n}, tile={max(int(n_tile), MIN_SVE_N_TILE)}, threads={threads}: "
            f"the smallest achievable per-worker window is {min(windows)} bytes"
        )
    return windows[max(fitting)][1]


def fused_moe_weight_windows(
    *,
    hidden_size: int,
    intermediate_size: int,
    n_tile: int,
    target_bytes: int,
    w13_fallback_ranges: int,
) -> tuple[WeightWindowGeometry, WeightWindowGeometry]:
    """Return W13 and W2 window geometry for one fused expert."""
    return fused_moe_task_weight_windows(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        n_tile=n_tile,
        inherited_target_bytes=target_bytes,
        w13_target_bytes=INHERIT_WEIGHT_WINDOW,
        w2_target_bytes=INHERIT_WEIGHT_WINDOW,
        w13_fallback_ranges=w13_fallback_ranges,
    )


def fused_moe_task_weight_windows(
    *,
    hidden_size: int,
    intermediate_size: int,
    n_tile: int,
    inherited_target_bytes: int,
    w13_target_bytes: int,
    w2_target_bytes: int,
    w13_fallback_ranges: int,
) -> tuple[WeightWindowGeometry, WeightWindowGeometry]:
    """Resolve independent per-task stage targets against the global policy."""
    if inherited_target_bytes < 0:
        raise ValueError(f"inherited_target_bytes must be non-negative, got {inherited_target_bytes}")
    if w13_target_bytes < INHERIT_WEIGHT_WINDOW or w2_target_bytes < INHERIT_WEIGHT_WINDOW:
        raise ValueError("stage targets must be -1 or non-negative")
    resolved_w13 = inherited_target_bytes if w13_target_bytes == INHERIT_WEIGHT_WINDOW else w13_target_bytes
    resolved_w2 = inherited_target_bytes if w2_target_bytes == INHERIT_WEIGHT_WINDOW else w2_target_bytes
    w13 = stage_weight_window_geometry(
        k=hidden_size,
        n=2 * intermediate_size,
        n_tile=n_tile,
        target_bytes=resolved_w13,
        fallback_ranges=w13_fallback_ranges,
    )
    w2 = stage_weight_window_geometry(
        k=intermediate_size,
        n=hidden_size,
        n_tile=n_tile,
        target_bytes=resolved_w2,
    )
    return w13, w2
