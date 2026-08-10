"""Full-N packed-B geometry for production fused-MoE GEMM stages.

A stage's computation pattern is determined by the pair `(threads, window_tiles)`,
where `window_tiles` is the per-thread owner window measured in whole packed-B N
tiles. A tile is the smallest addressable unit because the packed layout is
tile-contiguous and the microkernel's output tile is `n_tile` wide, so a window is
always a whole number of tiles.

The team consumes `threads * window_tiles` tiles per window, and the stage is
covered by `ceil(total_tiles / range_tiles)` windows. The final window may be
short; its tiles are spread over the team the same way a full window's are.

Setting `window_tiles = tiles_per_worker(threads)` yields exactly one window, which
is the `full_n_team_stripes` geometry: every worker owns one contiguous stripe and
traverses it once. That is the default and this module's `window_plan` reproduces
it bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass


MIN_SVE_N_TILE = 8


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _split_evenly(units: int, group_size: int, local_tid: int) -> tuple[int, int]:
    """Mirror of the native `split_evenly`: leading threads absorb the remainder."""
    if units <= 0 or group_size <= 0 or local_tid < 0 or local_tid >= group_size:
        return (0, 0)
    per_thread, extra = divmod(units, group_size)
    if local_tid < extra:
        return (local_tid * (per_thread + 1), per_thread + 1)
    begin = extra * (per_thread + 1) + (local_tid - extra) * per_thread
    return (begin, per_thread)


@dataclass(frozen=True)
class ThreadWindowRange:
    """One worker's tile range inside one window."""

    begin_tile: int
    tiles: int

    @property
    def is_empty(self) -> bool:
        return self.tiles <= 0


@dataclass(frozen=True)
class StageWindowPlan:
    """The unique computation pattern implied by `(threads, window_tiles)`."""

    n_tile: int
    total_tiles: int
    bytes_per_tile: int
    threads: int
    window_tiles: int

    @property
    def range_tiles(self) -> int:
        """Tiles the whole team consumes per window."""
        return self.threads * self.window_tiles

    @property
    def windows(self) -> int:
        """`R`, the number of windows covering the stage."""
        return _ceil_div(self.total_tiles, self.range_tiles)

    @property
    def bytes_per_worker(self) -> int:
        return self.window_tiles * self.bytes_per_tile

    def window_tile_span(self, window_index: int) -> tuple[int, int]:
        """`(begin_tile, tiles)` for one window; the last one may be short."""
        if window_index < 0 or window_index >= self.windows:
            raise IndexError(f"window index out of range: {window_index} of {self.windows}")
        begin = window_index * self.range_tiles
        return (begin, min(self.range_tiles, self.total_tiles - begin))

    def thread_range(self, window_index: int, local_tid: int) -> ThreadWindowRange:
        window_begin, window_tiles = self.window_tile_span(window_index)
        offset, tiles = _split_evenly(window_tiles, self.threads, local_tid)
        return ThreadWindowRange(begin_tile=window_begin + offset, tiles=tiles)

    def idle_threads(self, window_index: int) -> int:
        """Workers that receive no tiles in this window."""
        return sum(1 for tid in range(self.threads) if self.thread_range(window_index, tid).is_empty)

    def starves_any_thread(self) -> bool:
        """True when some window leaves a worker with no work.

        Only reachable through a short tail window: a full window holds
        `threads * window_tiles >= threads` tiles. When `threads` divides
        `total_tiles` the tail is a multiple of `threads`, so this is always False.
        """
        return any(self.idle_threads(index) > 0 for index in range(self.windows))


@dataclass(frozen=True)
class FullStageGeometry:
    """One full-N stage partitioned into tile windows per worker."""

    n_tile: int
    total_tiles: int
    bytes_per_tile: int

    @property
    def stage_bytes(self) -> int:
        return self.total_tiles * self.bytes_per_tile

    def tiles_per_worker(self, threads: int) -> int:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        return _ceil_div(self.total_tiles, threads)

    def bytes_per_worker(self, threads: int) -> int:
        return self.tiles_per_worker(threads) * self.bytes_per_tile

    def active_threads(self, threads: int) -> int:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        return min(threads, self.total_tiles)

    def window_tiles_from_bytes(self, window_bytes: int) -> int:
        """Convert a byte budget to whole tiles.

        This is the only place a byte-denominated window may be interpreted. The
        runtime ABI carries tile counts, so policies that reason in bytes convert
        here and nowhere else. A budget below one tile clamps to one tile, because a
        tile cannot be subdivided, and a budget above the stage clamps to the stage.
        """
        if window_bytes <= 0:
            raise ValueError(f"window bytes must be positive, got {window_bytes}")
        tiles = int(window_bytes) // self.bytes_per_tile
        return max(1, min(tiles, self.total_tiles))

    def window_plan(self, threads: int, window_tiles: int) -> StageWindowPlan:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        if window_tiles <= 0:
            raise ValueError(f"window tiles must be positive, got {window_tiles}")
        if window_tiles > self.total_tiles:
            raise ValueError(f"window tiles {window_tiles} exceeds stage tiles {self.total_tiles}")
        return StageWindowPlan(
            n_tile=self.n_tile,
            total_tiles=self.total_tiles,
            bytes_per_tile=self.bytes_per_tile,
            threads=threads,
            window_tiles=window_tiles,
        )

    def full_stripe_plan(self, threads: int) -> StageWindowPlan:
        """The `R = 1` endpoint, identical to `full_n_team_stripes`."""
        return self.window_plan(threads, self.tiles_per_worker(threads))


def full_stage_geometry(*, k: int, n: int, n_tile: int) -> FullStageGeometry:
    tile = max(int(n_tile), MIN_SVE_N_TILE)
    if k <= 0 or n <= 0:
        raise ValueError(f"full-stage GEMM dimensions must be positive: K={k}, N={n}")
    if n % tile:
        raise ValueError(f"full-stage N must be tile aligned: N={n}, tile={tile}")
    return FullStageGeometry(
        n_tile=tile,
        total_tiles=n // tile,
        bytes_per_tile=k * tile * 2,
    )
