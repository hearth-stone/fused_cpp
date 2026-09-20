"""Full-N packed-B geometry for production fused-MoE GEMM stages.

A stage's computation pattern is determined by the pair `(threads, window_tiles)`,
where `window_tiles` is the per-thread owner window measured in whole packed-B N
tiles. A tile is the smallest addressable unit because the packed layout is
tile-contiguous and the microkernel's output tile is `n_tile` wide, so a window is
always a whole number of tiles.

Order (2026-09-20): ownership first, windows second. The stage's tiles are split
across the team once, so every worker owns one contiguous stripe of
`total_tiles / threads` tiles, and the worker then walks its own stripe in windows
of `window_tiles` tiles. A worker therefore always touches one contiguous region of
packed B, and the last window of a stripe is the only short one. The team still
consumes `threads * window_tiles` tiles per pass, so the per-worker L2 footprint and
the number of passes over A are the same as in the window-first order this replaces.

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
        """`R`, the number of window passes; the longest stripe sets it."""
        longest = max((self.thread_stripe(tid)[1] for tid in range(self.threads)), default=0)
        return _ceil_div(longest, self.window_tiles) if longest > 0 else 0

    @property
    def bytes_per_worker(self) -> int:
        return self.window_tiles * self.bytes_per_tile

    def thread_stripe(self, local_tid: int) -> tuple[int, int]:
        """`(begin_tile, tiles)` of the contiguous stripe this worker owns."""
        return _split_evenly(self.total_tiles, self.threads, local_tid)

    def window_tile_span(self, window_index: int) -> tuple[int, int]:
        """`(begin_tile, tiles)` covered by the team in one pass.

        The pass is the union of every worker's `window_index`-th window, which is not
        contiguous once the stage is wider than one pass; the span reports its hull, and
        `thread_range` is what the kernel iterates.
        """
        if window_index < 0 or window_index >= self.windows:
            raise IndexError(f"window index out of range: {window_index} of {self.windows}")
        ranges = [self.thread_range(window_index, tid) for tid in range(self.threads)]
        active = [item for item in ranges if not item.is_empty]
        if not active:
            return (0, 0)
        begin = min(item.begin_tile for item in active)
        end = max(item.begin_tile + item.tiles for item in active)
        return (begin, end - begin)

    def thread_range(self, window_index: int, local_tid: int) -> ThreadWindowRange:
        stripe_begin, stripe_tiles = self.thread_stripe(local_tid)
        begin = stripe_begin + window_index * self.window_tiles
        remaining = stripe_begin + stripe_tiles - begin
        return ThreadWindowRange(begin_tile=begin, tiles=max(0, min(self.window_tiles, remaining)))

    def idle_threads(self, window_index: int) -> int:
        """Workers that receive no tiles in this window."""
        return sum(1 for tid in range(self.threads) if self.thread_range(window_index, tid).is_empty)

    def starves_any_thread(self) -> bool:
        """True when some pass leaves a worker with no work.

        With ownership first this only happens when a worker's stripe is shorter than
        the longest one by a whole window, or when there are more workers than tiles.
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
