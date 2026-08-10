"""Full-N packed-B geometry for production fused-MoE GEMM stages."""

from __future__ import annotations

from dataclasses import dataclass


MIN_SVE_N_TILE = 8


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class FullStageGeometry:
    """One full-N stage partitioned into one tile stripe per worker."""

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
