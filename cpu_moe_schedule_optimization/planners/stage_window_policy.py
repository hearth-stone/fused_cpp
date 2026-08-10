"""Per-thread stage-window policy, denominated in whole packed-B N tiles.

A stage's computation pattern is determined by `(threads, window_tiles)`; see
`cost_model/full_stage_geometry.py` for the derivation and the runtime mirror. This
module holds the calibrated choice of `window_tiles` per stage.

The policy is a band table over the route count, with per-width overrides. It is a
deterministic function of the already-selected `(routes, threads)`, so it adds no
planner search dimension: the planner picks a width, then reads the windows off.

`FULL_STRIPE` (0) means one window per worker, which is the `full_n_team_stripes`
geometry. Route counts outside every band fall back to it, so an uncalibrated shape
keeps the pre-window behaviour rather than inheriting a guess.

Windows are tile counts, not bytes. The runtime ABI carries tiles and
`FullStageGeometry.window_tiles_from_bytes` is the only sanctioned byte conversion,
so a byte-denominated table cannot be mapped back to a unique pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


FULL_STRIPE = 0

# TP4 H=4096 F=512 with backend_n_tile=8: a W13 tile is 2*4096*8 = 64 KiB and a W2
# tile is 2*512*8 = 8 KiB, so W13 holds 128 tiles and W2 holds 512.
_W13_TILE_BYTES = 64 * 1024
_W2_TILE_BYTES = 8 * 1024


@dataclass(frozen=True)
class StageWindowBand:
    """One route band. `overrides` maps a team width to its own (w13, w2) tiles.

    A stage's entry may be `FULL_STRIPE` to leave that stage unwindowed while the
    other one is windowed, which is what an override calibrated on only one stage
    should say rather than guessing the other.
    """

    min_routes: int
    max_routes: int
    w13_tiles: int
    w2_tiles: int
    widths: tuple[int, ...] = (1, 2, 4, 8)
    overrides: Mapping[int, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_routes <= 0 or self.max_routes < self.min_routes:
            raise ValueError(f"invalid band range [{self.min_routes}, {self.max_routes}]")
        if self.w13_tiles < 0 or self.w2_tiles < 0:
            raise ValueError("band windows must be non-negative tile counts")
        if self.w13_tiles == FULL_STRIPE and self.w2_tiles == FULL_STRIPE:
            raise ValueError("a band that windows neither stage should be omitted instead")
        for width, (w13, w2) in self.overrides.items():
            if width not in self.widths:
                raise ValueError(f"override width {width} is not in {self.widths}")
            if w13 < 0 or w2 < 0:
                raise ValueError(f"override for width {width} must be non-negative tile counts")

    def contains(self, routes: int) -> bool:
        return self.min_routes <= routes <= self.max_routes

    def select(self, threads: int) -> tuple[int, int] | None:
        if threads not in self.widths:
            return None
        return self.overrides.get(threads, (self.w13_tiles, self.w2_tiles))


@dataclass(frozen=True)
class StageWindowPolicy:
    """Bands searched in order; the first containing band wins."""

    name: str
    hidden_size: int
    intermediate_size: int
    backend_n_tile: int
    bands: tuple[StageWindowBand, ...]

    def select(self, routes: int, threads: int) -> tuple[int, int]:
        """`(w13_tiles, w2_tiles)`, or `(FULL_STRIPE, FULL_STRIPE)` when uncovered."""
        if routes <= 0 or threads <= 0:
            return (FULL_STRIPE, FULL_STRIPE)
        for band in self.bands:
            if band.contains(int(routes)):
                chosen = band.select(int(threads))
                return chosen if chosen is not None else (FULL_STRIPE, FULL_STRIPE)
        return (FULL_STRIPE, FULL_STRIPE)

    def matches_shape(self, hidden_size: int, intermediate_size: int, backend_n_tile: int) -> bool:
        return (
            int(hidden_size) == self.hidden_size
            and int(intermediate_size) == self.intermediate_size
            and int(backend_n_tile) == self.backend_n_tile
        )


# Calibrated on AmazonC5192Cores NUMA0, TP4 H=4096 F=512, SVE JIT exact-M, 32 MiB
# HugeTLB. The 1/2/4/8-thread values are the V4 empirical table converted from
# per-thread bytes to tiles; `results/amazon_192c_stage_window_tiles_20260810.md`
# re-measured the overlapping cells on the tile-counted runtime and reproduced its
# choices.
#
# The 16-thread entries were measured directly, W13 only, and stop at route 143:
# a 16-thread stripe is 8 tiles and windowing it is worth 15.2/11.2/9.2/4.1% at
# routes 48/72/96/120, but at routes 144 and above the full stripe wins and a
# 1-tile window costs 7-32%. That is the large-M regime, where A no longer fits and
# a wide window is what amortizes its rescans. Their W2 entry is FULL_STRIPE because
# only the W13 axis was swept there.
#
# 32 threads is deliberately absent: its stripe is already 4 tiles and every window
# landed within 0.6% of it, inside the noise floor.
AMAZON_C5_192C_TP4_F512_V5 = StageWindowPolicy(
    name="amazon_c5_192c_tp4_f512_v5_tiles",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=8,
    bands=(
        StageWindowBand(
            min_routes=13,
            max_routes=48,
            w13_tiles=4,
            w2_tiles=32,
            widths=(1, 2, 4, 8, 16),
            overrides={8: (2, 16), 16: (1, FULL_STRIPE)},
        ),
        StageWindowBand(
            min_routes=49,
            max_routes=95,
            w13_tiles=2,
            w2_tiles=16,
            widths=(1, 2, 4, 8, 16),
            overrides={4: (1, 8), 16: (1, FULL_STRIPE)},
        ),
        StageWindowBand(
            min_routes=96,
            max_routes=143,
            w13_tiles=2,
            w2_tiles=16,
            widths=(1, 2, 4, 8, 16),
            overrides={2: (1, 16), 4: (1, 16), 8: (2, 8), 16: (2, FULL_STRIPE)},
        ),
        StageWindowBand(min_routes=144, max_routes=215, w13_tiles=2, w2_tiles=16, overrides={8: (2, 8)}),
        StageWindowBand(min_routes=216, max_routes=287, w13_tiles=8, w2_tiles=16),
        StageWindowBand(min_routes=288, max_routes=575, w13_tiles=8, w2_tiles=16, overrides={1: (16, 64)}),
    ),
)

_POLICIES: tuple[StageWindowPolicy, ...] = (AMAZON_C5_192C_TP4_F512_V5,)


def default_stage_window_policy(
    *, hidden_size: int, intermediate_size: int, backend_n_tile: int
) -> StageWindowPolicy | None:
    """The calibrated policy for this shape, or None when none was calibrated."""
    for policy in _POLICIES:
        if policy.matches_shape(hidden_size, intermediate_size, backend_n_tile):
            return policy
    return None


def stage_geometry_name(w13_windows: Sequence[int], w2_windows: Sequence[int]) -> str:
    """Report name for a plan's windows.

    An all-full-stripe plan is exactly the pre-window geometry and keeps reporting
    `full_n_team_stripes`, so profiles calibrated under that name stay valid.
    """
    if all(int(w) == FULL_STRIPE for w in w13_windows) and all(int(w) == FULL_STRIPE for w in w2_windows):
        return "full_n_team_stripes"
    return "windowed_team_stripes"


__all__ = [
    "AMAZON_C5_192C_TP4_F512_V5",
    "FULL_STRIPE",
    "StageWindowBand",
    "StageWindowPolicy",
    "default_stage_window_policy",
    "stage_geometry_name",
]
