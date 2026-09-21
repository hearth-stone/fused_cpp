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
    # Measured full-load T(window) / T(full stripe) per width. A width without an
    # entry scales by 1, i.e. the cost model's full-stripe time is kept.
    time_scales: Mapping[int, float] = field(default_factory=dict)

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
        for width, scale in self.time_scales.items():
            if width not in self.widths:
                raise ValueError(f"time scale width {width} is not in {self.widths}")
            if not 0.0 < float(scale) <= 1.0:
                raise ValueError(f"time scale for width {width} must be in (0, 1], got {scale}")

    def contains(self, routes: int) -> bool:
        return self.min_routes <= routes <= self.max_routes

    def select(self, threads: int) -> tuple[int, int] | None:
        if threads not in self.widths:
            return None
        return self.overrides.get(threads, (self.w13_tiles, self.w2_tiles))

    def time_scale(self, threads: int) -> float:
        if threads not in self.widths:
            return 1.0
        return float(self.time_scales.get(threads, 1.0))


@dataclass(frozen=True)
class StageWindowPolicy:
    """Bands searched in order; the first containing band wins."""

    name: str
    hidden_size: int
    intermediate_size: int
    backend_n_tile: int
    bands: tuple[StageWindowBand, ...]
    # Machines this table was measured on. Empty means "any machine with the shape", which is
    # how the tables predating the field are resolved; a table measured on one machine lists it.
    machine_ids: tuple[str, ...] = ()

    def select(self, routes: int, threads: int) -> tuple[int, int]:
        """`(w13_tiles, w2_tiles)`, or `(FULL_STRIPE, FULL_STRIPE)` when uncovered."""
        if routes <= 0 or threads <= 0:
            return (FULL_STRIPE, FULL_STRIPE)
        for band in self.bands:
            if band.contains(int(routes)):
                chosen = band.select(int(threads))
                return chosen if chosen is not None else (FULL_STRIPE, FULL_STRIPE)
        return (FULL_STRIPE, FULL_STRIPE)

    def time_scale(self, routes: int, threads: int) -> float:
        """Full-load window/full-stripe time ratio of the selected windows; 1 when uncovered."""
        if routes <= 0 or threads <= 0:
            return 1.0
        for band in self.bands:
            if band.contains(int(routes)):
                return band.time_scale(int(threads))
        return 1.0

    def matches_machine(self, machine_id: str | None) -> bool:
        return not self.machine_ids or (machine_id is not None and str(machine_id) in self.machine_ids)

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

# Superseded by V4 below and no longer registered; kept because the lab records and the
# frozen Lab parity value of the event model were produced with it.
# Measured for the calibrated machine (2026-09-19): Arm-codex NUMA3 80 cores (2 x 40-core LLC),
# TP4 H=4096 F=512, SVE BF16 n_tile 16 (W13 tile 128 KiB, W2 tile 16 KiB).
# Measured at full load (all lanes busy with the same width and window), producer-hot
# A and cold B, two sessions, with jemalloc preloaded and purging disabled
# (`tmp/jemalloc_rerun_20260919/decision.md`): the window table grid and the W2 sweep
# at W13 = 1 tile, composed as for V2 (`build_v3.py` -> `table_v3.json`). V1/V2 were
# measured under glibc, where the per-call route output page-faulted every call; that
# cost hid most large-M window gains and caused the old W2 large-M penalty. Windows are
# (W13 tiles, W2 tiles); `time_scales` is the measured full-load T(window)/T(full
# stripe) the planner multiplies into T(M, t). Routes 1-16 are DRAM bound and every
# window ties; above 720 there is no evidence, so both keep the full stripe. Widths
# outside (2, 4, 8, 16) are uncalibrated and keep the full stripe.
ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3 = StageWindowPolicy(
    name="arm_codex_numa3_80c_tp4_f512_n16_v3_full_load_jemalloc",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=16,
    machine_ids=("arm_codex_320c_numa3_80c_sve256_jemalloc_narrow_merge_v9",),
    bands=(
        StageWindowBand(
            min_routes=17,
            max_routes=33,
            w13_tiles=1,
            w2_tiles=8,
            widths=(2, 4, 8, 16),
            overrides={2: (1, 4)},
            time_scales={2: 0.6555, 4: 0.7317, 8: 0.8483, 16: 0.9191},
        ),
        StageWindowBand(
            min_routes=34,
            max_routes=67,
            w13_tiles=1,
            w2_tiles=8,
            widths=(2, 4, 8, 16),
            overrides={4: (1, 4), 16: (2, 16)},
            time_scales={2: 0.6061, 4: 0.748, 8: 0.8566, 16: 0.9687},
        ),
        StageWindowBand(
            min_routes=68,
            max_routes=117,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.6593, 4: 0.8169, 8: 0.8926},
        ),
        StageWindowBand(
            min_routes=118,
            max_routes=166,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.6999, 4: 0.8638, 8: 0.9323},
        ),
        StageWindowBand(
            min_routes=167,
            max_routes=235,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={8: (1, 8), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.7187, 4: 0.885, 8: 0.9457},
        ),
        StageWindowBand(
            min_routes=236,
            max_routes=371,
            w13_tiles=1,
            w2_tiles=FULL_STRIPE,
            widths=(2, 4, 8, 16),
            overrides={4: (1, 4), 8: (1, 8), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.7789, 4: 0.911, 8: 0.9578},
        ),
        StageWindowBand(
            min_routes=372,
            max_routes=587,
            w13_tiles=1,
            w2_tiles=FULL_STRIPE,
            widths=(2, 4, 8, 16),
            overrides={2: (1, 4), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.8123, 4: 0.9601, 8: 0.9727},
        ),
        StageWindowBand(
            min_routes=588,
            max_routes=720,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={4: (1, FULL_STRIPE), 8: (FULL_STRIPE, FULL_STRIPE), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.8345, 4: 0.9716},
        ),
    ),
)

# Registered for the calibrated machine (2026-09-21): the same grids re-measured after the
# stage window order became thread-major (`optimizations/fused_moe_sve/results/
# window_table_thread_major_20260921.md`), so table and kernel now share one order - V3's 2T
# rows had been measured under the previous one. Composition is unchanged (the window grid
# plus the W2 sweep at W13 = 1 tile, `build_v4.py`), and so are the adoption rules. On 18
# fresh layers V4 measured 0.27% faster than V3 with 16 of 18 layers better, inside the
# frozen tie band, so this replacement is a lineage decision, not a measured speedup; the
# two tables' 16 differing cells are ties the grid cannot separate from its own repeat
# (scales move by up to 0.046 between two runs of the same cells). Windows are (W13 tiles,
# W2 tiles); `time_scales` is the measured full-load T(window)/T(full stripe) the planner
# multiplies into T(M, t). Routes 1-16 tie on every window and above 720 there is no
# evidence, so both keep the full stripe; widths outside (2, 4, 8, 16) are uncalibrated.
ARM_CODEX_NUMA3_80C_TP4_F512_N16_V4 = StageWindowPolicy(
    name="arm_codex_numa3_80c_tp4_f512_n16_v4_thread_major_jemalloc",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=16,
    machine_ids=("arm_codex_320c_numa3_80c_sve256_jemalloc_narrow_merge_v9",),
    bands=(
        StageWindowBand(
            min_routes=17,
            max_routes=33,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={8: (1, FULL_STRIPE), 16: (1, 8)},
            time_scales={2: 0.6561, 4: 0.7324, 8: 0.9091, 16: 0.9181},
        ),
        StageWindowBand(
            min_routes=34,
            max_routes=67,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={16: (1, 8)},
            time_scales={2: 0.5992, 4: 0.7486, 8: 0.8439, 16: 0.9596},
        ),
        StageWindowBand(
            min_routes=68,
            max_routes=117,
            w13_tiles=1,
            w2_tiles=4,
            widths=(2, 4, 8, 16),
            overrides={4: (1, 8), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.6635, 4: 0.8253, 8: 0.9098},
        ),
        StageWindowBand(
            min_routes=118,
            max_routes=166,
            w13_tiles=1,
            w2_tiles=FULL_STRIPE,
            widths=(2, 4, 8, 16),
            overrides={4: (1, 4), 8: (1, 8), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.7426, 4: 0.8645, 8: 0.938},
        ),
        StageWindowBand(
            min_routes=167,
            max_routes=235,
            w13_tiles=1,
            w2_tiles=FULL_STRIPE,
            widths=(2, 4, 8, 16),
            overrides={4: (1, 4), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.7517, 4: 0.8866, 8: 0.9607},
        ),
        StageWindowBand(
            min_routes=236,
            max_routes=371,
            w13_tiles=1,
            w2_tiles=FULL_STRIPE,
            widths=(2, 4, 8, 16),
            overrides={4: (1, 4), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.7784, 4: 0.9107, 8: 0.9712},
        ),
        StageWindowBand(
            min_routes=372,
            max_routes=587,
            w13_tiles=1,
            w2_tiles=FULL_STRIPE,
            widths=(2, 4, 8, 16),
            overrides={4: (2, 16), 8: (1, 4), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.8396, 4: 0.9596, 8: 0.9623},
        ),
        StageWindowBand(
            min_routes=588,
            max_routes=720,
            w13_tiles=1,
            w2_tiles=8,
            widths=(2, 4, 8, 16),
            overrides={4: (1, FULL_STRIPE), 8: (1, 4), 16: (FULL_STRIPE, FULL_STRIPE)},
            time_scales={2: 0.8349, 4: 0.9752, 8: 0.9708},
        ),
    ),
)

_POLICIES: tuple[StageWindowPolicy, ...] = (
    AMAZON_C5_192C_TP4_F512_V5,
    ARM_CODEX_NUMA3_80C_TP4_F512_N16_V4,
)


def default_stage_window_policy(
    *, hidden_size: int, intermediate_size: int, backend_n_tile: int, machine_id: str | None = None
) -> StageWindowPolicy | None:
    """The calibrated policy for this shape and machine, or None when none was calibrated.

    A table measured on one machine (``machine_ids``) resolves only for that machine; tables
    without the field resolve on shape alone, as before.
    """
    for policy in _POLICIES:
        if policy.matches_shape(hidden_size, intermediate_size, backend_n_tile) and policy.matches_machine(machine_id):
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
    "ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3",
    "ARM_CODEX_NUMA3_80C_TP4_F512_N16_V4",
    "FULL_STRIPE",
    "StageWindowBand",
    "StageWindowPolicy",
    "default_stage_window_policy",
    "stage_geometry_name",
]
