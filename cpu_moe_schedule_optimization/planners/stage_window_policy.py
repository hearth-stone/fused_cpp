"""Profile-bound per-task packed-B window policies for MoE plans."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from weight_window import range_bytes_for_worker_window  # noqa: E402


INHERIT_STAGE_WINDOW = -1
MIB = 1024 * 1024


class TaskStageWindowPolicy(Protocol):
    name: str

    def select(self, routes: int, threads: int) -> tuple[int, int]: ...

    def cost_model_entries(self) -> tuple["StageWindowPolicyEntry", ...]: ...


class StageWindowProfilePolicy(Protocol):
    mode: str
    degree: int
    hidden_size: int
    intermediate_size: int
    global_experts: int
    local_experts: int
    backend: str
    backend_n_tile: int
    sve_implementation: str
    m_tail_policy: str
    activation: str
    dtype: str
    measurement_experts: int
    cores_per_rank: int
    concurrent_ranks: int
    llc_bytes_per_rank: int
    numa_nodes: tuple[int, ...]
    cpu_ids_by_rank: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class StageWindowBand:
    """One route band of a stage-window policy.

    The measured invariant of packed-B reuse is the per-thread window
    ``omega = range_bytes / threads``, not the per-range budget: at equal
    ``omega`` the calibrated widths agree within a few percent, while a fixed
    per-range budget spans the full width ratio. A band is therefore normally
    expressed with ``w13_bytes_per_thread`` / ``w2_bytes_per_thread`` and lowered
    to per-range budgets by the owning policy, which holds the stage geometry.
    ``literal_windows`` carries per-range budgets directly, which is the
    historical form and the only one available without geometry.

    ``widths`` is the explicit coverage set. Widths outside it inherit the
    operator-wide window, which also keeps the cost model's full-workload anchor
    eligible for them, so widening coverage is always a deliberate change that
    needs its own measurement.
    """

    min_routes: int
    max_routes: int
    widths: tuple[int, ...]
    w13_bytes_per_thread: int = 0
    w2_bytes_per_thread: int = 0
    thread_overrides: tuple[tuple[int, int, int], ...] = ()
    literal_windows: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.min_routes <= 0 or self.max_routes < self.min_routes:
            raise ValueError("route band must be positive and non-empty")
        if not self.widths:
            raise ValueError("route band must cover at least one thread width")
        if any(threads <= 0 for threads in self.widths):
            raise ValueError("thread widths must be positive")
        if len(self.widths) != len(set(self.widths)):
            raise ValueError("thread widths must be unique within a route band")

        requested = (self.w13_bytes_per_thread, self.w2_bytes_per_thread)
        if min(requested) < 0:
            raise ValueError("per-thread stage windows must be non-negative")
        if max(requested) > 0 and min(requested) == 0:
            raise ValueError("per-thread route bands must set both the W13 and the W2 window")
        if (max(requested) > 0) == bool(self.literal_windows):
            raise ValueError(
                "a route band must carry either positive per-thread windows or literal "
                "per-range windows, not both and not neither"
            )

        if self.per_thread:
            override_widths = [threads for threads, _, _ in self.thread_overrides]
            if len(override_widths) != len(set(override_widths)):
                raise ValueError("thread overrides must be unique within a route band")
            unknown = sorted(set(override_widths) - set(self.widths))
            if unknown:
                raise ValueError(f"thread overrides must target covered widths, got {unknown}")
            if any(min(w13_bytes, w2_bytes) <= 0 for _, w13_bytes, w2_bytes in self.thread_overrides):
                raise ValueError("per-thread stage window overrides must be positive")
            return

        if self.thread_overrides:
            raise ValueError("literal per-range route bands do not support thread overrides")
        literal_widths = [threads for threads, _, _ in self.literal_windows]
        if len(literal_widths) != len(set(literal_widths)):
            raise ValueError("thread widths must be unique within a route band")
        if set(literal_widths) != set(self.widths):
            raise ValueError("literal per-range windows must cover exactly the band's widths")
        if any(min(w13_bytes, w2_bytes) < 0 for _, w13_bytes, w2_bytes in self.literal_windows):
            raise ValueError("explicit stage windows must be non-negative")

    @classmethod
    def from_thread_windows(
        cls,
        min_routes: int,
        max_routes: int,
        thread_windows: Sequence[tuple[int, int, int]],
    ) -> StageWindowBand:
        """Build a band from the historical ``(threads, w13, w2)`` per-range form."""
        windows = tuple((int(threads), int(w13), int(w2)) for threads, w13, w2 in thread_windows)
        return cls(
            min_routes=int(min_routes),
            max_routes=int(max_routes),
            widths=tuple(threads for threads, _, _ in windows),
            literal_windows=windows,
        )

    @property
    def per_thread(self) -> bool:
        """True when the band is expressed in per-thread windows."""
        return not self.literal_windows

    def covers(self, routes: int) -> bool:
        return self.min_routes <= int(routes) <= self.max_routes

    def target_windows(self, threads: int) -> tuple[int, int] | None:
        """Return this band's request for ``threads``, in the band's own unit."""
        threads = int(threads)
        if threads not in self.widths:
            return None
        if not self.per_thread:
            for candidate, w13_bytes, w2_bytes in self.literal_windows:
                if candidate == threads:
                    return w13_bytes, w2_bytes
            return None
        for candidate, w13_bytes, w2_bytes in self.thread_overrides:
            if candidate == threads:
                return w13_bytes, w2_bytes
        return self.w13_bytes_per_thread, self.w2_bytes_per_thread


@dataclass(frozen=True)
class StageWindowPolicyEntry:
    min_routes: int
    max_routes: int
    threads: int
    w13_window_bytes: int
    w2_window_bytes: int

    def __post_init__(self) -> None:
        if self.min_routes <= 0 or self.max_routes < self.min_routes:
            raise ValueError("route band must be positive and non-empty")
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if min(self.w13_window_bytes, self.w2_window_bytes) < 0:
            raise ValueError("cost-model stage windows must be non-negative")


@dataclass(frozen=True)
class StaticStageWindowPolicy:
    """A finite route-band table, lowered to per-range budgets at construction.

    ``hidden_size`` / ``intermediate_size`` / ``backend_n_tile`` describe the
    stage geometry needed to lower per-thread windows. They are required only
    when some band is expressed that way, and must match the bound profile.
    """

    name: str
    bands: tuple[StageWindowBand, ...]
    hidden_size: int = 0
    intermediate_size: int = 0
    backend_n_tile: int = 0
    _lowered: tuple[dict[int, tuple[int, int]], ...] = field(
        init=False,
        repr=False,
        compare=False,
        default=(),
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage-window policy name must be non-empty")
        ordered = sorted(self.bands, key=lambda band: band.min_routes)
        if tuple(ordered) != self.bands:
            raise ValueError("stage-window route bands must be sorted")
        if any(left.max_routes >= right.min_routes for left, right in zip(ordered, ordered[1:])):
            raise ValueError("stage-window route bands must not overlap")
        object.__setattr__(self, "_lowered", self._lower_bands())

    def _lower_bands(self) -> tuple[dict[int, tuple[int, int]], ...]:
        geometry = (self.hidden_size, self.intermediate_size, self.backend_n_tile)
        if any(band.per_thread for band in self.bands) and min(geometry) <= 0:
            raise ValueError(
                f"stage-window policy {self.name!r} has per-thread route bands, which require "
                "hidden_size, intermediate_size and backend_n_tile"
            )
        lowered = []
        for band in self.bands:
            resolved: dict[int, tuple[int, int]] = {}
            for threads in band.widths:
                targets = band.target_windows(threads)
                if targets is None:
                    raise ValueError(f"route band {band.min_routes}-{band.max_routes} lacks width {threads}")
                resolved[threads] = self._lower(band, threads, targets) if band.per_thread else targets
            lowered.append(resolved)
        return tuple(lowered)

    def _lower(self, band: StageWindowBand, threads: int, targets: tuple[int, int]) -> tuple[int, int]:
        """Invert ``bytes_per_worker`` for both stages at one width."""
        stages = (
            ("w13", self.hidden_size, 2 * self.intermediate_size, targets[0]),
            ("w2", self.intermediate_size, self.hidden_size, targets[1]),
        )
        resolved = []
        for stage, k, n, target_worker_bytes in stages:
            try:
                resolved.append(
                    range_bytes_for_worker_window(
                        k=k,
                        n=n,
                        n_tile=self.backend_n_tile,
                        threads=threads,
                        target_worker_bytes=target_worker_bytes,
                    )
                )
            except ValueError as error:
                raise ValueError(
                    f"stage-window policy {self.name!r} cannot lower {stage} for routes "
                    f"{band.min_routes}-{band.max_routes} at {threads} threads: {error}"
                ) from error
        return resolved[0], resolved[1]

    def select(self, routes: int, threads: int) -> tuple[int, int]:
        routes = int(routes)
        threads = int(threads)
        if routes <= 0 or threads <= 0:
            raise ValueError("routes and threads must be positive")
        for band, resolved in zip(self.bands, self._lowered):
            if not band.covers(routes):
                continue
            selected = resolved.get(threads)
            if selected is not None:
                return selected
        return INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW

    def worker_windows(self, routes: int, threads: int) -> tuple[int, int]:
        """Return the achieved per-thread windows, or ``-1`` where inherited."""
        w13_bytes, w2_bytes = self.select(routes, threads)
        if (w13_bytes, w2_bytes) == (INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW):
            return INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW
        tile_bytes = (self.hidden_size * self.backend_n_tile * 2, self.intermediate_size * self.backend_n_tile * 2)
        return tuple(  # type: ignore[return-value]
            -(-(range_bytes // tile) // int(threads)) * tile
            for range_bytes, tile in zip((w13_bytes, w2_bytes), tile_bytes)
        )

    def cost_model_entries(self) -> tuple[StageWindowPolicyEntry, ...]:
        """Return the finite deterministic mapping consumed by native planning."""
        return tuple(
            StageWindowPolicyEntry(
                min_routes=band.min_routes,
                max_routes=band.max_routes,
                threads=threads,
                w13_window_bytes=w13_bytes,
                w2_window_bytes=w2_bytes,
            )
            for band, resolved in zip(self.bands, self._lowered)
            for threads, (w13_bytes, w2_bytes) in sorted(resolved.items())
        )


# Calibrated on both 96-core NUMA ranks of AmazonC5192Cores for the TP4
# H=4096/F=512 SVE R13=2/R2=1 path. Each band carries one per-thread window per
# stage, which is the invariant the isolated sweeps measured: at equal per-thread
# window the four widths agree within a few percent, while the per-range budget
# they lower to spans up to 8x within a single band. The overrides are cells
# where the original per-cell search landed one factor-of-two step away; they are
# retained so this table reproduces the measured bytes exactly.
#
# ``widths`` is deliberately narrow. Unsupported route/width combinations inherit
# the profile's operator-wide window instead of being extrapolated, and that also
# keeps the cost model's full-workload anchor eligible for them.
AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1 = StaticStageWindowPolicy(
    name="amazon_c5_192c_tp4_f512_v1",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=8,
    bands=(
        StageWindowBand(
            min_routes=49,
            max_routes=95,
            widths=(8,),
            w13_bytes_per_thread=MIB // 8,
            w2_bytes_per_thread=MIB // 8,
        ),
        StageWindowBand(
            min_routes=96,
            max_routes=143,
            widths=(1, 2, 4, 8),
            w13_bytes_per_thread=MIB // 8,
            w2_bytes_per_thread=MIB // 8,
            thread_overrides=(
                (2, MIB // 16, MIB // 8),
                (4, MIB // 16, MIB // 8),
                (8, MIB // 8, MIB // 16),
            ),
        ),
        StageWindowBand(
            min_routes=144,
            max_routes=287,
            widths=(1, 2, 4, 8),
            w13_bytes_per_thread=MIB // 8,
            w2_bytes_per_thread=MIB // 8,
            thread_overrides=((8, MIB // 8, MIB // 16),),
        ),
        StageWindowBand(
            min_routes=288,
            max_routes=575,
            widths=(1, 2, 4, 8),
            w13_bytes_per_thread=MIB // 2,
            w2_bytes_per_thread=MIB // 8,
            thread_overrides=((1, 1 * MIB, MIB // 2),),
        ),
    ),
)

# Compatibility alias for the initial NUMA0-only experiment name.
AMAZON_C5_192C_NUMA0_TP4_F512_STAGE_WINDOWS_V1 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1


# Short-route band added in V2. V1 had no band below 49 routes, so every expert
# with M < 49 inherited the operator-wide R13=2 geometry: two 4 MiB W13
# ranges and one 4 MiB W2 range, which is 1 MiB per thread at 4T and 4 MiB at 1T.
# Isolated sweeps on this host show short-route experts lose a large share of
# their useful packed-B bandwidth there, because each additional M12 panel
# re-reads the whole window and only two panels' worth of reuse is available to
# amortise the fill.
AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND = StageWindowBand(
    min_routes=13,
    max_routes=48,
    widths=(1, 2, 4, 8),
    w13_bytes_per_thread=MIB // 4,
    w2_bytes_per_thread=MIB // 4,
    thread_overrides=((8, MIB // 8, MIB // 8),),
)

AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2 = StaticStageWindowPolicy(
    name="amazon_c5_192c_tp4_f512_v2",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=8,
    bands=(
        AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND,
        *AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1.bands,
    ),
)


# V1 calibrated the 49-95 band at 8 threads only, leaving narrower teams on the
# operator-wide legacy geometry. That geometry divides one 4 MiB range across the
# team, so its per-thread window is 4 MiB / t: near the measured optimum at 16 and
# 32 threads, but 32x to 8x too large at 1 to 4 threads. Filling those three cells
# is worth 1.65x to 3.39x of isolated useful packed-B bandwidth at M=72. The 8
# thread cell is left exactly as V1 calibrated it.
AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V3 = StageWindowBand(
    min_routes=49,
    max_routes=95,
    widths=(1, 2, 4, 8),
    w13_bytes_per_thread=MIB // 8,
    w2_bytes_per_thread=MIB // 8,
    thread_overrides=((4, MIB // 16, MIB // 16),),
)

AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V3 = StaticStageWindowPolicy(
    name="amazon_c5_192c_tp4_f512_v3",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=8,
    bands=(
        AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND,
        AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V3,
        *AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1.bands[1:],
    ),
)


# V1 gave 144-287 a single 1/8 MiB per-thread W13 window, but the optimum inside
# that range is not constant: the per-thread shared-A scan is 2*M*H bytes, so it
# outgrows the 2 MiB private L2 at M=256 and range multiplication stops being
# cheap. Measured on 192 homogeneous experts, the optimum steps up from 1/8 MiB
# to 1/2 MiB between M=200, where 1/8 still wins by 3.9%, and M=224, where 1/2
# wins by 1.6%; interpolating puts the crossover at M~217. The boundary is placed
# at 216, an integral number of M12 panels, which costs at most 0.2% at the seam.
#
# The step is width-independent, as the mechanism predicts since every thread
# scans all of A: at M=256 all four widths peak at 1/2 MiB per thread, and the
# 1/8 MiB the old band prescribed costs 61% at 1T, 16% at 2T, 23% at 4T and 27%
# at 8T. So the new band carries no thread overrides.
AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V4 = replace(
    AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1.bands[2],
    max_routes=215,
)

AMAZON_C5_192C_TP4_F512_LARGE_A_BAND_V4 = StageWindowBand(
    min_routes=216,
    max_routes=287,
    widths=(1, 2, 4, 8),
    w13_bytes_per_thread=MIB // 2,
    w2_bytes_per_thread=MIB // 8,
)

AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4 = StaticStageWindowPolicy(
    name="amazon_c5_192c_tp4_f512_v4",
    hidden_size=4096,
    intermediate_size=512,
    backend_n_tile=8,
    bands=(
        AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND,
        AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V3,
        AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1.bands[1],
        AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V4,
        AMAZON_C5_192C_TP4_F512_LARGE_A_BAND_V4,
        AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1.bands[3],
    ),
)


def default_task_stage_window_policy(
    profile: StageWindowProfilePolicy | None,
    *,
    num_cores: int,
    cpu_ids: tuple[int, ...],
) -> TaskStageWindowPolicy | None:
    """Return the measured default only for its exact machine/profile domain."""
    if profile is None or int(num_cores) != 96:
        return None
    expected = {
        "mode": "tp",
        "degree": 4,
        "hidden_size": 4096,
        "intermediate_size": 512,
        "global_experts": 256,
        "local_experts": 256,
        "backend": "sve",
        "backend_n_tile": 8,
        "sve_implementation": "jit",
        "m_tail_policy": "xbyak_exact_m",
        "activation": "silu",
        "dtype": "bf16",
        "measurement_experts": 256,
        "cores_per_rank": 96,
        "concurrent_ranks": 2,
        "llc_bytes_per_rank": 96 * MIB,
        "numa_nodes": (0, 1),
        "cpu_ids_by_rank": (tuple(range(96)), tuple(range(96, 192))),
    }
    if any(getattr(profile, name, None) != value for name, value in expected.items()):
        return None
    if tuple(int(cpu) for cpu in cpu_ids) not in profile.cpu_ids_by_rank:
        return None
    return AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4


__all__ = [
    "AMAZON_C5_192C_NUMA0_TP4_F512_STAGE_WINDOWS_V1",
    "AMAZON_C5_192C_TP4_F512_LARGE_A_BAND_V4",
    "AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V3",
    "AMAZON_C5_192C_TP4_F512_MID_ROUTE_BAND_V4",
    "AMAZON_C5_192C_TP4_F512_SHORT_ROUTE_BAND",
    "AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1",
    "AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V2",
    "AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V3",
    "AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V4",
    "INHERIT_STAGE_WINDOW",
    "StageWindowBand",
    "StageWindowPolicyEntry",
    "StageWindowProfilePolicy",
    "StaticStageWindowPolicy",
    "TaskStageWindowPolicy",
    "default_task_stage_window_policy",
]
