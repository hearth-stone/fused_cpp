"""Profile-bound per-task packed-B window policies for MoE plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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
    w13_split: bool
    w13_split_chunks: int
    weight_window_bytes: int
    w13_window_ranges: int
    w2_window_ranges: int
    measurement_experts: int
    cores_per_rank: int
    concurrent_ranks: int
    llc_bytes_per_rank: int
    numa_nodes: tuple[int, ...]
    cpu_ids_by_rank: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class StageWindowBand:
    min_routes: int
    max_routes: int
    thread_windows: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if self.min_routes <= 0 or self.max_routes < self.min_routes:
            raise ValueError("route band must be positive and non-empty")
        widths = [threads for threads, _, _ in self.thread_windows]
        if any(threads <= 0 for threads in widths) or len(widths) != len(set(widths)):
            raise ValueError("thread widths must be positive and unique within a route band")
        if any(min(w13_bytes, w2_bytes) < 0 for _, w13_bytes, w2_bytes in self.thread_windows):
            raise ValueError("explicit stage windows must be non-negative")

    def select(self, routes: int, threads: int) -> tuple[int, int] | None:
        if not self.min_routes <= routes <= self.max_routes:
            return None
        for candidate_threads, w13_bytes, w2_bytes in self.thread_windows:
            if candidate_threads == threads:
                return w13_bytes, w2_bytes
        return None


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
    name: str
    bands: tuple[StageWindowBand, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage-window policy name must be non-empty")
        ordered = sorted(self.bands, key=lambda band: band.min_routes)
        if tuple(ordered) != self.bands:
            raise ValueError("stage-window route bands must be sorted")
        if any(left.max_routes >= right.min_routes for left, right in zip(ordered, ordered[1:])):
            raise ValueError("stage-window route bands must not overlap")

    def select(self, routes: int, threads: int) -> tuple[int, int]:
        routes = int(routes)
        threads = int(threads)
        if routes <= 0 or threads <= 0:
            raise ValueError("routes and threads must be positive")
        for band in self.bands:
            selected = band.select(routes, threads)
            if selected is not None:
                return selected
        return INHERIT_STAGE_WINDOW, INHERIT_STAGE_WINDOW

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
            for band in self.bands
            for threads, w13_bytes, w2_bytes in band.thread_windows
        )


# Calibrated on both 96-core NUMA ranks of AmazonC5192Cores for the TP4
# H=4096/F=512 split-W13 SVE path. Unsupported route/width combinations inherit
# the profile's operator-wide window instead of being extrapolated.
AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1 = StaticStageWindowPolicy(
    name="amazon_c5_192c_tp4_f512_v1",
    bands=(
        StageWindowBand(
            min_routes=49,
            max_routes=95,
            thread_windows=((8, 1 * MIB, 1 * MIB),),
        ),
        StageWindowBand(
            min_routes=96,
            max_routes=143,
            thread_windows=(
                (1, MIB // 8, MIB // 8),
                (2, MIB // 8, MIB // 4),
                (4, MIB // 4, MIB // 2),
                (8, 1 * MIB, MIB // 2),
            ),
        ),
        StageWindowBand(
            min_routes=144,
            max_routes=287,
            thread_windows=(
                (1, MIB // 8, MIB // 8),
                (2, MIB // 4, MIB // 4),
                (4, MIB // 2, MIB // 2),
                (8, 1 * MIB, MIB // 2),
            ),
        ),
        StageWindowBand(
            min_routes=288,
            max_routes=575,
            thread_windows=(
                (1, 1 * MIB, MIB // 2),
                (2, 1 * MIB, MIB // 4),
                (4, 2 * MIB, MIB // 2),
                (8, 4 * MIB, 1 * MIB),
            ),
        ),
    ),
)

# Compatibility alias for the initial NUMA0-only experiment name.
AMAZON_C5_192C_NUMA0_TP4_F512_STAGE_WINDOWS_V1 = AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1


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
        "w13_split": True,
        "w13_split_chunks": 2,
        "weight_window_bytes": 0,
        "w13_window_ranges": 2,
        "w2_window_ranges": 1,
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
    return AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1


__all__ = [
    "AMAZON_C5_192C_TP4_F512_STAGE_WINDOWS_V1",
    "AMAZON_C5_192C_NUMA0_TP4_F512_STAGE_WINDOWS_V1",
    "INHERIT_STAGE_WINDOW",
    "StageWindowBand",
    "StageWindowPolicyEntry",
    "StageWindowProfilePolicy",
    "StaticStageWindowPolicy",
    "TaskStageWindowPolicy",
    "default_task_stage_window_policy",
]
