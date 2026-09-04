from __future__ import annotations

import io
from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_stream_pressure_pmu import (
    EXPERT_COUNT,
    PMU_MODES,
    SCRUB_CORES,
    PerfControl,
    _bridge,
    _scrub_bridge,
    pmu_live_teams,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, threads: int) -> tuple[int, int]:
        return routes + threads, routes + 2 * threads


class _BridgeModel:
    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


def test_pmu_bridge_contains_only_overlapping_tasks() -> None:
    cpu_ids = tuple(range(80))
    expected = {
        "isolated_head": (1, 0, 0),
        "wide16_same_head": (2, 1, 16),
        "two_8t_same_head": (3, 2, 16),
        "four_4t_same_head": (5, 4, 16),
        "four_2t_same_head": (5, 4, 8),
        "eight_2t_same_head": (9, 8, 16),
        "eight_1t_same_head": (9, 8, 8),
        "four_1t_same_head": (5, 4, 4),
        "many16_1t_same_head": (17, 16, 16),
        "grid4_1t_same_head": (5, 4, 4),
        "grid4_2t_same_head": (5, 4, 8),
        "grid6_1t_same_head": (7, 6, 6),
        "grid6_2t_same_head": (7, 6, 12),
        "grid8_1t_same_head": (9, 8, 8),
        "grid8_2t_same_head": (9, 8, 16),
    }
    for mode in PMU_MODES:
        bridge, routes = _bridge(_BridgeModel(), thread_cpu_ids=cpu_ids, mode=mode)
        tasks, streams, threads = expected[mode]
        assert len(bridge["task_expert_ids"]) == tasks
        assert len(routes) == tasks
        assert bridge["task_deps"] == []
        assert bridge["task_dep_offsets"] == [0] * (tasks + 1)
        assert len(bridge["task_threads"]) - 1 == streams
        assert sum(bridge["task_threads"][1:]) == threads


def test_pmu_count_sweep_keeps_sixteen_peer_threads() -> None:
    for mode in (
        "wide16_same_head",
        "two_8t_same_head",
        "four_4t_same_head",
        "eight_2t_same_head",
        "many16_1t_same_head",
    ):
        teams = pmu_live_teams(mode)
        assert sum(width for _core, width in teams) == 16


def test_request_shape_modes_keep_expert_starts_and_counts() -> None:
    assert pmu_live_teams("four_4t_same_head") == [(48, 4), (52, 4), (56, 4), (60, 4)]
    assert pmu_live_teams("four_2t_same_head") == [(48, 2), (52, 2), (56, 2), (60, 2)]
    assert pmu_live_teams("four_1t_same_head") == [(48, 1), (52, 1), (56, 1), (60, 1)]
    assert pmu_live_teams("eight_2t_same_head") == [(core, 2) for core in range(48, 64, 2)]
    assert pmu_live_teams("eight_1t_same_head") == [(core, 1) for core in range(48, 64, 2)]


def test_proxy_grid_uses_nested_start_prefixes() -> None:
    for count in (4, 6, 8):
        starts = [48 + 2 * index for index in range(count)]
        for width in (1, 2):
            assert pmu_live_teams(f"grid{count}_{width}t_same_head") == [
                (core, width) for core in starts
            ]


def test_scrub_bridge_touches_every_expert_on_disjoint_copy() -> None:
    bridge, routes = _scrub_bridge(_BridgeModel(), thread_cpu_ids=tuple(range(80)))
    assert bridge["task_expert_ids"] == list(range(EXPERT_COUNT))
    assert bridge["task_core_begins"] == list(SCRUB_CORES)
    assert routes == {expert: 1 for expert in range(EXPERT_COUNT)}


def test_perf_control_requires_both_fifos() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        PerfControl([], [Path("ack.fifo")])


def test_perf_control_checks_acknowledgement() -> None:
    control = PerfControl([], [])
    control._controls = [io.BytesIO(), io.BytesIO()]
    control._acks = [io.BytesIO(b"ack\n"), io.BytesIO(b"ack\n")]
    control.command("enable")
    assert all(stream.getvalue() == b"enable\n" for stream in control._controls)

    control._acks = [io.BytesIO(b"\x00ack\n"), io.BytesIO(b"ack\n")]
    control.command("disable")

    control._acks = [io.BytesIO(b"ack\n"), io.BytesIO(b"error\n")]
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        control.command("disable")
