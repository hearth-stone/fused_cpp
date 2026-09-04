from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_fill_port_vs_stream import (
    BACKGROUND_EXPERTS,
    CROSS_LLC_TEAM,
    MODES,
    SAME_LLC_TEAM,
    TARGET_CORE_BEGIN,
    TEAM_WIDTH,
    _build_bridge,
    _task_specs,
    decide_fill_ports,
    parse_mode,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, threads: int) -> tuple[int, int]:
        return (routes + threads, routes + 2 * threads)


class _BridgeModel:
    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


def _dependencies(bridge: dict[str, object], task: int) -> list[int]:
    offsets = bridge["task_dep_offsets"]
    dependencies = bridge["task_deps"]
    return dependencies[offsets[task] : offsets[task + 1]]


def _delta(median_ms: float) -> dict[str, dict[str, float]]:
    return {"delta": {"median_ms": median_ms}}


def test_modes_cover_one_stream_wide_and_many_1t() -> None:
    assert parse_mode("wide16_same_head") == ("wide", "16t", "same_llc")
    assert parse_mode("many16_1t_cross_head") == ("many", "1t", "cross_llc")
    assert "one_1t_same_head" in MODES
    with pytest.raises(ValueError, match="unsupported fill-port mode"):
        parse_mode("wide32_same_head")


def test_wide_and_many_share_histogram_and_keep_cores_disjoint_from_victim() -> None:
    cpu_ids = tuple(range(80))
    routes = None
    for mode in MODES:
        bridge, mode_routes = _build_bridge(_BridgeModel(), thread_cpu_ids=cpu_ids, mode=mode)
        if routes is None:
            routes = mode_routes
        assert mode_routes == routes
        assert sorted(mode_routes) == list(range(2 + BACKGROUND_EXPERTS))
        assert all(mode_routes[expert] == 1 for expert in mode_routes)
        assert bridge["task_core_begins"][:2] == [TARGET_CORE_BEGIN, TARGET_CORE_BEGIN]
        assert _dependencies(bridge, 1) == [0]
        assert TARGET_CORE_BEGIN not in bridge["task_core_begins"][2:]


def test_wide16_uses_one_sixteen_thread_team_on_the_same_cores_as_many_1t() -> None:
    wide = _task_specs("wide16_same_head")
    many = _task_specs("many16_1t_same_head")
    one = _task_specs("one_1t_same_head")
    isolated = _task_specs("isolated_head")

    assert wide[2][2:] == (SAME_LLC_TEAM[0], TEAM_WIDTH, False)
    assert all(task[3] == 1 and task[4] for task in wide[3:])
    assert [task[2] for task in wide[3:]] == list(CROSS_LLC_TEAM[: BACKGROUND_EXPERTS - 1])
    assert [task[2] for task in many[2:]] == list(SAME_LLC_TEAM)
    assert all(task[3] == 1 and not task[4] for task in many[2:])
    assert one[2] == (2, 1, SAME_LLC_TEAM[0], 1, False)
    assert all(task[4] for task in one[3:])
    assert all(task[4] for task in isolated[2:])
    assert [task[2] for task in isolated[2:]] == list(SAME_LLC_TEAM)


def test_decide_fill_ports_when_wide_matches_sixteen_1t() -> None:
    comparisons = {
        "one_1t_same_head_vs_isolated": _delta(0.20),
        "wide16_same_head_vs_isolated": _delta(0.66),
        "many16_1t_same_head_vs_isolated": _delta(0.68),
        "wide16_cross_head_vs_isolated": _delta(0.02),
        "many16_1t_cross_head_vs_isolated": _delta(0.03),
    }
    calls = {
        "one_1t_same_head": {"peer_overlap_experts": [1.0]},
        "wide16_same_head": {"peer_overlap_experts": [1.0]},
        "many16_1t_same_head": {"peer_overlap_experts": [16.0]},
    }

    decision = decide_fill_ports(comparisons, calls)

    assert decision["signature"] == "fill_ports"
    assert decision["add_default_off_structure"] is False


def test_decide_one_stream_when_wide_matches_single_1t() -> None:
    comparisons = {
        "one_1t_same_head_vs_isolated": _delta(0.20),
        "wide16_same_head_vs_isolated": _delta(0.22),
        "many16_1t_same_head_vs_isolated": _delta(0.66),
        "wide16_cross_head_vs_isolated": _delta(0.01),
        "many16_1t_cross_head_vs_isolated": _delta(0.02),
    }
    calls = {
        "one_1t_same_head": {"peer_overlap_experts": [1.0]},
        "wide16_same_head": {"peer_overlap_experts": [1.0]},
        "many16_1t_same_head": {"peer_overlap_experts": [16.0]},
    }

    decision = decide_fill_ports(comparisons, calls)

    assert decision["signature"] == "one_stream"
    assert decision["one_stream"] is True
    assert decision["fill_ports"] is False
    assert decision["add_default_off_structure"] is False
