from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_stream_count_composition import (
    BACKGROUND_EXPERTS,
    CROSS21,
    MIX21,
    MIX_WIDTHS,
    MODES,
    SHEET16,
    TARGET_CORE_BEGIN,
    _build_bridge,
    _task_specs,
    decide_stream_count,
    live_teams,
    occupied_cores,
    parse_mode,
    place_widths,
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


def _calls(overlap: float) -> dict[str, list[float]]:
    return {"peer_overlap_experts": [overlap]}


def test_mix_covers_twenty_one_cores_as_four_streams() -> None:
    placed = place_widths(MIX21, MIX_WIDTHS)
    assert placed == [(43, 8), (51, 8), (59, 4), (63, 1)]
    assert occupied_cores(placed) == list(MIX21)
    assert TARGET_CORE_BEGIN not in occupied_cores(placed)
    with pytest.raises(ValueError, match="overflow"):
        place_widths(SHEET16, MIX_WIDTHS)


def test_modes_keep_histogram_and_leave_victim_core_free() -> None:
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
        live = [task for task in _task_specs(mode)[2:] if not task[4]]
        cores = occupied_cores([(task[2], task[3]) for task in live])
        assert len(cores) == len(set(cores))
        assert TARGET_CORE_BEGIN not in cores


def test_equal_thread_ladder_and_mix_controls() -> None:
    assert parse_mode("mix_8841_cross_head") == ("mix_8841", "cross_llc")
    assert occupied_cores(live_teams("wide16_same_head")) == list(SHEET16)
    assert occupied_cores(live_teams("two_8t_same_head")) == list(SHEET16)
    assert occupied_cores(live_teams("four_4t_same_head")) == list(SHEET16)
    assert occupied_cores(live_teams("many16_1t_same_head")) == list(SHEET16)
    assert occupied_cores(live_teams("many21_1t_same_head")) == list(MIX21)
    assert occupied_cores(live_teams("mix_8841_same_head")) == list(MIX21)
    assert occupied_cores(live_teams("mix_8841_cross_head")) == list(CROSS21)
    assert live_teams("four_1t_same_head") == [(48, 1), (52, 1), (56, 1), (60, 1)]
    assert live_teams("four_1t_mix_same_head") == [(43, 1), (51, 1), (59, 1), (63, 1)]
    assert sum(width for _core, width in live_teams("mix_8841_same_head")) == 21
    assert len(live_teams("mix_8841_same_head")) == 4
    with pytest.raises(ValueError, match="unsupported stream-count mode"):
        parse_mode("eight_2t_same_head")


def test_decide_stream_count_when_mix_matches_four_1t() -> None:
    comparisons = {
        "wide16_same_head_vs_isolated": _delta(0.02),
        "two_8t_same_head_vs_isolated": _delta(0.05),
        "four_4t_same_head_vs_isolated": _delta(0.08),
        "four_1t_same_head_vs_isolated": _delta(0.07),
        "many16_1t_same_head_vs_isolated": _delta(0.16),
        "mix_8841_same_head_vs_isolated": _delta(0.09),
        "four_1t_mix_same_head_vs_isolated": _delta(0.08),
        "many21_1t_same_head_vs_isolated": _delta(0.20),
        "mix_8841_cross_head_vs_isolated": _delta(0.02),
        "many21_1t_cross_head_vs_isolated": _delta(0.03),
    }
    calls = {
        "wide16_same_head": _calls(1.0),
        "two_8t_same_head": _calls(2.0),
        "four_4t_same_head": _calls(4.0),
        "four_1t_same_head": _calls(4.0),
        "many16_1t_same_head": _calls(16.0),
        "mix_8841_same_head": _calls(4.0),
        "four_1t_mix_same_head": _calls(4.0),
        "many21_1t_same_head": _calls(21.0),
    }

    decision = decide_stream_count(comparisons, calls)

    assert decision["signature"] == "stream_count"
    assert decision["add_default_off_structure"] is False
    assert decision["thread_count"] is False


def test_decide_thread_count_when_mix_matches_many_1t() -> None:
    comparisons = {
        "wide16_same_head_vs_isolated": _delta(0.02),
        "two_8t_same_head_vs_isolated": _delta(0.14),
        "four_4t_same_head_vs_isolated": _delta(0.15),
        "four_1t_same_head_vs_isolated": _delta(0.07),
        "many16_1t_same_head_vs_isolated": _delta(0.16),
        "mix_8841_same_head_vs_isolated": _delta(0.19),
        "four_1t_mix_same_head_vs_isolated": _delta(0.08),
        "many21_1t_same_head_vs_isolated": _delta(0.20),
        "mix_8841_cross_head_vs_isolated": _delta(0.02),
        "many21_1t_cross_head_vs_isolated": _delta(0.03),
    }
    calls = {
        "wide16_same_head": _calls(1.0),
        "two_8t_same_head": _calls(2.0),
        "four_4t_same_head": _calls(4.0),
        "four_1t_same_head": _calls(4.0),
        "many16_1t_same_head": _calls(16.0),
        "mix_8841_same_head": _calls(4.0),
        "four_1t_mix_same_head": _calls(4.0),
        "many21_1t_same_head": _calls(21.0),
    }

    decision = decide_stream_count(comparisons, calls)

    assert decision["signature"] == "thread_count"
    assert decision["stream_count"] is False
    assert decision["add_default_off_structure"] is False
