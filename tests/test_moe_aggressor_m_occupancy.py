from __future__ import annotations

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_aggressor_m_occupancy import (
    ANCHOR_ROUTES,
    DEFAULT_AGGRESSOR_COUNTS,
    DEFAULT_AGGRESSOR_ROUTES,
    OUTPUT_KIND,
    SCHEMA_VERSION,
    cell_name,
    decide_occupancy,
    occupancy_modes,
    parse_cell_name,
    parse_occupancy_counts,
    parse_occupancy_routes,
)
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (
    BACKGROUND_EXPERTS,
    _build_bridge,
    _task_specs,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, threads: int) -> tuple[int, int]:
        return (routes + threads, routes + 2 * threads)


class _BridgeModel:
    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


def _delta(median_ms: float) -> dict[str, dict[str, float]]:
    return {"delta": {"median_ms": median_ms}}


def _comparisons_from_curve(
    *,
    same_n1: dict[int, float],
    same_n4: dict[int, float],
    cross_n4: dict[int, float],
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    return {
        routes: {
            "same_llc_head_n1_vs_isolated": _delta(same_n1[routes]),
            "same_llc_head_n4_vs_isolated": _delta(same_n4[routes]),
            "cross_llc_head_n4_vs_isolated": _delta(cross_n4[routes]),
        }
        for routes in ANCHOR_ROUTES + (4, 16)
        if routes in same_n1
    }


def _overlap_calls(routes: tuple[int, ...], count: int = 4) -> dict[str, dict[str, list[float]]]:
    calls: dict[str, dict[str, list[float]]] = {}
    for value in routes:
        for peer_count in (1, 4):
            calls[cell_name(value, f"same_llc_head_n{peer_count}")] = {
                "peer_overlap_experts": [float(peer_count)],
            }
    return calls


def test_occupancy_parsers_lock_anchors() -> None:
    assert parse_occupancy_counts("0,1,4") == DEFAULT_AGGRESSOR_COUNTS
    assert parse_occupancy_routes("1,4,16,68") == DEFAULT_AGGRESSOR_ROUTES
    assert OUTPUT_KIND == "moe_aggressor_m_occupancy_probe"
    assert SCHEMA_VERSION == 1
    assert parse_cell_name("m16/same_llc_head_n4") == (16, "same_llc_head_n4")
    with pytest.raises(ValueError, match="0,1,4"):
        parse_occupancy_counts("0,1,2")
    with pytest.raises(ValueError, match="1 and 68"):
        parse_occupancy_routes("1,4,16")


def test_occupancy_modes_skip_split_and_keep_isolated_control() -> None:
    modes = occupancy_modes((0, 1, 4))

    assert modes[:2] == ("isolated_head", "isolated_after_1")
    assert "same_llc_head_n4" in modes
    assert "cross_llc_head_n1" in modes
    assert "split_head_n4" not in modes
    assert "same_llc_head_n8" not in modes
    assert len(modes) == 10


def test_background_routes_change_histogram_not_placement_or_deps() -> None:
    cpu_ids = tuple(range(80))
    default_bridge, default_routes = _build_bridge(
        _BridgeModel(),
        thread_cpu_ids=cpu_ids,
        mode="same_llc_head_n1",
    )
    thin_bridge, thin_routes = _build_bridge(
        _BridgeModel(),
        thread_cpu_ids=cpu_ids,
        mode="same_llc_head_n1",
        background_routes=4,
    )

    assert default_routes[0] == default_routes[1] == 1
    assert all(default_routes[expert] == 68 for expert in range(2, 2 + BACKGROUND_EXPERTS))
    assert all(thin_routes[expert] == 4 for expert in range(2, 2 + BACKGROUND_EXPERTS))
    assert thin_routes[0] == 1
    assert default_bridge["task_core_begins"] == thin_bridge["task_core_begins"]
    assert default_bridge["task_deps"] == thin_bridge["task_deps"]
    assert default_bridge["task_dep_offsets"] == thin_bridge["task_dep_offsets"]
    assert [task[1] for task in _task_specs("same_llc_head_n1", background_routes=1)][2:] == [1] * 15


def test_decide_occupancy_accepts_stable_same_llc_tax() -> None:
    routes = (1, 4, 16, 68)
    comparisons = _comparisons_from_curve(
        same_n1={1: 0.144, 4: 0.140, 16: 0.146, 68: 0.144},
        same_n4={1: 0.291, 4: 0.300, 16: 0.305, 68: 0.312},
        cross_n4={1: 0.010, 4: 0.012, 16: 0.020, 68: 0.043},
    )

    decision = decide_occupancy(
        comparisons_by_routes=comparisons,
        calls_by_cell=_overlap_calls(routes),
        routes=routes,
    )

    assert decision["signature"] == "occupancy"
    assert decision["add_default_off_structure"] is False
    assert decision["occupancy_stable"] is True
    assert decision["utilization_growth"] is False


def test_decide_utilization_when_tax_grows_with_aggressor_m() -> None:
    routes = (1, 4, 16, 68)
    comparisons = _comparisons_from_curve(
        same_n1={1: 0.02, 4: 0.05, 16: 0.08, 68: 0.14},
        same_n4={1: 0.05, 4: 0.20, 16: 0.50, 68: 0.90},
        cross_n4={1: 0.02, 4: 0.08, 16: 0.20, 68: 0.40},
    )

    decision = decide_occupancy(
        comparisons_by_routes=comparisons,
        calls_by_cell=_overlap_calls(routes),
        routes=routes,
    )

    assert decision["signature"] == "utilization"
    assert decision["add_default_off_structure"] is False
    assert decision["utilization_growth"] is True
    assert decision["occupancy_stable"] is False


def test_decide_duration_occupancy_when_tax_appears_after_peer_covers_w13() -> None:
    routes = (1, 4, 16, 68)
    comparisons = _comparisons_from_curve(
        same_n1={1: 0.01, 4: 0.04, 16: 0.14, 68: 0.14},
        same_n4={1: 0.02, 4: 0.08, 16: 0.30, 68: 0.31},
        cross_n4={1: 0.00, 4: 0.01, 16: 0.02, 68: 0.03},
    )

    decision = decide_occupancy(
        comparisons_by_routes=comparisons,
        calls_by_cell=_overlap_calls(routes),
        routes=routes,
    )

    assert decision["signature"] == "duration_occupancy"
    assert decision["duration_occupancy"] is True
    assert decision["add_default_off_structure"] is False


def test_decide_rejects_missing_overlap() -> None:
    routes = (1, 68)
    comparisons = _comparisons_from_curve(
        same_n1={1: 0.14, 68: 0.14},
        same_n4={1: 0.31, 68: 0.31},
        cross_n4={1: 0.02, 68: 0.02},
    )
    calls = _overlap_calls(routes)
    calls[cell_name(1, "same_llc_head_n4")] = {"peer_overlap_experts": [0.0]}

    decision = decide_occupancy(
        comparisons_by_routes=comparisons,
        calls_by_cell=calls,
        routes=routes,
    )

    assert decision["signature"] == "invalid_overlap"
    assert decision["overlap_ok"] is False
    assert decision["add_default_off_structure"] is False
