from __future__ import annotations

import argparse
import json

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_narrow_lane_merge_transition import (
    MODES,
    TARGET_ROUTE_CASES,
    _build_bridge,
    _write_calibration,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, threads: int) -> tuple[int, int]:
        return (routes + threads, routes + 2 * threads)


class _BridgeModel:
    @staticmethod
    def T_iso(routes: int, threads: int) -> float:
        return routes / threads

    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


@pytest.mark.parametrize("mode", MODES)
def test_transition_bridge_preserves_tasks_and_uses_fixed_target_cores(mode: str) -> None:
    bridge, routes_by_expert = _build_bridge(
        _BridgeModel(),
        TARGET_ROUTE_CASES["balanced"],
        thread_cpu_ids=tuple(range(80)),
        mode=mode,
    )

    target_width = 2 if mode.startswith("merge") else 1
    pair_order = list(range(10))
    if mode.startswith("pair_swapped"):
        pair_order[0], pair_order[5] = pair_order[5], pair_order[0]
    elif mode.startswith("pair_tail_head_swapped"):
        pair_order[4], pair_order[5] = pair_order[5], pair_order[4]
    assert sorted(routes_by_expert) == list(range(60))
    assert bridge["task_expert_ids"][:10] == (
        sorted(range(10), key=lambda expert: (-routes_by_expert[expert] / 2, expert))
        if target_width == 2
        else pair_order
    )
    assert bridge["task_threads"][:10] == [target_width] * 10
    assert set(bridge["task_core_begins"][:10]) == (
        {64} if target_width == 2 else {64, 65}
    )
    assert bridge["early_merge"] is False


def test_swapped_pair_exchanges_lane_heads_without_changing_expert_routes() -> None:
    common = {
        "thread_cpu_ids": tuple(range(80)),
        "target_routes": TARGET_ROUTE_CASES["swap_sensitive"],
    }
    baseline, baseline_routes = _build_bridge(_BridgeModel(), mode="pair_background", **common)
    swapped, swapped_routes = _build_bridge(_BridgeModel(), mode="pair_swapped_background", **common)

    assert swapped_routes == baseline_routes
    assert baseline["task_expert_ids"][:10] == list(range(10))
    assert swapped["task_expert_ids"][:10] == [5, 1, 2, 3, 4, 0, 6, 7, 8, 9]
    assert baseline["task_core_begins"] == swapped["task_core_begins"]
    assert baseline["task_threads"] == swapped["task_threads"]


def test_tail_head_swap_reproduces_large_tail_moving_to_other_lane_head() -> None:
    common = {
        "thread_cpu_ids": tuple(range(80)),
        "target_routes": TARGET_ROUTE_CASES["tail_head_transition"],
    }
    baseline, baseline_routes = _build_bridge(_BridgeModel(), mode="pair_background", **common)
    swapped, swapped_routes = _build_bridge(
        _BridgeModel(),
        mode="pair_tail_head_swapped_background",
        **common,
    )

    assert swapped_routes == baseline_routes
    assert baseline["task_expert_ids"][:10] == list(range(10))
    assert swapped["task_expert_ids"][:10] == [0, 1, 2, 3, 5, 4, 6, 7, 8, 9]
    assert [baseline_routes[expert] for expert in baseline["task_expert_ids"][:10]] == [
        1,
        2,
        5,
        19,
        68,
        1,
        2,
        6,
        24,
        57,
    ]
    assert [swapped_routes[expert] for expert in swapped["task_expert_ids"][:10]] == [
        1,
        2,
        5,
        19,
        1,
        68,
        2,
        6,
        24,
        57,
    ]
    assert baseline["task_core_begins"] == swapped["task_core_begins"]


def test_transition_calibration_adds_discrete_narrow_team_corrections(tmp_path) -> None:
    source = tmp_path / "base.json"
    output = tmp_path / "fitted.json"
    source.write_text(
        json.dumps(
            {
                "planner": {
                    "wide_team_pressure": {
                        "isolated_dilation": [{"threads": 8, "dilation": 1.1}],
                        "full_cohort_dilation": [{"threads": 8, "dilation": 1.3}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(runs=31, weight_copies=4)

    _write_calibration(
        source,
        output,
        width2_expert_fixed_ns=120_000.0,
        width2_route_ns=4_000.0,
        width1_correction=0.95,
        width2_correction=0.72,
        args=args,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["planner"]["wide_team_pressure"] == {
        "isolated_dilation": [{"threads": 8, "dilation": 1.1}],
        "full_cohort_dilation": [{"threads": 8, "dilation": 1.3}],
    }
    assert payload["planner"]["narrow_team_contention_correction"] == {
        "full_cohort_correction": [
            {"threads": 1, "correction": 0.95},
            {"threads": 2, "correction": 0.72},
        ]
    }
    assert payload["overheads"]["by_width"] == [
        {
            "threads": 2,
            "expert_fixed_ns": 120_000.0,
            "route_ns": 4_000.0,
        }
    ]
    assert payload["provenance"]["narrow_lane_merge_transition"]["holdout_policy"].startswith(
        "isolated pair"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_calibration(
            source,
            output,
            width2_expert_fixed_ns=100_000.0,
            width2_route_ns=3_000.0,
            width1_correction=0.9,
            width2_correction=0.7,
            args=args,
        )


def test_transition_calibration_rejects_an_already_fitted_source(tmp_path) -> None:
    source = tmp_path / "already-fitted.json"
    output = tmp_path / "refitted.json"
    source.write_text(
        json.dumps(
            {
                "provenance": {
                    "narrow_lane_merge_transition": {"fit_statistic": "existing"},
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(runs=31, weight_copies=4)

    with pytest.raises(ValueError, match="refusing a second fit"):
        _write_calibration(
            source,
            output,
            width2_expert_fixed_ns=100_000.0,
            width2_route_ns=3_000.0,
            width1_correction=0.9,
            width2_correction=0.7,
            args=args,
        )
