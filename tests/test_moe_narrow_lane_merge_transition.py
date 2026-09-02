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
    assert sorted(routes_by_expert) == list(range(60))
    assert bridge["task_expert_ids"][:10] == (
        sorted(range(10), key=lambda expert: (-routes_by_expert[expert] / 2, expert))
        if target_width == 2
        else list(range(10))
    )
    assert bridge["task_threads"][:10] == [target_width] * 10
    assert set(bridge["task_core_begins"][:10]) == (
        {64} if target_width == 2 else {64, 65}
    )
    assert bridge["early_merge"] is False


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
