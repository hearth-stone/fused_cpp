from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))

from search_expert_working_set import (  # noqa: E402
    available_cpu_ids,
    assign_by_thread_capacity,
    balanced_thread_shape,
    default_active_experts,
    expert_flops,
    parse_active_experts,
    summarize_search,
)


def test_balanced_thread_shape_uses_every_core() -> None:
    assert available_cpu_ids()
    assert balanced_thread_shape(96, 3) == [32, 32, 32]
    assert balanced_thread_shape(96, 5) == [20, 19, 19, 19, 19]
    assert sum(balanced_thread_shape(8, 3)) == 8
    with pytest.raises(ValueError):
        balanced_thread_shape(8, 9)


def test_active_expert_candidates_are_exhaustive_through_64() -> None:
    assert default_active_experts(96, 32) == list(range(1, 33))
    assert parse_active_experts("1-3,8", 8, 32) == [1, 2, 3, 8]
    with pytest.raises(ValueError):
        parse_active_experts("1,9", 8, 32)


def test_capacity_assignment_preserves_all_experts() -> None:
    lanes = assign_by_thread_capacity(13, [4, 2, 2])
    assert sorted(expert for lane in lanes for expert in lane) == list(range(13))
    assert len(lanes[0]) >= len(lanes[1])


def test_expert_flops_covers_w13_and_w2() -> None:
    assert expert_flops(12, 4096, 2048) == 6 * 12 * 4096 * 2048


def test_search_summary_reports_near_peak_working_set_range() -> None:
    entries = [
        {
            "allocation": "uniform-cores",
            "routes": 192,
            "active_experts": active,
            "working_set_bytes": active * 16,
            "aggregate_tflops": throughput,
        }
        for active, throughput in [(1, 8.0), (2, 9.2), (3, 10.0), (4, 9.1), (5, 8.8)]
    ]
    summary = summarize_search(entries, 0.10)[0]
    assert summary["best_active_experts"] == 3
    assert summary["near_peak_min_active_experts"] == 2
    assert summary["near_peak_max_active_experts"] == 4
