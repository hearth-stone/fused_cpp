from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"))

from profile_gemm_memory_services import MemoryState, plan_stream_window  # noqa: E402
from validate_gemm_four_state import stage_prediction  # noqa: E402


def test_stream_window_evicts_private_and_shared_cache() -> None:
    l2_bytes = 2 * 2**20
    llc_bytes = 96 * 2**20
    state = MemoryState("a_hot_b_stream", stream_a=False, stream_b=True)

    window = plan_stream_window(
        width=48,
        a_bytes=12 * 728 * 2,
        b_bytes=728 * 16 * 2,
        state=state,
        warmup=4,
        minimum_timed_scans=128,
        l2_bytes_per_core=l2_bytes,
        llc_bytes_per_rank=llc_bytes,
        private_cache_multiple=2.0,
        shared_cache_multiple=2.0,
    )

    assert window.target_bytes_per_worker == 2 * l2_bytes
    assert window.copies_per_worker >= 132
    assert window.timed_scans == window.copies_per_worker - 4
    assert window.aggregate_window_bytes >= 2 * llc_bytes
    assert window.stream_bytes_per_scan == 728 * 16 * 2


def test_stream_window_counts_only_streaming_operands() -> None:
    common = {
        "width": 8,
        "a_bytes": 96,
        "b_bytes": 160,
        "warmup": 1,
        "minimum_timed_scans": 2,
        "l2_bytes_per_core": 1024,
        "llc_bytes_per_rank": 8192,
        "private_cache_multiple": 1.0,
        "shared_cache_multiple": 1.0,
    }

    a_stream = plan_stream_window(state=MemoryState("a", True, False), **common)
    b_stream = plan_stream_window(state=MemoryState("b", False, True), **common)
    both_stream = plan_stream_window(state=MemoryState("ab", True, True), **common)

    assert a_stream.stream_bytes_per_scan == 96
    assert b_stream.stream_bytes_per_scan == 160
    assert both_stream.stream_bytes_per_scan == 256


def test_stream_window_rejects_a_fully_hot_state() -> None:
    with pytest.raises(ValueError, match="at least one operand must stream"):
        plan_stream_window(
            width=1,
            a_bytes=96,
            b_bytes=160,
            state=MemoryState("hot", False, False),
            warmup=0,
            minimum_timed_scans=1,
            l2_bytes_per_core=1024,
            llc_bytes_per_rank=8192,
            private_cache_multiple=1.0,
            shared_cache_multiple=1.0,
        )


def test_four_state_stage_prediction_counts_exact_panel_states() -> None:
    state_us = {
        "cold_a_cold_b": 1.0,
        "hot_a_cold_b": 1.0,
        "cold_a_hot_b": 1.0,
        "hot_a_hot_b_l2": 1.0,
    }

    prediction = stage_prediction(
        state_us,
        m=24,
        k=728,
        q_tiles_per_worker=3,
        windows=2,
    )

    assert prediction["counts"] == {
        "cold_a_cold_b": 2,
        "hot_a_cold_b": 4,
        "cold_a_hot_b": 2,
        "hot_a_hot_b_l2": 4,
    }
    assert prediction["predicted_ms"] == pytest.approx(0.012)


def test_four_state_stage_prediction_requires_full_m12_panels() -> None:
    with pytest.raises(ValueError, match="positive multiple of 12"):
        stage_prediction(
            {"cold_a_cold_b": 1.0},
            m=13,
            k=728,
            q_tiles_per_worker=1,
            windows=1,
        )
