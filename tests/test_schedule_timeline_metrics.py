from __future__ import annotations

import sys
from pathlib import Path

import pytest


BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(BENCHMARK_DIR))

from schedule_timeline_metrics import (  # noqa: E402
    annotate_gemm_throughput,
    compute_idle_metrics,
)


def test_annotate_gemm_throughput_assigns_per_core_n_work() -> None:
    tasks = [
        {
            "task": 0,
            "routes": 4,
            "threads": 2,
        }
    ]
    actual = {
        "cores": {
            str(core): [
                {
                    "task": 0,
                    "local_tid": core,
                    "stage": stage,
                    "start_ms": index * 0.002,
                    "end_ms": index * 0.002 + 0.001,
                }
                for index, stage in enumerate(("w13_fused_silu_packc", "w2_direct_route"))
            ]
            for core in range(2)
        }
    }

    summary = annotate_gemm_throughput(
        actual,
        tasks=tasks,
        hidden_size=32,
        intermediate_size=16,
        n_tile=8,
        color_max_gflops=400.0,
    )

    for segments in actual["cores"].values():
        w13, w2 = segments
        assert w13["n_columns"] == 16
        assert w13["logical_flops"] == 4096
        assert w13["gflops"] == pytest.approx(4.096)
        assert w2["n_columns"] == 16
        assert w2["logical_flops"] == 2048
        assert w2["gflops"] == pytest.approx(2.048)
    assert summary["segments"] == 4
    assert summary["observed_max_gflops"] == pytest.approx(4.096)


def test_compute_idle_metrics_separates_internal_and_tail_gaps() -> None:
    actual = {
        "host_segments": [
            {
                "stage": "scheduled_compute",
                "start_ms": 0.0,
                "end_ms": 5.1,
            }
        ],
        "cores": {
            "0": [
                {"stage": "w13_fused_silu_packc", "start_ms": 0.0, "end_ms": 1.0},
                {"stage": "w2_direct_route", "start_ms": 2.0, "end_ms": 3.0},
                {"stage": "merge_ready_token", "start_ms": 3.5, "end_ms": 4.0},
            ],
            "1": [
                {"stage": "w13_fused_silu_packc", "start_ms": 0.0, "end_ms": 5.0},
            ],
        },
    }

    metrics = compute_idle_metrics(actual, cores=2)

    assert metrics["window_ms"] == pytest.approx(5.0)
    assert metrics["internal_idle_core_ms"] == pytest.approx(1.0)
    assert metrics["internal_idle_pct"] == pytest.approx(10.0)
    assert metrics["tail_wait_core_ms"] == pytest.approx(2.0)
    assert metrics["tail_idle_core_ms"] == pytest.approx(1.5)
    assert metrics["tail_idle_pct"] == pytest.approx(15.0)
    assert metrics["per_core"]["0"]["tail_active_ms"] == pytest.approx(0.5)
    assert metrics["per_core"]["1"]["tail_idle_ms"] == pytest.approx(0.0)
