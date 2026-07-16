from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from run_fragmented_route_pipeline import (  # noqa: E402
    benchmark_command,
    build_points,
    parse_output,
)
from profile_fragmented_route_pipeline import (  # noqa: E402
    parse_perf_csv,
    profiled_command,
    summarize_rows,
)


def test_fragmentation_keeps_total_routes_and_active_stage_constant() -> None:
    points = build_points(
        teams=24,
        base_routes=2040,
        hidden=4096,
        intermediate=512,
        split_factors=[1, 2, 5, 10],
        replaced_teams=[6, 12, 24],
    )

    assert len(points) == 10
    assert {point.total_routes for point in points} == {24 * 2040}
    assert {point.active_stage_bytes for point in points} == {96 * 1024 * 1024}
    assert [(point.split_factor, point.replaced_teams) for point in points[:4]] == [
        (1, 0),
        (2, 6),
        (2, 12),
        (2, 24),
    ]


def test_full_replacement_increases_tasks_and_unique_weights_only() -> None:
    point = build_points(
        teams=24,
        base_routes=2040,
        hidden=4096,
        intermediate=512,
        split_factors=[10],
        replaced_teams=[24],
    )[0]

    assert point.fragment_routes == 204
    assert point.task_count == 240
    assert point.unique_weight_bytes == 240 * 12 * 1024 * 1024
    assert point.total_routes == 48_960


def test_fragmentation_rejects_non_m12_route() -> None:
    with pytest.raises(ValueError, match="M12-aligned"):
        build_points(
            teams=24,
            base_routes=2040,
            hidden=4096,
            intermediate=512,
            split_factors=[3],
            replaced_teams=[24],
        )


def test_command_and_output_parser_preserve_measurement_metadata() -> None:
    point = build_points(
        teams=24,
        base_routes=2040,
        hidden=4096,
        intermediate=512,
        split_factors=[2],
        replaced_teams=[6],
    )[0]
    command = benchmark_command(
        Path("bench_fragmented_route_pipeline"),
        point,
        teams=24,
        base_routes=2040,
        hidden=4096,
        intermediate=512,
        threads_per_team=4,
        schedule="dynamic",
        copies=9,
        cpu_start=0,
        warmup=2,
        runs=7,
    )
    assert command[command.index("--replaced-teams") + 1] == "6"
    assert command[command.index("--split-factor") + 1] == "2"
    assert command[command.index("--copies") + 1] == "9"
    assert command[command.index("--schedule") + 1] == "dynamic"

    output = """\
config teams=24 tasks=30 active_stage_mib=96.000 unique_weight_mib=360.000 allocated_gib=4.000
stage name=gather_pack_a1 median_max_team_sum_ms=2.000 min_ms=1.900 mean_ms=2.100
stage name=fused_w13_silu_mul_packc median_max_team_sum_ms=17.000 min_ms=16.900 mean_ms=17.100
stage name=w2_gemm median_max_team_sum_ms=8.000 min_ms=7.900 mean_ms=8.100
RESULT_JSON {"median_ms":27.0,"p99_ms":27.2,"tflops":22.8,"tasks":30,"total_routes":48960}
"""
    result, stages, config = parse_output(output)
    assert result["tasks"] == 30
    assert stages["fused_w13_silu_mul_packc"] == pytest.approx(17.0)
    assert config == {
        "active_stage_mib": pytest.approx(96.0),
        "unique_weight_mib": pytest.approx(360.0),
        "allocated_gib": pytest.approx(4.0),
    }


def test_profile_command_stops_after_worker_creation() -> None:
    point = build_points(
        teams=24,
        base_routes=2040,
        hidden=4096,
        intermediate=512,
        split_factors=[10],
        replaced_teams=[24],
    )[0]

    command = profiled_command(
        Path("bench_fragmented_route_pipeline"),
        point,
        teams=24,
        base_routes=2040,
        hidden=4096,
        intermediate=512,
        threads_per_team=4,
        schedule="dynamic",
        copies=9,
        cpu_start=0,
        warmup=2,
        runs=7,
        numa_node=0,
    )

    assert command[:6] == ["numactl", "--cpunodebind=0", "--membind=0", "taskset", "-c", "0-95"]
    assert "--stop-before-run" in command
    assert command[command.index("--copies") + 1] == "9"


def test_perf_parser_and_summary_normalize_repeated_measurements() -> None:
    perf_output = """\
900000,,cycles:u,1000000,100.00
450000,,instructions:u,1000000,100.00
2000,,l2d_cache_refill:u,1000000,100.00
1200,,ll_cache_rd:u,1000000,100.00
600,,ll_cache_miss_rd:u,1000000,100.00
180000,,stall_backend_mem:u,1000000,100.00
"""
    events = (
        "cycles",
        "instructions",
        "l2d_cache_refill",
        "ll_cache_rd",
        "ll_cache_miss_rd",
        "stall_backend_mem",
    )
    counters, running = parse_perf_csv(perf_output, events)

    assert counters["l2d_cache_refill"] == 2000
    assert running == {event: pytest.approx(100.0) for event in events}

    rows = [
        {
            "split_factor": 10,
            "fragment_routes": 204,
            "task_count": 240,
            "active_stage_bytes": 96 * 2**20,
            "unique_weight_bytes": 2880 * 2**20,
            "median_ms": median_ms,
            "aggregate_tflops": tflops,
            "ipc": 0.5,
            "memory_stall_fraction": 0.2,
            "l2_refill_gib_per_invocation": refill,
            "ll_read_gib_per_invocation": 1.0,
            "ll_miss_gib_per_invocation": 0.5,
        }
        for median_ms, tflops, refill in ((35.0, 17.6, 3.0), (37.0, 16.7, 3.2), (36.0, 17.1, 3.1))
    ]
    summary = summarize_rows(rows)

    assert summary[0]["median_ms"] == pytest.approx(36.0)
    assert summary[0]["l2_refill_gib_per_invocation"] == pytest.approx(3.1)
