from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from run_thread_weight_working_set import (  # noqa: E402
    build_points,
    build_team_points,
    intermediate_for_total_stage,
    max_nsplit_thread_stage_bytes,
    parse_benchmark_output,
    stage_bytes,
)


def test_nsplit_keeps_total_stage_constant_and_reduces_thread_stripe() -> None:
    assert stage_bytes(4096, 2048) == 16 * 1024 * 1024
    assert max_nsplit_thread_stage_bytes(4096, 2048, 1, 8) == 16 * 1024 * 1024
    assert max_nsplit_thread_stage_bytes(4096, 2048, 8, 8) == 2 * 1024 * 1024


def test_fixed_total_expert_mapping_reduces_each_expert_weight() -> None:
    intermediate = intermediate_for_total_stage(4096, 64.0, 32, 8)
    assert intermediate == 256
    assert 32 * stage_bytes(4096, intermediate) == 64 * 1024 * 1024


def test_point_builder_skips_nsplit_thread_counts_without_an_n_tile() -> None:
    points = build_points(
        experiments=["nsplit"],
        routes=[192],
        threads=[32, 64, 96],
        hidden=4096,
        n_tile=8,
        nsplit_stage_mib=[4.0],
        expert_stage_mib=[1.0],
        total_stage_mib=[64.0],
        max_total_stage_mib=192.0,
    )
    assert [point.threads for point in points] == [32, 64]


def test_team_point_builder_keeps_total_threads_and_fixed_work_constant() -> None:
    points = build_team_points(
        experiments=["team-fixed-route", "team-fixed-work"],
        expert_counts=[1, 2, 3, 4, 6, 8, 12],
        total_threads=96,
        fixed_route=2040,
        total_routes=2304,
        hidden=4096,
        n_tile=8,
        stage_mib=16.0,
    )
    fixed_route = [point for point in points if point.experiment == "team-fixed-route"]
    fixed_work = [point for point in points if point.experiment == "team-fixed-work"]

    assert len(fixed_route) == len(fixed_work) == 7
    assert all(point.experts * point.threads_per_expert == 96 for point in points)
    assert all(point.routes == 2040 for point in fixed_route)
    assert all(point.experts * point.routes == 2304 for point in fixed_work)
    assert all(point.routes % 12 == 0 for point in points)


def test_team_point_builder_rejects_non_integral_team_shapes() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        build_team_points(
            experiments=["team-fixed-work"],
            expert_counts=[5],
            total_threads=96,
            fixed_route=2040,
            total_routes=2304,
            hidden=4096,
            n_tile=8,
            stage_mib=16.0,
        )


def test_output_parser_requires_production_fused_stage_data() -> None:
    output = """\
config experts=1 routes=192 hidden=4096 intermediate=512 threads_per_expert=1 workers=1 w13_ranges=2 n_tile=8 copies=3 allocated_gib=0.125
stage variant=production_fused name=fused_w13_silu_mul_packc median_max_team_ms=1.250 min_ms=1.200 mean_ms=1.300
RESULT_JSON {"variant":"production_fused","median_ms":2.5,"p99_ms":2.7,"tflops":1.25}
"""
    result, stages, allocated_gib = parse_benchmark_output(output)
    assert result["median_ms"] == pytest.approx(2.5)
    assert stages == {"fused_w13_silu_mul_packc": pytest.approx(1.25)}
    assert allocated_gib == pytest.approx(0.125)
