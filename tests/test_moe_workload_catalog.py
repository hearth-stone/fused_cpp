from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
MOE_BENCHMARKS = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(PLANNERS))
sys.path.insert(0, str(MOE_BENCHMARKS))

from bench_vllm_staged_schedule import (  # noqa: E402
    make_large_medium_stream_partition_plan,
    make_large_small_partition_plan,
    make_static_16t_to_4x4t_schedule,
    materialize_topk_ids,
    parse_large_medium_stream_partition,
    parse_large_small_partition,
)
from capture_schedule_timeline import plan_to_bridge  # noqa: E402
from fused_cpp.moe import AsyncMoEPlanV2  # noqa: E402
from workload_catalog import (  # noqa: E402
    PAPER_ACTIVE_SET_SIZES,
    PAPER_NUM_EXPERTS,
    PAPER_TOKENS,
    PAPER_TOP_K,
    default_offline_workloads,
    synthetic_offline_workloads,
)
from simulate_schedules import PRESETS  # noqa: E402


def test_paper_workloads_are_valid_global_topk_histograms() -> None:
    workloads = synthetic_offline_workloads()
    expected_names = {
        "moe256-uniform",
        *(f"moe256-active-set-{active}" for active in PAPER_ACTIVE_SET_SIZES),
        "moe256-tiered-hotspot",
        "moe256-long-short-bimodal",
    }

    assert set(workloads) == expected_names
    for name, workload in workloads.items():
        active = [routes for routes in workload.histogram if routes > 0]
        assert workload.name == name
        assert workload.routes == PAPER_TOKENS * PAPER_TOP_K == 12_288
        assert len(workload.histogram) == workload.num_experts == PAPER_NUM_EXPERTS
        assert len(active) == workload.observed_active_experts
        assert len(active) >= PAPER_TOP_K
        assert max(active) <= PAPER_TOKENS
        assert workload.source["kind"] == "synthetic"
        assert not workload.tail_reconstructed

        mean = sum(active) / len(active)
        population_std = math.sqrt(sum((routes - mean) ** 2 for routes in active) / len(active))
        assert workload.observed_routes_std == population_std


def test_uniform_and_active_set_sweep_hold_total_routes_constant() -> None:
    workloads = synthetic_offline_workloads()
    expected = {
        8: 1536,
        16: 768,
        32: 384,
        64: 192,
        128: 96,
        256: 48,
    }

    for active_experts, routes_per_expert in expected.items():
        name = "moe256-uniform" if active_experts == 256 else f"moe256-active-set-{active_experts}"
        active = [routes for routes in workloads[name].histogram if routes > 0]
        assert active == [routes_per_expert] * active_experts


def test_tiered_and_bimodal_workloads_have_expected_modes() -> None:
    workloads = synthetic_offline_workloads()

    tiered = Counter(workloads["moe256-tiered-hotspot"].histogram)
    assert tiered == Counter({0: 192, 96: 48, 384: 12, 768: 4})

    bimodal = Counter(workloads["moe256-long-short-bimodal"].histogram)
    assert bimodal == Counter({0: 77, 12: 174, 2040: 5})


def test_default_catalog_includes_synthetic_and_captured_workloads() -> None:
    workloads = default_offline_workloads()

    assert set(synthetic_offline_workloads()) < set(workloads)
    assert "dsv4-real-2048-seq70" in workloads
    for name, workload in workloads.items():
        assert PRESETS[name] == workload.experts


def test_paper_workloads_materialize_to_exact_distinct_topk_rows() -> None:
    for workload in synthetic_offline_workloads().values():
        topk_ids = materialize_topk_ids(
            workload.histogram,
            tokens=workload.tokens,
            top_k=workload.top_k,
            seed=20260722,
        )

        assert tuple(topk_ids.shape) == (workload.tokens, workload.top_k)
        assert all(len(set(row)) == workload.top_k for row in topk_ids.tolist())
        actual = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=workload.num_experts)
        assert actual.tolist() == list(workload.histogram)

        repeated = materialize_topk_ids(
            workload.histogram,
            tokens=workload.tokens,
            top_k=workload.top_k,
            seed=20260722,
        )
        assert torch.equal(repeated, topk_ids)


def test_static_16t_to_4x4t_schedule_releases_four_short_lanes_per_long_team() -> None:
    workload = synthetic_offline_workloads()["moe256-long-short-bimodal"]
    route_counts = torch.tensor(workload.histogram, dtype=torch.int64)

    def isolated_time_ns(routes: int, threads: int) -> float:
        if routes == 2040 and threads == 16:
            return 6_121_973.0
        if routes == 12 and threads == 4:
            return 179_066.0
        raise AssertionError(f"unexpected isolated-time query: routes={routes}, threads={threads}")

    schedule, metadata = make_static_16t_to_4x4t_schedule(
        route_counts,
        threads=96,
        long_route_threshold=12,
        iso_time_ns=isolated_time_ns,
    )
    expert_ids, core_begins, team_threads, dep_offsets, deps = (tensor.tolist() for tensor in schedule)

    assert len(expert_ids) == 179
    assert len(set(expert_ids)) == 179
    assert set(expert_ids) == {expert for expert, routes in enumerate(workload.histogram) if routes > 0}
    assert expert_ids[:5] == [0, 1, 2, 3, 4]
    assert core_begins[:5] == [0, 16, 32, 48, 64]
    assert team_threads[:5] == [16] * 5
    assert team_threads[5:] == [4] * 174
    assert dep_offsets[0] == 0
    assert dep_offsets[-1] == len(deps)

    short_tasks_by_core: dict[int, list[int]] = {}
    for task_id in range(5, len(expert_ids)):
        task_deps = deps[dep_offsets[task_id] : dep_offsets[task_id + 1]]
        assert all(dependency < task_id for dependency in task_deps)
        short_tasks_by_core.setdefault(core_begins[task_id], []).append(task_id)

    assert set(short_tasks_by_core) == set(range(0, 96, 4))
    for core_begin, task_ids in short_tasks_by_core.items():
        first_task = task_ids[0]
        first_deps = deps[dep_offsets[first_task] : dep_offsets[first_task + 1]]
        if core_begin < 80:
            assert first_deps == [core_begin // 16]
        else:
            assert first_deps == []
        for previous, current in zip(task_ids, task_ids[1:]):
            assert deps[dep_offsets[current] : dep_offsets[current + 1]] == [previous]

    assert metadata["long_tasks"] == 5
    assert metadata["short_tasks"] == 174
    assert metadata["short_lane_task_counts"] == [2] * 10 + [1] * 10 + [36] * 4


def test_dsv4_large_small_partition_keeps_classes_on_disjoint_core_regions() -> None:
    workload = default_offline_workloads()["dsv4-real-2048-seq70"]
    route_counts = torch.tensor(workload.histogram, dtype=torch.int64)

    def isolated_time_ns(routes: int, threads: int) -> float:
        return 20_000.0 + 2_000.0 * routes / threads + 100.0 * threads

    def stage_windows(routes: int, threads: int) -> tuple[int, int]:
        return (2 if routes <= 48 else 8, 16 if threads <= 8 else 0)

    plan, metadata = make_large_small_partition_plan(
        route_counts,
        cpu_ids=list(range(96)),
        route_threshold=48,
        large_core_count=64,
        large_team_threads=8,
        small_team_threads=2,
        iso_time_ns=isolated_time_ns,
        stage_window_select=stage_windows,
        early_merge=True,
    )

    experts = plan.task_expert_ids.tolist()
    begins = plan.task_core_begins.tolist()
    widths = plan.task_threads.tolist()
    dep_offsets = plan.task_dep_offsets.tolist()
    deps = plan.task_deps.tolist()
    expected = {expert for expert, routes in enumerate(workload.histogram) if routes > 0}
    assert len(experts) == len(expected) == 223
    assert set(experts) == expected
    assert metadata["large_tasks"] == 28
    assert metadata["small_tasks"] == 195

    for task, expert in enumerate(experts):
        routes = workload.histogram[expert]
        task_deps = deps[dep_offsets[task] : dep_offsets[task + 1]]
        assert all(dependency < task for dependency in task_deps)
        assert all(begins[dependency] == begins[task] for dependency in task_deps)
        if routes > 48:
            assert widths[task] == 8
            assert 0 <= begins[task] < 64
            assert plan.task_w13_window_tiles[task].item() == 8
        else:
            assert widths[task] == 2
            assert 64 <= begins[task] < 96
            assert plan.task_w13_window_tiles[task].item() == 2
        assert plan.task_w2_window_tiles[task].item() == 16
    assert plan.early_merge is True
    round_trip = AsyncMoEPlanV2.from_dict(plan_to_bridge(plan))
    assert torch.equal(round_trip.task_expert_ids, plan.task_expert_ids)
    assert torch.equal(round_trip.task_w13_window_tiles, plan.task_w13_window_tiles)
    assert round_trip.early_merge is True


def test_dsv4_large_medium_stream_partition_starts_bounded_short_lanes() -> None:
    workload = default_offline_workloads()["dsv4-real-2048-seq70"]
    route_counts = torch.tensor(workload.histogram, dtype=torch.int64)

    def isolated_time_ns(routes: int, threads: int) -> float:
        return 20_000.0 + 2_000.0 * routes / threads + 100.0 * threads

    def stage_windows(routes: int, threads: int) -> tuple[int, int]:
        return (2 if routes <= 48 else 8, 16 if threads <= 8 else 0)

    plan, metadata = make_large_medium_stream_partition_plan(
        route_counts,
        cpu_ids=list(range(96)),
        large_route_threshold=48,
        short_route_threshold=12,
        large_core_count=48,
        large_team_threads=8,
        short_stream_core_count=4,
        iso_time_ns=isolated_time_ns,
        stage_window_select=stage_windows,
        early_merge=None,
    )

    experts = plan.task_expert_ids.tolist()
    begins = plan.task_core_begins.tolist()
    widths = plan.task_threads.tolist()
    dep_offsets = plan.task_dep_offsets.tolist()
    deps = plan.task_deps.tolist()
    assert metadata["large_tasks"] == 28
    assert metadata["medium_tasks"] == 120
    assert metadata["short_tasks"] == 75
    assert metadata["large_routes"] == 8875
    assert metadata["medium_routes"] == 3314
    assert metadata["short_routes"] == 99
    assert len(metadata["short_lane_task_counts"]) == 4
    assert sum(metadata["short_lane_task_counts"]) == 75

    first_task_by_core: dict[int, int] = {}
    for task, expert in enumerate(experts):
        routes = workload.histogram[expert]
        first_task_by_core.setdefault(begins[task], task)
        task_deps = deps[dep_offsets[task] : dep_offsets[task + 1]]
        assert all(dependency < task for dependency in task_deps)
        assert all(begins[dependency] == begins[task] for dependency in task_deps)
        if routes > 48:
            assert widths[task] == 8
            assert 0 <= begins[task] < 48
            assert plan.task_w13_window_tiles[task].item() == 8
        elif routes > 12:
            assert widths[task] == 1
            assert 48 <= begins[task] < 92
            assert plan.task_w13_window_tiles[task].item() == 2
        else:
            assert widths[task] == 1
            assert 92 <= begins[task] < 96
            assert plan.task_w13_window_tiles[task].item() == 2
        assert plan.task_w2_window_tiles[task].item() == 16

    assert set(first_task_by_core) == {
        *range(0, 48, 8),
        *range(48, 96),
    }
    for core in range(92, 96):
        first_task = first_task_by_core[core]
        assert deps[dep_offsets[first_task] : dep_offsets[first_task + 1]] == []


def test_large_small_partition_spec_parser_rejects_malformed_values() -> None:
    assert parse_large_small_partition("48:64:8:2") == (48, 64, 8, 2)
    with pytest.raises(ValueError):
        parse_large_small_partition("48:64:8")

    assert parse_large_medium_stream_partition("48:12:48:8:4") == (48, 12, 48, 8, 4)
    with pytest.raises(ValueError):
        parse_large_medium_stream_partition("48:12:48:8")
    with pytest.raises(ValueError):
        parse_large_medium_stream_partition("12:48:48:8:4")
