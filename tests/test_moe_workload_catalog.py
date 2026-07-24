from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
MOE_BENCHMARKS = ROOT / "optimizations" / "fused_moe_sve" / "benchmarks"
sys.path.insert(0, str(PLANNERS))
sys.path.insert(0, str(MOE_BENCHMARKS))

from bench_vllm_staged_schedule import (  # noqa: E402
    make_static_16t_to_4x4t_schedule,
    materialize_topk_ids,
)
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
