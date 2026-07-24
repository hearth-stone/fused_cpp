from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANNERS = ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path.insert(0, str(PLANNERS))

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
