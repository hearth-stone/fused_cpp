from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))

from iso_formula import IsoFormula  # noqa: E402
from working_set_model import (  # noqa: E402
    OwnerCacheModel,
    ScanObservation,
    fit_owner_cache_model,
    isolated_baseline_ns,
    max_full_stage_bytes,
    recommend_working_sets,
)


def test_full_stage_size_and_v3_owner_cache_band() -> None:
    stage_bytes = max_full_stage_bytes(4096, 2048)
    model = OwnerCacheModel(
        cores=96,
        private_cache_bytes_per_core=2 * 2**20,
        cache_ways=8,
        reserved_ways=2,
        bandwidth_limit_bytes_per_second=9e12,
        stream_saturation=0.85,
        target_bandwidth_utilization=0.95,
    )
    assert stage_bytes == 32 * 2**20
    assert model.owner_cache_budget_bytes == 144 * 2**20
    assert model.minimum_streams() == 3
    assert model.maximum_streams(stage_bytes) == 4


def test_owner_cache_fit_recovers_resident_stream_saturation() -> None:
    limit = 9e12
    saturation = 0.85
    stream = 16 * 2**20
    observations = [
        ScanObservation(
            active_streams=n,
            stream_bytes=stream,
            working_set_bytes=n * stream,
            bandwidth_bytes_per_second=(limit * (1.0 - math.exp(-n / saturation)) if n <= 8 else 4e12),
        )
        for n in (1, 2, 3, 4, 6, 8, 12)
    ]
    fitted = fit_owner_cache_model(
        observations,
        cores=96,
        private_cache_bytes_per_core=2 * 2**20,
        cache_ways=8,
        reserved_ways=2,
        target_bandwidth_utilization=0.95,
    )
    assert fitted.bandwidth_limit_bytes_per_second == pytest.approx(limit, rel=0.01)
    assert fitted.stream_saturation == pytest.approx(saturation, abs=0.03)


def test_small_owner_cache_degenerates_to_one_stream() -> None:
    stream = 4 * 2**20
    observations = [ScanObservation(n, stream, n * stream, (800 - 50 * n) * 1e9) for n in range(1, 9)]
    fitted = fit_owner_cache_model(
        observations,
        cores=8,
        private_cache_bytes_per_core=1 * 2**20,
        cache_ways=8,
        reserved_ways=2,
        target_bandwidth_utilization=0.95,
    )
    assert fitted.owner_cache_budget_bytes == 6 * 2**20
    assert fitted.minimum_streams() == 1
    assert fitted.maximum_streams(stream) == 1


def test_iso_headroom_selects_smallest_robust_working_set() -> None:
    formula = IsoFormula(
        o0=0.0,
        o1=0.0,
        alpha=0.0,
        beta=0.0,
        c_pts=[(2040, 100.0)],
        phi_pts=[(1, 1.0), (12, 0.09), (24, 0.045), (32, 0.04)],
    )
    assert isolated_baseline_ns(2040, [24] * 4, 32, formula) > 0
    model = OwnerCacheModel(
        cores=96,
        private_cache_bytes_per_core=2 * 2**20,
        cache_ways=8,
        reserved_ways=2,
        bandwidth_limit_bytes_per_second=9e12,
        stream_saturation=0.85,
        target_bandwidth_utilization=0.95,
    )
    candidates = [
        {
            "routes": 2040,
            "shape": [32] * 3,
            "active_experts": 3,
            "isolated_baseline_ns": 106.0,
            "measured_ns": 103.0,
        },
        {
            "routes": 2040,
            "shape": [24] * 4,
            "active_experts": 4,
            "isolated_baseline_ns": 101.0,
            "measured_ns": 100.0,
        },
        {
            "routes": 2040,
            "shape": [12] * 8,
            "active_experts": 8,
            "isolated_baseline_ns": 100.0,
            "measured_ns": 101.0,
        },
    ]
    summary = recommend_working_sets(
        candidates,
        stage_bytes=32 * 2**20,
        model=model,
        iso_headroom=0.05,
    )[0]
    assert summary["recommended_active_experts"] == 3
    assert summary["measured_regret"] == pytest.approx(0.03)
