from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COST_MODEL = ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path.insert(0, str(COST_MODEL))

from tiso_roofline import (  # noqa: E402
    RooflineCaps,
    fit_bulk_observations,
    fused_expert_work,
    m12_tail_capacity,
    panel_histogram,
    predict_roofline,
)


def test_m12_panel_mapping() -> None:
    assert [m12_tail_capacity(value) for value in range(12)] == [
        0,
        2,
        2,
        4,
        4,
        8,
        8,
        8,
        8,
        12,
        12,
        12,
    ]
    assert panel_histogram(0) == {}
    assert panel_histogram(11) == {12: 1}
    assert panel_histogram(25) == {12: 2, 2: 1}


def test_nsplit_fused_expert_work_matches_kernel_loops() -> None:
    h, f, threads = 4096, 1024, 8
    work = fused_expert_work(12, threads, h, f)

    assert work.effective_rows == 12
    assert work.packed_rows == 12
    assert work.store_rows == 12
    assert work.panel_count == 1
    assert work.w13.flops == 4 * 12 * h * f
    assert work.w13.a_read_bytes == 2 * 12 * h * threads
    assert work.w13.b_read_bytes == 4 * h * f
    assert work.w13.c_write_bytes == 2 * 12 * f
    assert work.w2.flops == 2 * 12 * h * f
    assert work.w2.a_read_bytes == 2 * 12 * f * threads
    assert work.w2.b_read_bytes == 2 * h * f
    assert work.w2.c_write_bytes == 4 * 12 * h
    assert work.gemm_flops == 72 * h * f

    split_work = fused_expert_work(12, threads, h, f, w13_n_ranges=2)
    assert split_work.w13.a_read_bytes == 2 * work.w13.a_read_bytes
    assert split_work.w13.b_read_bytes == work.w13.b_read_bytes
    assert split_work.w13.c_write_bytes == work.w13.c_write_bytes


def test_tiso_tail_uses_distinct_compute_pack_and_store_rows() -> None:
    work = fused_expert_work(1, 1, 64, 32)

    assert work.effective_rows == 2
    assert work.packed_rows == 8
    assert work.store_rows == 1
    assert work.w13.flops == 4 * 2 * 64 * 32
    assert work.w13.c_write_bytes == 2 * 1 * 32
    assert work.w2.c_write_bytes == 4 * 1 * 64
    assert work.aux.packed_a_write_bytes == 2 * 8 * 64


def test_roofline_uses_active_compute_or_memory_ceiling() -> None:
    work = fused_expert_work(12, 1, 64, 32)
    caps = RooflineCaps(
        w13_flops_per_second=1e12,
        w2_flops_per_second=1e12,
        l3_bytes_per_second=1e9,
        copy_bytes_per_second=2e9,
        fixed_ns=100.0,
    )
    prediction = predict_roofline(work, caps)

    assert prediction.w13_ns == pytest.approx(prediction.w13_memory_ns)
    assert prediction.w2_ns == pytest.approx(prediction.w2_memory_ns)
    assert prediction.total_ns == pytest.approx(
        prediction.fixed_ns + prediction.w13_ns + prediction.w2_ns + prediction.aux_ns
    )


def test_64core_profile_bulk_observations_have_physical_units() -> None:
    path = (
        COST_MODEL
        / "profiles"
        / "contention_async_amazon_c5_64c_tp2_sve_F1024_splitw13_v2_r1_20260713.json"
    )
    profile = json.loads(path.read_text(encoding="utf-8"))
    observations = {row.threads: row for row in fit_bulk_observations(profile)}

    assert tuple(observations) == (1, 2, 4, 8, 16, 32)
    assert observations[1].required_tflops == pytest.approx(0.310, abs=0.002)
    assert observations[8].required_tflops == pytest.approx(2.451, abs=0.003)
    assert observations[32].required_tflops == pytest.approx(7.616, abs=0.003)
    assert observations[1].required_l3_gbs == pytest.approx(26.3, abs=0.5)
    assert observations[8].required_l3_gbs == pytest.approx(220.4, abs=1.0)
    assert observations[32].required_l3_gbs == pytest.approx(818.8, abs=1.0)
