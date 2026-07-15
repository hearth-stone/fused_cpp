# -*- coding: utf-8 -*-
"""``flash2_neon_cache`` 解耦微内核的接口级测试。"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("fused_cpp._C")

from fused_cpp import _C  # noqa: E402


@pytest.mark.equiv
@pytest.mark.parametrize("dtype", ["fp32", "bf16"], ids=["fp32", "bf16"])
def test_flash2_neon_cache_microkernels_validate_against_scalar(dtype):
    """QKᵀ/P·V 的 ``8x8``、``8x4`` 和 tail dispatcher 与标量参考一致。"""
    result = _C.validate_sdpa_flash2_neon_cache_microkernels(
        dtype=dtype,
        E=17,
        Sk=19,
    )
    atol = 1e-4 if dtype == "fp32" else 5e-3
    for key in (
        "qkt_8x8_max_abs",
        "qkt_8x4_max_abs",
        "qkt_tail_max_abs",
        "pv_8x8_max_abs",
        "pv_tail_max_abs",
    ):
        value = float(result[key])
        assert math.isfinite(value), (key, result)
        assert value <= atol, f"{key}={value:.3e} > {atol:.3e} ({result})"


@pytest.mark.equiv
def test_flash2_neon_cache_microkernels_benchmark_smoke():
    """现有 benchmark 入口应调用解耦后的微内核并返回完整指标。"""
    result = _C.benchmark_sdpa_flash2_neon_cache_microkernels(
        dtype="fp32",
        E=17,
        Sk=19,
        iterations=2,
        warmup=1,
    )
    for key in (
        "direct_microkernel",
        "qkt_8x8_checksum",
        "qkt_8x4_checksum",
        "pv_8x8_checksum",
        "qkt_8x8_gflops",
        "qkt_8x4_gflops",
        "pv_8x8_gflops",
    ):
        assert key in result, result
        assert math.isfinite(float(result[key])), (key, result)
    assert result["direct_microkernel"] == 1.0
