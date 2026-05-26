# -*- coding: utf-8 -*-
"""W4A8 int8 GEMM 的 ``torch._int_mm`` 与 fallback 性能对比。"""
from __future__ import annotations

import importlib.util

import pytest
import torch

from fused_cpp.w4a8_linear import _int8_gemm, _supports_int_mm


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pytest_benchmark") is None,
    reason="pytest-benchmark 未安装",
)


@pytest.mark.bench
@pytest.mark.skipif(not _supports_int_mm(), reason="torch._int_mm 不可用")
@pytest.mark.parametrize(
    "use_int_mm",
    [True, False],
    ids=["int_mm", "int32_matmul_fallback"],
)
def test_int_mm_vs_fallback_perf(benchmark, use_int_mm: bool) -> None:
    """同一组 int8 输入下，对比 int_mm 与 fallback 的 GEMM 性能。"""
    # Arrange
    torch.manual_seed(0)
    m, k, n = 128, 1024, 512
    x_q = torch.randint(-128, 127, (m, k), dtype=torch.int8)
    w_q = torch.randint(-15, 16, (k, n), dtype=torch.int8)

    ref = _int8_gemm(x_q, w_q, use_int_mm=False)
    out = _int8_gemm(x_q, w_q, use_int_mm=use_int_mm)
    assert torch.equal(out, ref)

    def run() -> torch.Tensor:
        return _int8_gemm(x_q, w_q, use_int_mm=use_int_mm)

    # Warmup
    for _ in range(3):
        run()

    benchmark.pedantic(run, iterations=10, rounds=5, warmup_rounds=0)
