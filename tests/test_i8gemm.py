# -*- coding: utf-8 -*-
"""Tests for the public ``fused_cpp.i8gemm`` wrapper."""

from __future__ import annotations

import pytest
import torch

from fused_cpp import i8gemm


pytestmark = pytest.mark.skipif(
    not i8gemm._supports_i8gemm,
    reason="i8gemm C++ bindings are unavailable",
)


def _reference_dynamic_scaled_mm(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
    max_abs = x_2d.abs().amax(dim=-1)
    x_scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    x_q = torch.round(x_2d / x_scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)

    acc = x_q.to(torch.int32) @ weight_int8.t().contiguous().to(torch.int32)
    scale = weight_scale.reshape(-1).to(torch.float32)
    if scale.numel() == 1:
        scale = scale.expand(weight_int8.shape[0])
    out = acc.to(torch.float32) * x_scale.unsqueeze(-1) * scale.unsqueeze(0)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(out_dtype).reshape(*x.shape[:-1], weight_int8.shape[0])


@pytest.mark.parametrize(
    "M,K,N,out_dtype",
    [
        (1, 7, 5, torch.float32),
        (4, 16, 8, torch.bfloat16),
        (9, 31, 13, torch.bfloat16),
    ],
)
def test_dynamic_scaled_mm_matches_reference(M: int, K: int, N: int, out_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    weight = torch.randint(-16, 17, (N, K), dtype=torch.int8)
    weight_scale = torch.rand(N, dtype=torch.float32) * 0.05 + 0.01
    bias = torch.randn(N, dtype=torch.bfloat16)
    x = torch.randn(M, K, dtype=torch.bfloat16)

    packed = i8gemm.prepare(weight, weight_scale)
    out = i8gemm.dynamic_scaled_mm(x, packed, bias=bias, out_dtype=out_dtype, nthreads=0)
    expected = _reference_dynamic_scaled_mm(x, weight, weight_scale, bias=bias, out_dtype=out_dtype)

    assert out.shape == (M, N)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_dynamic_scaled_mm_supports_batched_input_and_scalar_scale() -> None:
    torch.manual_seed(1)
    batch, M, K, N = 2, 3, 19, 11
    weight = torch.randint(-8, 9, (N, K), dtype=torch.int8)
    weight_scale = torch.tensor(0.02, dtype=torch.float32)
    x = torch.randn(batch, M, K, dtype=torch.float32)

    packed = i8gemm.prepare(weight, weight_scale)
    out = i8gemm.dynamic_scaled_mm(x, packed, out_dtype=torch.bfloat16)
    expected = _reference_dynamic_scaled_mm(x, weight, weight_scale, out_dtype=torch.bfloat16)

    assert out.shape == (batch, M, N)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_prepare_rejects_bad_layout_dtype() -> None:
    weight = torch.randn(4, 8)
    scale = torch.ones(4)
    with pytest.raises(RuntimeError, match="torch.int8"):
        i8gemm.prepare(weight, scale)


def test_dynamic_scaled_mm_rejects_bad_k() -> None:
    weight = torch.randint(-8, 9, (4, 8), dtype=torch.int8)
    packed = i8gemm.prepare(weight, torch.ones(4))
    x = torch.randn(2, 7, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="last dim"):
        i8gemm.dynamic_scaled_mm(x, packed)
