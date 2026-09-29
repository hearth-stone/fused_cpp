# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest
import torch

from fused_cpp import bf16_linear, prepare_bf16_linear_weight, bf16_linear_mm


@pytest.mark.skipif(
    not bf16_linear._supports_bf16_linear,
    reason="bf16_linear C++ backend is unavailable",
)
def test_bf16_linear_prepacked_matches_raw() -> None:
    torch.manual_seed(0)
    x = (torch.randn(2, 3, 16) * 0.2).to(torch.bfloat16)
    weight = (torch.randn(7, 16) * 0.2).to(torch.bfloat16)

    packed = bf16_linear.prepare(weight)
    packed_from_top_level = prepare_bf16_linear_weight(weight)

    for out_dtype in (torch.bfloat16, torch.float32):
        actual = bf16_linear.linear(x, packed, out_dtype=out_dtype)
        actual_top_level = bf16_linear_mm(
            x,
            packed_from_top_level,
            out_dtype=out_dtype,
        )
        raw = bf16_linear.linear_raw(x, weight, out_dtype=out_dtype)
        expected = (x.float() @ weight.float().T).to(out_dtype)

        assert actual.shape == (2, 3, 7)
        assert actual.dtype == out_dtype
        torch.testing.assert_close(actual.float(), actual_top_level.float())
        torch.testing.assert_close(actual.float(), raw.float())
        torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(
    not bf16_linear._supports_bf16_linear,
    reason="bf16_linear C++ backend is unavailable",
)
def test_bf16_linear_forced_m_n_split_match(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(1)
    x = (torch.randn(17, 32) * 0.2).to(torch.bfloat16)
    weight = (torch.randn(64, 32) * 0.2).to(torch.bfloat16)
    packed = bf16_linear.prepare(weight)

    monkeypatch.setenv("BF16_NEON_CLAMP_THREADS", "0")
    monkeypatch.setenv("BF16_NEON_SPLIT", "m")
    split_m = bf16_linear.linear(x, packed, out_dtype=torch.float32, nthreads=2)
    monkeypatch.setenv("BF16_NEON_SPLIT", "n")
    split_n = bf16_linear.linear(x, packed, out_dtype=torch.float32, nthreads=2)

    expected = x.float() @ weight.float().T
    torch.testing.assert_close(split_m, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(split_n, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(split_m, split_n, atol=5e-2, rtol=5e-2)
