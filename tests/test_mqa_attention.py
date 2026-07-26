# -*- coding: utf-8 -*-
"""Correctness tests for dense multi-query attention."""

from __future__ import annotations

import pytest
import torch

from fused_cpp.mqa import multi_query_attention, multi_query_attention_torch


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("kv_rank4", [False, True], ids=["kv3d", "kv4d"])
def test_multi_query_attention_matches_torch_reference(
    dtype: torch.dtype,
    is_causal: bool,
    kv_rank4: bool,
) -> None:
    torch.manual_seed(123)
    B, N, L, S, E, Ev = 2, 5, 7, 9, 16, 12
    q = torch.randn(B, N, L, E, dtype=dtype)
    k = torch.randn(B, S, E, dtype=dtype)
    v = torch.randn(B, S, Ev, dtype=dtype)
    if kv_rank4:
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    actual = multi_query_attention(q, k, v, is_causal=is_causal)
    expected = multi_query_attention_torch(q, k, v, is_causal=is_causal)

    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_multi_query_attention_supports_broadcast_masks(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    B, N, L, S, E, Ev = 2, 4, 5, 6, 8, 10
    q = torch.randn(B, N, L, E, dtype=dtype)
    k = torch.randn(B, S, E, dtype=dtype)
    v = torch.randn(B, S, Ev, dtype=dtype)
    masks = [
        torch.randn(L, S, dtype=dtype),
        torch.randn(B, L, S, dtype=dtype),
        torch.randn(B, 1, L, S, dtype=dtype),
        torch.randn(B, N, L, S, dtype=dtype),
    ]

    for mask in masks:
        actual = multi_query_attention(q, k, v, attn_mask=mask)
        expected = multi_query_attention_torch(q, k, v, attn_mask=mask)
        _assert_close(actual, expected, dtype)


def test_multi_query_attention_uses_lower_right_causal_semantics() -> None:
    torch.manual_seed(9)
    q = torch.randn(1, 3, 3, 8)
    k = torch.randn(1, 7, 8)
    v = torch.randn(1, 7, 4)

    actual = multi_query_attention(q, k, v, is_causal=True)
    expected = multi_query_attention_torch(q, k, v, is_causal=True)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_multi_query_attention_rejects_multi_kv_heads() -> None:
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 4)

    with pytest.raises((RuntimeError, ValueError), match=r"\[B, S, [DE]\].*\[B, 1, S, [DE]\]"):
        multi_query_attention(q, k, v)
