# -*- coding: utf-8 -*-
"""Tests for the naive FlashMLA sparse prefill MLA reference."""

from __future__ import annotations

import pytest
import torch

from fused_cpp.sparse_mla import (
    flash_mla_sparse_fwd_naive,
    sparse_mla_naive,
)


def _flash_mla_sparse_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    *,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    gather_indices = indices.clone().squeeze(1)
    if topk_length is not None:
        valid_by_length = (
            torch.arange(topk, device=q.device).unsqueeze(0)
            < topk_length.to(torch.long).unsqueeze(1)
        )
        gather_indices = gather_indices.masked_fill(~valid_by_length, -1)

    invalid = (gather_indices < 0) | (gather_indices >= s_kv)
    gather_indices = gather_indices.masked_fill(invalid, 0)
    gathered_kv = kv[:, 0, :].index_select(
        0,
        gather_indices.reshape(-1).to(torch.long),
    ).reshape(s_q, topk, d_qk).float()

    logits = torch.matmul(q.float(), gathered_kv.transpose(1, 2))
    logits = logits * sm_scale
    logits = logits.masked_fill(invalid.unsqueeze(1), float("-inf"))

    sparse_lse = torch.logsumexp(logits, dim=-1)
    max_logits = logits.max(dim=-1).values

    if attn_sink is None:
        output_lse = sparse_lse
    else:
        sink = attn_sink.float().reshape(1, h_q).expand(s_q, h_q)
        output_lse = torch.logsumexp(torch.stack((sparse_lse, sink), dim=0), dim=0)

    output_lse = output_lse.clone()
    output_lse[output_lse == float("-inf")] = float("+inf")
    weights = torch.exp(logits - output_lse.unsqueeze(-1))
    out = torch.matmul(weights, gathered_kv[..., :d_v]).to(q.dtype)

    sparse_lse = sparse_lse.clone()
    sparse_lse[sparse_lse == float("-inf")] = float("+inf")
    return out, max_logits, sparse_lse


def test_sparse_mla_naive_matches_flash_mla_reference_features() -> None:
    torch.manual_seed(7)
    s_q, h_q, s_kv, d_qk, d_v = 4, 3, 7, 8, 6
    q = torch.randn(s_q, h_q, d_qk).bfloat16()
    kv = torch.randn(s_kv, 1, d_qk).bfloat16()
    indices = torch.tensor(
        [
            [[0, 3, -1, 6, 7]],
            [[2, 5, 1, 4, 0]],
            [[6, 0, 1, 2, 3]],
            [[4, -2, 2, 1, 8]],
        ],
        dtype=torch.int32,
    )
    topk_length = torch.tensor([5, 3, 0, 2], dtype=torch.int32)
    attn_sink = torch.tensor([float("-inf"), 0.25, float("+inf")], dtype=torch.float32)
    sm_scale = 0.125

    actual = sparse_mla_naive(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    expected = _flash_mla_sparse_reference(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


def test_sparse_mla_naive_all_invalid_outputs_zero_and_lse_inf() -> None:
    q = torch.randn(2, 2, 4).bfloat16()
    kv = torch.randn(3, 1, 4).bfloat16()
    indices = torch.full((2, 1, 3), -1, dtype=torch.int32)

    out, max_logits, lse = sparse_mla_naive(q, kv, indices, 0.5, d_v=3)

    assert torch.equal(out, torch.zeros_like(out))
    assert torch.all(torch.isneginf(max_logits))
    assert torch.all(torch.isposinf(lse))


def test_flash_mla_sparse_fwd_naive_reuses_out_buffer() -> None:
    torch.manual_seed(11)
    q = torch.randn(3, 2, 6).bfloat16()
    kv = torch.randn(5, 1, 6).bfloat16()
    indices = torch.tensor([[[0, 1]], [[2, 3]], [[4, -1]]], dtype=torch.int32)
    d_v = 4

    expected = flash_mla_sparse_fwd_naive(q, kv, indices, 0.25, d_v=d_v)
    out_buffer = torch.empty_like(expected[0])
    actual = flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        0.25,
        d_v=d_v,
        out=out_buffer,
    )

    assert actual[0] is out_buffer
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


def test_sparse_mla_naive_rejects_multi_kv_head_sparse_prefill() -> None:
    q = torch.randn(2, 2, 4).bfloat16()
    kv = torch.randn(3, 2, 4).bfloat16()
    indices = torch.zeros(2, 2, 1, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="h_kv == 1"):
        sparse_mla_naive(q, kv, indices, 0.5, d_v=4)
