# -*- coding: utf-8 -*-
"""Tests for the vLLM-style naive sparse prefill MLA reference."""

from __future__ import annotations

import pytest
import torch

from fused_cpp.sparse_mla import (
    _HAS_CPP_SPARSE_MLA,
    _build_sparse_mla_plans,
    flash_mla_sparse_fwd,
    flash_mla_sparse_fwd_naive,
    sparse_mla_naive,
)


def _vllm_cpu_sparse_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    *,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _gather_kv(valid_indices: torch.Tensor) -> torch.Tensor:
        if valid_indices.numel() == 1:
            start = int(valid_indices[0].item())
            return kv_2d.narrow(0, start, 1)

        breaks = (valid_indices[1:] != valid_indices[:-1] + 1).nonzero(as_tuple=False).flatten()
        if breaks.numel() == 0:
            start = int(valid_indices[0].item())
            return kv_2d.narrow(0, start, valid_indices.numel())

        num_runs = int(breaks.numel()) + 1
        if num_runs > 4:
            return kv_2d.index_select(0, valid_indices)

        parts: list[torch.Tensor] = []
        run_start = 0
        for break_idx_t in breaks:
            run_end = int(break_idx_t.item()) + 1
            start = int(valid_indices[run_start].item())
            parts.append(kv_2d.narrow(0, start, run_end - run_start))
            run_start = run_end
        start = int(valid_indices[run_start].item())
        parts.append(kv_2d.narrow(0, start, valid_indices.numel() - run_start))
        return torch.cat(parts, dim=0)

    if kv.ndim == 3:
        assert kv.shape[1] == 1
        kv_2d = kv.squeeze(1)
    else:
        kv_2d = kv
    s_q, h_q, _ = q.shape
    out = torch.zeros((s_q, h_q, d_v), dtype=torch.float32, device=q.device)
    max_logits = torch.full((s_q, h_q), float("-inf"), device=q.device)
    lse = torch.full((s_q, h_q), float("+inf"), device=q.device)
    indices_2d = indices.reshape(s_q, -1).to(torch.long)
    sink = attn_sink[:h_q].to(torch.float32) if attn_sink is not None else None

    for token_idx in range(s_q):
        valid_indices = indices_2d[token_idx]
        valid_indices = valid_indices[valid_indices >= 0]
        if valid_indices.numel() == 0:
            continue

        k_i = _gather_kv(valid_indices).to(torch.float32)
        logits = torch.matmul(q[token_idx].to(torch.float32), k_i.T) * sm_scale
        max_logits[token_idx] = logits.max(dim=-1).values
        lse[token_idx] = torch.logsumexp(logits, dim=-1)
        if sink is not None:
            logits = torch.cat([logits, sink[:, None]], dim=-1)
        probs = torch.softmax(logits, dim=-1)
        out[token_idx] = torch.matmul(probs[..., : k_i.shape[0]], k_i[:, :d_v])

    return out.to(q.dtype), max_logits, lse


def _assert_sparse_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[2], expected[2], atol=1e-5, rtol=1e-5)


def test_sparse_mla_naive_matches_vllm_cpu_sparse_attention() -> None:
    torch.manual_seed(7)
    s_q, h_q, s_kv, d_qk, d_v = 4, 3, 7, 8, 6
    q = torch.randn(s_q, h_q, d_qk).bfloat16()
    kv = torch.randn(s_kv, 1, d_qk).bfloat16()
    indices = torch.tensor(
        [
            [[0, 3, -1, 6, 5]],
            [[2, 5, 1, 4, 0]],
            [[6, 0, 1, 2, 3]],
            [[4, -2, 2, 1, 6]],
        ],
        dtype=torch.int32,
    )
    topk_length = torch.tensor([2, 1, 0, 2], dtype=torch.int32)
    attn_sink = torch.tensor([float("-inf"), 0.25, 1.0], dtype=torch.float32)
    sm_scale = 0.125

    actual = sparse_mla_naive(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        return_stats=True,
    )
    expected = _vllm_cpu_sparse_attention_reference(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink=attn_sink,
    )

    _assert_sparse_close(actual, expected)


def test_sparse_mla_naive_ignores_topk_length_like_vllm() -> None:
    torch.manual_seed(17)
    q = torch.randn(2, 2, 6).bfloat16()
    kv = torch.randn(5, 1, 6).bfloat16()
    indices = torch.tensor([[[0, 1, 2]], [[3, -1, 4]]], dtype=torch.int32)
    topk_length = torch.zeros(2, dtype=torch.int32)

    actual = sparse_mla_naive(
        q,
        kv,
        indices,
        0.25,
        topk_length=topk_length,
        return_stats=True,
    )
    expected = sparse_mla_naive(
        q,
        kv,
        indices,
        0.25,
        topk_length=None,
        return_stats=True,
    )

    assert actual[0].shape == q.shape
    _assert_sparse_close(actual, expected)


def test_sparse_mla_naive_positive_out_of_range_raises_like_vllm() -> None:
    q = torch.randn(1, 1, 4).bfloat16()
    kv = torch.randn(3, 1, 4).bfloat16()
    indices = torch.tensor([[[3]]], dtype=torch.int32)

    with pytest.raises(RuntimeError):
        sparse_mla_naive(q, kv, indices, 0.5)


def test_sparse_mla_plan_extracts_shared_dense_run_and_indexes_tail() -> None:
    indices = torch.arange(20, dtype=torch.int32).reshape(1, 1, 20)
    indices = indices.expand(8, 1, 20).clone()

    plan = _build_sparse_mla_plans(indices.reshape(8, -1).to(torch.long))[0]

    assert plan.lq_eff == 8
    assert len(plan.dense_segments) == 1
    assert plan.dense_segments[0].start == 0
    assert plan.dense_segments[0].length == 16
    assert len(plan.indexed_tiles) == 1
    assert plan.indexed_tiles[0].valid_mask == (1 << 32) - 1
    for row in range(8):
        assert plan.indexed_tiles[0].idx[row] == (16, 17, 18, 19)


def test_sparse_mla_naive_plan_dense_and_indexed_matches_reference() -> None:
    torch.manual_seed(23)
    q = torch.randn(8, 2, 8).bfloat16()
    kv = torch.randn(24, 1, 8).bfloat16()
    indices = torch.arange(20, dtype=torch.int32).reshape(1, 1, 20)
    indices = indices.expand(8, 1, 20).clone()

    actual = sparse_mla_naive(q, kv, indices, 0.25, d_v=6, return_stats=True)
    expected = _vllm_cpu_sparse_attention_reference(q, kv, indices, 0.25, 6)

    _assert_sparse_close(actual, expected)


def test_sparse_mla_naive_all_invalid_outputs_zero_and_lse_inf() -> None:
    q = torch.randn(2, 2, 4).bfloat16()
    kv = torch.randn(3, 1, 4).bfloat16()
    indices = torch.full((2, 1, 3), -1, dtype=torch.int32)

    out, max_logits, lse = sparse_mla_naive(
        q,
        kv,
        indices,
        0.5,
        d_v=3,
        return_stats=True,
    )

    assert torch.equal(out, torch.zeros_like(out))
    assert torch.all(torch.isneginf(max_logits))
    assert torch.all(torch.isposinf(lse))


def test_flash_mla_sparse_fwd_naive_reuses_out_buffer() -> None:
    torch.manual_seed(11)
    q = torch.randn(3, 2, 6).bfloat16()
    kv = torch.randn(5, 1, 6).bfloat16()
    indices = torch.tensor([[[0, 1]], [[2, 3]], [[4, -1]]], dtype=torch.int32)
    d_v = 4

    expected = flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        0.25,
        d_v=d_v,
        return_stats=True,
    )
    out_buffer = torch.empty_like(expected[0])
    actual = flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        0.25,
        d_v=d_v,
        out=out_buffer,
        return_stats=True,
    )

    assert actual[0] is out_buffer
    _assert_sparse_close(actual, expected)


def test_flash_mla_sparse_fwd_alias_matches_naive() -> None:
    torch.manual_seed(13)
    q = torch.randn(2, 3, 8).bfloat16()
    kv = torch.randn(6, 1, 8).bfloat16()
    indices = torch.tensor([[[0, 3, 5]], [[4, -1, 2]]], dtype=torch.int32)

    actual = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        0.375,
        d_v=5,
        return_stats=True,
    )
    expected = flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        0.375,
        d_v=5,
        return_stats=True,
    )

    _assert_sparse_close(actual, expected)


@pytest.mark.skipif(not _HAS_CPP_SPARSE_MLA, reason="C++ sparse MLA extension unavailable")
def test_flash_mla_sparse_fwd_cpp_hybrid_dense_and_indexed_matches_naive() -> None:
    torch.manual_seed(29)
    q = torch.randn(8, 2, 8).bfloat16()
    kv = torch.randn(24, 1, 8).bfloat16()
    indices = torch.arange(20, dtype=torch.int32).reshape(1, 1, 20)
    indices = indices.expand(8, 1, 20).clone()
    attn_sink = torch.tensor([float("-inf"), 0.5], dtype=torch.float32)

    actual = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        0.25,
        d_v=6,
        attn_sink=attn_sink,
        return_stats=True,
    )
    expected = flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        0.25,
        d_v=6,
        attn_sink=attn_sink,
        return_stats=True,
    )

    _assert_sparse_close(actual, expected)


@pytest.mark.skipif(not _HAS_CPP_SPARSE_MLA, reason="C++ sparse MLA extension unavailable")
def test_flash_mla_sparse_fwd_cpp_dense_packqkv_fast_path_matches_naive() -> None:
    torch.manual_seed(31)
    s_q, h_q, s_kv, d_qk, d_v, topk = 16, 4, 32, 16, 16, 16
    q = torch.randn(s_q, h_q, d_qk).bfloat16()
    kv = torch.randn(s_kv, 1, d_qk).bfloat16()
    indices = torch.arange(3, 3 + topk, dtype=torch.int32).reshape(1, 1, topk)
    indices = indices.expand(s_q, 1, topk).clone()

    output_only = flash_mla_sparse_fwd(q, kv, indices, 1.0 / (d_qk**0.5), d_v=d_v)
    actual = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        1.0 / (d_qk**0.5),
        d_v=d_v,
        return_stats=True,
    )
    expected = flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        1.0 / (d_qk**0.5),
        d_v=d_v,
        return_stats=True,
    )

    assert isinstance(output_only, torch.Tensor)
    torch.testing.assert_close(output_only, expected[0], atol=1e-2, rtol=1e-2)
    _assert_sparse_close(actual, expected)


def test_sparse_mla_naive_rejects_multi_kv_head_sparse_prefill() -> None:
    q = torch.randn(2, 2, 4).bfloat16()
    kv = torch.randn(3, 2, 4).bfloat16()
    indices = torch.zeros(2, 2, 1, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="h_kv == 1"):
        sparse_mla_naive(q, kv, indices, 0.5, d_v=4)
