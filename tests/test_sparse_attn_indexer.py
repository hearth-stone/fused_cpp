# -*- coding: utf-8 -*-
"""Tests for the DeepSeek V4 sparse attention indexer CPU baseline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fused_cpp import cpu_sparse_attn_indexer_op
import fused_cpp.sparse_attn_indexer as sparse_attn_indexer_mod


def _fold_q_weights(q_quant: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (q_quant.float() * weights.float().unsqueeze(-1)).sum(dim=1)


def _gather_paged_k(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    block_size = kv_cache.shape[1]
    head_dim = kv_cache.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size
    gathered = kv_cache.index_select(0, block_ids[:num_blocks].to(torch.long))
    return gathered.reshape(num_blocks * block_size, head_dim)[:seq_len].float()


def _topk_for_row(
    q_w: torch.Tensor, k_rows: torch.Tensor, topk_tokens: int
) -> torch.Tensor:
    k_take = min(topk_tokens, k_rows.shape[0])
    _, indices = torch.topk(torch.mv(k_rows, q_w), k_take, dim=-1)
    return indices.to(torch.int32)


def test_cpu_sparse_attn_indexer_short_prefill_skips_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short prefill rows should fill arange indices without scoring logits."""

    def linear_should_not_run(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("short prefill should not score logits")

    monkeypatch.setattr(sparse_attn_indexer_mod.F, "linear", linear_should_not_run)

    chunk = SimpleNamespace(
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        cu_seqlen_ks=torch.tensor([0, 0, 0], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([1, 2, 3], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 3], dtype=torch.int32),
        total_seq_lens=3,
        token_start=0,
        token_end=3,
        num_reqs=1,
    )
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        prefill=SimpleNamespace(chunks=[chunk]),
    )
    topk_indices = torch.full((3, 4), -1, dtype=torch.int32)

    out = cpu_sparse_attn_indexer_op(
        q_quant=torch.ones((3, 2, 4), dtype=torch.bfloat16),
        weights=torch.ones((3, 2), dtype=torch.float32),
        kv_cache=torch.empty((1, 4, 4), dtype=torch.bfloat16),
        topk_indices_buffer=topk_indices,
        topk_tokens=4,
        attn_metadata=metadata,
    )

    expected = torch.tensor(
        [
            [0, -1, -1, -1],
            [0, 1, -1, -1],
            [0, 1, 2, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(out, expected)
    assert out is topk_indices


def test_cpu_sparse_attn_indexer_prefill_scores_paged_cache() -> None:
    """Prefill scoring should gather paged K rows and write local topk indices."""
    torch.manual_seed(3)

    q_quant = torch.randn(3, 2, 4).bfloat16()
    weights = torch.randn(3, 2)
    kv_cache = torch.randn(4, 4, 4).bfloat16()
    topk_indices = torch.full((3, 3), 99, dtype=torch.int32)
    topk_tokens = 2
    block_table = torch.tensor(
        [
            [0, 2],
            [3, 1],
        ],
        dtype=torch.int32,
    )
    chunk = SimpleNamespace(
        block_table=block_table,
        cu_seqlen_ks=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([5, 5, 8], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 5, 8], dtype=torch.int32),
        total_seq_lens=8,
        token_start=0,
        token_end=3,
        num_reqs=2,
    )
    metadata = SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        prefill=SimpleNamespace(chunks=[chunk]),
    )

    out = cpu_sparse_attn_indexer_op(
        q_quant=q_quant,
        weights=weights,
        kv_cache=kv_cache,
        topk_indices_buffer=topk_indices,
        topk_tokens=topk_tokens,
        attn_metadata=metadata,
    )

    q_w = _fold_q_weights(q_quant, weights)
    req0_k = _gather_paged_k(kv_cache, block_table[0], 5)
    req1_k = _gather_paged_k(kv_cache, block_table[1], 3)
    all_k = torch.cat([req0_k, req1_k], dim=0)
    expected = torch.full((3, 3), -1, dtype=torch.int32)
    expected[0, :topk_tokens] = _topk_for_row(q_w[0], all_k[0:5], topk_tokens)
    expected[1, :topk_tokens] = _topk_for_row(q_w[1], all_k[2:5], topk_tokens)
    expected[2, :topk_tokens] = _topk_for_row(q_w[2], all_k[5:8], topk_tokens)

    torch.testing.assert_close(out, expected)


def test_cpu_sparse_attn_indexer_decode_expands_block_table() -> None:
    """Decode should expand request block tables by decode_lens for MTP batches."""
    torch.manual_seed(5)

    q_quant = torch.randn(3, 2, 4).bfloat16()
    weights = torch.randn(3, 2)
    kv_cache = torch.randn(5, 4, 4).bfloat16()
    topk_indices = torch.full((3, 3), -1, dtype=torch.int32)
    topk_tokens = 2
    block_table = torch.tensor(
        [
            [0, 1],
            [3, 4],
        ],
        dtype=torch.int32,
    )
    decode = SimpleNamespace(
        block_table=block_table,
        seq_lens=torch.tensor([3, 6, 2], dtype=torch.int32),
        decode_lens=torch.tensor([2, 1], dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=3,
        num_prefills=0,
        decode=decode,
    )

    out = cpu_sparse_attn_indexer_op(
        q_quant=q_quant,
        weights=weights,
        kv_cache=kv_cache,
        topk_indices_buffer=topk_indices,
        topk_tokens=topk_tokens,
        attn_metadata=metadata,
    )

    q_w = _fold_q_weights(q_quant, weights)
    expected = torch.full((3, 3), -1, dtype=torch.int32)
    token0_k = _gather_paged_k(kv_cache, block_table[0], 3)
    token1_k = _gather_paged_k(kv_cache, block_table[0], 6)
    expected[0, :topk_tokens] = _topk_for_row(q_w[0], token0_k, topk_tokens)
    expected[1, :topk_tokens] = _topk_for_row(q_w[1], token1_k, topk_tokens)
    expected[2, :2] = torch.tensor([0, 1], dtype=torch.int32)

    torch.testing.assert_close(out, expected)
