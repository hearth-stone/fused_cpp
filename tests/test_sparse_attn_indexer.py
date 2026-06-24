# -*- coding: utf-8 -*-
"""Tests for the DeepSeek V4 sparse attention indexer CPU baseline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fused_cpp import (
    _HAS_CPP_SPARSE_ATTN_INDEXER,
    available_sparse_attn_indexer_versions,
    cpu_sparse_attn_indexer_op,
    cpu_sparse_attn_indexer_op_cpp_v0,
    cpu_sparse_attn_indexer_op_torch_baseline,
)
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


def _make_prefill_metadata(
    *,
    block_table: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    total_seq_lens: int,
    token_start: int,
    token_end: int,
    num_reqs: int,
) -> SimpleNamespace:
    chunk = SimpleNamespace(
        block_table=block_table,
        cu_seqlen_ks=cu_seqlen_ks,
        cu_seqlen_ke=cu_seqlen_ke,
        cu_seq_lens=cu_seq_lens,
        total_seq_lens=total_seq_lens,
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
    )
    return SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=num_reqs,
        prefill=SimpleNamespace(chunks=[chunk]),
    )


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


def test_cpu_sparse_attn_indexer_version_registry() -> None:
    """Version registry should expose a stable Python baseline and auto selector."""
    versions = available_sparse_attn_indexer_versions()

    assert "torch" in versions
    assert "auto" in versions
    assert ("cpp_v0" in versions) == _HAS_CPP_SPARSE_ATTN_INDEXER


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
    metadata = _make_prefill_metadata(
        block_table=block_table,
        cu_seqlen_ks=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([5, 5, 8], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 5, 8], dtype=torch.int32),
        total_seq_lens=8,
        token_start=0,
        token_end=3,
        num_reqs=2,
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


@pytest.mark.skipif(
    not _HAS_CPP_SPARSE_ATTN_INDEXER,
    reason="sparse attention indexer C++ extension is unavailable",
)
@pytest.mark.parametrize("version", ["cpp_v0", "auto"])
def test_cpu_sparse_attn_indexer_cpp_v0_matches_torch_prefill(version: str) -> None:
    """The initial C++ prefill path should exactly match the Torch baseline."""
    torch.manual_seed(11)

    q_quant = torch.randn(5, 3, 7).bfloat16()
    weights = torch.randn(5, 3)
    kv_cache = torch.randn(8, 4, 7).bfloat16()
    block_table = torch.tensor(
        [
            [1, 3, 0],
            [5, 2, 7],
        ],
        dtype=torch.int32,
    )
    metadata = _make_prefill_metadata(
        block_table=block_table,
        cu_seqlen_ks=torch.tensor([0, 1, 4, 6, 8], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([5, 5, 8, 8, 10], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 6, 10], dtype=torch.int32),
        total_seq_lens=10,
        token_start=0,
        token_end=5,
        num_reqs=2,
    )
    topk_tokens = 3

    ref_buffer = torch.full((5, 4), 77, dtype=torch.int32)
    actual_buffer = torch.full((5, 4), 77, dtype=torch.int32)
    ref = cpu_sparse_attn_indexer_op_torch_baseline(
        q_quant,
        weights,
        kv_cache,
        ref_buffer,
        topk_tokens,
        metadata,
    )
    actual = cpu_sparse_attn_indexer_op(
        q_quant,
        weights,
        kv_cache,
        actual_buffer,
        topk_tokens,
        metadata,
        version=version,
    )
    assert actual is actual_buffer
    torch.testing.assert_close(actual, ref)


@pytest.mark.skipif(
    not _HAS_CPP_SPARSE_ATTN_INDEXER,
    reason="sparse attention indexer C++ extension is unavailable",
)
def test_cpu_sparse_attn_indexer_cpp_v0_writes_sorted_topk() -> None:
    """C++ topk output should stay sorted by descending score."""
    q_quant = torch.ones((1, 1, 1), dtype=torch.float32)
    weights = torch.ones((1, 1), dtype=torch.float32)
    kv_cache = torch.tensor(
        [
            [[0.1], [0.9], [0.3], [0.8]],
            [[0.2], [0.7], [0.4], [0.6]],
        ],
        dtype=torch.float32,
    )
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = _make_prefill_metadata(
        block_table=block_table,
        cu_seqlen_ks=torch.tensor([0], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([8], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 8], dtype=torch.int32),
        total_seq_lens=8,
        token_start=0,
        token_end=1,
        num_reqs=1,
    )
    topk_indices = torch.full((1, 4), -1, dtype=torch.int32)

    out = cpu_sparse_attn_indexer_op(
        q_quant,
        weights,
        kv_cache,
        topk_indices,
        4,
        metadata,
        version="cpp_v0",
    )

    torch.testing.assert_close(
        out,
        torch.tensor([[1, 3, 5, 7]], dtype=torch.int32),
    )


@pytest.mark.skipif(
    not _HAS_CPP_SPARSE_ATTN_INDEXER,
    reason="sparse attention indexer C++ extension is unavailable",
)
def test_cpu_sparse_attn_indexer_cpp_v0_respects_output_column_stride() -> None:
    """The custom topk path should support non-contiguous output views."""
    q_quant = torch.ones((1, 1, 1), dtype=torch.float32)
    weights = torch.ones((1, 1), dtype=torch.float32)
    kv_cache = torch.tensor(
        [
            [[0.1], [0.9], [0.3], [0.8]],
            [[0.2], [0.7], [0.4], [0.6]],
        ],
        dtype=torch.float32,
    )
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = _make_prefill_metadata(
        block_table=block_table,
        cu_seqlen_ks=torch.tensor([0], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([8], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 8], dtype=torch.int32),
        total_seq_lens=8,
        token_start=0,
        token_end=1,
        num_reqs=1,
    )
    backing = torch.full((1, 8), -1, dtype=torch.int32)
    topk_indices = backing[:, ::2]

    out = cpu_sparse_attn_indexer_op(
        q_quant,
        weights,
        kv_cache,
        topk_indices,
        4,
        metadata,
        version="cpp_v0",
    )

    assert out is topk_indices
    torch.testing.assert_close(
        topk_indices,
        torch.tensor([[1, 3, 5, 7]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        backing[:, 1::2],
        torch.full((1, 4), -1, dtype=torch.int32),
    )


@pytest.mark.skipif(
    not _HAS_CPP_SPARSE_ATTN_INDEXER,
    reason="sparse attention indexer C++ extension is unavailable",
)
def test_cpu_sparse_attn_indexer_cpp_v0_rejects_decode() -> None:
    """cpp_v0 is intentionally prefill-only while decode is out of scope."""
    metadata = SimpleNamespace(num_decodes=1, num_decode_tokens=1, num_prefills=0)

    with pytest.raises(NotImplementedError, match="prefill only"):
        cpu_sparse_attn_indexer_op_cpp_v0(
            torch.ones((1, 1, 4), dtype=torch.bfloat16),
            torch.ones((1, 1), dtype=torch.float32),
            torch.ones((1, 4, 4), dtype=torch.bfloat16),
            torch.full((1, 2), -1, dtype=torch.int32),
            2,
            metadata,
        )


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
