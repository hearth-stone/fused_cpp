# -*- coding: utf-8 -*-
"""DeepSeek V4 sparse attention indexer CPU baselines.

This module carries the current vLLM v0.22.0-dsv4 Torch CPU fallback for
``cpu_sparse_attn_indexer_op`` into fused_cpp as a standalone baseline.  It
accepts the same metadata shape by duck typing so it can run with vLLM's
``DeepseekV32IndexerMetadata`` objects without making vLLM a package
dependency.

The public entrypoint is versioned so future native implementations can be
compared against the strict Torch baseline without changing test and benchmark
call sites.  Set ``FUSED_CPP_SPARSE_ATTN_INDEXER_VERSION`` or pass
``version=...`` explicitly.  Supported names:

* ``torch``: strict Python/Torch copy of vLLM's CPU fallback.
* ``auto``: alias for ``torch``.

The old standalone ``cpp_v0`` sparse indexer backend has been retired.  The
native post-GEMM migration lives in ``deepseek_v4_post_gemm_stage`` instead.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

_cpp_prefill_v0_impl = None
_HAS_CPP_SPARSE_ATTN_INDEXER = False

__all__ = [
    "_HAS_CPP_SPARSE_ATTN_INDEXER",
    "available_sparse_attn_indexer_versions",
    "cpu_sparse_attn_indexer_op",
    "cpu_sparse_attn_indexer_op_cpp_v0",
    "cpu_sparse_attn_indexer_op_torch_baseline",
    "sparse_attn_indexer_op",
]


def _fold_q_weights(q_quant: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Fold per-head indexer weights into Q for scalar K dot products."""
    return (q_quant.to(torch.float32) * weights.to(torch.float32).unsqueeze(-1)).sum(
        dim=1
    )


def cpu_sparse_attn_indexer_op_torch_baseline(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    attn_metadata: Any,
) -> torch.Tensor:
    """Torch baseline for vLLM DeepSeek V4 ``SparseAttnIndexer.forward``.

    Args:
        q_quant: ``[num_tokens, num_heads, head_dim]`` bf16/float tensor after
            indexer RoPE.  The CPU baseline keeps Q unquantized.
        weights: ``[num_tokens, num_heads]`` fp32 tensor with softmax/head
            scaling already folded by the indexer-Q path.
        kv_cache: ``[num_blocks, block_size, head_dim]`` paged indexer K cache.
        topk_indices_buffer: ``[max_tokens, max_topk]`` int32 buffer mutated in
            place.  Written indices are local, cache-relative positions;
            padded entries are ``-1``.
        topk_tokens: Number of top entries to keep for each token.
        attn_metadata: Object matching vLLM's
            ``DeepseekV32IndexerMetadata`` attribute contract.

    Returns:
        ``topk_indices_buffer`` after in-place update.
    """
    num_tokens = q_quant.shape[0]
    head_dim = q_quant.shape[-1]
    block_size = kv_cache.shape[1]

    topk_indices_buffer[:num_tokens] = -1

    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    q_w: torch.Tensor | None = None

    def get_q_w() -> torch.Tensor:
        nonlocal q_w
        if q_w is None:
            q_w = _fold_q_weights(q_quant, weights)
        return q_w

    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            token_start = chunk.token_start
            token_end = chunk.token_end
            num_chunk_tokens = token_end - token_start
            if num_chunk_tokens == 0:
                continue

            cu_seq_lens_cpu = chunk.cu_seq_lens.to("cpu")
            cu_seqlen_ks_cpu = chunk.cu_seqlen_ks.to("cpu")
            cu_seqlen_ke_cpu = chunk.cu_seqlen_ke.to("cpu")
            valid_lens_cpu = cu_seqlen_ke_cpu - cu_seqlen_ks_cpu
            if (
                valid_lens_cpu.numel() > 0
                and int(valid_lens_cpu.max().item()) <= topk_tokens
            ):
                for i in range(num_chunk_tokens):
                    valid_len = int(valid_lens_cpu[i].item())
                    if valid_len <= 0:
                        continue
                    topk_indices_buffer[token_start + i, :valid_len] = torch.arange(
                        valid_len,
                        dtype=torch.int32,
                        device=topk_indices_buffer.device,
                    )
                continue

            block_table_cpu = chunk.block_table.to("cpu")
            num_reqs = chunk.num_reqs
            total_seq_lens = int(chunk.total_seq_lens)
            k_gathered = torch.empty(
                (total_seq_lens, head_dim),
                dtype=torch.float32,
                device=q_quant.device,
            )
            for req_idx in range(num_reqs):
                ks = int(cu_seq_lens_cpu[req_idx].item())
                ke = int(cu_seq_lens_cpu[req_idx + 1].item())
                seq_len = ke - ks
                if seq_len == 0:
                    continue
                num_blocks = (seq_len + block_size - 1) // block_size
                block_ids = block_table_cpu[req_idx, :num_blocks].to(torch.long)
                gathered = kv_cache.index_select(0, block_ids).reshape(
                    num_blocks * block_size,
                    head_dim,
                )
                k_gathered[ks:ke] = gathered[:seq_len].to(torch.float32)

            q_w_chunk = get_q_w()[token_start:token_end]
            logits = F.linear(q_w_chunk, k_gathered)
            for i in range(num_chunk_tokens):
                ks_i = int(cu_seqlen_ks_cpu[i].item())
                ke_i = int(cu_seqlen_ke_cpu[i].item())
                valid_len = ke_i - ks_i
                if valid_len <= 0:
                    continue
                row = logits[i, ks_i:ke_i]
                k_take = min(topk_tokens, valid_len)
                _, idx_local = torch.topk(row, k_take, dim=-1)
                topk_indices_buffer[token_start + i, :k_take] = idx_local.to(
                    torch.int32
                )

    if has_decode:
        decode_metadata = attn_metadata.decode
        assert decode_metadata is not None
        block_table_d = decode_metadata.block_table.to("cpu").to(torch.long)
        seq_lens_d = decode_metadata.seq_lens.to("cpu")
        if seq_lens_d.dim() == 2:
            seq_lens_per_token = seq_lens_d.reshape(-1)
        else:
            seq_lens_per_token = seq_lens_d

        decode_lens_cpu = decode_metadata.decode_lens.to("cpu")
        if block_table_d.shape[0] == num_decode_tokens:
            per_token_block_table = block_table_d
        else:
            per_token_block_table = torch.repeat_interleave(
                block_table_d,
                decode_lens_cpu.to(torch.long),
                dim=0,
            )

        for token_idx in range(num_decode_tokens):
            seq_len = int(seq_lens_per_token[token_idx].item())
            if seq_len <= 0:
                continue
            if seq_len <= topk_tokens:
                topk_indices_buffer[token_idx, :seq_len] = torch.arange(
                    seq_len,
                    dtype=torch.int32,
                    device=topk_indices_buffer.device,
                )
                continue
            num_blocks = (seq_len + block_size - 1) // block_size
            block_ids = per_token_block_table[token_idx, :num_blocks]
            gathered = kv_cache.index_select(0, block_ids).reshape(
                num_blocks * block_size,
                head_dim,
            )
            k_t = gathered[:seq_len].to(torch.float32)
            row = F.linear(get_q_w()[token_idx], k_t)
            k_take = min(topk_tokens, seq_len)
            _, idx_local = torch.topk(row, k_take, dim=-1)
            topk_indices_buffer[token_idx, :k_take] = idx_local.to(torch.int32)

    return topk_indices_buffer


def _metadata_has_decode(attn_metadata: Any) -> bool:
    return (
        int(getattr(attn_metadata, "num_decodes", 0)) > 0
        or int(getattr(attn_metadata, "num_decode_tokens", 0)) > 0
    )


def available_sparse_attn_indexer_versions() -> tuple[str, ...]:
    """Return sparse attention indexer versions importable in this runtime."""
    return ("torch", "auto")


def _normalize_sparse_attn_indexer_version(version: str | None) -> str:
    value = (
        version
        if version is not None
        else os.environ.get("FUSED_CPP_SPARSE_ATTN_INDEXER_VERSION", "torch")
    )
    value = value.lower().replace("-", "_")
    aliases = {
        "baseline": "torch",
        "python": "torch",
        "pytorch": "torch",
        "cpp": "cpp_v0",
        "cxx": "cpp_v0",
        "native": "cpp_v0",
    }
    return aliases.get(value, value)


def cpu_sparse_attn_indexer_op_cpp_v0(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    attn_metadata: Any,
) -> torch.Tensor:
    """Retired standalone C++ implementation.

    Kept only to give callers a clear failure if they still request the old
    backend explicitly.
    """
    raise RuntimeError(
        "sparse attention indexer cpp_v0 has been retired; use version='torch' "
        "for the standalone baseline or fused_cpp.deepseek_v4_post_gemm_stage "
        "for the migrated native post-GEMM path"
    )


def cpu_sparse_attn_indexer_op(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    attn_metadata: Any,
    *,
    version: str | None = None,
) -> torch.Tensor:
    """Versioned DeepSeek V4 sparse attention indexer entrypoint."""
    selected = _normalize_sparse_attn_indexer_version(version)
    if selected == "auto":
        selected = "torch"
    if selected == "torch":
        return cpu_sparse_attn_indexer_op_torch_baseline(
            q_quant,
            weights,
            kv_cache,
            topk_indices_buffer,
            topk_tokens,
            attn_metadata,
        )
    if selected == "cpp_v0":
        return cpu_sparse_attn_indexer_op_cpp_v0(
            q_quant,
            weights,
            kv_cache,
            topk_indices_buffer,
            topk_tokens,
            attn_metadata,
        )
    raise ValueError(
        "unknown sparse attention indexer version "
        f"{selected!r}; available={available_sparse_attn_indexer_versions()}"
    )


sparse_attn_indexer_op = cpu_sparse_attn_indexer_op
