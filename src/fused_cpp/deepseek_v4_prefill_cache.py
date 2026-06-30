# -*- coding: utf-8 -*-
"""DeepSeek V4 CPU prefill cache/index helper baselines."""
from __future__ import annotations

import torch

try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        deepseek_v4_combine_topk_swa_indices as _cpp_combine_topk_swa_indices,
        deepseek_v4_dequantize_and_gather_dual_k_cache as _cpp_dual_gather,
        deepseek_v4_dequantize_and_gather_k_cache as _cpp_dequantize_and_gather_k_cache,
    )

    _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS = True
except (ImportError, AttributeError):
    _cpp_combine_topk_swa_indices = None
    _cpp_dual_gather = None
    _cpp_dequantize_and_gather_k_cache = None
    _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS = False

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def _require_backend() -> None:
    if not _HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS:
        raise RuntimeError(
            "DeepSeek V4 prefill cache/index C++ backend unavailable: "
            "fused_cpp._C is not built with the required symbols"
        )


def dequantize_and_gather_k_cache_cpp(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Gather the recent bf16 paged K-cache window into ``out`` in place."""
    _require_backend()
    assert _cpp_dequantize_and_gather_k_cache is not None
    _cpp_dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        int(block_size),
        int(offset),
    )


def dequantize_and_gather_dual_k_cache_cpp(
    out: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_block_table: torch.Tensor,
    compressed_block_size: int,
    compressed_offset: int,
    has_compressed: bool,
    swa_k_cache: torch.Tensor,
    swa_seq_lens: torch.Tensor,
    swa_gather_lens: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_block_size: int,
    swa_offset: int,
) -> None:
    """Gather compressed-cache and SWA bf16 paged K-cache regions in one call."""
    _require_backend()
    assert _cpp_dual_gather is not None
    _cpp_dual_gather(
        out,
        compressed_k_cache,
        compressed_seq_lens,
        compressed_block_table,
        int(compressed_block_size),
        int(compressed_offset),
        bool(has_compressed),
        swa_k_cache,
        swa_seq_lens,
        swa_gather_lens,
        swa_block_table,
        int(swa_block_size),
        int(swa_offset),
    )


def combine_topk_swa_indices_cpp(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine compressed-cache top-k indices with SWA local-buffer indices."""
    _require_backend()
    assert _cpp_combine_topk_swa_indices is not None
    return _cpp_combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        int(window_size),
        int(compress_ratio),
        int(topk),
        int(M),
        int(N),
    )


def dequantize_and_gather_k_cache_torch_baseline(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Torch reference matching vLLM's CPU ``cpu_dequantize_and_gather_k_cache``."""
    if k_cache.numel() == 0:
        return
    head_dim = k_cache.shape[-1]
    chunk_size = seq_lens.shape[0]
    seq_lens_cpu = seq_lens.to("cpu")
    gather_lens_cpu = gather_lens.to("cpu") if gather_lens is not None else None

    for i in range(chunk_size):
        n_full = int(seq_lens_cpu[i].item())
        if n_full == 0:
            continue
        n_to_gather = (
            int(gather_lens_cpu[i].item()) if gather_lens_cpu is not None else n_full
        )
        if n_to_gather == 0:
            continue

        num_blocks = (n_full + block_size - 1) // block_size
        block_ids = block_table[i, :num_blocks]
        gathered = k_cache.index_select(0, block_ids.to(torch.long))
        gathered = gathered.reshape(num_blocks * block_size, head_dim)
        start = max(0, n_full - n_to_gather)
        out[i, offset : offset + n_to_gather, :] = gathered[start:n_full].to(
            out.dtype
        )


def dequantize_and_gather_dual_k_cache_torch_baseline(
    out: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_block_table: torch.Tensor,
    compressed_block_size: int,
    compressed_offset: int,
    has_compressed: bool,
    swa_k_cache: torch.Tensor,
    swa_seq_lens: torch.Tensor,
    swa_gather_lens: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_block_size: int,
    swa_offset: int,
) -> None:
    """Torch reference for the dual gather optimized entrypoint."""
    if has_compressed:
        dequantize_and_gather_k_cache_torch_baseline(
            out,
            compressed_k_cache,
            compressed_seq_lens,
            None,
            compressed_block_table,
            compressed_block_size,
            compressed_offset,
        )
    dequantize_and_gather_k_cache_torch_baseline(
        out,
        swa_k_cache,
        swa_seq_lens,
        swa_gather_lens,
        swa_block_table,
        swa_block_size,
        swa_offset,
    )


def combine_topk_swa_indices_torch_baseline(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference matching vLLM's CPU ``cpu_combine_topk_swa_indices``."""
    num_tokens, _ = topk_indices.shape
    num_reqs = seq_lens.shape[0]

    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.zeros(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    base = int(query_start_loc[0].item())
    qsl = (query_start_loc - base).to(torch.long).tolist()
    seq_lens_l = seq_lens.to(torch.long).tolist()
    gather_lens_l = gather_lens.to(torch.long).tolist()
    topk_indices_i32 = topk_indices.to(torch.int32)

    for batch_idx in range(num_reqs):
        q_start = qsl[batch_idx]
        q_end = qsl[batch_idx + 1]
        query_len = q_end - q_start
        if query_len <= 0:
            continue
        seq_len = seq_lens_l[batch_idx]
        gather_len = gather_lens_l[batch_idx]
        gather_start = seq_len - gather_len
        start_pos = seq_len - query_len

        for i in range(query_len):
            tok = q_start + i
            pos = start_pos + i
            topk_len = (
                min((pos + 1) // compress_ratio, topk)
                if compress_ratio > 0
                else 0
            )
            swa_len = min(pos + 1, window_size)

            if topk_len > 0:
                src = topk_indices_i32[tok, :topk_len]
                combined_indices[tok, :topk_len] = src + (M * batch_idx)
            if swa_len > 0:
                offsets = torch.arange(
                    swa_len, dtype=torch.int32, device=topk_indices.device
                )
                base_off = M * batch_idx + N + (pos - swa_len + 1 - gather_start)
                combined_indices[tok, topk_len : topk_len + swa_len] = (
                    base_off + offsets
                )
            combined_lens[tok] = topk_len + swa_len

    return combined_indices, combined_lens


__all__ = [
    "_HAS_DEEPSEEK_V4_PREFILL_CACHE_OPS",
    "combine_topk_swa_indices_cpp",
    "combine_topk_swa_indices_torch_baseline",
    "dequantize_and_gather_dual_k_cache_cpp",
    "dequantize_and_gather_dual_k_cache_torch_baseline",
    "dequantize_and_gather_k_cache_cpp",
    "dequantize_and_gather_k_cache_torch_baseline",
]
