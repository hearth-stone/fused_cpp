# -*- coding: utf-8 -*-
"""DeepSeek V4 post-GEMM attention stage baselines.

This module mirrors the CPU prefill semantics of the second
``execute_in_parallel`` block in vLLM's DeepSeek V4 attention path:

* main Q branch: ``wq_b(qr)`` + per-head Q RMSNorm + RoPE + SWA KV cache insert;
* MLA compressor branch: save partial states + compress/norm/RoPE/cache insert;
* indexer branch: indexer ``wq_b(qr)`` + Q RoPE/weight scaling + indexer
  compressor + sparse top-k indexer.

The ``torch`` version is the precision baseline. The ``cpp`` version calls a
native fused_cpp op that intentionally uses torch C++ APIs first; it is an
optimization baseline for later scheduling and kernel fusion work, not a final
micro-kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from fused_cpp.bf16_linear import (
    PreparedBF16LinearWeight,
    prepare as _prepare_bf16_linear_weight,
)

try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        deepseek_v4_post_gemm_parallel_stage as _cpp_post_gemm_stage,
    )

    _HAS_DEEPSEEK_V4_POST_GEMM_STAGE = True
except (ImportError, AttributeError):
    _cpp_post_gemm_stage = None
    _HAS_DEEPSEEK_V4_POST_GEMM_STAGE = False

try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        deepseek_v4_post_gemm_parallel_stage_prepacked as _cpp_post_gemm_stage_prepacked,
    )

    _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED = True
except (ImportError, AttributeError):
    _cpp_post_gemm_stage_prepacked = None
    _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED = False

PostGemmStageVersion = Literal["auto", "torch", "cpp"]


@dataclass
class SWACacheState:
    """State needed by the main Q/KV RoPE and SWA cache insertion branch."""

    kv_cache: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class CompressorState:
    """Explicit tensor state for one DeepSeek V4 compressor branch."""

    ape: torch.Tensor
    state_cache: torch.Tensor
    state_slot_mapping: torch.Tensor
    token_to_req_indices: torch.Tensor
    block_table: torch.Tensor
    kv_cache: torch.Tensor
    kv_slot_mapping: torch.Tensor
    norm_weight: torch.Tensor
    compress_ratio: int
    rms_norm_eps: float


@dataclass
class SparseIndexerPrefillMetadata:
    """Single prefill chunk metadata for sparse indexer top-k."""

    cu_seq_lens: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    block_table: torch.Tensor
    topk_tokens: int


@dataclass(frozen=True)
class PreparedDeepSeekV4PostGemmWeights:
    """Prepacked weights for the two post-GEMM bf16 linear layers.

    The source weights use PyTorch linear layout ``[N, K]`` and are packed with
    the same opaque backend format as ``fused_cpp.bf16_linear``.
    """

    main_wq_b: PreparedBF16LinearWeight
    indexer_wq_b: PreparedBF16LinearWeight


@dataclass
class PostGemmStageInputs:
    """All explicit tensors needed to replay the post-GEMM stage."""

    qr: torch.Tensor
    kv: torch.Tensor
    kv_score: torch.Tensor
    indexer_kv_score: torch.Tensor
    indexer_weights: torch.Tensor
    positions: torch.Tensor
    main_wq_b_weight: torch.Tensor
    indexer_wq_b_weight: torch.Tensor
    main_cos_sin_cache: torch.Tensor
    indexer_cos_sin_cache: torch.Tensor
    swa: SWACacheState
    mla_compressor: CompressorState
    indexer_compressor: CompressorState
    topk_indices_buffer: torch.Tensor
    prefill: SparseIndexerPrefillMetadata
    main_head_dim: int
    q_eps: float
    prepared_weights: PreparedDeepSeekV4PostGemmWeights | None = None


def _linear_to_dtype(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (input_tensor.to(torch.float32) @ weight.to(torch.float32).T).to(dtype)


def _gptj_rope_apply(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor | int,
    rope_head_dim: int,
) -> torch.Tensor:
    if rope_head_dim == 0:
        return x.to(torch.float32)
    assert rope_head_dim % 2 == 0
    head_dim = x.shape[-1]
    nope = head_dim - rope_head_dim
    half = rope_head_dim // 2
    x_f = x.to(torch.float32)
    out = x_f.clone()
    rope = x_f[..., nope:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    cache = cos_sin_cache.to(torch.float32)
    if isinstance(positions, torch.Tensor) and positions.dim() >= 1:
        pos = positions.to(torch.int64)
        rows = cache[pos]
        while rows.dim() < x.dim():
            rows = rows.unsqueeze(-2)
        cos_v = rows[..., :half]
        sin_v = rows[..., half : 2 * half]
    else:
        pos_int = int(positions.item()) if isinstance(positions, torch.Tensor) else int(positions)
        row = cache[pos_int]
        cos_v = row[:half]
        sin_v = row[half : 2 * half]
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = even * cos_v - odd * sin_v
    rotated[..., 1::2] = odd * cos_v + even * sin_v
    out[..., nope:] = rotated
    return out


def _per_head_rmsnorm_no_weight(q: torch.Tensor, eps: float) -> torch.Tensor:
    q_f = q.to(torch.float32)
    return (q_f * torch.rsqrt(q_f.pow(2).mean(dim=-1, keepdim=True) + eps)).to(q.dtype)


def _qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa: SWACacheState,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> None:
    q.copy_(_per_head_rmsnorm_no_weight(q, eps))
    rope_head_dim = cos_sin_cache.shape[-1]
    nope_head_dim = q.shape[-1] - rope_head_dim
    q_pe_rot = _gptj_rope_apply(
        q[..., nope_head_dim:].contiguous(),
        cos_sin_cache,
        positions,
        rope_head_dim,
    )
    k_pe_rot = _gptj_rope_apply(
        kv[..., nope_head_dim:].unsqueeze(1).contiguous(),
        cos_sin_cache,
        positions,
        rope_head_dim,
    ).squeeze(1)
    q[..., nope_head_dim:] = q_pe_rot.to(q.dtype)
    kv[..., nope_head_dim:] = k_pe_rot.to(kv.dtype)

    if swa.kv_cache.numel() == 0:
        return
    slots = swa.slot_mapping.to(torch.int64).flatten()
    valid = slots >= 0
    if not bool(valid.any()):
        return
    valid_slots = slots[valid]
    rows = kv.to(swa.kv_cache.dtype)[valid]
    if swa.kv_cache.dim() == 2:
        swa.kv_cache[valid_slots] = rows
    elif swa.kv_cache.dim() == 3:
        block_size = swa.kv_cache.shape[1]
        swa.kv_cache[valid_slots // block_size, valid_slots % block_size] = rows
    else:
        raise AssertionError(f"swa kv_cache must be 2-D or 3-D, got {swa.kv_cache.shape}")


def _save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    state: CompressorState,
    positions: torch.Tensor,
) -> None:
    if state.state_cache.numel() == 0:
        return
    slots = state.state_slot_mapping.to(torch.int64).flatten()
    valid = slots >= 0
    if not bool(valid.any()):
        return
    valid_slots = slots[valid]
    block_size = state.state_cache.shape[1]
    state_width = state.state_cache.shape[-1] // 2
    block_idx = valid_slots // block_size
    pos_in_block = valid_slots % block_size
    ape_rows = (positions[valid].to(torch.int64) % state.compress_ratio).clamp_min(0)
    state.state_cache[block_idx, pos_in_block, :state_width] = kv[valid].to(
        state.state_cache.dtype
    )
    state.state_cache[block_idx, pos_in_block, state_width:] = (
        score[valid] + state.ape[ape_rows]
    ).to(state.state_cache.dtype)


def _kv_compress_norm_rope_insert(
    state: CompressorState,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    if state.kv_cache.numel() == 0 or state.state_cache.numel() == 0:
        return

    num_tokens = positions.shape[0]
    pos_cpu = positions.to(torch.int64).cpu()
    slot_cpu = state.state_slot_mapping.to(torch.int64).cpu()
    kv_slot_cpu = state.kv_slot_mapping.to(torch.int64).cpu()
    req_cpu = state.token_to_req_indices.to(torch.int64).cpu()
    state_block_size = state.state_cache.shape[1]
    state_width = state.state_cache.shape[-1] // 2
    head_dim = state.norm_weight.shape[0]
    coff = state_width // head_dim
    assert coff in (1, 2)
    window = coff * state.compress_ratio
    kv_block_size = state.kv_cache.shape[1]
    rms_w = state.norm_weight.to(torch.float32)
    rope_head_dim = cos_sin_cache.shape[-1]

    for i in range(num_tokens):
        slot = int(slot_cpu[i].item())
        if slot < 0:
            continue
        position = int(pos_cpu[i].item())
        if (position + 1) % state.compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_cpu[i].item())
        if kv_slot < 0:
            continue
        req_idx = int(req_cpu[i].item())
        start = position - window + 1
        kv_rows = []
        score_rows = []
        for t in range(window):
            p = start + t
            if p < 0:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=torch.float32, device=state.state_cache.device)
                )
                score_rows.append(
                    torch.full(
                        (head_dim,),
                        float("-inf"),
                        dtype=torch.float32,
                        device=state.state_cache.device,
                    )
                )
                continue
            logical_block = p // state_block_size
            logical_offset = p % state_block_size
            block_id = int(state.block_table[req_idx, logical_block].item())
            row = state.state_cache[block_id, logical_offset].to(torch.float32)
            if coff == 2 and t >= state.compress_ratio:
                kv_rows.append(row[head_dim : 2 * head_dim])
                score_rows.append(row[state_width + head_dim : state_width + 2 * head_dim])
            else:
                kv_rows.append(row[:head_dim])
                score_rows.append(row[state_width : state_width + head_dim])
        kv_stack = torch.stack(kv_rows, dim=0)
        score_stack = torch.stack(score_rows, dim=0)
        all_neg_inf = torch.isneginf(score_stack).all(dim=0, keepdim=True)
        if bool(all_neg_inf.any()):
            score_stack = torch.where(
                all_neg_inf.expand_as(score_stack),
                torch.zeros_like(score_stack),
                score_stack,
            )
        weights = torch.softmax(score_stack, dim=0)
        compressed = (kv_stack * weights).sum(dim=0)
        normed = compressed * torch.rsqrt(compressed.pow(2).mean() + state.rms_norm_eps) * rms_w
        compressed_pos = (position // state.compress_ratio) * state.compress_ratio
        rotated = _gptj_rope_apply(normed, cos_sin_cache, compressed_pos, rope_head_dim)
        state.kv_cache[kv_slot // kv_block_size, kv_slot % kv_block_size] = rotated.to(
            state.kv_cache.dtype
        )


def _run_compressor(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state: CompressorState,
    cos_sin_cache: torch.Tensor,
) -> None:
    state_width = state.ape.shape[1]
    kv, score = kv_score.split([state_width, state_width], dim=-1)
    _save_partial_states(kv, score, state, positions)
    _kv_compress_norm_rope_insert(state, positions, cos_sin_cache)


def _indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    indexer_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = _gptj_rope_apply(q, cos_sin_cache, positions, cos_sin_cache.shape[-1])
    softmax_scale = q.shape[-1] ** -0.5
    head_scale = q.shape[1] ** -0.5
    weights = indexer_weights.to(torch.float32) * softmax_scale * head_scale
    return q_rot.to(torch.bfloat16), weights


def _sparse_indexer_prefill(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    prefill: SparseIndexerPrefillMetadata,
) -> torch.Tensor:
    num_tokens = q_quant.shape[0]
    head_dim = q_quant.shape[-1]
    block_size = kv_cache.shape[1]
    topk_indices_buffer[:num_tokens] = -1

    cu_seq_lens_cpu = prefill.cu_seq_lens.to(torch.int64).cpu()
    cu_seqlen_ks_cpu = prefill.cu_seqlen_ks.to(torch.int64).cpu()
    cu_seqlen_ke_cpu = prefill.cu_seqlen_ke.to(torch.int64).cpu()
    valid_lens_cpu = cu_seqlen_ke_cpu - cu_seqlen_ks_cpu
    if valid_lens_cpu.numel() > 0 and int(valid_lens_cpu.max().item()) <= prefill.topk_tokens:
        for i in range(num_tokens):
            valid_len = int(valid_lens_cpu[i].item())
            if valid_len <= 0:
                continue
            topk_indices_buffer[i, :valid_len] = torch.arange(
                valid_len,
                dtype=torch.int32,
                device=topk_indices_buffer.device,
            )
        return topk_indices_buffer

    num_reqs = cu_seq_lens_cpu.numel() - 1
    total_seq_lens = int(cu_seq_lens_cpu[-1].item())
    k_gathered = torch.empty(
        (total_seq_lens, head_dim),
        dtype=torch.float32,
        device=q_quant.device,
    )
    for req_idx in range(num_reqs):
        seq_start = int(cu_seq_lens_cpu[req_idx].item())
        seq_end = int(cu_seq_lens_cpu[req_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len == 0:
            continue
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ids = prefill.block_table[req_idx, :num_blocks].to(torch.long)
        gathered = kv_cache.index_select(0, block_ids).reshape(
            num_blocks * block_size,
            head_dim,
        )
        k_gathered[seq_start:seq_end] = gathered[:seq_len].to(torch.float32)

    q_w = (q_quant.to(torch.float32) * weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)
    logits = F.linear(q_w, k_gathered)
    for i in range(num_tokens):
        row_start = int(cu_seqlen_ks_cpu[i].item())
        row_end = int(cu_seqlen_ke_cpu[i].item())
        valid_len = row_end - row_start
        if valid_len <= 0:
            continue
        k_take = min(prefill.topk_tokens, valid_len)
        _, idx_local = torch.topk(logits[i, row_start:row_end], k_take, dim=-1)
        topk_indices_buffer[i, :k_take] = idx_local.to(torch.int32)
    return topk_indices_buffer


def post_gemm_parallel_stage_torch_baseline(
    inputs: PostGemmStageInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the precision baseline with the same side effects as vLLM's CPU path."""
    main_num_heads = inputs.main_wq_b_weight.shape[0] // inputs.main_head_dim
    q = _linear_to_dtype(
        inputs.qr,
        inputs.main_wq_b_weight,
        inputs.qr.dtype,
    ).view(-1, main_num_heads, inputs.main_head_dim)
    _qnorm_rope_kv_insert(
        q,
        inputs.kv,
        inputs.swa,
        inputs.positions,
        inputs.main_cos_sin_cache,
        inputs.q_eps,
    )

    indexer_head_dim = inputs.indexer_compressor.norm_weight.shape[0]
    indexer_num_heads = inputs.indexer_wq_b_weight.shape[0] // indexer_head_dim
    indexer_q = _linear_to_dtype(
        inputs.qr,
        inputs.indexer_wq_b_weight,
        inputs.qr.dtype,
    ).view(-1, indexer_num_heads, indexer_head_dim)
    q_quant, scaled_weights = _indexer_q_rope_quant(
        inputs.positions,
        indexer_q,
        inputs.indexer_cos_sin_cache,
        inputs.indexer_weights,
    )

    _run_compressor(
        inputs.kv_score,
        inputs.positions,
        inputs.mla_compressor,
        inputs.main_cos_sin_cache,
    )
    _run_compressor(
        inputs.indexer_kv_score,
        inputs.positions,
        inputs.indexer_compressor,
        inputs.indexer_cos_sin_cache,
    )
    _sparse_indexer_prefill(
        q_quant,
        scaled_weights,
        inputs.indexer_compressor.kv_cache,
        inputs.topk_indices_buffer,
        inputs.prefill,
    )
    return q, inputs.topk_indices_buffer


def prepare_deepseek_v4_post_gemm_weights(
    main_wq_b_weight: torch.Tensor,
    indexer_wq_b_weight: torch.Tensor,
) -> PreparedDeepSeekV4PostGemmWeights:
    """Prepack the two post-GEMM bf16 linear weights for repeated prefill calls."""
    return PreparedDeepSeekV4PostGemmWeights(
        main_wq_b=_prepare_bf16_linear_weight(main_wq_b_weight),
        indexer_wq_b=_prepare_bf16_linear_weight(indexer_wq_b_weight),
    )


def post_gemm_parallel_stage_cpp_prepacked(
    inputs: PostGemmStageInputs,
    weights: PreparedDeepSeekV4PostGemmWeights | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the native C++ baseline with prepacked post linear weights."""
    if not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED:
        raise RuntimeError("DeepSeek V4 post-GEMM prepacked C++ stage is unavailable")
    selected_weights = weights if weights is not None else inputs.prepared_weights
    if selected_weights is None:
        raise RuntimeError(
            "post_gemm_parallel_stage_cpp_prepacked requires weights from "
            "prepare_deepseek_v4_post_gemm_weights"
        )
    return _cpp_post_gemm_stage_prepacked(
        inputs.qr,
        inputs.kv,
        inputs.kv_score,
        inputs.indexer_kv_score,
        inputs.indexer_weights,
        inputs.positions,
        selected_weights.main_wq_b.packed_weight,
        int(selected_weights.main_wq_b.k),
        int(selected_weights.main_wq_b.n),
        int(selected_weights.main_wq_b.n_padded),
        selected_weights.indexer_wq_b.packed_weight,
        int(selected_weights.indexer_wq_b.k),
        int(selected_weights.indexer_wq_b.n),
        int(selected_weights.indexer_wq_b.n_padded),
        inputs.main_cos_sin_cache,
        inputs.indexer_cos_sin_cache,
        inputs.swa.kv_cache,
        inputs.swa.slot_mapping,
        inputs.mla_compressor.ape,
        inputs.mla_compressor.state_cache,
        inputs.mla_compressor.state_slot_mapping,
        inputs.mla_compressor.token_to_req_indices,
        inputs.mla_compressor.block_table,
        inputs.mla_compressor.kv_cache,
        inputs.mla_compressor.kv_slot_mapping,
        inputs.mla_compressor.norm_weight,
        inputs.indexer_compressor.ape,
        inputs.indexer_compressor.state_cache,
        inputs.indexer_compressor.state_slot_mapping,
        inputs.indexer_compressor.token_to_req_indices,
        inputs.indexer_compressor.block_table,
        inputs.indexer_compressor.kv_cache,
        inputs.indexer_compressor.kv_slot_mapping,
        inputs.indexer_compressor.norm_weight,
        inputs.topk_indices_buffer,
        inputs.prefill.cu_seq_lens,
        inputs.prefill.cu_seqlen_ks,
        inputs.prefill.cu_seqlen_ke,
        inputs.prefill.block_table,
        int(inputs.main_head_dim),
        float(inputs.q_eps),
        int(inputs.mla_compressor.compress_ratio),
        float(inputs.mla_compressor.rms_norm_eps),
        int(inputs.indexer_compressor.compress_ratio),
        float(inputs.indexer_compressor.rms_norm_eps),
        int(inputs.prefill.topk_tokens),
    )


def post_gemm_parallel_stage_cpp(
    inputs: PostGemmStageInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the native torch-API C++ baseline."""
    if not _HAS_DEEPSEEK_V4_POST_GEMM_STAGE:
        raise RuntimeError("DeepSeek V4 post-GEMM C++ stage is unavailable")
    if inputs.prepared_weights is not None:
        return post_gemm_parallel_stage_cpp_prepacked(inputs, inputs.prepared_weights)
    return _cpp_post_gemm_stage(
        inputs.qr,
        inputs.kv,
        inputs.kv_score,
        inputs.indexer_kv_score,
        inputs.indexer_weights,
        inputs.positions,
        inputs.main_wq_b_weight,
        inputs.indexer_wq_b_weight,
        inputs.main_cos_sin_cache,
        inputs.indexer_cos_sin_cache,
        inputs.swa.kv_cache,
        inputs.swa.slot_mapping,
        inputs.mla_compressor.ape,
        inputs.mla_compressor.state_cache,
        inputs.mla_compressor.state_slot_mapping,
        inputs.mla_compressor.token_to_req_indices,
        inputs.mla_compressor.block_table,
        inputs.mla_compressor.kv_cache,
        inputs.mla_compressor.kv_slot_mapping,
        inputs.mla_compressor.norm_weight,
        inputs.indexer_compressor.ape,
        inputs.indexer_compressor.state_cache,
        inputs.indexer_compressor.state_slot_mapping,
        inputs.indexer_compressor.token_to_req_indices,
        inputs.indexer_compressor.block_table,
        inputs.indexer_compressor.kv_cache,
        inputs.indexer_compressor.kv_slot_mapping,
        inputs.indexer_compressor.norm_weight,
        inputs.topk_indices_buffer,
        inputs.prefill.cu_seq_lens,
        inputs.prefill.cu_seqlen_ks,
        inputs.prefill.cu_seqlen_ke,
        inputs.prefill.block_table,
        int(inputs.main_head_dim),
        float(inputs.q_eps),
        int(inputs.mla_compressor.compress_ratio),
        float(inputs.mla_compressor.rms_norm_eps),
        int(inputs.indexer_compressor.compress_ratio),
        float(inputs.indexer_compressor.rms_norm_eps),
        int(inputs.prefill.topk_tokens),
    )


def post_gemm_parallel_stage(
    inputs: PostGemmStageInputs,
    *,
    version: PostGemmStageVersion | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the requested post-GEMM stage baseline."""
    selected = version or os.getenv("FUSED_CPP_DEEPSEEK_V4_POST_GEMM_STAGE", "auto")
    if selected == "auto":
        selected = "cpp" if _HAS_DEEPSEEK_V4_POST_GEMM_STAGE else "torch"
    if selected == "torch":
        return post_gemm_parallel_stage_torch_baseline(inputs)
    if selected == "cpp":
        return post_gemm_parallel_stage_cpp(inputs)
    raise ValueError(f"unknown DeepSeek V4 post-GEMM stage version: {selected!r}")


__all__ = [
    "CompressorState",
    "PostGemmStageInputs",
    "PostGemmStageVersion",
    "PreparedDeepSeekV4PostGemmWeights",
    "SWACacheState",
    "SparseIndexerPrefillMetadata",
    "_HAS_DEEPSEEK_V4_POST_GEMM_STAGE",
    "_HAS_DEEPSEEK_V4_POST_GEMM_STAGE_PREPACKED",
    "post_gemm_parallel_stage_cpp_prepacked",
    "post_gemm_parallel_stage",
    "post_gemm_parallel_stage_cpp",
    "post_gemm_parallel_stage_torch_baseline",
    "prepare_deepseek_v4_post_gemm_weights",
]
