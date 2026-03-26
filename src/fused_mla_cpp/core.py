# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""CPU Fused MLA (Multi-head Latent Attention) pure torch implementation.

Fuses the complete MLA forward computation (projection, RoPE, KV cache write,
attention, o_proj) into a single class. All computation uses PyTorch only —
no external framework dependencies beyond torch.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── C++ extension import (graceful fallback to PyTorch) ──────────────────────
try:
    from fused_mla_cpp import _C
    _HAS_CPP = True
except ImportError:
    _C = None
    _HAS_CPP = False
    logger.warning("C++ extension not available, using PyTorch fallback")

# ── DEBUG switches (environment variable controlled) ─────────────────────────
_DEBUG_USE_ORIG_RMSNORM = os.environ.get("FUSED_MLA_USE_ORIG_RMSNORM", "0") == "1"
_DEBUG_USE_ORIG_ROPE = os.environ.get("FUSED_MLA_USE_ORIG_ROPE", "0") == "1"
if _DEBUG_USE_ORIG_RMSNORM:
    logger.warning("[DEBUG] FUSED_MLA_USE_ORIG_RMSNORM=1, using original RMSNorm")
if _DEBUG_USE_ORIG_ROPE:
    logger.warning("[DEBUG] FUSED_MLA_USE_ORIG_ROPE=1, using original RoPE")


# ── Pure PyTorch kernel implementations ──────────────────────────────────────

def _pytorch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm: normalize in float32, cast back, multiply by weight.

    Shape:
        - x: [T, D]
        - weight: [D]
        - returns: [T, D]
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_f32 = x_f32 * torch.rsqrt(variance + eps)
    return x_f32.to(orig_dtype) * weight


def _pytorch_apply_rope(
    x: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox_style: bool | None = True,
) -> torch.Tensor:
    """RoPE: supports NeoX (first/second half) and GPT-J (even/odd) styles.

    Shape:
        - x: [T, H, d_rope] or [T, 1, d_rope] (for k_pe)
        - cos_sin_cache: [max_pos, d_rope * 2]
        - positions: [T]
        - returns: same shape as x
    """
    if is_neox_style is None:
        is_neox_style = True
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def _pytorch_write_kv_cache(
    kv_c: torch.Tensor, k_pe: torch.Tensor,
    kv_cache: torch.Tensor, slot_mapping: torch.Tensor,
) -> None:
    """Write concatenated kv_c+k_pe to paged KV cache via slot_mapping.

    Shape:
        - kv_c: [T, R_kv]
        - k_pe: [T, d_rope]
        - kv_cache: [num_blocks, block_size, R_kv + d_rope]
        - slot_mapping: [T]
    """
    kv_combined = torch.cat([kv_c, k_pe], dim=-1)
    block_size = kv_cache.shape[1]
    for i in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        kv_cache[slot // block_size, slot % block_size] = kv_combined[i]


def _pytorch_gather_kv_cache(
    kv_cache: torch.Tensor, block_table: torch.Tensor,
    seq_len: int, block_size: int,
) -> torch.Tensor:
    """Gather seq_len tokens from paged KV cache.

    Shape:
        - kv_cache: [num_blocks, block_size, R_kv + d_rope]
        - block_table: [num_blocks_per_seq]
        - returns: [seq_len, R_kv + d_rope]
    """
    head_size = kv_cache.shape[2]
    gathered = torch.empty(seq_len, head_size, dtype=kv_cache.dtype, device=kv_cache.device)
    num_full_blocks = seq_len // block_size
    remainder = seq_len % block_size
    for block_idx in range(num_full_blocks):
        block_num = int(block_table[block_idx].item())
        start = block_idx * block_size
        gathered[start:start + block_size] = kv_cache[block_num]
    if remainder > 0:
        block_num = int(block_table[num_full_blocks].item())
        start = num_full_blocks * block_size
        gathered[start:start + remainder] = kv_cache[block_num, :remainder]
    return gathered


def _pytorch_concat_k_nope_k_pe(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """Broadcast k_pe to num_heads, concatenate with k_nope.

    Shape:
        - k_nope: [T, H, d_nope]
        - k_pe: [T, 1, d_rope]
        - returns: [T, H, d_nope + d_rope]
    """
    k_pe_expanded = k_pe.expand(-1, k_nope.shape[1], -1)
    return torch.cat([k_nope, k_pe_expanded], dim=-1)


def _pytorch_merge_attn_states(
    prefix_output: torch.Tensor, prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor, suffix_lse: torch.Tensor,
) -> torch.Tensor:
    """LSE merge of two attention outputs, computed in float32.

    Shape:
        - prefix_output: [T, H, d_v]
        - prefix_lse: [H, T]
        - suffix_output: [T, H, d_v]
        - suffix_lse: [H, T]
        - returns: [T, H, d_v]
    """
    p_lse = prefix_lse.transpose(0, 1).unsqueeze(-1).float()
    s_lse = suffix_lse.transpose(0, 1).unsqueeze(-1).float()
    max_lse = torch.maximum(p_lse, s_lse)
    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    return (p_scale * prefix_output.float() + s_scale * suffix_output.float()).to(prefix_output.dtype)


# ── C++ wrapper for apply_rope (normalizes is_neox_style=None) ───────────────

def _cpp_apply_rope(x, cos_sin_cache, positions, is_neox_style=True):
    return _C.apply_rope(x, cos_sin_cache, positions, is_neox_style if is_neox_style is not None else True)


def _cpp_write_kv_cache(kv_c, k_pe, kv_cache, slot_mapping):
    _C.write_kv_cache(kv_c, k_pe, kv_cache, slot_mapping)


class CPUFusedMLAImpl:
    """CPU Fused MLA pure torch implementation.

    Implements all MLA computation steps (projection -> RoPE -> KV cache ->
    attention -> o_proj). All linear transforms use F.linear, attention uses
    torch SDPA or manual softmax.

    Kernel dispatch is resolved once at __init__ time via a method table.
    If the C++ extension is available, C++ kernels are used; otherwise
    pure PyTorch fallbacks are used. Debug env vars can override individual
    kernels back to PyTorch.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str,
        # MLA-specific parameters
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: Any,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        # Try to extract kv_b_proj weights eagerly.
        self._kv_b_proj_ref: Any | None = None
        self._kv_b_proj_weight: torch.Tensor | None = None
        self._kv_b_proj_bias: torch.Tensor | None = None
        self._extract_kv_b_proj_weight(kv_b_proj)

        self.cos_sin_cache: torch.Tensor | None = None
        self.is_neox_style: bool | None = None
        self.W_UK_T: torch.Tensor | None = None
        self.W_UV: torch.Tensor | None = None

        # ── Kernel dispatch table ────────────────────────────────────────────
        if _HAS_CPP:
            self._rms_norm = _C.rms_norm
            self._apply_rope = _cpp_apply_rope
            self._write_kv_cache_cpu = _cpp_write_kv_cache
            self._gather_kv_cache = _C.gather_kv_cache
            self._concat_k_nope_k_pe = _C.concat_k_nope_k_pe
            self._merge_attn_states = _C.merge_attn_states
            self._linear_raw = _C.linear
        else:
            self._rms_norm = _pytorch_rms_norm
            self._apply_rope = _pytorch_apply_rope
            self._write_kv_cache_cpu = _pytorch_write_kv_cache
            self._gather_kv_cache = _pytorch_gather_kv_cache
            self._concat_k_nope_k_pe = _pytorch_concat_k_nope_k_pe
            self._merge_attn_states = _pytorch_merge_attn_states
            self._linear_raw = F.linear

        # Debug env var overrides (force PyTorch fallback for specific kernels)
        if _DEBUG_USE_ORIG_RMSNORM:
            self._rms_norm = _pytorch_rms_norm
        if _DEBUG_USE_ORIG_ROPE:
            self._apply_rope = _pytorch_apply_rope

    def set_attn_impl(self, attn_impl: Any) -> None:
        """No-op for compatibility with FusedMLAAttention."""

    def _extract_kv_b_proj_weight(self, kv_b_proj: Any) -> None:
        """Extract weight and bias from kv_b_proj layer eagerly."""
        weight = getattr(kv_b_proj, "weight", None)
        if weight is not None and weight.numel() > 0:
            self._kv_b_proj_weight = weight.data.clone()
        else:
            self._kv_b_proj_ref = kv_b_proj
        bias = getattr(kv_b_proj, "bias", None)
        if bias is not None:
            self._kv_b_proj_bias = bias.data.clone()

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Extract loaded weights from kv_b_proj reference, build W_UK_T and W_UV."""
        if self._kv_b_proj_ref is not None:
            weight = getattr(self._kv_b_proj_ref, "weight", None)
            if weight is not None and weight.numel() > 0:
                self._kv_b_proj_weight = weight.data.clone()
            bias = getattr(self._kv_b_proj_ref, "bias", None)
            if bias is not None:
                self._kv_b_proj_bias = bias.data.clone()
            self._kv_b_proj_ref = None

        if self._kv_b_proj_weight is None or self._kv_b_proj_weight.numel() == 0:
            logger.warning(
                "CPUFusedMLAImpl.process_weights_after_loading: "
                "kv_b_proj weight is empty, ensure weights are loaded correctly"
            )
            return

        kv_b_proj_weight = self._kv_b_proj_weight.to(act_dtype).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )

        if _HAS_CPP:
            self.W_UK_T, self.W_UV = _C.build_absorption_matrices(
                self._kv_b_proj_weight, self.num_heads,
                self.qk_nope_head_dim, self.v_head_dim,
                self.kv_lora_rank, act_dtype,
            )
        else:
            kv_b_proj_weight = kv_b_proj_weight.view(
                self.kv_lora_rank, self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            w_uk, w_uv = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            self.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
            self.W_UV = w_uv.transpose(0, 1).contiguous()

        assert self.W_UK_T is not None and self.W_UV is not None
        logger.debug(
            "CPUFusedMLAImpl.process_weights_after_loading: W_UK_T=%s, W_UV=%s",
            tuple(self.W_UK_T.shape), tuple(self.W_UV.shape),
        )


    def forward_fused(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        wrapper: Any,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Fused MLA forward: from hidden_states to attn_output."""
        num_tokens = hidden_states.shape[0]

        # ── 1. Q projection ───────────────────────────────────────────────────
        if wrapper.q_lora_rank is not None:
            # DeepSeek V3
            # TODO: support fused for V3/R1
            fused_w = wrapper.fused_qkv_a_proj
            qkv_lora = self._linear(fused_w, hidden_states)  # [T, R_q + R_kv + d_rope]
            q_c, kv_lora = qkv_lora.split(
                [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
                dim=-1,
            )  # q_c: [T, R_q],  kv_lora: [T, R_kv + d_rope]
            q_a_ln = wrapper.q_a_layernorm
            if _DEBUG_USE_ORIG_RMSNORM:
                q_c = q_a_ln(q_c)  # [T, R_q]
            else:
                q_c = self._rms_norm(q_c, q_a_ln.weight, q_a_ln.variance_epsilon)  # [T, R_q]
            q = self._linear(wrapper.q_b_proj, q_c)  # [T, H * d_qk]
        else:
            # DeepSeek V2
            kv_lora = self._linear(wrapper.kv_a_proj_with_mqa, hidden_states)  # [T, R_kv + d_rope]
            q = self._linear(wrapper.q_proj, hidden_states)  # [T, H * d_qk]

        # ── 2. KV compressed projection + kv_a_layernorm ──────────────────────
        kv_c, k_pe = kv_lora.split(
            [wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1
        )  # kv_c: [T, R_kv],  k_pe: [T, d_rope]
        kv_a_ln = wrapper.kv_a_layernorm
        if _DEBUG_USE_ORIG_RMSNORM:
            kv_c_normed = kv_a_ln(kv_c)  # [T, R_kv]
        else:
            kv_c_normed = self._rms_norm(kv_c, kv_a_ln.weight, kv_a_ln.variance_epsilon)  # [T, R_kv]

        # ── 3. reshape + RoPE ─────────────────────────────────────────────────
        q = q.view(num_tokens, wrapper.num_heads, wrapper.qk_head_dim)  # [T, H, d_qk]
        k_pe = k_pe.unsqueeze(1)  # [T, 1, d_rope]

        if wrapper.rotary_emb is not None:
            if self.cos_sin_cache is None:
                self.cos_sin_cache = wrapper.rotary_emb.cos_sin_cache
            if self.is_neox_style is None:
                self.is_neox_style = getattr(wrapper.rotary_emb, "is_neox_style", True)
            assert self.cos_sin_cache is not None
            cos_sin_cache = self.cos_sin_cache
            if _DEBUG_USE_ORIG_ROPE:
                q[..., wrapper.qk_nope_head_dim:], k_pe = wrapper.rotary_emb(
                    positions, q[..., wrapper.qk_nope_head_dim:], k_pe
                )  # q: [T, H, d_qk],  k_pe: [T, 1, d_rope]
            else:
                q_rope = q[..., wrapper.qk_nope_head_dim:]  # [T, H, d_rope]
                q[..., wrapper.qk_nope_head_dim:] = self._apply_rope(
                    q_rope, cos_sin_cache, positions, self.is_neox_style
                )  # in-place update q[..., d_nope:]: [T, H, d_rope]
                k_pe = self._apply_rope(
                    k_pe, cos_sin_cache, positions, self.is_neox_style
                )  # [T, 1, d_rope]

        # ── 4. Write to KV cache ──────────────────────────────────────────────
        # kv_c_normed: [T, R_kv],  k_pe.squeeze(1): [T, d_rope]
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if kv_cache.numel() > 0 and slot_mapping is not None:
            self._write_kv_cache_cpu(
                kv_c_normed, k_pe.squeeze(1), kv_cache, slot_mapping.flatten(),
            )

        # ── 5. Attention computation ──────────────────────────────────────────
        num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", None) or 0
        has_decode = (getattr(attn_metadata, "num_decodes", None) or 0) > 0
        has_prefill = (getattr(attn_metadata, "num_prefills", None) or 0) > 0

        attn_output = torch.zeros(
            num_tokens, wrapper.num_heads * wrapper.v_head_dim,
            dtype=hidden_states.dtype, device=hidden_states.device,
        )  # [T, H * d_v]

        if has_prefill:
            self._forward_prefill_torch(
                q=q[num_decode_tokens:],          # [T_p, H, d_qk]
                kv_c_normed=kv_c_normed[num_decode_tokens:],  # [T_p, R_kv]
                k_pe=k_pe[num_decode_tokens:],    # [T_p, 1, d_rope]
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=attn_output[num_decode_tokens:],  # [T_p, H * d_v]
            )

        if has_decode:
            decode_q = q[:num_decode_tokens]  # [T_d, H, d_qk]
            mqa_q_nope, mqa_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )  # mqa_q_nope: [T_d, H, d_nope],  mqa_q_pe: [T_d, H, d_rope]
            assert self.W_UK_T is not None, "W_UK_T not initialized"
            # W_UK_T: [H, d_nope, R_kv];  bmm over H heads: [H, T_d, d_nope] x [H, d_nope, R_kv]
            mqa_ql_nope = torch.bmm(
                mqa_q_nope.transpose(0, 1), self.W_UK_T,
            ).transpose(0, 1)  # [T_d, H, R_kv]

            attn_out = self._forward_decode_torch(
                q_nope_proj=mqa_ql_nope, q_pe=mqa_q_pe,
                kv_cache=kv_cache, attn_metadata=attn_metadata,
            )  # [T_d, H, R_kv]

            assert self.W_UV is not None, "W_UV not initialized"
            # W_UV: [H, R_kv, d_v];  bmm over H heads: [H, T_d, R_kv] x [H, R_kv, d_v]
            decode_v = torch.bmm(
                attn_out.transpose(0, 1), self.W_UV,
            ).transpose(0, 1)  # [T_d, H, d_v]
            attn_output[:num_decode_tokens] = decode_v.reshape(
                num_decode_tokens, wrapper.num_heads * wrapper.v_head_dim
            )  # [T_d, H * d_v]

        # ── 6. o_proj ─────────────────────────────────────────────────────────
        # attn_output: [T, H * d_v]  ->  returns: [T, D]
        return self._linear(wrapper.o_proj, attn_output)


    # ── Prefill attention ─────────────────────────────────────────────────────

    def _forward_prefill_torch(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata, output):
        """Prefill stage MHA forward."""
        prefill_metadata = getattr(attn_metadata, "prefill", None)
        if prefill_metadata is None:
            return

        has_context = getattr(prefill_metadata, "chunked_context", None) is not None

        kv_nope = self._kv_b_proj_forward(kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )  # [T_p, H, d_nope + d_v]
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # k_nope: [T_p, H, d_nope],  v: [T_p, H, d_v]
        k = self._concat_k_nope_k_pe(k_nope, k_pe)  # [T_p, H, d_qk]

        cu_seqlens_q = prefill_metadata.query_start_loc
        output_prefill = self._varlen_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=prefill_metadata.max_query_len,
            causal=True, return_softmax_lse=has_context,
        )  # [T_p, H, d_v] or tuple([T_p, H, d_v], [H, T_p])

        if has_context:
            assert isinstance(output_prefill, tuple)
            suffix_output, suffix_lse = output_prefill
            # suffix_output: [T_p, H, d_qk],  suffix_lse: [H, T_p]
            assert suffix_lse is not None
            suffix_output = suffix_output[..., :self.v_head_dim]  # [T_p, H, d_v]

            context_output, context_lse = self._compute_prefill_context(
                q, kv_cache, attn_metadata
            )  # context_output: [T_p, H, d_v],  context_lse: [H, T_p]
            merged = self._merge_attn_states(
                prefix_output=context_output, prefix_lse=context_lse,
                suffix_output=suffix_output, suffix_lse=suffix_lse,
            )  # [T_p, H, d_v]
            output.copy_(merged.flatten(start_dim=-2))  # [T_p, H * d_v]
        else:
            assert isinstance(output_prefill, torch.Tensor)
            output.copy_(output_prefill[..., :v.shape[-1]].flatten(start_dim=-2))  # [T_p, H * d_v]

    def _compute_prefill_context(self, q, kv_cache, attn_metadata):
        """Gather context tokens from KV cache, compute prefill context attention."""
        prefill_metadata = attn_metadata.prefill
        block_size = kv_cache.shape[1]
        block_table = prefill_metadata.block_table
        num_prefills = block_table.shape[0]

        cu_seq_lens_first = prefill_metadata.chunked_context.cu_seq_lens[0]
        context_lens = (cu_seq_lens_first[1:] - cu_seq_lens_first[:-1]).cpu()

        cu_seqlens_k = torch.zeros(num_prefills + 1, dtype=torch.int32, device=q.device)
        for i in range(num_prefills):
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(context_lens[i])

        total_context_tokens = int(cu_seqlens_k[-1].item())
        if total_context_tokens == 0:
            num_tokens, num_heads = q.shape[0], q.shape[1]
            return (
                torch.zeros(num_tokens, num_heads, self.v_head_dim, dtype=q.dtype, device=q.device),  # [T_p, H, d_v]
                torch.full((num_heads, num_tokens), float("-inf"), dtype=torch.float32, device=q.device),  # [H, T_p]
            )

        gathered_kv = torch.empty(
            total_context_tokens, self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=kv_cache.dtype, device=q.device,
        )  # [T_ctx, R_kv + d_rope]
        for i in range(num_prefills):
            ctx_len = int(context_lens[i])
            if ctx_len == 0:
                continue
            dst_start = int(cu_seqlens_k[i].item())
            gathered_kv[dst_start:dst_start + ctx_len] = self._gather_kv_cache(
                kv_cache, block_table[i], ctx_len, block_size
            )  # fills gathered_kv[dst_start:dst_start+ctx_len]: [ctx_len, R_kv + d_rope]

        kv_c_ctx = gathered_kv[:, :self.kv_lora_rank]  # [T_ctx, R_kv]
        k_pe_ctx = gathered_kv[:, self.kv_lora_rank:].unsqueeze(1)  # [T_ctx, 1, d_rope]

        kv_nope = self._kv_b_proj_forward(kv_c_ctx).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )  # [T_ctx, H, d_nope + d_v]
        k_nope_ctx, v_ctx = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )  # k_nope_ctx: [T_ctx, H, d_nope],  v_ctx: [T_ctx, H, d_v]
        k_ctx = self._concat_k_nope_k_pe(k_nope_ctx, k_pe_ctx)  # [T_ctx, H, d_qk]

        result = self._varlen_attention(
            q=q, k=k_ctx, v=v_ctx,
            cu_seqlens_q=prefill_metadata.query_start_loc,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=int(context_lens.max().item()),
            causal=False, return_softmax_lse=True,
        )  # tuple([T_p, H, d_v], [H, T_p])
        assert isinstance(result, tuple)
        context_output, context_lse = result
        # context_output: [T_p, H, d_v],  context_lse: [H, T_p]
        assert context_lse is not None
        return context_output, context_lse


    # ── Decode attention ──────────────────────────────────────────────────────

    def _forward_decode_torch(self, q_nope_proj, q_pe, kv_cache, attn_metadata):
        """Decode stage MQA forward (absorption trick)."""
        decode_metadata = getattr(attn_metadata, "decode", None)
        if decode_metadata is None:
            return torch.zeros_like(q_nope_proj)

        if _HAS_CPP:
            return _C.forward_decode(
                q_nope_proj, q_pe, kv_cache,
                decode_metadata.block_table, decode_metadata.seq_lens,
                self.scale, self.kv_lora_rank, self.qk_rope_head_dim,
            )

        batch_size = q_nope_proj.shape[0]
        num_heads = q_nope_proj.shape[1]
        block_table = decode_metadata.block_table
        seq_lens = decode_metadata.seq_lens
        seq_lens_cpu = seq_lens.cpu()
        block_size = kv_cache.shape[1]

        output = torch.zeros(
            batch_size, num_heads, self.kv_lora_rank,
            dtype=q_nope_proj.dtype, device=q_nope_proj.device,
        )  # [B, H, R_kv]

        for b in range(batch_size):
            seq_len = int(seq_lens_cpu[b].item())
            if seq_len == 0:
                continue
            gathered_kv = self._gather_kv_cache(
                kv_cache, block_table[b], seq_len, block_size
            )  # [seq_len, R_kv + d_rope]
            kv_c = gathered_kv[:, :self.kv_lora_rank]   # [seq_len, R_kv]
            k_pe_seq = gathered_kv[:, self.kv_lora_rank:]  # [seq_len, d_rope]
            # q_nope_proj[b]: [H, R_kv],  kv_c.T: [R_kv, seq_len]
            # q_pe[b]: [H, d_rope],  k_pe_seq.T: [d_rope, seq_len]
            attn_scores = (
                torch.matmul(q_nope_proj[b], kv_c.T)
                + torch.matmul(q_pe[b], k_pe_seq.T)
            ) * self.scale  # [H, seq_len]
            attn_weights = F.softmax(attn_scores, dim=-1)  # [H, seq_len]
            output[b] = torch.matmul(attn_weights, kv_c)  # [H, R_kv]

        return output  # [B, H, R_kv]

    # ── varlen_attention ──────────────────────────────────────────────────────

    def _varlen_attention(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                          max_seqlen_q, max_seqlen_k, causal=True,
                          return_softmax_lse=False):
        """Variable-length multi-head attention.

        Shape:
            - q: [T, H, d_qk]
            - k: [T_k, H, d_qk]
            - v: [T_k, H, d_v]
            - cu_seqlens_q: [batch_size + 1]
            - cu_seqlens_k: [batch_size + 1]
            - returns: [T, H, d_v]  or  tuple([T, H, d_v], [H, T]) when return_softmax_lse=True
        """
        if _HAS_CPP:
            return _C.varlen_attention(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, self.scale,
                causal, return_softmax_lse,
            )

        batch_size = cu_seqlens_q.shape[0] - 1
        total_q_tokens = q.shape[0]
        num_heads = q.shape[1]
        v_head_dim = v.shape[2]
        qk_head_dim = q.shape[2]

        output = torch.zeros(
            total_q_tokens, num_heads, v_head_dim,
            dtype=q.dtype, device=q.device,
        )  # [T, H, d_v]
        lse = (
            torch.full((num_heads, total_q_tokens), float("-inf"),
                        dtype=torch.float32, device=q.device)
            if return_softmax_lse else None
        )  # [H, T] or None

        cu_seqlens_q_cpu = cu_seqlens_q.cpu().numpy()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().numpy()

        for i in range(batch_size):
            q_start = cu_seqlens_q_cpu[i]
            q_end = cu_seqlens_q_cpu[i + 1]
            k_start = cu_seqlens_k_cpu[i]
            k_end = cu_seqlens_k_cpu[i + 1]
            if q_end <= q_start or k_end <= k_start:
                continue

            q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)  # [1, H, T_qi, d_qk]
            k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)  # [1, H, T_ki, d_qk]
            v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)  # [1, H, T_ki, d_v]

            if qk_head_dim != v_head_dim:
                v_i = F.pad(v_i, [0, qk_head_dim - v_head_dim], value=0.0)  # [1, H, T_ki, d_qk]

            if return_softmax_lse:
                attn_scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * self.scale  # [1, H, T_qi, T_ki]
                if causal:
                    seq_q = q_end - q_start
                    seq_k = k_end - k_start
                    q_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)
                    k_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)
                    causal_mask = q_idx >= k_idx
                    attn_scores = attn_scores.masked_fill(
                        ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )  # [1, H, T_qi, T_ki]
                lse_i = torch.logsumexp(attn_scores, dim=-1)  # [1, H, T_qi]
                assert lse is not None
                lse[:, q_start:q_end] = lse_i.squeeze(0)  # fills lse[:, q_start:q_end]: [H, T_qi]
                attn_weights = torch.softmax(attn_scores, dim=-1)  # [1, H, T_qi, T_ki]
                output_i = torch.matmul(attn_weights, v_i)  # [1, H, T_qi, d_qk]
            else:
                output_i = F.scaled_dot_product_attention(
                    q_i, k_i, v_i, attn_mask=None, dropout_p=0.0,
                    is_causal=causal, scale=self.scale,
                )  # [1, H, T_qi, d_qk]

            output[q_start:q_end] = output_i[0, :, :, :v_head_dim].transpose(0, 1)  # [T_qi, H, d_v]

        if return_softmax_lse:
            return output, lse  # [T, H, d_v], [H, T]
        return output  # [T, H, d_v]

    # ── Helper methods ────────────────────────────────────────────────────────

    def _kv_b_proj_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear transform using kv_b_proj weights."""
        assert self._kv_b_proj_weight is not None, "kv_b_proj weight not initialized"
        return self._linear_raw(x, self._kv_b_proj_weight, self._kv_b_proj_bias)

    @staticmethod
    def _linear(layer: Any, x: torch.Tensor) -> torch.Tensor:
        """Pure torch linear transform (F.linear)."""
        bias = getattr(layer, "bias", None)
        if getattr(layer, "skip_bias_add", False):
            bias = None
        weight = getattr(layer, "weight", None)
        if weight is None or weight.numel() == 0:
            raise RuntimeError(
                f"Layer {type(layer).__name__} weight is empty or missing, "
                f"cannot perform linear transform"
            )
        return F.linear(x, weight, bias)

    @staticmethod
    def debug_linear_layer(name: str, layer: Any) -> None:
        """Debug linear layer information."""
        weight = getattr(layer, "weight", None)
        bias = getattr(layer, "bias", None)
        logger.debug(
            "%s: type=%s, has_cpu_linear=%s, weight_shape=%s, "
            "weight_numel=%s, has_bias=%s, skip_bias_add=%s",
            name, type(layer).__name__,
            hasattr(layer, "cpu_linear"),
            None if weight is None else tuple(weight.shape),
            None if weight is None else weight.numel(),
            bias is not None,
            getattr(layer, "skip_bias_add", None),
        )
