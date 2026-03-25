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

# ── DEBUG switches (environment variable controlled) ─────────────────────────
_DEBUG_USE_ORIG_RMSNORM = os.environ.get("FUSED_MLA_USE_ORIG_RMSNORM", "0") == "1"
_DEBUG_USE_ORIG_ROPE = os.environ.get("FUSED_MLA_USE_ORIG_ROPE", "0") == "1"
if _DEBUG_USE_ORIG_RMSNORM:
    logger.warning("[DEBUG] FUSED_MLA_USE_ORIG_RMSNORM=1, using original RMSNorm")
if _DEBUG_USE_ORIG_ROPE:
    logger.warning("[DEBUG] FUSED_MLA_USE_ORIG_ROPE=1, using original RoPE")


class CPUFusedMLAImpl:
    """CPU Fused MLA pure torch implementation.

    Implements all MLA computation steps (projection -> RoPE -> KV cache ->
    attention -> o_proj). All linear transforms use F.linear, attention uses
    torch SDPA or manual softmax.
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

        # Try to extract kv_b_proj weights eagerly.  If the weight is
        # already populated (numel > 0) we clone it now; otherwise we keep
        # a reference so process_weights_after_loading can read it later.
        self._kv_b_proj_ref: Any | None = None
        self._kv_b_proj_weight: torch.Tensor | None = None
        self._kv_b_proj_bias: torch.Tensor | None = None
        self._extract_kv_b_proj_weight(kv_b_proj)

        # cos_sin_cache managed by this class
        self.cos_sin_cache: torch.Tensor | None = None
        # RoPE style: False = GPT-J (interleaved), True = NeoX (first/second half)
        self.is_neox_style: bool | None = None

        # W_UK_T and W_UV for decode path absorption trick
        self.W_UK_T: torch.Tensor | None = None
        self.W_UV: torch.Tensor | None = None

    def set_attn_impl(self, attn_impl: Any) -> None:
        """Receive external attention impl (compatibility with FusedMLAAttention).

        The pure torch implementation is self-contained and does not need an
        external impl, so this is a no-op.

        :param attn_impl: Attention implementation (ignored here)
        """

    def _extract_kv_b_proj_weight(self, kv_b_proj: Any) -> None:
        """Extract weight and bias from kv_b_proj layer eagerly.

        If the weight tensor is already populated (numel > 0), clone it now.
        Otherwise, save the reference for process_weights_after_loading.

        :param kv_b_proj: Linear layer object (any type with a weight attr)
        """
        weight = getattr(kv_b_proj, "weight", None)
        if weight is not None and weight.numel() > 0:
            self._kv_b_proj_weight = weight.data.clone()
        else:
            # Weights not yet loaded; keep reference for later extraction.
            self._kv_b_proj_ref = kv_b_proj
        bias = getattr(kv_b_proj, "bias", None)
        if bias is not None:
            self._kv_b_proj_bias = bias.data.clone()

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Extract loaded weights from kv_b_proj reference, build W_UK_T and W_UV.

        Called after model weights are loaded. At this point kv_b_proj.weight
        contains the real weights. We clone them and release the reference.
        """
        # Extract real weights from saved reference
        if self._kv_b_proj_ref is not None:
            weight = getattr(self._kv_b_proj_ref, "weight", None)
            if weight is not None and weight.numel() > 0:
                self._kv_b_proj_weight = weight.data.clone()
            bias = getattr(self._kv_b_proj_ref, "bias", None)
            if bias is not None:
                self._kv_b_proj_bias = bias.data.clone()
            # Release reference to avoid circular dependencies
            self._kv_b_proj_ref = None

        if self._kv_b_proj_weight is None or self._kv_b_proj_weight.numel() == 0:
            logger.warning(
                "CPUFusedMLAImpl.process_weights_after_loading: "
                "kv_b_proj weight is empty, ensure weights are loaded correctly"
            )
            return

        # kv_b_proj.weight: [out_features, kv_lora_rank]
        # Transpose to: [kv_lora_rank, out_features]
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

        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        w_uk, w_uv = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        # W_UK_T: [num_heads, qk_nope_head_dim, kv_lora_rank]
        self.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
        # W_UV: [num_heads, kv_lora_rank, v_head_dim]
        self.W_UV = w_uv.transpose(0, 1).contiguous()

        assert self.W_UK_T is not None and self.W_UV is not None
        logger.debug(
            "CPUFusedMLAImpl.process_weights_after_loading: W_UK_T=%s, W_UV=%s",
            tuple(self.W_UK_T.shape),
            tuple(self.W_UV.shape),
        )

    def forward_fused(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        wrapper: Any,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Fused MLA forward: from hidden_states to attn_output.

        Full MLA computation pipeline. KV cache management is handled
        externally via attn_metadata.slot_mapping.

        Pipeline:
        1. Q projection (supports both with/without q_lora_rank)
        2. KV compressed projection + kv_a_layernorm
        3. RoPE (rotary position embedding on q_rope and k_pe)
        4. Write to KV cache (via slot_mapping)
        5. Attention (prefill: MHA, decode: MQA absorption)
        6. o_proj
        7. Return attn_output

        Args:
            hidden_states: Input hidden states, shape = [num_tokens, hidden_size]
            positions: Position indices, shape = [num_tokens]
            wrapper: Wrapper object providing all MLA sub-modules
            kv_cache: Paged KV cache, shape = [num_blocks, block_size, head_size]
            attn_metadata: Attention metadata (slot_mapping, block_table, etc.)

        Returns:
            attn_output, shape = [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]

        # ── 1. Q projection ───────────────────────────────────────────────────
        if wrapper.q_lora_rank is not None:
            # With q_lora_rank: fused_qkv_a_proj -> split -> q_a_layernorm -> q_b_proj
            fused_w = wrapper.fused_qkv_a_proj
            qkv_lora = self._linear(fused_w, hidden_states)
            q_c, kv_lora = qkv_lora.split(
                [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
                dim=-1,
            )
            q_a_ln = wrapper.q_a_layernorm
            # DEBUG: optionally use original RMSNorm
            if _DEBUG_USE_ORIG_RMSNORM:
                q_c = q_a_ln(q_c)
            else:
                q_c = self._rms_norm(q_c, q_a_ln.weight, q_a_ln.variance_epsilon)
            q_b_w = wrapper.q_b_proj
            q = self._linear(q_b_w, q_c)
        else:
            # Without q_lora_rank: direct q_proj
            kv_a_w = wrapper.kv_a_proj_with_mqa
            kv_lora = self._linear(kv_a_w, hidden_states)
            q_w = wrapper.q_proj
            q = self._linear(q_w, hidden_states)

        # ── 2. KV compressed projection + kv_a_layernorm ──────────────────────
        kv_c, k_pe = kv_lora.split(
            [wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1
        )
        kv_a_ln = wrapper.kv_a_layernorm
        # DEBUG: optionally use original RMSNorm
        if _DEBUG_USE_ORIG_RMSNORM:
            kv_c_normed = kv_a_ln(kv_c)
        else:
            kv_c_normed = self._rms_norm(kv_c, kv_a_ln.weight, kv_a_ln.variance_epsilon)

        # ── 3. reshape + RoPE ─────────────────────────────────────────────────
        # q: [num_tokens, num_heads, qk_head_dim]
        q = q.view(num_tokens, wrapper.num_heads, wrapper.qk_head_dim)
        # k_pe: [num_tokens, 1, qk_rope_head_dim]
        k_pe = k_pe.unsqueeze(1)

        if wrapper.rotary_emb is not None:
            if self.cos_sin_cache is None:
                self.cos_sin_cache = wrapper.rotary_emb.cos_sin_cache
            if self.is_neox_style is None:
                self.is_neox_style = getattr(
                    wrapper.rotary_emb, "is_neox_style", True
                )
            assert self.cos_sin_cache is not None
            cos_sin_cache = self.cos_sin_cache
            # DEBUG: optionally use original RoPE
            if _DEBUG_USE_ORIG_ROPE:
                q[..., wrapper.qk_nope_head_dim:], k_pe = wrapper.rotary_emb(
                    positions, q[..., wrapper.qk_nope_head_dim:], k_pe
                )
            else:
                q_rope = q[..., wrapper.qk_nope_head_dim:]
                q[..., wrapper.qk_nope_head_dim:] = self._apply_rope(
                    q_rope, cos_sin_cache, positions, self.is_neox_style
                )
                k_pe = self._apply_rope(
                    k_pe, cos_sin_cache, positions, self.is_neox_style
                )

        # ── 4. Write to KV cache ──────────────────────────────────────────────
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if kv_cache.numel() > 0 and slot_mapping is not None:
            self._write_kv_cache_cpu(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
            )

        # ── 5. Attention computation ──────────────────────────────────────────
        num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", None) or 0
        has_decode = (getattr(attn_metadata, "num_decodes", None) or 0) > 0
        has_prefill = (getattr(attn_metadata, "num_prefills", None) or 0) > 0

        # Output buffer: [num_tokens, num_heads * v_head_dim]
        attn_output = torch.zeros(
            num_tokens,
            wrapper.num_heads * wrapper.v_head_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Prefill path (MHA) — pure torch
        if has_prefill:
            self._forward_prefill_torch(
                q=q[num_decode_tokens:],
                kv_c_normed=kv_c_normed[num_decode_tokens:],
                k_pe=k_pe[num_decode_tokens:],
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=attn_output[num_decode_tokens:],
            )

        # Decode path (MQA absorption) — pure torch
        if has_decode:
            decode_q = q[:num_decode_tokens]
            mqa_q_nope, mqa_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # W_UK_T: [N, qk_nope_head_dim, kv_lora_rank]
            assert self.W_UK_T is not None, "W_UK_T not initialized"
            mqa_ql_nope = torch.bmm(
                mqa_q_nope.transpose(0, 1),
                self.W_UK_T,
            ).transpose(0, 1)

            attn_out = self._forward_decode_torch(
                q_nope_proj=mqa_ql_nope,
                q_pe=mqa_q_pe,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )

            # v_up_proj: (B, N, L) -> (B, N, V) -> (B, N * V)
            assert self.W_UV is not None, "W_UV not initialized"
            decode_v = torch.bmm(
                attn_out.transpose(0, 1),
                self.W_UV,
            ).transpose(0, 1)
            attn_output[:num_decode_tokens] = decode_v.reshape(
                num_decode_tokens, wrapper.num_heads * wrapper.v_head_dim
            )

        # ── 6. o_proj ─────────────────────────────────────────────────────────
        suffix = attn_output[num_decode_tokens:]
        logger.warning(
            "forward_fused [before o_proj]: "
            "attn_output shape=%s, absmax=%.6f, all_zero=%s, "
            "attn_output[%d:] absmax=%.6f",
            attn_output.shape,
            attn_output.abs().max().item(),
            (attn_output == 0).all().item(),
            num_decode_tokens,
            suffix.abs().max().item() if suffix.numel() > 0 else 0.0,
        )
        o_w = wrapper.o_proj
        result = self._linear(o_w, attn_output)
        logger.warning(
            "forward_fused [after o_proj]: "
            "result shape=%s, absmax=%.6f, all_zero=%s",
            result.shape,
            result.abs().max().item(),
            (result == 0).all().item(),
        )
        return result

    # ── Prefill attention (pure torch) ────────────────────────────────────────

    def _forward_prefill_torch(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
    ) -> None:
        """Prefill stage MHA forward (pure torch).

        Up-projects kv_c_normed through kv_b_proj weights to full k/v,
        then uses torch SDPA for standard multi-head attention. When
        historical context exists, gathers context tokens from KV cache
        and merges with LSE merge.

        Args:
            q: shape = [num_prefill_tokens, num_heads, qk_head_dim]
            kv_c_normed: shape = [num_prefill_tokens, kv_lora_rank]
            k_pe: shape = [num_prefill_tokens, 1, qk_rope_head_dim]
            kv_cache: KV cache tensor
            attn_metadata: Attention metadata
            output: Output tensor, shape = [num_prefill_tokens, num_heads * v_head_dim]
        """
        prefill_metadata = getattr(attn_metadata, "prefill", None)
        logger.warning(
            "_forward_prefill_torch debug: "
            "prefill_metadata is None: %s, "
            "q.shape=%s, kv_c_normed.shape=%s, k_pe.shape=%s, "
            "output.shape=%s, "
            "_kv_b_proj_weight is None: %s",
            prefill_metadata is None,
            q.shape,
            kv_c_normed.shape,
            k_pe.shape,
            output.shape,
            self._kv_b_proj_weight is None,
        )
        if prefill_metadata is None:
            logger.warning(
                "_forward_prefill_torch: prefill_metadata is None! "
                "attn_metadata type=%s, "
                "attn_metadata attrs=%s",
                type(attn_metadata).__name__,
                [a for a in dir(attn_metadata) if not a.startswith("_")],
            )
            return

        has_context = getattr(prefill_metadata, "chunked_context", None) is not None
        logger.warning(
            "_forward_prefill_torch: has_context=%s, "
            "query_start_loc=%s, max_query_len=%s",
            has_context,
            prefill_metadata.query_start_loc,
            prefill_metadata.max_query_len,
        )

        # Up-project compressed kv_c through kv_b_proj weights to full k_nope and v
        kv_nope = self._kv_b_proj_forward(kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        logger.warning(
            "_forward_prefill_torch [step1] kv_b_proj done: "
            "kv_nope absmax=%.6f, k_nope absmax=%.6f, v absmax=%.6f",
            kv_nope.abs().max().item(),
            k_nope.abs().max().item(),
            v.abs().max().item(),
        )

        # Concatenate k_nope and k_pe to get full k
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        logger.warning(
            "_forward_prefill_torch [step2] concat done: "
            "q absmax=%.6f, k absmax=%.6f, v absmax=%.6f, "
            "q.shape=%s, k.shape=%s, v.shape=%s",
            q.abs().max().item(),
            k.abs().max().item(),
            v.abs().max().item(),
            q.shape, k.shape, v.shape,
        )

        # New token causal attention
        cu_seqlens_q = prefill_metadata.query_start_loc
        output_prefill = self._varlen_attention(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=prefill_metadata.max_query_len,
            causal=True,
            return_softmax_lse=has_context,
        )

        logger.warning(
            "_forward_prefill_torch [step3] varlen_attn done: "
            "type=%s, is_tensor=%s",
            type(output_prefill).__name__,
            isinstance(output_prefill, torch.Tensor),
        )
        if isinstance(output_prefill, torch.Tensor):
            logger.warning(
                "_forward_prefill_torch [step3] output_prefill: "
                "shape=%s, absmax=%.6f, all_zero=%s",
                output_prefill.shape,
                output_prefill.abs().max().item(),
                (output_prefill == 0).all().item(),
            )

        if has_context:
            assert isinstance(output_prefill, tuple)
            suffix_output, suffix_lse = output_prefill
            assert suffix_lse is not None
            suffix_output = suffix_output[..., :self.v_head_dim]

            context_output, context_lse = self._compute_prefill_context(
                q, kv_cache, attn_metadata
            )

            merged = self._merge_attn_states(
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
            output.copy_(merged.flatten(start_dim=-2))
        else:
            assert isinstance(output_prefill, torch.Tensor)
            output_prefill = output_prefill[..., :v.shape[-1]].flatten(start_dim=-2)
            logger.warning(
                "_forward_prefill_torch [step4] before copy: "
                "output_prefill shape=%s, absmax=%.6f, all_zero=%s, "
                "output shape=%s, output data_ptr=%s",
                output_prefill.shape,
                output_prefill.abs().max().item(),
                (output_prefill == 0).all().item(),
                output.shape,
                output.data_ptr(),
            )
            output.copy_(output_prefill)
            logger.warning(
                "_forward_prefill_torch [step5] after copy: "
                "output absmax=%.6f, all_zero=%s",
                output.abs().max().item(),
                (output == 0).all().item(),
            )

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather context tokens from KV cache, compute prefill context attention.

        Args:
            q: shape = [num_prefill_tokens, num_heads, qk_head_dim]
            kv_cache: shape = [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
            attn_metadata: Attention metadata

        Returns:
            (context_output, context_lse)
        """
        prefill_metadata = attn_metadata.prefill
        block_size = kv_cache.shape[1]
        block_table = prefill_metadata.block_table
        num_prefills = block_table.shape[0]

        cu_seq_lens_first = prefill_metadata.chunked_context.cu_seq_lens[0]
        context_lens = (
            cu_seq_lens_first[1:] - cu_seq_lens_first[:-1]
        ).cpu()

        cu_seqlens_k = torch.zeros(num_prefills + 1, dtype=torch.int32, device=q.device)
        for i in range(num_prefills):
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(context_lens[i])

        total_context_tokens = int(cu_seqlens_k[-1].item())
        if total_context_tokens == 0:
            num_tokens = q.shape[0]
            num_heads = q.shape[1]
            context_output = torch.zeros(
                num_tokens, num_heads, self.v_head_dim,
                dtype=q.dtype, device=q.device,
            )
            context_lse = torch.full(
                (num_heads, num_tokens), float("-inf"),
                dtype=torch.float32, device=q.device,
            )
            return context_output, context_lse

        # Gather all context token kv_c and k_pe
        gathered_kv = torch.empty(
            total_context_tokens,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=kv_cache.dtype,
            device=q.device,
        )
        for i in range(num_prefills):
            ctx_len = int(context_lens[i])
            if ctx_len == 0:
                continue
            dst_start = int(cu_seqlens_k[i].item())
            gathered_kv[dst_start:dst_start + ctx_len] = self._gather_kv_cache(
                kv_cache, block_table[i], ctx_len, block_size
            )

        kv_c_ctx = gathered_kv[:, :self.kv_lora_rank]
        k_pe_ctx = gathered_kv[:, self.kv_lora_rank:].unsqueeze(1)

        # Up-project through kv_b_proj weights
        kv_nope = self._kv_b_proj_forward(kv_c_ctx).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope_ctx, v_ctx = kv_nope.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_ctx = self._concat_k_nope_k_pe(k_nope_ctx, k_pe_ctx)

        cu_seqlens_q = prefill_metadata.query_start_loc

        result = self._varlen_attention(
            q=q,
            k=k_ctx,
            v=v_ctx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=prefill_metadata.max_query_len,
            max_seqlen_k=int(context_lens.max().item()),
            causal=False,
            return_softmax_lse=True,
        )
        assert isinstance(result, tuple)
        context_output, context_lse = result
        assert context_lse is not None
        return context_output, context_lse

    # ── Decode attention (pure torch) ─────────────────────────────────────────

    def _forward_decode_torch(
        self,
        q_nope_proj: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
    ) -> torch.Tensor:
        """Decode stage MQA forward (pure torch, absorption trick).

        Args:
            q_nope_proj: [B, N, kv_lora_rank] (already projected through W_UK_T)
            q_pe: [B, N, qk_rope_head_dim]
            kv_cache: shape = [num_blocks, block_size, head_size]
            attn_metadata: Attention metadata

        Returns:
            output: shape = [B, N, kv_lora_rank]
        """
        decode_metadata = getattr(attn_metadata, "decode", None)
        if decode_metadata is None:
            return torch.zeros_like(q_nope_proj)

        batch_size = q_nope_proj.shape[0]
        num_heads = q_nope_proj.shape[1]

        block_table = decode_metadata.block_table
        seq_lens = decode_metadata.seq_lens
        seq_lens_cpu = seq_lens.cpu()

        output = torch.zeros(
            batch_size,
            num_heads,
            self.kv_lora_rank,
            dtype=q_nope_proj.dtype,
            device=q_nope_proj.device,
        )

        block_size = kv_cache.shape[1]

        for b in range(batch_size):
            seq_len = int(seq_lens_cpu[b].item())
            if seq_len == 0:
                continue

            gathered_kv = self._gather_kv_cache(
                kv_cache, block_table[b], seq_len, block_size
            )

            kv_c = gathered_kv[:, :self.kv_lora_rank]
            k_pe_seq = gathered_kv[:, self.kv_lora_rank:]

            # Attention scores = q_l @ kv_c.T + q_pe @ k_pe.T
            attn_scores = (
                torch.matmul(q_nope_proj[b], kv_c.T)
                + torch.matmul(q_pe[b], k_pe_seq.T)
            ) * self.scale

            attn_weights = F.softmax(attn_scores, dim=-1)
            output[b] = torch.matmul(attn_weights, kv_c)

        return output

    # ── Helper methods ────────────────────────────────────────────────────────

    def _kv_b_proj_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear transform using kv_b_proj weights (pure F.linear).

        :param x: Input tensor, shape = [..., kv_lora_rank]
        :return: Output tensor, shape = [..., num_heads * (qk_nope_head_dim + v_head_dim)]
        """
        assert self._kv_b_proj_weight is not None, "kv_b_proj weight not initialized"
        return F.linear(x, self._kv_b_proj_weight, self._kv_b_proj_bias)

    @staticmethod
    def _linear(layer: Any, x: torch.Tensor) -> torch.Tensor:
        """Pure torch linear transform (F.linear).

        :param layer: Linear layer object (must have weight attribute)
        :param x: Input tensor
        :return: Linear transform output
        """
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
    def _rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Pure torch RMSNorm: x -> weight * x / sqrt(mean(x^2) + eps).

        Normalizes in float32, casts back to orig_dtype, then multiplies by weight.
        """
        orig_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_f32 = x_f32 * torch.rsqrt(variance + eps)
        return x_f32.to(orig_dtype) * weight

    @staticmethod
    def _apply_rope(
        x: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        is_neox_style: bool | None = True,
    ) -> torch.Tensor:
        """Pure torch RoPE, supports NeoX and GPT-J styles.

        cos_sin_cache: [max_pos, rot_dim], first half cos, second half sin
        x: [num_tokens, num_heads, rot_dim]
        positions: [num_tokens]
        is_neox_style: True for NeoX (first/second half split),
                       False for GPT-J (even/odd interleaved split).

        Returns tensor with same shape as x after rotation.
        """
        if is_neox_style is None:
            is_neox_style = True

        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # cos/sin: [num_tokens, rot_dim/2] -> [num_tokens, 1, rot_dim/2]
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        if is_neox_style:
            # NeoX style: first/second half split
            x1, x2 = torch.chunk(x, 2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.cat((o1, o2), dim=-1)
        else:
            # GPT-J style: even/odd interleaved split
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    @staticmethod
    def _write_kv_cache_cpu(
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Concatenate kv_c and k_pe, write to paged KV cache via slot_mapping.

        Args:
            kv_c: shape = [num_tokens, kv_lora_rank]
            k_pe: shape = [num_tokens, qk_rope_head_dim]
            kv_cache: shape = [num_blocks, block_size, head_size]
            slot_mapping: shape = [num_tokens]
        """
        kv_combined = torch.cat([kv_c, k_pe], dim=-1)
        block_size = kv_cache.shape[1]
        for i in range(slot_mapping.shape[0]):
            slot = int(slot_mapping[i].item())
            if slot < 0:
                continue
            kv_cache[slot // block_size, slot % block_size] = kv_combined[i]

    @staticmethod
    def _concat_k_nope_k_pe(
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate k_nope and k_pe to get full k.

        k_nope: [num_tokens, num_heads, qk_nope_head_dim]
        k_pe: [num_tokens, 1, qk_rope_head_dim]

        Returns:
            k: [num_tokens, num_heads, qk_head_dim]
        """
        # Broadcast k_pe to num_heads
        k_pe_expanded = k_pe.expand(-1, k_nope.shape[1], -1)
        return torch.cat([k_nope, k_pe_expanded], dim=-1)

    @staticmethod
    def _gather_kv_cache(
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather all tokens for a sequence from paged KV cache.

        Args:
            kv_cache: shape = [num_blocks, block_size, head_size]
            block_table: Block indices for this request, shape = [max_blocks]
            seq_len: Sequence length
            block_size: Tokens per block

        Returns:
            gathered: shape = [seq_len, head_size]
        """
        head_size = kv_cache.shape[2]
        gathered = torch.empty(
            seq_len,
            head_size,
            dtype=kv_cache.dtype,
            device=kv_cache.device,
        )

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

    def _varlen_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        """Pure torch variable-length multi-head attention.

        Args:
            q: shape = [total_tokens, num_heads, qk_head_dim]
            k: shape = [total_tokens, num_heads, qk_head_dim]
            v: shape = [total_tokens, num_heads, v_head_dim]
            cu_seqlens_q: Cumulative query sequence lengths, shape = [batch_size + 1]
            cu_seqlens_k: Cumulative key sequence lengths, shape = [batch_size + 1]
            max_seqlen_q: Maximum query sequence length
            max_seqlen_k: Maximum key sequence length
            causal: Whether to use causal mask
            return_softmax_lse: Whether to return log-sum-exp

        Returns:
            Attention output, shape = [total_tokens, num_heads, v_head_dim]
            If return_softmax_lse=True, also returns lse
        """
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q_tokens = q.shape[0]
        num_heads = q.shape[1]
        v_head_dim = v.shape[2]
        qk_head_dim = q.shape[2]

        output = torch.zeros(
            total_q_tokens, num_heads, v_head_dim,
            dtype=q.dtype, device=q.device,
        )

        lse = torch.full(
            (num_heads, total_q_tokens), float("-inf"),
            dtype=torch.float32, device=q.device,
        ) if return_softmax_lse else None

        cu_seqlens_q_cpu = cu_seqlens_q.cpu().numpy()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().numpy()

        for i in range(batch_size):
            q_start = cu_seqlens_q_cpu[i]
            q_end = cu_seqlens_q_cpu[i + 1]
            k_start = cu_seqlens_k_cpu[i]
            k_end = cu_seqlens_k_cpu[i + 1]

            if q_end <= q_start or k_end <= k_start:
                continue

            # q_i: [1, num_heads, seq_q, qk_head_dim]
            q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
            k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
            v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

            # If qk_head_dim != v_head_dim, pad v
            if qk_head_dim != v_head_dim:
                v_i = F.pad(v_i, [0, qk_head_dim - v_head_dim], value=0.0)

            if return_softmax_lse:
                # Manual attention scores for LSE
                attn_scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * self.scale

                if causal:
                    seq_q = q_end - q_start
                    seq_k = k_end - k_start
                    q_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)
                    k_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)
                    causal_mask = q_idx >= k_idx
                    attn_scores = attn_scores.masked_fill(
                        ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

                lse_i = torch.logsumexp(attn_scores, dim=-1)
                assert lse is not None
                lse[:, q_start:q_end] = lse_i.squeeze(0)

                attn_weights = torch.softmax(attn_scores, dim=-1)
                output_i = torch.matmul(attn_weights, v_i)
            else:
                # Use torch SDPA (more efficient)
                output_i = F.scaled_dot_product_attention(
                    q_i, k_i, v_i,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=causal,
                    scale=self.scale,
                )

            output[q_start:q_end] = output_i[0, :, :, :v_head_dim].transpose(0, 1)

        if return_softmax_lse:
            return output, lse

        return output

    @staticmethod
    def _merge_attn_states(
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
    ) -> torch.Tensor:
        """Pure torch LSE merge, combining two attention outputs.

        Args:
            prefix_output: shape = [num_tokens, num_heads, head_size]
            prefix_lse: shape = [num_heads, num_tokens]
            suffix_output: shape = [num_tokens, num_heads, head_size]
            suffix_lse: shape = [num_heads, num_tokens]

        Returns:
            merged output, shape = [num_tokens, num_heads, head_size]
        """
        p_lse = prefix_lse.transpose(0, 1).unsqueeze(-1).float()
        s_lse = suffix_lse.transpose(0, 1).unsqueeze(-1).float()

        max_lse = torch.maximum(p_lse, s_lse)
        p_se = torch.exp(p_lse - max_lse)
        s_se = torch.exp(s_lse - max_lse)
        out_se = p_se + s_se

        p_scale = p_se / out_se
        s_scale = s_se / out_se

        merged = (
            p_scale * prefix_output.float()
            + s_scale * suffix_output.float()
        ).to(prefix_output.dtype)
        return merged

    @staticmethod
    def debug_linear_layer(name: str, layer: Any) -> None:
        """Debug linear layer information."""
        weight = getattr(layer, "weight", None)
        bias = getattr(layer, "bias", None)
        logger.debug(
            "%s: type=%s, has_cpu_linear=%s, weight_shape=%s, "
            "weight_numel=%s, has_bias=%s, skip_bias_add=%s",
            name,
            type(layer).__name__,
            hasattr(layer, "cpu_linear"),
            None if weight is None else tuple(weight.shape),
            None if weight is None else weight.numel(),
            bias is not None,
            getattr(layer, "skip_bias_add", None),
        )
