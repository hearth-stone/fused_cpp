# -*- coding: utf-8 -*-
"""Tests for FUSED_MLA_USE_ORIG_RMSNORM and FUSED_MLA_USE_ORIG_ROPE env vars.

These tests verify that when the debug env vars are set to "1", the wrapper's
layernorm / rotary_emb are called instead of the internal implementations.

We use importlib.reload to re-evaluate the module-level env var checks.
"""
import importlib
import os
import torch
import pytest
from unittest.mock import MagicMock, patch, call


NUM_HEADS = 2
KV_LORA_RANK = 16
QK_NOPE_HEAD_DIM = 8
QK_ROPE_HEAD_DIM = 8
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 8
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
HIDDEN_SIZE = NUM_HEADS * QK_HEAD_DIM  # for q_proj output


def _make_wrapper(q_lora_rank=512):
    """Build a mock wrapper with all required attributes."""
    w = MagicMock()
    w.q_lora_rank = q_lora_rank
    w.kv_lora_rank = KV_LORA_RANK
    w.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    w.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    w.qk_head_dim = QK_HEAD_DIM
    w.num_heads = NUM_HEADS
    w.v_head_dim = V_HEAD_DIM
    return w


class TestOrigRmsNormEnvVar:
    """When FUSED_MLA_USE_ORIG_RMSNORM=1, wrapper's layernorm is called."""

    def test_orig_rmsnorm_calls_wrapper_layernorm(self, monkeypatch):
        # Set env var BEFORE reloading the module
        monkeypatch.setenv("FUSED_MLA_USE_ORIG_RMSNORM", "1")

        import fused_cpp.core as core_module
        importlib.reload(core_module)

        try:
            assert core_module._DEBUG_USE_ORIG_RMSNORM is True

            # Build an impl from the reloaded module
            kv_b_proj_weight = torch.randn(
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK
            )
            kv_b_proj = type("FakeLinear", (), {
                "weight": kv_b_proj_weight, "bias": None
            })()

            impl = core_module.CPUFusedMLAImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZE,
                scale=0.125,
                num_kv_heads=1,
                kv_cache_dtype="auto",
                q_lora_rank=512,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_head_dim=QK_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                kv_b_proj=kv_b_proj,
            )
            impl.process_weights_after_loading(act_dtype=torch.float32)

            # Build wrapper with mock layernorms
            wrapper = _make_wrapper(q_lora_rank=512)

            # q_a_layernorm: should be called directly when debug flag is on
            q_a_ln = MagicMock()
            q_a_ln.weight = torch.ones(512)
            q_a_ln.variance_epsilon = 1e-5
            q_a_ln.return_value = torch.randn(1, 512)
            wrapper.q_a_layernorm = q_a_ln

            # kv_a_layernorm: should be called directly
            kv_a_ln = MagicMock()
            kv_a_ln.weight = torch.ones(KV_LORA_RANK)
            kv_a_ln.variance_epsilon = 1e-5
            kv_a_ln.return_value = torch.randn(1, KV_LORA_RANK)
            wrapper.kv_a_layernorm = kv_a_ln

            # fused_qkv_a_proj: returns [q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            fused_out_dim = 512 + KV_LORA_RANK + QK_ROPE_HEAD_DIM
            fused_w = MagicMock()
            fused_w.weight = torch.randn(fused_out_dim, 64)
            fused_w.bias = None
            fused_w.skip_bias_add = False
            wrapper.fused_qkv_a_proj = fused_w

            # q_b_proj
            q_b_out_dim = NUM_HEADS * QK_HEAD_DIM
            q_b_w = MagicMock()
            q_b_w.weight = torch.randn(q_b_out_dim, 512)
            q_b_w.bias = None
            q_b_w.skip_bias_add = False
            wrapper.q_b_proj = q_b_w

            # rotary_emb
            wrapper.rotary_emb = None

            # o_proj
            o_w = MagicMock()
            o_w.weight = torch.randn(64, NUM_HEADS * V_HEAD_DIM)
            o_w.bias = None
            o_w.skip_bias_add = False
            wrapper.o_proj = o_w

            # Minimal attn_metadata
            attn_metadata = MagicMock()
            attn_metadata.slot_mapping = None
            attn_metadata.num_decode_tokens = 0
            attn_metadata.num_decodes = 0
            attn_metadata.num_prefills = 0

            hidden_states = torch.randn(1, 64)
            positions = torch.tensor([0])
            kv_cache = torch.empty(0)

            impl.forward_fused(hidden_states, positions, wrapper, kv_cache, attn_metadata)

            # Both layernorms should have been called directly (as callables)
            q_a_ln.assert_called_once()
            kv_a_ln.assert_called_once()
        finally:
            # Restore module to default state
            monkeypatch.delenv("FUSED_MLA_USE_ORIG_RMSNORM", raising=False)
            importlib.reload(core_module)


class TestOrigRopeEnvVar:
    """When FUSED_MLA_USE_ORIG_ROPE=1, wrapper's rotary_emb is called."""

    def test_orig_rope_calls_wrapper_rotary_emb(self, monkeypatch):
        monkeypatch.setenv("FUSED_MLA_USE_ORIG_ROPE", "1")

        import fused_cpp.core as core_module
        importlib.reload(core_module)

        try:
            assert core_module._DEBUG_USE_ORIG_ROPE is True

            kv_b_proj_weight = torch.randn(
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK
            )
            kv_b_proj = type("FakeLinear", (), {
                "weight": kv_b_proj_weight, "bias": None
            })()

            impl = core_module.CPUFusedMLAImpl(
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZE,
                scale=0.125,
                num_kv_heads=1,
                kv_cache_dtype="auto",
                q_lora_rank=512,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_head_dim=QK_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                kv_b_proj=kv_b_proj,
            )
            impl.process_weights_after_loading(act_dtype=torch.float32)

            wrapper = _make_wrapper(q_lora_rank=512)

            # Layernorms (use real RMSNorm path, not mocked)
            q_a_ln = MagicMock()
            q_a_ln.weight = torch.ones(512)
            q_a_ln.variance_epsilon = 1e-5
            wrapper.q_a_layernorm = q_a_ln

            kv_a_ln = MagicMock()
            kv_a_ln.weight = torch.ones(KV_LORA_RANK)
            kv_a_ln.variance_epsilon = 1e-5
            wrapper.kv_a_layernorm = kv_a_ln

            # fused_qkv_a_proj
            fused_out_dim = 512 + KV_LORA_RANK + QK_ROPE_HEAD_DIM
            fused_w = MagicMock()
            fused_w.weight = torch.randn(fused_out_dim, 64)
            fused_w.bias = None
            fused_w.skip_bias_add = False
            wrapper.fused_qkv_a_proj = fused_w

            # q_b_proj
            q_b_out_dim = NUM_HEADS * QK_HEAD_DIM
            q_b_w = MagicMock()
            q_b_w.weight = torch.randn(q_b_out_dim, 512)
            q_b_w.bias = None
            q_b_w.skip_bias_add = False
            wrapper.q_b_proj = q_b_w

            # rotary_emb: mock that returns modified tensors
            rotary_emb = MagicMock()
            cos_sin_cache = torch.randn(100, QK_ROPE_HEAD_DIM)
            rotary_emb.cos_sin_cache = cos_sin_cache
            rotary_emb.is_neox_style = True
            # rotary_emb(positions, q_rope, k_pe) -> (q_rope_out, k_pe_out)
            rotary_emb.return_value = (
                torch.randn(1, NUM_HEADS, QK_ROPE_HEAD_DIM),
                torch.randn(1, 1, QK_ROPE_HEAD_DIM),
            )
            wrapper.rotary_emb = rotary_emb

            # o_proj
            o_w = MagicMock()
            o_w.weight = torch.randn(64, NUM_HEADS * V_HEAD_DIM)
            o_w.bias = None
            o_w.skip_bias_add = False
            wrapper.o_proj = o_w

            attn_metadata = MagicMock()
            attn_metadata.slot_mapping = None
            attn_metadata.num_decode_tokens = 0
            attn_metadata.num_decodes = 0
            attn_metadata.num_prefills = 0

            hidden_states = torch.randn(1, 64)
            positions = torch.tensor([0])
            kv_cache = torch.empty(0)

            impl.forward_fused(hidden_states, positions, wrapper, kv_cache, attn_metadata)

            # rotary_emb should have been called as a callable
            rotary_emb.assert_called_once()
        finally:
            monkeypatch.delenv("FUSED_MLA_USE_ORIG_ROPE", raising=False)
            importlib.reload(core_module)
