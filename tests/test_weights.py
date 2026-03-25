# -*- coding: utf-8 -*-
"""Tests for process_weights_after_loading verifying W_UK_T and W_UV shapes."""
import torch
import pytest
from fused_mla_cpp import CPUFusedMLAImpl


NUM_HEADS = 4
QK_NOPE_HEAD_DIM = 32
QK_ROPE_HEAD_DIM = 16
V_HEAD_DIM = 24
KV_LORA_RANK = 64
OUT_FEATURES = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)  # 4 * 56 = 224


def _make_impl():
    """Create a CPUFusedMLAImpl with a real kv_b_proj weight tensor."""
    weight = torch.randn(OUT_FEATURES, KV_LORA_RANK)
    kv_b_proj = type("FakeLinear", (), {"weight": weight, "bias": None})()

    impl = CPUFusedMLAImpl(
        num_heads=NUM_HEADS,
        head_size=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        scale=0.125,
        num_kv_heads=1,
        kv_cache_dtype="auto",
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=kv_b_proj,
    )
    return impl


class TestProcessWeightsAfterLoading:
    """Verify W_UK_T and W_UV shapes after process_weights_after_loading."""

    def test_w_uk_t_shape(self):
        impl = _make_impl()
        impl.process_weights_after_loading(act_dtype=torch.float32)

        assert impl.W_UK_T is not None
        assert impl.W_UK_T.shape == (NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK)

    def test_w_uv_shape(self):
        impl = _make_impl()
        impl.process_weights_after_loading(act_dtype=torch.float32)

        assert impl.W_UV is not None
        assert impl.W_UV.shape == (NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM)
