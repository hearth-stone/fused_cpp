# -*- coding: utf-8 -*-
"""Tests for package importability, constructor signature, and set_attn_impl."""
import pytest
import torch


class TestPackageImport:
    """Verify the package is importable."""

    def test_import_fused_cpp(self):
        import fused_cpp  # noqa: F401

    def test_import_cpu_fused_mla_impl(self):
        from fused_cpp import CPUFusedMLAImpl  # noqa: F401


class TestConstructor:
    """Verify CPUFusedMLAImpl constructor accepts all required params."""

    def test_constructor_stores_params(self):
        from fused_cpp import CPUFusedMLAImpl

        # Minimal mock for kv_b_proj
        kv_b_proj = type("FakeLinear", (), {"weight": torch.zeros(1), "bias": None})()

        impl = CPUFusedMLAImpl(
            num_heads=8,
            head_size=128,
            scale=0.125,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=512,
            kv_lora_rank=256,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            qk_head_dim=96,
            v_head_dim=64,
            kv_b_proj=kv_b_proj,
        )

        assert impl.num_heads == 8
        assert impl.head_size == 128
        assert impl.scale == 0.125
        assert impl.num_kv_heads == 1
        assert impl.kv_cache_dtype == "auto"
        assert impl.q_lora_rank == 512
        assert impl.kv_lora_rank == 256
        assert impl.qk_nope_head_dim == 64
        assert impl.qk_rope_head_dim == 32
        assert impl.qk_head_dim == 96
        assert impl.v_head_dim == 64

    def test_constructor_accepts_kwargs(self):
        from fused_cpp import CPUFusedMLAImpl

        kv_b_proj = type("FakeLinear", (), {"weight": torch.zeros(1), "bias": None})()

        # Should not raise even with extra kwargs
        impl = CPUFusedMLAImpl(
            num_heads=4,
            head_size=64,
            scale=0.25,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            qk_head_dim=48,
            v_head_dim=32,
            kv_b_proj=kv_b_proj,
            extra_param="should_be_ignored",
        )
        assert impl.q_lora_rank is None


class TestSetAttnImpl:
    """Verify set_attn_impl is a no-op."""

    def test_set_attn_impl_returns_none(self):
        from fused_cpp import CPUFusedMLAImpl

        kv_b_proj = type("FakeLinear", (), {"weight": torch.zeros(1), "bias": None})()

        impl = CPUFusedMLAImpl(
            num_heads=4,
            head_size=64,
            scale=0.25,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            qk_head_dim=48,
            v_head_dim=32,
            kv_b_proj=kv_b_proj,
        )

        result = impl.set_attn_impl("some_impl")
        assert result is None

    def test_set_attn_impl_no_side_effects(self):
        from fused_cpp import CPUFusedMLAImpl

        kv_b_proj = type("FakeLinear", (), {"weight": torch.zeros(1), "bias": None})()

        impl = CPUFusedMLAImpl(
            num_heads=4,
            head_size=64,
            scale=0.25,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            qk_head_dim=48,
            v_head_dim=32,
            kv_b_proj=kv_b_proj,
        )

        # Capture state before
        attrs_before = {k: v for k, v in impl.__dict__.items()}

        impl.set_attn_impl("anything")

        # State should be unchanged
        attrs_after = {k: v for k, v in impl.__dict__.items()}
        assert attrs_before.keys() == attrs_after.keys()
        for key in attrs_before:
            before_val = attrs_before[key]
            after_val = attrs_after[key]
            if isinstance(before_val, torch.Tensor):
                assert torch.equal(before_val, after_val), f"Tensor {key} changed"
            else:
                assert before_val == after_val, f"Attribute {key} changed"
