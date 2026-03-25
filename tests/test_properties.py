# -*- coding: utf-8 -*-
"""Property-based tests for fused_mla_cpp using hypothesis.

Each test validates a correctness property from the design document.
"""
from __future__ import annotations

import glob
import math
import os

import torch
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from fused_mla_cpp.core import CPUFusedMLAImpl


# ── Hypothesis strategies ────────────────────────────────────────────────────

@st.composite
def rms_norm_inputs(draw):
    """Generate random inputs for _rms_norm testing.

    Returns (x, weight, eps) where weight is all-ones so we can check
    that the normalized output has unit RMS.
    """
    # Use larger dimensions so eps has negligible effect on RMS
    batch = draw(st.integers(min_value=1, max_value=4))
    dim = draw(st.integers(min_value=8, max_value=64))
    x = torch.randn(batch, dim)
    # Ensure x is non-zero
    assume(x.abs().max().item() > 1e-6)
    weight = torch.ones(dim)
    eps = draw(st.floats(min_value=1e-8, max_value=1e-6))
    return x, weight, eps


@st.composite
def rope_inputs(draw):
    """Generate random inputs for _apply_rope testing.

    Returns (x, cos_sin_cache, positions, is_neox_style).
    cos_sin_cache contains actual cos/sin values so rotation is norm-preserving.
    """
    num_tokens = draw(st.integers(min_value=1, max_value=4))
    num_heads = draw(st.integers(min_value=1, max_value=4))
    # rot_dim must be even
    half_rot_dim = draw(st.integers(min_value=1, max_value=8))
    rot_dim = half_rot_dim * 2

    x = torch.randn(num_tokens, num_heads, rot_dim)
    # Ensure x is non-zero per token-head
    assume(x.abs().max().item() > 1e-6)

    max_pos = draw(st.integers(min_value=num_tokens, max_value=num_tokens + 10))
    # cos_sin_cache: [max_pos, rot_dim] — first half cos, second half sin
    # Must contain actual cos/sin values for norm preservation
    angles = torch.randn(max_pos, half_rot_dim)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    cos_sin_cache = torch.cat([cos_vals, sin_vals], dim=-1)

    positions = torch.randint(0, max_pos, (num_tokens,))
    is_neox_style = draw(st.booleans())
    return x, cos_sin_cache, positions, is_neox_style


@st.composite
def kv_cache_round_trip_inputs(draw):
    """Generate random inputs for KV cache write-then-gather round trip.

    Returns (kv_c, k_pe, block_size).
    """
    num_tokens = draw(st.integers(min_value=1, max_value=8))
    kv_lora_rank = draw(st.integers(min_value=2, max_value=16))
    qk_rope_head_dim = draw(st.integers(min_value=2, max_value=8))
    block_size = draw(st.integers(min_value=1, max_value=4))

    kv_c = torch.randn(num_tokens, kv_lora_rank)
    k_pe = torch.randn(num_tokens, qk_rope_head_dim)
    return kv_c, k_pe, block_size


@st.composite
def lse_merge_inputs(draw):
    """Generate random inputs for LSE merge identity test.

    Returns (output, lse) where output is [T, H, D] and lse is [H, T].
    """
    num_tokens = draw(st.integers(min_value=1, max_value=4))
    num_heads = draw(st.integers(min_value=1, max_value=4))
    head_dim = draw(st.integers(min_value=2, max_value=8))

    output = torch.randn(num_tokens, num_heads, head_dim)
    # LSE values should be finite
    lse = torch.randn(num_heads, num_tokens)
    return output, lse


@st.composite
def causal_attention_inputs(draw):
    """Generate random inputs for causal attention masking test.

    Returns (q, k, v, seq_len) for a single sequence.
    """
    seq_len = draw(st.integers(min_value=2, max_value=6))
    num_heads = draw(st.integers(min_value=1, max_value=2))
    head_dim = draw(st.integers(min_value=2, max_value=8))

    q = torch.randn(seq_len, num_heads, head_dim)
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)
    return q, k, v, seq_len


# ── Property 1: No vllm imports in library source ────────────────────────────


class TestProperty1NoVllmImports:
    """**Validates: Requirements 1.4**

    Property 1: No vllm imports in library source.
    For any .py file in fused_mla_cpp/fused_mla_cpp/, the file shall contain
    zero import statements referencing vllm.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_no_vllm_imports(self, data):
        """Scan all .py files in fused_mla_cpp/fused_mla_cpp/ for absence of
        vllm imports."""
        package_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, "src", "fused_mla_cpp"
        )
        package_dir = os.path.normpath(package_dir)
        py_files = glob.glob(os.path.join(package_dir, "**", "*.py"), recursive=True)
        # Filter out __pycache__
        py_files = [f for f in py_files if "__pycache__" not in f]
        assume(len(py_files) > 0)

        # Pick a random file from the list
        idx = data.draw(st.integers(min_value=0, max_value=len(py_files) - 1))
        filepath = py_files[idx]

        with open(filepath, "r") as fh:
            content = fh.read()

        for line in content.splitlines():
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            assert "import vllm" not in stripped, (
                f"Found vllm import in {filepath}: {stripped}"
            )
            assert "from vllm" not in stripped, (
                f"Found vllm from-import in {filepath}: {stripped}"
            )


# ── Property 2: RMSNorm output has unit RMS ──────────────────────────────────


class TestProperty2RMSNormUnitRMS:
    """**Validates: Requirements 4.2, 4.3**

    Property 2: RMSNorm output has unit RMS.
    For random non-zero x, weight=ones, positive eps, the output of _rms_norm
    should have RMS ≈ 1.0.
    """

    @settings(max_examples=100)
    @given(inputs=rms_norm_inputs())
    def test_rms_norm_unit_rms(self, inputs):
        x, weight, eps = inputs
        result = CPUFusedMLAImpl._rms_norm(x, weight, eps)

        # With weight=ones, result IS the normalized output.
        # RMS of each row should be ≈ 1.0
        rms = result.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
            f"RMS not approximately 1.0: {rms}"
        )
        # Output dtype should match input dtype
        assert result.dtype == x.dtype


# ── Property 3: RoPE preserves vector norm ────────────────────────────────────


class TestProperty3RoPEPreservesNorm:
    """**Validates: Requirements 5.1, 5.2, 5.3**

    Property 3: RoPE preserves vector norm.
    For random x, valid cos_sin_cache, positions, and either RoPE style,
    _apply_rope output L2 norm per-token-per-head equals input L2 norm.
    """

    @settings(max_examples=100)
    @given(inputs=rope_inputs())
    def test_rope_preserves_norm(self, inputs):
        x, cos_sin_cache, positions, is_neox_style = inputs

        result = CPUFusedMLAImpl._apply_rope(
            x, cos_sin_cache, positions, is_neox_style
        )

        # L2 norm per token per head
        input_norms = torch.norm(x.float(), dim=-1)
        output_norms = torch.norm(result.float(), dim=-1)

        assert torch.allclose(input_norms, output_norms, atol=1e-4), (
            f"Norms differ: input={input_norms}, output={output_norms}"
        )


# ── Property 4: KV cache write-then-gather round trip ────────────────────────


class TestProperty4KVCacheRoundTrip:
    """**Validates: Requirements 6.1, 6.2, 11.1, 11.2**

    Property 4: KV cache write-then-gather round trip.
    For random kv_c and k_pe, write via _write_kv_cache_cpu then gather via
    _gather_kv_cache returns torch.cat([kv_c, k_pe], dim=-1) exactly.
    """

    @settings(max_examples=100)
    @given(inputs=kv_cache_round_trip_inputs())
    def test_kv_cache_round_trip(self, inputs):
        kv_c, k_pe, block_size = inputs
        num_tokens = kv_c.shape[0]
        kv_lora_rank = kv_c.shape[1]
        qk_rope_head_dim = k_pe.shape[1]
        head_size = kv_lora_rank + qk_rope_head_dim

        # Create slot_mapping: sequential slots 0..num_tokens-1
        slot_mapping = torch.arange(num_tokens, dtype=torch.long)

        # Allocate KV cache with enough blocks
        num_blocks = (num_tokens + block_size - 1) // block_size
        kv_cache = torch.zeros(num_blocks, block_size, head_size)

        # Write
        CPUFusedMLAImpl._write_kv_cache_cpu(kv_c, k_pe, kv_cache, slot_mapping)

        # Build block_table for gathering: sequential block indices
        block_table = torch.arange(num_blocks, dtype=torch.long)

        # Gather
        gathered = CPUFusedMLAImpl._gather_kv_cache(
            kv_cache, block_table, num_tokens, block_size
        )

        expected = torch.cat([kv_c, k_pe], dim=-1)
        assert torch.equal(gathered, expected), (
            f"Round trip mismatch:\ngathered={gathered}\nexpected={expected}"
        )


# ── Property 5: LSE merge with zero-contribution is identity ──────────────────


class TestProperty5LSEMergeIdentity:
    """**Validates: Requirements 10.1, 10.3**

    Property 5: LSE merge with zero-contribution is identity.
    For random attention output x and LSE, merging x with zeros output and
    -inf LSE returns the original output.
    """

    @settings(max_examples=100)
    @given(inputs=lse_merge_inputs())
    def test_lse_merge_identity(self, inputs):
        output, lse = inputs
        num_tokens, num_heads, head_dim = output.shape

        # Zero-contribution chunk: zeros output, -inf LSE
        zero_output = torch.zeros_like(output)
        neg_inf_lse = torch.full_like(lse, float("-inf"))

        merged = CPUFusedMLAImpl._merge_attn_states(
            prefix_output=output,
            prefix_lse=lse,
            suffix_output=zero_output,
            suffix_lse=neg_inf_lse,
        )

        assert torch.allclose(merged.float(), output.float(), atol=1e-5), (
            f"Merge with zero-contribution should be identity.\n"
            f"merged={merged}\noriginal={output}"
        )


# ── Property 7: Causal attention masks future tokens ─────────────────────────


class TestProperty7CausalAttentionMasking:
    """**Validates: Requirements 7.3**

    Property 7: Causal attention masks future tokens.
    For random Q/K/V sequences, causal _varlen_attention output at position i
    is unchanged when future K/V entries are zeroed.
    """

    @settings(max_examples=100)
    @given(inputs=causal_attention_inputs())
    def test_causal_attention_masking(self, inputs):
        q, k, v, seq_len = inputs
        num_heads = q.shape[1]
        head_dim = q.shape[2]

        # Create a CPUFusedMLAImpl instance for calling _varlen_attention
        impl = CPUFusedMLAImpl(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / math.sqrt(head_dim),
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=4,
            qk_nope_head_dim=head_dim,
            qk_rope_head_dim=0,
            qk_head_dim=head_dim,
            v_head_dim=head_dim,
            kv_b_proj=None,
        )

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        # Full causal attention
        out_full = impl._varlen_attention(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            causal=True,
            return_softmax_lse=False,
        )

        # For each position i, zero out future K/V (positions > i)
        # and verify output at position i is unchanged
        for i in range(seq_len):
            k_masked = k.clone()
            v_masked = v.clone()
            # Zero out positions > i
            if i + 1 < seq_len:
                k_masked[i + 1:] = 0.0
                v_masked[i + 1:] = 0.0

            out_masked = impl._varlen_attention(
                q=q, k=k_masked, v=v_masked,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                causal=True,
                return_softmax_lse=False,
            )

            assert torch.allclose(
                out_full[i].float(), out_masked[i].float(), atol=1e-5
            ), (
                f"Causal masking violated at position {i}.\n"
                f"full={out_full[i]}\nmasked={out_masked[i]}"
            )
