# -*- coding: utf-8 -*-
"""Property-based tests for fused_cpp using hypothesis.

Each test validates a correctness property from the design document.
"""

from __future__ import annotations

import glob
import math
import os

import torch
import torch.nn.functional as F
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from fused_cpp.mla.impl import (
    CPUFusedMLAImpl,
    _pytorch_apply_rope as _impl_pytorch_apply_rope,
    _pytorch_gather_kv_cache as _impl_pytorch_gather_kv_cache,
    _pytorch_merge_attn_states as _impl_pytorch_merge_attn_states,
    _pytorch_rms_norm as _impl_pytorch_rms_norm,
    _pytorch_write_kv_cache as _impl_pytorch_write_kv_cache,
)


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
    For any .py file in fused_cpp/, the file shall contain
    zero import statements referencing vllm.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_no_vllm_imports(self, data):
        """Scan all .py files in fused_cpp/ for absence of
        vllm imports."""
        package_dir = os.path.join(os.path.dirname(__file__), os.pardir, "src", "fused_cpp")
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
            assert "import vllm" not in stripped, f"Found vllm import in {filepath}: {stripped}"
            assert "from vllm" not in stripped, f"Found vllm from-import in {filepath}: {stripped}"


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
        result = _impl_pytorch_rms_norm(x, weight, eps)

        # RMS of each row should be ≈ 1.0
        rms = result.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), f"RMS not approximately 1.0: {rms}"
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

        result = _impl_pytorch_apply_rope(x, cos_sin_cache, positions, is_neox_style)

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
        _impl_pytorch_write_kv_cache(kv_c, k_pe, kv_cache, slot_mapping)

        # Build block_table for gathering: sequential block indices
        block_table = torch.arange(num_blocks, dtype=torch.long)

        # Gather
        gathered = _impl_pytorch_gather_kv_cache(kv_cache, block_table, num_tokens, block_size)

        expected = torch.cat([kv_c, k_pe], dim=-1)
        assert torch.equal(gathered, expected), f"Round trip mismatch:\ngathered={gathered}\nexpected={expected}"


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

        merged = _impl_pytorch_merge_attn_states(
            prefix_output=output,
            prefix_lse=lse,
            suffix_output=zero_output,
            suffix_lse=neg_inf_lse,
        )

        assert torch.allclose(merged.float(), output.float(), atol=1e-5), (
            f"Merge with zero-contribution should be identity.\nmerged={merged}\noriginal={output}"
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
            q=q,
            k=k,
            v=v,
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
                k_masked[i + 1 :] = 0.0
                v_masked[i + 1 :] = 0.0

            out_masked = impl._varlen_attention(
                q=q,
                k=k_masked,
                v=v_masked,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                causal=True,
                return_softmax_lse=False,
            )

            assert torch.allclose(out_full[i].float(), out_masked[i].float(), atol=1e-5), (
                f"Causal masking violated at position {i}.\nfull={out_full[i]}\nmasked={out_masked[i]}"
            )


# ── Hypothesis strategy for absorption matrices ──────────────────────────────


@st.composite
def absorption_inputs(draw):
    """Generate random inputs for build_absorption_matrices testing.

    Returns (kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim,
             kv_lora_rank).
    """
    num_heads = draw(st.integers(min_value=1, max_value=4))
    qk_nope_head_dim = draw(st.integers(min_value=2, max_value=8))
    v_head_dim = draw(st.integers(min_value=2, max_value=8))
    kv_lora_rank = draw(st.integers(min_value=2, max_value=8))

    out_features = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_b_proj_weight = torch.randn(out_features, kv_lora_rank)
    return kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank


# ── C++ kernel property tests ────────────────────────────────────────────────

# Import C++ extension (skip tests if unavailable)
try:
    from fused_cpp import _C

    _HAS_CPP = True
except ImportError:
    _C = None
    _HAS_CPP = False

_cpp_required = pytest.mark.skipif(not _HAS_CPP, reason="C++ extension not available")


# ── Property 2 (C++): RMSNorm unit RMS and dtype preservation ────────────────


@_cpp_required
class TestProperty2RMSNormUnitRMSCpp:
    """# Feature: cpp-kernel-migration, Property 2: RMSNorm unit RMS and dtype preservation

    **Validates: Requirements 3.3, 3.4**

    Call _C.rms_norm(x, ones, eps) and verify per-row RMS ≈ 1.0 and output
    dtype matches input dtype.
    """

    @settings(max_examples=100)
    @given(inputs=rms_norm_inputs())
    def test_rms_norm_unit_rms_cpp(self, inputs):
        x, weight, eps = inputs
        result = _C.rms_norm(x, weight, eps)

        # With weight=ones, per-row RMS should be ≈ 1.0
        rms = result.float().pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), f"C++ RMS not approximately 1.0: {rms}"
        # Output dtype must match input dtype
        assert result.dtype == x.dtype, f"dtype mismatch: expected {x.dtype}, got {result.dtype}"


# ── Property 3 (C++): RoPE preserves vector norm ─────────────────────────────


@_cpp_required
class TestProperty3RoPEPreservesNormCpp:
    """# Feature: cpp-kernel-migration, Property 3: RoPE preserves vector norm

    **Validates: Requirements 4.4**

    Call _C.apply_rope(x, cos_sin_cache, positions, is_neox_style) and verify
    per-token-per-head L2 norm is preserved.
    """

    @settings(max_examples=100)
    @given(inputs=rope_inputs())
    def test_rope_preserves_norm_cpp(self, inputs):
        x, cos_sin_cache, positions, is_neox_style = inputs
        result = _C.apply_rope(x, cos_sin_cache, positions, is_neox_style)

        input_norms = torch.norm(x.float(), dim=-1)
        output_norms = torch.norm(result.float(), dim=-1)

        assert torch.allclose(input_norms, output_norms, atol=1e-4), (
            f"C++ RoPE norms differ: input={input_norms}, output={output_norms}"
        )


# ── Property 8 (C++): Absorption matrices shape and contiguity ───────────────


@_cpp_required
class TestProperty8AbsorptionMatricesShapeCpp:
    """# Feature: cpp-kernel-migration, Property 8: Absorption matrices shape and contiguity

    **Validates: Requirements 15.2**

    Call _C.build_absorption_matrices(...) and verify W_UK_T shape
    [N, qk_nope_head_dim, kv_lora_rank], W_UV shape
    [N, kv_lora_rank, v_head_dim], both contiguous.
    """

    @settings(max_examples=100)
    @given(inputs=absorption_inputs())
    def test_absorption_matrices_shape_cpp(self, inputs):
        kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank = inputs

        W_UK_T, W_UV = _C.build_absorption_matrices(
            kv_b_proj_weight,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            kv_lora_rank,
            torch.float32,
        )

        assert W_UK_T.shape == (num_heads, qk_nope_head_dim, kv_lora_rank), (
            f"W_UK_T shape mismatch: expected {(num_heads, qk_nope_head_dim, kv_lora_rank)}, got {tuple(W_UK_T.shape)}"
        )
        assert W_UV.shape == (num_heads, kv_lora_rank, v_head_dim), (
            f"W_UV shape mismatch: expected {(num_heads, kv_lora_rank, v_head_dim)}, got {tuple(W_UV.shape)}"
        )
        assert W_UK_T.is_contiguous(), "W_UK_T is not contiguous"
        assert W_UV.is_contiguous(), "W_UV is not contiguous"


# ── Property 4 (C++): KV cache write-gather round trip ───────────────────────


@_cpp_required
class TestProperty4KVCacheRoundTripCpp:
    """# Feature: cpp-kernel-migration, Property 4: KV cache write-gather round trip

    **Validates: Requirements 5.2, 5.5**

    Write via _C.write_kv_cache, gather via _C.gather_kv_cache, verify result
    equals torch.cat([kv_c, k_pe], dim=-1) exactly.
    """

    @settings(max_examples=100)
    @given(inputs=kv_cache_round_trip_inputs())
    def test_kv_cache_round_trip_cpp(self, inputs):
        # Feature: cpp-kernel-migration, Property 4: KV cache write-gather round trip
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

        # Write via C++ kernel
        _C.write_kv_cache(kv_c, k_pe, kv_cache, slot_mapping)

        # Build block_table for gathering: sequential block indices
        block_table = torch.arange(num_blocks, dtype=torch.long)

        # Gather via C++ kernel
        gathered = _C.gather_kv_cache(kv_cache, block_table, num_tokens, block_size)

        expected = torch.cat([kv_c, k_pe], dim=-1)
        assert torch.equal(gathered, expected), f"C++ round trip mismatch:\ngathered={gathered}\nexpected={expected}"


# ── Property 6 (C++): LSE merge with zero-contribution is identity ────────────


@_cpp_required
class TestProperty6LSEMergeIdentityCpp:
    """# Feature: cpp-kernel-migration, Property 6: LSE merge with zero-contribution is identity

    **Validates: Requirements 9.3**

    Merge output with (zeros, -inf LSE) via _C.merge_attn_states, verify
    result equals original output within atol=1e-5.
    """

    @settings(max_examples=100)
    @given(inputs=lse_merge_inputs())
    def test_lse_merge_identity_cpp(self, inputs):
        # Feature: cpp-kernel-migration, Property 6: LSE merge with zero-contribution is identity
        output, lse = inputs

        # Zero-contribution chunk: zeros output, -inf LSE
        zero_output = torch.zeros_like(output)
        neg_inf_lse = torch.full_like(lse, float("-inf"))

        merged = _C.merge_attn_states(
            prefix_output=output,
            prefix_lse=lse,
            suffix_output=zero_output,
            suffix_lse=neg_inf_lse,
        )

        assert torch.allclose(merged.float(), output.float(), atol=1e-5), (
            f"C++ merge with zero-contribution should be identity.\nmerged={merged}\noriginal={output}"
        )


# ── Property 5 (C++): Causal attention masks future tokens ────────────────────


@_cpp_required
class TestProperty5CausalAttentionMaskingCpp:
    """# Feature: cpp-kernel-migration, Property 5: Causal attention masks future tokens

    **Validates: Requirements 7.2**

    Call _C.varlen_attention with causal=True, verify zeroing future K/V does
    not change output at position i.
    """

    @settings(max_examples=100)
    @given(inputs=causal_attention_inputs())
    def test_causal_attention_masking_cpp(self, inputs):
        # Feature: cpp-kernel-migration, Property 5: Causal attention masks future tokens
        q, k, v, seq_len = inputs
        head_dim = q.shape[2]
        scale = 1.0 / math.sqrt(head_dim)

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        # Full causal attention via C++ kernel
        out_full = _C.varlen_attention(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            seq_len,
            seq_len,
            scale,
            True,
            False,
        )

        # For each position i, zero out future K/V (positions > i)
        # and verify output at position i is unchanged
        for i in range(seq_len):
            k_masked = k.clone()
            v_masked = v.clone()
            if i + 1 < seq_len:
                k_masked[i + 1 :] = 0.0
                v_masked[i + 1 :] = 0.0

            out_masked = _C.varlen_attention(
                q,
                k_masked,
                v_masked,
                cu_seqlens,
                cu_seqlens,
                seq_len,
                seq_len,
                scale,
                True,
                False,
            )

            assert torch.allclose(out_full[i].float(), out_masked[i].float(), atol=1e-5), (
                f"C++ causal masking violated at position {i}.\nfull={out_full[i]}\nmasked={out_masked[i]}"
            )


# ── Property 1 (C++): C++ source code constraints ────────────────────────────
# Feature: cpp-kernel-migration, Property 1: C++ source code constraints

# Standard C++ headers that are allowed in #include <...> directives
_ALLOWED_STD_HEADERS = frozenset(
    {
        "algorithm",
        "any",
        "array",
        "atomic",
        "bitset",
        "cassert",
        "ccomplex",
        "cctype",
        "cerrno",
        "cfenv",
        "cfloat",
        "charconv",
        "chrono",
        "cinttypes",
        "climits",
        "clocale",
        "cmath",
        "codecvt",
        "complex",
        "condition_variable",
        "csetjmp",
        "csignal",
        "cstdarg",
        "cstddef",
        "cstdint",
        "cstdio",
        "cstdlib",
        "cstring",
        "ctime",
        "cuchar",
        "cwchar",
        "cwctype",
        "deque",
        "exception",
        "execution",
        "filesystem",
        "format",
        "forward_list",
        "fstream",
        "functional",
        "future",
        "initializer_list",
        "iomanip",
        "ios",
        "iosfwd",
        "iostream",
        "istream",
        "iterator",
        "limits",
        "list",
        "locale",
        "map",
        "memory",
        "memory_resource",
        "mutex",
        "new",
        "numeric",
        "optional",
        "ostream",
        "queue",
        "random",
        "ranges",
        "ratio",
        "regex",
        "scoped_allocator",
        "set",
        "shared_mutex",
        "span",
        "sstream",
        "stack",
        "stdexcept",
        "streambuf",
        "string",
        "string_view",
        "system_error",
        "thread",
        "tuple",
        "type_traits",
        "typeindex",
        "typeinfo",
        "unordered_map",
        "unordered_set",
        "utility",
        "valarray",
        "variant",
        "vector",
        "version",
    }
)


def _is_allowed_include(include_path: str) -> bool:
    """Check if an #include directive references an allowed header.

    Allowed:
    - torch/ prefixed headers (e.g., <torch/extension.h>)
    - Standard C++ headers (e.g., <cmath>, <vector>)
    - Local includes with quotes (e.g., "my_header.h")
    """
    # Local includes with quotes are always allowed
    if include_path.startswith('"') and include_path.endswith('"'):
        return True
    # Angle-bracket includes
    if include_path.startswith("<") and include_path.endswith(">"):
        header = include_path[1:-1]
        # torch/ prefixed headers are allowed
        if header.startswith("torch/"):
            return True
        # Standard C++ headers (no path separator)
        if "/" not in header:
            # Strip .h suffix for C compat headers
            base = header.removesuffix(".h")
            if base in _ALLOWED_STD_HEADERS:
                return True
            # Also allow bare C headers like <stdint.h>, <math.h>, etc.
            # that may not be in the frozenset but have no path separator
            # and are not third-party (no path separator = standard)
            # Be conservative: only allow if no dots except .h
            if "." not in header or header.endswith(".h"):
                return True
        return False
    return False


@_cpp_required
class TestProperty1CppSourceConstraints:
    """# Feature: cpp-kernel-migration, Property 1: C++ source code constraints

    **Validates: Requirements 1.5, 1.6**

    For any .cpp file in fused_mla_cpp/csrc/, the file shall contain no
    #include directives referencing third-party libraries beyond torch/ and
    standard C++ headers, and shall not use torch::autograd or define any
    backward functions.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_cpp_source_constraints(self, data):
        """Scan all .cpp files in fused_mla_cpp/csrc/ for absence of
        third-party includes, torch::autograd usage, and backward function
        definitions."""
        import re

        csrc_dir = os.path.join(os.path.dirname(__file__), os.pardir, "csrc")
        csrc_dir = os.path.normpath(csrc_dir)
        cpp_files = glob.glob(os.path.join(csrc_dir, "**", "*.cpp"), recursive=True)
        assume(len(cpp_files) > 0)

        # Pick a random file from the list
        idx = data.draw(st.integers(min_value=0, max_value=len(cpp_files) - 1))
        filepath = cpp_files[idx]

        with open(filepath, "r") as fh:
            content = fh.read()

        filename = os.path.basename(filepath)

        for line_no, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()

            # Skip comments (single-line)
            if stripped.startswith("//"):
                continue

            # Check #include directives
            include_match = re.match(r'#include\s+([<"][^>"]+[>"])', stripped)
            if include_match:
                include_path = include_match.group(1)
                assert _is_allowed_include(include_path), (
                    f"Disallowed third-party include in {filename}:{line_no}: {stripped}"
                )

            # Check for torch::autograd usage
            assert "torch::autograd" not in stripped, f"Found torch::autograd usage in {filename}:{line_no}: {stripped}"

            # Check for backward function definitions
            backward_match = re.search(r"\b(backward)\s*\(", stripped)
            if backward_match:
                # Allow comments mentioning backward, only flag definitions
                # A definition would look like: type backward(...) or
                # something backward(...)
                # Exclude if it's inside a string or comment
                if not stripped.startswith("//") and not stripped.startswith("/*"):
                    assert False, f"Found backward function definition in {filename}:{line_no}: {stripped}"


# ── Hypothesis strategies for Property 7 ──────────────────────────────────────


@st.composite
def rms_norm_inputs_with_weight(draw):
    """Generate random inputs for rms_norm with non-ones weight.

    Returns (x, weight, eps).
    """
    batch = draw(st.integers(min_value=1, max_value=4))
    dim = draw(st.integers(min_value=8, max_value=64))
    x = torch.randn(batch, dim)
    assume(x.abs().max().item() > 1e-6)
    weight = torch.randn(dim)
    eps = draw(st.floats(min_value=1e-8, max_value=1e-6))
    return x, weight, eps


@st.composite
def linear_inputs(draw):
    """Generate random inputs for linear kernel testing.

    Returns (x, weight, bias_or_none).
    """
    batch = draw(st.integers(min_value=1, max_value=4))
    in_features = draw(st.integers(min_value=2, max_value=16))
    out_features = draw(st.integers(min_value=2, max_value=16))
    x = torch.randn(batch, in_features)
    weight = torch.randn(out_features, in_features)
    has_bias = draw(st.booleans())
    bias = torch.randn(out_features) if has_bias else None
    return x, weight, bias


@st.composite
def concat_inputs(draw):
    """Generate random inputs for concat_k_nope_k_pe testing.

    Returns (k_nope, k_pe).
    """
    num_tokens = draw(st.integers(min_value=1, max_value=4))
    num_heads = draw(st.integers(min_value=1, max_value=4))
    qk_nope_head_dim = draw(st.integers(min_value=2, max_value=8))
    qk_rope_head_dim = draw(st.integers(min_value=2, max_value=8))
    k_nope = torch.randn(num_tokens, num_heads, qk_nope_head_dim)
    k_pe = torch.randn(num_tokens, 1, qk_rope_head_dim)
    return k_nope, k_pe


@st.composite
def forward_decode_inputs(draw):
    """Generate random inputs for forward_decode testing.

    Returns (q_nope_proj, q_pe, kv_cache, block_table, seq_lens,
             scale, kv_lora_rank, qk_rope_head_dim).
    """
    batch_size = draw(st.integers(min_value=1, max_value=3))
    num_heads = draw(st.integers(min_value=1, max_value=3))
    kv_lora_rank = draw(st.integers(min_value=2, max_value=8))
    qk_rope_head_dim = draw(st.integers(min_value=2, max_value=8))
    head_size = kv_lora_rank + qk_rope_head_dim
    block_size = draw(st.integers(min_value=1, max_value=4))

    # Each sequence has some length
    seq_lens_list = [draw(st.integers(min_value=1, max_value=6)) for _ in range(batch_size)]
    max_seq_len = max(seq_lens_list)
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    # Total blocks needed
    total_blocks = sum((s + block_size - 1) // block_size for s in seq_lens_list)

    q_nope_proj = torch.randn(batch_size, num_heads, kv_lora_rank)
    q_pe = torch.randn(batch_size, num_heads, qk_rope_head_dim)
    kv_cache = torch.randn(total_blocks, block_size, head_size)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.long)

    # Build block_table: [batch_size, max_blocks_per_seq]
    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.long)
    block_offset = 0
    for b in range(batch_size):
        n_blocks = (seq_lens_list[b] + block_size - 1) // block_size
        for j in range(n_blocks):
            block_table[b, j] = block_offset + j
        block_offset += n_blocks

    scale = 1.0 / math.sqrt(kv_lora_rank + qk_rope_head_dim)
    return (q_nope_proj, q_pe, kv_cache, block_table, seq_lens, scale, kv_lora_rank, qk_rope_head_dim)


# ── Pure PyTorch reference implementations (no C++ dispatch) ──────────────────


def _pytorch_rms_norm(x, weight, eps):
    """Pure PyTorch RMSNorm (no C++ dispatch)."""
    orig_dtype = x.dtype
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_f32 = x_f32 * torch.rsqrt(variance + eps)
    return x_f32.to(orig_dtype) * weight


def _pytorch_apply_rope(x, cos_sin_cache, positions, is_neox_style):
    """Pure PyTorch RoPE (no C++ dispatch)."""
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


def _pytorch_write_kv_cache(kv_c, k_pe, kv_cache, slot_mapping):
    """Pure PyTorch write_kv_cache (no C++ dispatch)."""
    kv_combined = torch.cat([kv_c, k_pe], dim=-1)
    block_size = kv_cache.shape[1]
    for i in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        kv_cache[slot // block_size, slot % block_size] = kv_combined[i]


def _pytorch_gather_kv_cache(kv_cache, block_table, seq_len, block_size):
    """Pure PyTorch gather_kv_cache (no C++ dispatch)."""
    head_size = kv_cache.shape[2]
    gathered = torch.empty(seq_len, head_size, dtype=kv_cache.dtype, device=kv_cache.device)
    num_full_blocks = seq_len // block_size
    remainder = seq_len % block_size
    for block_idx in range(num_full_blocks):
        block_num = int(block_table[block_idx].item())
        start = block_idx * block_size
        gathered[start : start + block_size] = kv_cache[block_num]
    if remainder > 0:
        block_num = int(block_table[num_full_blocks].item())
        start = num_full_blocks * block_size
        gathered[start : start + remainder] = kv_cache[block_num, :remainder]
    return gathered


def _pytorch_varlen_attention(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, causal, return_softmax_lse
):
    """Pure PyTorch varlen_attention (no C++ dispatch)."""
    batch_size = cu_seqlens_q.shape[0] - 1
    total_q_tokens = q.shape[0]
    num_heads = q.shape[1]
    v_head_dim = v.shape[2]
    qk_head_dim = q.shape[2]

    output = torch.zeros(total_q_tokens, num_heads, v_head_dim, dtype=q.dtype, device=q.device)
    lse = (
        torch.full((num_heads, total_q_tokens), float("-inf"), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )

    cu_q_cpu = cu_seqlens_q.cpu().numpy()
    cu_k_cpu = cu_seqlens_k.cpu().numpy()

    for i in range(batch_size):
        q_start, q_end = int(cu_q_cpu[i]), int(cu_q_cpu[i + 1])
        k_start, k_end = int(cu_k_cpu[i]), int(cu_k_cpu[i + 1])
        if q_end <= q_start or k_end <= k_start:
            continue
        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        if qk_head_dim != v_head_dim:
            v_i = F.pad(v_i, [0, qk_head_dim - v_head_dim], value=0.0)
        if return_softmax_lse:
            attn_scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
            if causal:
                seq_q = q_end - q_start
                seq_k = k_end - k_start
                q_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)
                k_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)
                causal_mask = q_idx >= k_idx
                attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            lse_i = torch.logsumexp(attn_scores, dim=-1)
            lse[:, q_start:q_end] = lse_i.squeeze(0)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            output_i = torch.matmul(attn_weights, v_i)
        else:
            output_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
            )
        output[q_start:q_end] = output_i[0, :, :, :v_head_dim].transpose(0, 1)

    if return_softmax_lse:
        return output, lse
    return output


def _pytorch_merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse):
    """Pure PyTorch merge_attn_states (no C++ dispatch)."""
    p_lse = prefix_lse.transpose(0, 1).unsqueeze(-1).float()
    s_lse = suffix_lse.transpose(0, 1).unsqueeze(-1).float()
    max_lse = torch.maximum(p_lse, s_lse)
    p_se = torch.exp(p_lse - max_lse)
    s_se = torch.exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    merged = (p_scale * prefix_output.float() + s_scale * suffix_output.float()).to(prefix_output.dtype)
    return merged


def _pytorch_concat_k_nope_k_pe(k_nope, k_pe):
    """Pure PyTorch concat_k_nope_k_pe (no C++ dispatch)."""
    k_pe_expanded = k_pe.expand(-1, k_nope.shape[1], -1)
    return torch.cat([k_nope, k_pe_expanded], dim=-1)


def _pytorch_build_absorption_matrices(kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank, dtype):
    """Pure PyTorch build_absorption_matrices (no C++ dispatch)."""
    kv_b_proj_weight_t = kv_b_proj_weight.to(dtype).T
    kv_b_proj_weight_t = kv_b_proj_weight_t.view(kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim)
    w_uk, w_uv = kv_b_proj_weight_t.split([qk_nope_head_dim, v_head_dim], dim=-1)
    W_UK_T = w_uk.permute(1, 2, 0).contiguous()
    W_UV = w_uv.transpose(0, 1).contiguous()
    return W_UK_T, W_UV


def _pytorch_forward_decode(q_nope_proj, q_pe, kv_cache, block_table, seq_lens, scale, kv_lora_rank, qk_rope_head_dim):
    """Pure PyTorch forward_decode (no C++ dispatch)."""
    batch_size = q_nope_proj.shape[0]
    num_heads = q_nope_proj.shape[1]
    block_size = kv_cache.shape[1]
    output = torch.zeros(batch_size, num_heads, kv_lora_rank, dtype=q_nope_proj.dtype, device=q_nope_proj.device)
    seq_lens_cpu = seq_lens.cpu()
    for b in range(batch_size):
        seq_len = int(seq_lens_cpu[b].item())
        if seq_len == 0:
            continue
        gathered_kv = _pytorch_gather_kv_cache(kv_cache, block_table[b], seq_len, block_size)
        kv_c = gathered_kv[:, :kv_lora_rank]
        k_pe_seq = gathered_kv[:, kv_lora_rank:]
        attn_scores = (torch.matmul(q_nope_proj[b], kv_c.T) + torch.matmul(q_pe[b], k_pe_seq.T)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        output[b] = torch.matmul(attn_weights, kv_c)
    return output


# ── Property 7: C++ vs PyTorch per-kernel numerical equivalence ───────────────
# Feature: cpp-kernel-migration, Property 7: C++ vs PyTorch per-kernel numerical equivalence


@_cpp_required
class TestProperty7CppPytorchEquivalence:
    """# Feature: cpp-kernel-migration, Property 7: C++ vs PyTorch per-kernel numerical equivalence

    **Validates: Requirements 3.2, 4.2, 4.3, 6.1, 6.2, 8.2, 8.3, 9.2, 10.2, 11.3, 12.1, 13.2, 15.3**

    For any valid random inputs to each kernel function, the C++ extension
    output shall match the corresponding pure PyTorch implementation output
    within floating-point tolerance (atol=1e-5 for float32).
    """

    @settings(max_examples=100)
    @given(inputs=rms_norm_inputs_with_weight())
    def test_rms_norm_equivalence(self, inputs):
        """rms_norm: C++ vs PyTorch equivalence."""
        x, weight, eps = inputs
        cpp_result = _C.rms_norm(x, weight, eps)
        py_result = _pytorch_rms_norm(x, weight, eps)
        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"rms_norm mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=rope_inputs())
    def test_apply_rope_equivalence(self, inputs):
        """apply_rope: C++ vs PyTorch equivalence."""
        x, cos_sin_cache, positions, is_neox_style = inputs
        cpp_result = _C.apply_rope(x, cos_sin_cache, positions, is_neox_style)
        py_result = _pytorch_apply_rope(x, cos_sin_cache, positions, is_neox_style)
        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"apply_rope mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=kv_cache_round_trip_inputs())
    def test_kv_cache_write_gather_equivalence(self, inputs):
        """write_kv_cache + gather_kv_cache: C++ vs PyTorch equivalence."""
        kv_c, k_pe, block_size = inputs
        num_tokens = kv_c.shape[0]
        head_size = kv_c.shape[1] + k_pe.shape[1]
        slot_mapping = torch.arange(num_tokens, dtype=torch.long)
        num_blocks = (num_tokens + block_size - 1) // block_size

        # C++ path
        kv_cache_cpp = torch.zeros(num_blocks, block_size, head_size)
        _C.write_kv_cache(kv_c, k_pe, kv_cache_cpp, slot_mapping)
        block_table = torch.arange(num_blocks, dtype=torch.long)
        cpp_result = _C.gather_kv_cache(kv_cache_cpp, block_table, num_tokens, block_size)

        # PyTorch path
        kv_cache_py = torch.zeros(num_blocks, block_size, head_size)
        _pytorch_write_kv_cache(kv_c, k_pe, kv_cache_py, slot_mapping)
        py_result = _pytorch_gather_kv_cache(kv_cache_py, block_table, num_tokens, block_size)

        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"kv_cache mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=linear_inputs())
    def test_linear_equivalence(self, inputs):
        """linear: C++ vs PyTorch (F.linear) equivalence."""
        x, weight, bias = inputs
        cpp_result = _C.linear(x, weight, bias)
        py_result = F.linear(x, weight, bias)
        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"linear mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=linear_inputs())
    def test_kv_b_proj_forward_equivalence(self, inputs):
        """kv_b_proj_forward: C++ vs PyTorch (F.linear) equivalence."""
        x, weight, bias = inputs
        cpp_result = _C.kv_b_proj_forward(x, weight, bias)
        py_result = F.linear(x, weight, bias)
        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"kv_b_proj_forward mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=causal_attention_inputs())
    def test_varlen_attention_equivalence(self, inputs):
        """varlen_attention: C++ vs PyTorch equivalence."""
        q, k, v, seq_len = inputs
        head_dim = q.shape[2]
        scale = 1.0 / math.sqrt(head_dim)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        cpp_result = _C.varlen_attention(q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, scale, True, False)
        py_result = _pytorch_varlen_attention(q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, scale, True, False)

        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"varlen_attention mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=forward_decode_inputs())
    def test_forward_decode_equivalence(self, inputs):
        """forward_decode: C++ vs PyTorch equivalence."""
        (q_nope_proj, q_pe, kv_cache, block_table, seq_lens, scale, kv_lora_rank, qk_rope_head_dim) = inputs

        cpp_result = _C.forward_decode(
            q_nope_proj, q_pe, kv_cache, block_table, seq_lens, scale, kv_lora_rank, qk_rope_head_dim
        )
        py_result = _pytorch_forward_decode(
            q_nope_proj, q_pe, kv_cache, block_table, seq_lens, scale, kv_lora_rank, qk_rope_head_dim
        )

        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"forward_decode mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=lse_merge_inputs())
    def test_merge_attn_states_equivalence(self, inputs):
        """merge_attn_states: C++ vs PyTorch equivalence."""
        output, lse = inputs
        suffix_output = torch.randn_like(output)
        suffix_lse = torch.randn_like(lse)

        cpp_result = _C.merge_attn_states(output, lse, suffix_output, suffix_lse)
        py_result = _pytorch_merge_attn_states(output, lse, suffix_output, suffix_lse)

        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"merge_attn_states mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=concat_inputs())
    def test_concat_k_nope_k_pe_equivalence(self, inputs):
        """concat_k_nope_k_pe: C++ vs PyTorch equivalence."""
        k_nope, k_pe = inputs
        cpp_result = _C.concat_k_nope_k_pe(k_nope, k_pe)
        py_result = _pytorch_concat_k_nope_k_pe(k_nope, k_pe)

        assert torch.allclose(cpp_result, py_result, atol=1e-5), (
            f"concat_k_nope_k_pe mismatch: max diff={(cpp_result - py_result).abs().max()}"
        )

    @settings(max_examples=100)
    @given(inputs=absorption_inputs())
    def test_build_absorption_matrices_equivalence(self, inputs):
        """build_absorption_matrices: C++ vs PyTorch equivalence."""
        (kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank) = inputs

        cpp_W_UK_T, cpp_W_UV = _C.build_absorption_matrices(
            kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank, torch.float32
        )
        py_W_UK_T, py_W_UV = _pytorch_build_absorption_matrices(
            kv_b_proj_weight, num_heads, qk_nope_head_dim, v_head_dim, kv_lora_rank, torch.float32
        )

        assert torch.allclose(cpp_W_UK_T, py_W_UK_T, atol=1e-5), (
            f"W_UK_T mismatch: max diff={(cpp_W_UK_T - py_W_UK_T).abs().max()}"
        )
        assert torch.allclose(cpp_W_UV, py_W_UV, atol=1e-5), (
            f"W_UV mismatch: max diff={(cpp_W_UV - py_W_UV).abs().max()}"
        )
