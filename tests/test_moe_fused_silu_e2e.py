# -*- coding: utf-8 -*-
"""Task 6: end-to-end MoE equivalence for the fused SiLU-and-mul path.

The opt-in fused path (prepare with fuse_silu=True) must match the baseline
w13+activation MoE within a tolerance covering bf16 rounding + the poly exp.
"""
from __future__ import annotations

import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE backend requires AArch64",
)


def _bf16(*shape, scale=0.2):
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _moe_case(num_tokens, hidden, ffn, num_experts, top_k, seed=0):
    torch.manual_seed(seed)
    x = _bf16(num_tokens, hidden)
    w13 = _bf16(num_experts, 2 * ffn, hidden)
    w2 = _bf16(num_experts, hidden, ffn)
    topk_ids = torch.tensor(
        [[(i + j) % num_experts for j in range(top_k)] for i in range(num_tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    return x, w13, w2, topk_weights, topk_ids


@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("num_threads", [1, 4])
@pytest.mark.parametrize("num_tokens", [1, 8, 37, 128])
def test_fused_silu_moe_matches_baseline(degree, num_threads, num_tokens):
    # F must be a multiple of 8 for the fused path.
    hidden, ffn, num_experts, top_k = 128, 64, 4, 2
    x, w13, w2, tw, ti = _moe_case(
        num_tokens, hidden, ffn, num_experts, top_k, seed=degree + num_tokens
    )

    base_w = prepare_fused_moe_bf16_tiled_weights(w13, w2)
    fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    assert fused_w.fused_silu is True

    ref = fused_moe_bf16_tiled(
        x, base_w, tw, ti, num_threads=num_threads, activation="silu"
    )
    out = fused_moe_bf16_tiled(
        x,
        fused_w,
        tw,
        ti,
        num_threads=num_threads,
        activation="silu",
        silu_poly_degree=degree,
    )
    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    # Baseline uses std::exp; fused uses poly exp -> allow a modest tolerance.
    torch.testing.assert_close(out.float(), ref.float(), atol=6e-2, rtol=6e-2)


def test_fused_silu_requires_multiple_of_8_F():
    # F=11 is not a multiple of 8 -> prepare(fuse_silu=True) must raise.
    w13 = _bf16(4, 2 * 11, 16)
    w2 = _bf16(4, 16, 11)
    with pytest.raises(Exception):
        prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
