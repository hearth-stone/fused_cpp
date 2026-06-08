# -*- coding: utf-8 -*-
from __future__ import annotations

import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE kernel is only available on AArch64",
)


def _bf16_randn(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape) * 0.2).to(torch.bfloat16)


def _case(seed: int = 0) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    num_tokens = 37
    hidden_size = 13
    ffn_hidden_size = 11
    num_experts = 4
    top_k = 2
    hidden_states = _bf16_randn(num_tokens, hidden_size)
    w13_weight = _bf16_randn(num_experts, 2 * ffn_hidden_size, hidden_size)
    w2_weight = _bf16_randn(num_experts, hidden_size, ffn_hidden_size)
    w13_bias = torch.randn(num_experts, 2 * ffn_hidden_size) * 0.1
    w2_bias = torch.randn(num_experts, hidden_size) * 0.1
    topk_ids = torch.tensor(
        [[(i + j) % num_experts for j in range(top_k)] for i in range(num_tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    return (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    )


@pytest.mark.parametrize("activation", ["silu", "swigluoai"])
def test_fused_moe_bf16_tiled_matches_naive(activation: str) -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=123)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    out = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
        activation=activation,
    )
    ref = fused_moe_naive(
        hidden_states.float(),
        w13_weight.float(),
        w2_weight.float(),
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation=activation,
    ).to(torch.bfloat16)

    assert out.dtype == torch.bfloat16
    assert out.shape == hidden_states.shape
    torch.testing.assert_close(out.float(), ref.float(), atol=7e-2, rtol=7e-2)


def test_fused_moe_bf16_tiled_threaded_matches_single_thread() -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=7)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    serial = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
    )
    threaded = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=3,
    )

    torch.testing.assert_close(threaded.float(), serial.float(), atol=0, rtol=0)


def test_fused_moe_bf16_tiled_out_buffer() -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        _w13_bias,
        _w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=11)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)
    out_buffer = torch.empty_like(hidden_states)

    ret = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        num_threads=2,
        out=out_buffer,
    )

    assert ret is out_buffer
    ref = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
    )
    torch.testing.assert_close(out_buffer.float(), ref.float(), atol=0, rtol=0)
