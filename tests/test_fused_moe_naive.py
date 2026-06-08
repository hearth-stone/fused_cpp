# -*- coding: utf-8 -*-
"""Tests for the naive fused MoE expert FFN."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        d = gate_up.shape[-1] // 2
        return F.silu(gate_up[..., :d]) * gate_up[..., d:]
    if activation == "gelu":
        d = gate_up.shape[-1] // 2
        return F.gelu(gate_up[..., :d], approximate="none") * gate_up[..., d:]
    if activation == "swigluoai":
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=7.0)
        up = up.clamp(min=-7.0, max=7.0)
        return (up + 1) * gate * torch.sigmoid(gate * 1.702)
    raise AssertionError(f"test does not implement activation {activation!r}")


def _token_reference(
    input: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: str = "silu",
    skip_weighted: bool = False,
) -> torch.Tensor:
    out = torch.empty_like(input)
    for token_idx in range(input.shape[0]):
        acc = torch.zeros_like(input[token_idx])
        for slot_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, slot_idx].item())
            gate_up = F.linear(
                input[token_idx:token_idx + 1],
                w13_weight[expert_id],
                None if w13_bias is None else w13_bias[expert_id],
            )
            intermediate = _activation(gate_up, activation)
            expert_out = F.linear(
                intermediate,
                w2_weight[expert_id],
                None if w2_bias is None else w2_bias[expert_id],
            )[0]
            if skip_weighted:
                acc = expert_out
            else:
                acc = acc + expert_out * topk_weights[token_idx, slot_idx]
        out[token_idx] = acc
    return out


def _random_case(seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    num_tokens = 7
    hidden_size = 5
    ffn_hidden_size = 9
    num_experts = 4
    top_k = 2
    input = torch.randn(num_tokens, hidden_size, generator=generator)
    w13_weight = torch.randn(
        num_experts, 2 * ffn_hidden_size, hidden_size, generator=generator
    )
    w2_weight = torch.randn(
        num_experts, hidden_size, ffn_hidden_size, generator=generator
    )
    w13_bias = torch.randn(num_experts, 2 * ffn_hidden_size, generator=generator)
    w2_bias = torch.randn(num_experts, hidden_size, generator=generator)
    topk_ids = torch.tensor(
        [
            [3, 1],
            [0, 2],
            [2, 3],
            [1, 0],
            [3, 0],
            [2, 1],
            [0, 3],
        ],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, generator=generator), dim=-1
    )
    return input, w13_weight, w2_weight, w13_bias, w2_bias, topk_weights, topk_ids


def test_fused_moe_naive_imports() -> None:
    from fused_cpp import fused_moe_naive as top_level
    from fused_cpp.moe import fused_moe_naive as moe_level
    from fused_cpp.moe import naive_fused_moe

    assert top_level is moe_level
    assert naive_fused_moe is moe_level


@pytest.mark.parametrize("activation", ["silu", "gelu", "swigluoai"])
def test_fused_moe_naive_matches_token_reference(activation: str) -> None:
    from fused_cpp.moe import fused_moe_naive

    input, w13_weight, w2_weight, w13_bias, w2_bias, topk_weights, topk_ids = (
        _random_case(seed=123)
    )

    out = fused_moe_naive(
        input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation=activation,
        global_num_experts=w13_weight.shape[0],
    )
    ref = _token_reference(
        input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation=activation,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fused_moe_naive_out_buffer() -> None:
    from fused_cpp.moe import fused_moe_naive

    input, w13_weight, w2_weight, w13_bias, w2_bias, topk_weights, topk_ids = (
        _random_case(seed=7)
    )
    out_buffer = torch.empty_like(input)
    ret = fused_moe_naive(
        input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        out=out_buffer,
    )
    expected = _token_reference(
        input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )
    assert ret is out_buffer
    torch.testing.assert_close(out_buffer, expected, rtol=1e-5, atol=1e-5)


def test_fused_moe_naive_skip_weighted_matches_preweighted_input() -> None:
    from fused_cpp.moe import fused_moe_naive

    input, w13_weight, w2_weight, w13_bias, w2_bias, _, topk_ids_2 = (
        _random_case(seed=99)
    )
    topk_ids = topk_ids_2[:, :1].contiguous()
    topk_weights = torch.rand(input.shape[0], 1)
    weighted_input = input * topk_weights.to(input.dtype)

    out = fused_moe_naive(
        weighted_input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        skip_weighted=True,
    )
    ref = _token_reference(
        weighted_input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        skip_weighted=True,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_fused_moe_naive_rejects_bad_expert_id() -> None:
    from fused_cpp.moe import fused_moe_naive

    input, w13_weight, w2_weight, _, _, topk_weights, topk_ids = _random_case(seed=5)
    bad_ids = topk_ids.clone()
    bad_ids[0, 0] = w13_weight.shape[0]

    with pytest.raises(ValueError, match="topk_ids out of range"):
        fused_moe_naive(input, w13_weight, w2_weight, topk_weights, bad_ids)


def test_fused_moe_naive_rejects_skip_weighted_topk_gt_one() -> None:
    from fused_cpp.moe import fused_moe_naive

    input, w13_weight, w2_weight, _, _, topk_weights, topk_ids = _random_case(seed=6)

    with pytest.raises(AssertionError, match="top_k == 1"):
        fused_moe_naive(
            input,
            w13_weight,
            w2_weight,
            topk_weights,
            topk_ids,
            skip_weighted=True,
        )
