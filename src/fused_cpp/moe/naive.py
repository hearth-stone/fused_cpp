# -*- coding: utf-8 -*-
"""Naive PyTorch Fused MoE expert FFN.

This module mirrors the tensor semantics of vLLM's ``cpu_fused_moe_torch``:
``topk_weights`` and ``topk_ids`` are already computed by the router, tokens are
grouped by expert, each expert runs gate/up -> activation -> down, then top-k
expert outputs are weighted and reduced back to one output per token.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["fused_moe_naive", "naive_fused_moe"]


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="none") * x[..., d:]


def _swigluoai_and_mul(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


_ACTIVATION_FNS = {
    "silu": _silu_and_mul,
    "gelu": _gelu_and_mul,
    "swigluoai": _swigluoai_and_mul,
}


def _activation_name(activation: Any) -> str:
    value = getattr(activation, "value", activation)
    if not isinstance(value, str):
        value = str(value)
    value = {"gelu_pytorch_tanh": "gelu_tanh"}.get(value, value)
    if value not in _ACTIVATION_FNS:
        supported = ", ".join(sorted(_ACTIVATION_FNS))
        raise ValueError(
            f"Unsupported MoE activation {value!r}; supported activations: {supported}"
        )
    return value


def _check_inputs(
    input: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    global_num_experts: int,
    out: torch.Tensor | None,
) -> int:
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D [tokens, hidden], got {tuple(input.shape)}")
    if w13_weight.dim() != 3:
        raise ValueError(
            "w13_weight must be 3-D [experts, 2 * ffn_hidden, hidden], got "
            f"{tuple(w13_weight.shape)}"
        )
    if w2_weight.dim() != 3:
        raise ValueError(
            "w2_weight must be 3-D [experts, hidden, ffn_hidden], got "
            f"{tuple(w2_weight.shape)}"
        )
    if topk_ids.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError(
            "topk_ids and topk_weights must both be 2-D [tokens, top_k], got "
            f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights shapes must match, got "
            f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if topk_ids.shape[0] != input.shape[0]:
        raise ValueError(
            f"topk first dimension must equal token count {input.shape[0]}, got "
            f"{topk_ids.shape[0]}"
        )
    if topk_ids.shape[1] == 0:
        raise ValueError("top_k dimension must be non-zero")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(
            f"topk_weights must use a floating dtype, got {topk_weights.dtype}"
        )
    if (
        input.device != w13_weight.device
        or input.device != w2_weight.device
        or input.device != topk_ids.device
        or input.device != topk_weights.device
    ):
        raise ValueError("input, weights, topk_ids, and topk_weights must share device")

    num_weight_experts, gate_up_size, hidden_size = w13_weight.shape
    if input.shape[1] != hidden_size:
        raise ValueError(
            f"input hidden size {input.shape[1]} does not match w13 hidden size "
            f"{hidden_size}"
        )
    if gate_up_size % 2 != 0:
        raise ValueError(f"w13 output dimension must be even, got {gate_up_size}")
    ffn_hidden_size = gate_up_size // 2
    expected_w2 = (num_weight_experts, hidden_size, ffn_hidden_size)
    if tuple(w2_weight.shape) != expected_w2:
        raise ValueError(
            f"w2_weight must have shape {expected_w2}, got {tuple(w2_weight.shape)}"
        )

    if w13_bias is not None:
        if tuple(w13_bias.shape) != (num_weight_experts, gate_up_size):
            raise ValueError(
                "w13_bias must have shape "
                f"{(num_weight_experts, gate_up_size)}, got {tuple(w13_bias.shape)}"
            )
        if w13_bias.device != input.device:
            raise ValueError("w13_bias must be on the same device as input")
    if w2_bias is not None:
        if tuple(w2_bias.shape) != (num_weight_experts, hidden_size):
            raise ValueError(
                "w2_bias must have shape "
                f"{(num_weight_experts, hidden_size)}, got {tuple(w2_bias.shape)}"
            )
        if w2_bias.device != input.device:
            raise ValueError("w2_bias must be on the same device as input")

    if global_num_experts < 0:
        num_experts = num_weight_experts
    else:
        num_experts = int(global_num_experts)
        if num_experts <= 0:
            raise ValueError(f"global_num_experts must be positive, got {num_experts}")
        if num_experts > num_weight_experts:
            raise ValueError(
                "global_num_experts cannot exceed available expert weights: "
                f"{num_experts} > {num_weight_experts}"
            )

    if topk_ids.numel() > 0:
        id_min = int(topk_ids.min().item())
        id_max = int(topk_ids.max().item())
        if id_min < 0 or id_max >= num_experts:
            raise ValueError(
                f"topk_ids out of range: min={id_min}, max={id_max}, "
                f"valid range [0, {num_experts})"
            )

    if out is not None:
        if tuple(out.shape) != tuple(input.shape):
            raise ValueError(f"out must have shape {tuple(input.shape)}, got {tuple(out.shape)}")
        if out.device != input.device:
            raise ValueError("out must be on the same device as input")
        if out.dtype != input.dtype:
            raise ValueError(f"out dtype must match input dtype, got {out.dtype} vs {input.dtype}")

    return num_experts


def fused_moe_naive(
    input: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a naive fused MoE expert FFN.

    Args:
        input: ``[T, H]`` hidden states.
        w13_weight: ``[E, 2 * F, H]`` fused gate/up weights.
        w2_weight: ``[E, H, F]`` down weights.
        topk_weights: ``[T, K]`` router weights aligned with ``topk_ids``.
        topk_ids: ``[T, K]`` selected expert ids.
        w13_bias: Optional ``[E, 2 * F]`` fused gate/up bias.
        w2_bias: Optional ``[E, H]`` down bias.
        activation: One of ``"silu"``, ``"gelu"``, or ``"swigluoai"``. Enum-like
            objects with a string ``.value`` are accepted.
        global_num_experts: Expert count used for routing. ``-1`` infers it from
            ``w13_weight.shape[0]``.
        skip_weighted: Match ``cpu_fused_moe_torch``: skip final top-k weighting.
            The caller is responsible for pre-weighting ``input`` and ``K`` must
            be 1.
        out: Optional output buffer, matching vLLM's mutating op style.

    Returns:
        ``[T, H]`` output tensor. When ``out`` is provided, the same tensor is
        returned after being filled.
    """
    act = _activation_name(activation)
    num_experts = _check_inputs(
        input,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        w13_bias,
        w2_bias,
        global_num_experts,
        out,
    )

    num_tokens, hidden_size = input.shape
    top_k = topk_ids.shape[1]
    if skip_weighted and top_k != 1:
        raise AssertionError("skip_weighted is only valid when top_k == 1")

    if num_tokens == 0:
        final_out = input.new_empty((0, hidden_size))
        if out is not None:
            out.copy_(final_out)
            return out
        return final_out

    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    idxs = flat_ids.argsort()
    sorted_token_idx = torch.div(idxs, top_k, rounding_mode="floor")
    sorted_tokens = input.index_select(0, sorted_token_idx)
    tokens_per_expert = torch.bincount(flat_ids, minlength=num_experts)[:num_experts]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for expert_id, num_for_expert_t in enumerate(tokens_per_expert):
        num_for_expert = int(num_for_expert_t.item())
        end_idx = start_idx + num_for_expert
        if num_for_expert == 0:
            continue

        tokens_for_expert = sorted_tokens[start_idx:end_idx]
        gate_up = F.linear(
            tokens_for_expert,
            w13_weight[expert_id],
            None if w13_bias is None else w13_bias[expert_id],
        )
        intermediate = _ACTIVATION_FNS[act](gate_up)
        expert_out = F.linear(
            intermediate,
            w2_weight[expert_id],
            None if w2_bias is None else w2_bias[expert_id],
        )
        outputs.append(expert_out)
        start_idx = end_idx

    if outputs:
        outs = torch.cat(outputs, dim=0)
    else:
        outs = input.new_empty((0, hidden_size))

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs

    if skip_weighted:
        final_out = new_x
    else:
        final_out = (
            new_x.view(num_tokens, top_k, hidden_size)
            .to(topk_weights.dtype)
            .mul_(topk_weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(new_x.dtype)
        )

    if out is not None:
        out.copy_(final_out)
        return out
    return final_out


naive_fused_moe = fused_moe_naive
