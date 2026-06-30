# -*- coding: utf-8 -*-
"""BF16 tiled fused MoE wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch

PreparedWeight = Tuple[torch.Tensor, int, int]


@dataclass(frozen=True)
class PreparedBF16TiledFusedMoEWeights:
    """Packed BF16 weights for the tiled fused MoE path."""

    w13: PreparedWeight
    w2: PreparedWeight


try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        fused_moe_bf16_tiled as _fused_moe_bf16_tiled_impl,
    )
    from fused_cpp._C import (  # type: ignore[import-untyped]
        fused_moe_bf16_tiled_prepare_weights as _prepare_bf16_tiled_impl,
    )
    try:
        from fused_cpp._C import (  # type: ignore[import-untyped]
            fused_moe_bf16_tiled_scheduled as _scheduled_impl,
        )

        _fused_moe_bf16_tiled_scheduled_impl = _scheduled_impl
    except (ImportError, AttributeError):
        _fused_moe_bf16_tiled_scheduled_impl = None

    _HAS_BF16_TILED_FUSED_MOE = True
except (ImportError, AttributeError):
    _fused_moe_bf16_tiled_impl = None
    _fused_moe_bf16_tiled_scheduled_impl = None
    _prepare_bf16_tiled_impl = None
    _HAS_BF16_TILED_FUSED_MOE = False


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_backend() -> None:
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError(
            "BF16 tiled fused MoE backend is unavailable "
            "(requires the C++ extension on AArch64)"
        )


def _activation_name(activation: Any) -> str:
    value = getattr(activation, "value", activation)
    if not isinstance(value, str):
        value = str(value)
    value = {"gelu_pytorch_tanh": "gelu_tanh"}.get(value, value)
    if value not in {"silu", "gelu", "swigluoai"}:
        raise ValueError(
            "Unsupported MoE activation "
            f"{value!r}; supported activations: gelu, silu, swigluoai"
        )
    return value


def _check_integer_schedule_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(tensor.shape)}")


def prepare_fused_moe_bf16_tiled_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> PreparedBF16TiledFusedMoEWeights:
    """Pack dense bf16 expert weights for :func:`fused_moe_bf16_tiled`.

    ``w13_weight`` follows the vLLM layout ``[E, 2 * F, H]`` and ``w2_weight``
    follows ``[E, H, F]``. The returned object is reusable across decode steps.
    Set ``FUSED_CPP_MOE_PREPACK_THREADS`` to parallelize packing by expert.
    """
    _require_backend()
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise TypeError("BF16 tiled fused MoE weights must be torch.bfloat16")
    if not w13_weight.device.type == w2_weight.device.type == "cpu":
        raise ValueError("BF16 tiled fused MoE weights must be CPU tensors")

    packed = _prepare_bf16_tiled_impl(
        w13_weight.contiguous(),
        w2_weight.contiguous(),
    )
    return PreparedBF16TiledFusedMoEWeights(
        w13=(packed[0], int(packed[1]), int(packed[2])),
        w2=(packed[3], int(packed[4]), int(packed[5])),
    )


def fused_moe_bf16_tiled(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    num_threads: int = 1,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the C++ tiled fused MoE path using BF16 GEMMs."""
    _require_backend()
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(
            f"topk_weights must use a floating dtype, got {topk_weights.dtype}"
        )
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    if out is not None:
        if tuple(out.shape) != tuple(input.shape):
            raise ValueError(f"out must have shape {tuple(input.shape)}")
        if out.dtype != input.dtype or out.device != input.device:
            raise ValueError("out must match input dtype and device")

    def _contiguous_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        if bias.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("MoE bias must be torch.float32 or torch.bfloat16")
        if bias.device.type != "cpu":
            raise ValueError("MoE bias must be a CPU tensor")
        return bias.contiguous()

    result = _fused_moe_bf16_tiled_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        _contiguous_bias(w13_bias),
        _contiguous_bias(w2_bias),
        int(num_threads),
        _activation_name(activation),
        int(global_num_experts),
        bool(skip_weighted),
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


def fused_moe_bf16_tiled_scheduled(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    wave_offsets: torch.Tensor,
    team_expert_ids: torch.Tensor,
    team_threads: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    num_threads: int = 1,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the BF16 tiled MoE path using an externally supplied schedule.

    ``wave_offsets`` has shape ``[num_waves + 1]`` and indexes into
    ``team_expert_ids`` / ``team_threads``. Each team computes one active
    expert using ``team_threads[i]`` cooperative threads.
    """
    _require_backend()
    if _fused_moe_bf16_tiled_scheduled_impl is None:
        raise RuntimeError(
            "BF16 tiled scheduled MoE backend is unavailable; rebuild the C++ "
            "extension with fused_moe_bf16_tiled_scheduled support."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(
            f"topk_weights must use a floating dtype, got {topk_weights.dtype}"
        )
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    _check_integer_schedule_tensor(wave_offsets, "wave_offsets")
    _check_integer_schedule_tensor(team_expert_ids, "team_expert_ids")
    _check_integer_schedule_tensor(team_threads, "team_threads")
    if out is not None:
        if tuple(out.shape) != tuple(input.shape):
            raise ValueError(f"out must have shape {tuple(input.shape)}")
        if out.dtype != input.dtype or out.device != input.device:
            raise ValueError("out must match input dtype and device")

    def _contiguous_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        if bias.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("MoE bias must be torch.float32 or torch.bfloat16")
        if bias.device.type != "cpu":
            raise ValueError("MoE bias must be a CPU tensor")
        return bias.contiguous()

    result = _fused_moe_bf16_tiled_scheduled_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        wave_offsets.contiguous(),
        team_expert_ids.contiguous(),
        team_threads.contiguous(),
        _contiguous_bias(w13_bias),
        _contiguous_bias(w2_bias),
        int(num_threads),
        _activation_name(activation),
        int(global_num_experts),
        bool(skip_weighted),
    )
    if out is not None:
        out.copy_(result)
        return out
    return result


bf16_tiled_fused_moe = fused_moe_bf16_tiled
bf16_tiled_fused_moe_scheduled = fused_moe_bf16_tiled_scheduled
prepare_bf16_tiled_fused_moe_weights = prepare_fused_moe_bf16_tiled_weights


__all__ = [
    "PreparedBF16TiledFusedMoEWeights",
    "PreparedWeight",
    "_HAS_BF16_TILED_FUSED_MOE",
    "fused_moe_bf16_tiled",
    "fused_moe_bf16_tiled_scheduled",
    "bf16_tiled_fused_moe",
    "bf16_tiled_fused_moe_scheduled",
    "prepare_fused_moe_bf16_tiled_weights",
    "prepare_bf16_tiled_fused_moe_weights",
]
