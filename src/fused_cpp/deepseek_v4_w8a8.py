# -*- coding: utf-8 -*-
"""Explicit W8A8 linear selection for DeepSeek V4 attention projections."""

from __future__ import annotations

from dataclasses import dataclass
import platform
import sys

import torch

from fused_cpp import i8gemm


def _w8a8_available() -> bool:
    if sys.platform != "linux" or platform.machine() not in {"aarch64", "arm64"} or not i8gemm._supports_i8gemm:
        return False
    try:
        cpuinfo = open("/proc/cpuinfo", encoding="utf-8", errors="ignore").read().lower()
    except OSError:
        return False
    flags = set(cpuinfo.replace("\n", " ").split())
    return "sve" in flags and ("i8mm" in flags or "asimdi8mm" in flags)


_HAS_DEEPSEEK_V4_W8A8 = _w8a8_available()


@dataclass(frozen=True)
class PreparedDeepSeekV4W8A8LinearWeight:
    """Opaque per-output-channel W8 weight for dynamic per-row A8 GEMM."""

    packed: i8gemm.PreparedI8GEMMWeight

    @property
    def k(self) -> int:
        return self.packed.k

    @property
    def n(self) -> int:
        return self.packed.n


def _quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16:
        raise TypeError("DeepSeek V4 W8A8 source weight must be a CPU torch.bfloat16 tensor")
    if weight.dim() != 2:
        raise ValueError(f"DeepSeek V4 W8A8 source weight must be [N, K], got {tuple(weight.shape)}")
    fp32 = weight.float()
    maximum = fp32.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    quantized = (fp32 / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.squeeze(1).contiguous()


def prepare_deepseek_v4_w8a8_linear_weight(
    weight: torch.Tensor,
    *,
    backend: str = "sve",
) -> PreparedDeepSeekV4W8A8LinearWeight:
    """Quantize one BF16 ``[N, K]`` attention weight and pack it once."""
    quantized, scale = _quantize_weight_per_output_channel(weight)
    return prepare_deepseek_v4_w8a8_linear_quantized_weight(quantized, scale, backend=backend)


def prepare_deepseek_v4_w8a8_linear_quantized_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    backend: str = "sve",
) -> PreparedDeepSeekV4W8A8LinearWeight:
    """Pack checkpoint INT8 ``[N, K]`` weight with FP32 channel scales."""
    if weight.device.type != "cpu" or weight.dtype != torch.int8 or weight.dim() != 2:
        raise TypeError("DeepSeek V4 quantized weight must be a CPU torch.int8 [N, K] tensor")
    if scale.device.type != "cpu" or scale.dtype != torch.float32:
        raise TypeError("DeepSeek V4 W8 scale must be a CPU torch.float32 tensor")
    if scale.numel() != weight.shape[0]:
        raise ValueError(f"DeepSeek V4 W8 scale must contain N={weight.shape[0]} values")
    return PreparedDeepSeekV4W8A8LinearWeight(
        packed=i8gemm.prepare(weight.contiguous(), scale.reshape(-1).contiguous(), backend=backend)
    )


def deepseek_v4_w8a8_linear(
    input: torch.Tensor,
    weight: PreparedDeepSeekV4W8A8LinearWeight,
    *,
    num_threads: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse dynamic A8 quantization, i8mm GEMM, channel dequant, and BF16 store."""
    if not isinstance(weight, PreparedDeepSeekV4W8A8LinearWeight):
        raise TypeError("weight must be PreparedDeepSeekV4W8A8LinearWeight")
    if input.device.type != "cpu" or input.dtype != torch.bfloat16:
        raise TypeError("DeepSeek V4 W8A8 input must be a CPU torch.bfloat16 tensor")
    if input.shape[-1] != weight.k:
        raise ValueError(f"DeepSeek V4 W8A8 input K must be {weight.k}, got {input.shape[-1]}")
    if int(num_threads) < 0:
        raise ValueError("num_threads must be non-negative; zero selects the backend default")
    result = i8gemm.dynamic_scaled_mm(
        input,
        weight.packed,
        out_dtype=torch.bfloat16,
        nthreads=int(num_threads),
    )
    if out is None:
        return result
    if out.device != result.device or out.dtype != result.dtype or out.shape != result.shape or not out.is_contiguous():
        raise ValueError("out must be contiguous CPU BF16 with shape [*, N]")
    out.copy_(result)
    return out


def deepseek_v4_wo_b_w8a8(
    input: torch.Tensor,
    weight: PreparedDeepSeekV4W8A8LinearWeight,
    *,
    num_threads: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the TP-local ``attn.wo_b`` projection; collective reduction stays external."""
    return deepseek_v4_w8a8_linear(input, weight, num_threads=num_threads, out=out)


__all__ = [
    "PreparedDeepSeekV4W8A8LinearWeight",
    "_HAS_DEEPSEEK_V4_W8A8",
    "deepseek_v4_w8a8_linear",
    "deepseek_v4_wo_b_w8a8",
    "prepare_deepseek_v4_w8a8_linear_quantized_weight",
    "prepare_deepseek_v4_w8a8_linear_weight",
]
