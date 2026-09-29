# -*- coding: utf-8 -*-
"""I8 GEMM wrapper with dynamic activation scaling.

Public API:

    from fused_cpp import i8gemm

    packed = i8gemm.prepare(weight_int8, weight_scale, backend="auto")
    out = i8gemm.dynamic_scaled_mm(x, packed, bias=None,
                                   out_dtype=torch.bfloat16, nthreads=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        i8gemm_dynamic_scaled_mm as _i8gemm_dynamic_scaled_mm,
        i8gemm_prepare as _i8gemm_prepare,
    )

    _supports_i8gemm = True
except ImportError:
    _supports_i8gemm = False

try:
    from fused_cpp._C import i8gemm_dynamic_scaled_mm_pair as _i8gemm_dynamic_scaled_mm_pair  # type: ignore[import-untyped]
except (ImportError, AttributeError):
    _i8gemm_dynamic_scaled_mm_pair = None


def _require_backend() -> None:
    if not _supports_i8gemm:
        raise RuntimeError("i8gemm backend unavailable: fused_cpp._C is not built")


@dataclass(frozen=True)
class PreparedI8GEMMWeight:
    """Packed int8 linear weight and metadata.

    ``weight_int8`` uses the vLLM linear layout ``[N, K]``.  ``packed_weight``
    is backend-specific and should be treated as opaque by callers.
    """

    packed_weight: torch.Tensor
    weight_scale: torch.Tensor
    k: int
    n: int
    k_padded: int
    n_padded: int
    backend: str


def prepare(
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    backend: str = "auto",
) -> PreparedI8GEMMWeight:
    """Prepare an int8 ``[N, K]`` vLLM linear weight for repeated GEMMs."""
    _require_backend()

    if weight_int8.dim() != 2:
        raise RuntimeError(f"i8gemm.prepare: weight_int8 must be 2D [N, K], got {weight_int8.dim()}D")
    if weight_int8.dtype != torch.int8:
        raise RuntimeError(f"i8gemm.prepare: weight_int8 dtype must be torch.int8, got {weight_int8.dtype}")
    if weight_scale.numel() not in (1, int(weight_int8.shape[0])):
        raise RuntimeError("i8gemm.prepare: weight_scale must be scalar or shape [N]")

    packed, scale, k, n, k_padded, n_padded, selected = _i8gemm_prepare(
        weight_int8.contiguous(), weight_scale.contiguous(), str(backend)
    )
    return PreparedI8GEMMWeight(
        packed_weight=packed,
        weight_scale=scale,
        k=int(k),
        n=int(n),
        k_padded=int(k_padded),
        n_padded=int(n_padded),
        backend=str(selected),
    )


def dynamic_scaled_mm(
    x: torch.Tensor,
    packed: PreparedI8GEMMWeight,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    nthreads: int = 0,
) -> torch.Tensor:
    """Run ``x @ packed.weight.T`` using dynamic per-token int8 quantization.

    The compute path quantizes ``x`` per row, runs int8 GEMM, dequantizes with
    ``x_scale[row] * packed.weight_scale[col]``, applies optional bias, and
    returns float32 or bfloat16.
    """
    _require_backend()

    if not isinstance(packed, PreparedI8GEMMWeight):
        raise RuntimeError("i8gemm.dynamic_scaled_mm: packed must come from i8gemm.prepare")
    if x.shape[-1] != packed.k:
        raise RuntimeError(f"i8gemm.dynamic_scaled_mm: x last dim must be K={packed.k}, got {x.shape[-1]}")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"i8gemm.dynamic_scaled_mm: x dtype must be float32/bfloat16, got {x.dtype}")
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"i8gemm.dynamic_scaled_mm: out_dtype must be float32/bfloat16, got {out_dtype}")
    if bias is not None:
        if bias.dim() != 1 or int(bias.shape[0]) != packed.n:
            raise RuntimeError(f"i8gemm.dynamic_scaled_mm: bias must be shape [{packed.n}]")
        bias = bias.contiguous()

    batch_shape = tuple(x.shape[:-1])
    out_shape = (*batch_shape, packed.n)
    if x.numel() == 0:
        return torch.empty(out_shape, dtype=out_dtype, device=x.device)

    x_2d = x.contiguous().reshape(-1, packed.k)
    out_2d = torch.empty((x_2d.shape[0], packed.n), dtype=out_dtype, device=x.device)
    _i8gemm_dynamic_scaled_mm(
        out_2d,
        x_2d,
        packed.packed_weight,
        packed.weight_scale,
        bias,
        packed.k,
        packed.n,
        packed.k_padded,
        packed.n_padded,
        int(nthreads),
    )
    return out_2d.reshape(out_shape)


def mm(
    x: torch.Tensor,
    packed: PreparedI8GEMMWeight,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    nthreads: int = 0,
) -> torch.Tensor:
    """Alias for :func:`dynamic_scaled_mm`."""
    return dynamic_scaled_mm(x, packed, bias=bias, out_dtype=out_dtype, nthreads=nthreads)


def dynamic_scaled_mm_pair(
    x: torch.Tensor,
    first: PreparedI8GEMMWeight,
    second: PreparedI8GEMMWeight,
    *,
    nthreads: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two BF16-output GEMMs while sharing dynamic per-row activation quantization."""
    _require_backend()
    if _i8gemm_dynamic_scaled_mm_pair is None:
        raise RuntimeError("i8gemm dynamic pair kernel is unavailable in this build")
    if not isinstance(first, PreparedI8GEMMWeight) or not isinstance(second, PreparedI8GEMMWeight):
        raise TypeError("first and second must be PreparedI8GEMMWeight")
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.dim() != 2:
        raise TypeError("i8gemm.dynamic_scaled_mm_pair input must be CPU BF16 [M, K]")
    if first.k != second.k or first.k_padded != second.k_padded or x.shape[1] != first.k:
        raise ValueError("paired i8gemm weights and input must share K and Kp")
    if int(nthreads) < 0:
        raise ValueError("nthreads must be non-negative; zero selects the backend default")
    x = x.contiguous()
    first_output = torch.empty((x.shape[0], first.n), dtype=torch.bfloat16)
    second_output = torch.empty((x.shape[0], second.n), dtype=torch.bfloat16)
    _i8gemm_dynamic_scaled_mm_pair(
        first_output,
        second_output,
        x,
        first.packed_weight,
        first.weight_scale,
        first.k,
        first.n,
        first.k_padded,
        first.n_padded,
        second.packed_weight,
        second.weight_scale,
        second.n,
        second.n_padded,
        int(nthreads),
    )
    return first_output, second_output


__all__ = [
    "PreparedI8GEMMWeight",
    "_supports_i8gemm",
    "dynamic_scaled_mm",
    "dynamic_scaled_mm_pair",
    "mm",
    "prepare",
]
