# -*- coding: utf-8 -*-
"""BF16 linear wrapper backed by refs/i8gemm bf16gemm.

Public API:

    from fused_cpp import bf16_linear

    packed = bf16_linear.prepare(weight_bf16)
    out = bf16_linear.linear(x_bf16, packed, out_dtype=torch.bfloat16)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        bf16_linear_prepare_weight as _bf16_linear_prepare_weight,
        bf16_linear_prepacked_to_dtype as _bf16_linear_prepacked_to_dtype,
        bf16_linear_to_dtype as _bf16_linear_to_dtype,
    )

    _supports_bf16_linear = True
except (ImportError, AttributeError):
    _supports_bf16_linear = False


def _require_backend() -> None:
    if not _supports_bf16_linear:
        raise RuntimeError("bf16_linear backend unavailable: fused_cpp._C is not built")


@dataclass(frozen=True)
class PreparedBF16LinearWeight:
    """Packed bf16 linear weight and metadata.

    ``weight`` uses the PyTorch linear layout ``[N, K]``. ``packed_weight`` is
    backend-specific and should be treated as opaque by callers.
    """

    packed_weight: torch.Tensor
    k: int
    n: int
    n_padded: int


def prepare(weight: torch.Tensor) -> PreparedBF16LinearWeight:
    """Prepack a bf16 ``[N, K]`` linear weight for repeated GEMMs."""
    _require_backend()
    if weight.dim() != 2:
        raise RuntimeError(
            f"bf16_linear.prepare: weight must be 2D [N, K], got {weight.dim()}D"
        )
    if weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"bf16_linear.prepare: weight dtype must be torch.bfloat16, got {weight.dtype}"
        )
    packed, k, n, n_padded = _bf16_linear_prepare_weight(weight.contiguous())
    return PreparedBF16LinearWeight(
        packed_weight=packed,
        k=int(k),
        n=int(n),
        n_padded=int(n_padded),
    )


def linear(
    x: torch.Tensor,
    packed: PreparedBF16LinearWeight,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    nthreads: int = 0,
) -> torch.Tensor:
    """Run ``x @ packed.weight.T`` using a prepacked bf16 weight."""
    _require_backend()
    if not isinstance(packed, PreparedBF16LinearWeight):
        raise RuntimeError("bf16_linear.linear: packed must come from bf16_linear.prepare")
    if x.shape[-1] != packed.k:
        raise RuntimeError(
            f"bf16_linear.linear: x last dim must be K={packed.k}, got {x.shape[-1]}"
        )
    if x.dtype != torch.bfloat16:
        raise RuntimeError(
            f"bf16_linear.linear: x dtype must be torch.bfloat16, got {x.dtype}"
        )
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(
            f"bf16_linear.linear: out_dtype must be float32/bfloat16, got {out_dtype}"
        )

    batch_shape = tuple(x.shape[:-1])
    if x.numel() == 0:
        return torch.empty((*batch_shape, packed.n), dtype=out_dtype, device=x.device)

    x_2d = x.contiguous().reshape(-1, packed.k)
    out_2d = _bf16_linear_prepacked_to_dtype(
        x_2d,
        packed.packed_weight,
        packed.k,
        packed.n,
        packed.n_padded,
        out_dtype == torch.bfloat16,
        int(nthreads),
    )
    return out_2d.reshape((*batch_shape, packed.n))


def linear_raw(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    nthreads: int = 0,
) -> torch.Tensor:
    """Run bf16 linear with a raw ``[N, K]`` weight, packing it per call."""
    _require_backend()
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(
            f"bf16_linear.linear_raw: out_dtype must be float32/bfloat16, got {out_dtype}"
        )
    batch_shape = tuple(x.shape[:-1])
    if x.numel() == 0:
        if weight.dim() != 2:
            raise RuntimeError(
                f"bf16_linear.linear_raw: weight must be 2D [N, K], got {weight.dim()}D"
            )
        return torch.empty((*batch_shape, weight.shape[0]), dtype=out_dtype, device=x.device)
    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    out_2d = _bf16_linear_to_dtype(
        x_2d,
        weight.contiguous(),
        out_dtype == torch.bfloat16,
        int(nthreads),
    )
    return out_2d.reshape((*batch_shape, weight.shape[0]))


mm = linear


__all__ = [
    "PreparedBF16LinearWeight",
    "_supports_bf16_linear",
    "linear",
    "linear_raw",
    "mm",
    "prepare",
]
