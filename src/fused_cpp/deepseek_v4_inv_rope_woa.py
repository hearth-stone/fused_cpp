"""DeepSeek-V4 inverse-RoPE plus grouped WO_A operator contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


try:
    from fused_cpp._moe_C import (  # type: ignore[attr-defined, import-untyped]
        deepseek_v4_inv_rope_grouped_woa as _native_execute,
        deepseek_v4_inv_rope_woa_available as _native_available,
        deepseek_v4_inv_rope_woa_prepare as _native_prepare,
    )

    _HAS_DEEPSEEK_V4_INV_ROPE_WOA = bool(_native_available())
except (ImportError, AttributeError):
    _native_available = None
    _native_execute = None
    _native_prepare = None
    _HAS_DEEPSEEK_V4_INV_ROPE_WOA = False

if not _HAS_DEEPSEEK_V4_INV_ROPE_WOA:
    try:
        from fused_cpp._C import (  # type: ignore[attr-defined, import-untyped]
            deepseek_v4_inv_rope_grouped_woa as _native_execute,
            deepseek_v4_inv_rope_woa_available as _native_available,
            deepseek_v4_inv_rope_woa_prepare as _native_prepare,
        )

        _HAS_DEEPSEEK_V4_INV_ROPE_WOA = bool(_native_available())
    except (ImportError, AttributeError):
        _native_available = None
        _native_execute = None
        _native_prepare = None
        _HAS_DEEPSEEK_V4_INV_ROPE_WOA = False


@dataclass(frozen=True)
class PreparedDeepseekV4InvRopeWoa:
    """Opaque grouped WO_A weight plus execution geometry."""

    packed_weight: torch.Tensor
    n_groups: int
    heads_per_group: int
    head_dim: int
    nope_dim: int
    rope_dim: int
    output_rank: int
    backend: str


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _resolve_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    if backend == "auto":
        return "arm_sve_bf16" if _HAS_DEEPSEEK_V4_INV_ROPE_WOA else "torch"
    if backend not in {"torch", "arm_sve_bf16"}:
        raise ValueError(f"unsupported inverse-RoPE WO_A backend {backend!r}")
    if backend == "arm_sve_bf16" and not _HAS_DEEPSEEK_V4_INV_ROPE_WOA:
        raise RuntimeError("arm_sve_bf16 inverse-RoPE WO_A backend is unavailable in this build")
    return backend


def prepare_deepseek_v4_inv_rope_woa(
    wo_a_weight: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    head_dim: int,
    rope_dim: int,
    backend: str = "auto",
) -> PreparedDeepseekV4InvRopeWoa:
    """Prepare TP-local BF16 WO_A weight for grouped output projection."""
    n_groups = _positive_int(n_groups, "n_groups")
    heads_per_group = _positive_int(heads_per_group, "heads_per_group")
    head_dim = _positive_int(head_dim, "head_dim")
    rope_dim = int(rope_dim)
    if rope_dim < 0 or rope_dim > head_dim or rope_dim % 2:
        raise ValueError(f"rope_dim must be even and in [0, head_dim], got {rope_dim} for head_dim={head_dim}")
    if wo_a_weight.device.type != "cpu" or wo_a_weight.dtype != torch.bfloat16:
        raise TypeError("wo_a_weight must be a CPU torch.bfloat16 tensor")
    if wo_a_weight.dim() != 2 or not wo_a_weight.is_contiguous():
        raise ValueError("wo_a_weight must be contiguous [G * R, P * DH]")
    grouped_dim = heads_per_group * head_dim
    if int(wo_a_weight.shape[1]) != grouped_dim:
        raise ValueError(
            f"wo_a_weight input dimension must be P * DH={grouped_dim}, got {int(wo_a_weight.shape[1])}"
        )
    if int(wo_a_weight.shape[0]) == 0 or int(wo_a_weight.shape[0]) % n_groups:
        raise ValueError("wo_a_weight rows must be a positive multiple of n_groups")
    output_rank = int(wo_a_weight.shape[0]) // n_groups
    selected = _resolve_backend(backend)
    if selected == "torch":
        packed_weight = wo_a_weight.view(n_groups, output_rank, grouped_dim)
    else:
        assert _native_prepare is not None
        packed_weight, output_rank, _, _ = _native_prepare(
            wo_a_weight,
            n_groups,
            heads_per_group,
            head_dim,
            rope_dim,
            selected,
        )
    return PreparedDeepseekV4InvRopeWoa(
        packed_weight=packed_weight,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        nope_dim=head_dim - rope_dim,
        rope_dim=rope_dim,
        output_rank=int(output_rank),
        backend=selected,
    )


def _check_no_storage_alias(output: torch.Tensor, tensors: Sequence[torch.Tensor]) -> None:
    output_storage = output.untyped_storage().data_ptr()
    for tensor in tensors:
        if tensor.numel() and tensor.untyped_storage().data_ptr() == output_storage:
            raise ValueError("out must not alias any operator input or prepared weight")


def _prepare_output(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: PreparedDeepseekV4InvRopeWoa,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if not isinstance(weights, PreparedDeepseekV4InvRopeWoa):
        raise TypeError("weights must be PreparedDeepseekV4InvRopeWoa")
    if o.device.type != "cpu" or o.dtype != torch.bfloat16:
        raise TypeError("o must be a CPU torch.bfloat16 tensor")
    if o.dim() != 3 or not o.is_contiguous():
        raise ValueError("o must be contiguous [T, NH, DH]")
    if positions.device.type != "cpu" or positions.dtype != torch.int64:
        raise TypeError("positions must be a CPU torch.int64 tensor")
    if positions.dim() != 1 or not positions.is_contiguous() or int(positions.shape[0]) != int(o.shape[0]):
        raise ValueError("positions must be contiguous [T] and match o.shape[0]")
    if cos_sin_cache.device.type != "cpu" or cos_sin_cache.dtype != torch.float32:
        raise TypeError("cos_sin_cache must be a CPU torch.float32 tensor")
    if cos_sin_cache.dim() != 2 or not cos_sin_cache.is_contiguous():
        raise ValueError("cos_sin_cache must be contiguous [max_position, rope_dim]")
    if int(cos_sin_cache.shape[1]) != weights.rope_dim:
        raise ValueError(
            f"cos_sin_cache second dimension must be rope_dim={weights.rope_dim}, "
            f"got {int(cos_sin_cache.shape[1])}"
        )
    expected_heads = weights.n_groups * weights.heads_per_group
    if int(o.shape[1]) != expected_heads or int(o.shape[2]) != weights.head_dim:
        raise ValueError(
            f"o shape must be [T, {expected_heads}, {weights.head_dim}], got {tuple(o.shape)}"
        )
    if positions.numel():
        minimum = int(positions.min().item())
        maximum = int(positions.max().item())
        if minimum < 0 or maximum >= int(cos_sin_cache.shape[0]):
            raise ValueError(
                f"positions must be in [0, {int(cos_sin_cache.shape[0])}), got [{minimum}, {maximum}]"
            )
    shape = (int(o.shape[0]), weights.n_groups, weights.output_rank)
    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device="cpu")
    if out.device.type != "cpu" or out.dtype != torch.bfloat16:
        raise TypeError("out must be a CPU torch.bfloat16 tensor")
    if tuple(out.shape) != shape or not out.is_contiguous():
        raise ValueError(f"out must be contiguous with shape {shape}, got {tuple(out.shape)}")
    if out.requires_grad:
        raise ValueError("out with requires_grad=True is unsupported")
    if torch._debug_has_internal_overlap(out) != 0:
        raise ValueError("out must not have internal overlap")
    _check_no_storage_alias(out, (o, positions, cos_sin_cache, weights.packed_weight))
    return out


def _inverse_gptj_rope_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0 or o.numel() == 0:
        return o.clone()
    half = rope_dim // 2
    nope_dim = int(o.shape[-1]) - rope_dim
    source = o.float()
    cache = cos_sin_cache.index_select(0, positions)
    cos = cache[:, :half].view(int(o.shape[0]), 1, half)
    sin = cache[:, half:].view(int(o.shape[0]), 1, half)
    rope = source[..., nope_dim:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1).flatten(-2)
    result = source.clone()
    result[..., nope_dim:] = rotated
    return result.to(torch.bfloat16)


def _grouped_woa_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    grouped_weight: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    o_ref = _inverse_gptj_rope_torch(o, positions, cos_sin_cache, rope_dim)
    grouped = o_ref.view(int(o.shape[0]), n_groups, heads_per_group * head_dim)
    return torch.einsum("tgd,grd->tgr", grouped.float(), grouped_weight.float()).to(torch.bfloat16)


def deepseek_v4_inv_rope_grouped_woa_torch_reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    rope_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialized Torch reference matching vLLM's inverse-RoPE + einsum."""
    if o.dim() != 3:
        raise ValueError("o must be 3-D [T, NH, DH]")
    prepared = prepare_deepseek_v4_inv_rope_woa(
        wo_a_weight,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        head_dim=int(o.shape[2]),
        rope_dim=rope_dim,
        backend="torch",
    )
    output = _prepare_output(o, positions, cos_sin_cache, prepared, out)
    if o.shape[0] == 0:
        return output
    result = _grouped_woa_torch(
        o,
        positions,
        cos_sin_cache,
        prepared.packed_weight,
        n_groups=prepared.n_groups,
        heads_per_group=prepared.heads_per_group,
        head_dim=prepared.head_dim,
        rope_dim=prepared.rope_dim,
    )
    output.copy_(result)
    return output


def deepseek_v4_inv_rope_grouped_woa(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: PreparedDeepseekV4InvRopeWoa,
    *,
    out: torch.Tensor | None = None,
    core_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Execute inverse GPT-J RoPE and TP-local grouped WO_A projection."""
    output = _prepare_output(o, positions, cos_sin_cache, weights, out)
    if core_ids is not None:
        resolved_core_ids = tuple(int(core_id) for core_id in core_ids)
        if not resolved_core_ids or min(resolved_core_ids) < 0 or len(set(resolved_core_ids)) != len(resolved_core_ids):
            raise ValueError("core_ids must contain unique non-negative CPU ids")
    else:
        resolved_core_ids = ()
    if o.shape[0] == 0:
        return output
    if weights.backend == "torch":
        result = _grouped_woa_torch(
            o,
            positions,
            cos_sin_cache,
            weights.packed_weight,
            n_groups=weights.n_groups,
            heads_per_group=weights.heads_per_group,
            head_dim=weights.head_dim,
            rope_dim=weights.rope_dim,
        )
        output.copy_(result)
        return output
    if weights.backend != "arm_sve_bf16" or _native_execute is None:
        raise RuntimeError(f"unsupported prepared inverse-RoPE WO_A backend {weights.backend!r}")
    native_core_ids = torch.tensor(resolved_core_ids, dtype=torch.int32) if resolved_core_ids else None
    return _native_execute(
        o,
        positions,
        cos_sin_cache,
        weights.packed_weight,
        weights.n_groups,
        weights.heads_per_group,
        weights.head_dim,
        weights.rope_dim,
        weights.output_rank,
        native_core_ids,
        output,
    )


__all__ = [
    "PreparedDeepseekV4InvRopeWoa",
    "_HAS_DEEPSEEK_V4_INV_ROPE_WOA",
    "deepseek_v4_inv_rope_grouped_woa",
    "deepseek_v4_inv_rope_grouped_woa_torch_reference",
    "prepare_deepseek_v4_inv_rope_woa",
]
