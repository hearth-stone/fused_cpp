"""Torch reference implementation of DeepSeek V4 Multi-Head Hyper-Connections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

try:
    from fused_cpp._C import (  # type: ignore[attr-defined, import-untyped]
        deepseek_v4_mhc_sve_control_postprocess as _native_sve_control_postprocess,
        deepseek_v4_mhc_sve_pre_apply_rmsnorm as _native_sve_pre_apply_rmsnorm,
        deepseek_v4_mhc_sve_pre_rmsnorm as _native_sve_pre_rmsnorm,
        deepseek_v4_mhc_sve_post as _native_sve_post,
        deepseek_v4_mhc_sve_post_head_rmsnorm as _native_sve_post_head_rmsnorm,
        deepseek_v4_mhc_sve_projection as _native_sve_projection,
        deepseek_v4_mhc_sve_projection_available as _native_sve_projection_available,
        deepseek_v4_mhc_sve_projection_control as _native_sve_projection_control,
    )

    _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION = bool(_native_sve_projection_available())
except (ImportError, AttributeError):
    _native_sve_control_postprocess = None
    _native_sve_pre_apply_rmsnorm = None
    _native_sve_pre_rmsnorm = None
    _native_sve_post = None
    _native_sve_post_head_rmsnorm = None
    _native_sve_projection = None
    _native_sve_projection_available = None
    _native_sve_projection_control = None
    _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION = False


MHCWeightKind = Literal["pre", "head"]


@dataclass(frozen=True)
class PreparedDeepSeekV4MHCWeight:
    """Owned FP32 projection weight prepared for one mHC reference operation."""

    packed: torch.Tensor
    kind: MHCWeightKind
    backend: Literal["fp32"]
    c: int
    h: int
    n: int


def _infer_geometry(fn: torch.Tensor, kind: MHCWeightKind) -> tuple[int, int, int]:
    n, k = (int(value) for value in fn.shape)
    if kind == "head":
        c = n
    else:
        c = math.isqrt(n + 1) - 1
        if c * c + 2 * c != n:
            raise ValueError(f"pre mHC weight N must equal C*C+2*C, got N={n}")
    if c <= 0 or k % c != 0:
        raise ValueError(f"mHC weight K must be divisible by positive C={c}, got K={k}")
    return c, k // c, n


def prepare_mhc_weight(
    fn: torch.Tensor,
    *,
    kind: MHCWeightKind,
    backend: str = "fp32",
) -> PreparedDeepSeekV4MHCWeight:
    """Own one contiguous CPU FP32 ``[N, C*H]`` mHC projection weight."""
    if kind not in {"pre", "head"}:
        raise ValueError(f"mHC weight kind must be 'pre' or 'head', got {kind!r}")
    if backend != "fp32":
        raise ValueError(f"mHC baseline backend must be 'fp32', got {backend!r}")
    if fn.device.type != "cpu" or fn.dtype != torch.float32 or fn.dim() != 2:
        raise TypeError("mHC weight must be a CPU FP32 [N, C*H] tensor")
    if not fn.is_contiguous():
        raise ValueError("mHC weight must be contiguous")
    c, h, n = _infer_geometry(fn, kind)
    return PreparedDeepSeekV4MHCWeight(
        packed=fn.detach().t().contiguous(),
        kind=kind,
        backend="fp32",
        c=c,
        h=h,
        n=n,
    )


def _check_tensor(tensor: torch.Tensor, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if tensor.device.type != "cpu" or tensor.dtype != dtype:
        raise TypeError(f"{name} must be a CPU {dtype} tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_common_options(*, sinkhorn_repeat: int | None = None, num_threads: int) -> None:
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    if sinkhorn_repeat is not None and sinkhorn_repeat < 1:
        raise ValueError("sinkhorn_repeat must be at least one")


def _rmsnorm_from_bf16(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = input.float()
    scale = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * scale * weight.float()).to(torch.bfloat16)


def _sinkhorn(logits: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    result = torch.softmax(logits, dim=-1) + eps
    result = result / (result.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        result = result / (result.sum(dim=-1, keepdim=True) + eps)
        result = result / (result.sum(dim=-2, keepdim=True) + eps)
    return result


def _validate_pre_inputs(
    residual: torch.Tensor,
    prepared: PreparedDeepSeekV4MHCWeight,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[int, int, int]:
    if not isinstance(prepared, PreparedDeepSeekV4MHCWeight) or prepared.kind != "pre":
        raise TypeError("prepared_next_fn must be a prepared pre mHC weight")
    if residual.dim() != 3:
        raise ValueError(f"residual must have shape [T,C,H], got {tuple(residual.shape)}")
    t = int(residual.shape[0])
    c, h = prepared.c, prepared.h
    _check_tensor(residual, "residual", (t, c, h), torch.bfloat16)
    _check_tensor(hc_scale, "hc_scale", (3,), torch.float32)
    _check_tensor(hc_base, "hc_base", (prepared.n,), torch.float32)
    _check_tensor(norm_weight, "norm_weight", (h,), torch.bfloat16)
    return t, c, h


def _mhc_pre_rmsnorm_impl(
    residual: torch.Tensor,
    prepared_next_fn: PreparedDeepSeekV4MHCWeight,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, c, h = _validate_pre_inputs(residual, prepared_next_fn, hc_scale, hc_base, norm_weight)
    residual_fp32 = residual.reshape(t, c * h).float()
    mixes = residual_fp32 @ prepared_next_fn.packed
    sqrsum = residual_fp32.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (c * h) + rms_eps)

    return _mhc_pre_postprocess(
        residual,
        mixes,
        hc_scale,
        hc_base,
        norm_weight,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
    )


def _mhc_pre_postprocess(
    residual: torch.Tensor,
    normalized_mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, c, _ = residual.shape
    mixes = normalized_mixes

    pre_mix = torch.sigmoid(mixes[:, :c] * hc_scale[0] + hc_base[:c]) + hc_pre_eps
    post_mix = (torch.sigmoid(mixes[:, c : 2 * c] * hc_scale[1] + hc_base[c : 2 * c]) * hc_post_mult_value).reshape(
        t, c, 1
    )
    comb_logits = mixes[:, 2 * c :].reshape(t, c, c) * hc_scale[2] + hc_base[2 * c :].reshape(1, c, c)
    comb_mix = _sinkhorn(comb_logits, sinkhorn_repeat, hc_sinkhorn_eps)

    raw_input = (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    normed_input = _rmsnorm_from_bf16(raw_input, norm_weight, norm_eps)
    return post_mix, comb_mix, normed_input


def _mhc_pre_rmsnorm_sve_impl(
    residual: torch.Tensor,
    prepared_next_fn: PreparedDeepSeekV4MHCWeight,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    num_threads: int,
    b_window_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, c, h = _validate_pre_inputs(residual, prepared_next_fn, hc_scale, hc_base, norm_weight)
    if (
        not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION
        or _native_sve_projection is None
        or _native_sve_control_postprocess is None
        or _native_sve_pre_apply_rmsnorm is None
        or _native_sve_pre_rmsnorm is None
        or _native_sve_post is None
        or _native_sve_projection_control is None
    ):
        raise RuntimeError("DeepSeek V4 mHC SVE projection is unavailable in this build")
    if prepared_next_fn.n != 24:
        raise ValueError(f"SVE mHC projection requires N=24, got N={prepared_next_fn.n}")
    if b_window_bytes <= 0:
        raise ValueError("b_window_bytes must be positive")
    post_mix, comb_mix, normed_input, _, _ = _native_sve_pre_rmsnorm(
        residual,
        prepared_next_fn.packed,
        hc_scale,
        hc_base,
        norm_weight,
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_post_mult_value),
        float(hc_sinkhorn_eps),
        int(sinkhorn_repeat),
        float(norm_eps),
        int(num_threads),
        int(b_window_bytes),
    )
    return post_mix, comb_mix, normed_input


@torch.inference_mode()
def mhc_pre_rmsnorm_torch_baseline(
    residual: torch.Tensor,
    prepared_next_fn: PreparedDeepSeekV4MHCWeight,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-6,
    num_threads: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run strict CPU-Torch mHC pre, BF16 boundary, and RMSNorm semantics."""
    _check_common_options(sinkhorn_repeat=sinkhorn_repeat, num_threads=num_threads)
    post_mix, comb_mix, normed_input = _mhc_pre_rmsnorm_impl(
        residual,
        prepared_next_fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
    )
    return post_mix, comb_mix, normed_input


@torch.inference_mode()
def mhc_pre_rmsnorm_sve_candidate(
    residual: torch.Tensor,
    prepared_next_fn: PreparedDeepSeekV4MHCWeight,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-6,
    num_threads: int = 0,
    b_window_bytes: int = 1 << 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the SVE M8/M4 narrow projection and Torch mHC postprocessing."""
    _check_common_options(sinkhorn_repeat=sinkhorn_repeat, num_threads=num_threads)
    return _mhc_pre_rmsnorm_sve_impl(
        residual,
        prepared_next_fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        num_threads=num_threads,
        b_window_bytes=b_window_bytes,
    )


def _mhc_post(
    layer_output: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
) -> torch.Tensor:
    if residual_prev.dim() != 3:
        raise ValueError(f"residual_prev must have shape [T,C,H], got {tuple(residual_prev.shape)}")
    t, c, h = (int(value) for value in residual_prev.shape)
    _check_tensor(residual_prev, "residual_prev", (t, c, h), torch.bfloat16)
    _check_tensor(layer_output, "layer_output", (t, h), torch.bfloat16)
    _check_tensor(post_mix_prev, "post_mix_prev", (t, c, 1), torch.float32)
    _check_tensor(comb_mix_prev, "comb_mix_prev", (t, c, c), torch.float32)
    mixed = torch.bmm(comb_mix_prev.transpose(1, 2), residual_prev.float())
    injected = post_mix_prev * layer_output.float().unsqueeze(1)
    return (mixed + injected).to(torch.bfloat16)


@torch.inference_mode()
def mhc_post_pre_rmsnorm_torch_baseline(
    layer_output: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    prepared_next_fn: PreparedDeepSeekV4MHCWeight,
    next_hc_scale: torch.Tensor,
    next_hc_base: torch.Tensor,
    next_norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-6,
    num_threads: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run strict BF16 post followed by the next Torch mHC pre and RMSNorm."""
    _check_common_options(sinkhorn_repeat=sinkhorn_repeat, num_threads=num_threads)
    _validate_pre_inputs(residual_prev, prepared_next_fn, next_hc_scale, next_hc_base, next_norm_weight)
    residual_cur = _mhc_post(layer_output, residual_prev, post_mix_prev, comb_mix_prev)
    post_mix_cur, comb_mix_cur, normed_input_cur = mhc_pre_rmsnorm_torch_baseline(
        residual_cur,
        prepared_next_fn,
        next_hc_scale,
        next_hc_base,
        next_norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        num_threads=num_threads,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, normed_input_cur


@torch.inference_mode()
def mhc_post_pre_rmsnorm_sve_candidate(
    layer_output: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    prepared_next_fn: PreparedDeepSeekV4MHCWeight,
    next_hc_scale: torch.Tensor,
    next_hc_base: torch.Tensor,
    next_norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-6,
    num_threads: int = 0,
    b_window_bytes: int = 1 << 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused SVE post followed by the SVE narrow projection candidate."""
    _check_common_options(sinkhorn_repeat=sinkhorn_repeat, num_threads=num_threads)
    if not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION or _native_sve_post is None:
        raise RuntimeError("DeepSeek V4 mHC SVE post-pre is unavailable in this build")
    _validate_pre_inputs(residual_prev, prepared_next_fn, next_hc_scale, next_hc_base, next_norm_weight)
    residual_cur = _native_sve_post(layer_output, residual_prev, post_mix_prev, comb_mix_prev, int(num_threads))
    post_mix_cur, comb_mix_cur, normed_input_cur = _mhc_pre_rmsnorm_sve_impl(
        residual_cur,
        prepared_next_fn,
        next_hc_scale,
        next_hc_base,
        next_norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        num_threads=num_threads,
        b_window_bytes=b_window_bytes,
    )
    return residual_cur, post_mix_cur, comb_mix_cur, normed_input_cur


@torch.inference_mode()
def mhc_post_hc_head_rmsnorm_torch_baseline(
    layer_output: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    prepared_head_fn: PreparedDeepSeekV4MHCWeight,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    final_norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_threads: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run final strict BF16 post, HC head, and final RMSNorm."""
    _check_common_options(num_threads=num_threads)
    if not isinstance(prepared_head_fn, PreparedDeepSeekV4MHCWeight) or prepared_head_fn.kind != "head":
        raise TypeError("prepared_head_fn must be a prepared head mHC weight")
    if residual_prev.dim() != 3:
        raise ValueError(f"residual_prev must have shape [T,C,H], got {tuple(residual_prev.shape)}")
    t, c, h = (int(value) for value in residual_prev.shape)
    if (c, h) != (prepared_head_fn.c, prepared_head_fn.h):
        raise ValueError(
            f"final residual geometry must be C={prepared_head_fn.c}, H={prepared_head_fn.h}, got C={c}, H={h}"
        )
    _check_tensor(head_scale, "head_scale", (1,), torch.float32)
    _check_tensor(head_base, "head_base", (c,), torch.float32)
    _check_tensor(final_norm_weight, "final_norm_weight", (h,), torch.bfloat16)

    final_residual = _mhc_post(layer_output, residual_prev, post_mix_prev, comb_mix_prev)
    residual_fp32 = final_residual.reshape(t, c * h).float()
    mixes = residual_fp32 @ prepared_head_fn.packed
    sqrsum = residual_fp32.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (c * h) + rms_eps)
    pre_mix = torch.sigmoid(mixes * head_scale[0] + head_base) + hc_eps
    head_output = (pre_mix.unsqueeze(-1) * final_residual.float()).sum(dim=1).to(torch.bfloat16)
    hidden_states = _rmsnorm_from_bf16(head_output, final_norm_weight, norm_eps)
    return hidden_states, final_residual


@torch.inference_mode()
def mhc_post_hc_head_rmsnorm_sve_candidate(
    layer_output: torch.Tensor,
    residual_prev: torch.Tensor,
    post_mix_prev: torch.Tensor,
    comb_mix_prev: torch.Tensor,
    prepared_head_fn: PreparedDeepSeekV4MHCWeight,
    head_scale: torch.Tensor,
    head_base: torch.Tensor,
    final_norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_threads: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the SVE post and fixed NEON M12xN4 final HC head candidate."""
    _check_common_options(num_threads=num_threads)
    if not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION or _native_sve_post_head_rmsnorm is None:
        raise RuntimeError("DeepSeek V4 mHC SVE post-head is unavailable in this build")
    if not isinstance(prepared_head_fn, PreparedDeepSeekV4MHCWeight) or prepared_head_fn.kind != "head":
        raise TypeError("prepared_head_fn must be a prepared head mHC weight")
    if residual_prev.dim() != 3:
        raise ValueError(f"residual_prev must have shape [T,C,H], got {tuple(residual_prev.shape)}")
    t, c, h = (int(value) for value in residual_prev.shape)
    if c != 4:
        raise ValueError(f"SVE mHC post-head requires C=4, got C={c}")
    if (c, h) != (prepared_head_fn.c, prepared_head_fn.h):
        raise ValueError(
            f"final residual geometry must be C={prepared_head_fn.c}, H={prepared_head_fn.h}, got C={c}, H={h}"
        )
    _check_tensor(residual_prev, "residual_prev", (t, c, h), torch.bfloat16)
    _check_tensor(layer_output, "layer_output", (t, h), torch.bfloat16)
    _check_tensor(post_mix_prev, "post_mix_prev", (t, c, 1), torch.float32)
    _check_tensor(comb_mix_prev, "comb_mix_prev", (t, c, c), torch.float32)
    _check_tensor(head_scale, "head_scale", (1,), torch.float32)
    _check_tensor(head_base, "head_base", (c,), torch.float32)
    _check_tensor(final_norm_weight, "final_norm_weight", (h,), torch.bfloat16)
    return _native_sve_post_head_rmsnorm(
        layer_output,
        residual_prev,
        post_mix_prev,
        comb_mix_prev,
        prepared_head_fn.packed,
        head_scale,
        head_base,
        final_norm_weight,
        float(rms_eps),
        float(hc_eps),
        float(norm_eps),
        int(num_threads),
    )


mhc_pre_rmsnorm = mhc_pre_rmsnorm_torch_baseline
mhc_post_pre_rmsnorm = mhc_post_pre_rmsnorm_torch_baseline
mhc_post_hc_head_rmsnorm = mhc_post_hc_head_rmsnorm_torch_baseline


__all__ = [
    "PreparedDeepSeekV4MHCWeight",
    "mhc_post_hc_head_rmsnorm",
    "mhc_post_hc_head_rmsnorm_sve_candidate",
    "mhc_post_hc_head_rmsnorm_torch_baseline",
    "mhc_post_pre_rmsnorm",
    "mhc_post_pre_rmsnorm_sve_candidate",
    "mhc_post_pre_rmsnorm_torch_baseline",
    "mhc_pre_rmsnorm",
    "mhc_pre_rmsnorm_sve_candidate",
    "mhc_pre_rmsnorm_torch_baseline",
    "prepare_mhc_weight",
]
