# -*- coding: utf-8 -*-
"""Naive sparse MLA reference implementations."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

__all__ = [
    "flash_mla_sparse_fwd_naive",
    "sparse_mla",
    "sparse_mla_naive",
]


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _check_sparse_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
) -> None:
    if q.dim() != 3:
        raise ValueError(f"q must be 3-D [s_q, h_q, d_qk], got {tuple(q.shape)}")
    if kv.dim() != 3:
        raise ValueError(f"kv must be 3-D [s_kv, h_kv, d_qk], got {tuple(kv.shape)}")
    if indices.dim() != 3:
        raise ValueError(
            f"indices must be 3-D [s_q, h_kv, topk], got {tuple(indices.shape)}"
        )
    if indices.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"indices must use an integer dtype, got {indices.dtype}")
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, kv_d = kv.shape
    if indices.shape[-1] == 0:
        raise ValueError("indices topk dimension must be non-zero")
    if kv_d != d_qk:
        raise ValueError(f"q/kv d_qk mismatch: q={d_qk}, kv={kv_d}")
    if h_kv != 1:
        raise NotImplementedError(
            "sparse_mla_naive currently matches FlashMLA sparse prefill kernels, "
            "which require h_kv == 1"
        )
    if indices.shape[0] != s_q or indices.shape[1] != h_kv:
        raise ValueError(
            "indices shape must be [s_q, h_kv, topk], got "
            f"{tuple(indices.shape)} for s_q={s_q}, h_kv={h_kv}"
        )
    if d_v <= 0 or d_v > d_qk:
        raise ValueError(f"d_v must satisfy 0 < d_v <= d_qk, got d_v={d_v}, d_qk={d_qk}")
    if q.device != kv.device or q.device != indices.device:
        raise ValueError("q, kv, and indices must be on the same device")
    if attn_sink is not None:
        if attn_sink.dim() != 1 or attn_sink.numel() != h_q:
            raise ValueError(
                f"attn_sink must be [h_q], got {tuple(attn_sink.shape)} for h_q={h_q}"
            )
        if attn_sink.device != q.device:
            raise ValueError("attn_sink must be on the same device as q")
    if topk_length is not None:
        if topk_length.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"topk_length must use an integer dtype, got {topk_length.dtype}"
            )
        if topk_length.dim() != 1 or topk_length.numel() != s_q:
            raise ValueError(
                "topk_length must be [s_q], got "
                f"{tuple(topk_length.shape)} for s_q={s_q}"
            )
        if topk_length.device != q.device:
            raise ValueError("topk_length must be on the same device as q")
    if out is not None:
        expected = (s_q, h_q, d_v)
        if tuple(out.shape) != expected:
            raise ValueError(f"out must have shape {expected}, got {tuple(out.shape)}")
        if out.device != q.device:
            raise ValueError("out must be on the same device as q")
        if out.dtype != q.dtype:
            raise ValueError(f"out dtype must match q dtype, got {out.dtype} vs {q.dtype}")
        if out.stride(-1) != 1:
            raise ValueError("out must be contiguous on the last dimension")
    if s_kv == 0:
        raise ValueError("kv must contain at least one row so invalid indices can clamp to 0")


def sparse_mla_naive(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: Optional[float] = None,
    *,
    d_v: Optional[int] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Naive PyTorch implementation of FlashMLA sparse prefill attention.

    The function follows ``flash_mla_sparse_fwd`` prefill semantics:

    - ``q`` has shape ``[s_q, h_q, d_qk]``.
    - ``kv`` has shape ``[s_kv, 1, d_qk]``. The first ``d_v`` channels are V.
    - ``indices`` has shape ``[s_q, 1, topk]``. Entries ``< 0`` or ``>= s_kv``
      are masked out.
    - ``topk_length[i]`` keeps only the leftmost indices for q row ``i``.
    - ``attn_sink`` changes output normalization only; returned ``lse`` and
      ``max_logits`` are computed from sparse attention logits alone.

    Returns ``(output, max_logits, lse)`` with shapes
    ``[s_q, h_q, d_v]``, ``[s_q, h_q]``, and ``[s_q, h_q]``.
    """
    if d_v is None:
        d_v = kv.shape[-1] if kv.shape[-1] < 512 else 512
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    _check_sparse_inputs(q, kv, indices, d_v, attn_sink, topk_length, out)

    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    indices_2d = indices[:, 0, :].to(torch.long).clone()
    if topk_length is not None:
        keep_mask = (
            torch.arange(topk, device=q.device).unsqueeze(0)
            < topk_length.to(torch.long).unsqueeze(1)
        )
        indices_2d = indices_2d.masked_fill(~keep_mask, -1)

    invalid_mask = (indices_2d < 0) | (indices_2d >= s_kv)
    gather_index = indices_2d.masked_fill(invalid_mask, 0).reshape(-1)

    kv0 = kv[:, 0, :]
    gathered_kv = kv0.index_select(0, gather_index).reshape(s_q, topk, d_qk).float()

    scores = torch.matmul(q.float(), gathered_kv.transpose(1, 2))
    scores = scores * float(sm_scale)
    scores = scores.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))

    lse_raw = torch.logsumexp(scores, dim=-1)
    max_logits = scores.max(dim=-1).values

    if attn_sink is None:
        lse_for_output = lse_raw
    else:
        sink = attn_sink.float().reshape(1, h_q).expand(s_q, h_q)
        lse_for_output = torch.logsumexp(torch.stack((lse_raw, sink), dim=0), dim=0)

    lse_for_output = lse_for_output.clone()
    lse_for_output[lse_for_output == float("-inf")] = float("+inf")
    weights = torch.exp(scores - lse_for_output.unsqueeze(-1))
    out_fp32 = torch.matmul(weights, gathered_kv[..., :d_v])
    out_value = out_fp32.to(q.dtype) if q.dtype != torch.float32 else out_fp32

    lse = lse_raw.clone()
    lse[lse == float("-inf")] = float("+inf")

    if out is not None:
        out.copy_(out_value)
        output = out
    else:
        output = out_value
    return output, max_logits, lse


def flash_mla_sparse_fwd_naive(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper matching ``flash_mla_sparse_fwd`` arguments."""
    return sparse_mla_naive(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )


sparse_mla = sparse_mla_naive
