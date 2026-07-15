# -*- coding: utf-8 -*-
"""Multi-query attention helpers."""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "multi_query_attention",
    "multi_query_attention_torch",
]


try:
    from fused_cpp._C import multi_query_attention as _cpp_mqa

    _HAS_CPP_MQA = True
except ImportError:
    _cpp_mqa = None
    _HAS_CPP_MQA = False


def _normalize_kv(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:
        return tensor[:, 0]
    raise ValueError(f"{name} must have shape [B, S, D] or [B, 1, S, D], got {tuple(tensor.shape)}")


def _normalize_attn_mask(
    attn_mask: torch.Tensor,
    B: int,
    N: int,
    L: int,
    S: int,
) -> torch.Tensor:
    if attn_mask.dim() == 2:
        if attn_mask.shape != (L, S):
            raise ValueError(f"attn_mask [L, S] shape mismatch: got {tuple(attn_mask.shape)}")
        return attn_mask
    if attn_mask.dim() == 3:
        if attn_mask.shape != (B, L, S):
            raise ValueError(f"attn_mask [B, L, S] shape mismatch: got {tuple(attn_mask.shape)}")
        return attn_mask[:, None, :, :]
    if attn_mask.dim() == 4:
        if attn_mask.shape != (B, 1, L, S) and attn_mask.shape != (B, N, L, S):
            raise ValueError(f"attn_mask [B, N, L, S] shape mismatch: got {tuple(attn_mask.shape)}")
        return attn_mask
    raise ValueError(
        f"attn_mask must be [L, S], [B, L, S], [B, 1, L, S], or [B, N, L, S], got {tuple(attn_mask.shape)}"
    )


def multi_query_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure PyTorch MQA reference with lower-right causal semantics.

    Shapes:
    - ``query``: ``[B, N, L, E]``
    - ``key``: ``[B, S, E]`` or ``[B, 1, S, E]``
    - ``value``: ``[B, S, Ev]`` or ``[B, 1, S, Ev]``
    - output: ``[B, N, L, Ev]``
    """
    del dropout_p
    if query.dim() != 4:
        raise ValueError(f"query must be [B, N, L, E], got {tuple(query.shape)}")

    orig_dtype = query.dtype
    key = _normalize_kv(key, "key")
    value = _normalize_kv(value, "value")

    B, N, L, E = query.shape
    if key.shape[0] != B or key.shape[2] != E:
        raise ValueError(f"key shape mismatch: expected [B, S, {E}], got {tuple(key.shape)}")
    S = key.shape[1]
    if value.shape[0] != B or value.shape[1] != S:
        raise ValueError(f"value shape mismatch: expected [B, S, Ev], got {tuple(value.shape)}")
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    scores = torch.einsum("bnle,bse->bnls", query.float(), key.float())
    scores = scores * float(scale)

    if attn_mask is not None:
        attn_mask = _normalize_attn_mask(attn_mask, B, N, L, S)
        scores = scores + attn_mask.float()

    if is_causal:
        offset = S - L
        l_idx = torch.arange(L, device=query.device).unsqueeze(-1)
        s_idx = torch.arange(S, device=query.device).unsqueeze(0)
        causal_mask = s_idx > (l_idx + offset)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.einsum("bnls,bsv->bnlv", weights, value.float())
    return out.to(orig_dtype) if orig_dtype != torch.float32 else out


def multi_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Dense MQA attention, using the C++ kernel when available."""
    if _HAS_CPP_MQA:
        return _cpp_mqa(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
        )
    return multi_query_attention_torch(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
