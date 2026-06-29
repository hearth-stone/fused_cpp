# -*- coding: utf-8 -*-
"""DeepSeek V4 attn_gemm_parallel_execute fused GEMM helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

PreparedWeight = Tuple[torch.Tensor, int, int]


@dataclass(frozen=True)
class PreparedDeepSeekV4AttnGemmWeights:
    """Packed weights for the fused attn GEMM variants.

    The selected variant is fixed by which optional weights are present:
    dense runs only ``fused_wqa_wkv``; C128A adds ``compressor_kv_score``;
    C4A adds the two indexer GEMMs.
    """

    fused_wqa_wkv: PreparedWeight
    compressor_kv_score: PreparedWeight | None = None
    indexer_compressor_kv_score: PreparedWeight | None = None
    indexer_weights_proj: PreparedWeight | None = None

    @property
    def variant(self) -> str:
        has_compressor = self.compressor_kv_score is not None
        has_indexer = (
            self.indexer_compressor_kv_score is not None
            or self.indexer_weights_proj is not None
        )
        if not has_compressor and has_indexer:
            raise ValueError("indexer GEMM weights require compressor_kv_score")
        if (
            self.indexer_compressor_kv_score is None
            and self.indexer_weights_proj is not None
        ) or (
            self.indexer_compressor_kv_score is not None
            and self.indexer_weights_proj is None
        ):
            raise ValueError(
                "indexer_compressor_kv_score and indexer_weights_proj "
                "must be provided together"
            )
        if not has_compressor:
            return "dense"
        if not has_indexer:
            return "c128a"
        return "c4a"


try:
    from fused_cpp._C import (
        fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused
        as _fused_impl,
    )
    from fused_cpp._C import (
        fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt
        as _fused_mt_impl,
    )
    from fused_cpp._C import (
        fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare
        as _prepare_impl,
    )
    from fused_cpp._C import (
        fused_wqa_wkv_compressor_kv_score_fused as _c128a_impl,
    )
    from fused_cpp._C import (
        fused_wqa_wkv_compressor_kv_score_fused_mt as _c128a_mt_impl,
    )
    from fused_cpp._C import fused_wqa_wkv_fused as _dense_impl
    from fused_cpp._C import fused_wqa_wkv_fused_mt as _dense_mt_impl

    _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED = True
except (ImportError, AttributeError):
    _prepare_impl = None
    _fused_impl = None
    _fused_mt_impl = None
    _dense_impl = None
    _dense_mt_impl = None
    _c128a_impl = None
    _c128a_mt_impl = None
    _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED = False


def fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare(
    weight: torch.Tensor,
) -> PreparedWeight:
    """Pack a bf16 [K, N] weight for a fused attn GEMM variant."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    return _prepare_impl(weight)


def fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare_weights(
    fused_wqa_wkv: torch.Tensor,
    compressor_kv_score: torch.Tensor | None = None,
    indexer_compressor_kv_score: torch.Tensor | None = None,
    indexer_weights_proj: torch.Tensor | None = None,
) -> PreparedDeepSeekV4AttnGemmWeights:
    """Pack present bf16 [K, N] weights for reuse across fused calls."""
    prepare_weight = (
        fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare
    )
    has_indexer = (
        indexer_compressor_kv_score is not None
        or indexer_weights_proj is not None
    )
    if compressor_kv_score is None and has_indexer:
        raise ValueError("indexer GEMM weights require compressor_kv_score")
    if (
        indexer_compressor_kv_score is None
        and indexer_weights_proj is not None
    ) or (
        indexer_compressor_kv_score is not None
        and indexer_weights_proj is None
    ):
        raise ValueError(
            "indexer_compressor_kv_score and indexer_weights_proj "
            "must be provided together"
        )
    return PreparedDeepSeekV4AttnGemmWeights(
        fused_wqa_wkv=prepare_weight(fused_wqa_wkv),
        compressor_kv_score=(
            prepare_weight(compressor_kv_score)
            if compressor_kv_score is not None
            else None
        ),
        indexer_compressor_kv_score=(
            prepare_weight(indexer_compressor_kv_score)
            if indexer_compressor_kv_score is not None
            else None
        ),
        indexer_weights_proj=(
            prepare_weight(indexer_weights_proj)
            if indexer_weights_proj is not None
            else None
        ),
    )


def fused_wqa_wkv_fused(
    hidden_states: torch.Tensor,
    fused_wqa_wkv: PreparedWeight,
) -> torch.Tensor:
    """Run the dense fused path and return qr_kv."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    return _dense_impl(
        hidden_states,
        fused_wqa_wkv[0],
        fused_wqa_wkv[1],
        fused_wqa_wkv[2],
    )


def fused_wqa_wkv_fused_mt(
    hidden_states: torch.Tensor,
    fused_wqa_wkv: PreparedWeight,
    core_ids: Sequence[int],
) -> torch.Tensor:
    """Run the dense fused path with OpenMP row partitioning."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    normalized_core_ids = _normalize_core_ids(core_ids)
    if not normalized_core_ids:
        return fused_wqa_wkv_fused(hidden_states, fused_wqa_wkv)
    return _dense_mt_impl(
        hidden_states,
        fused_wqa_wkv[0],
        fused_wqa_wkv[1],
        fused_wqa_wkv[2],
        normalized_core_ids,
    )


def fused_wqa_wkv_compressor_kv_score_fused(
    hidden_states: torch.Tensor,
    fused_wqa_wkv: PreparedWeight,
    compressor_kv_score: PreparedWeight,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the C128A fused path and return qr_kv, kv_score."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    return _c128a_impl(
        hidden_states,
        fused_wqa_wkv[0],
        fused_wqa_wkv[1],
        fused_wqa_wkv[2],
        compressor_kv_score[0],
        compressor_kv_score[1],
        compressor_kv_score[2],
    )


def fused_wqa_wkv_compressor_kv_score_fused_mt(
    hidden_states: torch.Tensor,
    fused_wqa_wkv: PreparedWeight,
    compressor_kv_score: PreparedWeight,
    core_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the C128A fused path with OpenMP row partitioning."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    normalized_core_ids = _normalize_core_ids(core_ids)
    if not normalized_core_ids:
        return fused_wqa_wkv_compressor_kv_score_fused(
            hidden_states,
            fused_wqa_wkv,
            compressor_kv_score,
        )
    return _c128a_mt_impl(
        hidden_states,
        fused_wqa_wkv[0],
        fused_wqa_wkv[1],
        fused_wqa_wkv[2],
        compressor_kv_score[0],
        compressor_kv_score[1],
        compressor_kv_score[2],
        normalized_core_ids,
    )


def fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
    hidden_states: torch.Tensor,
    fused_wqa_wkv: PreparedWeight,
    compressor_kv_score: PreparedWeight,
    indexer_compressor_kv_score: PreparedWeight,
    indexer_weights_proj: PreparedWeight,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the serial fused path and return qr_kv, kv_score, indexer_kv_score, indexer_weights."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    return _fused_impl(
        hidden_states,
        fused_wqa_wkv[0],
        fused_wqa_wkv[1],
        fused_wqa_wkv[2],
        compressor_kv_score[0],
        compressor_kv_score[1],
        compressor_kv_score[2],
        indexer_compressor_kv_score[0],
        indexer_compressor_kv_score[1],
        indexer_compressor_kv_score[2],
        indexer_weights_proj[0],
        indexer_weights_proj[1],
        indexer_weights_proj[2],
    )


def _normalize_core_ids(core_ids: Optional[Sequence[int]]) -> list[int]:
    if core_ids is None:
        return []
    normalized = [int(core_id) for core_id in core_ids]
    if any(core_id < 0 for core_id in normalized):
        raise ValueError(f"core_ids must be non-negative, got {normalized}")
    return normalized


def fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt(
    hidden_states: torch.Tensor,
    fused_wqa_wkv: PreparedWeight,
    compressor_kv_score: PreparedWeight,
    indexer_compressor_kv_score: PreparedWeight,
    indexer_weights_proj: PreparedWeight,
    core_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused path with OpenMP row partitioning and per-thread CPU affinity."""
    if not _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED:
        raise RuntimeError("DeepSeek V4 attn GEMM fused kernel is unavailable")
    normalized_core_ids = _normalize_core_ids(core_ids)
    if not normalized_core_ids:
        return fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
            hidden_states,
            fused_wqa_wkv,
            compressor_kv_score,
            indexer_compressor_kv_score,
            indexer_weights_proj,
        )
    return _fused_mt_impl(
        hidden_states,
        fused_wqa_wkv[0],
        fused_wqa_wkv[1],
        fused_wqa_wkv[2],
        compressor_kv_score[0],
        compressor_kv_score[1],
        compressor_kv_score[2],
        indexer_compressor_kv_score[0],
        indexer_compressor_kv_score[1],
        indexer_compressor_kv_score[2],
        indexer_weights_proj[0],
        indexer_weights_proj[1],
        indexer_weights_proj[2],
        normalized_core_ids,
    )


def fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepacked(
    hidden_states: torch.Tensor,
    weights: PreparedDeepSeekV4AttnGemmWeights,
    cores: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Run the fused path with pre-packed weights."""
    normalized_core_ids = _normalize_core_ids(cores)
    variant = weights.variant
    if variant == "dense":
        if normalized_core_ids:
            qr_kv = fused_wqa_wkv_fused_mt(
                hidden_states,
                weights.fused_wqa_wkv,
                normalized_core_ids,
            )
        else:
            qr_kv = fused_wqa_wkv_fused(hidden_states, weights.fused_wqa_wkv)
        return qr_kv, None, None, None
    if variant == "c128a":
        assert weights.compressor_kv_score is not None
        if normalized_core_ids:
            qr_kv, kv_score = fused_wqa_wkv_compressor_kv_score_fused_mt(
                hidden_states,
                weights.fused_wqa_wkv,
                weights.compressor_kv_score,
                normalized_core_ids,
            )
        else:
            qr_kv, kv_score = fused_wqa_wkv_compressor_kv_score_fused(
                hidden_states,
                weights.fused_wqa_wkv,
                weights.compressor_kv_score,
            )
        return qr_kv, kv_score, None, None

    assert weights.compressor_kv_score is not None
    assert weights.indexer_compressor_kv_score is not None
    assert weights.indexer_weights_proj is not None
    if normalized_core_ids:
        return fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt(
            hidden_states,
            weights.fused_wqa_wkv,
            weights.compressor_kv_score,
            weights.indexer_compressor_kv_score,
            weights.indexer_weights_proj,
            normalized_core_ids,
        )
    return fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
        hidden_states,
        weights.fused_wqa_wkv,
        weights.compressor_kv_score,
        weights.indexer_compressor_kv_score,
        weights.indexer_weights_proj,
    )


prepare_deepseek_v4_attn_gemm_weights = (
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare_weights
)
deepseek_v4_attn_gemm_fused_prepacked = (
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepacked
)


__all__ = [
    "deepseek_v4_attn_gemm_fused_prepacked",
    "prepare_deepseek_v4_attn_gemm_weights",
    "PreparedDeepSeekV4AttnGemmWeights",
    "PreparedWeight",
    "_HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED",
    "fused_wqa_wkv_fused",
    "fused_wqa_wkv_fused_mt",
    "fused_wqa_wkv_compressor_kv_score_fused",
    "fused_wqa_wkv_compressor_kv_score_fused_mt",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare_weights",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepacked",
]
