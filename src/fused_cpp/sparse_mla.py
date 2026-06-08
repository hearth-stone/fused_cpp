# -*- coding: utf-8 -*-
"""Naive sparse MLA reference implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    from fused_cpp._C import flash_mla_sparse_fwd as _cpp_flash_mla_sparse_fwd

    _HAS_CPP_SPARSE_MLA = True
except (ImportError, AttributeError):
    _cpp_flash_mla_sparse_fwd = None
    _HAS_CPP_SPARSE_MLA = False

__all__ = [
    "flash_mla_sparse_fwd",
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

_QUERY_BLOCK = 8
_INDEXED_KT = 4
_DENSE_THRESHOLD = 16


@dataclass(frozen=True)
class _DenseSeg:
    start: int
    length: int


@dataclass(frozen=True)
class _IndexedTile:
    idx: tuple[tuple[int, ...], ...]
    valid_mask: int


@dataclass(frozen=True)
class _BlockPlan:
    token0: int
    lq_eff: int
    dense_segments: tuple[_DenseSeg, ...]
    indexed_tiles: tuple[_IndexedTile, ...]


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
            "sparse_mla_naive currently matches vLLM DeepSeek V4 sparse CPU "
            "fallback, which requires h_kv == 1"
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
        if attn_sink.dim() != 1 or attn_sink.numel() < h_q:
            raise ValueError(
                "attn_sink must be 1-D with at least h_q entries, got "
                f"{tuple(attn_sink.shape)} for h_q={h_q}"
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
        raise ValueError("kv must contain at least one row")


def _gather_kv_vllm_cpu(kv: torch.Tensor, valid_indices: torch.Tensor) -> torch.Tensor:
    """Gather KV rows like DeepSeek V4's vLLM CPU fallback.

    vLLM only treats negative indices as padding. Positive out-of-range entries
    are intentionally left to ``narrow`` / ``index_select`` to fail, matching the
    original fallback instead of silently masking them out.
    """
    if valid_indices.numel() == 1:
        start = int(valid_indices[0].item())
        return kv.narrow(0, start, 1)

    breaks = (
        valid_indices[1:] != valid_indices[:-1] + 1
    ).nonzero(as_tuple=False).flatten()
    if breaks.numel() == 0:
        start = int(valid_indices[0].item())
        return kv.narrow(0, start, valid_indices.numel())

    # DeepSeek V4 prefill often forms a few dense runs, e.g. compressed prefix
    # plus SWA window. Preserve vLLM's small-run fast path before falling back
    # to generic indexing.
    num_runs = int(breaks.numel()) + 1
    if num_runs > 4:
        return kv.index_select(0, valid_indices)

    parts: list[torch.Tensor] = []
    run_start = 0
    for break_idx_t in breaks:
        run_end = int(break_idx_t.item()) + 1
        start = int(valid_indices[run_start].item())
        parts.append(kv.narrow(0, start, run_end - run_start))
        run_start = run_end
    start = int(valid_indices[run_start].item())
    parts.append(kv.narrow(0, start, valid_indices.numel() - run_start))
    return torch.cat(parts, dim=0)


def _contiguous_runs(values: list[int]) -> list[tuple[int, int, int]]:
    if not values:
        return []
    runs: list[tuple[int, int, int]] = []
    run_pos = 0
    run_start = values[0]
    run_len = 1
    for pos in range(1, len(values)):
        if values[pos] == values[pos - 1] + 1:
            run_len += 1
            continue
        runs.append((run_pos, run_start, run_len))
        run_pos = pos
        run_start = values[pos]
        run_len = 1
    runs.append((run_pos, run_start, run_len))
    return runs


def _build_block_plan(
    token0: int,
    rows: list[list[int]],
    *,
    dense_threshold: int,
) -> _BlockPlan:
    lq_eff = len(rows)
    consumed = [[False] * len(row) for row in rows]
    dense_segments: list[_DenseSeg] = []

    # First version only uses dense segments for full 8-query blocks. Tail
    # blocks remain indexed so the dense kernel can stay pure 8x8.
    if lq_eff == _QUERY_BLOCK:
        row_runs = [_contiguous_runs(row) for row in rows]
        for pos0, start, length0 in row_runs[0]:
            if length0 < dense_threshold:
                continue
            matches: list[tuple[int, int]] = [(pos0, length0)]
            for runs in row_runs[1:]:
                match = next(
                    ((pos, length) for pos, run_start, length in runs if run_start == start),
                    None,
                )
                if match is None:
                    break
                matches.append(match)
            if len(matches) != _QUERY_BLOCK:
                continue

            dense_len = min(length for _, length in matches)
            if dense_len < dense_threshold:
                continue
            dense_len = (dense_len // _QUERY_BLOCK) * _QUERY_BLOCK
            if dense_len == 0:
                continue

            for row_idx, (pos, _) in enumerate(matches):
                # Avoid overlapping dense segments if repeated runs share a
                # start value. Repeated index positions must keep their count.
                if any(consumed[row_idx][pos : pos + dense_len]):
                    break
            else:
                dense_segments.append(_DenseSeg(start=start, length=dense_len))
                for row_idx, (pos, _) in enumerate(matches):
                    for offset in range(dense_len):
                        consumed[row_idx][pos + offset] = True

    leftovers: list[list[int]] = []
    for row, row_consumed in zip(rows, consumed):
        leftovers.append(
            [value for value, was_consumed in zip(row, row_consumed) if not was_consumed]
        )

    indexed_tiles: list[_IndexedTile] = []
    max_leftover = max((len(row) for row in leftovers), default=0)
    for start_col in range(0, max_leftover, _INDEXED_KT):
        tile_rows: list[tuple[int, ...]] = []
        valid_mask = 0
        for row_idx in range(_QUERY_BLOCK):
            row_values = leftovers[row_idx] if row_idx < lq_eff else []
            cols: list[int] = []
            for col in range(_INDEXED_KT):
                value_pos = start_col + col
                if value_pos < len(row_values):
                    cols.append(row_values[value_pos])
                    valid_mask |= 1 << (row_idx * _INDEXED_KT + col)
                else:
                    cols.append(0)
            tile_rows.append(tuple(cols))
        indexed_tiles.append(
            _IndexedTile(idx=tuple(tile_rows), valid_mask=valid_mask)
        )

    return _BlockPlan(
        token0=token0,
        lq_eff=lq_eff,
        dense_segments=tuple(dense_segments),
        indexed_tiles=tuple(indexed_tiles),
    )


def _build_sparse_mla_plans(
    indices_2d: torch.Tensor,
    *,
    dense_threshold: int = _DENSE_THRESHOLD,
) -> tuple[_BlockPlan, ...]:
    s_q = indices_2d.shape[0]
    plans: list[_BlockPlan] = []
    for token0 in range(0, s_q, _QUERY_BLOCK):
        rows: list[list[int]] = []
        for token_idx in range(token0, min(token0 + _QUERY_BLOCK, s_q)):
            row = indices_2d[token_idx]
            rows.append([int(v) for v in row[row >= 0].tolist()])
        plans.append(
            _build_block_plan(token0, rows, dense_threshold=dense_threshold)
        )
    return tuple(plans)


def _update_online_row(
    scores: torch.Tensor,
    values: torch.Tensor,
    row: int,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    output_acc: torch.Tensor,
    real_max: torch.Tensor,
    real_sum: torch.Tensor,
) -> None:
    row_max = scores.max()

    real_new_max = torch.maximum(real_max[row], row_max)
    real_sum[row] = (
        real_sum[row] * torch.exp(real_max[row] - real_new_max)
        + torch.exp(scores - real_new_max).sum()
    )
    real_max[row] = real_new_max

    new_max = torch.maximum(running_max[row], row_max)
    old_scale = torch.exp(running_max[row] - new_max)
    probs = torch.exp(scores - new_max)
    output_acc[row].mul_(old_scale)
    output_acc[row].add_(torch.matmul(probs, values))
    running_sum[row] = running_sum[row] * old_scale + probs.sum()
    running_max[row] = new_max


def _apply_sink_row(
    sink_score: torch.Tensor,
    row: int,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    output_acc: torch.Tensor,
) -> None:
    if torch.isneginf(sink_score):
        return
    if torch.isposinf(sink_score):
        running_max[row] = sink_score
        running_sum[row] = torch.ones((), device=running_sum.device)
        output_acc[row].zero_()
        return
    new_max = torch.maximum(running_max[row], sink_score)
    old_scale = torch.exp(running_max[row] - new_max)
    sink_scale = torch.exp(sink_score - new_max)
    output_acc[row].mul_(old_scale)
    running_sum[row] = running_sum[row] * old_scale + sink_scale
    running_max[row] = new_max


def _index_select_kv_vllm(kv: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    try:
        return kv.index_select(0, index)
    except IndexError as exc:
        raise RuntimeError(str(exc)) from exc


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
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Naive PyTorch implementation of vLLM DeepSeek V4 sparse attention.

    The output path intentionally follows
    ``vllm.models.deepseek_v4.cpu._cpu_sparse_attention``:

    - ``q`` has shape ``[s_q, h_q, d_qk]``.
    - ``kv`` has shape ``[s_kv, 1, d_qk]``. The first ``d_v`` channels are V.
    - ``indices`` has shape ``[s_q, 1, topk]``. Entries ``< 0`` are padding.
      Positive out-of-range entries are not masked; the gather raises just like
      vLLM's CPU fallback.
    - ``topk_length`` is accepted for FlashMLA signature compatibility but is
      ignored by this vLLM-style reference. Upstream must encode padding as
      ``-1`` in ``indices``.
    - ``attn_sink`` is appended to logits as one extra zero-value key per head.

    Differences from the older FlashMLA-style naive reference in this repo:
    it no longer uses ``topk_length`` to truncate rows, no longer masks
    ``indices >= s_kv``, and gathers contiguous index runs with ``narrow`` /
    ``cat`` before falling back to ``index_select``. By default this matches
    vLLM's helper and returns only ``output``; set ``return_stats=True`` to
    return compatibility ``max_logits`` and sparse ``lse``. When ``d_v`` is
    smaller than ``d_qk``, only the value side is sliced; vLLM's DeepSeek V4 CPU
    fallback uses the full head dimension.

    Returns ``output`` with shape ``[s_q, h_q, d_v]`` by default, or
    ``(output, max_logits, lse)`` when ``return_stats=True``.
    """
    if d_v is None:
        d_v = kv.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    _check_sparse_inputs(q, kv, indices, d_v, attn_sink, topk_length, out)

    s_q, h_q, _ = q.shape
    indices_2d = indices.reshape(s_q, -1).to(torch.long)
    q_fp32 = q.to(torch.float32)
    kv0 = kv.squeeze(1)
    scale = float(sm_scale)

    out_fp32 = torch.zeros(
        (s_q, h_q, d_v),
        device=q.device,
        dtype=torch.float32,
    )
    max_logits = torch.full(
        (s_q, h_q),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    lse = torch.full(
        (s_q, h_q),
        float("+inf"),
        device=q.device,
        dtype=torch.float32,
    )

    sink = attn_sink[:h_q].to(torch.float32) if attn_sink is not None else None
    plans = _build_sparse_mla_plans(indices_2d)

    for plan in plans:
        token0 = plan.token0
        lq_eff = plan.lq_eff
        for head_idx in range(h_q):
            q_block = q_fp32[token0 : token0 + lq_eff, head_idx]
            running_max = torch.full(
                (lq_eff,),
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )
            running_sum = torch.zeros(
                (lq_eff,),
                device=q.device,
                dtype=torch.float32,
            )
            output_acc = torch.zeros(
                (lq_eff, d_v),
                device=q.device,
                dtype=torch.float32,
            )
            real_max = torch.full(
                (lq_eff,),
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )
            real_sum = torch.zeros(
                (lq_eff,),
                device=q.device,
                dtype=torch.float32,
            )

            for seg in plan.dense_segments:
                kv_seg = kv0.narrow(0, seg.start, seg.length).to(torch.float32)
                values = kv_seg[:, :d_v]
                scores_block = torch.matmul(q_block, kv_seg.T) * scale
                for row in range(lq_eff):
                    _update_online_row(
                        scores_block[row],
                        values,
                        row,
                        running_max,
                        running_sum,
                        output_acc,
                        real_max,
                        real_sum,
                    )

            for tile in plan.indexed_tiles:
                for row in range(lq_eff):
                    row_indices: list[int] = []
                    for col in range(_INDEXED_KT):
                        if tile.valid_mask & (1 << (row * _INDEXED_KT + col)):
                            row_indices.append(tile.idx[row][col])
                    if not row_indices:
                        continue
                    index_tensor = torch.tensor(
                        row_indices,
                        device=q.device,
                        dtype=torch.long,
                    )
                    kv_i = _index_select_kv_vllm(kv0, index_tensor).to(torch.float32)
                    scores = torch.matmul(q_block[row], kv_i.T) * scale
                    _update_online_row(
                        scores,
                        kv_i[:, :d_v],
                        row,
                        running_max,
                        running_sum,
                        output_acc,
                        real_max,
                        real_sum,
                    )

            if sink is not None:
                sink_score = sink[head_idx]
                for row in range(lq_eff):
                    _apply_sink_row(
                        sink_score,
                        row,
                        running_max,
                        running_sum,
                        output_acc,
                    )

            for row in range(lq_eff):
                token_idx = token0 + row
                if real_sum[row] > 0:
                    max_logits[token_idx, head_idx] = real_max[row]
                    lse[token_idx, head_idx] = real_max[row] + torch.log(real_sum[row])
                if running_sum[row] > 0:
                    out_fp32[token_idx, head_idx] = output_acc[row] / running_sum[row]

    out_value = out_fp32.to(q.dtype) if q.dtype != torch.float32 else out_fp32

    if out is not None:
        out.copy_(out_value)
        output = out
    else:
        output = out_value
    if not return_stats:
        return output
    return output, max_logits, lse


def flash_mla_sparse_fwd_naive(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: Optional[int] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return_stats=return_stats,
    )


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: Optional[int] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper using the C++ sparse MLA kernel when available."""
    if _HAS_CPP_SPARSE_MLA:
        return _cpp_flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            d_v,
            attn_sink,
            topk_length,
            out,
            return_stats,
        )
    return flash_mla_sparse_fwd_naive(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
        return_stats=return_stats,
    )


sparse_mla = flash_mla_sparse_fwd
