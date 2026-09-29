# -*- coding: utf-8 -*-
"""Unit tests for the refactored MoE GEMM layers.

Task 1: the bottom-layer single-thread GEMM (`fused_moe_test_single_thread_gemm`)
must match a reference fp32 matmul across M values, both MoE stage shapes, and
with/without bias.
"""

from __future__ import annotations

import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE backend requires AArch64",
)


def _single_thread_gemm(A, B, bias=None):
    from fused_cpp import _C

    return _C.fused_moe_test_single_thread_gemm(A, B, bias)


# w13: A[M, H] x W13[2F, H]^T -> [M, 2F];  w2: A[M, F] x W2[H, F]^T -> [M, H]
_SHAPES = {
    "w13": (256, 512),  # (K=H, N=2F) for H=256, F=256
    "w2": (256, 256),  # (K=F, N=H)
}
_M_VALUES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 64, 100, 128]


@pytest.mark.parametrize("stage", sorted(_SHAPES))
@pytest.mark.parametrize("m", _M_VALUES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_single_thread_gemm_matches_reference(stage, m, with_bias):
    torch.manual_seed(0)
    k, n = _SHAPES[stage]
    a = torch.randn(m, k, dtype=torch.bfloat16) * 0.05
    b = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n, dtype=torch.float32) * 0.1 if with_bias else None

    out = _single_thread_gemm(a, b, bias)
    assert out.shape == (m, n)
    assert out.dtype == torch.float32

    ref = a.float() @ b.float().t()
    if bias is not None:
        ref = ref + bias.float()

    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)


def test_single_thread_gemm_unpadded_dims():
    """K and N not multiples of 8 must still produce correct [M, N]."""
    torch.manual_seed(1)
    m, k, n = 6, 20, 12  # neither k nor n is a multiple of 8
    a = torch.randn(m, k, dtype=torch.bfloat16) * 0.05
    b = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    out = _single_thread_gemm(a, b, None)
    ref = a.float() @ b.float().t()
    assert out.shape == (m, n)
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)


# ── Task 2: split-plan property tests ────────────────────────────────────


def _split_plan(stage, m, k, n, group_size):
    from fused_cpp import _C

    return _C.fused_moe_test_split_plan(stage, m, k, n, group_size)


def _assert_full_disjoint_cover(ranges, extent):
    """Nonzero ranges must tile [0, extent) exactly, in order, no overlap."""
    nonzero = [(b, s) for (b, s) in ranges if s > 0]
    nonzero.sort()
    cursor = 0
    for begin, size in nonzero:
        assert begin == cursor, f"gap/overlap: {begin} != {cursor}"
        cursor += size
    assert cursor == extent, f"coverage {cursor} != extent {extent}"


_TEAM_SIZES = [1, 2, 3, 4, 8, 16]
_M_GRID = [1, 6, 7, 8, 9, 16, 17, 20, 63, 64, 100, 256, 2048]
_N_GRID = [8, 16, 24, 64, 512, 1024]  # padded (multiple of 8) as in the MoE path


@pytest.mark.parametrize("stage", ["w13", "w2"])
@pytest.mark.parametrize("group_size", _TEAM_SIZES)
@pytest.mark.parametrize("m", _M_GRID)
def test_m_split_plan_properties(stage, group_size, m):
    n, k = 512, 256
    _sel, m_ranges, _n_ranges = _split_plan(stage, m, k, n, group_size)
    assert len(m_ranges) == group_size

    # Full, disjoint coverage of all M rows.
    _assert_full_disjoint_cover(m_ranges, m)

    # All-but-one thread's rows are multiples of 8; at most one carries the
    # <8 remainder tail.
    non_multiples = [s for (_b, s) in m_ranges if s > 0 and s % 8 != 0]
    assert len(non_multiples) <= 1, f"more than one non-8 M range: {m_ranges}"

    # Minimal idle threads: number of busy threads == min(group_size, blocks|1).
    blocks = m // 8
    expected_busy = 1 if blocks == 0 else min(group_size, blocks)
    busy = sum(1 for (_b, s) in m_ranges if s > 0)
    assert busy == expected_busy, f"busy={busy} expected={expected_busy} {m_ranges}"


@pytest.mark.parametrize("group_size", _TEAM_SIZES)
@pytest.mark.parametrize("n", _N_GRID)
def test_n_split_plan_properties(group_size, n):
    m, k = 64, 256
    _sel, _m_ranges, n_ranges = _split_plan("w13", m, k, n, group_size)
    assert len(n_ranges) == group_size

    _assert_full_disjoint_cover(n_ranges, n)

    # Every N range is a multiple of kKernelTile (block-aligned).
    for _b, s in n_ranges:
        assert s % 8 == 0, f"N range not 8-aligned: {n_ranges}"

    blocks = n // 8
    expected_busy = min(group_size, blocks) if blocks else 1
    busy = sum(1 for (_b, s) in n_ranges if s > 0)
    assert busy == expected_busy


@pytest.mark.parametrize("stage", ["w13", "w2"])
def test_split_plan_group_size_one_is_whole(stage):
    _sel, m_ranges, n_ranges = _split_plan(stage, 100, 256, 512, 1)
    assert m_ranges == [(0, 100)]
    assert n_ranges == [(0, 512)]


def test_split_selector_returns_valid_choice():
    for group_size in _TEAM_SIZES:
        sel, _m, _n = _split_plan("w13", 2048, 4096, 1024, group_size)
        assert sel in ("m", "n")


# ── Task 3: team_gemm cooperative-execution equivalence ──────────────────


def _team_gemm(A, B, group_size, split, bias=None):
    from fused_cpp import _C

    return _C.fused_moe_test_team_gemm(A, B, group_size, split, bias)


@pytest.mark.parametrize("stage", sorted(_SHAPES))
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("split", ["m", "n", "auto"])
@pytest.mark.parametrize("m", [1, 7, 8, 17, 64, 100])
@pytest.mark.parametrize("with_bias", [False, True])
def test_team_gemm_matches_reference(stage, group_size, split, m, with_bias):
    torch.manual_seed(0)
    k, n = _SHAPES[stage]
    a = torch.randn(m, k, dtype=torch.bfloat16) * 0.05
    b = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n, dtype=torch.float32) * 0.1 if with_bias else None

    out = _team_gemm(a, b, group_size, split, bias)
    assert out.shape == (m, n)

    ref = a.float() @ b.float().t()
    if bias is not None:
        ref = ref + bias.float()
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)


@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("split", ["m", "n", "auto"])
def test_team_gemm_matches_single_thread(group_size, split):
    """team_gemm (any split/T) must match the single-thread bottom layer."""
    torch.manual_seed(3)
    m, k, n = 129, 256, 512
    a = torch.randn(m, k, dtype=torch.bfloat16) * 0.05
    b = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    team = _team_gemm(a, b, group_size, split, None)
    solo = _single_thread_gemm(a, b, None)
    torch.testing.assert_close(team, solo, atol=1e-3, rtol=1e-3)


# ── Task 5: hierarchical N-split path routed through team_gemm ────────────


def _build_moe_case(seed=0, E=8, H=128, F=128, T=64, top_k=2):
    torch.manual_seed(seed)
    from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

    inp = torch.randn(T, H, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(E, 2 * F, H, dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, H, F, dtype=torch.bfloat16) * 0.05
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2)
    scores = torch.softmax(torch.randn(T, E).float(), dim=-1)
    topk_w, topk_i = torch.topk(scores, top_k, dim=-1)
    return inp, packed, topk_w.to(torch.float32), topk_i.to(torch.int32)


@pytest.mark.parametrize("num_threads", [2, 4, 8])
def test_hierarchical_matches_default(monkeypatch, num_threads):
    """Hierarchical N-split (one cooperative group of `num_threads`) must match
    the default single-thread-per-expert path."""
    from fused_cpp.moe import fused_moe_bf16_tiled

    inp, packed, topk_w, topk_i = _build_moe_case()

    monkeypatch.delenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", raising=False)
    out_default = fused_moe_bf16_tiled(inp, packed, topk_w, topk_i, num_threads=num_threads, activation="silu")

    # 1 partition, 1 group -> group_size == num_threads (cooperative team GEMM).
    monkeypatch.setenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_CORE_BASES", "0")
    monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION", "1")
    out_hier = fused_moe_bf16_tiled(inp, packed, topk_w, topk_i, num_threads=num_threads, activation="silu")

    torch.testing.assert_close(out_hier, out_default, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_threads", [2, 4])
def test_hierarchical_multi_group(monkeypatch, num_threads):
    """Two partitions x default groups also matches the default path."""
    from fused_cpp.moe import fused_moe_bf16_tiled

    inp, packed, topk_w, topk_i = _build_moe_case(seed=1)

    monkeypatch.delenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", raising=False)
    out_default = fused_moe_bf16_tiled(inp, packed, topk_w, topk_i, num_threads=num_threads, activation="silu")

    # groups == num_threads -> group_size == 1 (degenerate teams, still routed).
    monkeypatch.setenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_CORE_BASES", "0")
    monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION", str(num_threads))
    out_hier = fused_moe_bf16_tiled(inp, packed, topk_w, topk_i, num_threads=num_threads, activation="silu")

    torch.testing.assert_close(out_hier, out_default, atol=2e-2, rtol=2e-2)


# ── Task 6: scheduled / async entry points routed through team_gemm ───────


@pytest.mark.parametrize("num_threads", [2, 4])
def test_async_matches_default(num_threads):
    """Async task-DAG path (one task per active expert on the full team, linear
    dependency chain) must match the default path."""
    from fused_cpp.moe import fused_moe_bf16_tiled, fused_moe_bf16_tiled_async

    inp, packed, topk_w, topk_i = _build_moe_case(seed=2)
    out_default = fused_moe_bf16_tiled(inp, packed, topk_w, topk_i, num_threads=num_threads, activation="silu")

    active = torch.unique(topk_i.reshape(-1)).to(torch.int32).tolist()
    n = len(active)
    deps: list[int] = []
    dep_offsets = [0]
    for i in range(n):
        if i > 0:
            deps.append(i - 1)  # serialize: task i waits for task i-1
        dep_offsets.append(len(deps))

    out_async = fused_moe_bf16_tiled_async(
        inp,
        packed,
        topk_w,
        topk_i,
        torch.tensor(active, dtype=torch.int32),
        torch.zeros(n, dtype=torch.int32),  # task_core_begins
        torch.full((n,), num_threads, dtype=torch.int32),  # task_threads
        torch.tensor(dep_offsets, dtype=torch.int32),
        torch.tensor(deps, dtype=torch.int32),
        num_threads=num_threads,
        activation="silu",
    )
    torch.testing.assert_close(out_async, out_default, atol=2e-2, rtol=2e-2)
