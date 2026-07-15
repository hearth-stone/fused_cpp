# -*- coding: utf-8 -*-
"""N-split single-expert fused SiLU path.

Task 1: the cooperative N-split fused w13 dispatch (team_fused_w13_silu) must
assemble an intermediate that is bit-for-bit identical to the whole-slice
single-thread fused kernel (same kernel, disjoint column slices).
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


def _single(A, w13, degree):
    from fused_cpp import _C

    return _C.fused_moe_test_fused_w13_silu(A, w13, degree)


def _team(A, w13, group_size, degree):
    from fused_cpp import _C

    return _C.fused_moe_test_team_fused_w13_silu(A, w13, group_size, degree)


@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
@pytest.mark.parametrize("h,f", [(256, 64), (128, 512), (4096, 512)])
@pytest.mark.parametrize("m", [8, 64, 37])  # includes a tail (37 = 32 + 4 + 1)
@pytest.mark.parametrize("degree", [4, 5, 6])
def test_team_fused_w13_matches_single_thread(group_size, h, f, m, degree):
    torch.manual_seed(group_size + h + f + m + degree)
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.05

    ref = _single(a, w13, degree)
    out = _team(a, w13, group_size, degree)

    assert out.shape == ref.shape == (m, f)
    assert out.dtype == torch.bfloat16
    # Same kernel, disjoint column slices -> must be bit-identical.
    assert torch.equal(out, ref), (
        f"mismatch g={group_size} h={h} f={f} m={m} d={degree}; "
        f"max|diff|={(out.float() - ref.float()).abs().max().item()}"
    )


# ── Task 3: end-to-end threaded equivalence (hierarchical N-split enabled) ──
import contextlib  # noqa: E402
import os  # noqa: E402

from fused_cpp.moe import fused_moe_bf16_tiled  # noqa: E402
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights  # noqa: E402

_NSPLIT_ENV_KEYS = (
    "FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT",
    "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
    "FUSED_CPP_MOE_N_SPLIT_CORE_BASES",
)


@contextlib.contextmanager
def _nsplit_env(groups_per_partition=1, core_bases="0"):
    """Enable the hierarchical N-split path for the duration of the block."""
    saved = {k: os.environ.get(k) for k in _NSPLIT_ENV_KEYS}
    os.environ["FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT"] = "1"
    os.environ["FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION"] = str(groups_per_partition)
    os.environ["FUSED_CPP_MOE_N_SPLIT_CORE_BASES"] = core_bases
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bf16(*shape, scale=0.2):
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _moe_case(num_tokens, hidden, ffn, num_experts, top_k, seed):
    torch.manual_seed(seed)
    x = _bf16(num_tokens, hidden)
    w13 = _bf16(num_experts, 2 * ffn, hidden)
    w2 = _bf16(num_experts, hidden, ffn)
    topk_ids = torch.tensor(
        [[(i + j) % num_experts for j in range(top_k)] for i in range(num_tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    return x, w13, w2, topk_weights, topk_ids


@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("num_tokens", [2, 64, 128])
def test_nsplit_fused_moe_matches_baseline(degree, num_tokens):
    # F % 8 == 0 required. A group of 4 threads teams over each expert.
    hidden, ffn, num_experts, top_k = 128, 64, 4, 2
    x, w13, w2, tw, ti = _moe_case(num_tokens, hidden, ffn, num_experts, top_k, seed=degree + num_tokens)

    base_w = prepare_fused_moe_bf16_tiled_weights(w13, w2)
    fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    # Reference: non-fused, default single-thread path (no N-split).
    ref = fused_moe_bf16_tiled(x, base_w, tw, ti, num_threads=1, activation="silu")

    # Fused via the hierarchical N-split team path: 1 group of 4 threads/expert.
    with _nsplit_env(groups_per_partition=1, core_bases="0"):
        out = fused_moe_bf16_tiled(
            x,
            fused_w,
            tw,
            ti,
            num_threads=4,
            activation="silu",
            silu_poly_degree=degree,
        )

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), atol=6e-2, rtol=6e-2)
