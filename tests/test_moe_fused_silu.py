# -*- coding: utf-8 -*-
"""Unit tests for the fused w13-GEMM + SiLU-and-mul epilogue.

Task 1: the interleaved w13 prepack must place gate/up columns per 8-col
N-block as [g0 g1 g2 g3 | u0 u1 u2 u3], i.e.
  gate feature f -> packed column 8*(f//4) + (f%4)
  up   feature f -> packed column 8*(f//4) + 4 + (f%4)
Verified by running the fp32 GEMM on the interleaved-packed weights and
checking each interleaved column against the reference gate/up feature.
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


def _pack_interleaved_gemm(A, w13):
    from fused_cpp import _C

    return _C.fused_moe_test_pack_interleaved_gemm(A, w13)


def _gate_col(f: int) -> int:
    return 8 * (f // 4) + (f % 4)


def _up_col(f: int) -> int:
    return 8 * (f // 4) + 4 + (f % 4)


# (H, F): F multiple of 4 gives a clean interleave; F=6 exercises padding.
@pytest.mark.parametrize("h,f", [(256, 256), (128, 512), (256, 64), (64, 6)])
@pytest.mark.parametrize("m", [1, 4, 8, 17, 64])
def test_interleaved_prepack_permutation(h, f, m):
    torch.manual_seed(0)
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.05

    c = _pack_interleaved_gemm(a, w13)  # [M, 2 * ceil(F,4)] interleaved order
    f_pad4 = (f + 3) // 4 * 4
    assert c.shape == (m, 2 * f_pad4)
    assert c.dtype == torch.float32

    ref = a.float() @ w13.float().t()  # [M, 2F]
    gate_ref = ref[:, :f]
    up_ref = ref[:, f:]

    for feat in range(f):
        torch.testing.assert_close(c[:, _gate_col(feat)], gate_ref[:, feat], atol=8e-2, rtol=8e-2)
        torch.testing.assert_close(c[:, _up_col(feat)], up_ref[:, feat], atol=8e-2, rtol=8e-2)

    # Padded features [F, F_pad4) must be zero (weights zero-filled there).
    for feat in range(f, f_pad4):
        assert torch.all(c[:, _gate_col(feat)] == 0.0)
        assert torch.all(c[:, _up_col(feat)] == 0.0)


def _fused_w13_linear(A, w13):
    from fused_cpp import _C

    return _C.fused_moe_test_fused_w13_linear(A, w13)


# Task 2: fused kernel store epilogue computing gate*up (no silu). Validates
# layout / de-interleave / column-advance / bf16 store, isolated from the exp.
@pytest.mark.parametrize("h,f", [(256, 256), (128, 512), (256, 64)])
@pytest.mark.parametrize("m", [8, 16, 64, 128])
def test_fused_w13_linear_matches_gate_mul_up(h, f, m):
    torch.manual_seed(1)
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.05

    out = _fused_w13_linear(a, w13)  # [M, F] bf16 = gate*up
    assert out.shape == (m, f)
    assert out.dtype == torch.bfloat16

    ref = a.float() @ w13.float().t()  # [M, 2F]
    gate = ref[:, :f]
    up = ref[:, f:]
    ref_lin = gate * up

    torch.testing.assert_close(out.float(), ref_lin, atol=6e-2, rtol=6e-2)


def _fused_w13_silu(A, w13, degree=5):
    from fused_cpp import _C

    return _C.fused_moe_test_fused_w13_silu(A, w13, degree)


def _ref_silu_mul(a, w13, f):
    ref = a.float() @ w13.float().t()  # [M, 2F]
    gate = ref[:, :f]
    up = ref[:, f:]
    silu = gate / (1.0 + torch.exp(-gate))
    return silu * up


# Task 3: fused SiLU-and-mul (poly5) must match the fp32 reference silu(gate)*up
# within a tolerance that covers bf16 rounding + the poly-5 exp approximation.
@pytest.mark.parametrize("h,f", [(256, 256), (128, 512), (256, 64)])
@pytest.mark.parametrize("m", [8, 16, 64, 128])
def test_fused_w13_silu_poly5_matches_reference(h, f, m):
    torch.manual_seed(2)
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.05

    out = _fused_w13_silu(a, w13, degree=5)
    assert out.shape == (m, f)
    assert out.dtype == torch.bfloat16

    ref = _ref_silu_mul(a, w13, f)
    torch.testing.assert_close(out.float(), ref, atol=6e-2, rtol=6e-2)


# Wider input range to exercise the exp poly / clamp envelope.
@pytest.mark.parametrize("scale", [0.2, 1.0])
def test_fused_w13_silu_poly5_wide_range(scale):
    torch.manual_seed(3)
    h, f, m = 256, 256, 64
    a = torch.randn(m, h, dtype=torch.bfloat16) * scale
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * scale

    out = _fused_w13_silu(a, w13, degree=5).float()
    ref = _ref_silu_mul(a, w13, f)
    # Relative error on the silu-mul output; poly5 exp is ~1-2 ULP so the
    # dominant error is bf16 rounding of inputs/outputs.
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)


# Task 4: all three exp-poly degrees produce silu(gate)*up. Higher degree =
# more accurate; the tolerance is dominated by bf16 rounding anyway.
@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("m", [8, 64])
def test_fused_w13_silu_all_degrees(degree, m):
    torch.manual_seed(4)
    h, f = 256, 256
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.1

    out = _fused_w13_silu(a, w13, degree=degree).float()
    ref = _ref_silu_mul(a, w13, f)
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)


def test_fused_w13_silu_degree_error_ordering():
    # poly6 should be at least as accurate as poly4 against an fp32-exp
    # reference (both bounded by bf16 rounding, but the exp term differs).
    torch.manual_seed(5)
    h, f, m = 512, 512, 64
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.3
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.3
    ref = _ref_silu_mul(a, w13, f)

    errs = {}
    for d in (4, 5, 6):
        out = _fused_w13_silu(a, w13, degree=d).float()
        errs[d] = (out - ref).abs().max().item()
    # Sanity: all within a loose bound; higher degree not worse than lower+slack.
    for d in (4, 5, 6):
        assert errs[d] < 0.2, (d, errs)
    assert errs[6] <= errs[4] + 1e-3


def test_fused_w13_silu_invalid_degree_raises():
    a = torch.randn(8, 64, dtype=torch.bfloat16) * 0.05
    w13 = torch.randn(128, 64, dtype=torch.bfloat16) * 0.05
    with pytest.raises(Exception):
        _fused_w13_silu(a, w13, degree=3)


# Task 5: arbitrary M (m=8 blocks + m=1/2/4 tails) handled fully in-kernel.
@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 9, 13, 15, 17, 31, 130])
def test_fused_w13_silu_arbitrary_M(degree, m):
    torch.manual_seed(6)
    h, f = 256, 256
    a = torch.randn(m, h, dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(2 * f, h, dtype=torch.bfloat16) * 0.1

    out = _fused_w13_silu(a, w13, degree=degree).float()
    assert out.shape == (m, f)
    ref = _ref_silu_mul(a, w13, f)
    torch.testing.assert_close(out, ref, atol=8e-2, rtol=8e-2)
