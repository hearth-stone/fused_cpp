"""Correctness tests for the DeepSeek V4 mHC Torch baseline."""

from __future__ import annotations

import pytest
import torch

import fused_cpp.deepseek_v4_mhc as mhc_module
from fused_cpp.deepseek_v4_mhc import (
    _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION,
    _native_sve_control_postprocess,
    _native_sve_pre_apply_rmsnorm,
    _native_sve_post,
    _native_sve_projection,
    mhc_post_hc_head_rmsnorm,
    mhc_post_hc_head_rmsnorm_sve_candidate,
    mhc_post_pre_rmsnorm,
    mhc_post_pre_rmsnorm_sve_candidate,
    mhc_pre_rmsnorm,
    mhc_pre_rmsnorm_sve_candidate,
    prepare_mhc_weight,
)


def _rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = input.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).bfloat16()


def _sinkhorn(logits: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = logits.softmax(-1) + eps
    value = value / (value.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(-1, keepdim=True) + eps)
        value = value / (value.sum(-2, keepdim=True) + eps)
    return value


def _pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, c, h = residual.shape
    flat = residual.flatten(1).float()
    mixes = flat @ fn.t()
    mixes = mixes * torch.rsqrt(flat.square().sum(-1, keepdim=True) / (c * h) + rms_eps)
    pre_mix = (mixes[:, :c] * hc_scale[0] + hc_base[:c]).sigmoid() + hc_pre_eps
    post_mix = ((mixes[:, c : 2 * c] * hc_scale[1] + hc_base[c : 2 * c]).sigmoid() * hc_post_mult_value).reshape(
        t, c, 1
    )
    comb_logits = mixes[:, 2 * c :].reshape(t, c, c) * hc_scale[2] + hc_base[2 * c :].reshape(1, c, c)
    comb_mix = _sinkhorn(comb_logits, sinkhorn_repeat, hc_sinkhorn_eps)
    raw = (pre_mix.unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    return post_mix, comb_mix, _rmsnorm(raw, norm_weight, norm_eps)


def _post_reference(
    layer_output: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    mixed = torch.bmm(comb_mix.transpose(1, 2), residual.float())
    return (mixed + post_mix * layer_output.float().unsqueeze(1)).bfloat16()


def _assert_mhc_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def _inputs(t: int, c: int = 2, h: int = 16) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260822 + t)
    p = 2 * c + c * c
    return {
        "residual": torch.randn((t, c, h), generator=generator).bfloat16().contiguous(),
        "layer_output": torch.randn((t, h), generator=generator).bfloat16().contiguous(),
        "pre_fn": (torch.randn((p, c * h), generator=generator) * 0.1).contiguous(),
        "head_fn": (torch.randn((c, c * h), generator=generator) * 0.1).contiguous(),
        "hc_scale": (torch.randn((3,), generator=generator) * 0.1).contiguous(),
        "hc_base": (torch.randn((p,), generator=generator) * 0.1).contiguous(),
        "head_scale": (torch.randn((1,), generator=generator) * 0.1).contiguous(),
        "head_base": (torch.randn((c,), generator=generator) * 0.1).contiguous(),
        "norm_weight": torch.randn((h,), generator=generator).bfloat16().contiguous(),
    }


@pytest.mark.parametrize("tokens", [0, 1, 7, 8, 16, 17, 128])
def test_mhc_pre_rmsnorm_matches_independent_reference(tokens: int) -> None:
    values = _inputs(tokens)
    residual_before = values["residual"].clone()
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")

    actual = mhc_pre_rmsnorm(
        values["residual"],
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        num_threads=2,
    )
    expected = _pre_reference(
        values["residual"],
        values["pre_fn"],
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
    )

    for result, reference in zip(actual, expected, strict=True):
        assert result.is_contiguous()
        _assert_mhc_close(result, reference)
    torch.testing.assert_close(values["residual"], residual_before, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("tokens", [0, 1, 7, 17])
def test_mhc_post_pre_rmsnorm_matches_strict_bf16_boundary(tokens: int) -> None:
    values = _inputs(tokens)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    previous_post, previous_comb, _ = _pre_reference(
        values["residual"],
        values["pre_fn"],
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
    )
    residual_expected = _post_reference(values["layer_output"], values["residual"], previous_post, previous_comb)
    next_expected = _pre_reference(
        residual_expected,
        values["pre_fn"],
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
    )

    actual = mhc_post_pre_rmsnorm(
        values["layer_output"],
        values["residual"],
        previous_post,
        previous_comb,
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
    )

    torch.testing.assert_close(actual[0], residual_expected, rtol=0.0, atol=0.0)
    for result, reference in zip(actual[1:], next_expected, strict=True):
        _assert_mhc_close(result, reference)


@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_mhc_post_hc_head_rmsnorm_matches_independent_reference(tokens: int) -> None:
    values = _inputs(tokens)
    prepared_head = prepare_mhc_weight(values["head_fn"], kind="head")
    previous_post, previous_comb, _ = _pre_reference(
        values["residual"],
        values["pre_fn"],
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
    )
    final_residual = _post_reference(values["layer_output"], values["residual"], previous_post, previous_comb)
    t, c, h = final_residual.shape
    flat = final_residual.flatten(1).float()
    mixes = flat @ values["head_fn"].t()
    mixes = mixes * torch.rsqrt(flat.square().sum(-1, keepdim=True) / (c * h) + 1e-6)
    pre_mix = (mixes * values["head_scale"] + values["head_base"]).sigmoid() + 1e-6
    head_output = (pre_mix.unsqueeze(-1) * final_residual.float()).sum(1).bfloat16()
    hidden_expected = _rmsnorm(head_output, values["norm_weight"], 1e-6)

    hidden_actual, residual_actual = mhc_post_hc_head_rmsnorm(
        values["layer_output"],
        values["residual"],
        previous_post,
        previous_comb,
        prepared_head,
        values["head_scale"],
        values["head_base"],
        values["norm_weight"],
    )

    assert hidden_actual.shape == (t, h)
    assert residual_actual.shape == (t, c, h)
    torch.testing.assert_close(hidden_actual, hidden_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual_actual, final_residual, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_mhc_post_hc_head_rmsnorm_sve_candidate_matches_torch(tokens: int) -> None:
    values = _inputs(tokens, c=4, h=16)
    prepared_head = prepare_mhc_weight(values["head_fn"], kind="head")
    previous_post, previous_comb, _ = _pre_reference(
        values["residual"],
        values["pre_fn"],
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
    )
    expected = mhc_post_hc_head_rmsnorm(
        values["layer_output"],
        values["residual"],
        previous_post,
        previous_comb,
        prepared_head,
        values["head_scale"],
        values["head_base"],
        values["norm_weight"],
        num_threads=8,
    )
    actual = mhc_post_hc_head_rmsnorm_sve_candidate(
        values["layer_output"],
        values["residual"],
        previous_post,
        previous_comb,
        prepared_head,
        values["head_scale"],
        values["head_base"],
        values["norm_weight"],
        num_threads=8,
    )

    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0.0, atol=0.03125)
    torch.testing.assert_close(actual[1].float(), expected[1].float(), rtol=0.0, atol=0.03125)


def test_mhc_pre_preserves_raw_input_bf16_round_trip() -> None:
    values = _inputs(7)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    actual = mhc_pre_rmsnorm(
        values["residual"], prepared, values["hc_scale"], values["hc_base"], values["norm_weight"]
    )[2]

    t, c, h = values["residual"].shape
    flat = values["residual"].flatten(1).float()
    mixes = flat @ values["pre_fn"].t()
    mixes *= torch.rsqrt(flat.square().sum(-1, keepdim=True) / (c * h) + 1e-6)
    pre_mix = (mixes[:, :c] * values["hc_scale"][0] + values["hc_base"][:c]).sigmoid() + 1e-6
    raw_fp32 = (pre_mix.unsqueeze(-1) * values["residual"].float()).sum(1)
    bypass = _rmsnorm(raw_fp32, values["norm_weight"], 1e-6)

    assert torch.count_nonzero(actual != bypass).item() > 0
    expected = _rmsnorm(raw_fp32.bfloat16(), values["norm_weight"], 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_prepare_mhc_weight_validates_geometry_and_owns_weight() -> None:
    weight = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32).contiguous()
    prepared = prepare_mhc_weight(weight, kind="pre")
    weight.zero_()

    assert (prepared.c, prepared.h, prepared.n, prepared.backend) == (2, 16, 8, "fp32")
    assert torch.count_nonzero(prepared.packed).item() > 0
    with pytest.raises(ValueError, match=r"C\*C\+2\*C"):
        prepare_mhc_weight(torch.ones((7, 32)), kind="pre")
    with pytest.raises(TypeError, match="CPU FP32"):
        prepare_mhc_weight(torch.ones((8, 32), dtype=torch.bfloat16), kind="pre")


def test_mhc_rejects_invalid_runtime_contracts() -> None:
    values = _inputs(1)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    with pytest.raises(ValueError, match="sinkhorn_repeat"):
        mhc_pre_rmsnorm(
            values["residual"],
            prepared,
            values["hc_scale"],
            values["hc_base"],
            values["norm_weight"],
            sinkhorn_repeat=0,
        )
    with pytest.raises(ValueError, match="num_threads"):
        mhc_pre_rmsnorm(
            values["residual"],
            prepared,
            values["hc_scale"],
            values["hc_base"],
            values["norm_weight"],
            num_threads=-1,
        )


def test_mhc_post_pre_sve_candidate_reports_unavailable_native_path(monkeypatch: pytest.MonkeyPatch) -> None:
    values = _inputs(1, c=4)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    monkeypatch.setattr(mhc_module, "_HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION", False)
    monkeypatch.setattr(mhc_module, "_native_sve_post", None)

    with pytest.raises(RuntimeError, match="SVE post-pre is unavailable"):
        mhc_post_pre_rmsnorm_sve_candidate(
            values["layer_output"],
            values["residual"],
            torch.zeros((1, 4, 1), dtype=torch.float32),
            torch.eye(4, dtype=torch.float32).unsqueeze(0),
            prepared,
            values["hc_scale"],
            values["hc_base"],
            values["norm_weight"],
        )


def test_mhc_dispatch_is_repeatable_across_requested_thread_counts() -> None:
    values = _inputs(17)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    first = mhc_pre_rmsnorm(
        values["residual"],
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        num_threads=1,
    )
    second = mhc_pre_rmsnorm(
        values["residual"],
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        num_threads=8,
    )
    for first_output, second_output in zip(first, second, strict=True):
        torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
@pytest.mark.parametrize(
    "tokens,b_window_bytes,sinkhorn_repeat",
    [(0, 4096, 1), (1, 4096, 1), (7, 4096, 2), (17, 1 << 20, 20), (64, 4096, 20)],
)
def test_mhc_sve_candidate_matches_torch_for_m_and_k_splits(
    tokens: int, b_window_bytes: int, sinkhorn_repeat: int
) -> None:
    values = _inputs(tokens, c=4, h=64)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    expected = mhc_pre_rmsnorm(
        values["residual"],
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        sinkhorn_repeat=sinkhorn_repeat,
        num_threads=8,
    )
    actual = mhc_pre_rmsnorm_sve_candidate(
        values["residual"],
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        sinkhorn_repeat=sinkhorn_repeat,
        num_threads=8,
        b_window_bytes=b_window_bytes,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[2].float(), expected[2].float(), rtol=0.0, atol=0.015625)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
def test_mhc_sve_control_postprocess_handles_extreme_logits_and_t_tail() -> None:
    """The SVE control kernel must keep extreme sigmoid inputs and a partial T vector finite and accurate."""
    assert _native_sve_control_postprocess is not None
    tokens = 9
    mixes = torch.empty((tokens, 24), dtype=torch.float32)
    first = torch.tensor([-100.0, -20.0, -2.0, 0.0, 2.0, 20.0, 100.0, -100.0])
    mixes[:, :8] = first
    mixes[:, 8:] = torch.linspace(-20.0, 20.0, 16)
    sqrsum = torch.zeros(tokens, dtype=torch.float32)
    hc_scale = torch.ones(3, dtype=torch.float32)
    hc_base = torch.zeros(24, dtype=torch.float32)

    pre, post, comb = _native_sve_control_postprocess(mixes, sqrsum, hc_scale, hc_base, 1, 1.0, 1e-6, 2.0, 1e-6, 20, 8)
    expected_pre = torch.sigmoid(mixes[:, :4]) + 1e-6
    expected_post = (torch.sigmoid(mixes[:, 4:8]) * 2.0).reshape(tokens, 4, 1)
    expected_comb = _sinkhorn(mixes[:, 8:].reshape(tokens, 4, 4), 20, 1e-6)

    assert torch.isfinite(pre).all()
    assert torch.isfinite(post).all()
    assert torch.isfinite(comb).all()
    torch.testing.assert_close(pre, expected_pre, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(post, expected_post, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(comb, expected_comb, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
@pytest.mark.parametrize("tokens,hidden_size", [(0, 65), (1, 65), (9, 65), (17, 128)])
def test_mhc_sve_pre_apply_rmsnorm_preserves_bf16_boundary_and_h_tail(tokens: int, hidden_size: int) -> None:
    """The native pre consumer must match FP32 four-stream reduction followed by the explicit BF16 boundary."""
    assert _native_sve_pre_apply_rmsnorm is not None
    generator = torch.Generator().manual_seed(20260824 + tokens + hidden_size)
    residual = torch.randn((tokens, 4, hidden_size), generator=generator).bfloat16().contiguous()
    pre_mix = torch.sigmoid(torch.randn((tokens, 4), generator=generator)).contiguous()
    norm_weight = torch.randn((hidden_size,), generator=generator).bfloat16().contiguous()

    actual = _native_sve_pre_apply_rmsnorm(residual, pre_mix, norm_weight, 1e-6, 8)
    raw = (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).bfloat16()
    expected = _rmsnorm(raw, norm_weight, 1e-6)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=0.03125)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
@pytest.mark.parametrize("tokens,hidden_size", [(0, 65), (1, 65), (9, 65), (17, 128)])
def test_mhc_sve_post_matches_k4_mixed_plus_rank_one_injection(tokens: int, hidden_size: int) -> None:
    """The U2 post kernel must preserve the fused fixed-K4 plus rank-one injection BF16 boundary."""
    assert _native_sve_post is not None
    generator = torch.Generator().manual_seed(20260825 + tokens + hidden_size)
    residual = torch.randn((tokens, 4, hidden_size), generator=generator).bfloat16().contiguous()
    layer_output = torch.randn((tokens, hidden_size), generator=generator).bfloat16().contiguous()
    post_mix = torch.randn((tokens, 4, 1), generator=generator).contiguous()
    comb_mix = torch.randn((tokens, 4, 4), generator=generator).contiguous()

    actual = _native_sve_post(layer_output, residual, post_mix, comb_mix, 8)
    expected = _post_reference(layer_output, residual, post_mix, comb_mix)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=0.03125)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
def test_mhc_sve_post_rejects_flat_post_mix() -> None:
    """The native ABI must not silently accept a layout other than contiguous [T,4,1]."""
    assert _native_sve_post is not None
    residual = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    layer_output = torch.zeros((1, 8), dtype=torch.bfloat16)
    post_mix = torch.zeros((1, 4), dtype=torch.float32)
    comb_mix = torch.eye(4, dtype=torch.float32).unsqueeze(0)

    with pytest.raises(RuntimeError, match=r"post_mix.*\[T,4,1\]"):
        _native_sve_post(layer_output, residual, post_mix, comb_mix, 1)


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
def test_mhc_sve_default_window_balances_deepseek_v4_k() -> None:
    values = _inputs(1, c=4, h=4096)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")

    assert _native_sve_projection is not None
    _, _, mr, kc = _native_sve_projection(values["residual"], prepared.packed, 1, 1 << 20)

    assert mr in {4, 7}
    assert kc == 8192


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
def test_mhc_sve_selects_m7_for_sve256_projection() -> None:
    values = _inputs(8, c=4, h=64)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")

    assert _native_sve_projection is not None
    low_mr = _native_sve_projection(values["residual"], prepared.packed, 8, 1 << 20)[2]
    high_mr = _native_sve_projection(values["residual"], prepared.packed, 32, 1 << 20)[2]

    assert (low_mr, high_mr) in {(7, 7), (4, 4)}


@pytest.mark.skipif(not _HAS_DEEPSEEK_V4_MHC_SVE_PROJECTION, reason="SVE VL128/VL256 projection is unavailable")
def test_mhc_post_pre_sve_candidate_preserves_strict_post_boundary() -> None:
    values = _inputs(17, c=4, h=64)
    prepared = prepare_mhc_weight(values["pre_fn"], kind="pre")
    previous_post, previous_comb, _ = mhc_pre_rmsnorm(
        values["residual"], prepared, values["hc_scale"], values["hc_base"], values["norm_weight"]
    )
    expected = mhc_post_pre_rmsnorm(
        values["layer_output"],
        values["residual"],
        previous_post,
        previous_comb,
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        num_threads=8,
    )
    actual = mhc_post_pre_rmsnorm_sve_candidate(
        values["layer_output"],
        values["residual"],
        previous_post,
        previous_comb,
        prepared,
        values["hc_scale"],
        values["hc_base"],
        values["norm_weight"],
        num_threads=8,
    )

    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0.0, atol=0.03125)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual[3].float(), expected[3].float(), rtol=0.0, atol=0.015625)
