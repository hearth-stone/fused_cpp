from __future__ import annotations

import math

import pytest
import torch

from fused_cpp.deepseek_v4_inv_rope_woa import (
    _HAS_DEEPSEEK_V4_INV_ROPE_WOA,
    deepseek_v4_inv_rope_grouped_woa,
    deepseek_v4_inv_rope_grouped_woa_torch_reference,
    prepare_deepseek_v4_inv_rope_woa,
)


def _cos_sin_cache(max_position: int, rope_dim: int) -> torch.Tensor:
    half = rope_dim // 2
    position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
    frequency = torch.arange(half, dtype=torch.float32).unsqueeze(0)
    angle = position * 0.17 + frequency * 0.031
    return torch.cat((torch.cos(angle), torch.sin(angle)), dim=1).contiguous()


def _inputs(
    *,
    tokens: int = 7,
    groups: int = 2,
    heads_per_group: int = 2,
    head_dim: int = 8,
    rope_dim: int = 4,
    output_rank: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    heads = groups * heads_per_group
    o = (torch.randn(tokens, heads, head_dim) * 0.2).to(torch.bfloat16)
    positions = torch.tensor([(index * 7) % 19 for index in range(tokens)], dtype=torch.int64)
    cache = _cos_sin_cache(19, rope_dim)
    weight = (
        torch.randn(groups * output_rank, heads_per_group * head_dim) * 0.2
    ).to(torch.bfloat16)
    return o, positions, cache, weight


@pytest.mark.parametrize("tokens", [0, 1, 2, 7, 64])
def test_public_torch_path_matches_materialized_reference(tokens: int) -> None:
    o, positions, cache, weight = _inputs(tokens=tokens)
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=2,
        heads_per_group=2,
        head_dim=8,
        rope_dim=4,
        backend="torch",
    )

    expected = deepseek_v4_inv_rope_grouped_woa_torch_reference(
        o,
        positions,
        cache,
        weight,
        n_groups=2,
        heads_per_group=2,
        rope_dim=4,
    )
    actual = deepseek_v4_inv_rope_grouped_woa(o, positions, cache, prepared)

    assert actual.shape == (tokens, 2, 5)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_inverse_rope_sign_and_group_mapping_against_scalar_oracle() -> None:
    o = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([1], dtype=torch.int64)
    cache = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float32)
    weight = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -0.25]],
        dtype=torch.bfloat16,
    )
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=2,
        heads_per_group=1,
        head_dim=4,
        rope_dim=2,
        backend="torch",
    )

    actual = deepseek_v4_inv_rope_grouped_woa(o, positions, cache, prepared)

    expected_values: list[float] = []
    for group in range(2):
        source = [float(value) for value in o[0, group]]
        even, odd = source[2], source[3]
        inverse = source[:2] + [even * 0.6 + odd * 0.8, odd * 0.6 - even * 0.8]
        inverse_bf16 = torch.tensor(inverse, dtype=torch.bfloat16).float()
        expected_values.append(float(torch.dot(inverse_bf16, weight[group].float())))
    expected = torch.tensor(expected_values, dtype=torch.bfloat16).view(1, 2, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_out_is_returned_and_inputs_are_read_only() -> None:
    o, positions, cache, weight = _inputs()
    original_o = o.clone()
    original_weight = weight.clone()
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=2,
        heads_per_group=2,
        head_dim=8,
        rope_dim=4,
    )
    out = torch.empty(7, 2, 5, dtype=torch.bfloat16)

    returned = deepseek_v4_inv_rope_grouped_woa(
        o,
        positions,
        cache,
        prepared,
        out=out,
        core_ids=(0, 2),
    )

    assert returned is out
    torch.testing.assert_close(o, original_o, atol=0, rtol=0)
    torch.testing.assert_close(weight, original_weight, atol=0, rtol=0)


def test_tp2_m5_geometry_empty_batch() -> None:
    weight = torch.empty(4 * 1024, 8 * 512, dtype=torch.bfloat16)
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=4,
        heads_per_group=8,
        head_dim=512,
        rope_dim=64,
        backend="torch",
    )

    output = deepseek_v4_inv_rope_grouped_woa(
        torch.empty(0, 32, 512, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.int64),
        torch.empty(1, 64, dtype=torch.float32),
        prepared,
    )

    assert output.shape == (0, 4, 1024)


def test_invalid_position_and_output_alias_are_rejected() -> None:
    o, positions, cache, weight = _inputs(tokens=1, groups=1, output_rank=1)
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=1,
        heads_per_group=2,
        head_dim=8,
        rope_dim=4,
        backend="torch",
    )
    positions.fill_(cache.shape[0])
    with pytest.raises(ValueError, match="positions must be in"):
        deepseek_v4_inv_rope_grouped_woa(o, positions, cache, prepared)

    aliasing_out = o.view(1, 1, 16)[:, :, :1]
    positions.zero_()
    with pytest.raises(ValueError, match="out must not alias"):
        deepseek_v4_inv_rope_grouped_woa(
            o,
            positions,
            cache,
            prepared,
            out=aliasing_out,
        )


def test_unavailable_native_backend_is_explicit() -> None:
    if _HAS_DEEPSEEK_V4_INV_ROPE_WOA:
        pytest.skip("native backend is available in this build")
    _, _, _, weight = _inputs()
    with pytest.raises(RuntimeError, match="unavailable"):
        prepare_deepseek_v4_inv_rope_woa(
            weight,
            n_groups=2,
            heads_per_group=2,
            head_dim=8,
            rope_dim=4,
            backend="arm_sve_bf16",
        )


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_INV_ROPE_WOA,
    reason="SVE BF16 inverse-RoPE grouped WO_A backend is unavailable",
)
@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 11, 12, 13, 24, 25])
def test_native_matches_torch_reference_for_m_tails(tokens: int) -> None:
    o, positions, cache, weight = _inputs(tokens=tokens, output_rank=17)
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=2,
        heads_per_group=2,
        head_dim=8,
        rope_dim=4,
        backend="arm_sve_bf16",
    )
    expected = deepseek_v4_inv_rope_grouped_woa_torch_reference(
        o,
        positions,
        cache,
        weight,
        n_groups=2,
        heads_per_group=2,
        rope_dim=4,
    )

    actual = deepseek_v4_inv_rope_grouped_woa(o, positions, cache, prepared, core_ids=(0,))

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.125, rtol=0.02)


@pytest.mark.skipif(
    not _HAS_DEEPSEEK_V4_INV_ROPE_WOA,
    reason="SVE BF16 inverse-RoPE grouped WO_A backend is unavailable",
)
def test_native_handles_k4_crossing_nope_rope_boundary() -> None:
    o, positions, cache, weight = _inputs(
        tokens=13,
        groups=2,
        heads_per_group=1,
        head_dim=4,
        rope_dim=2,
        output_rank=17,
    )
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=2,
        heads_per_group=1,
        head_dim=4,
        rope_dim=2,
        backend="arm_sve_bf16",
    )
    expected = deepseek_v4_inv_rope_grouped_woa_torch_reference(
        o,
        positions,
        cache,
        weight,
        n_groups=2,
        heads_per_group=1,
        rope_dim=2,
    )

    actual = deepseek_v4_inv_rope_grouped_woa(o, positions, cache, prepared, core_ids=(0,))

    torch.testing.assert_close(actual.float(), expected.float(), atol=0.125, rtol=0.02)


def test_cache_contains_valid_trigonometric_pairs() -> None:
    cache = _cos_sin_cache(5, 6)
    pair_norm = cache[:, :3].square() + cache[:, 3:].square()
    torch.testing.assert_close(pair_norm, torch.ones_like(pair_norm), atol=1e-6, rtol=1e-6)
    assert math.isfinite(float(cache.sum()))
