from __future__ import annotations

import pytest
import torch

from fused_cpp.deepseek_v4_w8a8 import (
    _HAS_DEEPSEEK_V4_W8A8,
    deepseek_v4_w8a8_linear,
    deepseek_v4_w8a8_linear_pair,
    deepseek_v4_wo_b_w8a8,
    prepare_deepseek_v4_w8a8_linear_quantized_weight,
    prepare_deepseek_v4_w8a8_linear_weight,
)


pytestmark = pytest.mark.skipif(not _HAS_DEEPSEEK_V4_W8A8, reason="ARM SVE i8mm W8A8 is unavailable")


def _quantize_rows(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp32 = input.float()
    maximum = fp32.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    return (fp32 / scale).round().clamp(-127, 127).to(torch.int32), scale


def test_w8a8_linear_matches_dynamic_quantization_reference() -> None:
    generator = torch.Generator().manual_seed(41)
    m, k, n = 13, 64, 96
    input = (torch.randn((m, k), generator=generator) * 0.2).to(torch.bfloat16)
    weight = torch.randint(-16, 17, (n, k), dtype=torch.int8, generator=generator)
    weight_scale = torch.rand((n,), generator=generator) * 0.004 + 0.0005
    prepared = prepare_deepseek_v4_w8a8_linear_quantized_weight(weight, weight_scale)

    actual = deepseek_v4_w8a8_linear(input, prepared, num_threads=2)
    input_q, input_scale = _quantize_rows(input)
    expected = ((input_q @ weight.to(torch.int32).T).float() * input_scale * weight_scale).to(torch.bfloat16)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=0.0)


def test_w8a8_dynamic_quantization_uses_ties_to_even() -> None:
    input = torch.zeros((2, 16), dtype=torch.bfloat16)
    input[0, :5] = torch.tensor([127.0, 0.5, 1.5, -0.5, -1.5], dtype=torch.bfloat16)
    weight = torch.zeros((8, 16), dtype=torch.int8)
    weight[:5, :5] = torch.eye(5, dtype=torch.int8)
    prepared = prepare_deepseek_v4_w8a8_linear_quantized_weight(weight, torch.ones(8))

    actual = deepseek_v4_w8a8_linear(input, prepared, num_threads=1)
    input_q, input_scale = _quantize_rows(input)
    expected = ((input_q @ weight.to(torch.int32).T).float() * input_scale).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_wo_b_alias_supports_out_and_bf16_prepare() -> None:
    generator = torch.Generator().manual_seed(43)
    input = (torch.randn((12, 64), generator=generator) * 0.2).to(torch.bfloat16)
    weight = (torch.randn((128, 64), generator=generator) * 0.02).to(torch.bfloat16)
    prepared = prepare_deepseek_v4_w8a8_linear_weight(weight)
    expected = deepseek_v4_w8a8_linear(input, prepared, num_threads=2)
    out = torch.empty_like(expected)
    actual = deepseek_v4_wo_b_w8a8(input, prepared, num_threads=2, out=out)
    assert actual is out
    assert torch.equal(actual, expected)


def test_w8a8_pair_matches_two_single_projections() -> None:
    generator = torch.Generator().manual_seed(47)
    input = (torch.randn((24, 64), generator=generator) * 0.2).to(torch.bfloat16)
    first = prepare_deepseek_v4_w8a8_linear_weight(
        (torch.randn((96, 64), generator=generator) * 0.02).to(torch.bfloat16)
    )
    second = prepare_deepseek_v4_w8a8_linear_weight(
        (torch.randn((128, 64), generator=generator) * 0.02).to(torch.bfloat16)
    )
    expected_first = deepseek_v4_w8a8_linear(input, first, num_threads=2)
    expected_second = deepseek_v4_w8a8_linear(input, second, num_threads=2)
    actual_first, actual_second = deepseek_v4_w8a8_linear_pair(input, first, second, num_threads=2)
    assert torch.equal(actual_first, expected_first)
    assert torch.equal(actual_second, expected_second)


def test_w8a8_pair_supports_logical_n_tails() -> None:
    input = torch.arange(128, dtype=torch.float32).reshape(2, 64).to(torch.bfloat16) / 127
    first = prepare_deepseek_v4_w8a8_linear_quantized_weight(
        torch.arange(13 * 64, dtype=torch.int32).remainder(17).sub(8).reshape(13, 64).to(torch.int8),
        torch.linspace(0.001, 0.002, 13),
    )
    second = prepare_deepseek_v4_w8a8_linear_quantized_weight(
        torch.arange(21 * 64, dtype=torch.int32).remainder(19).sub(9).reshape(21, 64).to(torch.int8),
        torch.linspace(0.002, 0.003, 21),
    )

    expected_first = deepseek_v4_w8a8_linear(input, first, num_threads=2)
    expected_second = deepseek_v4_w8a8_linear(input, second, num_threads=2)
    actual_first, actual_second = deepseek_v4_w8a8_linear_pair(input, first, second, num_threads=2)
    assert torch.equal(actual_first, expected_first)
    assert torch.equal(actual_second, expected_second)


def test_w8a8_pair_rejects_mismatched_k() -> None:
    first = prepare_deepseek_v4_w8a8_linear_quantized_weight(
        torch.ones((16, 32), dtype=torch.int8), torch.ones(16)
    )
    second = prepare_deepseek_v4_w8a8_linear_quantized_weight(
        torch.ones((16, 48), dtype=torch.int8), torch.ones(16)
    )
    with pytest.raises(ValueError, match="share K"):
        deepseek_v4_w8a8_linear_pair(torch.zeros((2, 32), dtype=torch.bfloat16), first, second)


def test_w8a8_linear_rejects_wrong_input_k() -> None:
    weight = torch.ones((16, 32), dtype=torch.int8)
    prepared = prepare_deepseek_v4_w8a8_linear_quantized_weight(weight, torch.ones(16))
    with pytest.raises(ValueError, match="input K"):
        deepseek_v4_w8a8_linear(torch.zeros((2, 16), dtype=torch.bfloat16), prepared)
