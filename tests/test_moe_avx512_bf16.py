# -*- coding: utf-8 -*-
from __future__ import annotations

import platform

import pytest
import torch

from fused_cpp.moe import available_fused_moe_bf16_tiled_backends
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


pytestmark = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64")
    or "x86_avx512_bf16" not in available_fused_moe_bf16_tiled_backends(),
    reason="requires an AVX-512 BF16 CPU and x86 MoE build",
)


def _case(
    *,
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.empty((tokens, hidden), dtype=torch.bfloat16).normal_(std=0.1, generator=generator)
    w13 = torch.empty((experts, 2 * intermediate, hidden), dtype=torch.bfloat16).normal_(
        std=0.1,
        generator=generator,
    )
    w2 = torch.empty((experts, hidden, intermediate), dtype=torch.bfloat16).normal_(
        std=0.1,
        generator=generator,
    )
    topk_ids = torch.tensor(
        [[(token + slot) % experts for slot in range(top_k)] for token in range(tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn((tokens, top_k), generator=generator), dim=-1)
    return inputs, w13, w2, topk_weights, topk_ids


def _assert_bf16_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = (actual_f32 - expected_f32).abs()
    max_abs = float(difference.max()) if difference.numel() else 0.0
    max_rel = float((difference / (expected_f32.abs() + 1e-12)).max()) if difference.numel() else 0.0
    cosine = float(
        torch.nn.functional.cosine_similarity(actual_f32.flatten(), expected_f32.flatten(), dim=0)
    )
    message = f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} cosine={cosine:.8f}"
    assert not torch.isnan(actual_f32).any(), message
    assert not torch.isinf(actual_f32).any(), message
    torch.testing.assert_close(actual_f32, expected_f32, atol=7e-2, rtol=7e-2, msg=message)
    assert cosine >= 0.999, message


@pytest.mark.parametrize(
    ("tokens", "hidden", "intermediate", "experts", "top_k"),
    [
        pytest.param(13, 31, 7, 1, 1, id="single-expert-all-tails"),
        pytest.param(25, 33, 17, 4, 2, id="multi-expert-all-tails"),
        pytest.param(12, 64, 32, 2, 2, id="full-tiles"),
        pytest.param(48, 64, 32, 1, 1, id="four-m12-panels"),
    ],
)
@pytest.mark.parametrize("num_threads", [1, 2], ids=["single-core", "dual-core"])
def test_avx512_fused_expert_matches_naive(
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    num_threads: int,
) -> None:
    """AVX-512 fused SiLU expert must match the PyTorch definition across tile tails."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        top_k=top_k,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_avx512_bf16",
    )

    actual = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=num_threads,
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    assert packed.gemm_backend == 101
    assert packed.backend_n_tile == 32
    assert packed.backend_name == "x86_avx512_bf16"
    _assert_bf16_close(actual, expected)


@pytest.mark.parametrize("degree", [4, 5, 6])
def test_avx512_silu_degrees_are_thread_deterministic(degree: int) -> None:
    """Single- and dual-thread schedules must produce identical BF16 values."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=29,
        hidden=65,
        intermediate=33,
        experts=3,
        top_k=2,
        seed=degree,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    serial = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
        silu_poly_degree=degree,
    )
    threaded = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=2,
        silu_poly_degree=degree,
    )

    torch.testing.assert_close(threaded.float(), serial.float(), atol=0, rtol=0)


def test_avx512_skip_weighted_and_out_buffer() -> None:
    """Top-1 direct BF16 store must honor skip_weighted and the supplied output buffer."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=25,
        hidden=47,
        intermediate=23,
        experts=2,
        top_k=1,
    )
    topk_weights.fill_(1.0)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    output = torch.empty_like(inputs)

    returned = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=2,
        skip_weighted=True,
        out=output,
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids, skip_weighted=True)

    assert returned is output
    _assert_bf16_close(output, expected)


def test_avx512_rejects_bias_and_more_than_two_threads() -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=2,
        hidden=16,
        intermediate=8,
        experts=1,
        top_k=1,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    with pytest.raises(RuntimeError, match="does not support expert bias"):
        fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            w13_bias=torch.zeros((1, 16)),
        )
    with pytest.raises(RuntimeError, match="num_threads=1 or 2"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=3)


def test_avx512_runtime_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime switch must remove AVX-512 from discovery and explicit dispatch."""
    _, w13, w2, _, _ = _case(
        tokens=1,
        hidden=16,
        intermediate=8,
        experts=1,
        top_k=1,
    )
    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_BF16", "0")

    assert "x86_avx512_bf16" not in available_fused_moe_bf16_tiled_backends()
    with pytest.raises(RuntimeError, match="not supported by this build/runtime"):
        prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend="x86_avx512_bf16",
        )
