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
    platform.machine() not in ("x86_64", "AMD64") or "x86_avx512_bf16" not in available_fused_moe_bf16_tiled_backends(),
    reason="requires an AVX-512 BF16 CPU and x86 MoE build",
)

requires_amx = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64") or "x86_amx_bf16" not in available_fused_moe_bf16_tiled_backends(),
    reason="requires an AMX BF16 CPU, Linux XTILEDATA permission, and Xbyak",
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
    cosine = float(torch.nn.functional.cosine_similarity(actual_f32.flatten(), expected_f32.flatten(), dim=0))
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
def test_avx512_silu_degrees_are_thread_deterministic(monkeypatch: pytest.MonkeyPatch, degree: int) -> None:
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
    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_IMPL", "intrinsic")
    intrinsic = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
        silu_poly_degree=degree,
    )
    _assert_bf16_close(serial, intrinsic)


@pytest.mark.parametrize("routes", range(1, 14), ids=lambda value: f"m{value}")
def test_avx512_exact_m_jit_matches_intrinsic(monkeypatch: pytest.MonkeyPatch, routes: int) -> None:
    """Every exact-M specialization and the M12+tail composition retain the fallback contract."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=routes,
        hidden=33,
        intermediate=17,
        experts=1,
        top_k=1,
        seed=routes,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_IMPL", "jit")
    jit = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)
    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_IMPL", "intrinsic")
    intrinsic = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)

    _assert_bf16_close(jit, intrinsic)


def test_avx512_rejects_unknown_implementation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=1,
        hidden=16,
        intermediate=8,
        experts=1,
        top_k=1,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_IMPL", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, jit, or intrinsic"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


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


@requires_amx
def test_amx_backend_is_explicit_and_preserves_avx512_auto() -> None:
    """AMX uses backend 102 while automatic x86 preparation remains on AVX-512."""
    _, w13, w2, _, _ = _case(
        tokens=1,
        hidden=33,
        intermediate=17,
        experts=1,
        top_k=1,
    )

    automatic = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    amx = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    assert automatic.gemm_backend == 101
    assert automatic.backend_name == "x86_avx512_bf16"
    assert amx.gemm_backend == 102
    assert amx.backend_n_tile == 32
    assert amx.backend_name == "x86_amx_bf16"
    assert amx.w13[0].shape[1] % 32 == 0
    assert amx.w2[0].shape[1] % 32 == 0


@requires_amx
@pytest.mark.parametrize("routes", range(1, 18), ids=lambda value: f"m{value}")
def test_amx_exact_m_matches_avx512_jit(routes: int) -> None:
    """AMX exact-M1..16 kernels and M16+tail composition retain the AVX result contract."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=routes,
        hidden=33,
        intermediate=17,
        experts=1,
        top_k=1,
        seed=100 + routes,
    )
    avx = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_avx512_bf16",
    )
    amx = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    expected = fused_moe_bf16_tiled(inputs, avx, topk_weights, topk_ids, num_threads=1)
    actual = fused_moe_bf16_tiled(inputs, amx, topk_weights, topk_ids, num_threads=1)

    _assert_bf16_close(actual, expected)


@requires_amx
@pytest.mark.parametrize("pattern", ["m1n2", "m2n2", "m1n4"])
@pytest.mark.parametrize("routes", [17, 32, 33, 64, 2048], ids=lambda value: f"m{value}")
def test_amx_patterns_match_naive_through_large_m(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    routes: int,
) -> None:
    """Each AMX pattern must preserve W13/W2 tails and routing through M=2048."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=routes,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=400 + routes,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    actual = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    _assert_bf16_close(actual, expected)


@requires_amx
@pytest.mark.parametrize("pattern", ["m2n2", "m1n4"])
def test_amx_patterns_hot_two_thread_direct_bf16(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
) -> None:
    """New kernels must support hot-expert N-split and direct BF16 stores."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=500,
    )
    topk_weights.fill_(1.0)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
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


@requires_amx
def test_amx_rejects_unknown_jit_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=1,
        hidden=32,
        intermediate=16,
        experts=1,
        top_k=1,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", "unknown")

    with pytest.raises(RuntimeError, match="must be m1n2, m2n2, or m1n4"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@requires_amx
@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_amx_fused_expert_matches_naive_for_tails_and_threads(degree: int, num_threads: int) -> None:
    """AMX W13, SiLU, W2, routing, and merge match the definition for shape tails and both workers."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=25,
        hidden=35,
        intermediate=19,
        experts=4,
        top_k=2,
        seed=200 + degree,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    actual = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=num_threads,
        silu_poly_degree=degree,
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    _assert_bf16_close(actual, expected)


@requires_amx
def test_amx_skip_weighted_direct_bf16_output() -> None:
    """AMX W2 writes a tail-sized top-1 result directly into the supplied BF16 buffer."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=17,
        hidden=47,
        intermediate=23,
        experts=2,
        top_k=1,
        seed=300,
    )
    topk_weights.fill_(1.0)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
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


@requires_amx
def test_amx_runtime_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The AMX kill switch hides only backend 102 and rejects explicit preparation."""
    _, w13, w2, _, _ = _case(
        tokens=1,
        hidden=32,
        intermediate=16,
        experts=1,
        top_k=1,
    )
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_BF16", "0")

    backends = available_fused_moe_bf16_tiled_backends()
    assert "x86_amx_bf16" not in backends
    assert "x86_avx512_bf16" in backends
    with pytest.raises(RuntimeError, match="not supported by this build/runtime"):
        prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend="x86_amx_bf16",
        )
