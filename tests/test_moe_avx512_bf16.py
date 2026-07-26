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

requires_amx_n64 = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64")
    or "x86_amx_bf16_n64" not in available_fused_moe_bf16_tiled_backends(),
    reason="requires the experimental AMX BF16 N64 packed layout",
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
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")

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


def test_avx512_persistent_intermediate_matches_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grow-only intermediate pool must preserve the AVX-512 exact-M and routing contract."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=29,
        hidden=65,
        intermediate=33,
        experts=3,
        top_k=2,
        seed=501,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")

    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE", "0")
    transient = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE", "1")
    persistent = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    reused = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)

    torch.testing.assert_close(persistent.float(), transient.float(), atol=0, rtol=0)
    torch.testing.assert_close(reused.float(), transient.float(), atol=0, rtol=0)


def test_avx512_persistent_input_matches_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gathered-input pool must preserve AVX-512 physical-M16 packing and route order."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=29,
        hidden=65,
        intermediate=33,
        experts=3,
        top_k=2,
        seed=505,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")

    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INPUT", "0")
    transient = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INPUT", "1")
    persistent = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    reused = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)

    torch.testing.assert_close(persistent.float(), transient.float(), atol=0, rtol=0)
    torch.testing.assert_close(reused.float(), transient.float(), atol=0, rtol=0)


def test_x86_rejects_unknown_persistent_intermediate_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=1,
        hidden=32,
        intermediate=16,
        experts=1,
        top_k=1,
        seed=504,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, 0, or 1"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


def test_x86_rejects_unknown_persistent_input_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=1,
        hidden=32,
        intermediate=16,
        experts=1,
        top_k=1,
        seed=506,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INPUT", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, 0, or 1"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


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
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")

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
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_IMPL", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, jit, or intrinsic"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@pytest.mark.parametrize(
    ("backend", "pattern"),
    [
        pytest.param("x86_avx512_bf16", None, id="avx512"),
        pytest.param("x86_amx_bf16", "m1n2", marks=requires_amx, id="amx-m1n2"),
        pytest.param("x86_amx_bf16", "m2n2", marks=requires_amx, id="amx-m2n2"),
        pytest.param("x86_amx_bf16", "m1n4", marks=requires_amx, id="amx-m1n4"),
    ],
)
@pytest.mark.parametrize(
    ("w13_blocks", "w2_blocks"),
    [pytest.param(2, 3, id="even-odd"), pytest.param(5, 2, id="odd-even")],
)
@pytest.mark.parametrize(
    "num_threads",
    [1, 2, 4, 8],
    ids=["one-thread", "two-threads", "four-threads", "eight-threads"],
)
def test_x86_cache_block_windows_preserve_tails_and_threads(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    pattern: str | None,
    w13_blocks: int,
    w2_blocks: int,
    num_threads: int,
) -> None:
    """N-window cache blocking must preserve odd block counts, tails, and cooperative N splitting."""
    monkeypatch.setenv("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS", str(w13_blocks))
    monkeypatch.setenv("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS", str(w2_blocks))
    if pattern is not None:
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=131,
        intermediate=83,
        experts=1,
        top_k=1,
        seed=601 + w13_blocks + w2_blocks,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend=backend)

    actual = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    _assert_bf16_close(actual, expected)


@pytest.mark.parametrize(
    ("environment", "value"),
    [
        pytest.param("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS", "-1", id="negative-w13"),
        pytest.param("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS", "abc", id="nonnumeric-w2"),
    ],
)
@pytest.mark.parametrize("num_threads", [1, 8])
def test_x86_rejects_invalid_cache_block_window(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    value: str,
    num_threads: int,
) -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=1,
        hidden=257,
        intermediate=129,
        experts=1,
        top_k=1,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
    monkeypatch.setenv(environment, value)

    with pytest.raises(RuntimeError, match="must be auto or a non-negative integer"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)


@pytest.mark.parametrize(
    ("backend", "pattern"),
    [
        pytest.param("x86_avx512_bf16", None, id="avx512"),
        pytest.param("x86_amx_bf16", "m1n2", marks=requires_amx, id="amx-m1n2"),
        pytest.param("x86_amx_bf16", "m2n2", marks=requires_amx, id="amx-m2n2"),
        pytest.param("x86_amx_bf16", "m1n4", marks=requires_amx, id="amx-m1n4"),
    ],
)
@pytest.mark.parametrize(
    "route_counts",
    [
        pytest.param([33], id="one-active-expert"),
        pytest.param([25, 7, 1], id="three-skewed-experts"),
        pytest.param([4] * 8, id="eight-active-experts"),
        pytest.param([129, 32, 31, 30, 29, 28, 27, 26], id="route-skewed-wave"),
    ],
)
def test_x86_eight_thread_expert_or_nsplit_is_thread_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    pattern: str | None,
    route_counts: list[int],
) -> None:
    """Eight workers split N for sparse/skewed experts and retain expert parallelism when routes are balanced."""
    if pattern is not None:
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    tokens = sum(route_counts)
    inputs, w13, w2, topk_weights, _ = _case(
        tokens=tokens,
        hidden=257,
        intermediate=129,
        experts=len(route_counts),
        top_k=1,
        seed=730 + len(route_counts),
    )
    topk_ids = torch.tensor(
        [expert for expert, routes in enumerate(route_counts) for _ in range(routes)],
        dtype=torch.int32,
    ).view(tokens, 1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend=backend)

    serial = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)
    threaded = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=8)
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    torch.testing.assert_close(threaded.float(), serial.float(), atol=0, rtol=0)
    _assert_bf16_close(threaded, expected)


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
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
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


@pytest.mark.parametrize("implementation", ["jit", "intrinsic"])
@pytest.mark.parametrize("num_threads", [1, 8], ids=["one-thread", "eight-threads"])
def test_avx512_weighted_top1_direct_matches_route_workspace(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str,
    num_threads: int,
) -> None:
    """Weighted top-1 direct output must match the FP32 route-workspace reduction exactly."""
    monkeypatch.setenv("FUSED_CPP_MOE_AVX512_IMPL", implementation)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=29,
        hidden=47,
        intermediate=23,
        experts=3,
        top_k=1,
        seed=351,
    )
    topk_weights.copy_(torch.linspace(0.125, 1.125, inputs.size(0)).view(-1, 1))
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")

    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "0")
    workspace = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)
    output = torch.empty_like(inputs)
    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "1")
    returned = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=num_threads,
        out=output,
    )

    assert returned is output
    torch.testing.assert_close(output.float(), workspace.float(), atol=0, rtol=0)
    _assert_bf16_close(output, fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids))


def test_x86_weighted_top1_direct_keeps_topk2_route_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in direct epilogue is top-1-only and must not change top-k reduction order."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=13,
        hidden=35,
        intermediate=19,
        experts=3,
        top_k=2,
        seed=352,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "0")
    disabled = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "1")
    enabled = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)

    torch.testing.assert_close(enabled.float(), disabled.float(), atol=0, rtol=0)


def test_x86_rejects_unknown_weighted_top1_direct_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=1,
        hidden=32,
        intermediate=16,
        experts=1,
        top_k=1,
        seed=353,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, 0, or 1"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


def test_avx512_rejects_bias_and_excessive_threads() -> None:
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=2,
        hidden=16,
        intermediate=8,
        experts=1,
        top_k=1,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="x86_avx512_bf16")

    with pytest.raises(RuntimeError, match="does not support expert bias"):
        fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            w13_bias=torch.zeros((1, 16)),
        )
    with pytest.raises(RuntimeError, match=r"num_threads in \[1, 256\]"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=257)


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
def test_auto_backend_uses_shared_k32_weights_with_explicit_isa_backends_available() -> None:
    """Automatic x86 preparation records its shared K32 dispatch identity."""
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
    avx512 = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_avx512_bf16",
    )

    assert automatic.gemm_backend == 104
    assert automatic.backend_name == "x86_bf16_auto"
    assert automatic.w13[0].shape == amx.w13[0].shape
    assert automatic.w2[0].shape == amx.w2[0].shape
    assert amx.gemm_backend == 102
    assert amx.backend_n_tile == 32
    assert amx.backend_name == "x86_amx_bf16"
    assert amx.w13[0].shape[1] % 32 == 0
    assert amx.w2[0].shape[1] % 32 == 0
    assert avx512.gemm_backend == 101
    assert avx512.backend_name == "x86_avx512_bf16"


@requires_amx
def test_dimension_aware_policy_profile_and_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The C8i calibration selects ISA, threads, pattern, cache, and skew from dimensions."""
    from fused_cpp import _moe_C

    monkeypatch.setenv("FUSED_CPP_MOE_X86_POLICY_PROFILE", "intel_06_ad_c8i_v1")
    monkeypatch.delenv("FUSED_CPP_MOE_X86_ISA", raising=False)

    tiny = _moe_C.fused_moe_test_x86_policy(64, 16, [1], 8)
    assert tiny["profile"] == "intel_06_ad_c8i_v1"
    assert tiny["isa"] == "avx512_bf16"
    assert tiny["execution_threads"] == 1
    assert tiny["nsplit_target_rows"] == 256
    assert tiny["experts"][0] == {
        "rows": 1,
        "amx_pattern": "m2n2",
        "w13_cache_blocks": 0,
        "w2_cache_blocks": 0,
    }

    isa_boundary = _moe_C.fused_moe_test_x86_policy(128, 64, [1], 8)
    isa_above_boundary = _moe_C.fused_moe_test_x86_policy(128, 64, [2], 8)
    assert isa_boundary["isa"] == "avx512_bf16"
    assert isa_above_boundary["isa"] == "amx_bf16"

    small_multi_expert = _moe_C.fused_moe_test_x86_policy(256, 64, [1] * 8, 8)
    medium_multi_expert = _moe_C.fused_moe_test_x86_policy(512, 128, [1] * 8, 8)
    assert small_multi_expert["execution_threads"] == 1
    assert medium_multi_expert["execution_threads"] == 8

    production = _moe_C.fused_moe_test_x86_policy(4096, 512, [1536, 256, 256], 8)
    assert production["isa"] == "amx_bf16"
    assert production["execution_threads"] == 8
    assert production["nsplit_target_rows"] == 64
    assert production["route_skewed"] is True
    assert production["experts"][0]["amx_pattern"] == "m1n4"
    assert production["experts"][0]["w13_cache_blocks"] == 4
    assert production["experts"][0]["w2_cache_blocks"] == 16

    narrow = _moe_C.fused_moe_test_x86_policy(1024, 256, [2048], 8)
    assert narrow["nsplit_target_rows"] == 256
    assert narrow["experts"][0]["amx_pattern"] == "m2n2"

    wide_f_96 = _moe_C.fused_moe_test_x86_policy(4096, 2048, [96], 1)
    wide_f_128 = _moe_C.fused_moe_test_x86_policy(4096, 2048, [128], 1)
    assert wide_f_96["nsplit_target_rows"] == 16
    assert wide_f_96["experts"][0]["amx_pattern"] == "m2n2"
    assert wide_f_128["experts"][0]["amx_pattern"] == "m1n4"


@requires_amx
def test_dimension_aware_policy_rejects_unknown_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelled profile must not silently select a different policy."""
    from fused_cpp import _moe_C

    monkeypatch.setenv("FUSED_CPP_MOE_X86_POLICY_PROFILE", "not-a-profile")
    with pytest.raises(RuntimeError, match="FUSED_CPP_MOE_X86_POLICY_PROFILE"):
        _moe_C.fused_moe_test_x86_policy(4096, 512, [64], 8)


@requires_amx
def test_auto_k32_weights_execute_with_both_isa_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """One automatic packed-weight copy is correct through AVX-512 and AMX."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=3,
        hidden=65,
        intermediate=17,
        experts=1,
        top_k=1,
        seed=909,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    assert packed.gemm_backend == 104

    monkeypatch.setenv("FUSED_CPP_MOE_X86_POLICY_PROFILE", "intel_06_ad_c8i_v1")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_ISA", "avx512")
    avx = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=4)
    monkeypatch.setenv("FUSED_CPP_MOE_X86_ISA", "amx")
    amx = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=4)

    torch.testing.assert_close(avx.float(), amx.float(), atol=0, rtol=0)
    _assert_bf16_close(avx, fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids))

    monkeypatch.delenv("FUSED_CPP_MOE_X86_ISA")
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_BF16", "0")
    avx_fallback = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=4)
    torch.testing.assert_close(avx_fallback.float(), avx.float(), atol=0, rtol=0)


@requires_amx
@pytest.mark.parametrize("num_threads", [1, 2], ids=["single-core", "dual-core"])
def test_amx_auto_pattern_and_cache_policy_support_mixed_expert_sizes(
    monkeypatch: pytest.MonkeyPatch,
    num_threads: int,
) -> None:
    """The generic fallback may select m2n2 and m1n4 in one invocation."""
    inputs, w13, w2, topk_weights, _ = _case(
        tokens=151,
        hidden=641,
        intermediate=545,
        experts=2,
        top_k=1,
        seed=702,
    )
    topk_ids = torch.tensor([0] * 75 + [1] * 76, dtype=torch.int32).view(151, 1)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    monkeypatch.setenv("FUSED_CPP_MOE_X86_POLICY_PROFILE", "generic_v1")
    monkeypatch.delenv("FUSED_CPP_MOE_AMX_PATTERN", raising=False)
    monkeypatch.delenv("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS", raising=False)
    monkeypatch.delenv("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS", raising=False)
    automatic = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)

    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", "auto")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS", "auto")
    monkeypatch.setenv("FUSED_CPP_MOE_X86_W2_CACHE_BLOCKS", "auto")
    explicit_auto = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    torch.testing.assert_close(explicit_auto.float(), automatic.float(), atol=0, rtol=0)
    _assert_bf16_close(automatic, expected)


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
@pytest.mark.parametrize("skip_weighted", [False, True], ids=["weighted", "direct-bf16"])
@pytest.mark.parametrize("num_threads", [1, 8], ids=["one-thread", "eight-threads"])
def test_amx_aligned_single_active_expert_matches_naive_without_gather(
    skip_weighted: bool,
    num_threads: int,
) -> None:
    """An aligned top-1 hot expert may read the original input without changing results."""
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=64,
        intermediate=37,
        experts=3,
        top_k=1,
        seed=460,
    )
    topk_ids.fill_(2)
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
        num_threads=num_threads,
        skip_weighted=skip_weighted,
        out=output,
    )
    expected = fused_moe_naive(
        inputs,
        w13,
        w2,
        topk_weights,
        topk_ids,
        skip_weighted=skip_weighted,
    )

    assert returned is output
    _assert_bf16_close(output, expected)


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


@requires_amx_n64
@pytest.mark.parametrize(
    ("pattern", "routes"),
    [
        pytest.param("m1n2", 17, id="m1n2"),
        pytest.param("m2n2", 33, id="m2n2"),
        pytest.param("m1n4", 77, id="m1n4"),
    ],
)
@pytest.mark.parametrize("num_threads", [1, 4], ids=["one-thread", "four-threads"])
def test_amx_n64_layout_matches_n32_and_naive_with_cache_isolation(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    routes: int,
    num_threads: int,
) -> None:
    """N32/N64 packed objects may alternate while every AMX pattern preserves tails and N-split ranges."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=routes,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=900 + routes,
    )
    packed_n32 = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    packed_n64 = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16_n64",
    )
    assert packed_n64.gemm_backend == 103
    assert packed_n64.backend_name == "x86_amx_bf16_n64"
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    for packed in (packed_n32, packed_n64, packed_n32, packed_n64):
        actual = fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=num_threads,
        )
        _assert_bf16_close(actual, expected)


@requires_amx
@pytest.mark.parametrize(
    ("pattern", "routes"),
    [
        pytest.param("m1n2", 17, id="m1n2"),
        pytest.param("m2n2", 33, id="m2n2"),
        pytest.param("m1n4", 77, id="m1n4"),
    ],
)
def test_amx_n32_b_load_hints_match_tileloadd_with_cache_isolation(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    routes: int,
) -> None:
    """Every N32 B-load hint must preserve exact output while occupying a distinct JIT cache entry."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=routes,
        hidden=129,
        intermediate=65,
        experts=1,
        top_k=1,
        seed=925 + routes,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    outputs = {}
    for hint in ("tileloadd", "tileloaddt1", "prefetch_t0", "prefetch_t1", "tileloadd"):
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_B_LOAD_HINT", hint)
        outputs[hint] = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)
        _assert_bf16_close(outputs[hint], expected)

    baseline = outputs["tileloadd"]
    for hint in ("tileloaddt1", "prefetch_t0", "prefetch_t1"):
        torch.testing.assert_close(outputs[hint].float(), baseline.float(), atol=0, rtol=0)


@requires_amx
@pytest.mark.parametrize(
    ("hidden", "intermediate"),
    [
        pytest.param(31, 31, id="single-k-block"),
        pytest.param(33, 65, id="w13-even-w2-odd"),
        pytest.param(65, 97, id="w13-odd-w2-even"),
    ],
)
@pytest.mark.parametrize("num_threads", [1, 2], ids=["single-core", "dual-core"])
def test_amx_m1n2_k_load_pipeline_matches_baseline_with_cache_isolation(
    monkeypatch: pytest.MonkeyPatch,
    hidden: int,
    intermediate: int,
    num_threads: int,
) -> None:
    """The m1n2 ping-pong K schedule must preserve K tails, N tails, and exact accumulation order."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", "m1n2")
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=17,
        hidden=hidden,
        intermediate=intermediate,
        experts=1,
        top_k=1,
        seed=1100 + hidden + intermediate,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    outputs = []
    for pipeline in ("baseline", "pipelined", "baseline"):
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_K_LOAD_PIPELINE", pipeline)
        output = fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=num_threads,
        )
        _assert_bf16_close(output, expected)
        outputs.append(output)

    torch.testing.assert_close(outputs[1].float(), outputs[0].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs[2].float(), outputs[0].float(), atol=0, rtol=0)


@requires_amx
@pytest.mark.parametrize(
    ("pattern", "routes"),
    [
        pytest.param("m1n2", 37, id="m1n2-full-panels-and-tail"),
        pytest.param("m2n2", 81, id="m2n2-full-pairs-and-tail"),
        pytest.param("m1n4", 77, id="m1n4-full-panels-and-tail"),
    ],
)
@pytest.mark.parametrize(
    "num_threads",
    [1, 2, 4, 8],
    ids=["one-thread", "two-threads", "four-threads", "eight-threads"],
)
def test_amx_macro_m_tile_state_matches_per_call(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    routes: int,
    num_threads: int,
) -> None:
    """One tile configuration across full M panels must preserve exact BF16 output and tail behavior."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=routes,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=950 + routes,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "per_call")
    per_call = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "macro_m")
    macro_m = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    torch.testing.assert_close(macro_m.float(), per_call.float(), atol=0, rtol=0)
    _assert_bf16_close(macro_m, expected)


@requires_amx
def test_amx_macro_m_tile_state_preserves_direct_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Macro-M must advance route rows correctly when W2 stores BF16 output directly."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", "m1n4")
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=77,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=1027,
    )
    topk_weights.fill_(1.0)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "per_call")
    per_call = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=2,
        skip_weighted=True,
    )
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "macro_m")
    macro_m = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=2,
        skip_weighted=True,
    )

    torch.testing.assert_close(macro_m.float(), per_call.float(), atol=0, rtol=0)


@requires_amx
def test_amx_macro_m_tile_state_advances_tile_store_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Macro-M must advance expert-contiguous FP32 output between full M panels."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", "m1n4")
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_W2_EPILOGUE", "tile_store")
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=77,
        hidden=67,
        intermediate=35,
        experts=2,
        top_k=2,
        seed=1031,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "per_call")
    per_call = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "macro_m")
    macro_m = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=2)
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids)

    torch.testing.assert_close(macro_m.float(), per_call.float(), atol=0, rtol=0)
    _assert_bf16_close(macro_m, expected)


@requires_amx
@pytest.mark.parametrize("pattern", ["m2n2", "m1n4"])
@pytest.mark.parametrize("num_threads", [2, 4, 8])
def test_amx_patterns_hot_multithread_direct_bf16(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    num_threads: int,
) -> None:
    """AMX kernels must support hot-expert N-split and direct BF16 stores."""
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
        num_threads=num_threads,
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

    with pytest.raises(RuntimeError, match="must be auto, m1n2, m2n2, or m1n4"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@requires_amx
def test_amx_rejects_unknown_tile_state_mode(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_TILE_STATE", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, per_call, or macro_m"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@requires_amx
def test_amx_rejects_unknown_b_load_hint(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_B_LOAD_HINT", "unknown")

    with pytest.raises(
        RuntimeError,
        match="must be auto, tileloadd, tileloaddt1, prefetch_t0, or prefetch_t1",
    ):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@requires_amx
def test_amx_rejects_unknown_k_load_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_K_LOAD_PIPELINE", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, baseline, or pipelined"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@requires_amx
@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("pattern", ["m1n2", "m2n2", "m1n4"])
@pytest.mark.parametrize("num_threads", [1, 2], ids=["single-core", "dual-core"])
def test_amx_w13_silu_epilogues_match_baseline(
    monkeypatch: pytest.MonkeyPatch,
    degree: int,
    pattern: str,
    num_threads: int,
) -> None:
    """Resident constants and row pipelining are exact; RCP14 remains within the BF16 contract."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=17,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=800 + degree,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    outputs: dict[str, torch.Tensor] = {}
    for epilogue in ("baseline", "auto", "resident", "pipelined", "rcp14"):
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_SILU_EPILOGUE", epilogue)
        outputs[epilogue] = fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=num_threads,
            silu_poly_degree=degree,
        )

    torch.testing.assert_close(outputs["auto"].float(), outputs["resident"].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs["resident"].float(), outputs["baseline"].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs["pipelined"].float(), outputs["baseline"].float(), atol=0, rtol=0)
    _assert_bf16_close(outputs["rcp14"], outputs["baseline"])


@requires_amx
def test_amx_rejects_unknown_silu_epilogue(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_SILU_EPILOGUE", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, baseline, resident, pipelined, or rcp14"):
        fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=1)


@requires_amx
@pytest.mark.parametrize("pattern", ["m1n2", "m2n2", "m1n4"])
@pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
def test_amx_w2_store_merge_epilogues_match_baseline(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    num_threads: int,
) -> None:
    """Combined stores and expert-contiguous TILESTORED output preserve flat-route merge order."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=67,
        intermediate=35,
        experts=3,
        top_k=2,
        seed=900,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    outputs: dict[str, torch.Tensor] = {}
    for epilogue in ("auto", "baseline", "combined", "tile_store"):
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_W2_EPILOGUE", epilogue)
        outputs[epilogue] = fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=num_threads,
        )

    torch.testing.assert_close(outputs["auto"].float(), outputs["baseline"].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs["combined"].float(), outputs["baseline"].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs["tile_store"].float(), outputs["baseline"].float(), atol=0, rtol=0)
    _assert_bf16_close(outputs["tile_store"], fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids))


@requires_amx
def test_amx_w2_tile_store_keeps_direct_bf16_epilogue(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tile-store selector falls back to the BF16 converting store for top-k=1 direct output."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", "m1n4")
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=901,
    )
    topk_weights.fill_(1.0)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    outputs: dict[str, torch.Tensor] = {}
    for epilogue in ("baseline", "combined", "tile_store"):
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_W2_EPILOGUE", epilogue)
        outputs[epilogue] = fused_moe_bf16_tiled(
            inputs,
            packed,
            topk_weights,
            topk_ids,
            num_threads=2,
            skip_weighted=True,
        )

    torch.testing.assert_close(outputs["combined"].float(), outputs["baseline"].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs["tile_store"].float(), outputs["combined"].float(), atol=0, rtol=0)


@requires_amx
def test_amx_rejects_unknown_w2_epilogue(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_W2_EPILOGUE", "unknown")

    with pytest.raises(RuntimeError, match="must be auto, baseline, combined, or tile_store"):
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
@pytest.mark.parametrize("num_threads", [1, 8], ids=["one-thread", "eight-threads"])
def test_amx_persistent_intermediate_clears_reused_k32_tail(
    monkeypatch: pytest.MonkeyPatch,
    num_threads: int,
) -> None:
    """A smaller F must not consume stale non-finite values from a previously wider persistent buffer."""
    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE", "1")
    wide_inputs, wide_w13, wide_w2, wide_topk_weights, wide_topk_ids = _case(
        tokens=33,
        hidden=64,
        intermediate=64,
        experts=1,
        top_k=1,
        seed=502,
    )
    wide_w13[:, 64:, :].fill_(float("nan"))
    wide_topk_weights.fill_(1.0)
    wide_packed = prepare_fused_moe_bf16_tiled_weights(
        wide_w13,
        wide_w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    fused_moe_bf16_tiled(
        wide_inputs,
        wide_packed,
        wide_topk_weights,
        wide_topk_ids,
        num_threads=num_threads,
        skip_weighted=True,
    )

    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=64,
        intermediate=33,
        experts=1,
        top_k=1,
        seed=503,
    )
    topk_weights.fill_(1.0)
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
        skip_weighted=True,
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids, skip_weighted=True)

    _assert_bf16_close(actual, expected)


@requires_amx
@pytest.mark.parametrize("num_threads", [1, 8], ids=["one-thread", "eight-threads"])
def test_amx_persistent_input_clears_reused_k32_tail(
    monkeypatch: pytest.MonkeyPatch,
    num_threads: int,
) -> None:
    """Gathering a narrower H must overwrite stale pooled values through AMX's observable K32 tail."""
    monkeypatch.setenv("FUSED_CPP_MOE_X86_PERSISTENT_INPUT", "1")
    wide_inputs, wide_w13, wide_w2, wide_topk_weights, wide_topk_ids = _case(
        tokens=34,
        hidden=64,
        intermediate=32,
        experts=2,
        top_k=1,
        seed=507,
    )
    wide_inputs[:, 33:].fill_(float("nan"))
    wide_topk_weights.fill_(1.0)
    wide_packed = prepare_fused_moe_bf16_tiled_weights(
        wide_w13,
        wide_w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )
    fused_moe_bf16_tiled(
        wide_inputs,
        wide_packed,
        wide_topk_weights,
        wide_topk_ids,
        num_threads=num_threads,
        skip_weighted=True,
    )

    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=34,
        hidden=33,
        intermediate=32,
        experts=2,
        top_k=1,
        seed=508,
    )
    topk_weights.fill_(1.0)
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
        skip_weighted=True,
    )
    expected = fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids, skip_weighted=True)

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
@pytest.mark.parametrize("pattern", ["m1n2", "m2n2", "m1n4"])
@pytest.mark.parametrize("num_threads", [1, 8], ids=["one-thread", "eight-threads"])
def test_amx_weighted_top1_direct_matches_route_workspace(
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    num_threads: int,
) -> None:
    """Every AMX M/N pattern scales TILESTORED rows before the direct BF16 conversion."""
    monkeypatch.setenv("FUSED_CPP_MOE_AMX_PATTERN", pattern)
    if pattern == "m1n4":
        # Weighted direct output must override the incompatible final
        # TILESTORED mode while retaining identical values.
        monkeypatch.setenv("FUSED_CPP_MOE_AMX_W2_EPILOGUE", "tile_store")
    inputs, w13, w2, topk_weights, topk_ids = _case(
        tokens=33,
        hidden=67,
        intermediate=35,
        experts=1,
        top_k=1,
        seed=354,
    )
    topk_weights.copy_(torch.linspace(0.0625, 1.0625, inputs.size(0)).view(-1, 1))
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="x86_amx_bf16",
    )

    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "0")
    workspace = fused_moe_bf16_tiled(inputs, packed, topk_weights, topk_ids, num_threads=num_threads)
    output = torch.empty_like(inputs)
    monkeypatch.setenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT", "1")
    returned = fused_moe_bf16_tiled(
        inputs,
        packed,
        topk_weights,
        topk_ids,
        num_threads=num_threads,
        out=output,
    )

    assert returned is output
    torch.testing.assert_close(output.float(), workspace.float(), atol=0, rtol=0)
    _assert_bf16_close(output, fused_moe_naive(inputs, w13, w2, topk_weights, topk_ids))


@requires_amx
def test_amx_runtime_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The AMX kill switch hides both layouts, rejects them explicitly, and restores AVX-512 auto fallback."""
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
    assert "x86_amx_bf16_n64" not in backends
    assert "x86_avx512_bf16" in backends
    automatic = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    assert automatic.gemm_backend == 101
    assert automatic.backend_name == "x86_avx512_bf16"
    for backend in ("x86_amx_bf16", "x86_amx_bf16_n64"):
        with pytest.raises(RuntimeError, match="not supported by this build/runtime"):
            prepare_fused_moe_bf16_tiled_weights(
                w13,
                w2,
                fuse_silu=True,
                backend=backend,
            )
