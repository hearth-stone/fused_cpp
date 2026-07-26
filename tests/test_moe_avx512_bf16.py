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
def test_amx_backend_is_preferred_by_auto_with_explicit_avx512_fallback() -> None:
    """Automatic x86 preparation prefers AMX while explicit AVX-512 remains available."""
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

    assert automatic.gemm_backend == 102
    assert automatic.backend_name == "x86_amx_bf16"
    assert amx.gemm_backend == 102
    assert amx.backend_n_tile == 32
    assert amx.backend_name == "x86_amx_bf16"
    assert amx.w13[0].shape[1] % 32 == 0
    assert amx.w2[0].shape[1] % 32 == 0
    assert avx512.gemm_backend == 101
    assert avx512.backend_name == "x86_avx512_bf16"


@requires_amx
@pytest.mark.parametrize("num_threads", [1, 2], ids=["single-core", "dual-core"])
def test_amx_auto_pattern_and_cache_policy_support_mixed_expert_sizes(
    monkeypatch: pytest.MonkeyPatch,
    num_threads: int,
) -> None:
    """Auto policy may select m2n2 and m1n4 in one invocation without relying on environment overrides."""
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
@pytest.mark.parametrize(
    ("pattern", "routes"),
    [
        pytest.param("m1n2", 37, id="m1n2-full-panels-and-tail"),
        pytest.param("m2n2", 81, id="m2n2-full-pairs-and-tail"),
        pytest.param("m1n4", 77, id="m1n4-full-panels-and-tail"),
    ],
)
@pytest.mark.parametrize("num_threads", [1, 8], ids=["one-thread", "eight-threads"])
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
    """The AMX kill switch hides backend 102, rejects it explicitly, and restores AVX-512 auto fallback."""
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
    automatic = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    assert automatic.gemm_backend == 101
    assert automatic.backend_name == "x86_avx512_bf16"
    with pytest.raises(RuntimeError, match="not supported by this build/runtime"):
        prepare_fused_moe_bf16_tiled_weights(
            w13,
            w2,
            fuse_silu=True,
            backend="x86_amx_bf16",
        )
