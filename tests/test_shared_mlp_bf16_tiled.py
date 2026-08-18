# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import available_fused_moe_bf16_tiled_backends
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_bf16_tiled_with_shared
from fused_cpp.moe import fused_moe_bf16_tiled_vllm_staged
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights
from fused_cpp.moe import prepare_routed_shared_moe_bf16_tiled_weights
from fused_cpp.moe import prepare_shared_mlp_bf16_tiled_weights
from fused_cpp.moe import set_default_moe_planner_runtime
from fused_cpp.moe import shared_mlp_bf16_tiled

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64")
    or not _HAS_BF16_TILED_FUSED_MOE
    or "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends(),
    reason="BF16 tiled shared MLP is only available on AArch64 SVE",
)


def _cpu_ids(count: int) -> torch.Tensor:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(count))
    if len(affinity) < count:
        pytest.skip(f"requires {count} available CPUs")
    return torch.tensor(affinity[:count], dtype=torch.int32)


@pytest.mark.parametrize("rows", [1, 13, 97])
def test_shared_mlp_matches_one_expert_staged_moe(rows: int) -> None:
    generator = torch.Generator().manual_seed(20260816 + rows)
    hidden_size = 192
    intermediate_size = 96
    num_threads = 8
    hidden = torch.empty((rows, hidden_size), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    w13 = torch.empty((2 * intermediate_size, hidden_size), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    w2 = torch.empty((hidden_size, intermediate_size), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    packed = prepare_shared_mlp_bf16_tiled_weights(w13, w2)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    cpu_ids = _cpu_ids(num_threads)
    reference = fused_moe_bf16_tiled_vllm_staged(
        hidden,
        packed,
        torch.ones((rows, 1), dtype=torch.float32),
        torch.zeros((rows, 1), dtype=torch.int32),
        thread_cpu_ids=cpu_ids,
        num_threads=num_threads,
    )
    candidate = shared_mlp_bf16_tiled(
        hidden,
        packed,
        thread_cpu_ids=cpu_ids,
        num_threads=num_threads,
    )

    torch.testing.assert_close(candidate, reference, atol=0, rtol=0)


def test_shared_mlp_out_and_empty_input() -> None:
    hidden_size = 192
    intermediate_size = 96
    w13 = torch.zeros((2 * intermediate_size, hidden_size), dtype=torch.bfloat16)
    w2 = torch.zeros((hidden_size, intermediate_size), dtype=torch.bfloat16)
    packed = prepare_shared_mlp_bf16_tiled_weights(w13, w2)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    hidden = torch.empty((0, hidden_size), dtype=torch.bfloat16)
    out = torch.empty_like(hidden)
    result = shared_mlp_bf16_tiled(hidden, packed, num_threads=1, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert result.shape == hidden.shape


def test_shared_mlp_rejects_multiple_experts() -> None:
    hidden_size = 192
    intermediate_size = 96
    w13 = torch.zeros((2, 2 * intermediate_size, hidden_size), dtype=torch.bfloat16)
    w2 = torch.zeros((2, hidden_size, intermediate_size), dtype=torch.bfloat16)

    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
    with pytest.raises(ValueError, match="exactly one packed expert"):
        shared_mlp_bf16_tiled(torch.zeros((1, hidden_size), dtype=torch.bfloat16), packed)


def test_routed_shared_packing_matches_explicit_combined_tensor() -> None:
    """Direct E+1 packing must match the copy-heavy reference packing bitwise."""
    generator = torch.Generator().manual_seed(20260817)
    routed_experts = 2
    hidden_size = 192
    intermediate_size = 96
    routed_w13 = torch.randn(
        (routed_experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    routed_w2 = torch.randn(
        (routed_experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    shared_w13 = torch.randn(
        (2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    shared_w2 = torch.randn(
        (hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        generator=generator,
    )

    actual = prepare_routed_shared_moe_bf16_tiled_weights(
        routed_w13,
        routed_w2,
        shared_w13,
        shared_w2,
    )
    reference = prepare_fused_moe_bf16_tiled_weights(
        torch.cat((routed_w13, shared_w13.unsqueeze(0)), dim=0),
        torch.cat((routed_w2, shared_w2.unsqueeze(0)), dim=0),
        fuse_silu=True,
        backend="arm_sve_bf16",
    )

    torch.testing.assert_close(actual.packed.w13[0], reference.w13[0], atol=0, rtol=0)
    torch.testing.assert_close(actual.packed.w2[0], reference.w2[0], atol=0, rtol=0)
    assert actual.routed_experts == routed_experts
    assert actual.shared_expert_id == routed_experts


def test_routed_shared_execution_matches_explicit_synthetic_route() -> None:
    """The public combined API must implement one unit-weight shared route per token."""
    generator = torch.Generator().manual_seed(20260818)
    tokens = 13
    routed_experts = 2
    hidden_size = 192
    intermediate_size = 96
    hidden = torch.randn((tokens, hidden_size), dtype=torch.bfloat16, generator=generator)
    routed_w13 = torch.randn(
        (routed_experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    routed_w2 = torch.randn(
        (routed_experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    shared_w13 = torch.randn(
        (2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    shared_w2 = torch.randn(
        (hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        generator=generator,
    )
    packed = prepare_routed_shared_moe_bf16_tiled_weights(
        routed_w13,
        routed_w2,
        shared_w13,
        shared_w2,
    )
    topk_ids = torch.arange(tokens, dtype=torch.int32).remainder(routed_experts).reshape(tokens, 1)
    topk_weights = torch.linspace(0.25, 0.75, tokens).reshape(tokens, 1)
    combined_ids = torch.cat(
        (topk_ids, torch.full((tokens, 1), routed_experts, dtype=torch.int32)),
        dim=1,
    )
    combined_weights = torch.cat((topk_weights * 2.5, torch.ones((tokens, 1))), dim=1)
    previous = set_default_moe_planner_runtime(None)
    try:
        actual = fused_moe_bf16_tiled_with_shared(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            num_threads=8,
            routed_scaling_factor=2.5,
        )
        reference = fused_moe_bf16_tiled(
            hidden,
            packed.packed,
            combined_weights,
            combined_ids,
            num_threads=8,
            activation="silu",
            global_num_experts=routed_experts + 1,
        )
        actual_clamped = fused_moe_bf16_tiled_with_shared(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            num_threads=8,
            routed_scaling_factor=2.5,
            swiglu_limit=10.0,
        )
        reference_clamped = fused_moe_bf16_tiled(
            hidden,
            packed.packed,
            combined_weights,
            combined_ids,
            num_threads=8,
            activation="silu",
            global_num_experts=routed_experts + 1,
            swiglu_limit=10.0,
        )
    finally:
        set_default_moe_planner_runtime(previous)

    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_clamped, reference_clamped, atol=0, rtol=0)


def test_routed_shared_rejects_mismatched_intermediate_size() -> None:
    hidden_size = 192
    routed_intermediate = 96
    shared_intermediate = 104
    with pytest.raises(RuntimeError, match="shared W13 shape must match"):
        prepare_routed_shared_moe_bf16_tiled_weights(
            torch.zeros((2, 2 * routed_intermediate, hidden_size), dtype=torch.bfloat16),
            torch.zeros((2, hidden_size, routed_intermediate), dtype=torch.bfloat16),
            torch.zeros((2 * shared_intermediate, hidden_size), dtype=torch.bfloat16),
            torch.zeros((hidden_size, shared_intermediate), dtype=torch.bfloat16),
        )
