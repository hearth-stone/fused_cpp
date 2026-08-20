from __future__ import annotations

import os

import pytest
import torch

from fused_cpp.moe.bf16_tiled import (
    PreparedW8A16TiledFusedMoEWeights,
    available_fused_moe_bf16_tiled_backends,
    fused_moe_bf16_tiled_async_plan,
    fused_moe_w8a16_tiled_async_plan,
    fused_moe_w8a16_tiled_with_shared,
    prepare_fused_moe_bf16_tiled_weights,
    prepare_fused_moe_w8a16_tiled_quantized_weights,
    prepare_fused_moe_w8a16_tiled_weights,
    prepare_routed_shared_moe_w8a16_tiled_quantized_weights,
)
from fused_cpp.moe.plan import AsyncMoEPlanV2, upgrade_legacy_async_plan


def _bf16_normal(shape: tuple[int, ...], generator: torch.Generator, std: float) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * std).to(torch.bfloat16)


def _quantize_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
    quantized = torch.round(weight.float() / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def test_prequantized_w8a16_pack_matches_bf16_source_pack() -> None:
    """Packing checkpoint INT8+scale must preserve the existing W8A16 opaque layout."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    generator = torch.Generator().manual_seed(20260818)
    w13 = _bf16_normal((2, 64, 64), generator, 0.05)
    w2 = _bf16_normal((2, 64, 32), generator, 0.05)
    q13, s13 = _quantize_per_output_channel(w13)
    q2, s2 = _quantize_per_output_channel(w2)

    reference = prepare_fused_moe_w8a16_tiled_weights(w13, w2)
    actual = prepare_fused_moe_w8a16_tiled_quantized_weights(q13, s13, q2, s2)

    torch.testing.assert_close(actual.w13[0], reference.w13[0], atol=0, rtol=0)
    torch.testing.assert_close(actual.w13[3], reference.w13[3], atol=0, rtol=0)
    torch.testing.assert_close(actual.w2[0], reference.w2[0], atol=0, rtol=0)
    torch.testing.assert_close(actual.w2[3], reference.w2[3], atol=0, rtol=0)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires Linux CPU affinity")
def test_prequantized_w8a16_routed_shared_clamp_matches_synthetic_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The W8A16 shared wrapper must append one unit route and preserve clamp execution."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    cpu_ids = sorted(os.sched_getaffinity(0))[:4]
    if len(cpu_ids) < 4:
        pytest.skip("requires four available CPUs")
    generator = torch.Generator().manual_seed(20260819)
    tokens = 13
    hidden_size = 64
    intermediate_size = 32
    hidden = _bf16_normal((tokens, hidden_size), generator, 0.1)
    routed_w13 = _bf16_normal((2, 2 * intermediate_size, hidden_size), generator, 0.05)
    routed_w2 = _bf16_normal((2, hidden_size, intermediate_size), generator, 0.05)
    shared_w13 = _bf16_normal((2 * intermediate_size, hidden_size), generator, 0.05)
    shared_w2 = _bf16_normal((hidden_size, intermediate_size), generator, 0.05)
    routed_q13, routed_s13 = _quantize_per_output_channel(routed_w13)
    routed_q2, routed_s2 = _quantize_per_output_channel(routed_w2)
    shared_q13, shared_s13 = _quantize_per_output_channel(shared_w13)
    shared_q2, shared_s2 = _quantize_per_output_channel(shared_w2)
    weights = prepare_routed_shared_moe_w8a16_tiled_quantized_weights(
        routed_q13,
        routed_s13,
        routed_q2,
        routed_s2,
        shared_q13,
        shared_s13,
        shared_q2,
        shared_s2,
    )
    topk_ids = torch.arange(tokens, dtype=torch.int32).remainder(2).reshape(tokens, 1)
    topk_weights = torch.linspace(0.25, 0.75, tokens).reshape(tokens, 1)
    combined_ids = torch.cat((topk_ids, torch.full((tokens, 1), 2, dtype=torch.int32)), dim=1)
    combined_weights = torch.cat((topk_weights * 2.5, torch.ones((tokens, 1))), dim=1)
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 4,
            "thread_cpu_ids": cpu_ids,
            "task_expert_ids": [0, 1, 2],
            "task_core_begins": [0, 1, 2],
            "task_threads": [1, 1, 2],
            "task_dep_offsets": [0, 0, 0, 0],
            "task_deps": [],
        }
    )
    bridge["early_merge"] = False
    bridge["task_w13_window_tiles"] = [1, 1, 1]
    bridge["task_w2_window_tiles"] = [1, 1, 1]
    plan = AsyncMoEPlanV2.from_dict(bridge)

    class _SharedPlanner:
        def plan_for_shared_dispatch(self, *args, **kwargs):
            return plan

    monkeypatch.setattr(
        "fused_cpp.moe.planner_runtime.get_default_moe_planner_runtime",
        lambda: _SharedPlanner(),
    )
    actual = fused_moe_w8a16_tiled_with_shared(
        hidden,
        weights,
        topk_weights,
        topk_ids,
        num_threads=4,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
    )
    reference = fused_moe_w8a16_tiled_async_plan(
        hidden,
        weights.packed,
        combined_weights,
        combined_ids,
        plan,
        swiglu_limit=10.0,
    )
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires Linux CPU affinity")
def test_w8a16_plan_v2_tracks_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    cpu_ids = sorted(os.sched_getaffinity(0))[:4]
    if len(cpu_ids) < 4:
        pytest.skip("requires four available CPUs")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_DIRECT_ROUTE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1")
    generator = torch.Generator().manual_seed(20260817)
    hidden = 64
    intermediate = 32
    route_counts = [24, 12]
    tokens = sum(route_counts)
    hidden_states = _bf16_normal((tokens, hidden), generator, 0.1)
    w13 = _bf16_normal((2, 2 * intermediate, hidden), generator, 0.05)
    w2 = _bf16_normal((2, hidden, intermediate), generator, 0.05)
    topk_ids = torch.cat(
        [torch.full((count,), expert, dtype=torch.int32) for expert, count in enumerate(route_counts)]
    ).reshape(tokens, 1)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32)
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 4,
            "thread_cpu_ids": cpu_ids,
            "task_expert_ids": [0, 1],
            "task_core_begins": [0, 2],
            "task_threads": [2, 2],
            "task_dep_offsets": [0, 0, 0],
            "task_deps": [],
        }
    )
    bridge["early_merge"] = False
    bridge["task_w13_window_tiles"] = [1, 1]
    bridge["task_w2_window_tiles"] = [1, 1]
    plan = AsyncMoEPlanV2.from_dict(bridge)
    bf16_weights = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    w8_weights = prepare_fused_moe_w8a16_tiled_weights(w13, w2)

    reference = fused_moe_bf16_tiled_async_plan(hidden_states, bf16_weights, topk_weights, topk_ids, plan)
    actual = fused_moe_w8a16_tiled_async_plan(hidden_states, w8_weights, topk_weights, topk_ids, plan)
    cached = fused_moe_w8a16_tiled_async_plan(
        hidden_states,
        w8_weights,
        topk_weights,
        topk_ids,
        plan,
        cache_dequant=True,
    )

    torch.testing.assert_close(actual.float(), reference.float(), atol=2.0e-3, rtol=3.0e-2)
    torch.testing.assert_close(cached.float(), reference.float(), atol=2.0e-3, rtol=3.0e-2)
    assert w8_weights.w13[0].dtype == torch.int8
    assert w8_weights.w2[0].dtype == torch.int8
    assert w8_weights.w13[0].numel() == bf16_weights.w13[0].numel()
    assert w8_weights.w13[0].element_size() * 2 == bf16_weights.w13[0].element_size()
    assert w8_weights.w13[3].shape == (2, 4 * w8_weights.w13[2])
    assert w8_weights.w2[3].shape == (2, 4 * w8_weights.w2[2])

    malformed = PreparedW8A16TiledFusedMoEWeights(
        w13=(*w8_weights.w13[:3], w8_weights.w13[3][:, :-1].contiguous()),
        w2=w8_weights.w2,
        backend_n_tile=w8_weights.backend_n_tile,
    )
    with pytest.raises(RuntimeError, match="scale shape mismatch"):
        fused_moe_w8a16_tiled_async_plan(hidden_states, malformed, topk_weights, topk_ids, plan)
