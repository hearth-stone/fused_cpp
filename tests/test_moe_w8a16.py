from __future__ import annotations

import os

import pytest
import torch

from fused_cpp.moe.bf16_tiled import (
    PreparedW8A16TiledFusedMoEWeights,
    available_fused_moe_bf16_tiled_backends,
    fused_moe_bf16_tiled_async_plan,
    fused_moe_w8a16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
    prepare_fused_moe_w8a16_tiled_weights,
)
from fused_cpp.moe.plan import AsyncMoEPlanV2, upgrade_legacy_async_plan


def _bf16_normal(shape: tuple[int, ...], generator: torch.Generator, std: float) -> torch.Tensor:
    return (torch.randn(shape, generator=generator) * std).to(torch.bfloat16)


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
