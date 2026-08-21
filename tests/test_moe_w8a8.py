from __future__ import annotations

import os

import pytest
import torch

from fused_cpp.moe import AsyncMoEPlanV2, upgrade_legacy_async_plan
from fused_cpp.moe import bf16_tiled as module
from fused_cpp.moe.bf16_tiled import (
    PreparedW8A8TiledFusedMoEWeights,
    PreparedW8A8TiledRoutedSharedMoEWeights,
    fused_moe_w8a8_tiled_with_shared,
    fused_moe_w8a8_tiled_async_plan,
    prepare_fused_moe_w8a8_tiled_weights,
    prepare_fused_moe_w8a8_tiled_quantized_weights,
    prepare_routed_shared_moe_w8a8_tiled_quantized_weights,
)


def _strict_plan(experts: int, team_width: int) -> AsyncMoEPlanV2:
    tasks = list(range(experts))
    cpu_ids = sorted(os.sched_getaffinity(0))[: experts * team_width]
    if len(cpu_ids) != experts * team_width:
        pytest.skip("not enough CPUs for the requested W8A8 test plan")
    raw = upgrade_legacy_async_plan(
        {
            "num_threads": experts * team_width,
            "thread_cpu_ids": cpu_ids,
            "task_expert_ids": tasks,
            "task_core_begins": [task * team_width for task in tasks],
            "task_threads": [team_width] * experts,
            "task_dep_offsets": [0] * (experts + 1),
            "task_deps": [],
        }
    )
    return AsyncMoEPlanV2.from_dict(raw)


def _quantize_row(row: torch.Tensor) -> tuple[torch.Tensor, float]:
    row = row.float()
    scale = max(float(row.abs().max()) / 127.0, 1.0e-30)
    return (row / scale).round().clamp(-127, 127).to(torch.int32), scale


def _reference(
    input: torch.Tensor,
    q13: torch.Tensor,
    s13: torch.Tensor,
    q2: torch.Tensor,
    s2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden = input.shape
    intermediate = q2.size(2)
    output = torch.zeros((tokens, hidden), dtype=torch.float32)
    for token in range(tokens):
        input_q, input_scale = _quantize_row(input[token])
        for slot in range(topk_ids.size(1)):
            expert = int(topk_ids[token, slot])
            w13 = (input_q @ q13[expert].to(torch.int32).T).float() * input_scale * s13[expert]
            gate = w13[:intermediate].clamp(max=10.0)
            up = w13[intermediate:].clamp(-10.0, 10.0)
            activated = (gate * up / (1.0 + torch.exp(-gate))).to(torch.bfloat16)
            intermediate_q, intermediate_scale = _quantize_row(activated)
            route = (intermediate_q @ q2[expert].to(torch.int32).T).float() * intermediate_scale * s2[expert]
            output[token].add_(route, alpha=float(topk_weights[token, slot]))
    return output.to(torch.bfloat16)


def test_w8a8_prepared_weight_contract_is_explicit() -> None:
    tensor = torch.empty((1, 256), dtype=torch.int8)
    scales = torch.ones((1, 16), dtype=torch.float32)
    weights = PreparedW8A8TiledFusedMoEWeights(
        w13=(tensor, 16, 16, scales),
        w2=(tensor, 16, 16, scales),
        backend_n_tile=16,
    )
    assert weights.fused_silu
    assert weights.gemm_backend == 1
    assert weights.backend_name == "arm_sve_w8a8_i8mm"


def test_fused_moe_tiled_dispatches_explicit_w8a8_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    tensor = torch.empty((1, 256), dtype=torch.int8)
    scales = torch.ones((1, 16), dtype=torch.float32)
    weights = PreparedW8A8TiledFusedMoEWeights(
        w13=(tensor, 16, 16, scales),
        w2=(tensor, 16, 16, scales),
        backend_n_tile=16,
    )
    expected = torch.zeros((1, 16), dtype=torch.bfloat16)
    monkeypatch.setattr(module, "fused_moe_w8a8_tiled", lambda *args, **kwargs: expected)
    actual = module.fused_moe_tiled(
        expected,
        weights,
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int64),
        swiglu_limit=10.0,
    )
    assert actual is expected


def test_w8a8_routed_shared_wrapper_appends_one_all_token_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.empty((3, 256), dtype=torch.int8)
    scales = torch.ones((3, 64), dtype=torch.float32)
    packed = PreparedW8A8TiledFusedMoEWeights(
        w13=(tensor, 16, 32, scales),
        w2=(tensor, 16, 16, scales),
        backend_n_tile=16,
    )
    weights = PreparedW8A8TiledRoutedSharedMoEWeights(
        packed=packed,
        routed_experts=2,
        shared_expert_id=2,
    )
    captured: dict[str, object] = {}
    expected = torch.empty((2, 16), dtype=torch.bfloat16)

    class FakeRuntime:
        def plan_for_shared_dispatch(self, _weights, combined_ids, **kwargs):
            captured["planned_ids"] = combined_ids.clone()
            captured["plan_kwargs"] = kwargs
            return object()

    def fake_execute(_input, _packed, combined_weights, combined_ids, _plan, **kwargs):
        captured["executed_ids"] = combined_ids.clone()
        captured["executed_weights"] = combined_weights.clone()
        captured["execute_kwargs"] = kwargs
        return expected

    monkeypatch.setattr(module, "_require_backend", lambda: None)
    monkeypatch.setattr(
        "fused_cpp.moe.planner_runtime.get_default_moe_planner_runtime",
        lambda: FakeRuntime(),
    )
    monkeypatch.setattr(module, "fused_moe_w8a8_tiled_async_plan", fake_execute)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)

    actual = fused_moe_w8a8_tiled_with_shared(
        torch.zeros((2, 16), dtype=torch.bfloat16),
        weights,
        topk_weights,
        topk_ids,
        num_threads=4,
        swiglu_limit=10.0,
        out=expected,
    )

    assert actual is expected
    expected_ids = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.int64)
    expected_weights = torch.tensor(
        [[0.7, 0.3, 1.0], [0.4, 0.6, 1.0]], dtype=torch.float32
    )
    assert torch.equal(captured["planned_ids"], expected_ids)
    assert torch.equal(captured["executed_ids"], expected_ids)
    torch.testing.assert_close(captured["executed_weights"], expected_weights)
    assert captured["execute_kwargs"]["global_num_experts"] == -1


@pytest.mark.skipif(
    not module._HAS_W8A8_TILED_FUSED_MOE,
    reason="ARM SVE i8mm W8A8 extension is unavailable",
)
def test_w8a8_quantized_routed_shared_prepare_appends_shared_expert() -> None:
    generator = torch.Generator().manual_seed(31)
    routed_w13 = torch.randint(-12, 13, (2, 32, 16), dtype=torch.int8, generator=generator)
    routed_w2 = torch.randint(-12, 13, (2, 16, 16), dtype=torch.int8, generator=generator)
    shared_w13 = torch.randint(-12, 13, (32, 16), dtype=torch.int8, generator=generator)
    shared_w2 = torch.randint(-12, 13, (16, 16), dtype=torch.int8, generator=generator)
    routed_s13 = torch.ones((2, 32, 1), dtype=torch.float32)
    routed_s2 = torch.ones((2, 16, 1), dtype=torch.float32)
    shared_s13 = torch.ones((32, 1), dtype=torch.float32)
    shared_s2 = torch.ones((16, 1), dtype=torch.float32)

    weights = prepare_routed_shared_moe_w8a8_tiled_quantized_weights(
        routed_w13,
        routed_s13,
        routed_w2,
        routed_s2,
        shared_w13,
        shared_s13,
        shared_w2,
        shared_s2,
    )

    assert weights.routed_experts == 2
    assert weights.shared_expert_id == 2
    assert weights.packed.w13[0].shape[0] == 3
    assert weights.packed.w2[0].shape[0] == 3


@pytest.mark.skipif(
    not module._HAS_W8A8_TILED_FUSED_MOE,
    reason="ARM SVE i8mm W8A8 extension is unavailable",
)
def test_w8a8_plan_v2_matches_dynamic_quantization_reference() -> None:
    generator = torch.Generator().manual_seed(23)
    experts, hidden, intermediate, tokens = 2, 64, 64, 48
    q13 = torch.randint(-12, 13, (experts, 2 * intermediate, hidden), dtype=torch.int8, generator=generator)
    q2 = torch.randint(-12, 13, (experts, hidden, intermediate), dtype=torch.int8, generator=generator)
    s13 = torch.rand((experts, 2 * intermediate), generator=generator) * 0.003 + 0.0005
    s2 = torch.rand((experts, hidden), generator=generator) * 0.003 + 0.0005
    input = (torch.randn((tokens, hidden), generator=generator) * 0.3).to(torch.bfloat16)
    topk_ids = torch.tensor([[token % experts] for token in range(tokens)], dtype=torch.int64)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32)

    weights = prepare_fused_moe_w8a8_tiled_quantized_weights(q13, s13, q2, s2)
    plan = _strict_plan(experts, team_width=2)
    actual = fused_moe_w8a8_tiled_async_plan(input, weights, topk_weights, topk_ids, plan)
    repeated = fused_moe_w8a8_tiled_async_plan(input, weights, topk_weights, topk_ids, plan)
    expected = _reference(input, q13, s13, q2, s2, topk_weights, topk_ids)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.0, atol=3.0e-6)
    assert torch.equal(actual, repeated)


@pytest.mark.skipif(
    not module._HAS_W8A8_TILED_FUSED_MOE,
    reason="ARM SVE i8mm W8A8 extension is unavailable",
)
def test_w8a8_bf16_prepare_matches_explicit_channel_quantization() -> None:
    generator = torch.Generator().manual_seed(29)
    w13 = (torch.randn((2, 32, 16), generator=generator) * 0.1).to(torch.bfloat16)
    w2 = (torch.randn((2, 16, 16), generator=generator) * 0.1).to(torch.bfloat16)

    def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fp32 = weight.float()
        maximum = fp32.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        return (fp32 / scale).round().clamp(-127, 127).to(torch.int8), scale.squeeze(-1)

    q13, s13 = quantize(w13)
    q2, s2 = quantize(w2)
    automatic = prepare_fused_moe_w8a8_tiled_weights(w13, w2)
    explicit = prepare_fused_moe_w8a8_tiled_quantized_weights(q13, s13, q2, s2)
    assert torch.equal(automatic.w13[0], explicit.w13[0])
    assert torch.equal(automatic.w13[3], explicit.w13[3])
    assert torch.equal(automatic.w2[0], explicit.w2[0])
    assert torch.equal(automatic.w2[3], explicit.w2[3])


@pytest.mark.skipif(
    not module._HAS_W8A8_TILED_FUSED_MOE,
    reason="ARM SVE i8mm W8A8 extension is unavailable",
)
def test_w8a8_rejects_unclamped_activation() -> None:
    tensor = torch.empty((1, 256), dtype=torch.int8)
    scales = torch.ones((1, 16), dtype=torch.float32)
    weights = PreparedW8A8TiledFusedMoEWeights(
        w13=(tensor, 16, 16, scales),
        w2=(tensor, 16, 16, scales),
        backend_n_tile=16,
    )
    with pytest.raises(ValueError, match="swiglu_limit=10.0"):
        fused_moe_w8a8_tiled_async_plan(
            torch.zeros((1, 16), dtype=torch.bfloat16),
            weights,
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int64),
            _strict_plan(1, team_width=1),
            swiglu_limit=0.0,
        )
