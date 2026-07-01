# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import platform

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_bf16_tiled_scheduled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not _HAS_BF16_TILED_FUSED_MOE,
    reason="BF16 tiled fused MoE kernel is only available on AArch64",
)


def _bf16_randn(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape) * 0.2).to(torch.bfloat16)


def _bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def _first_affinity_cpu() -> int:
    if hasattr(os, "sched_getaffinity"):
        cpus = os.sched_getaffinity(0)
        if cpus:
            return min(cpus)
    return 0


def _case(seed: int = 0) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    num_tokens = 37
    hidden_size = 13
    ffn_hidden_size = 11
    num_experts = 4
    top_k = 2
    hidden_states = _bf16_randn(num_tokens, hidden_size)
    w13_weight = _bf16_randn(num_experts, 2 * ffn_hidden_size, hidden_size)
    w2_weight = _bf16_randn(num_experts, hidden_size, ffn_hidden_size)
    w13_bias = torch.randn(num_experts, 2 * ffn_hidden_size) * 0.1
    w2_bias = torch.randn(num_experts, hidden_size) * 0.1
    topk_ids = torch.tensor(
        [[(i + j) % num_experts for j in range(top_k)] for i in range(num_tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    return (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    )


def _case_top1(seed: int = 0) -> tuple[torch.Tensor, ...]:
    """Single-route (top_k == 1) case for the skip_weighted fast path."""
    torch.manual_seed(seed)
    num_tokens = 37
    hidden_size = 13
    ffn_hidden_size = 11
    num_experts = 4
    hidden_states = _bf16_randn(num_tokens, hidden_size)
    w13_weight = _bf16_randn(num_experts, 2 * ffn_hidden_size, hidden_size)
    w2_weight = _bf16_randn(num_experts, hidden_size, ffn_hidden_size)
    w13_bias = torch.randn(num_experts, 2 * ffn_hidden_size) * 0.1
    w2_bias = torch.randn(num_experts, hidden_size) * 0.1
    topk_ids = torch.tensor(
        [[i % num_experts] for i in range(num_tokens)], dtype=torch.int32
    )
    topk_weights = torch.ones(num_tokens, 1)
    return (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    )


def test_fused_moe_bf16_tiled_skip_weighted_matches_unit_weighted() -> None:
    # 2b: with top_k == 1 and unit weights, the skip_weighted fast path
    # (w2 result written straight to the bf16 output, no route_out / merge)
    # must bit-match the weighted path. Covers impl A (single + multi thread)
    # and the scheduled split-GEMM path.
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case_top1(seed=5)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    ref = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
    )

    out_a = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
        skip_weighted=True,
    )
    assert out_a.dtype == torch.bfloat16
    torch.testing.assert_close(out_a.float(), ref.float(), atol=0, rtol=0)

    out_a_mt = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=3,
        skip_weighted=True,
    )
    torch.testing.assert_close(out_a_mt.float(), ref.float(), atol=0, rtol=0)

    out_sched = fused_moe_bf16_tiled_scheduled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        wave_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
        team_expert_ids=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        team_threads=torch.tensor([2, 1, 2, 1], dtype=torch.int32),
        thread_cpu_ids=torch.tensor([0, 1, 2], dtype=torch.int32),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=3,
        skip_weighted=True,
    )
    torch.testing.assert_close(out_sched.float(), ref.float(), atol=0, rtol=0)


@pytest.mark.parametrize("activation", ["silu", "swigluoai"])
def test_fused_moe_bf16_tiled_matches_naive(activation: str) -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=123)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    out = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
        activation=activation,
    )
    ref = fused_moe_naive(
        hidden_states.float(),
        w13_weight.float(),
        w2_weight.float(),
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        activation=activation,
    ).to(torch.bfloat16)

    assert out.dtype == torch.bfloat16
    assert out.shape == hidden_states.shape
    torch.testing.assert_close(out.float(), ref.float(), atol=7e-2, rtol=7e-2)


def test_fused_moe_bf16_tiled_threaded_matches_single_thread() -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=7)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    serial = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
    )
    threaded = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=3,
    )

    torch.testing.assert_close(threaded.float(), serial.float(), atol=0, rtol=0)


def test_fused_moe_bf16_tiled_scheduled_matches_single_thread() -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=19)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    serial = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
    )
    scheduled = fused_moe_bf16_tiled_scheduled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        wave_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
        team_expert_ids=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        team_threads=torch.tensor([2, 1, 2, 1], dtype=torch.int32),
        thread_cpu_ids=torch.tensor([0, 1, 2], dtype=torch.int32),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=3,
    )

    torch.testing.assert_close(scheduled.float(), serial.float(), atol=0, rtol=0)


def test_fused_moe_bf16_tiled_scheduled_rejects_bad_thread_cpu_ids() -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=29)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    with pytest.raises(ValueError, match="thread_cpu_ids"):
        fused_moe_bf16_tiled_scheduled(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            wave_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
            team_expert_ids=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            team_threads=torch.tensor([2, 1, 2, 1], dtype=torch.int32),
            thread_cpu_ids=torch.tensor([0, 1], dtype=torch.int32),
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            num_threads=3,
        )


def test_fused_moe_bf16_tiled_scheduled_m_split_matches_single_thread() -> None:
    torch.manual_seed(23)
    num_tokens = 32
    hidden_size = 8
    ffn_hidden_size = 3
    num_experts = 2
    hidden_states = _bf16_randn(num_tokens, hidden_size)
    w13_weight = _bf16_randn(num_experts, 2 * ffn_hidden_size, hidden_size)
    w2_weight = _bf16_randn(num_experts, hidden_size, ffn_hidden_size)
    w13_bias = torch.randn(num_experts, 2 * ffn_hidden_size) * 0.1
    w2_bias = torch.randn(num_experts, hidden_size) * 0.1
    topk_ids = torch.tensor([[i % num_experts] for i in range(num_tokens)])
    topk_weights = torch.ones(num_tokens, 1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    serial = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
    )
    scheduled = fused_moe_bf16_tiled_scheduled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        wave_offsets=torch.tensor([0, 2], dtype=torch.int32),
        team_expert_ids=torch.tensor([0, 1], dtype=torch.int32),
        team_threads=torch.tensor([2, 2], dtype=torch.int32),
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=4,
    )

    torch.testing.assert_close(scheduled.float(), serial.float(), atol=0, rtol=0)


def test_prepare_fused_moe_bf16_tiled_weights_prepack_threads_matches_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _hidden_states,
        w13_weight,
        w2_weight,
        _w13_bias,
        _w2_bias,
        _topk_weights,
        _topk_ids,
    ) = _case(seed=13)

    monkeypatch.setenv("FUSED_CPP_MOE_PREPACK_THREADS", "1")
    serial = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    monkeypatch.setenv("FUSED_CPP_MOE_PREPACK_THREADS", "3")
    threaded = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    assert threaded.w13[1:] == serial.w13[1:]
    assert threaded.w2[1:] == serial.w2[1:]
    torch.testing.assert_close(threaded.w13[0], serial.w13[0], atol=0, rtol=0)
    torch.testing.assert_close(threaded.w2[0], serial.w2[0], atol=0, rtol=0)


def test_fused_moe_bf16_tiled_hierarchical_core_skip_is_relative(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=17)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    monkeypatch.setenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", "2")
    monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_SCHEDULE_DEBUG", "1")
    monkeypatch.delenv("FUSED_CPP_MOE_N_SPLIT_CORE_BASES", raising=False)

    hierarchical = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=4,
    )
    captured = capfd.readouterr()

    monkeypatch.setenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SCHEDULE_DEBUG", "0")
    serial = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
        num_threads=1,
    )

    first_cpu = _first_affinity_cpu()
    assert (
        f"core_bases=[{first_cpu},{first_cpu + 2}]"
        in captured.err
    )
    torch.testing.assert_close(hierarchical.float(), serial.float(), atol=0, rtol=0)


def test_fused_moe_bf16_tiled_out_buffer() -> None:
    (
        hidden_states,
        w13_weight,
        w2_weight,
        _w13_bias,
        _w2_bias,
        topk_weights,
        topk_ids,
    ) = _case(seed=11)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)
    out_buffer = torch.empty_like(hidden_states)

    ret = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        num_threads=2,
        out=out_buffer,
    )

    assert ret is out_buffer
    ref = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
    )
    torch.testing.assert_close(out_buffer.float(), ref.float(), atol=0, rtol=0)


@pytest.mark.slow
def test_fused_moe_bf16_tiled_deepseek_v4_tp4_rank_shape_smoke() -> None:
    """DeepSeek V4 Flash TP=4 rank shape smoke with reduced experts."""
    generator = torch.Generator().manual_seed(20260218)
    hidden_size = 4096
    ffn_hidden_size_per_rank = 2048 // 4
    num_experts = 20
    top_k = 6
    total_tokens = 2048

    hidden_states = _bf16_normal(
        (total_tokens, hidden_size),
        generator=generator,
        std=0.01,
    )
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size_per_rank, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size_per_rank),
        generator=generator,
        std=0.01,
    )
    routing_scores = torch.rand(total_tokens, num_experts, generator=generator)
    topk_weights, topk_ids = torch.topk(routing_scores, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)
    del w13_weight, w2_weight

    out = fused_moe_bf16_tiled(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        num_threads=1,
        activation="silu",
    )

    assert out.shape == hidden_states.shape
    assert out.dtype == torch.bfloat16
