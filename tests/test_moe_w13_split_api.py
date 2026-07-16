from __future__ import annotations

import os
import platform

import pytest
import torch

import fused_cpp.moe.bf16_tiled as bf16_tiled


def test_async_w13_split_is_forwarded_as_tristate(monkeypatch) -> None:
    captured: list[tuple] = []

    def fake_async(*args):
        captured.append(args)
        return args[0]

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_impl", fake_async)
    packed = torch.empty(1, dtype=torch.bfloat16)
    weights = bf16_tiled.PreparedBF16TiledFusedMoEWeights(
        w13=(packed, 1, 2),
        w2=(packed, 1, 1),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
    )
    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    one = torch.tensor([0], dtype=torch.int32)
    threads = torch.tensor([1], dtype=torch.int32)
    dep_offsets = torch.tensor([0, 0], dtype=torch.int32)
    deps = torch.empty(0, dtype=torch.int32)

    for value, expected in ((None, -1), (False, 0), (True, 1)):
        result = bf16_tiled.fused_moe_bf16_tiled_async(
            hidden,
            weights,
            topk_weights,
            topk_ids,
            one,
            one,
            threads,
            dep_offsets,
            deps,
            num_threads=1,
            w13_split=value,
        )
        assert result is hidden
        assert captured[-1][-2] == expected
        assert captured[-1][-1] == -1


def test_weight_window_is_forwarded_by_all_entrypoints(monkeypatch) -> None:
    captured: dict[str, tuple] = {}

    def fake(name: str):
        def invoke(*args):
            captured[name] = args
            return args[0]

        return invoke

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_impl", fake("normal"))
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_scheduled_impl", fake("scheduled"))
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_impl", fake("async"))
    packed = torch.empty(1, dtype=torch.bfloat16)
    weights = bf16_tiled.PreparedBF16TiledFusedMoEWeights(
        w13=(packed, 1, 2),
        w2=(packed, 1, 1),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
    )
    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    zero = torch.tensor([0], dtype=torch.int32)
    one = torch.tensor([1], dtype=torch.int32)
    dep_offsets = torch.tensor([0, 0], dtype=torch.int32)
    deps = torch.empty(0, dtype=torch.int32)
    target = 2 << 20

    bf16_tiled.fused_moe_bf16_tiled(
        hidden,
        weights,
        topk_weights,
        topk_ids,
        weight_window_bytes=target,
    )
    bf16_tiled.fused_moe_bf16_tiled_scheduled(
        hidden,
        weights,
        topk_weights,
        topk_ids,
        torch.tensor([0, 1], dtype=torch.int32),
        zero,
        one,
        weight_window_bytes=target,
    )
    bf16_tiled.fused_moe_bf16_tiled_async(
        hidden,
        weights,
        topk_weights,
        topk_ids,
        zero,
        zero,
        one,
        dep_offsets,
        deps,
        weight_window_bytes=target,
    )

    assert captured["normal"][-1] == target
    assert captured["scheduled"][-1] == target
    assert captured["async"][-2:] == (-1, target)


@pytest.mark.parametrize(
    "value,error",
    [(-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_weight_window_rejects_invalid_values(monkeypatch, value, error) -> None:
    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_impl", lambda *args: args[0])
    packed = torch.empty(1, dtype=torch.bfloat16)
    weights = bf16_tiled.PreparedBF16TiledFusedMoEWeights(
        w13=(packed, 1, 2),
        w2=(packed, 1, 1),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
    )
    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)

    with pytest.raises(error, match="weight_window_bytes"):
        bf16_tiled.fused_moe_bf16_tiled(
            hidden,
            weights,
            torch.ones((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.int32),
            weight_window_bytes=value,
        )


@pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not bf16_tiled._HAS_BF16_TILED_FUSED_MOE,
    reason="requires the AArch64 BF16 MoE extension",
)
def test_async_explicit_w13_split_overrides_environment(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(20260711)
    hidden_size, intermediate_size, experts = 64, 32, 2
    w13 = torch.empty((experts, 2 * intermediate_size, hidden_size), dtype=torch.bfloat16).normal_(
        0.0, 0.01, generator=generator
    )
    w2 = torch.empty((experts, hidden_size, intermediate_size), dtype=torch.bfloat16).normal_(
        0.0, 0.01, generator=generator
    )
    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires the SVE backend")

    routes = (12, 24)
    tokens = sum(routes)
    hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    topk_ids = torch.cat(
        [torch.full((count,), expert, dtype=torch.int32) for expert, count in enumerate(routes)]
    ).reshape(tokens, 1)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32)
    task_experts = torch.arange(experts, dtype=torch.int32)
    core_begins = torch.zeros(experts, dtype=torch.int32)
    task_threads = torch.ones(experts, dtype=torch.int32)
    dep_offsets = torch.tensor([0, 0, 1], dtype=torch.int32)
    deps = torch.tensor([0], dtype=torch.int32)

    def run(split: bool | None, weight_window_bytes: int | None = None) -> torch.Tensor:
        return bf16_tiled.fused_moe_bf16_tiled_async(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            task_experts,
            core_begins,
            task_threads,
            dep_offsets,
            deps,
            num_threads=1,
            skip_weighted=True,
            w13_split=split,
            weight_window_bytes=weight_window_bytes,
        )

    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "0")
    reference = run(None)
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "1")
    explicit_no_split = run(False)
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "0")
    explicit_split = run(True)
    four_range_w13 = run(False, 2048)
    monkeypatch.setenv("FUSED_CPP_MOE_WEIGHT_WINDOW_BYTES", "2048")
    environment_window = run(False)
    torch.testing.assert_close(explicit_no_split, reference, atol=0, rtol=0)
    torch.testing.assert_close(explicit_split, reference, atol=0, rtol=0)
    torch.testing.assert_close(four_range_w13, reference, atol=0, rtol=0)
    torch.testing.assert_close(environment_window, reference, atol=0, rtol=0)


@pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64") or not bf16_tiled._HAS_BF16_TILED_FUSED_MOE,
    reason="requires the AArch64 BF16 MoE extension",
)
@pytest.mark.parametrize("bridge", ["scheduled", "async"])
def test_multithread_weight_window_preserves_w2_owner_scatter(monkeypatch, bridge: str) -> None:
    """Multi-window W2 scatter must follow each worker's discontiguous N ownership."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "0")
    generator = torch.Generator().manual_seed(20260716)
    hidden_size, intermediate_size, routes, threads = 64, 32, 24, 5
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(threads))
    if len(affinity) < threads:
        pytest.skip(f"requires {threads} available CPUs")

    w13 = torch.empty((1, 2 * intermediate_size, hidden_size), dtype=torch.bfloat16).normal_(
        0.0, 0.01, generator=generator
    )
    w2 = torch.empty((1, hidden_size, intermediate_size), dtype=torch.bfloat16).normal_(
        0.0, 0.01, generator=generator
    )
    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires the SVE backend")

    hidden = torch.empty((routes, hidden_size), dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)
    topk_weights = torch.ones((routes, 1), dtype=torch.float32)
    topk_ids = torch.zeros((routes, 1), dtype=torch.int32)
    expert_ids = torch.zeros(1, dtype=torch.int32)
    team_threads = torch.full((1,), threads, dtype=torch.int32)
    cpu_ids = torch.tensor(affinity[:threads], dtype=torch.int32)

    def run(weight_window_bytes: int) -> torch.Tensor:
        if bridge == "scheduled":
            return bf16_tiled.fused_moe_bf16_tiled_scheduled(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                torch.tensor([0, 1], dtype=torch.int32),
                expert_ids,
                team_threads,
                thread_cpu_ids=cpu_ids,
                num_threads=threads,
                skip_weighted=True,
                weight_window_bytes=weight_window_bytes,
            )
        return bf16_tiled.fused_moe_bf16_tiled_async(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            torch.zeros(1, dtype=torch.int32),
            team_threads,
            torch.tensor([0, 0], dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            thread_cpu_ids=cpu_ids,
            num_threads=threads,
            skip_weighted=True,
            weight_window_bytes=weight_window_bytes,
        )

    reference = run(0)
    candidate = run(2048)
    torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)
