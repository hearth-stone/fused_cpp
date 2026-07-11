from __future__ import annotations

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
        assert captured[-1][-1] == expected


@pytest.mark.skipif(
    platform.machine() not in ("aarch64", "arm64")
    or not bf16_tiled._HAS_BF16_TILED_FUSED_MOE,
    reason="requires the AArch64 BF16 MoE extension",
)
def test_async_explicit_w13_split_overrides_environment(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(20260711)
    hidden_size, intermediate_size, experts = 64, 32, 2
    w13 = torch.empty(
        (experts, 2 * intermediate_size, hidden_size), dtype=torch.bfloat16
    ).normal_(0.0, 0.01, generator=generator)
    w2 = torch.empty(
        (experts, hidden_size, intermediate_size), dtype=torch.bfloat16
    ).normal_(0.0, 0.01, generator=generator)
    packed = bf16_tiled.prepare_fused_moe_bf16_tiled_weights(
        w13, w2, fuse_silu=True
    )
    if packed.gemm_backend != 1:
        pytest.skip("requires the SVE backend")

    routes = (12, 24)
    tokens = sum(routes)
    hidden = torch.empty((tokens, hidden_size), dtype=torch.bfloat16).normal_(
        0.0, 0.01, generator=generator
    )
    topk_ids = torch.cat(
        [
            torch.full((count,), expert, dtype=torch.int32)
            for expert, count in enumerate(routes)
        ]
    ).reshape(tokens, 1)
    topk_weights = torch.ones((tokens, 1), dtype=torch.float32)
    task_experts = torch.arange(experts, dtype=torch.int32)
    core_begins = torch.zeros(experts, dtype=torch.int32)
    task_threads = torch.ones(experts, dtype=torch.int32)
    dep_offsets = torch.tensor([0, 0, 1], dtype=torch.int32)
    deps = torch.tensor([0], dtype=torch.int32)

    def run(split: bool | None) -> torch.Tensor:
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
        )

    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "0")
    reference = run(None)
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "1")
    explicit_no_split = run(False)
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "0")
    explicit_split = run(True)
    torch.testing.assert_close(explicit_no_split, reference, atol=0, rtol=0)
    torch.testing.assert_close(explicit_split, reference, atol=0, rtol=0)
