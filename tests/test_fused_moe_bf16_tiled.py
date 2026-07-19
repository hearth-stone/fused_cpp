# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_bf16_tiled_async
from fused_cpp.moe import fused_moe_bf16_tiled_scheduled
from fused_cpp.moe import fused_moe_bf16_tiled_vllm_staged
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
    topk_ids = torch.tensor([[i % num_experts] for i in range(num_tokens)], dtype=torch.int32)
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


@pytest.mark.parametrize("use_bf16_route", [False, True])
def test_vllm_staged_matches_fused_sve_with_multiple_n_tasks(
    monkeypatch: pytest.MonkeyPatch,
    use_bf16_route: bool,
) -> None:
    """The global W13/W2 task pools must preserve the production SVE result."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(12))
    if len(affinity) < 12:
        pytest.skip("requires 12 available CPUs to exercise multiple N tasks per expert")

    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if use_bf16_route else "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL", "1")
    generator = torch.Generator().manual_seed(20260717)
    num_tokens, hidden_size, intermediate_size = 29, 128, 64
    num_experts, top_k, num_threads = 8, 6, 12
    hidden = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13 = _bf16_normal(
        (num_experts, 2 * intermediate_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2 = _bf16_normal(
        (num_experts, hidden_size, intermediate_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.tensor(
        [[(token + slot) % num_experts for slot in range(top_k)] for token in range(num_tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn((num_tokens, top_k), generator=generator), dim=-1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    reference = fused_moe_bf16_tiled(
        hidden,
        packed,
        topk_weights,
        topk_ids,
        num_threads=num_threads,
    )
    candidate = fused_moe_bf16_tiled_vllm_staged(
        hidden,
        packed,
        topk_weights,
        topk_ids,
        thread_cpu_ids=torch.tensor(affinity[:num_threads], dtype=torch.int32),
        num_threads=num_threads,
    )

    torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)


def test_sve_m12_silu_and_w2_bf16_route_match_legacy_for_unit_top1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover M12, padded-M12 tails 9-11, and smaller tails through all bridges."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    generator = torch.Generator().manual_seed(20260710)
    hidden_size = 64
    ffn_hidden_size = 32
    route_counts = list(range(1, 24))
    num_experts = len(route_counts)
    num_tokens = sum(route_counts)
    hidden_states = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.cat(
        [torch.full((count,), expert, dtype=torch.int32) for expert, count in enumerate(route_counts)]
    ).reshape(num_tokens, 1)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    cpu = _first_affinity_cpu()
    wave_offsets = torch.arange(num_experts + 1, dtype=torch.int32)
    expert_ids = torch.arange(num_experts, dtype=torch.int32)
    one_thread = torch.ones(num_experts, dtype=torch.int32)
    async_dep_offsets = torch.arange(num_experts + 1, dtype=torch.int32)
    async_dep_offsets[1:] -= 1
    async_deps = torch.arange(num_experts - 1, dtype=torch.int32)

    calls = {
        "normal": lambda degree=5: fused_moe_bf16_tiled(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            num_threads=1,
            silu_poly_degree=degree,
        ),
        "scheduled": lambda degree=5: fused_moe_bf16_tiled_scheduled(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            wave_offsets,
            expert_ids,
            one_thread,
            thread_cpu_ids=torch.tensor([cpu], dtype=torch.int32),
            num_threads=1,
            silu_poly_degree=degree,
        ),
        "async": lambda degree=5: fused_moe_bf16_tiled_async(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            torch.zeros(num_experts, dtype=torch.int32),
            one_thread,
            async_dep_offsets,
            async_deps,
            thread_cpu_ids=torch.tensor([cpu], dtype=torch.int32),
            num_threads=1,
            silu_poly_degree=degree,
        ),
    }
    for bridge, call in calls.items():
        monkeypatch.setenv("FUSED_CPP_MOE_SILU_MINIMAX3", "0")
        monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "0")
        reference = call()
        monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1")
        candidate = call()
        torch.testing.assert_close(
            candidate.float(),
            reference.float(),
            atol=0,
            rtol=0,
            msg=lambda message: f"{bridge}: {message}",
        )

        for degree in (4, 5, 6):
            monkeypatch.setenv("FUSED_CPP_MOE_SILU_MINIMAX3", "0")
            monkeypatch.setenv("FUSED_CPP_MOE_SILU_RECIP_NR", "0")
            monkeypatch.setenv("FUSED_CPP_MOE_SILU_M12_OPT", "0")
            reference = call(degree)
            monkeypatch.setenv("FUSED_CPP_MOE_SILU_M12_OPT", "1")
            candidate = call(degree)
            torch.testing.assert_close(
                candidate.float(),
                reference.float(),
                atol=0,
                rtol=0,
                msg=lambda message, bridge=bridge, degree=degree: f"{bridge}/poly{degree}: {message}",
            )

            for recip_steps in (1, 2):
                monkeypatch.setenv("FUSED_CPP_MOE_SILU_RECIP_NR", str(recip_steps))
                reciprocal = call(degree)
                torch.testing.assert_close(
                    reciprocal.float(),
                    candidate.float(),
                    atol=1.0e-6,
                    rtol=1.0e-2,
                    msg=lambda message, bridge=bridge, degree=degree, recip_steps=recip_steps: (
                        f"{bridge}/poly{degree}/recip{recip_steps}: {message}"
                    ),
                )

            if degree == 5:
                monkeypatch.setenv("FUSED_CPP_MOE_SILU_RECIP_NR", "0")
                monkeypatch.setenv("FUSED_CPP_MOE_SILU_MINIMAX3", "1")
                minimax3 = call(degree)
                torch.testing.assert_close(
                    minimax3.float(),
                    candidate.float(),
                    atol=1.0e-6,
                    rtol=2.0e-2,
                    msg=lambda message, bridge=bridge: f"{bridge}/minimax3: {message}",
                )


@pytest.mark.parametrize("use_bf16_route", [False, True], ids=["fp32-route", "bf16-route"])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 13], ids=["m1", "m2", "m4", "m8", "m12", "m12-m1"])
def test_sve_kc_path_matches_reference_and_threaded_output(
    monkeypatch: pytest.MonkeyPatch,
    use_bf16_route: bool,
    rows: int,
) -> None:
    """Exercise every production Mr with multiple K chunks in W13 and W2."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(4))
    if len(affinity) < 4:
        pytest.skip("requires four available CPUs")

    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if use_bf16_route else "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "1")
    generator = torch.Generator().manual_seed(20260718)
    hidden_size = 1024
    ffn_hidden_size = 1024
    hidden = _bf16_normal((rows, hidden_size), generator=generator, std=0.01)
    w13 = _bf16_normal((1, 2 * ffn_hidden_size, hidden_size), generator=generator, std=0.01)
    w2 = _bf16_normal((1, hidden_size, ffn_hidden_size), generator=generator, std=0.01)
    topk_ids = torch.zeros((rows, 1), dtype=torch.int32)
    topk_weights = torch.ones((rows, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    serial = fused_moe_bf16_tiled(hidden, packed, topk_weights, topk_ids, num_threads=1)
    threaded = fused_moe_bf16_tiled(hidden, packed, topk_weights, topk_ids, num_threads=4)
    torch.testing.assert_close(threaded.float(), serial.float(), atol=0, rtol=0)

    reference = fused_moe_naive(hidden.float(), w13.float(), w2.float(), topk_weights, topk_ids).to(torch.bfloat16)
    torch.testing.assert_close(serial.float(), reference.float(), atol=2.0e-3, rtol=2.0e-2)


@pytest.mark.parametrize("bridge", ["normal", "scheduled", "async"])
@pytest.mark.parametrize("w2_bf16_route", [False, True])
@pytest.mark.parametrize("top_k", [2, 4, 5, 6, 8], ids=["fixed2", "fixed4", "dynamic5", "fixed6", "fixed8"])
def test_sve_route_merge_matches_sequential(
    monkeypatch: pytest.MonkeyPatch,
    bridge: str,
    w2_bf16_route: bool,
    top_k: int,
) -> None:
    """Cover fixed trees, dynamic fallback, H-unroll variants, and the U1 default."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if w2_bf16_route else "0")
    generator = torch.Generator().manual_seed(20260716)
    num_tokens = 19
    hidden_size = 64
    ffn_hidden_size = 32
    num_experts = 8
    threads = 4
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(threads))
    if len(affinity) < threads:
        pytest.skip(f"requires {threads} available CPUs")
    monkeypatch.setenv("FUSED_CPP_MOE_PIN_THREADS", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_PIN_THREAD_CPUS", ",".join(str(cpu) for cpu in affinity[:threads]))

    hidden_states = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.05)
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.05,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.05,
    )
    topk_ids = torch.arange(top_k, dtype=torch.int32).repeat(num_tokens, 1)
    topk_weights = torch.softmax(torch.randn((num_tokens, top_k), generator=generator), dim=-1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    expert_ids = torch.arange(top_k, dtype=torch.int32)
    team_threads = torch.full((top_k,), threads, dtype=torch.int32)
    thread_cpu_ids = torch.tensor(affinity[:threads], dtype=torch.int32)
    if bridge == "normal":

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                num_threads=threads,
            )

    elif bridge == "scheduled":
        wave_offsets = torch.arange(top_k + 1, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_scheduled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                wave_offsets,
                expert_ids,
                team_threads,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=threads,
            )

    else:
        dep_offsets = torch.arange(top_k + 1, dtype=torch.int32)
        dep_offsets[1:] -= 1
        deps = torch.arange(top_k - 1, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_async(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                expert_ids,
                torch.zeros(top_k, dtype=torch.int32),
                team_threads,
                dep_offsets,
                deps,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=threads,
            )

    merge_flag = "FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"
    legacy_merge_flag = "FUSED_CPP_MOE_SVE_ROUTE_MERGE_TREE_UNROLL"
    monkeypatch.setenv(merge_flag, "0")
    reference = run().float()
    candidates: dict[int, torch.Tensor] = {}
    for unroll in (1, 2, 4):
        monkeypatch.setenv(merge_flag, str(unroll))
        candidate = run().float()
        candidates[unroll] = candidate
        atol = 1.0e-5 if top_k in (2, 4, 6, 8) else 0.0
        rtol = 2.0e-2 if top_k in (2, 4, 6, 8) else 0.0
        torch.testing.assert_close(
            candidate,
            reference,
            atol=atol,
            rtol=rtol,
            msg=lambda message, unroll=unroll: (
                f"{bridge}/top_k={top_k}/bf16_route={w2_bf16_route}/sve_u{unroll}: {message}"
            ),
        )
    monkeypatch.delenv(merge_flag)
    monkeypatch.delenv(legacy_merge_flag, raising=False)
    default = run().float()
    torch.testing.assert_close(
        default,
        candidates[1],
        atol=0.0,
        rtol=0.0,
        msg=lambda message: f"{bridge}/top_k={top_k}/bf16_route={w2_bf16_route}/default_u1: {message}",
    )
    if bridge == "normal" and not w2_bf16_route and top_k == 6:
        monkeypatch.setenv(legacy_merge_flag, "1")
        legacy_alias = run().float()
        torch.testing.assert_close(legacy_alias, reference, atol=1.0e-5, rtol=2.0e-2)


@pytest.mark.parametrize("bridge", ["normal", "scheduled", "async"])
@pytest.mark.parametrize("split_2d", [False, True], ids=["nsplit", "2d-nsplit"])
@pytest.mark.parametrize("w2_bf16_route", [False, True], ids=["fp32-route", "bf16-route"])
def test_sve_w2_direct_route_store_matches_scatter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bridge: str,
    split_2d: bool,
    w2_bf16_route: bool,
) -> None:
    """Cover interleaved route IDs, every M tail, and multi-thread N ownership."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if w2_bf16_route else "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL", "0")
    monkeypatch.setenv("FUSED_CPP_MOE_FUSED_2D_SPLIT", "1" if split_2d else "0")
    generator = torch.Generator().manual_seed(20260716)
    hidden_size = 64
    ffn_hidden_size = 32
    route_counts = list(range(1, 24))
    num_experts = len(route_counts)
    top_k = 6
    num_routes = sum(route_counts)
    assert num_routes % top_k == 0
    num_tokens = num_routes // top_k
    threads = 5
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(threads))
    if len(affinity) < threads:
        pytest.skip(f"requires {threads} available CPUs")
    if bridge == "normal" and split_2d:
        monkeypatch.setenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", "1")
        monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_CORE_BASES", str(affinity[0]))
        monkeypatch.setenv("FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION", "1")
    else:
        monkeypatch.setenv("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", "0")

    hidden_states = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    remaining = route_counts.copy()
    flat_ids: list[int] = []
    while len(flat_ids) < num_routes:
        for expert in range(num_experts):
            if remaining[expert] > 0:
                flat_ids.append(expert)
                remaining[expert] -= 1
    topk_ids = torch.tensor(flat_ids, dtype=torch.int32).reshape(num_tokens, top_k)
    topk_weights = torch.softmax(torch.randn((num_tokens, top_k), generator=generator), dim=-1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    expert_ids = torch.arange(num_experts, dtype=torch.int32)
    team_threads = torch.full((num_experts,), threads, dtype=torch.int32)
    thread_cpu_ids = torch.tensor(affinity[:threads], dtype=torch.int32)
    if bridge == "normal":

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                num_threads=threads,
            )

    elif bridge == "scheduled":
        wave_offsets = torch.arange(num_experts + 1, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_scheduled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                wave_offsets,
                expert_ids,
                team_threads,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=threads,
            )

    else:
        dep_offsets = torch.arange(num_experts + 1, dtype=torch.int32)
        dep_offsets[1:] -= 1
        deps = torch.arange(num_experts - 1, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_async(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                expert_ids,
                torch.zeros(num_experts, dtype=torch.int32),
                team_threads,
                dep_offsets,
                deps,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=threads,
            )

    direct_flag = "FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"
    monkeypatch.setenv(direct_flag, "0")
    reference = run()
    monkeypatch.setenv(direct_flag, "1")
    candidate: Optional[torch.Tensor] = None
    for _ in range(3):
        candidate = run()
        torch.testing.assert_close(
            candidate.float(),
            reference.float(),
            atol=0,
            rtol=0,
            msg=lambda message: f"{bridge}/split_2d={split_2d}/bf16_route={w2_bf16_route}: {message}",
        )
    assert candidate is not None
    monkeypatch.delenv(direct_flag)
    if bridge == "async" and not split_2d:
        trace_path = tmp_path / "default_w2_direct_route.log"
        monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "1")
        monkeypatch.setenv("FUSED_CPP_MOE_TRACE_FILE", str(trace_path))
    default = run()
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "0")
    torch.testing.assert_close(default.float(), candidate.float(), atol=0, rtol=0)
    if bridge == "async" and not split_2d:
        trace = trace_path.read_text()
        assert "stage=w2_direct_route" in trace
        assert "stage=scatter_route_out" not in trace


@pytest.mark.parametrize("w2_bf16_route", [False, True], ids=["fp32-route", "bf16-route"])
def test_async_ready_token_merge_overlaps_imbalanced_experts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    w2_bf16_route: bool,
) -> None:
    """Merge short-group tokens while an independent long expert group runs."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W13_SPLIT_N", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if w2_bf16_route else "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_FUSED_2D_SPLIT", "0")

    generator = torch.Generator().manual_seed(20260716)
    hidden_size = 512
    ffn_hidden_size = 256
    num_experts = 4
    top_k = 2
    short_tokens = 8
    long_tokens = 256
    num_tokens = short_tokens + long_tokens
    threads = num_experts
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(threads))
    if len(affinity) < threads:
        pytest.skip(f"requires {threads} available CPUs")

    hidden_states = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.empty((num_tokens, top_k), dtype=torch.int32)
    topk_ids[:short_tokens] = torch.tensor([0, 1], dtype=torch.int32)
    topk_ids[short_tokens:] = torch.tensor([2, 3], dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn((num_tokens, top_k), generator=generator), dim=-1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    expert_ids = torch.arange(num_experts, dtype=torch.int32)
    task_core_begins = torch.arange(num_experts, dtype=torch.int32)
    task_threads = torch.ones(num_experts, dtype=torch.int32)
    dep_offsets = torch.zeros(num_experts + 1, dtype=torch.int32)
    deps = torch.empty(0, dtype=torch.int32)
    thread_cpu_ids = torch.tensor(affinity[:threads], dtype=torch.int32)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            task_core_begins,
            task_threads,
            dep_offsets,
            deps,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=threads,
        )

    ready_flag = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"
    monkeypatch.setenv(ready_flag, "0")
    reference = run()
    monkeypatch.setenv(ready_flag, "1")
    for _ in range(5):
        candidate = run()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    monkeypatch.delenv(ready_flag)
    default = run()
    torch.testing.assert_close(default.float(), candidate.float(), atol=0, rtol=0)

    trace_path = tmp_path / "async_ready_token_merge.log"
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE_FILE", str(trace_path))
    traced = run()
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "0")
    torch.testing.assert_close(traced.float(), reference.float(), atol=0, rtol=0)
    assert "stage=merge_ready_token" in trace_path.read_text()


@pytest.mark.parametrize("bridge", ["scheduled", "async"])
@pytest.mark.parametrize("w2_bf16_route", [False, True])
@pytest.mark.parametrize(
    "elide_zero,owner_scatter",
    [(True, False), (False, True), (True, True), (None, None)],
    ids=["no-zero", "owner-scatter", "combined", "default"],
)
def test_sve_expert_barrier_elision_reuses_dirty_scratch(
    monkeypatch: pytest.MonkeyPatch,
    bridge: str,
    w2_bf16_route: bool,
    elide_zero: Optional[bool],
    owner_scatter: Optional[bool],
) -> None:
    """Exercise all M tails while repeatedly reusing dirty team scratch."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if w2_bf16_route else "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "0")
    generator = torch.Generator().manual_seed(20260714)
    hidden_size = 64
    ffn_hidden_size = 32
    route_counts = list(range(24, 0, -1))
    num_experts = len(route_counts)
    num_tokens = sum(route_counts)
    threads = 5
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(threads))
    if len(affinity) < threads:
        pytest.skip(f"requires {threads} available CPUs")

    hidden_states = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.cat(
        [torch.full((count,), expert, dtype=torch.int32) for expert, count in enumerate(route_counts)]
    ).reshape(num_tokens, 1)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    expert_ids = torch.arange(num_experts, dtype=torch.int32)
    team_threads = torch.full((num_experts,), threads, dtype=torch.int32)
    thread_cpu_ids = torch.tensor(affinity[:threads], dtype=torch.int32)
    if bridge == "scheduled":
        wave_offsets = torch.arange(num_experts + 1, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_scheduled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                wave_offsets,
                expert_ids,
                team_threads,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=threads,
            )

    else:
        dep_offsets = torch.arange(num_experts + 1, dtype=torch.int32)
        dep_offsets[1:] -= 1
        deps = torch.arange(num_experts - 1, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_async(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                expert_ids,
                torch.zeros(num_experts, dtype=torch.int32),
                team_threads,
                dep_offsets,
                deps,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=threads,
            )

    zero_flag = "FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO"
    owner_flag = "FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER"
    monkeypatch.setenv(zero_flag, "0")
    monkeypatch.setenv(owner_flag, "0")
    reference = run()

    if elide_zero is None:
        monkeypatch.delenv(zero_flag)
        monkeypatch.delenv(owner_flag)
    else:
        monkeypatch.setenv(zero_flag, "1" if elide_zero else "0")
        monkeypatch.setenv(owner_flag, "1" if owner_scatter else "0")
    for _ in range(3):
        candidate = run()
        torch.testing.assert_close(
            candidate.float(),
            reference.float(),
            atol=0,
            rtol=0,
            msg=lambda message: (
                f"{bridge}/bf16_route={w2_bf16_route}/zero={elide_zero}/owner={owner_scatter}: {message}"
            ),
        )


def test_fused_moe_bf16_tiled_skip_weighted_matches_unit_weighted(monkeypatch: pytest.MonkeyPatch) -> None:
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
    # The top-k=1 direct-scatter path must bypass route-merge dispatch even
    # when an experimental SVE merge variant is requested.
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL", "4")

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
    assert f"core_bases=[{first_cpu},{first_cpu + 2}]" in captured.err
    torch.testing.assert_close(hierarchical.float(), serial.float(), atol=0, rtol=0)


@pytest.mark.parametrize("bridge", ["normal", "scheduled", "async"])
def test_fused_moe_bf16_tiled_native_out_buffer(bridge: str) -> None:
    generator = torch.Generator().manual_seed(11)
    num_tokens, hidden_size, ffn_hidden_size = 24, 64, 32
    num_experts, top_k = 8, 6
    hidden_states = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13_weight = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.tensor(
        [[(token + slot) % num_experts for slot in range(top_k)] for token in range(num_tokens)],
        dtype=torch.int32,
    )
    topk_weights = torch.softmax(torch.randn((num_tokens, top_k), generator=generator), dim=-1)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight, fuse_silu=True)
    if packed.gemm_backend != 1:
        pytest.skip("requires the SVE fused MoE backend")
    out_buffer = torch.empty_like(hidden_states)
    expert_ids = torch.arange(num_experts, dtype=torch.int32)
    team_threads = torch.ones(num_experts, dtype=torch.int32)

    def run(out: torch.Tensor | None = None) -> torch.Tensor:
        if bridge == "normal":
            return fused_moe_bf16_tiled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                num_threads=1,
                out=out,
            )
        if bridge == "scheduled":
            return fused_moe_bf16_tiled_scheduled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                torch.arange(num_experts + 1, dtype=torch.int32),
                expert_ids,
                team_threads,
                num_threads=1,
                out=out,
            )
        return fused_moe_bf16_tiled_async(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            torch.zeros(num_experts, dtype=torch.int32),
            team_threads,
            torch.cat((torch.zeros(1, dtype=torch.int32), torch.arange(num_experts, dtype=torch.int32))),
            torch.arange(num_experts - 1, dtype=torch.int32),
            num_threads=1,
            out=out,
        )

    out_buffer.fill_(float("nan"))
    version_before = out_buffer._version
    ret = run(out_buffer)
    assert ret is out_buffer
    assert ret.data_ptr() == out_buffer.data_ptr()
    assert out_buffer._version == version_before + 1
    ref = run()
    torch.testing.assert_close(out_buffer.float(), ref.float(), atol=0, rtol=0)


def test_fused_moe_bf16_tiled_native_out_rejects_input_alias() -> None:
    hidden_states, w13_weight, w2_weight, _, _, topk_weights, topk_ids = _case(seed=12)
    packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)

    with pytest.raises(RuntimeError, match="out must not overlap input"):
        fused_moe_bf16_tiled(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            num_threads=1,
            out=hidden_states,
        )


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
