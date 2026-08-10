# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Optional

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import ASYNC_MOE_EXECUTION_TAIL_POOL
from fused_cpp.moe import ASYNC_MOE_PLACEMENT_FIXED
from fused_cpp.moe import ASYNC_MOE_PLACEMENT_TAIL_POOL
from fused_cpp.moe import AsyncMoEPlanV2
from fused_cpp.moe import available_fused_moe_bf16_tiled_backends
from fused_cpp.moe import fused_moe_naive
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import fused_moe_bf16_tiled_async
from fused_cpp.moe import fused_moe_bf16_tiled_async_plan
from fused_cpp.moe import fused_moe_bf16_tiled_planned_staged
from fused_cpp.moe import fused_moe_bf16_tiled_scheduled
from fused_cpp.moe import fused_moe_bf16_tiled_vllm_staged
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights
from fused_cpp.moe import upgrade_legacy_async_plan

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
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if use_bf16_route else "0")
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


def test_sve_m12_silu_and_w2_bf16_route_for_unit_top1(
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
        monkeypatch.setenv("FUSED_CPP_MOE_SVE_IMPL", "auto")
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
            monkeypatch.setenv("FUSED_CPP_MOE_SVE_IMPL", "jit")
            reference = call(degree)
            monkeypatch.setenv("FUSED_CPP_MOE_SVE_IMPL", "asm")
            candidate = call(degree)
            torch.testing.assert_close(
                candidate.float(),
                reference.float(),
                atol=0,
                rtol=0,
                msg=lambda message, bridge=bridge, degree=degree: f"{bridge}/poly{degree}/asm-vs-jit: {message}",
            )


def test_sve_plan_v2_route_slices_match_full_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent route slices must preserve full-expert output and completion."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE", "1")
    generator = torch.Generator().manual_seed(20260730)
    hidden_size = 64
    ffn_hidden_size = 32
    route_counts = [48, 24]
    num_tokens = sum(route_counts)
    hidden_states = _bf16_normal(
        (num_tokens, hidden_size),
        generator=generator,
        std=0.01,
    )
    w13_weight = _bf16_normal(
        (len(route_counts), 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2_weight = _bf16_normal(
        (len(route_counts), hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.cat(
        [torch.full((count,), expert, dtype=torch.int32) for expert, count in enumerate(route_counts)]
    ).reshape(num_tokens, 1)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13_weight,
        w2_weight,
        fuse_silu=True,
        backend="sve",
    )
    if hasattr(os, "sched_getaffinity"):
        cpu_ids = sorted(os.sched_getaffinity(0))[:4]
    else:
        cpu_ids = list(range(4))
    if len(cpu_ids) < 4:
        pytest.skip("requires four available CPUs")

    full_bridge = upgrade_legacy_async_plan(
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
    sliced_bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 4,
            "thread_cpu_ids": cpu_ids,
            "task_expert_ids": [0, 0, 1, 1],
            "task_core_begins": [0, 1, 2, 3],
            "task_threads": [1, 1, 1, 1],
            "task_dep_offsets": [0, 0, 0, 0, 0],
            "task_deps": [],
        }
    )
    sliced_bridge["task_range_granularities"] = [24, 24, 12, 12]

    full = fused_moe_bf16_tiled_async_plan(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        AsyncMoEPlanV2.from_dict(full_bridge),
    )
    sliced = fused_moe_bf16_tiled_async_plan(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        AsyncMoEPlanV2.from_dict(sliced_bridge),
    )

    torch.testing.assert_close(sliced.float(), full.float(), atol=0, rtol=0)


def test_sve_plan_v2_strict_and_tail_pool_match_legacy_async(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Native Plan V2 must preserve results across fixed and pooled placement."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    generator = torch.Generator().manual_seed(20260726)
    hidden_size = 64
    ffn_hidden_size = 32
    route_counts = [48, 12, 36, 8]
    num_experts = len(route_counts)
    num_tokens = sum(route_counts)
    hidden_states = _bf16_normal(
        (num_tokens, hidden_size),
        generator=generator,
        std=0.01,
    )
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
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13_weight,
        w2_weight,
        fuse_silu=True,
        backend="sve",
    )
    if hasattr(os, "sched_getaffinity"):
        cpu_ids = sorted(os.sched_getaffinity(0))[:4]
    else:
        cpu_ids = list(range(4))
    if len(cpu_ids) < 4:
        pytest.skip("requires four available CPUs")
    strict_bridge = {
        "num_threads": 4,
        "thread_cpu_ids": cpu_ids,
        "task_expert_ids": [0, 1, 2, 3],
        "task_core_begins": [0, 0, 2, 2],
        "task_threads": [2, 2, 2, 2],
        "task_dep_offsets": [0, 0, 1, 1, 2],
        "task_deps": [0, 2],
    }
    strict_plan = AsyncMoEPlanV2.from_dict(upgrade_legacy_async_plan(strict_bridge))
    reference = fused_moe_bf16_tiled_async(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        *strict_plan.legacy_schedule(),
        thread_cpu_ids=strict_plan.thread_cpu_ids,
        num_threads=4,
    )

    monkeypatch.setenv("FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS", "invalid")
    strict = fused_moe_bf16_tiled_async_plan(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        strict_plan,
    )

    steal_bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 4,
            "thread_cpu_ids": cpu_ids,
            "task_expert_ids": [0, 1, 2, 3],
            "task_core_begins": [0, 0, 0, 2],
            "task_threads": [2, 2, 2, 2],
            "task_dep_offsets": [0, 0, 1, 2, 2],
            "task_deps": [0, 1],
        }
    )
    steal_bridge["early_merge"] = False
    monkeypatch.setenv("FUSED_CPP_MOE_STRICT_TAIL_STEAL", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_STAGE_TIMING", "1")
    strict_tail_steal = fused_moe_bf16_tiled_async_plan(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        AsyncMoEPlanV2.from_dict(steal_bridge),
    )
    strict_tail_stderr = capfd.readouterr().err
    steal_match = re.search(r"\[strict_tail_steal\].*stolen_tasks=(\d+)", strict_tail_stderr)
    assert steal_match is not None
    assert int(steal_match.group(1)) > 0
    steal_ready_bridge = dict(steal_bridge)
    steal_ready_bridge["early_merge"] = True
    strict_tail_steal_ready_merge = fused_moe_bf16_tiled_async_plan(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        AsyncMoEPlanV2.from_dict(steal_ready_bridge),
    )
    steal_ready_stderr = capfd.readouterr().err
    steal_ready_match = re.search(
        r"\[strict_tail_steal\].*stolen_tasks=(\d+)",
        steal_ready_stderr,
    )
    assert steal_ready_match is not None
    assert int(steal_ready_match.group(1)) > 0
    monkeypatch.delenv("FUSED_CPP_MOE_STRICT_TAIL_STEAL")
    monkeypatch.delenv("FUSED_CPP_MOE_STAGE_TIMING")

    tail_bridge = upgrade_legacy_async_plan(strict_bridge)
    tail_bridge.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_TAIL_POOL,
            "task_core_begins": [0, -1, 2, -1],
            "task_dep_offsets": [0, 0, 0, 0, 0],
            "task_deps": [],
            "task_placement_modes": [
                ASYNC_MOE_PLACEMENT_FIXED,
                ASYNC_MOE_PLACEMENT_TAIL_POOL,
                ASYNC_MOE_PLACEMENT_FIXED,
                ASYNC_MOE_PLACEMENT_TAIL_POOL,
            ],
        }
    )
    tail = fused_moe_bf16_tiled_async_plan(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        AsyncMoEPlanV2.from_dict(tail_bridge),
    )
    w2_bridge = {
        "num_threads": 4,
        "thread_cpu_ids": cpu_ids,
        "task_expert_ids": [0, 1, 2, 3],
        "task_core_begins": [0, 1, 2, 3],
        "task_threads": [1, 1, 1, 1],
        "task_dep_offsets": [0, 0, 0, 0, 0],
        "task_deps": [],
    }
    independently_planned = fused_moe_bf16_tiled_planned_staged(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        strict_plan,
        AsyncMoEPlanV2.from_dict(upgrade_legacy_async_plan(w2_bridge)),
    )
    staged_tail = fused_moe_bf16_tiled_planned_staged(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        AsyncMoEPlanV2.from_dict(tail_bridge),
        AsyncMoEPlanV2.from_dict(tail_bridge),
    )
    staged_mixed = fused_moe_bf16_tiled_planned_staged(
        hidden_states,
        packed,
        topk_weights,
        topk_ids,
        strict_plan,
        AsyncMoEPlanV2.from_dict(tail_bridge),
    )

    torch.testing.assert_close(strict.float(), reference.float(), atol=0, rtol=0)
    torch.testing.assert_close(strict_tail_steal.float(), strict.float(), atol=0, rtol=0)
    torch.testing.assert_close(strict_tail_steal_ready_merge.float(), strict.float(), atol=0, rtol=0)
    torch.testing.assert_close(tail.float(), strict.float(), atol=0, rtol=0)
    torch.testing.assert_close(independently_planned.float(), strict.float(), atol=0, rtol=0)
    torch.testing.assert_close(staged_tail.float(), strict.float(), atol=0, rtol=0)
    torch.testing.assert_close(staged_mixed.float(), strict.float(), atol=0, rtol=0)

    unordered_overlap_bridge = upgrade_legacy_async_plan(
        {
            **strict_bridge,
            "task_core_begins": [0, 1, 2, 2],
            "task_dep_offsets": [0, 0, 0, 0, 0],
            "task_deps": [],
        }
    )
    with pytest.raises(RuntimeError, match="overlapping fixed intervals require dependency ordering"):
        fused_moe_bf16_tiled_planned_staged(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            AsyncMoEPlanV2.from_dict(unordered_overlap_bridge),
            strict_plan,
        )

    # The materialized plan owns mutable tensors, so native must validate the
    # metadata again instead of trusting the Python construction check.
    strict_plan.task_resize_points[0] = 1
    with pytest.raises(RuntimeError, match="do not support resize points"):
        fused_moe_bf16_tiled_async_plan(
            hidden_states,
            packed,
            topk_weights,
            topk_ids,
            strict_plan,
        )


@pytest.mark.parametrize("degree", [4, 5, 6], ids=["poly4", "poly5", "poly6"])
def test_sve_xbyak_exact_m_matches_static_asm(
    monkeypatch: pytest.MonkeyPatch,
    degree: int,
) -> None:
    """Cover exact-tail W13/W2 JIT kernels through all bridges."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "0")

    generator = torch.Generator().manual_seed(20260720 + degree)
    route_counts = [*range(1, 14), 23, 24, 25, 35, 36, 37, 48, 192]
    num_experts = len(route_counts)
    num_tokens = sum(route_counts)
    hidden_size = 64
    ffn_hidden_size = 32
    hidden = _bf16_normal((num_tokens, hidden_size), generator=generator, std=0.01)
    w13 = _bf16_normal(
        (num_experts, 2 * ffn_hidden_size, hidden_size),
        generator=generator,
        std=0.01,
    )
    w2 = _bf16_normal(
        (num_experts, hidden_size, ffn_hidden_size),
        generator=generator,
        std=0.01,
    )
    topk_ids = torch.cat(
        [torch.full((count,), expert, dtype=torch.int32) for expert, count in enumerate(route_counts)]
    ).reshape(num_tokens, 1)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    cpu = _first_affinity_cpu()
    expert_ids = torch.arange(num_experts, dtype=torch.int32)
    one_thread = torch.ones(num_experts, dtype=torch.int32)
    wave_offsets = torch.arange(num_experts + 1, dtype=torch.int32)
    dep_offsets = torch.arange(num_experts + 1, dtype=torch.int32)
    dep_offsets[1:] -= 1
    deps = torch.arange(num_experts - 1, dtype=torch.int32)
    thread_cpu_ids = torch.tensor([cpu], dtype=torch.int32)

    calls = {
        "normal": lambda: fused_moe_bf16_tiled(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            num_threads=1,
            silu_poly_degree=degree,
        ),
        "scheduled": lambda: fused_moe_bf16_tiled_scheduled(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            wave_offsets,
            expert_ids,
            one_thread,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=1,
            silu_poly_degree=degree,
        ),
        "async": lambda: fused_moe_bf16_tiled_async(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            torch.zeros(num_experts, dtype=torch.int32),
            one_thread,
            dep_offsets,
            deps,
            thread_cpu_ids=thread_cpu_ids,
            num_threads=1,
            silu_poly_degree=degree,
        ),
    }
    for direct_route in ("0", "1"):
        monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", direct_route)
        for bridge, call in calls.items():
            monkeypatch.setenv("FUSED_CPP_MOE_SVE_IMPL", "asm")
            reference = call()
            monkeypatch.setenv("FUSED_CPP_MOE_SVE_IMPL", "jit")
            candidate = call()
            torch.testing.assert_close(
                candidate.float(),
                reference.float(),
                atol=0,
                rtol=0,
                msg=lambda message, bridge=bridge: f"{bridge}/poly{degree}: {message}",
            )


def test_sve_xbyak_pure_gemm_matches_static_asm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate the standalone FP32 GEMM operation for every exact-M shape."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    from fused_cpp import _moe_C

    generator = torch.Generator().manual_seed(20260725)
    K = 64
    N = 64
    w13 = _bf16_normal((1, N, K), generator=generator, std=0.05)
    w2 = _bf16_normal((1, K, N // 2), generator=generator, std=0.05)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    if packed.gemm_backend != 1:
        pytest.skip("requires an SVE BF16 build/runtime")

    for rows in [*range(1, 14), 23, 24, 25]:
        A = _bf16_normal((rows, K), generator=generator, std=0.05)
        reference = _moe_C.fused_moe_test_sve_packed_gemm(A, packed.w13[0], K, N, packed.backend_n_tile, False)
        candidate = _moe_C.fused_moe_test_sve_packed_gemm(A, packed.w13[0], K, N, packed.backend_n_tile, True)
        torch.testing.assert_close(
            candidate,
            reference,
            atol=0,
            rtol=0,
            msg=lambda message, rows=rows: f"M={rows}: {message}",
        )

    with pytest.raises(RuntimeError, match="K must be divisible by 8"):
        _moe_C.fused_moe_test_sve_packed_gemm(
            torch.empty((1, K - 1), dtype=torch.bfloat16),
            packed.w13[0],
            K - 1,
            N,
            packed.backend_n_tile,
            True,
        )


@pytest.mark.parametrize(
    "probe_mode",
    [4, 10],
    ids=["full-no-store", "matrix-only"],
)
def test_sve_xbyak_m12_service_probe_runs(probe_mode: int) -> None:
    """M12 calibration probes must execute while preserving their exact-M contract."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    from fused_cpp import _moe_C

    generator = torch.Generator().manual_seed(20260801)
    K = 64
    N = 64
    w13 = _bf16_normal((1, N, K), generator=generator, std=0.05)
    w2 = _bf16_normal((1, K, N // 2), generator=generator, std=0.05)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    A = _bf16_normal((12, K), generator=generator, std=0.05)

    samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        A,
        packed.w13[0],
        K,
        N,
        packed.backend_n_tile,
        1,
        1,
        3,
        probe_mode,
    )

    assert len(samples) == 3
    assert all(sample > 0.0 for sample in samples)
    b_only_samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        A[:1],
        packed.w13[0],
        K,
        N,
        packed.backend_n_tile,
        1,
        1,
        2,
        1,
    )
    assert len(b_only_samples) == 2
    assert all(sample > 0.0 for sample in b_only_samples)
    with pytest.raises(RuntimeError, match="M1/M2, or an M12-compatible mode"):
        _moe_C.fused_moe_bench_sve_jit_w13_gemm(
            A[:1],
            packed.w13[0],
            K,
            N,
            packed.backend_n_tile,
            1,
            0,
            1,
            10,
        )
    with pytest.raises(RuntimeError, match="probe_mode must be one of"):
        _moe_C.fused_moe_bench_sve_jit_w13_gemm(
            A,
            packed.w13[0],
            K,
            N,
            packed.backend_n_tile,
            1,
            0,
            1,
            2,
        )


def test_sve_xbyak_m12_service_probe_rotates_packed_a_copies() -> None:
    """The benchmark-only API accepts independent packed-A stream windows."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    from fused_cpp import _moe_C

    generator = torch.Generator().manual_seed(20260802)
    K = 64
    N = 64
    w13 = _bf16_normal((3, N, K), generator=generator, std=0.05)
    w2 = _bf16_normal((3, K, N // 2), generator=generator, std=0.05)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    A = _bf16_normal((3, 12, K), generator=generator, std=0.05)

    samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        A,
        packed.w13[0],
        K,
        N,
        packed.backend_n_tile,
        1,
        1,
        5,
        4,
    )

    assert len(samples) == 5
    assert all(sample > 0.0 for sample in samples)


def test_sve_xbyak_service_probe_traverses_full_m12_panels() -> None:
    """The no-store service can measure a complete pure GEMM."""
    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        pytest.skip("requires an SVE BF16 build/runtime")
    from fused_cpp import _moe_C

    generator = torch.Generator().manual_seed(20260802)
    M = 24
    K = 64
    N = 64
    w13 = _bf16_normal((1, N, K), generator=generator, std=0.05)
    w2 = _bf16_normal((1, K, N // 2), generator=generator, std=0.05)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    A = _bf16_normal((M, K), generator=generator, std=0.05)

    samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        A,
        packed.w13[0],
        K,
        N,
        packed.backend_n_tile,
        1,
        1,
        3,
        4,
    )

    assert len(samples) == 3
    assert all(sample > 0.0 for sample in samples)


@pytest.mark.parametrize("bridge", ["normal", "scheduled", "async"])
@pytest.mark.parametrize("w2_bf16_route", [False, True])
@pytest.mark.parametrize("top_k", [2, 4, 5, 6, 8], ids=["fixed2", "fixed4", "dynamic5", "fixed6", "fixed8"])
def test_sve_route_merge_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    bridge: str,
    w2_bf16_route: bool,
    top_k: int,
) -> None:
    """Cover the production fixed trees and dynamic TopK fallback."""
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

    reference = fused_moe_naive(
        hidden_states.float(),
        w13_weight.float(),
        w2_weight.float(),
        topk_weights,
        topk_ids,
    ).to(torch.bfloat16)
    candidate = run()
    torch.testing.assert_close(
        candidate.float(),
        reference.float(),
        atol=7.0e-2,
        rtol=7.0e-2,
        msg=lambda message: f"{bridge}/top_k={top_k}/bf16_route={w2_bf16_route}: {message}",
    )


@pytest.mark.parametrize("bridge", ["normal", "scheduled", "async"])
@pytest.mark.parametrize("w2_bf16_route", [False, True], ids=["fp32-route", "bf16-route"])
def test_sve_w2_direct_route_store_matches_scatter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bridge: str,
    w2_bf16_route: bool,
) -> None:
    """Cover interleaved route IDs, every M tail, and multi-thread N ownership."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if w2_bf16_route else "0")
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
            msg=lambda message: f"{bridge}/bf16_route={w2_bf16_route}: {message}",
        )
    assert candidate is not None
    monkeypatch.delenv(direct_flag)
    if bridge == "async":
        trace_path = tmp_path / "default_w2_direct_route.log"
        monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "1")
        monkeypatch.setenv("FUSED_CPP_MOE_TRACE_FILE", str(trace_path))
    default = run()
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "0")
    torch.testing.assert_close(default.float(), candidate.float(), atol=0, rtol=0)
    if bridge == "async":
        trace = trace_path.read_text()
        assert "stage=w2_direct_route" in trace
        assert "stage=scatter_route_out" not in trace


def test_async_short_pool_matches_fixed_dag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Release 4T long intervals as 2T dynamic short-expert groups."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE", "0")

    threads = 8
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(threads))
    if len(affinity) < threads:
        pytest.skip(f"requires {threads} available CPUs")

    generator = torch.Generator().manual_seed(20260722)
    route_counts = [24, 24, *([2] * 8)]
    num_experts = len(route_counts)
    num_tokens = sum(route_counts)
    hidden_size = 64
    ffn_hidden_size = 32
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
    task_core_begins = torch.tensor([0, 4, *([0, 4] * 4)], dtype=torch.int32)
    task_threads = torch.full((num_experts,), 4, dtype=torch.int32)
    dep_offsets = [0]
    deps: list[int] = []
    previous_by_lane = [0, 1]
    for task_id in range(num_experts):
        if task_id >= 2:
            lane = (task_id - 2) % 2
            deps.append(previous_by_lane[lane])
            previous_by_lane[lane] = task_id
        dep_offsets.append(len(deps))
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
            torch.tensor(dep_offsets, dtype=torch.int32),
            torch.tensor(deps, dtype=torch.int32),
            thread_cpu_ids=thread_cpu_ids,
            num_threads=threads,
        )

    monkeypatch.setenv("FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS", "0")
    reference = run()
    monkeypatch.setenv("FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS", "2")
    monkeypatch.setenv("FUSED_CPP_MOE_ASYNC_SHORT_POOL_MAX_ROWS", "2")
    for _ in range(3):
        candidate = run()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    trace_path = tmp_path / "async_short_pool.log"
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE_FILE", str(trace_path))
    traced = run()
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "0")
    torch.testing.assert_close(traced.float(), reference.float(), atol=0, rtol=0)
    assert "strategy=external_plan_async_short_pool" in trace_path.read_text()


@pytest.mark.parametrize("w2_bf16_route", [False, True], ids=["fp32-route", "bf16-route"])
def test_async_ready_token_merge_overlaps_imbalanced_experts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    w2_bf16_route: bool,
) -> None:
    """Merge short-group tokens while an independent long expert group runs."""
    monkeypatch.setenv("FUSED_CPP_MOE_SVE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_W2_BF16_ROUTE", "1" if w2_bf16_route else "0")
    monkeypatch.setenv("FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE", "1")
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
    drain_flag = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN"
    batch_flag = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_BATCH"
    prefetch_flag = "FUSED_CPP_MOE_ASYNC_READY_TOKEN_PREFETCH"
    monkeypatch.setenv(ready_flag, "0")
    reference = run()
    monkeypatch.setenv(ready_flag, "1")
    monkeypatch.setenv(drain_flag, "0")
    monkeypatch.setenv(batch_flag, "1")
    monkeypatch.setenv(prefetch_flag, "0")
    legacy = run()
    torch.testing.assert_close(legacy.float(), reference.float(), atol=0, rtol=0)

    def trace_phases(path: Path) -> list[dict[str, str]]:
        return [
            dict(field.split("=", 1) for field in line.split()[1:])
            for line in path.read_text().splitlines()
            if line.startswith("PHASE ")
        ]

    legacy_trace_path = tmp_path / "async_ready_token_merge_legacy.log"
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE_FILE", str(legacy_trace_path))
    traced_legacy = run()
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "0")
    torch.testing.assert_close(traced_legacy.float(), reference.float(), atol=0, rtol=0)
    assert "merge_routes" in [record["stage"] for record in trace_phases(legacy_trace_path)]

    monkeypatch.setenv(drain_flag, "1")
    monkeypatch.setenv(batch_flag, "7")
    monkeypatch.setenv(prefetch_flag, "1")
    for _ in range(5):
        candidate = run()
        torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

    monkeypatch.delenv(ready_flag)
    monkeypatch.delenv(drain_flag)
    monkeypatch.delenv(batch_flag)
    monkeypatch.delenv(prefetch_flag)
    default = run()
    torch.testing.assert_close(default.float(), candidate.float(), atol=0, rtol=0)

    trace_path = tmp_path / "async_ready_token_merge.log"
    monkeypatch.setenv(ready_flag, "1")
    monkeypatch.setenv(drain_flag, "1")
    monkeypatch.setenv(batch_flag, "7")
    monkeypatch.setenv(prefetch_flag, "1")
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "1")
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE_FILE", str(trace_path))
    traced = run()
    monkeypatch.setenv("FUSED_CPP_MOE_TRACE", "0")
    torch.testing.assert_close(traced.float(), reference.float(), atol=0, rtol=0)
    phase_records = trace_phases(trace_path)
    merge_records = [record for record in phase_records if record["stage"] == "merge_ready_token"]
    assert len(merge_records) == num_tokens
    assert all(record["stage"] != "merge_routes" for record in phase_records)
    tokens_per_owner = (num_tokens + threads - 1) // threads
    for record in merge_records:
        owner = int(record["tid"])
        token = int(record["group"])
        token_begin = min(num_tokens, owner * tokens_per_owner)
        token_end = min(num_tokens, token_begin + tokens_per_owner)
        assert token_begin <= token < token_end, (
            f"token {token} was merged by tid {owner}, outside owner range [{token_begin}, {token_end})"
        )


@pytest.mark.parametrize("bridge", ["scheduled", "async"])
@pytest.mark.parametrize("w2_bf16_route", [False, True])
def test_sve_expert_barrier_elision_reuses_dirty_scratch(
    monkeypatch: pytest.MonkeyPatch,
    bridge: str,
    w2_bf16_route: bool,
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

    reference = run()
    for _ in range(3):
        candidate = run()
        torch.testing.assert_close(
            candidate.float(),
            reference.float(),
            atol=0,
            rtol=0,
            msg=lambda message: f"{bridge}/bf16_route={w2_bf16_route}: {message}",
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
    # The top-k=1 direct-scatter path bypasses route merge entirely.

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
