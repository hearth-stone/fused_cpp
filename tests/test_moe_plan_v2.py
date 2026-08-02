from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fused_cpp.moe.bf16_tiled as bf16_tiled
from fused_cpp.moe.plan import (
    ASYNC_MOE_EXECUTION_ELASTIC,
    ASYNC_MOE_EXECUTION_TAIL_POOL,
    ASYNC_MOE_PLACEMENT_FIXED,
    ASYNC_MOE_PLACEMENT_TAIL_POOL,
    ASYNC_MOE_PLAN_VERSION,
    ASYNC_MOE_RESIZE_BEFORE_W2,
    AsyncMoEPlanV2,
    upgrade_legacy_async_plan,
)


def _legacy_bridge() -> dict[str, object]:
    return {
        "num_threads": 4,
        "thread_cpu_ids": [8, 9, 10, 11],
        "task_expert_ids": [3, 7],
        "task_core_begins": [0, 2],
        "task_threads": [2, 2],
        "task_dep_offsets": [0, 0, 1],
        "task_deps": [0],
    }


def _tail_pool_bridge() -> dict[str, object]:
    plan = upgrade_legacy_async_plan(_legacy_bridge())
    plan.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_TAIL_POOL,
            "task_core_begins": [0, -1],
            "task_dep_offsets": [0, 0, 0],
            "task_deps": [],
            "task_placement_modes": [
                ASYNC_MOE_PLACEMENT_FIXED,
                ASYNC_MOE_PLACEMENT_TAIL_POOL,
            ],
        }
    )
    return plan


def _prepared_weights() -> SimpleNamespace:
    packed = torch.empty(1, dtype=torch.bfloat16)
    return SimpleNamespace(
        w13=(packed, 1, 2),
        w2=(packed, 1, 1),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
    )


def test_upgrade_legacy_plan_produces_strict_singleton_widths() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    plan = AsyncMoEPlanV2.from_dict(upgraded)

    assert upgraded["plan_version"] == ASYNC_MOE_PLAN_VERSION
    assert upgraded["execution_mode"] == "strict"
    assert upgraded["task_allowed_thread_offsets"] == [0, 1, 2]
    assert upgraded["task_allowed_threads"] == [2, 2]
    assert upgraded["task_placement_modes"] == [
        ASYNC_MOE_PLACEMENT_FIXED,
        ASYNC_MOE_PLACEMENT_FIXED,
    ]
    assert plan.plan_version == ASYNC_MOE_PLAN_VERSION
    assert plan.num_threads == 4
    assert plan.thread_cpu_ids.tolist() == [8, 9, 10, 11]
    assert plan.task_threads.tolist() == [2, 2]
    assert plan.task_w13_window_bytes.tolist() == [-1, -1]
    assert plan.task_w2_window_bytes.tolist() == [-1, -1]
    assert plan.task_release_ns.tolist() == [0, 0]
    assert plan.task_preferred_core_begins.tolist() == [-1, -1]
    assert plan.early_merge is None
    assert plan.native_early_merge == -1
    assert plan.legacy_schedule()[0].tolist() == [3, 7]


@pytest.mark.parametrize(("early_merge", "native"), [(False, 0), (True, 1)])
def test_plan_v2_materializes_early_merge_control(early_merge: bool, native: int) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge["early_merge"] = early_merge

    plan = AsyncMoEPlanV2.from_dict(bridge)

    assert plan.early_merge is early_merge
    assert plan.native_early_merge == native


def test_plan_v2_rejects_invalid_early_merge_control() -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge["early_merge"] = 1

    with pytest.raises(TypeError, match="bool or None"):
        AsyncMoEPlanV2.from_dict(bridge)


def test_elastic_plan_rejects_forced_early_merge() -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge["execution_mode"] = ASYNC_MOE_EXECUTION_ELASTIC
    bridge["early_merge"] = True

    with pytest.raises(ValueError, match="does not support"):
        AsyncMoEPlanV2.from_dict(bridge)


def test_strict_plan_accepts_equal_contiguous_route_slices() -> None:
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 4,
            "thread_cpu_ids": [8, 9, 10, 11],
            "task_expert_ids": [3, 3, 7],
            "task_core_begins": [0, 2, 0],
            "task_threads": [2, 2, 4],
            "task_dep_offsets": [0, 0, 0, 2],
            "task_deps": [0, 1],
        }
    )
    bridge["task_range_granularities"] = [64, 64, 0]

    plan = AsyncMoEPlanV2.from_dict(bridge)

    assert plan.task_expert_ids.tolist() == [3, 3, 7]
    assert plan.task_range_granularities.tolist() == [64, 64, 0]


@pytest.mark.parametrize("granularities", ([0, 0, 0], [64, 32, 0]))
def test_strict_plan_rejects_invalid_duplicate_expert_ranges(granularities) -> None:
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 4,
            "thread_cpu_ids": [8, 9, 10, 11],
            "task_expert_ids": [3, 3, 7],
            "task_core_begins": [0, 2, 0],
            "task_threads": [2, 2, 4],
            "task_dep_offsets": [0, 0, 0, 2],
            "task_deps": [0, 1],
        }
    )
    bridge["task_range_granularities"] = granularities

    with pytest.raises(ValueError, match="route-sliced expert"):
        AsyncMoEPlanV2.from_dict(bridge)


def test_strict_plan_accepts_a_wider_future_width_envelope() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded.update(
        {
            "task_preferred_threads": [1, 2],
            "task_min_threads": [1, 2],
            "task_max_threads": [2, 2],
            "task_allowed_thread_offsets": [0, 2, 3],
            "task_allowed_threads": [1, 2, 2],
        }
    )

    plan = AsyncMoEPlanV2.from_dict(upgraded)

    assert plan.task_threads.tolist() == [2, 2]
    assert plan.task_preferred_threads.tolist() == [1, 2]
    assert plan.task_allowed_threads.tolist() == [1, 2, 2]


def test_plan_v2_rejects_unsupported_resize_semantics() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded["task_resize_points"] = [1, 0]

    try:
        AsyncMoEPlanV2.from_dict(upgraded)
    except ValueError as error:
        assert "do not support resize points" in str(error)
    else:
        raise AssertionError("unsupported resize metadata was accepted")


def test_elastic_plan_accepts_same_numa_w2_expansion(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_ELASTIC,
            "task_preferred_threads": [4, 4],
            "task_min_threads": [2, 2],
            "task_max_threads": [4, 4],
            "task_allowed_thread_offsets": [0, 2, 4],
            "task_allowed_threads": [2, 4, 2, 4],
            "task_numa_nodes": [0, 0],
            "task_resize_points": [ASYNC_MOE_RESIZE_BEFORE_W2] * 2,
            "task_resize_timeout_ns": [0, 5000],
        }
    )
    monkeypatch.setattr("fused_cpp.moe.plan._linux_cpu_numa_node", lambda cpu: 0)

    plan = AsyncMoEPlanV2.from_dict(bridge)

    assert plan.native_execution_mode == 2
    assert plan.task_preferred_threads.tolist() == [4, 4]
    assert plan.task_resize_timeout_ns.tolist() == [0, 5000]
    assert plan.task_preferred_core_begins.tolist() == [-1, -1]


def test_elastic_plan_accepts_disjoint_w2_migration(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 8,
            "thread_cpu_ids": list(range(8, 16)),
            "task_expert_ids": [3, 7],
            "task_core_begins": [0, 2],
            "task_threads": [2, 2],
            "task_dep_offsets": [0, 0, 0],
            "task_deps": [],
        }
    )
    bridge.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_ELASTIC,
            "task_preferred_threads": [4, 4],
            "task_min_threads": [2, 2],
            "task_max_threads": [4, 4],
            "task_allowed_thread_offsets": [0, 2, 4],
            "task_allowed_threads": [2, 4, 2, 4],
            "task_numa_nodes": [0, 0],
            "task_resize_points": [ASYNC_MOE_RESIZE_BEFORE_W2] * 2,
            "task_resize_timeout_ns": [1000, 1000],
            "task_preferred_core_begins": [0, 4],
        }
    )
    monkeypatch.setattr("fused_cpp.moe.plan._linux_cpu_numa_node", lambda cpu: 0)

    plan = AsyncMoEPlanV2.from_dict(bridge)

    assert plan.task_preferred_core_begins.tolist() == [0, 4]


def test_elastic_plan_rejects_partially_overlapping_w2_migration(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(
        {
            "num_threads": 8,
            "thread_cpu_ids": list(range(8, 16)),
            "task_expert_ids": [3],
            "task_core_begins": [3],
            "task_threads": [2],
            "task_dep_offsets": [0, 0],
            "task_deps": [],
        }
    )
    bridge.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_ELASTIC,
            "task_preferred_threads": [4],
            "task_min_threads": [2],
            "task_max_threads": [4],
            "task_allowed_thread_offsets": [0, 2],
            "task_allowed_threads": [2, 4],
            "task_numa_nodes": [0],
            "task_resize_points": [ASYNC_MOE_RESIZE_BEFORE_W2],
            "task_preferred_core_begins": [0],
        }
    )
    monkeypatch.setattr("fused_cpp.moe.plan._linux_cpu_numa_node", lambda cpu: 0)

    with pytest.raises(ValueError, match="contain or be disjoint"):
        AsyncMoEPlanV2.from_dict(bridge)


def test_elastic_plan_rejects_cross_numa_cohort(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_ELASTIC,
            "task_preferred_threads": [4, 4],
            "task_min_threads": [2, 2],
            "task_max_threads": [4, 4],
            "task_allowed_thread_offsets": [0, 2, 4],
            "task_allowed_threads": [2, 4, 2, 4],
            "task_numa_nodes": [0, 0],
            "task_resize_points": [ASYNC_MOE_RESIZE_BEFORE_W2] * 2,
        }
    )
    monkeypatch.setattr(
        "fused_cpp.moe.plan._linux_cpu_numa_node",
        lambda cpu: 1 if cpu == 11 else 0,
    )

    with pytest.raises(ValueError, match="crosses NUMA"):
        AsyncMoEPlanV2.from_dict(bridge)


def test_plan_v2_rejects_inconsistent_allowed_widths() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded["task_allowed_threads"] = [1, 2]

    try:
        AsyncMoEPlanV2.from_dict(upgraded)
    except ValueError as error:
        assert "task_threads[0] is not present" in str(error)
    else:
        raise AssertionError("an inconsistent selected width was accepted")


def test_plan_v2_accepts_optional_per_task_stage_windows() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded["task_w13_window_bytes"] = [1048576, -1]
    upgraded["task_w2_window_bytes"] = [524288, 0]

    plan = AsyncMoEPlanV2.from_dict(upgraded)

    assert plan.task_w13_window_bytes.tolist() == [1048576, -1]
    assert plan.task_w2_window_bytes.tolist() == [524288, 0]


def test_plan_v2_rejects_invalid_per_task_stage_windows() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded["task_w13_window_bytes"] = [-2, -1]

    with pytest.raises(ValueError, match="must be -1"):
        AsyncMoEPlanV2.from_dict(upgraded)


def test_strict_plan_accepts_timed_task_releases() -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded["task_release_ns"] = [0, 250_000]

    plan = AsyncMoEPlanV2.from_dict(upgraded)

    assert plan.task_release_ns.tolist() == [0, 250_000]


@pytest.mark.parametrize("execution_mode", [ASYNC_MOE_EXECUTION_TAIL_POOL, ASYNC_MOE_EXECUTION_ELASTIC])
def test_non_strict_plan_rejects_timed_task_releases(execution_mode: str) -> None:
    upgraded = upgrade_legacy_async_plan(_legacy_bridge())
    upgraded["execution_mode"] = execution_mode
    upgraded["task_release_ns"] = [0, 1]

    with pytest.raises(ValueError, match="requires strict execution"):
        AsyncMoEPlanV2.from_dict(upgraded)


def test_tail_pool_plan_accepts_explicit_whole_expert_placement() -> None:
    plan = AsyncMoEPlanV2.from_dict(_tail_pool_bridge())

    assert plan.execution_mode == ASYNC_MOE_EXECUTION_TAIL_POOL
    assert plan.native_execution_mode == 1
    assert plan.task_core_begins.tolist() == [0, -1]
    assert plan.task_placement_modes.tolist() == [
        ASYNC_MOE_PLACEMENT_FIXED,
        ASYNC_MOE_PLACEMENT_TAIL_POOL,
    ]
    with pytest.raises(RuntimeError, match="only strict Plan V2"):
        plan.legacy_schedule()


def test_tail_pool_plan_requires_a_pooled_task() -> None:
    plan = _tail_pool_bridge()
    plan["task_core_begins"] = [0, 2]
    plan["task_placement_modes"] = [
        ASYNC_MOE_PLACEMENT_FIXED,
        ASYNC_MOE_PLACEMENT_FIXED,
    ]

    with pytest.raises(ValueError, match="at least one pooled task"):
        AsyncMoEPlanV2.from_dict(plan)


def test_tail_pool_plan_rejects_unaligned_fixed_interval() -> None:
    plan = _tail_pool_bridge()
    plan["task_core_begins"] = [1, -1]

    with pytest.raises(ValueError, match="align to the tail-pool width"):
        AsyncMoEPlanV2.from_dict(plan)


def test_tail_pool_plan_rejects_pooled_dependencies() -> None:
    plan = _tail_pool_bridge()
    plan["task_dep_offsets"] = [0, 0, 1]
    plan["task_deps"] = [0]

    with pytest.raises(ValueError, match="must not have dependencies"):
        AsyncMoEPlanV2.from_dict(plan)


def test_async_plan_wrapper_calls_native_plan_v2(monkeypatch) -> None:
    plan = AsyncMoEPlanV2.from_dict(upgrade_legacy_async_plan(_legacy_bridge()))
    captured: dict[str, object] = {}
    sentinel = torch.empty(0)

    def fake_async_plan_v2(*args):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(
        bf16_tiled,
        "_fused_moe_bf16_tiled_async_plan_v2_impl",
        fake_async_plan_v2,
    )
    result = bf16_tiled.fused_moe_bf16_tiled_async_plan(
        torch.empty((1, 1), dtype=torch.bfloat16),
        _prepared_weights(),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        plan,
        skip_weighted=True,
        w13_split=True,
    )

    assert result is sentinel
    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[9].tolist() == [3, 7]
    assert args[10].tolist() == [0, 2]
    assert args[11].tolist() == [2, 2]
    assert args[14] == ASYNC_MOE_PLAN_VERSION
    assert args[15] == 0
    assert args[21].tolist() == [
        ASYNC_MOE_PLACEMENT_FIXED,
        ASYNC_MOE_PLACEMENT_FIXED,
    ]
    assert args[26].tolist() == [8, 9, 10, 11]
    assert args[29] == 4
    assert args[32] is True
    assert args[37] == 1
    assert args[40].tolist() == [-1, -1]
    assert args[41].tolist() == [-1, -1]
    assert args[42].tolist() == [0, 0]
    assert args[43] == -1


def test_async_plan_wrapper_calls_elastic_native_and_collects_stats(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge.update(
        {
            "execution_mode": ASYNC_MOE_EXECUTION_ELASTIC,
            "task_preferred_threads": [4, 4],
            "task_min_threads": [2, 2],
            "task_max_threads": [4, 4],
            "task_allowed_thread_offsets": [0, 2, 4],
            "task_allowed_threads": [2, 4, 2, 4],
            "task_numa_nodes": [0, 0],
            "task_resize_points": [ASYNC_MOE_RESIZE_BEFORE_W2] * 2,
            "task_resize_timeout_ns": [0, 1000],
        }
    )
    monkeypatch.setattr("fused_cpp.moe.plan._linux_cpu_numa_node", lambda cpu: 0)
    plan = AsyncMoEPlanV2.from_dict(bridge)
    captured: dict[str, object] = {}
    sentinel = torch.empty(0)

    def fake_elastic(*args):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_plan_v2_impl", lambda *args: sentinel)
    monkeypatch.setattr(
        bf16_tiled,
        "_fused_moe_bf16_tiled_async_plan_v2_elastic_impl",
        fake_elastic,
    )
    stats = bf16_tiled.make_async_moe_elastic_stats()

    result = bf16_tiled.fused_moe_bf16_tiled_async_plan(
        torch.empty((1, 1), dtype=torch.bfloat16),
        _prepared_weights(),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        plan,
        skip_weighted=True,
        elastic_stats_out=stats,
    )

    assert result is sentinel
    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[15] == 2
    assert args[42].tolist() == [0, 0]
    assert args[43].tolist() == [0, 1000]
    assert args[44] is stats
    assert args[45].tolist() == [-1, -1]
    assert args[46] == -1
    assert bf16_tiled.decode_async_moe_elastic_stats(stats)["eligible_tasks"] == 0


def test_async_plan_wrapper_falls_back_for_strict_plan(monkeypatch) -> None:
    plan = AsyncMoEPlanV2.from_dict(upgrade_legacy_async_plan(_legacy_bridge()))
    captured: dict[str, object] = {}
    sentinel = torch.empty(0)

    def fake_async(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_plan_v2_impl", None)
    monkeypatch.setattr(bf16_tiled, "fused_moe_bf16_tiled_async", fake_async)
    result = bf16_tiled.fused_moe_bf16_tiled_async_plan(
        torch.empty((1, 1), dtype=torch.bfloat16),
        object(),
        torch.ones((1, 1)),
        torch.zeros((1, 1), dtype=torch.int32),
        plan,
    )

    assert result is sentinel
    assert captured["args"][4].tolist() == [3, 7]
    assert captured["kwargs"]["num_threads"] == 4


def test_async_plan_wrapper_requires_native_tail_pool(monkeypatch) -> None:
    plan = AsyncMoEPlanV2.from_dict(_tail_pool_bridge())
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_plan_v2_impl", None)

    with pytest.raises(RuntimeError, match="tail_pool requires native"):
        bf16_tiled.fused_moe_bf16_tiled_async_plan(
            torch.empty((1, 1), dtype=torch.bfloat16),
            object(),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int32),
            plan,
        )


def test_async_plan_wrapper_requires_native_per_task_windows(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge["task_w13_window_bytes"] = [1048576, -1]
    plan = AsyncMoEPlanV2.from_dict(bridge)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_plan_v2_impl", None)

    with pytest.raises(RuntimeError, match="per-task W13/W2 windows require native"):
        bf16_tiled.fused_moe_bf16_tiled_async_plan(
            torch.empty((1, 1), dtype=torch.bfloat16),
            object(),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int32),
            plan,
        )


def test_async_plan_wrapper_requires_native_early_merge_control(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge["early_merge"] = False
    plan = AsyncMoEPlanV2.from_dict(bridge)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_plan_v2_impl", None)

    with pytest.raises(RuntimeError, match="early_merge control requires native"):
        bf16_tiled.fused_moe_bf16_tiled_async_plan(
            torch.empty((1, 1), dtype=torch.bfloat16),
            object(),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int32),
            plan,
        )


def test_async_plan_wrapper_requires_native_timed_releases(monkeypatch) -> None:
    bridge = upgrade_legacy_async_plan(_legacy_bridge())
    bridge["task_release_ns"] = [0, 1000]
    plan = AsyncMoEPlanV2.from_dict(bridge)
    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_async_plan_v2_impl", None)

    with pytest.raises(RuntimeError, match="timed task releases require native"):
        bf16_tiled.fused_moe_bf16_tiled_async_plan(
            torch.empty((1, 1), dtype=torch.bfloat16),
            object(),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.int32),
            plan,
        )
