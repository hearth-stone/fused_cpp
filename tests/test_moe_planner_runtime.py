from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import fused_cpp.moe.bf16_tiled as bf16_tiled
from fused_cpp.moe import (
    MoePlannerRuntime,
    PreparedBF16TiledFusedMoEWeights,
    calibrate_moe_planner_quick,
    enable_moe_planner_quick,
    get_default_moe_planner_runtime,
    set_default_moe_planner_runtime,
)


@pytest.fixture(autouse=True)
def _restore_default_runtime():
    previous = set_default_moe_planner_runtime(None)
    try:
        yield
    finally:
        set_default_moe_planner_runtime(previous)


def _weights(*, fused_silu: bool = True) -> PreparedBF16TiledFusedMoEWeights:
    return PreparedBF16TiledFusedMoEWeights(
        w13=(torch.empty((4, 1), dtype=torch.bfloat16), 64, 64),
        w2=(torch.empty((4, 1), dtype=torch.bfloat16), 32, 64),
        fused_silu=fused_silu,
        gemm_backend=1,
        backend_n_tile=8,
        backend_name="arm_sve_bf16",
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.zeros((2, 64), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    return hidden, topk_weights, topk_ids


def _runtime_with_plan(plan):
    runtime = object.__new__(MoePlannerRuntime)
    runtime.plan_for_dispatch = lambda *args, **kwargs: plan
    return runtime


def test_public_runtime_registry_is_reversible_and_validated() -> None:
    runtime = _runtime_with_plan(None)
    assert get_default_moe_planner_runtime() is None
    assert set_default_moe_planner_runtime(runtime) is None
    assert get_default_moe_planner_runtime() is runtime
    assert set_default_moe_planner_runtime(None) is runtime
    assert get_default_moe_planner_runtime() is None
    with pytest.raises(TypeError, match="MoePlannerRuntime"):
        set_default_moe_planner_runtime(object())
    assert callable(calibrate_moe_planner_quick)
    assert callable(enable_moe_planner_quick)


def test_enable_quick_planner_calibrates_before_installing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fused_cpp.moe.planner_runtime as planner_runtime

    calibration = object()
    calibration_result = SimpleNamespace(calibration=calibration, cpu_ids=(4, 5))
    runtime = _runtime_with_plan(None)
    events = []

    def calibrate(*args, **kwargs):
        events.append(("calibrate", args, kwargs))
        return calibration_result

    def construct(*args, **kwargs):
        events.append(("construct", args, kwargs))
        return runtime

    def install(value):
        events.append(("install", value))

    monkeypatch.setattr(planner_runtime, "calibrate_moe_planner_quick", calibrate)
    monkeypatch.setattr(planner_runtime, "MoePlannerRuntime", construct)
    monkeypatch.setattr(planner_runtime, "set_default_moe_planner_runtime", install)

    actual = planner_runtime.enable_moe_planner_quick(
        hidden_size=4096,
        intermediate_size=512,
        global_experts=256,
        local_experts=256,
        cpu_ids=(4, 5),
        mode="tp",
        degree=4,
        output="machine.json",
    )

    assert actual is runtime
    assert [event[0] for event in events] == ["calibrate", "construct", "install"]
    assert events[0][1] == ((4, 5),)
    assert events[1][1] == (calibration,)
    assert events[1][2]["cpu_ids"] == (4, 5)
    assert events[2][1] is runtime


def test_normal_dispatch_uses_plan_v2_when_runtime_accepts_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, topk_weights, topk_ids = _inputs()
    sentinel_plan = object()
    runtime = _runtime_with_plan(sentinel_plan)
    set_default_moe_planner_runtime(runtime)
    calls = []

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)
    monkeypatch.setattr(
        bf16_tiled,
        "_fused_moe_bf16_tiled_impl",
        lambda *args, **kwargs: pytest.fail("legacy dispatcher must not run"),
    )

    def planned(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(hidden, 2)

    monkeypatch.setattr(bf16_tiled, "fused_moe_bf16_tiled_async_plan", planned)
    output = bf16_tiled.fused_moe_bf16_tiled(
        hidden,
        _weights(),
        topk_weights,
        topk_ids,
        num_threads=4,
    )

    assert torch.equal(output, torch.full_like(hidden, 2))
    assert calls[0][0][4] is sentinel_plan


def test_normal_dispatch_preserves_legacy_path_when_runtime_declines_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, topk_weights, topk_ids = _inputs()
    set_default_moe_planner_runtime(_runtime_with_plan(None))
    calls = []

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)

    def legacy(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(hidden, 3)

    monkeypatch.setattr(bf16_tiled, "_fused_moe_bf16_tiled_impl", legacy)
    output = bf16_tiled.fused_moe_bf16_tiled(
        hidden,
        _weights(fused_silu=False),
        topk_weights,
        topk_ids,
        num_threads=4,
    )

    assert torch.equal(output, torch.full_like(hidden, 3))
    assert len(calls) == 1


def test_runtime_compatibility_is_limited_to_calibrated_tp_sve_shape() -> None:
    runtime = object.__new__(MoePlannerRuntime)
    runtime.local_experts = 4
    runtime.global_experts = 4
    runtime.mode = "tp"
    runtime.num_cores = 4
    runtime.hidden_size = 64
    runtime.intermediate_size = 32
    runtime.model = SimpleNamespace(policy=SimpleNamespace(backend_n_tile=8))

    assert runtime._is_compatible(
        _weights(),
        num_threads=4,
        activation="silu",
        global_num_experts=4,
    )
    assert not runtime._is_compatible(
        _weights(),
        num_threads=2,
        activation="silu",
        global_num_experts=4,
    )
    assert not runtime._is_compatible(
        _weights(),
        num_threads=4,
        activation="gelu",
        global_num_experts=4,
    )
