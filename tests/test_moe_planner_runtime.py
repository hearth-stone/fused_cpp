from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

import fused_cpp.moe.bf16_tiled as bf16_tiled
from fused_cpp.moe import (
    MoePlannerRuntime,
    PreparedBF16TiledFusedMoEWeights,
    PreparedBF16TiledRoutedSharedMoEWeights,
    PreparedW8A16TiledFusedMoEWeights,
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


def _w8_weights() -> PreparedW8A16TiledFusedMoEWeights:
    scales = torch.empty((4, 256), dtype=torch.float32)
    return PreparedW8A16TiledFusedMoEWeights(
        w13=(torch.empty((4, 1), dtype=torch.int8), 64, 64, scales),
        w2=(torch.empty((4, 1), dtype=torch.int8), 32, 64, scales),
        backend_n_tile=8,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.zeros((2, 64), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    return hidden, topk_weights, topk_ids


def _routed_shared_weights() -> PreparedBF16TiledRoutedSharedMoEWeights:
    packed = PreparedBF16TiledFusedMoEWeights(
        w13=(torch.empty((5, 1), dtype=torch.bfloat16), 64, 64),
        w2=(torch.empty((5, 1), dtype=torch.bfloat16), 32, 64),
        fused_silu=True,
        gemm_backend=1,
        backend_n_tile=8,
        backend_name="arm_sve_bf16",
    )
    return PreparedBF16TiledRoutedSharedMoEWeights(
        packed=packed,
        routed_experts=4,
        shared_expert_id=4,
    )


def test_swiglu_limit_validation_is_explicit_and_sve_only() -> None:
    weights = _weights()

    assert bf16_tiled._native_activation_name("silu", weights, None) == "silu"
    assert bf16_tiled._native_activation_name("silu", weights, 0.0) == "silu"
    assert bf16_tiled._native_activation_name("silu", weights, 10.0) == "silu_clamp10"
    with pytest.raises(ValueError, match="swiglu_limit=10.0"):
        bf16_tiled._native_activation_name("silu", weights, 7.0)
    with pytest.raises(ValueError, match="activation='silu'"):
        bf16_tiled._native_activation_name("gelu", weights, 10.0)

    neon_weights = PreparedBF16TiledFusedMoEWeights(
        w13=weights.w13,
        w2=weights.w2,
        fused_silu=True,
        gemm_backend=0,
        backend_n_tile=8,
        backend_name="arm_neon_bf16",
    )
    with pytest.raises(ValueError, match="SVE BF16"):
        bf16_tiled._native_activation_name("silu", neon_weights, 10.0)


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
        shared_experts=1,
        output="machine.json",
    )

    assert actual is runtime
    assert [event[0] for event in events] == ["calibrate", "construct", "install"]
    assert events[0][1] == ((4, 5),)
    assert events[1][1] == (calibration,)
    assert events[1][2]["cpu_ids"] == (4, 5)
    assert events[1][2]["shared_experts"] == 1
    assert events[2][1] is runtime


def test_runtime_planner_initialization_precomputes_dense_t_iso_table() -> None:
    class Model:
        supported_widths = (1, 2, 4)

        def __init__(self):
            self.entries = {}

        def T_iso(self, routes, threads):
            self.entries[(routes, threads)] = float(routes * threads)

        def export_t_iso_cache(self):
            return dict(self.entries)

    runtime = object.__new__(MoePlannerRuntime)
    runtime._lock = threading.RLock()
    runtime.model = Model()
    runtime._initialization_diagnostics = {}
    runtime._persist_cost_cache_if_needed = lambda: None

    first = runtime.initialize_planner(4)
    second = runtime.initialize_planner(4)

    assert first["initialized"] is True
    assert first["max_routes"] == 4
    assert first["generated_entries"] == 12
    assert first["total_entries"] == 12
    assert second["generated_entries"] == 0


def test_fixed_planner_threads_environment_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import fused_cpp.moe.planner_runtime as planner_runtime

    monkeypatch.delenv("FUSED_CPP_MOE_PLANNER_FIXED_THREADS", raising=False)
    assert planner_runtime._fixed_planner_threads_from_env() is None
    monkeypatch.setenv("FUSED_CPP_MOE_PLANNER_FIXED_THREADS", "0")
    assert planner_runtime._fixed_planner_threads_from_env() is None
    monkeypatch.setenv("FUSED_CPP_MOE_PLANNER_FIXED_THREADS", "8")
    assert planner_runtime._fixed_planner_threads_from_env() == 8
    monkeypatch.setenv("FUSED_CPP_MOE_PLANNER_FIXED_THREADS", "off")
    with pytest.raises(ValueError, match="non-negative integer"):
        planner_runtime._fixed_planner_threads_from_env()


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


def test_w8a16_operator_uses_installed_plan_and_forwards_cache_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, topk_weights, topk_ids = _inputs()
    sentinel_plan = object()
    set_default_moe_planner_runtime(_runtime_with_plan(sentinel_plan))
    calls = []

    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)

    def planned(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(hidden, 5)

    monkeypatch.setattr(bf16_tiled, "fused_moe_w8a16_tiled_async_plan", planned)
    output = bf16_tiled.fused_moe_w8a16_tiled(
        hidden,
        _w8_weights(),
        topk_weights,
        topk_ids,
        num_threads=4,
        cache_dequant=True,
    )

    assert torch.equal(output, torch.full_like(hidden, 5))
    assert calls[0][0][4] is sentinel_plan
    assert calls[0][1]["cache_dequant"] is True


def test_w8a16_operator_requires_compatible_installed_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, topk_weights, topk_ids = _inputs()
    monkeypatch.setattr(bf16_tiled, "_HAS_BF16_TILED_FUSED_MOE", True)

    with pytest.raises(RuntimeError, match="requires an installed MoePlannerRuntime"):
        bf16_tiled.fused_moe_w8a16_tiled(hidden, _w8_weights(), topk_weights, topk_ids)

    set_default_moe_planner_runtime(_runtime_with_plan(None))
    with pytest.raises(RuntimeError, match="incompatible with this W8A16 shape"):
        bf16_tiled.fused_moe_w8a16_tiled(hidden, _w8_weights(), topk_weights, topk_ids)


def test_generic_tiled_operator_selects_prepared_weight_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, topk_weights, topk_ids = _inputs()
    calls = []

    def bf16(*args, **kwargs):
        calls.append(("bf16", args, kwargs))
        return torch.full_like(hidden, 6)

    def w8(*args, **kwargs):
        calls.append(("w8a16", args, kwargs))
        return torch.full_like(hidden, 7)

    monkeypatch.setattr(bf16_tiled, "fused_moe_bf16_tiled", bf16)
    monkeypatch.setattr(bf16_tiled, "fused_moe_w8a16_tiled", w8)

    bf16_output = bf16_tiled.fused_moe_tiled(hidden, _weights(), topk_weights, topk_ids)
    w8_output = bf16_tiled.fused_moe_tiled(
        hidden,
        _w8_weights(),
        topk_weights,
        topk_ids,
        cache_dequant=True,
    )

    assert torch.equal(bf16_output, torch.full_like(hidden, 6))
    assert torch.equal(w8_output, torch.full_like(hidden, 7))
    assert [call[0] for call in calls] == ["bf16", "w8a16"]
    assert calls[1][2]["cache_dequant"] is True
    with pytest.raises(ValueError, match="only by W8A16"):
        bf16_tiled.fused_moe_tiled(
            hidden,
            _weights(),
            topk_weights,
            topk_ids,
            cache_dequant=True,
        )
    with pytest.raises(ValueError, match="does not support w13_bias"):
        bf16_tiled.fused_moe_tiled(
            hidden,
            _w8_weights(),
            topk_weights,
            topk_ids,
            w13_bias=torch.zeros(64),
        )


def test_combined_dispatch_appends_shared_route_and_uses_plan_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden, topk_weights, topk_ids = _inputs()
    sentinel_plan = object()
    runtime = _runtime_with_plan(None)
    runtime.plan_for_shared_dispatch = lambda *args, **kwargs: sentinel_plan
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
        return torch.full_like(hidden, 4)

    monkeypatch.setattr(bf16_tiled, "fused_moe_bf16_tiled_async_plan", planned)
    output = bf16_tiled.fused_moe_bf16_tiled_with_shared(
        hidden,
        _routed_shared_weights(),
        topk_weights,
        topk_ids,
        num_threads=4,
        routed_scaling_factor=2.5,
    )

    assert torch.equal(output, torch.full_like(hidden, 4))
    combined_weights = calls[0][0][2]
    combined_ids = calls[0][0][3]
    assert torch.equal(combined_ids, torch.tensor([[0, 4], [1, 4]], dtype=torch.int32))
    assert torch.equal(combined_weights, torch.tensor([[2.5, 1.0], [2.5, 1.0]]))
    assert calls[0][0][4] is sentinel_plan


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
    assert runtime._is_compatible(
        _w8_weights(),
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


def test_shared_runtime_declines_non_matrix_routes() -> None:
    runtime = object.__new__(MoePlannerRuntime)
    runtime.shared_experts = 1
    runtime.local_experts = 4
    runtime.global_experts = 4
    runtime.mode = "tp"
    runtime.num_cores = 4
    runtime.hidden_size = 64
    runtime.intermediate_size = 32
    runtime.model = SimpleNamespace(policy=SimpleNamespace(backend_n_tile=8))

    assert (
        runtime.plan_for_shared_dispatch(
            _routed_shared_weights(),
            torch.tensor([4], dtype=torch.int32),
            num_threads=4,
            activation="silu",
        )
        is None
    )
