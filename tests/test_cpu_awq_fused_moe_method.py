#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``CPUAWQFusedMoEMethod`` 的端到端 pipeline 测试（Step 4 / 方案 B）。

本测试在不拉起完整 vLLM engine 的前提下，验证新增的
``vllm.model_executor.layers.quantization.awq.CPUAWQFusedMoEMethod`` 的核心
pipeline：

1. ``create_weights``：正确注册 ``w13_*`` / ``w2_*`` 六个 Parameter，形状
   严格匹配 AWQ 原生布局（``[E, H, 2F//8]`` / ``[E, F, H//8]`` 等）。
2. ``weight_loader``：按 AutoAWQ checkpoint 原生形状喂入每个 expert 的
   ``gate_proj`` / ``up_proj`` / ``down_proj`` 的 ``qweight`` / ``qzeros`` /
   ``scales``，loader 应把 w1 写入 w13 的前半 F//8，w3 写入后半 F//8，
   w2 独立写入。
3. ``process_weights_after_loading``：按 N 维把 w13 拆回 gate/up，并构造
   ``AWQFusedMoEImpl``。
4. ``apply``：数值结果与 ``awq_moe_expert_ffn_reference`` 在 W4A8 容差内一致。

TP>1 的切分逻辑通过 monkeypatch 验证两点：
* TP rank=0/1 的切分索引正确；
* 两 rank 合并后结果与 TP=1 单机一致（仅验证切分代数正确性，不跑实际
  all-reduce）。

所有 vLLM 的分布式依赖由 monkeypatch 替身实现，避免初始化真正的 torch
distributed 环境。
"""
from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("vllm", reason="本测试依赖 vLLM 源码树；请在 vLLM 仓库内运行")

from fused_cpp.moe.awq_moe import (
    AWQExpertWeights,
    awq_moe_expert_ffn_reference,
)

_TOL_W4A8_MOE = dict(rtol=5e-2, atol=5e-2, cos_sim_threshold=0.999)


# ── 共享工具（与 test_awq_moe_vs_reference 保持一致的构造手段） ─────────────

_AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_awq_along_n(unpacked: torch.Tensor) -> torch.Tensor:
    assert unpacked.shape[-1] % 8 == 0
    lead = unpacked.shape[:-1]
    n = unpacked.shape[-1]
    reshaped = unpacked.reshape(*lead, n // 8, 8).to(torch.int32) & 0xF
    order_idx = torch.tensor(_AWQ_ORDER, dtype=torch.long)
    picked = reshaped.index_select(-1, order_idx)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32).view(
        *([1] * len(lead)), 1, 8,
    )
    return (picked << shifts).sum(dim=-1).to(torch.int32)


def _make_awq_weight(
    k: int,
    n: int,
    group_size: int,
    scales_dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    w_int4 = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    groups = k // group_size
    z_int4 = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.int32)
    scales_fp32 = (
        torch.rand((groups, n), generator=gen, dtype=torch.float32) * 5e-3 + 1e-3
    )
    qweight = _pack_awq_along_n(w_int4)
    qzeros = _pack_awq_along_n(z_int4)
    return qweight, qzeros, scales_fp32.to(scales_dtype)


def _make_expert(
    h: int,
    f_dim: int,
    g: int,
    scales_dtype: torch.dtype,
    seed: int,
) -> AWQExpertWeights:
    gate_qw, gate_qz, gate_s = _make_awq_weight(h, f_dim, g, scales_dtype, seed)
    up_qw, up_qz, up_s = _make_awq_weight(h, f_dim, g, scales_dtype, seed + 1)
    down_qw, down_qz, down_s = _make_awq_weight(f_dim, h, g, scales_dtype, seed + 2)
    return AWQExpertWeights(
        gate_qweight=gate_qw, gate_qzeros=gate_qz, gate_scales=gate_s,
        up_qweight=up_qw, up_qzeros=up_qz, up_scales=up_s,
        down_qweight=down_qw, down_qzeros=down_qz, down_scales=down_s,
    )


def _random_topk(
    num_tokens: int, num_experts: int, top_k: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=gen)[:top_k]
         for _ in range(num_tokens)]
    ).to(torch.int64)
    raw = torch.randn((num_tokens, top_k), generator=gen, dtype=torch.float32)
    return topk_ids, torch.softmax(raw, dim=-1)


def _assert_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float, atol: float, cos_sim_threshold: float,
) -> None:
    assert actual.shape == ref.shape
    assert actual.dtype == ref.dtype
    for name, t in (("actual", actual), ("ref", ref)):
        assert not torch.isnan(t).any(), f"{name} has NaN"
        assert not torch.isinf(t).any(), f"{name} has Inf"
    diff = (actual.float() - ref.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref.float().abs() + 1e-12)).max().item()
    cos = torch.nn.functional.cosine_similarity(
        actual.flatten().float().unsqueeze(0),
        ref.flatten().float().unsqueeze(0),
    ).item()
    msg = f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} cos={cos:.6f}"
    assert max_abs <= atol or max_rel <= rtol, msg
    assert cos >= cos_sim_threshold, msg


# ── vLLM 依赖的轻量 mock ───────────────────────────────────────────────────

def _patch_vllm_distributed(monkeypatch: pytest.MonkeyPatch, *, tp_rank: int, tp_size: int) -> None:
    """替换 awq.py 内的 ``get_tensor_model_parallel_rank`` / ``get_tp_group``，
    让 weight_loader 能在无真实分布式环境下运行。"""
    import vllm.model_executor.layers.quantization.awq as awq_mod

    fake_group = types.SimpleNamespace(world_size=tp_size)
    monkeypatch.setattr(awq_mod, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(awq_mod, "get_tp_group", lambda: fake_group)


def _build_fake_moe_config() -> object:
    """构造 FusedMoEMethodBase.__init__ 期望的最小 FusedMoEConfig 桩。"""
    # FusedMoEMethodBase.__init__(self, moe) 里持有引用即可；我们只读少量字段。
    return types.SimpleNamespace(disable_inplace=True)


def _instantiate_method(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    group_size: int = 32,
):
    """构造一个可用的 CPUAWQFusedMoEMethod 实例（已 patch 分布式依赖）。"""
    _patch_vllm_distributed(monkeypatch, tp_rank=tp_rank, tp_size=tp_size)
    from vllm.model_executor.layers.quantization.awq import (
        AWQConfig,
        CPUAWQFusedMoEMethod,
    )

    cfg = AWQConfig(
        weight_bits=4, group_size=group_size,
        zero_point=True, modules_to_not_convert=None,
    )
    moe_cfg = _build_fake_moe_config()
    return CPUAWQFusedMoEMethod(cfg, moe_cfg)  # type: ignore[arg-type]


class _FakeLayer(torch.nn.Module):
    """最简 layer 桩：只承接 register_parameter / setattr（类似 FusedMoE）。"""


# ── 测试 1：create_weights 形状契约 ────────────────────────────────────────

def test_create_weights_registers_correct_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_weights 应按 AWQ 原生布局注册 w13_*/w2_* 六个 Parameter。"""
    # Arrange
    num_experts, h, f_dim, g = 4, 128, 64, 32
    method = _instantiate_method(monkeypatch, group_size=g)
    layer = _FakeLayer()

    # Act
    method.create_weights(
        layer=layer, num_experts=num_experts,
        hidden_size=h, intermediate_size_per_partition=f_dim,
        params_dtype=torch.bfloat16,
    )

    # Assert
    assert layer.w13_qweight.shape == (num_experts, h, 2 * f_dim // 8)
    assert layer.w13_qweight.dtype == torch.int32
    assert layer.w13_qzeros.shape == (num_experts, h // g, 2 * f_dim // 8)
    assert layer.w13_qzeros.dtype == torch.int32
    assert layer.w13_scales.shape == (num_experts, h // g, 2 * f_dim)
    assert layer.w13_scales.dtype == torch.bfloat16

    assert layer.w2_qweight.shape == (num_experts, f_dim, h // 8)
    assert layer.w2_qweight.dtype == torch.int32
    assert layer.w2_qzeros.shape == (num_experts, f_dim // g, h // 8)
    assert layer.w2_qzeros.dtype == torch.int32
    assert layer.w2_scales.shape == (num_experts, f_dim // g, h)
    assert layer.w2_scales.dtype == torch.bfloat16

    # weight_loader 已注入
    assert callable(layer.weight_loader)
    assert getattr(layer.weight_loader, "supports_moe_loading", False) is True


# ── 测试 2：weight_loader 的 w13 前/后半写入正确 ────────────────────────────

def _load_expert(
    layer: torch.nn.Module,
    expert: AWQExpertWeights,
    expert_id: int,
) -> None:
    """通过 layer.weight_loader 把单个 expert 的 9 张量塞进 layer。

    模拟 vLLM ``default_loader`` 按 checkpoint key 触发的 6 次调用。
    """
    calls = (
        (layer.w13_qweight, expert.gate_qweight, "experts.N.gate_proj.qweight", "w1"),
        (layer.w13_qzeros,  expert.gate_qzeros,  "experts.N.gate_proj.qzeros",  "w1"),
        (layer.w13_scales,  expert.gate_scales,  "experts.N.gate_proj.scales",  "w1"),
        (layer.w13_qweight, expert.up_qweight,   "experts.N.up_proj.qweight",   "w3"),
        (layer.w13_qzeros,  expert.up_qzeros,    "experts.N.up_proj.qzeros",    "w3"),
        (layer.w13_scales,  expert.up_scales,    "experts.N.up_proj.scales",    "w3"),
        (layer.w2_qweight,  expert.down_qweight, "experts.N.down_proj.qweight", "w2"),
        (layer.w2_qzeros,   expert.down_qzeros,  "experts.N.down_proj.qzeros",  "w2"),
        (layer.w2_scales,   expert.down_scales,  "experts.N.down_proj.scales",  "w2"),
    )
    for param, loaded, name, shard_id in calls:
        layer.weight_loader(param, loaded, name, shard_id, expert_id)


def test_weight_loader_writes_w1_front_w3_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """w1 写入 w13 前半 F//8，w3 写入后半 F//8；w2 独立写入。"""
    # Arrange
    num_experts, h, f_dim, g = 2, 64, 32, 16
    method = _instantiate_method(monkeypatch, group_size=g)
    layer = _FakeLayer()
    method.create_weights(
        layer=layer, num_experts=num_experts, hidden_size=h,
        intermediate_size_per_partition=f_dim, params_dtype=torch.bfloat16,
    )

    experts = [
        _make_expert(h, f_dim, g, torch.bfloat16, seed=e * 13)
        for e in range(num_experts)
    ]

    # Act
    for e, exp in enumerate(experts):
        _load_expert(layer, exp, expert_id=e)

    # Assert：w1 占前半、w3 占后半，w2 原样
    pack_f = f_dim // 8
    for e, exp in enumerate(experts):
        assert torch.equal(
            layer.w13_qweight[e, :, :pack_f], exp.gate_qweight,
        ), f"expert {e}: w1 前半 qweight 写入错误"
        assert torch.equal(
            layer.w13_qweight[e, :, pack_f:], exp.up_qweight,
        ), f"expert {e}: w3 后半 qweight 写入错误"
        assert torch.equal(
            layer.w13_scales[e, :, :f_dim], exp.gate_scales,
        )
        assert torch.equal(
            layer.w13_scales[e, :, f_dim:], exp.up_scales,
        )
        assert torch.equal(
            layer.w13_qzeros[e, :, :pack_f], exp.gate_qzeros,
        )
        assert torch.equal(
            layer.w13_qzeros[e, :, pack_f:], exp.up_qzeros,
        )
        assert torch.equal(layer.w2_qweight[e], exp.down_qweight)
        assert torch.equal(layer.w2_qzeros[e], exp.down_qzeros)
        assert torch.equal(layer.w2_scales[e], exp.down_scales)


# ── 测试 3：完整 pipeline 数值等价性 ─────────────────────────────────────

@pytest.mark.parametrize(
    "scenario",
    [
        (4, 4, 2, 128, 64, 32),     # (T, E, top_k, H, F, g)
        (8, 8, 4, 256, 128, 32),
    ],
    ids=["tiny", "mid"],
)
def test_end_to_end_pipeline_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    scenario: tuple[int, int, int, int, int, int],
) -> None:
    """create_weights → weight_loader → process_weights_after_loading → apply
    的端到端输出与 reference 在 W4A8 容差内一致。"""
    # Arrange
    num_tokens, num_experts, top_k, h, f_dim, g = scenario
    method = _instantiate_method(monkeypatch, group_size=g)
    layer = _FakeLayer()
    method.create_weights(
        layer=layer, num_experts=num_experts, hidden_size=h,
        intermediate_size_per_partition=f_dim, params_dtype=torch.bfloat16,
    )
    torch.manual_seed(0)
    experts = [
        _make_expert(h, f_dim, g, torch.bfloat16, seed=e * 7)
        for e in range(num_experts)
    ]
    for e, exp in enumerate(experts):
        _load_expert(layer, exp, expert_id=e)

    method.process_weights_after_loading(layer)

    hidden = torch.randn(num_tokens, h, dtype=torch.bfloat16) * 0.1
    topk_ids, topk_w = _random_topk(num_tokens, num_experts, top_k, seed=42)

    # Act
    out = method.apply(
        layer=layer, x=hidden, topk_weights=topk_w,
        topk_ids=topk_ids, shared_experts_input=None,
    )
    ref = awq_moe_expert_ffn_reference(hidden, topk_ids, topk_w, experts)

    # Assert
    _assert_close(out, ref, **_TOL_W4A8_MOE)


# ── 测试 4：apply 对 3D 输入的 reshape 契约 ────────────────────────────────

def test_apply_handles_3d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """当调用方传入 ``[B, S, H]`` 时，apply 应内部压成 2D 并还原形状。"""
    # Arrange
    b, s, h, f_dim, g = 2, 3, 64, 32, 16
    num_experts, top_k = 3, 2
    method = _instantiate_method(monkeypatch, group_size=g)
    layer = _FakeLayer()
    method.create_weights(
        layer=layer, num_experts=num_experts, hidden_size=h,
        intermediate_size_per_partition=f_dim, params_dtype=torch.bfloat16,
    )
    experts = [
        _make_expert(h, f_dim, g, torch.bfloat16, seed=e)
        for e in range(num_experts)
    ]
    for e, exp in enumerate(experts):
        _load_expert(layer, exp, expert_id=e)
    method.process_weights_after_loading(layer)

    x_3d = torch.randn(b, s, h, dtype=torch.bfloat16) * 0.1
    topk_ids, topk_w = _random_topk(b * s, num_experts, top_k, seed=11)

    # Act
    out = method.apply(
        layer=layer, x=x_3d, topk_weights=topk_w,
        topk_ids=topk_ids, shared_experts_input=None,
    )

    # Assert：输出应保持 3D 且 shape 与输入一致
    assert out.shape == x_3d.shape
    assert out.dtype == x_3d.dtype


# ── 测试 5：TP=2 切分逻辑的代数正确性 ──────────────────────────────────────

def test_weight_loader_tp2_shards_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP=2 下，rank 0 与 rank 1 应各自拿到 loaded_weight 的 N 维/K 维
    对应前半/后半切片；两 rank 合并后应覆盖完整 weight。"""
    # Arrange：TP=2 的 intermediate_size_per_partition 是原 F 的一半
    full_f = 64
    h, g = 64, 16
    num_experts = 1

    # checkpoint 侧完整权重（gate: [H, full_f//8] 等）
    full_expert = _make_expert(h, full_f, g, torch.bfloat16, seed=777)

    # 两个 rank 的 layer
    layers: list[torch.nn.Module] = []
    for rank in range(2):
        method = _instantiate_method(
            monkeypatch, tp_rank=rank, tp_size=2, group_size=g,
        )
        layer = _FakeLayer()
        method.create_weights(
            layer=layer, num_experts=num_experts, hidden_size=h,
            intermediate_size_per_partition=full_f // 2,   # TP 切后
            params_dtype=torch.bfloat16,
        )
        # 只加载 gate_proj.qweight（沿 N 切的代表），验证切分
        layer.weight_loader(
            layer.w13_qweight, full_expert.gate_qweight,
            "experts.0.gate_proj.qweight", "w1", 0,
        )
        layers.append(layer)

    # Assert：
    #   - rank0 的 w13_qweight[0, :, :F//2//8] 应等于 full_expert.gate_qweight 的前 F//2//8 列
    #   - rank1 的 w13_qweight[0, :, :F//2//8] 应等于后 F//2//8 列
    half_packed = (full_f // 2) // 8
    assert torch.equal(
        layers[0].w13_qweight[0, :, :half_packed],
        full_expert.gate_qweight[:, :half_packed],
    )
    assert torch.equal(
        layers[1].w13_qweight[0, :, :half_packed],
        full_expert.gate_qweight[:, half_packed:],
    )


# ── 测试 6：w2 的 K 维（F）TP 切分 ──────────────────────────────────────────

def test_weight_loader_tp2_w2_shards_along_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TP=2 下，down_proj 应沿 K=F 维切：rank0 拿前 F/2 行，rank1 拿后 F/2 行。"""
    # Arrange
    full_f = 64
    h, g = 64, 16
    full_expert = _make_expert(h, full_f, g, torch.bfloat16, seed=888)

    layers: list[torch.nn.Module] = []
    for rank in range(2):
        method = _instantiate_method(
            monkeypatch, tp_rank=rank, tp_size=2, group_size=g,
        )
        layer = _FakeLayer()
        method.create_weights(
            layer=layer, num_experts=1, hidden_size=h,
            intermediate_size_per_partition=full_f // 2,
            params_dtype=torch.bfloat16,
        )
        layer.weight_loader(
            layer.w2_qweight, full_expert.down_qweight,
            "experts.0.down_proj.qweight", "w2", 0,
        )
        layers.append(layer)

    # Assert
    half_f = full_f // 2
    assert torch.equal(
        layers[0].w2_qweight[0],
        full_expert.down_qweight[:half_f, :],
    )
    assert torch.equal(
        layers[1].w2_qweight[0],
        full_expert.down_qweight[half_f:, :],
    )


# ── 测试 7（回归）：每个 Parameter 的 weight_loader 都被自定义 closure 覆盖 ─

def test_param_weight_loader_is_custom_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM 真实加载路径通过 ``param.weight_loader(...)`` 触发加载，因此
    必须把自定义 loader 挂到每个 Parameter 的 ``weight_loader`` 属性上。

    这个测试同时防御 ``FusedMoE.create_weights`` 传入的 ``extra_weight_attrs``
    里可能自带 ``weight_loader``（指向 ``FusedMoE.weight_loader`` 默认实现）
    导致我们的自定义 loader 被覆盖的回归。
    """
    # Arrange：模拟 FusedMoE 传入默认 weight_loader（我们必须覆盖它）
    num_experts, h, f_dim, g = 2, 64, 32, 16
    method = _instantiate_method(monkeypatch, group_size=g)
    layer = _FakeLayer()

    sentinel_default_loader_called = False

    def _sentinel_default_loader(*args: object, **kwargs: object) -> bool:
        # 如果这个被调用说明我们的覆盖失败了。
        nonlocal sentinel_default_loader_called
        sentinel_default_loader_called = True
        return False

    # Act：模拟 FusedMoE 的典型调用方式——把默认 loader 塞进 extra_weight_attrs
    method.create_weights(
        layer=layer, num_experts=num_experts, hidden_size=h,
        intermediate_size_per_partition=f_dim, params_dtype=torch.bfloat16,
        weight_loader=_sentinel_default_loader,
    )

    # Assert：每个 Parameter 的 weight_loader 都应是我们的 closure，而非默认
    custom = layer.weight_loader
    assert callable(custom)
    assert getattr(custom, "supports_moe_loading", False) is True

    for name in (
        "w13_qweight", "w13_qzeros", "w13_scales",
        "w2_qweight", "w2_qzeros", "w2_scales",
    ):
        param = getattr(layer, name)
        assert callable(param.weight_loader), f"{name}: weight_loader 不可调用"
        assert param.weight_loader is custom, (
            f"{name}: Parameter 上的 weight_loader 未被覆盖为自定义 closure"
        )

    # 触发一次真实加载，sentinel 永远不应被调用
    expert = _make_expert(h, f_dim, g, torch.bfloat16, seed=0)
    layer.w13_qweight.weight_loader(
        layer.w13_qweight, expert.gate_qweight,
        "experts.0.gate_proj.qweight", "w1", 0,
    )
    assert not sentinel_default_loader_called, (
        "默认 weight_loader 被误调用，覆盖逻辑失败"
    )
    # 校验真实写入效果（w1 前半 qweight 应与 checkpoint 一致）
    assert torch.equal(
        layer.w13_qweight[0, :, : f_dim // 8], expert.gate_qweight,
    )
