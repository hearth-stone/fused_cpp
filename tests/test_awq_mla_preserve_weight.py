#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``AWQLinearMethod.process_weights_after_loading`` + MLA ``kv_b_proj`` 适配测试。

覆盖点：

1. **MLA kv_b_proj 兼容**：当 ``layer._preserve_original_weight=True`` 且
   ``VLLM_CPU_AWQ_USE_FUSED_CPP=1`` 时，必须在 ``process_weights_after_loading``
   中反量化出 ``layer.weight`` 并挂 ``layer.cpu_linear``（MLA 吸收路径依赖）。
2. **数值正确**：反量化后的 bf16 稠密权重与 ``dequant_awq_to_bf16`` 参考一致。
3. **shape 契约**：``layer.weight`` 形状为 ``[N, K]``，与 ``nn.Linear.weight`` 一致。
4. **不影响其他 Linear**：未打 ``_preserve_original_weight`` 标记的普通 AWQ
   Linear 不应挂 ``layer.weight`` / ``layer.cpu_linear``，保持 w4a8 快路径。
5. **环境变量守卫**：当 ``VLLM_CPU_AWQ_USE_FUSED_CPP=0`` 时，即便标记也不触发
   （回到 GPU marlin 路径的职责范围，在 CPU 上不应被激活此测试所在场景）。

规范对齐：仅依赖 vLLM + fused_cpp 本体，不引入 LLMEngine / ModelRunner。
"""
from __future__ import annotations

import importlib

import pytest
import torch

from fused_cpp.moe import dequant_awq_to_bf16

vllm_awq = pytest.importorskip(
    "vllm.model_executor.layers.quantization.awq",
    reason="需要在 vLLM 可导入的环境下运行集成测试",
)
vllm_envs = importlib.import_module("vllm.envs")
AWQConfig = vllm_awq.AWQConfig
AWQLinearMethod = vllm_awq.AWQLinearMethod


_AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_awq_along_n(unpacked: torch.Tensor) -> torch.Tensor:
    """沿 N 维按 AWQ interleave 顺序将 int4 (0..15) pack 为 int32。"""
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


def _build_awq_params(
    k: int, n: int, group_size: int, *, seed: int = 0,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    """构造随机 AWQ (qweight, qzeros, scales=bf16) 三件套。"""
    gen = torch.Generator().manual_seed(seed)
    w_int4 = torch.randint(0, 16, (k, n), generator=gen, dtype=torch.int32)
    groups = k // group_size
    z_int4 = torch.randint(0, 16, (groups, n), generator=gen, dtype=torch.int32)
    scales_bf16 = (
        torch.rand((groups, n), generator=gen, dtype=torch.float32) * 0.02 + 0.001
    ).to(torch.bfloat16)
    qweight = torch.nn.Parameter(_pack_awq_along_n(w_int4), requires_grad=False)
    qzeros = torch.nn.Parameter(_pack_awq_along_n(z_int4), requires_grad=False)
    scales = torch.nn.Parameter(scales_bf16, requires_grad=False)
    return qweight, qzeros, scales


def _make_awq_method() -> AWQLinearMethod:
    """以最小合法配置构造 AWQLinearMethod 实例。"""
    cfg = AWQConfig(weight_bits=4, group_size=128, zero_point=True,
                    modules_to_not_convert=[])
    return AWQLinearMethod(cfg)


class _FakeLinear(torch.nn.Module):
    """最小 Linear stand-in，用于替代 ColumnParallelLinear / RowParallelLinear。

    只暴露 ``process_weights_after_loading`` 需要读取的属性：
    ``qweight`` / ``qzeros`` / ``scales`` / ``_preserve_original_weight``。
    """

    def __init__(
        self,
        k: int,
        n: int,
        group_size: int,
        *,
        preserve: bool,
    ) -> None:
        super().__init__()
        qw, qz, sc = _build_awq_params(k, n, group_size)
        self.qweight = qw
        self.qzeros = qz
        self.scales = sc
        if preserve:
            self._preserve_original_weight = True


# ── 1. MLA kv_b_proj 场景：必须生成 layer.weight 与 layer.cpu_linear ──

@pytest.mark.parametrize(
    "k,n,group_size",
    [
        (128, 64, 128),
        (512, 256, 128),
    ],
    ids=["tiny", "mid"],
)
def test_preserve_weight_materializes_dense_weight_and_cpu_linear(
    monkeypatch: pytest.MonkeyPatch,
    k: int,
    n: int,
    group_size: int,
) -> None:
    """MLA kv_b_proj 类 Linear 经 process_weights_after_loading 应暴露
    ``layer.weight`` (bf16, shape=[N, K]) 与可调用的 ``layer.cpu_linear``。"""
    monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True)
    # 强制 CPU 平台判定（在 macOS / AArch64 上都应返回 True，但这里显式保证）
    import vllm.platforms as _plat
    monkeypatch.setattr(_plat.current_platform, "is_cpu", lambda: True)

    layer = _FakeLinear(k=k, n=n, group_size=group_size, preserve=True)
    method = _make_awq_method()

    # Act
    method.process_weights_after_loading(layer)

    # Assert：weight 存在且 shape 正确
    assert hasattr(layer, "weight"), "layer.weight 未生成"
    assert layer.weight.dtype == torch.bfloat16
    assert tuple(layer.weight.shape) == (n, k), (
        f"expect [N={n}, K={k}], got {tuple(layer.weight.shape)}"
    )

    # Assert：cpu_linear 存在且可调用
    assert hasattr(layer, "cpu_linear"), "layer.cpu_linear 未挂载"
    assert callable(layer.cpu_linear)

    # 走一次 cpu_linear 确保函数路径完整（数值结果在第 2 个测试里做等价性）
    x = torch.randn(4, k, dtype=torch.bfloat16)
    out = layer.cpu_linear(x, layer.weight, None)
    assert out.shape == (4, n)


# ── 2. 反量化数值与 fused_cpp 参考一致 ──

def test_materialized_weight_matches_dequant_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``layer.weight`` 反量化结果必须与 fused_cpp.dequant_awq_to_bf16 一致（
    经 transpose 后）。"""
    monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True)
    import vllm.platforms as _plat
    monkeypatch.setattr(_plat.current_platform, "is_cpu", lambda: True)

    k, n, group_size = 256, 128, 64
    layer = _FakeLinear(k=k, n=n, group_size=group_size, preserve=True)
    method = _make_awq_method()

    # 参考：从 qweight/qzeros/scales 自己算一遍 [K, N] bf16
    w_kn_ref = dequant_awq_to_bf16(layer.qweight, layer.qzeros, layer.scales)

    # Act
    method.process_weights_after_loading(layer)

    # Assert：layer.weight == ref.t() （逐 bit 一致，因为来自同一个 dequant 实现）
    assert torch.equal(layer.weight, w_kn_ref.t().contiguous())


# ── 3. 非 MLA 场景不受影响 ──

def test_regular_awq_linear_does_not_materialize_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """普通 AWQ Linear（无 ``_preserve_original_weight``）必须保持原路径，
    不挂 ``layer.weight`` / ``layer.cpu_linear``。"""
    monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", True)
    import vllm.platforms as _plat
    monkeypatch.setattr(_plat.current_platform, "is_cpu", lambda: True)

    layer = _FakeLinear(k=128, n=64, group_size=128, preserve=False)
    method = _make_awq_method()
    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "weight") or layer.weight is None, (
        "普通 AWQ Linear 不应生成 layer.weight"
    )
    assert not hasattr(layer, "cpu_linear"), (
        "普通 AWQ Linear 不应挂 layer.cpu_linear"
    )


# ── 4. 环境变量守卫 ──

def test_preserve_weight_noop_when_fused_cpp_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``VLLM_CPU_AWQ_USE_FUSED_CPP=0`` 时即便打了 preserve 标记也不应触发
    CPU 反量化（此时 vLLM 走 GPU marlin，由 GPU 路径自己处理）。"""
    monkeypatch.setattr(vllm_envs, "VLLM_CPU_AWQ_USE_FUSED_CPP", False)

    layer = _FakeLinear(k=128, n=64, group_size=128, preserve=True)
    method = _make_awq_method()
    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "weight") or layer.weight is None
    assert not hasattr(layer, "cpu_linear")
