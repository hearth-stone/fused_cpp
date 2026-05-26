# -*- coding: utf-8 -*-
"""SDPA 多版本等价性测试矩阵。

矩阵维度：
  * ``sdpa_version`` —— 从注册表自动注入（见
    :mod:`tests.conftest::pytest_generate_tests`）；
  * ``shape``        —— 参数化，覆盖标准 + MLA + L != S；
  * ``dtype``        —— ``torch.float32`` / ``torch.bfloat16``；
  * ``is_causal``    —— True / False。

参考实现：使用注册表中的 ``naive_torch``（与 C++ 内核同 ``causal_offset =
S - L`` 语义），而**不**使用 ``F.scaled_dot_product_attention``——后者在
``L != S`` + ``is_causal=True`` 时采用 upper-left 语义，会产生歧义。

能力位过滤：
  * ``dtype not in info.supports_dtypes`` → skip
  * ``is_causal=True and not info.supports_causal`` → skip
  * ``E != Ev (MLA shape) and not info.supports_mla_shape`` → skip
"""
from __future__ import annotations

import pytest
import torch

from fused_cpp.sdpa_registry import VersionInfo, get_sdpa_version

from tests.conftest import assert_tensor_close, call_sdpa_version


# ── 测试矩阵：shape × dtype × is_causal ────────────────────────────────

# 「快」shape：默认收集；「慢」shape 通过 mark.slow 单独驱动。
FAST_SHAPES = [
    # (B, N, L, S, E, Ev)
    (1, 8, 1, 1, 64, 64),       # 极端短：L=S=1
    (2, 8, 16, 16, 64, 64),     # 标准小
    (1, 4, 8, 16, 32, 32),      # L != S
    (2, 4, 8, 8, 64, 64),       # 标准多 batch
    (1, 4, 16, 16, 192, 128),   # MLA 小（qk != v）
]

SLOW_SHAPES = [
    (2, 8, 128, 128, 64, 64),
    (1, 16, 512, 512, 192, 128),
    (1, 16, 1024, 1024, 192, 128),
]

DTYPES = [torch.float32, torch.bfloat16]
CAUSAL_VALUES = [False, True]


# 用作参考实现的版本名（必须存在于注册表）
REF_VERSION = "naive_torch"


# ── 辅助 ──────────────────────────────────────────────────────────────────


def _check_capabilities(
    info: VersionInfo, dtype: torch.dtype, is_causal: bool, shape
) -> None:
    """根据 :class:`VersionInfo` 能力位决定是否 skip。"""
    B, N, L, S, E, Ev = shape
    is_mla = E != Ev
    ok, reason = info.supports(
        dtype=dtype, is_causal=is_causal, mla_shape=is_mla
    )
    if not ok:
        pytest.skip(reason)


def _shape_id(shape) -> str:
    B, N, L, S, E, Ev = shape
    return f"B{B}-N{N}-L{L}-S{S}-E{E}-Ev{Ev}"


def _dtype_id(dt: torch.dtype) -> str:
    return {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
    }.get(dt, str(dt))


def _make_qkv(shape, dtype, seed: int = 42):
    B, N, L, S, E, Ev = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, N, L, E, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(B, N, S, E, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, N, S, Ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


# ── 等价性测试 ────────────────────────────────────────────────────────────


# PyTorch 内置 SDPA 走上层 ``F.scaled_dot_product_attention`` 的版本集合：
# 在 ``is_causal=True`` 且 ``L != S`` 时它们使用 **upper-left** causal，
# 与本框架其它版本的 ``causal_offset = S - L``（lower-right）语义
# 不一致。该集合用于统一跳过语义不兼容的子集，避免与参考实现直接互比。
_PYTORCH_BUILTIN_SDPA_NAMES = frozenset({"pytorch_sdpa", "pytorch_sdpa_math"})


@pytest.mark.equiv
@pytest.mark.parametrize("shape", FAST_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("is_causal", CAUSAL_VALUES,
                         ids=["noncausal", "causal"])
def test_sdpa_versions_equiv_fast(sdpa_version, shape, dtype, is_causal):
    """快速等价性矩阵：每个版本对照 ``naive_torch`` 参考实现。"""
    info = sdpa_version
    _check_capabilities(info, dtype, is_causal, shape)

    B, N, L, S, E, Ev = shape

    # ── 语义兼容性过滤 ──
    # PyTorch 内置 SDPA（``pytorch_sdpa`` / ``pytorch_sdpa_math``）在
    # ``is_causal=True`` 且 ``L != S`` 时使用 upper-left causal（与本框架的
    # lower-right ``S - L`` offset 不一致）。直接互比会产生大幅偏差；
    # 这里跳过该子集，但保留 L == S 时的对比。
    if info.name in _PYTORCH_BUILTIN_SDPA_NAMES and is_causal and L != S:
        pytest.skip(
            f"{info.name} uses upper-left causal mask when L != S; "
            "skipped to avoid semantic mismatch with the lower-right "
            "convention used by all other versions in this framework."
        )

    # 「自我参考」：当 sdpa_version 自身就是 naive_torch 时，跳过（无意义）。
    # 但同时为防止 naive_torch 失效，使用 C++ naive 兜底；
    # 若 C++ 不可用且当前测的就是 naive_torch，则与 PyTorch SDPA 比较，
    # 仅在 L == S 且 ! causal 这种 PyTorch 与本框架一致的语义下做。
    q, k, v = _make_qkv(shape, dtype)
    out = call_sdpa_version(info, q, k, v, is_causal=is_causal)

    if info.name == REF_VERSION:
        # 用 C++ naive 作为对照（如果可用），否则跟 pytorch_sdpa 在 L==S 时比较
        try:
            ref_info = get_sdpa_version("naive")
        except KeyError:
            ref_info = get_sdpa_version("pytorch_sdpa")
        if ref_info.source == "cpp":
            ref = call_sdpa_version(ref_info, q, k, v, is_causal=is_causal)
        else:
            if is_causal and L != S:
                pytest.skip(
                    "naive_torch self-reference: PyTorch SDPA uses upper-left "
                    "causal when L != S, semantics differ; skipped."
                )
            ref = call_sdpa_version(ref_info, q, k, v, is_causal=is_causal)
    else:
        ref_info = get_sdpa_version(REF_VERSION)
        ref = call_sdpa_version(ref_info, q, k, v, is_causal=is_causal)

    ctx = (
        f"version={info.name} source={info.source} shape={_shape_id(shape)} "
        f"dtype={_dtype_id(dtype)} is_causal={is_causal}"
    )
    metrics = assert_tensor_close(out, ref, dtype=dtype, context=ctx)
    # 留下一个非空 print 让 -s 模式可见，但不污染默认 -q 输出
    if False:  # pragma: no cover
        print(ctx, metrics)


# ── 大 shape 等价性（默认 skip，需 -m slow）──


@pytest.mark.equiv
@pytest.mark.slow
@pytest.mark.parametrize("shape", SLOW_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("is_causal", CAUSAL_VALUES,
                         ids=["noncausal", "causal"])
def test_sdpa_versions_equiv_slow(sdpa_version, shape, dtype, is_causal):
    """大 shape 等价性矩阵：默认跳过，仅在 ``-m slow`` 时执行。"""
    info = sdpa_version
    _check_capabilities(info, dtype, is_causal, shape)

    q, k, v = _make_qkv(shape, dtype)
    out = call_sdpa_version(info, q, k, v, is_causal=is_causal)

    ref_name = "naive" if info.name != "naive" else REF_VERSION
    try:
        ref_info = get_sdpa_version(ref_name)
    except KeyError:
        ref_info = get_sdpa_version(REF_VERSION)
    ref = call_sdpa_version(ref_info, q, k, v, is_causal=is_causal)

    ctx = (
        f"version={info.name} source={info.source} shape={_shape_id(shape)} "
        f"dtype={_dtype_id(dtype)} is_causal={is_causal}"
    )
    assert_tensor_close(out, ref, dtype=dtype, context=ctx)


# ── 注册表健康度检查（不需要 sdpa_version fixture）──


@pytest.mark.equiv
def test_sdpa_registry_has_expected_versions():
    """框架默认应注册以下五个版本（部分可能为 cpp/python）。"""
    from fused_cpp import available_sdpa_versions

    names = {vi.name for vi in available_sdpa_versions()}
    expected = {
        "naive_torch", "pytorch_sdpa", "pytorch_sdpa_math",
        "naive", "flash1", "flash2",
    }
    assert expected.issubset(names), (
        f"missing versions: {expected - names}; got {sorted(names)}"
    )


# ── flash2_neon vs flash2 交叉验证（NEON 重排误差边界）─────────────────


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算 ``a`` 与 ``b`` 的逐元素最大绝对误差（fp32 精度）。"""
    return float((a.detach().float() - b.detach().float()).abs().max().item())


@pytest.mark.equiv
@pytest.mark.parametrize("shape", FAST_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("is_causal", CAUSAL_VALUES,
                         ids=["noncausal", "causal"])
def test_flash2_neon_close_to_flash2(shape, dtype, is_causal):
    """``flash2_neon`` 与 ``flash2`` 的累加顺序差异在 2× 容差内。

    NEON 向量化（4 路 ``vfmaq_f32`` / 8 路 ``vbfdotq_f32``）改变了
    ``Q*K^T`` 与 ``output_acc`` 的累加顺序，导致与标量 ``flash2`` 在
    bf16 / 长序列下出现非零的逐元素误差。本测试断言：

        max_abs(flash2_neon, naive_torch)
            <=  2 * max_abs(flash2, naive_torch)  + atol_floor

    其中 ``atol_floor`` 取自 :data:`tests.conftest.SDPA_TOLERANCE`，避免
    在 ``flash2`` 与 ``naive_torch`` 已经逐元素一致（误差 ≈ 0）时把
    NEON 路径的浮点累加顺序差直接判 fail。
    """
    from tests.conftest import SDPA_TOLERANCE

    try:
        neon_info = get_sdpa_version("flash2_neon")
        flash2_info = get_sdpa_version("flash2")
        ref_info = get_sdpa_version(REF_VERSION)
    except KeyError as exc:
        pytest.skip(f"required version not registered: {exc!r}")

    # 能力位检查（与 flash2 对齐：fp32+bf16 / causal / mask / MLA）
    _check_capabilities(neon_info, dtype, is_causal, shape)
    _check_capabilities(flash2_info, dtype, is_causal, shape)

    q, k, v = _make_qkv(shape, dtype)
    out_neon = call_sdpa_version(neon_info, q, k, v, is_causal=is_causal)
    out_flash2 = call_sdpa_version(flash2_info, q, k, v, is_causal=is_causal)
    out_ref = call_sdpa_version(ref_info, q, k, v, is_causal=is_causal)

    err_neon = _max_abs(out_neon, out_ref)
    err_flash2 = _max_abs(out_flash2, out_ref)
    tol = SDPA_TOLERANCE.get(dtype, SDPA_TOLERANCE[torch.float32])
    bound = 2.0 * err_flash2 + tol.atol

    ctx = (
        f"shape={_shape_id(shape)} dtype={_dtype_id(dtype)} "
        f"is_causal={is_causal} err_neon={err_neon:.3e} "
        f"err_flash2={err_flash2:.3e} bound={bound:.3e}"
    )
    assert err_neon <= bound, (
        f"flash2_neon vs naive_torch error {err_neon:.3e} exceeds "
        f"2*flash2_err + atol = {bound:.3e} ({ctx})"
    )


# ── attn_mask 等价性测试 ─────────────────────────────────────────────────
#
# P1-2 把 `mask_ptr != nullptr` / `is_causal` 提到模板参数里靠 if constexpr
# 编译期消去——这意味着 mask 路径的代码会被实例化到一个**单独的**模板
# 副本里。注册时框架既支持 attn_mask 又支持 causal 的版本（默认所有
# C++ flash 系列都是这样），需要新增 mask 不为空的等价性用例确保模板
# 特化没改变数值行为。
#
# 覆盖 4 个组合：(causal, mask non-null) ∈ {F, T} × {T}（mask 永远非空——
# mask 为空的路径已被 test_sdpa_versions_equiv_fast 覆盖）。
# Shape 选小一点的 N=4 / L=S=32 / E=Ev=64 既能踩到 8x8 主路径也能踩到
# tail（Sc_cur=32 不是 64 的倍数则触发 tail，这里 32=4*8 走主路径 +
# 末尾 0 tail，Lc_eff=8 所以 inner 全跑 Lq_eff=8）。

MASK_SHAPES = [
    (1, 4, 32, 32, 64, 64),    # 主路径 + 8x8
    (1, 2, 12, 20, 64, 64),    # tail：Lq=12（一个 8 + 一个 4 tail）、Sk=20 → 8x8 + 8x4 + 8x4 tail
]


@pytest.mark.equiv
@pytest.mark.parametrize("shape", MASK_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("dtype", DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("is_causal", CAUSAL_VALUES,
                         ids=["noncausal", "causal"])
def test_sdpa_versions_equiv_with_attn_mask(
    sdpa_version, shape, dtype, is_causal
):
    """带 fp32 attn_mask 时各 SDPA 版本相对 ``naive_torch`` 仍然等价。

    动机：``flash2_neon_l3kv*`` 在 P1 优化中把 ``mask_ptr != nullptr`` 提
    到模板参数里靠 ``if constexpr`` 编译期消去。需要测例确保 mask 模板
    实例（kHasMask=true）的数值与原标量分支等价。
    """
    info = sdpa_version
    _check_capabilities(info, dtype, is_causal, shape)

    B, N, L, S, E, Ev = shape

    # PyTorch 内置 SDPA 在 ``L != S`` + ``is_causal=True`` 用 upper-left 语义
    if info.name in _PYTORCH_BUILTIN_SDPA_NAMES and is_causal and L != S:
        pytest.skip(
            f"{info.name} uses upper-left causal mask when L != S; "
            "skipped for semantic mismatch."
        )

    # PyTorch 内置 SDPA 在 ``is_causal=True`` 时拒绝显式 ``attn_mask``
    # （F.scaled_dot_product_attention 的 API 约束）。本测试目标是 C++ kernel
    # 自身 mask 路径的数值正确性，而非 PyTorch API 兼容性，故这里 skip。
    if info.name in _PYTORCH_BUILTIN_SDPA_NAMES and is_causal:
        pytest.skip(
            f"{info.name} disallows attn_mask + is_causal=True at API level"
        )

    ok, reason = info.supports(attn_mask=True)
    if not ok:
        pytest.skip(reason)

    q, k, v = _make_qkv(shape, dtype)
    g = torch.Generator(device="cpu").manual_seed(123)
    # additive mask 必须是 fp32（C++ kernel 假设），数值幅度小一点避免
    # 把整行打成 -inf。
    mask = torch.randn(B, N, L, S, generator=g, dtype=torch.float32) * 0.1

    out = call_sdpa_version(
        info, q, k, v, attn_mask=mask, is_causal=is_causal
    )

    if info.name == REF_VERSION:
        try:
            ref_info = get_sdpa_version("naive")
        except KeyError:
            ref_info = get_sdpa_version("pytorch_sdpa")
        ok_ref, _ = ref_info.supports(attn_mask=True)
        if not ok_ref:
            pytest.skip("reference version does not support attn_mask")
        if (
            ref_info.name in _PYTORCH_BUILTIN_SDPA_NAMES
            and is_causal and L != S
        ):
            pytest.skip("reference uses upper-left causal; semantic mismatch")
        ref = call_sdpa_version(
            ref_info, q, k, v, attn_mask=mask, is_causal=is_causal
        )
    else:
        ref_info = get_sdpa_version(REF_VERSION)
        ref = call_sdpa_version(
            ref_info, q, k, v, attn_mask=mask, is_causal=is_causal
        )

    ctx = (
        f"version={info.name} source={info.source} shape={_shape_id(shape)} "
        f"dtype={_dtype_id(dtype)} is_causal={is_causal} mask=non-null"
    )
    assert_tensor_close(out, ref, dtype=dtype, context=ctx)
