# -*- coding: utf-8 -*-
"""Scaled Dot-Product Attention (SDPA) 封装模块。

优先使用 C++ 扩展实现，若不可用则回退到 PyTorch 原生实现。

本模块同时是 SDPA 多版本框架的 Python 入口：
  * 默认接口 :func:`scaled_dot_product_attention` 行为保持不变
    （当前由 :mod:`fused_cpp._C` 提供 C++ FlashAttention-2 风格内核）。
  * 多版本接口 :func:`sdpa_versioned` 通过 :mod:`fused_cpp.sdpa_registry`
    暴露：

      - 模块导入时自动把 C++ 层 ``_C.list_sdpa_versions()`` 返回的全部内核
        注册到 Python 注册表（``source="cpp"``）；
      - 同时注册两个内置 PyTorch fallback：``"naive_torch"`` /
        ``"pytorch_sdpa"``（``source="python"``）；
      - C++ 扩展不可用时，对 ``source="cpp"`` 的版本调用 SHALL 降级到
        **同名** Python fallback；若不存在同名 fallback，则给出明确错误。

新增 SDPA 变种只需通过 :func:`register_sdpa_version` 装饰器注册即可，
框架本身**不感知**具体内核实现，详见 :mod:`fused_cpp.sdpa_registry`。
"""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F

from fused_cpp.sdpa_registry import (
    VersionInfo,
    available_sdpa_versions,
    get_sdpa_version,
    register_sdpa_version,
)

logger = logging.getLogger(__name__)

try:
    from fused_cpp._C import (
        scaled_dot_product_attention as _cpp_sdpa,
        scaled_dot_product_attention_versioned as _cpp_sdpa_versioned,
        list_sdpa_versions as _cpp_list_sdpa_versions,
    )

    _HAS_CPP_SDPA = True
except ImportError:
    _HAS_CPP_SDPA = False
    _cpp_sdpa = None
    _cpp_sdpa_versioned = None

    def _cpp_list_sdpa_versions():  # type: ignore[no-redef]
        return []

    logger.info("fused_cpp C++ SDPA 扩展不可用，将回退到 PyTorch 原生实现")

# ``torch.nn.attention.sdpa_kernel`` 与 ``SDPBackend`` 在 PyTorch 2.3+
# 提供；旧版本不存在时 ``pytorch_sdpa_math`` 会降级为“不限定后端”
# 调用 :func:`F.scaled_dot_product_attention`，同时上报 warning。
try:
    from torch.nn.attention import SDPBackend as _SDPBackend  # type: ignore
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel  # type: ignore

    _HAS_SDP_BACKEND_API = True
except ImportError:  # pragma: no cover - 只在 torch < 2.3 出现
    _SDPBackend = None  # type: ignore[assignment]
    _sdpa_kernel = None  # type: ignore[assignment]
    _HAS_SDP_BACKEND_API = False
    logger.info("torch.nn.attention.sdpa_kernel API 不可用，pytorch_sdpa_math 将回退到默认后端选择")

__all__ = [
    "VersionInfo",
    "available_sdpa_versions",
    "get_sdpa_version",
    "register_sdpa_version",
    "sdpa_versioned",
    "scaled_dot_product_attention",
]


# ── 默认入口（向后兼容，行为零回归）────────────────────────────────────


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Scaled Dot-Product Attention，接口与 ``F.scaled_dot_product_attention`` 完全一致。

    优先调用 C++ 扩展实现，若不可用则回退到 PyTorch 原生实现。

    :param query: Query 张量，shape ``[B, N, L, E]``
    :param key: Key 张量，shape ``[B, N, S, E]``
    :param value: Value 张量，shape ``[B, N, S, Ev]``
    :param attn_mask: 可选的 additive attention mask
    :param dropout_p: Dropout 概率（推理阶段忽略）
    :param is_causal: 是否应用因果掩码
    :param scale: 缩放因子，``None`` 时自动计算 ``1/sqrt(E)``
    :param enable_gqa: 是否启用 GQA（当前不支持）
    :return: 注意力输出张量，shape ``[B, N, L, Ev]``
    """
    if _HAS_CPP_SDPA:
        return _cpp_sdpa(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


# ── 内置纯 PyTorch fallback ───────────────────────────────────────────────


def _naive_torch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """两遍 softmax 的朴素 SDPA 实现。

    与 C++ ``"naive"`` 内核语义一致：``causal_offset = S - L``（lower-right
    causal）。所有中间张量在 fp32 精度下计算。
    """
    orig_dtype = query.dtype
    L = query.size(-2)
    S = key.size(-2)
    E = query.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    q = query.float()
    k = key.float()
    v = value.float()

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, N, L, S]
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    if is_causal:
        offset = S - L
        l_idx = torch.arange(L, device=scores.device).unsqueeze(-1)
        s_idx = torch.arange(S, device=scores.device).unsqueeze(0)
        causal_mask = s_idx > (l_idx + offset)  # [L, S]
        scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    # 处理「整行被 mask」的情形：softmax(全 -inf)=NaN，按内核语义置 0。
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.matmul(weights, v)
    return out.to(orig_dtype) if orig_dtype != torch.float32 else out


def _pytorch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """直接转发到 :func:`torch.nn.functional.scaled_dot_product_attention`。

    作为 PyTorch 官方对照基线（PyTorch 内部根据 CPU/GPU 自动选择 FA / FlashAtt
    后端）。注意：对 ``L != S`` 且 ``is_causal=True`` 的情形，PyTorch 默认采
    用 **upper-left** causal 语义（与 ``naive_torch`` / C++ 内核的 ``S-L``
    offset **不一致**）。等价性测试中应避免该组合直接互比。
    """
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


# 注册两个 PyTorch fallback。
# 注意：装饰器在模块导入期立即生效；clear_registry() 后需重新 import
# 该模块（或测试 fixture 中显式调用 _ensure_builtins()）才能恢复。
register_sdpa_version(
    "naive_torch",
    source="python",
    supports_dtypes=(torch.float32, torch.bfloat16),
    supports_causal=True,
    supports_attn_mask=True,
    supports_mla_shape=True,
    description=(
        "Pure PyTorch naive SDPA: matmul + softmax + matmul, with the same "
        "lower-right causal semantics as the C++ kernels."
    ),
    tags=("naive", "python_fallback"),
)(_naive_torch_sdpa)

register_sdpa_version(
    "pytorch_sdpa",
    source="python",
    supports_dtypes=(torch.float32, torch.bfloat16),
    # PyTorch 内置 SDPA 在 L != S 且 is_causal=True 时使用 upper-left 语义，
    # 与本框架其它版本不一致；测试矩阵中通过 supports_causal=True 仍允许
    # 跑用例，但 conftest 会确保 L == S 时才与其它版本互比，否则降级为
    # 仅作为「自我一致性」基线。
    supports_causal=True,
    supports_attn_mask=True,
    supports_mla_shape=True,
    description=("Direct call to torch.nn.functional.scaled_dot_product_attention; the official PyTorch baseline."),
    tags=("baseline", "python_fallback"),
)(_pytorch_sdpa)


def _pytorch_sdpa_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """调用 :func:`F.scaled_dot_product_attention` 并强制底层走 ``MATH`` 后端。

    在 macOS Apple Silicon 上 PyTorch 带的 wheel 默认未启用 oneDNN/MKLDNN，
    “默认后端选择”会把 bf16 SDPA dispatch 到 ``FLASH_ATTENTION`` 路径，
    但该路径会退化为标量实现，性能低于 ``MATH`` 后端的 BLAS gemm 一个量级。

    本函数作为 **公平 baseline**，在所有平台上都强制走同一条
    ``MATH`` 路径（``torch.matmul`` + ``softmax``），使 bench 结果不受
    后端分派差异的干扰。

    语义与 ``pytorch_sdpa`` 完全一致；在 ``L != S`` 且 ``is_causal=True``
    场景下同样使用 PyTorch 默认的 **upper-left** causal 语义，等价性
    测试中不应与 C++ 内核 / ``naive_torch`` 直接互比。
    """
    if _HAS_SDP_BACKEND_API:
        with _sdpa_kernel(_SDPBackend.MATH):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
            )
    # 旧版 PyTorch 不提供 sdpa_kernel；退为默认后端调用 + 一条 warning，
    # 避免静默失去“MATH 后端”语义。
    logger.warning("pytorch_sdpa_math: torch.nn.attention.sdpa_kernel 不可用，本次调用将使用 PyTorch 默认后端选择")
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


register_sdpa_version(
    "pytorch_sdpa_math",
    source="python",
    supports_dtypes=(torch.float32, torch.bfloat16),
    # 与 ``pytorch_sdpa`` 同语义：L != S + is_causal 走 upper-left。
    # 同样需要 conftest 保护，仅在 L == S 时与其它版本互比。
    supports_causal=True,
    supports_attn_mask=True,
    supports_mla_shape=True,
    description=(
        "PyTorch SDPA forced onto the MATH backend via "
        "torch.nn.attention.sdpa_kernel(SDPBackend.MATH); a fair bf16 "
        "baseline on platforms where mkldnn is unavailable (e.g. macOS "
        "Apple Silicon)."
    ),
    tags=("baseline", "python_fallback", "math_backend"),
)(_pytorch_sdpa_math)


# ── C++ 内核能力位（与 csrc/sdpa*.cpp 实现对齐）───────────────
# C++ 三个内核（naive / flash1 / flash2）支持的能力位完全一致：
# fp32 + bf16 / causal / attn_mask / MLA shape。
# 若未来某个 C++ 内核能力受限，可在该字典中收紧。
_CPP_DEFAULT_CAPS = dict(
    supports_dtypes=(torch.float32, torch.bfloat16),
    supports_causal=True,
    supports_attn_mask=True,
    supports_mla_shape=True,
)

_CPP_VERSION_META: dict = {
    "naive": dict(
        description=(
            "C++ naive SDPA: explicitly materialize attn_scores[L, S] per "
            "(b, n) and run two-pass softmax; reference for FLOPs."
        ),
        tags=("naive", "reference"),
    ),
    "flash1": dict(
        description=(
            "C++ FlashAttention-1 style: outer K/V tile + inner Q tile + "
            "two-pass online softmax with persistent (m, l, O) per i-block."
        ),
        tags=("flash", "online_softmax", "fa1"),
    ),
    "flash2": dict(
        description=(
            "C++ FlashAttention-2 style: per-Q-row online softmax with "
            "in-line correction factor (current default kernel)."
        ),
        tags=("flash", "online_softmax", "fa2"),
    ),
    "flash2_neon": dict(
        description=(
            "NEON-vectorized FlashAttention-2 kernel: same online-softmax "
            "skeleton as flash2, with Q*K^T / softmax-exp / attn*V hot "
            "paths replaced by ARM NEON intrinsics (fp32 vfmaq_f32, bf16 "
            "vbfdotq_f32 when __ARM_FEATURE_BF16, otherwise widen+fma). "
            "Falls back to a scalar C++ path on non-AArch64 platforms."
        ),
        tags=("flash", "online_softmax", "fa2", "neon"),
    ),
    "flash2_neon_cache": dict(
        description=(
            "Cache-aware multi-thread NEON FlashAttention-2 kernel: "
            "explicit L1/L2/L3 tile-size derivation (configurable at "
            "compile time via FUSED_CPP_SDPA_L{1,2,3}_BYTES / _RATIO), "
            "8x8 BFMMLA micro-kernel with three-level fallback "
            "(BFMMLA -> BFMLALB/T -> widen+FMLA), 8x4 / scalar Sk-tail "
            "micro-kernels, double-buffered software prefetch and "
            "OpenMP (b, n, q_tile) collapse(3). Two-stage GEMM precision "
            "split: bf16 BFMMLA for Q*K^T, fp32 FMLA + V widen for P_hat*V. "
            "Built only on AArch64 platforms."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
        ),
    ),
    "flash2_neon_l3kv": dict(
        description=(
            "L3-resident-K/V FlashAttention-2 kernel built on the same MK "
            "trait framework as flash2_neon_cache, but adds three pieces "
            "the cache-aware kernel only sketched: (1) when K/V doesn't "
            "fit L3, threads are head-grouped via `omp parallel { single { "
            "for(b,n) taskloop nogroup } }` so workers on the same (b, n) "
            "share one L3-resident K/V copy; (2) outer Q tile uses Lc_l2 "
            "(<=64, capped by Ev to fit L2) with Lc_l2/8 inner 8-row "
            "groups reusing each L2 K/V tile; (3) PLDL1KEEP prefetch is "
            "issued two inner-groups ahead, in addition to existing "
            "PLDL2KEEP for the next L2 K/V tile. Numerically equivalent "
            "to flash2_neon_cache row-by-row."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
        ),
    ),
    "flash2_neon_l3kv_packqkv": dict(
        description=(
            "L3-resident FlashAttention-2 with K + V multi-thread pre-packed "
            "at SDPA entry, Q packed once per q-tile inside "
            "process_q_tile_lc_packqkv (q-tile 内复用，不重复 pack). BFMMLA "
            "inner uses 4 independent vld1q_u16 + B-major schedule "
            "(inherits qk_packqk_seq4_bmajor microkernel). fp32 input uses "
            "full K pre-pack plus a packed-K 8x8 lane-FMLA QK^T microkernel; "
            "Q stays row-major and is not pre-packed."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
            "packed_k",
            "packed_q",
            "packed_v",
        ),
    ),
    "flash2_neon_l3kv_packqkv_pbf16pv": dict(
        description=(
            "Experimental packqkv variant that keeps the same packed Q/K/V "
            "layout as flash2_neon_l3kv_packqkv, but writes softmax P_hat "
            "directly to bf16 scratch and feeds pv_8x8_pbf16. The row sum "
            "and output normalization remain fp32. fp32 input uses full K "
            "pre-pack plus a packed-K 8x8 lane-FMLA QK^T microkernel and "
            "PV pquad."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
            "packed_k",
            "packed_q",
            "packed_v",
            "pbf16",
        ),
    ),
    "flash2_neon_l3kv_packqkv_pbf16pv_exp_poly4": dict(
        description=(
            "Same packed Q/K/V + bf16 P_hat PV path as "
            "flash2_neon_l3kv_packqkv_pbf16pv, but uses a degree-4 NEON "
            "polynomial approximation for softmax exp in the bf16 P_hat "
            "materialization path."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
            "packed_k",
            "packed_q",
            "packed_v",
            "pbf16",
            "exp_poly4",
            "experimental",
        ),
    ),
    "flash2_neon_l3kv_packqkv_pbf16pv_exp_poly6": dict(
        description=(
            "Same packed Q/K/V + bf16 P_hat PV path as "
            "flash2_neon_l3kv_packqkv_pbf16pv, but uses a degree-6 NEON "
            "polynomial approximation for softmax exp in the bf16 P_hat "
            "materialization path."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
            "packed_k",
            "packed_q",
            "packed_v",
            "pbf16",
            "exp_poly6",
            "experimental",
        ),
    ),
    "flash2_neon_l3kv_packv_l1_bfmlal_layout": dict(
        description=(
            "Experimental L3KV packv SDPA using MK_L1BfmlalLayout: V is "
            "pre-packed to [B,N,Ev/8,S,8], bf16 QK^T uses the K_col "
            "BFMLAL microkernel, and bf16 PV converts P_hat scratch to bf16 "
            "before calling pv_8x8_pbf16. This is the first end-to-end "
            "wiring of the L1-only 85%+ peak microkernels; K_col/P_bf16 "
            "production cost is still paid in the SDPA outer path."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
            "packed_v",
            "k_col",
            "pbf16",
            "l1_bfmlal_layout",
            "experimental",
        ),
    ),
    "flash2_neon_l3kv_l1_bfmlal_layout": dict(
        description=(
            "Short alias for flash2_neon_l3kv_packv_l1_bfmlal_layout, the "
            "experimental SDPA path that wires MK_L1BfmlalLayout into the "
            "packv L3KV topology."
        ),
        tags=(
            "flash",
            "online_softmax",
            "fa2",
            "neon",
            "cache_aware",
            "multi_thread",
            "l3_kv_resident",
            "packed_v",
            "k_col",
            "pbf16",
            "l1_bfmlal_layout",
            "experimental",
            "alias",
        ),
    ),
}


def _make_cpp_callable(version_name: str):
    """构造一个调用底层 C++ 内核的 callable。

    自动捕获 ``version_name``，在 C++ 扩展不可用时降级到同名 Python fallback。
    """

    def _cpp_call(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if _HAS_CPP_SDPA:
            return _cpp_sdpa_versioned(
                query,
                key,
                value,
                attn_mask,
                0.0,
                is_causal,
                scale,
                False,
                version_name,
            )
        # 降级到同名 Python fallback（若存在）
        try:
            info = get_sdpa_version(version_name)
        except KeyError:
            info = None
        if info is not None and info.source == "python":
            return info.callable(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
            )
        # 降级到 naive_torch 作为最后兜底
        try:
            info = get_sdpa_version("naive_torch")
            return info.callable(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
            )
        except KeyError:
            raise RuntimeError(
                f"sdpa version {version_name!r} is registered as C++ kernel "
                f"but the C++ extension is unavailable, and no Python "
                f"fallback was found."
            )

    _cpp_call.__name__ = f"_cpp_sdpa_{version_name}"
    _cpp_call.__qualname__ = _cpp_call.__name__
    return _cpp_call


def _register_cpp_versions() -> None:
    """把 C++ 层报告的内核名注册到 Python 注册表。

    在模块导入时调用。注册项标记 ``source="cpp"``，能力位按
    :data:`_CPP_DEFAULT_CAPS` + :data:`_CPP_VERSION_META` 合并填充。
    若 C++ 扩展可用，只注册 ``_C.list_sdpa_versions()`` 实际报告的名字，
    从而让架构相关的 source filtering 同步反映到 Python registry。扩展完全
    不可用时，仍按 :data:`_CPP_VERSION_META` 的预声明名单占位注册，callable
    在实际调用时降级到 Python fallback。
    """
    if _HAS_CPP_SDPA:
        try:
            cpp_names = list(_cpp_list_sdpa_versions())
        except Exception as exc:  # pragma: no cover - 防御
            logger.warning("failed to list C++ SDPA versions: %s", exc)
            cpp_names = []
    else:
        cpp_names = []

    names = cpp_names if _HAS_CPP_SDPA else list(_CPP_VERSION_META.keys())

    for name in names:
        meta = _CPP_VERSION_META.get(name, {})
        register_sdpa_version(
            name,
            source="cpp",
            supports_dtypes=_CPP_DEFAULT_CAPS["supports_dtypes"],
            supports_causal=_CPP_DEFAULT_CAPS["supports_causal"],
            supports_attn_mask=_CPP_DEFAULT_CAPS["supports_attn_mask"],
            supports_mla_shape=_CPP_DEFAULT_CAPS["supports_mla_shape"],
            description=meta.get("description", ""),
            tags=meta.get("tags", ()),
            override=True,
        )(_make_cpp_callable(name))


_register_cpp_versions()


# ── 多版本调度入口 ────────────────────────────────────────────────────────


def sdpa_versioned(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    version: str,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """按 ``version`` 选择 SDPA 内核执行。

    :param query: shape ``[B, N, L, E]``
    :param key: shape ``[B, N, S, E]``
    :param value: shape ``[B, N, S, Ev]``
    :param version: 已注册的版本名（参见 :func:`available_sdpa_versions`）。
    :param attn_mask: 可选 additive mask。
    :param dropout_p: 推理时忽略；仅用于接口一致性。
    :param is_causal: 是否应用因果掩码（``causal_offset = S - L`` 语义）。
    :param scale: 缩放因子，``None`` 时自动取 ``1/sqrt(E)``。
    :return: 输出张量，shape ``[B, N, L, Ev]``，dtype 与 ``query`` 一致。
    :raises ValueError: ``version`` 未注册；错误信息会列出当前可用版本。
    """
    if dropout_p != 0.0:
        logger.warning("sdpa_versioned: dropout_p=%s ignored (inference only)", dropout_p)
    try:
        info = get_sdpa_version(version)
    except KeyError:
        avail = sorted(v.name for v in available_sdpa_versions())
        raise ValueError(f"sdpa_versioned: version={version!r} is not registered; available versions: {avail}")

    with torch.no_grad():
        return info.callable(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
        )
