# -*- coding: utf-8 -*-
"""SDPA 多版本注册表。

对外提供：
  * :class:`VersionInfo`             -- 描述一个 SDPA 变种的元数据。
  * :func:`register_sdpa_version`    -- 装饰器，把一个 callable 注册为版本。
  * :func:`available_sdpa_versions`  -- 列出当前已注册的全部版本。
  * :func:`get_sdpa_version`         -- 按名查找单个版本。
  * :func:`clear_registry`           -- 仅供测试使用，清空注册表。

设计要点：
  * 注册表是 **进程级单例**，按导入顺序无关：版本可以在 ``fused_cpp`` 包导入
    时（自动）或测试 session 启动后（手动）注册，效果一致。
  * 重名时默认抛 :class:`ValueError`；显式传入 ``override=True`` 才覆盖。
  * 元数据中 ``supports_*`` 系列「能力位」用于 ``pytest_generate_tests`` 钩子
    在收集阶段自动 skip 不兼容的测试组合。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, List, Optional, Tuple

import torch

__all__ = [
    "VersionInfo",
    "register_sdpa_version",
    "available_sdpa_versions",
    "get_sdpa_version",
    "clear_registry",
]

# ── VersionInfo ───────────────────────────────────────────────────────────

#: SDPA 内核的 callable 协议。
#:
#: ``fn(query, key, value, *, attn_mask, is_causal, scale) -> Tensor``
SdpaCallable = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class VersionInfo:
    """单个 SDPA 版本的元数据。

    :param name: 唯一字符串 ID，例如 ``"naive"``、``"flash1"``、``"flash2"``。
    :param callable: 实际执行函数；签名见 :data:`SdpaCallable`。
    :param source: 版本来源；取值 ``"cpp"`` / ``"python"`` / ``"hybrid"``。
    :param supports_dtypes: 该版本支持的 dtype 元组。
    :param supports_causal: 是否支持 ``is_causal=True``。
    :param supports_attn_mask: 是否支持外部 ``attn_mask``。
    :param supports_mla_shape: 是否支持 ``qk_head_dim != v_head_dim`` 的 MLA 形状。
    :param description: 一句话描述算法范式。
    :param tags: 标签集合，用于按类别过滤（如 ``"flash"`` / ``"experimental"``）。
    """

    name: str
    callable: SdpaCallable
    source: str = "python"
    supports_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16)
    supports_causal: bool = True
    supports_attn_mask: bool = True
    supports_mla_shape: bool = True
    description: str = ""
    tags: frozenset = field(default_factory=frozenset)

    def supports(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        is_causal: Optional[bool] = None,
        attn_mask: Optional[bool] = None,
        mla_shape: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """检查该版本是否支持给定的能力组合。

        :return: ``(ok, reason)``；``ok`` 为 False 时 ``reason`` 给出缺失能力。
        """
        if dtype is not None and dtype not in self.supports_dtypes:
            return False, (
                f"version={self.name!r} does not support dtype={dtype}"
            )
        if is_causal and not self.supports_causal:
            return False, (
                f"version={self.name!r} does not support is_causal=True"
            )
        if attn_mask and not self.supports_attn_mask:
            return False, (
                f"version={self.name!r} does not support attn_mask"
            )
        if mla_shape and not self.supports_mla_shape:
            return False, (
                f"version={self.name!r} does not support MLA shape "
                f"(qk_head_dim != v_head_dim)"
            )
        return True, ""


# ── 全局注册表 ─────────────────────────────────────────────────────────────

# 使用普通 dict；注册操作发生在 import 阶段或测试 fixture 中，无并发风险。
_REGISTRY: dict = {}

_VALID_SOURCES = frozenset({"cpp", "python", "hybrid"})


def _normalize_tags(tags: Optional[Iterable[str]]) -> frozenset:
    if tags is None:
        return frozenset()
    return frozenset(str(t) for t in tags)


def register_sdpa_version(
    name: str,
    *,
    source: str = "python",
    supports_dtypes: Tuple[torch.dtype, ...] = (
        torch.float32,
        torch.bfloat16,
    ),
    supports_causal: bool = True,
    supports_attn_mask: bool = True,
    supports_mla_shape: bool = True,
    description: str = "",
    tags: Optional[Iterable[str]] = None,
    override: bool = False,
):
    """把一个 callable 注册为 SDPA 版本的装饰器。

    用法：

    .. code-block:: python

        @register_sdpa_version(
            "my_flash",
            source="python",
            supports_dtypes=(torch.float32,),
            tags=("experimental",),
            description="My experimental flash variant",
        )
        def my_flash(query, key, value, *, attn_mask, is_causal, scale):
            ...

    :param name: 版本唯一 ID。
    :param source: ``"cpp"`` / ``"python"`` / ``"hybrid"``。
    :param supports_dtypes: 该版本支持的 dtype 元组。
    :param supports_causal: 是否支持 ``is_causal=True``。
    :param supports_attn_mask: 是否支持 ``attn_mask`` 参数。
    :param supports_mla_shape: 是否支持 MLA 形状。
    :param description: 一句话描述。
    :param tags: 标签集合，用于按类别过滤。
    :param override: 是否允许覆盖已存在的同名注册项；默认 False，重复注册抛
        :class:`ValueError`。
    :return: 直接返回原始 callable，便于链式调用。
    """
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"invalid source={source!r}; must be one of "
            f"{sorted(_VALID_SOURCES)}"
        )
    if not isinstance(name, str) or not name:
        raise ValueError(f"version name must be a non-empty str, got {name!r}")

    def _decorator(fn: SdpaCallable) -> SdpaCallable:
        if name in _REGISTRY and not override:
            raise ValueError(
                f"SDPA version {name!r} is already registered; "
                f"pass override=True to replace it"
            )
        info = VersionInfo(
            name=name,
            callable=fn,
            source=source,
            supports_dtypes=tuple(supports_dtypes),
            supports_causal=bool(supports_causal),
            supports_attn_mask=bool(supports_attn_mask),
            supports_mla_shape=bool(supports_mla_shape),
            description=description,
            tags=_normalize_tags(tags),
        )
        _REGISTRY[name] = info
        return fn

    return _decorator


def available_sdpa_versions() -> List[VersionInfo]:
    """返回所有已注册版本的 :class:`VersionInfo` 列表（按注册顺序）。"""
    return list(_REGISTRY.values())


def get_sdpa_version(name: str) -> VersionInfo:
    """按名查找单个版本。

    :raises KeyError: 未找到指定 ``name``；异常信息中会列出当前可用版本。
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"SDPA version {name!r} is not registered; "
            f"available versions: {available}"
        )
    return _REGISTRY[name]


def clear_registry() -> None:
    """清空注册表。**仅供测试使用**。"""
    _REGISTRY.clear()


def _replace_version(name: str, **changes) -> VersionInfo:
    """内部工具：原地替换某个版本的部分字段，返回新的 :class:`VersionInfo`。

    主要服务于 ``sdpa.py`` 中「先注册一个 stub，确认 C++ 可用后再回填真正
    callable」这类场景。**非公开 API**。
    """
    if name not in _REGISTRY:
        raise KeyError(f"cannot replace non-existent version {name!r}")
    new_info = replace(_REGISTRY[name], **changes)
    _REGISTRY[name] = new_info
    return new_info
