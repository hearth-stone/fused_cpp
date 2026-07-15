# -*- coding: utf-8 -*-
"""ACL GEMM 后端 Python 封装层。

提供 ACLGEMMHandler 类管理 handler 生命周期，
以及 create_acl_gemm / acl_gemm / set_acl_affinity / get_acl_affinity 等公开接口。
"""

import logging

import torch

logger = logging.getLogger(__name__)

# 延迟检测 ACL 后端是否可用
try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        create_acl_gemm_handler as _create_acl_gemm_handler,
        acl_gemm as _acl_gemm,
        release_acl_gemm_handler as _release_acl_gemm_handler,
        set_acl_thread_affinity as _set_acl_thread_affinity,
        get_acl_thread_affinity as _get_acl_thread_affinity,
    )

    _supports_acl = True
except ImportError:
    _supports_acl = False


class ACLGEMMHandler:
    """ACL GEMM handler 的 Python 封装，管理 C++ 侧资源的生命周期。

    Attributes:
        k: 输入维度（权重的第 0 维）。
        n: 输出维度（权重的第 1 维）。
        fast_math: 是否启用低精度加速路径。
    """

    def __init__(self, handler_ptr: int, k: int, n: int, fast_math: bool = False) -> None:
        self._handler_ptr = handler_ptr
        self.k = k
        self.n = n
        self.fast_math = fast_math

    def __del__(self) -> None:
        if hasattr(self, "_handler_ptr") and self._handler_ptr != 0:
            try:
                _release_acl_gemm_handler(self._handler_ptr)
            except Exception:
                pass
            self._handler_ptr = 0


def create_acl_gemm(
    weight: torch.Tensor,
    num_threads: int = 0,
    fast_math: bool = False,
) -> ACLGEMMHandler:
    """创建 ACL GEMM handler 并预打包权重。

    :param weight: 权重张量，形状 [K, N]，dtype 为 float32 或 bfloat16。
    :param num_threads: ACL Scheduler 线程数，0 表示使用默认值。
    :param fast_math: 是否启用低精度加速路径（FP32 会用 BF16 中间精度）。
    :returns: ACLGEMMHandler 实例。
    :raises RuntimeError: 权重维度或数据类型不支持时抛出。
    """
    if not _supports_acl:
        raise RuntimeError("ACL GEMM 后端不可用（C++ 扩展未编译或非 AArch64 平台）")

    if weight.dim() != 2:
        raise RuntimeError(f"ACL GEMM: 权重张量必须为 2D，当前维度: {weight.dim()}")

    k, n = weight.shape
    handler_ptr = _create_acl_gemm_handler(weight, num_threads, fast_math)
    return ACLGEMMHandler(handler_ptr, k, n, fast_math)


def acl_gemm(
    handler: ACLGEMMHandler,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """使用 ACL GEMM handler 执行矩阵乘法。

    计算 output = x @ W^T + bias，其中 W 为预打包的权重。

    :param handler: 由 create_acl_gemm 创建的 handler。
    :param x: 输入张量，形状 [*, K]。
    :param bias: 可选偏置张量，形状 [N]。
    :returns: 输出张量，形状 [*, N]。
    """
    # M=0 边界情况：直接返回空张量
    if x.numel() == 0:
        return torch.empty((*x.shape[:-1], handler.n), dtype=x.dtype)

    # 确保输入连续
    x = x.contiguous()

    # 分配输出张量
    output = torch.empty((*x.shape[:-1], handler.n), dtype=x.dtype)

    # reshape 为 2D 执行 GEMM
    x_2d = x.reshape(-1, handler.k)

    _acl_gemm(output.reshape(-1, handler.n), x_2d, bias, handler._handler_ptr)

    return output


def set_acl_affinity(
    core_start: int,
    core_end: int,
    num_threads: int,
) -> None:
    """设置 ACL 核心绑定策略。

    :param core_start: 绑定的起始核心编号。
    :param core_end: 绑定的结束核心编号（不含）。
    :param num_threads: ACL Scheduler 使用的线程数。
    """
    if not _supports_acl:
        logger.warning("ACL 后端不可用，无法设置核心绑定策略")
        return
    _set_acl_thread_affinity(core_start, core_end, num_threads)


def get_acl_affinity() -> tuple[int, int, int]:
    """查询当前 ACL 核心绑定状态。

    :returns: (core_start, core_end, num_threads) 元组，-1 表示未设置。
    """
    if not _supports_acl:
        return (-1, -1, -1)
    return _get_acl_thread_affinity()
