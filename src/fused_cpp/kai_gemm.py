# -*- coding: utf-8 -*-
"""KleidiAI GEMM 后端 Python 封装层。

提供 :class:`KAIThreadPool`、:class:`KAIGEMMHandler` 两个资源持有类，
以及 ``kai_gemm_prepare`` / ``create_kai_gemm`` / ``kai_gemm`` 等公开接口。

设计要点：
    - ``KAIThreadPool`` 是独立的一等公民资源，按绑核 CPU 列表创建，
      生命周期由调用者显式管理；支持 ``with`` 语句自动释放。
    - ``KAIGEMMHandler`` 仅持有 packed_weight / K / N 等纯数据，不再
      内嵌线程池。执行时通过 ``kai_gemm(..., pool=...)`` 显式传入所需
      的线程池；不传则走单线程路径。
    - ``kai_gemm_prepare`` 为无状态的纯函数，返回 ``(packed_weight, K, N)`` 元组。
    - 后端不可用时（非 AArch64 或 C++ 扩展未编译），所有入口抛 ``RuntimeError``。
"""

import logging
from typing import List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# 延迟检测 KleidiAI 后端是否可用
try:
    from fused_cpp._C import (  # type: ignore[import-untyped]
        kai_gemm_prepare as _kai_gemm_prepare,
        create_kai_gemm_handler as _create_kai_gemm_handler,
        kai_gemm as _kai_gemm_impl,
        release_kai_gemm_handler as _release_kai_gemm_handler,
        create_kai_thread_pool as _create_kai_thread_pool,
        destroy_kai_thread_pool as _destroy_kai_thread_pool,
    )

    _supports_kai = True
except ImportError:
    _supports_kai = False


def _require_backend() -> None:
    """后端不可用时抛出统一的 RuntimeError。"""
    if not _supports_kai:
        raise RuntimeError("KleidiAI GEMM 后端不可用（C++ 扩展未编译或非 AArch64 平台）")


class KAIThreadPool:
    """KleidiAI GEMM 线程池的 Python 封装，管理 C++ 侧 pool 的生命周期。

    线程池按绑核 CPU 列表创建：``cpu_ids[0]`` 为调用 ``parallel_for`` 时
    主线程的绑核位置，``cpu_ids[1..]`` 由 worker 线程启动时各自绑定。
    总并发度 = ``len(cpu_ids)``；``cpu_ids`` 为空则退化为单线程、不绑核。

    推荐通过 ``with`` 语句使用以保证异常路径下也能正确释放 worker 线程：

    .. code-block:: python

        with KAIThreadPool([0, 1, 2, 3]) as pool:
            kai_gemm(handler, x, pool=pool)
    """

    def __init__(self, cpu_ids: Optional[List[int]] = None) -> None:
        _require_backend()
        cpu_ids = [] if cpu_ids is None else [int(c) for c in cpu_ids]
        self._cpu_ids = cpu_ids
        self._handle: int = _create_kai_thread_pool(cpu_ids)

    @property
    def handle(self) -> int:
        """C++ 侧 pool 的 int64 句柄；销毁后返回 0。"""
        return self._handle

    @property
    def cpu_ids(self) -> List[int]:
        """构造时传入的 CPU ID 列表（拷贝，避免外部修改）。"""
        return list(self._cpu_ids)

    @property
    def num_threads(self) -> int:
        """总并发度 = 主线程 + worker 数。"""
        return max(len(self._cpu_ids), 1)

    def close(self) -> None:
        """显式释放 pool 与 worker 线程；重复调用是安全的 no-op。"""
        if self._handle != 0:
            try:
                _destroy_kai_thread_pool(self._handle)
            except Exception:  # pylint: disable=broad-except
                pass
            self._handle = 0

    def __enter__(self) -> "KAIThreadPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # 进程退出时 C++ 侧的 atexit 也会做兜底，这里仍尝试释放。
        self.close()


class KAIGEMMHandler:
    """KleidiAI GEMM handler 的 Python 封装，管理 C++ 侧纯数据 handler 的生命周期。

    handler 只持有 packed_weight 的引用和 K/N 元信息，**不包含**任何线程
    资源。运行时需要通过 :func:`kai_gemm` 的 ``pool`` 参数显式传入所需
    的 :class:`KAIThreadPool`；不传则走单线程路径。
    """

    def __init__(
        self,
        handler_ptr: int,
        k: int,
        n: int,
        packed_weight: torch.Tensor,
    ) -> None:
        self._handler_ptr = handler_ptr
        self.k = k
        self.n = n
        # 保活：确保在 handler 使用期间 packed_weight 不被释放。
        self.packed_weight = packed_weight

    def __del__(self) -> None:
        if hasattr(self, "_handler_ptr") and self._handler_ptr != 0:
            try:
                _release_kai_gemm_handler(self._handler_ptr)
            except Exception:  # pylint: disable=broad-except
                pass
            self._handler_ptr = 0


def kai_gemm_prepare(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, int]:
    """对权重 B 做离线预打包。

    :param weight: 权重张量，形状 ``[K, N]``，dtype 为 float32 或 bfloat16，
        必须为行主序（contiguous）。
    :param bias: 可选偏置张量，形状 ``[N]``，dtype 必须为 float32。
    :returns: ``(packed_weight, K, N)`` 元组：
        - packed_weight: 一维 uint8 张量，包含 KleidiAI packed RHS 数据；
        - K: 原始权重第 0 维；
        - N: 原始权重第 1 维。
    :raises RuntimeError: 后端不可用或输入不合法时抛出。
    """
    _require_backend()

    if weight.dim() != 2:
        raise RuntimeError(f"KAI GEMM prepare: 权重张量必须为 2D [K, N]，当前 dim={weight.dim()}")
    if not weight.is_contiguous():
        raise RuntimeError("KAI GEMM prepare: 权重张量必须 contiguous（行主序）")
    if weight.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"KAI GEMM prepare: 权重 dtype 仅支持 float32/bfloat16，当前 {weight.dtype}")

    k, n = int(weight.shape[0]), int(weight.shape[1])

    if bias is not None:
        if bias.dim() != 1:
            raise RuntimeError(f"KAI GEMM prepare: bias 必须为 1D，当前 dim={bias.dim()}")
        if int(bias.shape[0]) != n:
            raise RuntimeError(f"KAI GEMM prepare: bias 长度必须等于 N={n}，实际 {int(bias.shape[0])}")
        if bias.dtype != torch.float32:
            raise RuntimeError(f"KAI GEMM prepare: bias dtype 必须为 float32，当前 {bias.dtype}")
        bias = bias.contiguous()

    packed_weight = _kai_gemm_prepare(weight, bias)
    return packed_weight, k, n


def create_kai_gemm(
    packed_weight: torch.Tensor,
    k: int,
    n: int,
    max_threads: int = 1,
) -> KAIGEMMHandler:
    """基于预打包的权重创建 KleidiAI GEMM handler。

    :param packed_weight: 由 :func:`kai_gemm_prepare` 返回的一维 uint8 张量。
    :param k: 原始权重第 0 维。
    :param n: 原始权重第 1 维。
    :param max_threads: 兼容旧接口，仅用于校验 ``>=1``。新架构下 handler
        不再持有线程资源；执行时请显式通过 :func:`kai_gemm` 的 ``pool``
        参数传入所需的 :class:`KAIThreadPool`。
    :returns: :class:`KAIGEMMHandler` 实例。
    :raises RuntimeError: 后端不可用或参数不合法时抛出。
    """
    _require_backend()

    if max_threads < 1:
        raise RuntimeError(f"KAI GEMM: max_threads 必须 >=1，当前 {max_threads}")
    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()

    handler_ptr = _create_kai_gemm_handler(
        packed_weight,
        int(k),
        int(n),
    )
    return KAIGEMMHandler(
        handler_ptr=handler_ptr,
        k=int(k),
        n=int(n),
        packed_weight=packed_weight,
    )


def kai_gemm(
    handler: KAIGEMMHandler,
    x: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
    pool: Optional[Union[KAIThreadPool, int]] = None,
) -> torch.Tensor:
    """使用 KleidiAI GEMM handler 执行矩阵乘法。

    计算 ``output = x @ W + bias``，其中 ``W`` 与可选 ``bias`` 已经通过
    :func:`kai_gemm_prepare` 预打包到 handler 中。

    :param handler: 由 :func:`create_kai_gemm` 创建的 handler。
    :param x: 输入张量，形状 ``[*, K]``，dtype 必须为 float32。
    :param output_dtype: 输出 dtype，支持 ``torch.float32`` 或
        ``torch.bfloat16``；默认与 ``x.dtype`` 一致。
    :param pool: 用于并行计算的线程池；可传 :class:`KAIThreadPool` 实例
        或其底层 int64 句柄。``None`` 或 ``0`` 表示单线程路径（在调用
        线程上顺序执行）。
    :returns: 输出张量，形状 ``[*, N]``，dtype 为 ``output_dtype``。
    :raises RuntimeError: 后端不可用或输入不合法时抛出。
    """
    _require_backend()

    if x.shape[-1] != handler.k:
        raise RuntimeError(f"KAI GEMM: 输入最后一维 ({x.shape[-1]}) 与 handler.k ({handler.k}) 不一致")
    if x.dtype != torch.float32:
        raise RuntimeError(f"KAI GEMM: 输入 dtype 必须为 float32，当前 {x.dtype}")

    # 默认输出 dtype 与输入一致（float32）。
    out_dtype = output_dtype if output_dtype is not None else x.dtype
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"KAI GEMM: output_dtype 仅支持 float32/bfloat16，当前 {out_dtype}")

    batch_shape = x.shape[:-1]
    output_shape = (*batch_shape, handler.n)

    # M=0 边界：直接返回形状正确的空张量，不调用 C++ 层。
    if x.numel() == 0:
        return torch.empty(output_shape, dtype=out_dtype)

    # 解析 pool 参数为 int64 句柄。
    if pool is None:
        pool_handle = 0
    elif isinstance(pool, KAIThreadPool):
        pool_handle = int(pool.handle)
    else:
        pool_handle = int(pool)

    # 确保 contiguous，reshape 为 2D 后调用 C++。
    x_contig = x.contiguous()
    x_2d = x_contig.reshape(-1, handler.k)

    output = torch.empty(output_shape, dtype=out_dtype)
    _kai_gemm_impl(
        output.reshape(-1, handler.n),
        x_2d,
        handler._handler_ptr,
        pool_handle,
    )

    return output
