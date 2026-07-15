# -*- coding: utf-8 -*-
"""SDPA 多版本基准测试（默认 skip，需 ``-m bench`` 触发）。

实现要点：

1. ``version`` 维度通过 :mod:`tests.conftest` 的 ``pytest_generate_tests``
   钩子从注册表自动注入，新注册的版本会**立即**进入基准矩阵，无需修改
   本文件。
2. 每条 ``(version, shape, dtype, is_causal)`` 用例独立测量：
   * ≥ 5 次 warmup（不计入统计）；
   * ≥ 20 轮稳态测量；
   * 用 ``time.perf_counter()`` 计时，输出 mean / std / median / P90 / P99。
3. 每条结果通过 :func:`tests.conftest.record_sdpa_bench` 加入 session 级
   聚合表；session 结束时由 :func:`tests.conftest.pytest_sessionfinish`
   统一落盘到 ``<output_dir>/sdpa_versions_<timestamp>.{csv,json}``。
4. **Thread Sweep 模式**：当传入 ``--sdpa-thread-sweep N1,N2,...`` 时，
   父进程跑完普通 bench 后，由 conftest 子进程模式以
   ``OMP_NUM_THREADS=N`` 重跑同一基准并写出
   ``<output_dir>/sdpa_versions_<timestamp>_sweep.{csv,json}`` +
   scaling 表 + ``efficiency = gflops(N) / (N * gflops(1))``。
   ``_C.has_openmp() == False`` 时自动 skip Thread Sweep。
"""

from __future__ import annotations

import gc
import os
import statistics
import time
from typing import Any, Dict

import pytest
import torch

from fused_cpp.sdpa_flops import compute_sdpa_flops, gflops
from fused_cpp.sdpa_registry import VersionInfo

from tests.conftest import (
    _apply_thread_pin,
    _resolve_pinned_thread_count,
    call_sdpa_version,
    get_sdpa_bench_records,
    record_sdpa_bench,
)


# ── benchmark 配置 ────────────────────────────────────────────────────────

WARMUP_ITERS = 5
BENCH_ITERS = 20

# 较小的形状集合 + MLA 形状；避免 macOS 无 OpenMP 时基准过慢。
BENCH_SHAPES = [
    # (B, N, L, S, E, Ev)
    (1, 8, 64, 64, 64, 64),
    (1, 8, 256, 256, 64, 64),
    # BGE-small-zh / vLLM-style request batching:
    # hidden=512, heads=8, head_dim=64, seq_len=512.
    (1, 8, 512, 512, 64, 64),
    (2, 8, 512, 512, 64, 64),
    (4, 8, 512, 512, 64, 64),
    (8, 8, 512, 512, 64, 64),
    (1, 16, 512, 512, 192, 128),  # MLA
    # DeepSeek-R1 / V3 在 TP=4 下的 prefill MLA 真实形状（non-absorbed 路径）：
    #   num_attention_heads = 128 → 每 rank N = 128 / 4 = 32
    #   qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192    → E
    #   v_head_dim = 128                                          → Ev
    #   L = S = 2048 取自常见的 chunked-prefill chunk size
    (1, 32, 2048, 2048, 192, 128),  # DeepSeek-R1 TP=4 prefill MLA
]

BENCH_DTYPES = [torch.float32, torch.bfloat16]
BENCH_CAUSAL = [False, True]


def _shape_id(shape) -> str:
    B, N, L, S, E, Ev = shape
    return f"B{B}-N{N}-L{L}-S{S}-E{E}-Ev{Ev}"


def _dtype_id(dt: torch.dtype) -> str:
    return {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
    }.get(dt, str(dt))


def _make_qkv(shape, dtype, seed: int = 1234):
    B, N, L, S, E, Ev = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, N, L, E, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(B, N, S, E, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, N, S, Ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


def _omp_snapshot() -> Dict[str, str]:
    """记录当前线程/OMP 状态快照，并附带 PyTorch intra-op 线程数。

    加入 ``torch_num_threads`` 与 ``torch_num_interop_threads`` 两项是为了
    让用户从 bench 输出中直观看到 PyTorch 路径（``pytorch_sdpa``）
    实际使用的线程池大小——这项在 Linux 上与 ``OMP_NUM_THREADS`` /
    ``omp_get_max_threads()`` 可能完全不一致。
    """
    snap: Dict[str, str] = {}
    try:
        from fused_cpp import _C  # type: ignore

        snap.update(dict(_C.get_omp_runtime_info()))
    except Exception:
        snap.update(
            {
                "max_threads": "1",
                "num_procs": "1",
                "has_openmp": "false",
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "null"),
                "OMP_SCHEDULE": os.environ.get("OMP_SCHEDULE", "null"),
                "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "null"),
                "OMP_PLACES": os.environ.get("OMP_PLACES", "null"),
            }
        )
    snap["torch_num_threads"] = str(torch.get_num_threads())
    try:
        snap["torch_num_interop_threads"] = str(torch.get_num_interop_threads())
    except RuntimeError:
        snap["torch_num_interop_threads"] = "unknown"
    return snap


def _bench_once(fn, args, kwargs) -> float:
    """跑一次 fn(*args, **kwargs)，返回耗时（秒）。"""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def _run_bench(fn, args, kwargs) -> Dict[str, float]:
    """warmup + 多轮测量；返回延迟统计（秒为单位）。"""
    for _ in range(WARMUP_ITERS):
        fn(*args, **kwargs)
    gc.collect()
    samples = [_bench_once(fn, args, kwargs) for _ in range(BENCH_ITERS)]
    samples.sort()
    n = len(samples)
    p90 = samples[max(int(round(n * 0.9)) - 1, 0)] if n > 0 else 0.0
    p99 = samples[max(int(round(n * 0.99)) - 1, 0)] if n > 0 else 0.0
    return {
        "mean": statistics.mean(samples),
        "stddev": statistics.pstdev(samples) if n > 1 else 0.0,
        "median": statistics.median(samples),
        "p90": p90,
        "p99": p99,
        "min": samples[0],
        "max": samples[-1],
    }


# ── pytest 用例 ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _sdpa_bench_pin_threads(request):
    """在 bench session 启动时一次性钉住 PyTorch / OpenMP 线程数。

    优先级：``--sdpa-num-threads`` > ``OMP_NUM_THREADS`` env。两者都未设时
    保持 PyTorch 默认（Linux 上会是 #cpus），只记录不钉住。

    这个 fixture 作为 session-scope autouse 只会在首次使用时设一次，
    避免用例间反复 set_num_threads 出现不必要开销。
    """
    n = _resolve_pinned_thread_count(request.config)
    if n is not None:
        actual = _apply_thread_pin(n, set_env=True, verbose=True)
        # 警告“钉住后仍不一致”——主要是 set_num_interop_threads 被锁住
        # 的情况（已运行过 op），不影响 intra-op 结果。
        if actual != n:
            print(
                f"[sdpa] WARNING: requested {n} threads but "
                f"torch.get_num_threads()={actual}; PyTorch may have already "
                "locked the pool. Restart the test session if exact pinning "
                "is required."
            )
    yield


@pytest.mark.bench
@pytest.mark.parametrize("shape", BENCH_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("dtype", BENCH_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("is_causal", BENCH_CAUSAL, ids=["noncausal", "causal"])
def test_sdpa_bench(sdpa_version, shape, dtype, is_causal):
    """单次基准：测量 (version, shape, dtype, is_causal) 配置。"""
    info: VersionInfo = sdpa_version
    B, N, L, S, E, Ev = shape

    # ── 能力位过滤 ──
    is_mla = E != Ev
    ok, reason = info.supports(dtype=dtype, is_causal=is_causal, mla_shape=is_mla)
    if not ok:
        pytest.skip(reason)

    q, k, v = _make_qkv(shape, dtype)

    def _call():
        return call_sdpa_version(info, q, k, v, is_causal=is_causal)

    stats = _run_bench(_call, (), {})

    # FLOPs 与 GFLOP/s
    flops = compute_sdpa_flops(B, N, L, S, E, Ev, is_causal=is_causal)
    gflops_total = gflops(flops["total_flops"], stats["mean"])
    gflops_causal = gflops(flops["total_flops_causal"], stats["mean"])
    effective_flops_key = "total_flops_causal" if is_causal else "total_flops"
    gflops_effective = gflops(flops[effective_flops_key], stats["mean"])

    # PyTorch 加速比基线（如同 (shape, dtype, is_causal) 的 pytorch_sdpa
    # 已被这一矩阵跑过，则取最近一次）
    speedup = float("nan")
    for prev in reversed(get_sdpa_bench_records()):
        if (
            prev["shape"] == _shape_id(shape)
            and prev["dtype"] == _dtype_id(dtype)
            and prev["is_causal"] == is_causal
            and prev["version"] == "pytorch_sdpa"
        ):
            if stats["mean"] > 0:
                speedup = prev["mean_ms"] / (stats["mean"] * 1e3)
            break

    omp = _omp_snapshot()
    record: Dict[str, Any] = {
        "version": info.name,
        "source": info.source,
        "shape": _shape_id(shape),
        "dtype": _dtype_id(dtype),
        "is_causal": bool(is_causal),
        "B": B,
        "N": N,
        "L": L,
        "S": S,
        "E": E,
        "Ev": Ev,
        "warmup_iters": WARMUP_ITERS,
        "bench_iters": BENCH_ITERS,
        "mean_ms": stats["mean"] * 1e3,
        "stddev_ms": stats["stddev"] * 1e3,
        "median_ms": stats["median"] * 1e3,
        "p90_ms": stats["p90"] * 1e3,
        "p99_ms": stats["p99"] * 1e3,
        "min_ms": stats["min"] * 1e3,
        "max_ms": stats["max"] * 1e3,
        "total_flops": flops["total_flops"],
        "total_flops_causal": flops["total_flops_causal"],
        "effective_flops": flops[effective_flops_key],
        "gflops": gflops_effective,
        "gflops_total": gflops_total,
        "gflops_causal": gflops_causal,
        "speedup_vs_pytorch": speedup,
        # 优先记录 PyTorch intra-op 线程数（它才是 pytorch_sdpa 路径的
        # 真实并行度）；omp_get_max_threads() 同时作为备考信息记在
        # ``omp_max_threads`` 里。
        "num_threads": str(torch.get_num_threads()),
        "omp_max_threads": omp.get("max_threads"),
        "omp_schedule": omp.get("OMP_SCHEDULE"),
        "omp_proc_bind": omp.get("OMP_PROC_BIND"),
        "has_openmp": omp.get("has_openmp"),
    }
    record_sdpa_bench(record)
