#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDPA 中等形状单线程基准（远端 Huawei aarch64 适用）。

约定：
  * 单线程：OMP_NUM_THREADS=1 + torch.set_num_threads(1)。
    单线程下 sdpa_flash2_neon_l3kv 永远走 Path A（head-grouping 路径需要
    多线程才有意义），与 sdpa_flash2_neon_cache 拓扑一致，便于纯 micro
    级别对比。
  * 形状集合：覆盖标准 / MLA / 不同 (L, S) / 不同 head_dim 的中等规模，
    与 macOS 上跑过的那一套对齐，方便横向对比。
  * 度量：5 次 warmup + 20 次正式测量；输出 median ms + GFLOPS。

用法：
  cd /home/zhangxu/vllm-aarch64/fused_cpp
  python tests/run_medium_bench_st.py
"""

from __future__ import annotations

import gc
import os
import statistics
import time

# 必须在 import torch 之前设。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_PROC_BIND", "true")
os.environ.setdefault("OMP_PLACES", "cores")

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

from fused_cpp._C import (  # noqa: E402 - thread environment must be set first
    scaled_dot_product_attention_versioned as cpp_sdpa,
    list_sdpa_versions,
)
from fused_cpp.sdpa_flops import compute_sdpa_flops, gflops  # noqa: E402


SHAPES = [
    # (label, B, N, L, S, E, Ev)
    ("std    256", 1, 8, 256, 256, 64, 64),
    ("std    512", 1, 8, 512, 512, 64, 64),
    ("std    1024", 1, 8, 1024, 1024, 64, 64),
    ("std128 1024", 1, 8, 1024, 1024, 128, 128),
    ("MLA    512", 1, 16, 512, 512, 192, 128),
    ("MLA    1024", 1, 16, 1024, 1024, 192, 128),
    ("multi-bn", 2, 16, 1024, 1024, 128, 128),
]

DTYPES = [("fp32", torch.float32), ("bf16", torch.bfloat16)]
IS_CAUSAL = False
WARMUP = 5
ITERS = 20

VERSIONS = list_sdpa_versions()


def make_qkv(B, N, L, S, E, Ev, dtype, seed=1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, N, L, E, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(B, N, S, E, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, N, S, Ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


def time_one(fn):
    for _ in range(WARMUP):
        fn()
    gc.collect()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def bench_cpp(version, q, k, v):
    return time_one(
        lambda: cpp_sdpa(q, k, v, None, 0.0, IS_CAUSAL, None, False, version),
    )


def bench_torch_math(q, k, v):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def call():
            with sdpa_kernel(SDPBackend.MATH):
                F.scaled_dot_product_attention(q, k, v, is_causal=IS_CAUSAL)
    except Exception:

        def call():
            F.scaled_dot_product_attention(q, k, v, is_causal=IS_CAUSAL)

    return time_one(call)


def fmt_row(name, ms_med, gfs, ms_min):
    return f"  {name:<32} {ms_med:>9.2f} ms  {gfs:>8.1f} GF/s   (min {ms_min:.2f} ms)"


def main() -> None:
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, torch.get_num_threads()={torch.get_num_threads()}")
    print(f"warmup={WARMUP}, iters={ITERS}, is_causal={IS_CAUSAL}")
    print(f"Registered C++ versions: {VERSIONS}")
    print()

    for label, B, N, L, S, E, Ev in SHAPES:
        flops = compute_sdpa_flops(B, N, L, S, E, Ev, is_causal=IS_CAUSAL)["total_flops"]
        print(f"=== {label}  shape=(B={B},N={N},L={L},S={S},E={E},Ev={Ev})  total_flops={flops / 1e9:.2f} G ===")
        for dt_label, dtype in DTYPES:
            q, k, v = make_qkv(B, N, L, S, E, Ev, dtype)
            print(f"  -- dtype={dt_label} --")

            try:
                ms, mmin = bench_torch_math(q, k, v)
                print(fmt_row("torch.F.sdpa(MATH)", ms * 1000, gflops(flops, ms), mmin * 1000))
            except Exception as exc:
                print(f"  torch.F.sdpa(MATH) FAILED: {exc}")

            # 大 fp32 形状下 naive 太慢，跳过
            skip_naive_fp32 = (
                L * S * E * 4 > 64 * 1024 * 1024  # >64MB scores buffer
                and dtype == torch.float32
            )

            for ver in VERSIONS:
                if ver == "naive" and skip_naive_fp32:
                    print(f"  {ver:<32} {dt_label}  [skipped: scores buffer too large]")
                    continue
                try:
                    ms, mmin = bench_cpp(ver, q, k, v)
                    print(fmt_row(ver, ms * 1000, gflops(flops, ms), mmin * 1000))
                except Exception as exc:
                    print(f"  {ver:<32} FAILED: {exc}")
            gc.collect()
        print()


if __name__ == "__main__":
    main()
