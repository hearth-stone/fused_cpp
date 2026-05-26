#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 SDPA 性能差异的根因。"""
import gc
import math
import time

import torch
import torch.nn.functional as F

from fused_cpp.sdpa import scaled_dot_product_attention


def bench(fn, *args, warmup=5, iters=20, **kwargs):
    """计时函数。"""
    for __ in range(warmup):
        fn(*args, **kwargs)
    gc.collect()
    t0 = time.perf_counter()
    for __ in range(iters):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / iters


def manual_sdpa_python(q, k, v, scale=None):
    """纯 Python/PyTorch 手写的 SDPA，与 C++ 实现逻辑完全一致。"""
    dtype = q.dtype
    scale = scale or (1.0 / math.sqrt(q.size(-1)))
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_f)
    return out.to(dtype) if dtype != torch.float32 else out


def main():
    torch.manual_seed(42)
    B, N, L, E, Ev = 4, 16, 512, 192, 128

    q = torch.randn(B, N, L, E, dtype=torch.bfloat16)
    k = torch.randn(B, N, L, E, dtype=torch.bfloat16)
    v = torch.randn(B, N, L, Ev, dtype=torch.bfloat16)

    print(f"Shape: Q={list(q.shape)}, K={list(k.shape)}, V={list(v.shape)}")
    print(f"dtype: {q.dtype}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    with torch.no_grad():
        # ── 测试 1: 三种实现对比 ──
        print("=" * 70)
        print("测试 1: 三种实现对比 (均在 torch.no_grad 下)")
        print("=" * 70)

        t_custom = bench(scaled_dot_product_attention, q, k, v)
        t_pytorch = bench(F.scaled_dot_product_attention, q, k, v)
        t_manual = bench(manual_sdpa_python, q, k, v)

        print(f"  C++ SDPA      : {t_custom * 1e3:8.2f} ms")
        print(f"  PyTorch SDPA  : {t_pytorch * 1e3:8.2f} ms")
        print(f"  手写 Python   : {t_manual * 1e3:8.2f} ms")
        print(f"  C++ vs PyTorch: {t_pytorch / t_custom:.3f}x")
        print(f"  手写 vs PyTorch: {t_pytorch / t_manual:.3f}x")
        print()

        # ── 测试 2: 交换运行顺序 ──
        print("=" * 70)
        print("测试 2: 交换运行顺序 (先 PyTorch 后 C++)")
        print("=" * 70)

        t_pytorch2 = bench(F.scaled_dot_product_attention, q, k, v)
        t_custom2 = bench(scaled_dot_product_attention, q, k, v)

        print(f"  PyTorch SDPA  : {t_pytorch2 * 1e3:8.2f} ms")
        print(f"  C++ SDPA      : {t_custom2 * 1e3:8.2f} ms")
        print(f"  C++ vs PyTorch: {t_pytorch2 / t_custom2:.3f}x")
        print()

        # ── 测试 3: 分解 PyTorch SDPA 的各步骤 ──
        print("=" * 70)
        print("测试 3: 分解各步骤耗时")
        print("=" * 70)

        # 步骤 a: bf16 -> fp32 转换
        t_cast = bench(lambda: (q.float(), k.float(), v.float()))
        print(f"  bf16->fp32 转换: {t_cast * 1e3:8.2f} ms")

        q_f, k_f, v_f = q.float(), k.float(), v.float()

        # 步骤 b: matmul Q@K^T
        t_qk = bench(torch.matmul, q_f, k_f.transpose(-2, -1))
        print(f"  Q@K^T matmul  : {t_qk * 1e3:8.2f} ms")

        scores = torch.matmul(q_f, k_f.transpose(-2, -1))

        # 步骤 c: softmax
        t_sm = bench(torch.softmax, scores, -1)
        print(f"  softmax       : {t_sm * 1e3:8.2f} ms")

        weights = torch.softmax(scores, dim=-1)

        # 步骤 d: matmul attn@V
        t_av = bench(torch.matmul, weights, v_f)
        print(f"  attn@V matmul : {t_av * 1e3:8.2f} ms")

        total_manual = t_cast + t_qk + t_sm + t_av
        print(f"  手动步骤合计  : {total_manual * 1e3:8.2f} ms")
        print(f"  PyTorch SDPA  : {t_pytorch * 1e3:8.2f} ms")
        print(f"  差距          : {t_pytorch / total_manual:.3f}x")


if __name__ == "__main__":
    main()
