# -*- coding: utf-8 -*-
"""FLOP/s 性能基准测试：自定义 SDPA vs PyTorch F.scaled_dot_product_attention。"""
import gc
import time

import pytest
import torch
import torch.nn.functional as F

from fused_cpp.sdpa import scaled_dot_product_attention


# ── 配置 ──────────────────────────────────────────────────────────────────

WARMUP_ITERS = 5
BENCH_ITERS = 20

# MLA 特定参数
NUM_HEADS = 16
QK_HEAD_DIM = 192   # qk_nope_head_dim(128) + qk_rope_head_dim(64)
V_HEAD_DIM = 128
BATCH_SIZE = 4

SEQ_LENS = [128, 256, 512, 1024, 2048]


# ── 辅助函数 ──────────────────────────────────────────────────────────────

def _compute_flops(
    batch_size: int,
    num_heads: int,
    seq_q: int,
    seq_k: int,
    qk_head_dim: int,
    v_head_dim: int,
) -> int:
    """计算 SDPA 的理论 FLOPs。

    FLOPs = 2*B*N*L*S*E (Q@K^T) + 2*B*N*L*S*Ev (attn@V) + B*N*L*S (softmax)
    """
    qk_flops = 2 * batch_size * num_heads * seq_q * seq_k * qk_head_dim
    av_flops = 2 * batch_size * num_heads * seq_q * seq_k * v_head_dim
    softmax_flops = batch_size * num_heads * seq_q * seq_k
    return qk_flops + av_flops + softmax_flops


def _bench_fn(fn, *args, **kwargs) -> float:
    """对函数进行预热和计时，返回平均耗时（秒）。

    注意：调用方需自行确保在 torch.no_grad() 上下文中调用，
    以保证与 C++ 实现（内置 NoGradGuard）的公平对比。
    """
    # 预热
    for __ in range(WARMUP_ITERS):
        fn(*args, **kwargs)

    # 强制 GC，减少计时期间的干扰
    gc.collect()

    # 正式计时
    start = time.perf_counter()
    for __ in range(BENCH_ITERS):
        fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed / BENCH_ITERS


# ── 性能测试 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_sdpa_benchmark(seq_len):
    """SDPA 性能基准测试（MLA 参数）。"""
    torch.manual_seed(42)

    q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, QK_HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, QK_HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, V_HEAD_DIM, dtype=torch.bfloat16)

    flops = _compute_flops(BATCH_SIZE, NUM_HEADS, seq_len, seq_len, QK_HEAD_DIM, V_HEAD_DIM)

    # 两个实现都在 no_grad 下运行，确保公平对比
    with torch.no_grad():
        # 自定义 SDPA
        custom_time = _bench_fn(scaled_dot_product_attention, q, k, v)
        custom_gflops = flops / (custom_time * 1e9)

        # PyTorch SDPA
        pytorch_time = _bench_fn(F.scaled_dot_product_attention, q, k, v)
        pytorch_gflops = flops / (pytorch_time * 1e9)

    speedup = pytorch_time / custom_time if custom_time > 0 else float("inf")

    print(
        f"\n{'=' * 80}\n"
        f"[MLA SDPA Benchmark] seq_len={seq_len}, B={BATCH_SIZE}, "
        f"N={NUM_HEADS}, qk_dim={QK_HEAD_DIM}, v_dim={V_HEAD_DIM}\n"
        f"  自定义 SDPA : {custom_time * 1e3:8.2f} ms  |  {custom_gflops:8.2f} GFLOP/s\n"
        f"  PyTorch SDPA: {pytorch_time * 1e3:8.2f} ms  |  {pytorch_gflops:8.2f} GFLOP/s\n"
        f"  加速比      : {speedup:.3f}x\n"
        f"{'=' * 80}"
    )


# ── 汇总报告 ──────────────────────────────────────────────────────────────

def test_sdpa_benchmark_summary():
    """输出格式化的性能报告表格。"""
    torch.manual_seed(42)

    header = (
        f"\n{'=' * 100}\n"
        f"{'序列长度':>8} | {'Batch':>5} | "
        f"{'自定义 (ms)':>12} | {'自定义 GFLOP/s':>14} | "
        f"{'PyTorch (ms)':>12} | {'PyTorch GFLOP/s':>15} | "
        f"{'加速比':>6}\n"
        f"{'-' * 100}"
    )
    print(header)

    with torch.no_grad():
        for seq_len in SEQ_LENS:
            q = torch.randn(
                BATCH_SIZE, NUM_HEADS, seq_len, QK_HEAD_DIM, dtype=torch.bfloat16,
            )
            k = torch.randn(
                BATCH_SIZE, NUM_HEADS, seq_len, QK_HEAD_DIM, dtype=torch.bfloat16,
            )
            v = torch.randn(
                BATCH_SIZE, NUM_HEADS, seq_len, V_HEAD_DIM, dtype=torch.bfloat16,
            )

            flops = _compute_flops(
                BATCH_SIZE, NUM_HEADS, seq_len, seq_len, QK_HEAD_DIM, V_HEAD_DIM,
            )

            custom_time = _bench_fn(scaled_dot_product_attention, q, k, v)
            custom_gflops = flops / (custom_time * 1e9)

            pytorch_time = _bench_fn(F.scaled_dot_product_attention, q, k, v)
            pytorch_gflops = flops / (pytorch_time * 1e9)

            speedup = pytorch_time / custom_time if custom_time > 0 else float("inf")

            print(
                f"{seq_len:>8} | {BATCH_SIZE:>5} | "
                f"{custom_time * 1e3:>12.2f} | {custom_gflops:>14.2f} | "
                f"{pytorch_time * 1e3:>12.2f} | {pytorch_gflops:>15.2f} | "
                f"{speedup:>6.3f}x"
            )

    print(f"{'=' * 100}")
    print(
        "\n注意：自定义实现仅为最小化前向推理路径，跳过了 PyTorch SDPA 内部\n"
        "为通用性和反向传播准备的额外工作（如 logsumexp 计算、额外的\n"
        "contiguous 转换、scale 乘法策略差异等），因此可能出现自定义实现\n"
        "更快的情况，这并非真正的算法优化。"
    )
