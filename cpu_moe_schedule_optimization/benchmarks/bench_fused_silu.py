# -*- coding: utf-8 -*-
"""Benchmark: fused SiLU-and-mul w13 epilogue vs the baseline w13+activation.

Compares end-to-end fused_moe_bf16_tiled latency for:
  - baseline  (prepare fuse_silu=False, separate activation pass)
  - fused     (prepare fuse_silu=True, poly4/5/6 exp in the w13 store epilogue)

Run pinned:
  OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
    python cpu_moe_schedule_optimization/benchmarks/bench_fused_silu.py
"""

from __future__ import annotations

import argparse
import time

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


def _bf16(*shape, scale=0.2):
    return (torch.randn(*shape) * scale).to(torch.bfloat16)


def _time(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]  # median ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--ffn-hidden-size", type=int, default=512)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--tokens", default="8,64,256,1024")
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H, F = args.hidden_size, args.ffn_hidden_size
    E, top_k = args.num_experts, args.top_k
    torch.manual_seed(args.seed)

    w13 = _bf16(E, 2 * F, H)
    w2 = _bf16(E, H, F)
    base_w = prepare_fused_moe_bf16_tiled_weights(w13, w2)
    fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)

    token_list = [int(t) for t in args.tokens.split(",")]
    thread_list = [int(t) for t in args.threads.split(",")]

    print(f"H={H} F={F} E={E} top_k={top_k}; median ms; speedup = base/fused")
    print("tokens threads   base_ms  p5_ms  p6_ms  p4_ms  spd5   maxdiff5")
    for tokens in token_list:
        x = _bf16(tokens, H)
        topk_ids = torch.tensor(
            [[(i + j) % E for j in range(top_k)] for i in range(tokens)],
            dtype=torch.int32,
        )
        topk_weights = torch.softmax(torch.randn(tokens, top_k), dim=-1)
        for nt in thread_list:

            def _base():
                return fused_moe_bf16_tiled(x, base_w, topk_weights, topk_ids, num_threads=nt)

            def _fused(d):
                return fused_moe_bf16_tiled(
                    x,
                    fused_w,
                    topk_weights,
                    topk_ids,
                    num_threads=nt,
                    silu_poly_degree=d,
                )

            base_ms = _time(_base, args.warmup, args.runs)
            p5 = _time(lambda: _fused(5), args.warmup, args.runs)
            p6 = _time(lambda: _fused(6), args.warmup, args.runs)
            p4 = _time(lambda: _fused(4), args.warmup, args.runs)
            ref = _base().float()
            got = _fused(5).float()
            maxdiff = (ref - got).abs().max().item()
            print(
                f"{tokens:6d} {nt:6d}  {base_ms:8.3f} {p5:6.3f} {p6:6.3f} {p4:6.3f}  {base_ms / p5:5.2f}  {maxdiff:.4f}"
            )


if __name__ == "__main__":
    main()
