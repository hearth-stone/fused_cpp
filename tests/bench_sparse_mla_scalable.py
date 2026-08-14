# SPDX-License-Identifier: Apache-2.0
"""Benchmark the public Sparse MLA path on a DSV4-like sparse workload."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from fused_cpp.sparse_mla import flash_mla_sparse_fwd


def build_indices(
    s_q: int,
    compressed_capacity: int,
    window_size: int,
    compress_ratio: int,
    context_start: int,
) -> tuple[torch.Tensor, int, int]:
    topk = compressed_capacity + window_size
    values = torch.full((s_q, topk), -1, dtype=torch.int32)
    valid_pairs = 0
    max_position = context_start + s_q
    compressed_region = (max_position + compress_ratio - 1) // compress_ratio
    for token in range(s_q):
        position = context_start + token
        compressed_source = (position + 1) // compress_ratio
        compressed_len = min(compressed_source, compressed_capacity)
        if compressed_len:
            selected = (
                torch.linspace(
                    0,
                    compressed_source - 1,
                    compressed_len,
                    dtype=torch.float64,
                )
                .round()
                .to(torch.int32)
            )
            values[token, :compressed_len] = selected
        recent_len = min(position + 1, window_size)
        recent_start = position + 1 - recent_len
        values[token, compressed_len : compressed_len + recent_len] = (
            compressed_region
            + torch.arange(recent_start, position + 1, dtype=torch.int32)
        )
        valid_pairs += compressed_len + recent_len
    return (
        values.reshape(s_q, 1, topk).contiguous(),
        valid_pairs,
        compressed_region + max_position,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-q", type=int, default=2048)
    parser.add_argument("--h-q", type=int, default=32)
    parser.add_argument("--d-qk", type=int, default=192)
    parser.add_argument("--d-v", type=int, default=128)
    parser.add_argument("--compressed-capacity", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--context-start", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    indices, valid_pairs, s_kv = build_indices(
        args.s_q,
        args.compressed_capacity,
        args.window_size,
        args.compress_ratio,
        args.context_start,
    )
    q = torch.randn(args.s_q, args.h_q, args.d_qk).bfloat16()
    kv = torch.randn(s_kv, 1, args.d_qk).bfloat16()
    output = torch.empty(
        (args.s_q, args.h_q, args.d_v), dtype=torch.bfloat16
    )
    scale = 1.0 / (args.d_qk**0.5)

    def run() -> torch.Tensor:
        return flash_mla_sparse_fwd(
            q, kv, indices, scale, d_v=args.d_v, out=output
        )

    for _ in range(args.warmup):
        run()
    samples = []
    for _ in range(args.iters):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1e3)
    source_flops = 2.0 * valid_pairs * args.h_q * (args.d_qk + args.d_v)
    median_ms = statistics.median(samples)
    print(
        f"shape=q[{args.s_q},{args.h_q},{args.d_qk}],"
        f"kv[{s_kv},1,{args.d_qk}],topk={indices.shape[-1]},"
        f"d_v={args.d_v},valid_pairs={valid_pairs}"
    )
    print(
        f"threads={args.threads},warmup={args.warmup},iters={args.iters},"
        f"seed={args.seed},median_ms={median_ms:.3f},"
        f"min_ms={min(samples):.3f},max_ms={max(samples):.3f},"
        f"source_gflops={source_flops / (median_ms / 1e3) / 1e9:.2f},"
        f"checksum={output.float().sum().item():.6f}"
    )
    print("raw_ms=" + ",".join(f"{sample:.3f}" for sample in samples))


if __name__ == "__main__":
    main()
