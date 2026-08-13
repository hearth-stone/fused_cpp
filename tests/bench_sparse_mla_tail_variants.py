# -*- coding: utf-8 -*-
"""Compare sparse MLA ragged-tail kernels on V4-like or tail-only workloads.

The timed region includes native plan construction, packing, and the forward
kernel. Pinning and NUMA placement are intentionally left to the caller.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable

import torch

from fused_cpp import _C


TAIL_VARIANTS = (
    "indexed_4x4",
    "indexed_4x4_2d",
    "masked_dense_8x8",
    "masked_dense_8x8_pruned",
    "masked_dense_8x8_pruned_2d",
    "masked_dense_8x8_fused_fmla",
    "masked_dense_8x8_fused_bfmlal",
    "masked_dense_8x8_fused_bfmmla",
)
WORKLOADS = (
    "v4-forward",
    "dsv4-sparse",
    "tail-causal",
    "tail-v4-mix",
)


def _build_v4_like_indices(s_q: int, compressed_capacity: int) -> tuple[torch.Tensor, int]:
    """Build compressed-prefix + dense-causal indices with negative padding."""
    topk = compressed_capacity + s_q
    token = torch.arange(s_q, dtype=torch.int64).reshape(s_q, 1)
    col = torch.arange(topk, dtype=torch.int64).reshape(1, topk)
    compressed_len = (token + 1) // 4
    dense_len = token + 1
    compressed = col < compressed_len
    dense = (col >= compressed_len) & (col < compressed_len + dense_len)
    values = torch.where(
        compressed,
        col,
        torch.where(dense, compressed_capacity + col - compressed_len, torch.full_like(col, -1)),
    )
    valid_pairs = int((compressed_len + dense_len).sum().item())
    return values.to(torch.int32).reshape(s_q, 1, topk).contiguous(), valid_pairs


def _build_dsv4_sparse_indices(
    s_q: int,
    compressed_capacity: int,
    window_size: int,
    compress_ratio: int,
    context_start: int,
) -> tuple[torch.Tensor, int, int]:
    """Build DSV4 top-k compressed indices plus a sliding recent window."""
    if min(s_q, compressed_capacity, window_size, compress_ratio) <= 0 or context_start < 0:
        raise ValueError("DSV4 sparse shape parameters must be positive")

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
            # Model the selected 4a path as sorted, non-contiguous top-k values
            # from the compressed cache, rather than another dense triangle.
            selected = torch.linspace(
                0,
                compressed_source - 1,
                compressed_len,
                dtype=torch.float64,
            ).round().to(torch.int32)
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


def _build_tail_only_indices(tail_blocks: int, workload: str) -> tuple[torch.Tensor, int, int, int]:
    """Build independent K/V tiles so every eight-query block is a masked tail."""
    if tail_blocks <= 0:
        raise ValueError(f"tail_blocks must be positive, got {tail_blocks}")

    block = torch.arange(tail_blocks, dtype=torch.int64).reshape(tail_blocks, 1, 1)
    row = torch.arange(8, dtype=torch.int64).reshape(1, 8, 1)
    col = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8)
    invalid = torch.full((1,), -1, dtype=torch.int64)

    if workload == "tail-causal":
        starts = block * 8
        values = torch.where(col <= row, starts + col, invalid)
        valid_pairs = tail_blocks * 36
        return (
            values.to(torch.int32).reshape(tail_blocks * 8, 1, 8).contiguous(),
            tail_blocks * 8,
            valid_pairs,
            tail_blocks,
        )

    if workload != "tail-v4-mix":
        raise ValueError(f"unsupported tail-only workload: {workload}")

    starts = block * 16
    compressed_shape = torch.tensor((0, 0, 0, 1, 1, 1, 1, 2), dtype=torch.int64).reshape(1, 8, 1)
    compressed_len = 2 * (block % 4) + compressed_shape
    compressed = torch.where(col < compressed_len, starts + col, invalid)
    causal = torch.where(col <= row, starts + 8 + col, invalid)
    values = torch.cat((compressed, causal), dim=2)
    valid_pairs = int(compressed_len.sum().item()) + tail_blocks * 36
    return (
        values.to(torch.int32).reshape(tail_blocks * 8, 1, 16).contiguous(),
        tail_blocks * 16,
        valid_pairs,
        tail_blocks * 2,
    )


def _bench_interleaved(
    run: Callable[[str], torch.Tensor],
    variants: list[str],
    warmup: int,
    iters: int,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        for variant in variants:
            run(variant)
    times = {variant: [] for variant in variants}
    for iteration in range(iters):
        offset = iteration % len(variants)
        order = variants[offset:] + variants[:offset]
        for variant in order:
            start = time.perf_counter()
            run(variant)
            times[variant].append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, default="v4-forward")
    parser.add_argument("--s-q", type=int, default=2048)
    parser.add_argument("--h-q", type=int, default=32)
    parser.add_argument("--d-qk", type=int, default=192)
    parser.add_argument("--d-v", type=int, default=128)
    parser.add_argument("--compressed-capacity", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--context-start", type=int, default=0)
    parser.add_argument("--tail-blocks", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--variants", nargs="+", choices=TAIL_VARIANTS, default=list(TAIL_VARIANTS))
    args = parser.parse_args()

    if not hasattr(_C, "_flash_mla_sparse_fwd_variant"):
        raise RuntimeError("extension was built without sparse MLA tail variants")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(args.seed)
    if args.workload == "v4-forward":
        s_q = args.s_q
        s_kv = args.compressed_capacity + s_q
        indices, valid_pairs = _build_v4_like_indices(s_q, args.compressed_capacity)
        tail_tiles = (s_q // 8) * 2
    elif args.workload == "dsv4-sparse":
        s_q = args.s_q
        indices, valid_pairs, s_kv = _build_dsv4_sparse_indices(
            s_q,
            args.compressed_capacity,
            args.window_size,
            args.compress_ratio,
            args.context_start,
        )
        tail_tiles = (s_q // 8) * 2
    else:
        indices, s_kv, valid_pairs, tail_tiles = _build_tail_only_indices(args.tail_blocks, args.workload)
        s_q = int(indices.shape[0])

    q = torch.randn(s_q, args.h_q, args.d_qk).bfloat16()
    kv = torch.randn(s_kv, 1, args.d_qk).bfloat16()
    scale = 1.0 / (args.d_qk**0.5)
    output_buffers = {
        variant: torch.empty((s_q, args.h_q, args.d_v), dtype=q.dtype) for variant in TAIL_VARIANTS
    }

    def run(variant: str) -> torch.Tensor:
        return _C._flash_mla_sparse_fwd_variant(
            q,
            kv,
            indices,
            scale,
            variant,
            d_v=args.d_v,
            out=output_buffers[variant],
        )

    reference = run("indexed_4x4").clone()
    max_abs: dict[str, float] = {"indexed_4x4": 0.0}
    for variant in args.variants:
        candidate = run(variant)
        max_abs[variant] = float((candidate.float() - reference.float()).abs().max().item())
        torch.testing.assert_close(candidate, reference, atol=1e-2, rtol=1e-2)

    source_flops = 2.0 * valid_pairs * args.h_q * (args.d_qk + args.d_v)
    print(
        f"workload={args.workload},tail_tiles={tail_tiles},"
        f"shape=q[{s_q},{args.h_q},{args.d_qk}],kv[{s_kv},1,{args.d_qk}],d_v={args.d_v},"
        f"topk={indices.shape[-1]},context_start={args.context_start},"
        f"valid_pairs_per_head={valid_pairs},source_flops={source_flops:.0f}"
    )
    print(
        f"threads={args.threads},torch_threads={torch.get_num_threads()},warmup={args.warmup},iters={args.iters},"
        f"seed={args.seed},schedule=interleaved_rotating"
    )
    print("variant,median_ms,min_ms,max_ms,source_gflops,max_abs_vs_indexed,checksum")
    times_by_variant = _bench_interleaved(run, args.variants, args.warmup, args.iters)
    for variant in args.variants:
        times = times_by_variant[variant]
        median = statistics.median(times)
        checksum = float(output_buffers[variant].float().sum().item())
        print(
            f"{variant},{median * 1e3:.3f},{min(times) * 1e3:.3f},{max(times) * 1e3:.3f},"
            f"{source_flops / median / 1e9:.2f},{max_abs[variant]:.6g},{checksum:.6f}",
            flush=True,
        )
        print("raw_ms=" + ",".join(f"{value * 1e3:.3f}" for value in times), flush=True)


if __name__ == "__main__":
    main()
