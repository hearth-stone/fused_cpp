# -*- coding: utf-8 -*-
"""DeepSeek V4-style flash attention benchmark case.

Default case:
  total input tokens = 2048
  seq_lens = [1024, 1024]

In dense SDPA form this is represented as B=2, L=S=1024 with
is_causal=True, i.e. two independent lower-triangular attention matrices.
This matches a varlen prefill batch with cu_seqlens=[0, 1024, 2048] without
requiring a large block-diagonal additive mask.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable

import torch

from fused_cpp.sdpa import sdpa_versioned


def _parse_seq_lens(value: str) -> list[int]:
    seq_lens = [int(part) for part in value.split(",") if part]
    if not seq_lens:
        raise argparse.ArgumentTypeError("seq_lens must not be empty")
    if any(seq_len <= 0 for seq_len in seq_lens):
        raise argparse.ArgumentTypeError("all seq_lens must be positive")
    return seq_lens


def _dtype(value: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}") from exc


def _bench(fn: Callable[[], object], warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    gc.collect()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), min(times), max(times)


def _make_qkv(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    qk_dim: int,
    v_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(batch, heads, seq_len, qk_dim, generator=gen, dtype=torch.float32).to(dtype)
    k = torch.randn(batch, heads, seq_len, qk_dim, generator=gen, dtype=torch.float32).to(dtype)
    v = torch.randn(batch, heads, seq_len, v_dim, generator=gen, dtype=torch.float32).to(dtype)
    return q, k, v


def _effective_pairs(seq_lens: list[int]) -> int:
    return sum(seq_len * (seq_len + 1) // 2 for seq_len in seq_lens)


def _attention_flops(*, heads: int, pairs: int, qk_dim: int, v_dim: int) -> float:
    return float(heads * pairs * (2 * (qk_dim + v_dim) + 5))


def _run_equal_len_segments(args: argparse.Namespace, seq_lens: list[int]) -> None:
    if len(set(seq_lens)) != 1:
        raise ValueError(
            "The dense batched case requires equal seq_lens. "
            "Use equal segments such as 1024,1024 to model two lower triangles."
        )

    batch = len(seq_lens)
    seq_len = seq_lens[0]
    q, k, v = _make_qkv(
        batch=batch,
        heads=args.heads,
        seq_len=seq_len,
        qk_dim=args.qk_dim,
        v_dim=args.v_dim,
        dtype=args.dtype,
        seed=args.seed,
    )

    def fn() -> torch.Tensor:
        return sdpa_versioned(
            q,
            k,
            v,
            version=args.version,
            is_causal=True,
            scale=args.scale,
        )

    out = fn()
    assert out.shape == (batch, args.heads, seq_len, args.v_dim)
    median, min_t, max_t = _bench(fn, args.warmup, args.iters)

    pairs = _effective_pairs(seq_lens)
    flops = _attention_flops(
        heads=args.heads,
        pairs=pairs,
        qk_dim=args.qk_dim,
        v_dim=args.v_dim,
    )
    print(
        "deepseek_v4_flash_2048_two_triangles,"
        f"{args.version},{str(args.dtype).replace('torch.', '')},"
        f"{batch},{args.heads},{seq_len},{seq_len},{args.qk_dim},{args.v_dim},"
        f'{sum(seq_lens)},"{",".join(map(str, seq_lens))}",{pairs},'
        f"{median * 1e3:.3f},{min_t * 1e3:.3f},{max_t * 1e3:.3f},"
        f"{flops / median / 1e9:.2f}",
        flush=True,
    )


def _run_global(args: argparse.Namespace, total_tokens: int) -> None:
    q, k, v = _make_qkv(
        batch=1,
        heads=args.heads,
        seq_len=total_tokens,
        qk_dim=args.qk_dim,
        v_dim=args.v_dim,
        dtype=args.dtype,
        seed=args.seed + 1,
    )

    def fn() -> torch.Tensor:
        return sdpa_versioned(
            q,
            k,
            v,
            version=args.version,
            is_causal=True,
            scale=args.scale,
        )

    out = fn()
    assert out.shape == (1, args.heads, total_tokens, args.v_dim)
    median, min_t, max_t = _bench(fn, args.warmup, args.iters)

    pairs = _effective_pairs([total_tokens])
    flops = _attention_flops(
        heads=args.heads,
        pairs=pairs,
        qk_dim=args.qk_dim,
        v_dim=args.v_dim,
    )
    print(
        "deepseek_v4_flash_2048_one_triangle,"
        f"{args.version},{str(args.dtype).replace('torch.', '')},"
        f"1,{args.heads},{total_tokens},{total_tokens},{args.qk_dim},{args.v_dim},"
        f'{total_tokens},"{total_tokens}",{pairs},'
        f"{median * 1e3:.3f},{min_t * 1e3:.3f},{max_t * 1e3:.3f},"
        f"{flops / median / 1e9:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        default="flash2_neon_l3kv_packqkv_pbf16pv",
        help="SDPA version to run.",
    )
    parser.add_argument(
        "--seq-lens",
        type=_parse_seq_lens,
        default=[1024, 1024],
        help="Comma-separated sequence lengths. Default models two lower triangles.",
    )
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--qk-dim", type=int, default=192)
    parser.add_argument("--v-dim", type=int, default=128)
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20240608)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--compare-global",
        action="store_true",
        help="Also run one global 2048-token lower-triangular attention.",
    )
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    total_tokens = sum(args.seq_lens)
    print(f"torch_threads={torch.get_num_threads()}")
    print(
        "case,version,dtype,B,N,L,S,E,Ev,total_tokens,seq_lens,effective_pairs,median_ms,min_ms,max_ms,effective_gflops"
    )
    _run_equal_len_segments(args, args.seq_lens)
    if args.compare_global:
        _run_global(args, total_tokens)


if __name__ == "__main__":
    main()
