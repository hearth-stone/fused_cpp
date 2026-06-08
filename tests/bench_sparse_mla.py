# -*- coding: utf-8 -*-
"""Sparse MLA microbenchmark.

This script is intentionally dependency-light so it can run on the AWS arm64
benchmark host without NumPy.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import torch

from fused_cpp.sdpa import sdpa_versioned
from fused_cpp.sparse_mla import flash_mla_sparse_fwd


def _bench(fn: Callable[[], object], warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), min(times), max(times)


def _dense_indices(s_q: int, topk: int) -> torch.Tensor:
    return (
        torch.arange(topk, dtype=torch.int32)
        .reshape(1, 1, topk)
        .expand(s_q, 1, topk)
        .clone()
    )


def _hybrid_indices(s_q: int, s_kv: int, topk: int, shared: int = 64) -> torch.Tensor:
    common = (
        torch.arange(shared, dtype=torch.int32)
        .reshape(1, 1, shared)
        .expand(s_q, 1, shared)
    )
    tail = torch.randint(shared, s_kv, (s_q, 1, topk - shared), dtype=torch.int32)
    return torch.cat([common, tail], dim=-1).clone()


def _random_indices(s_q: int, s_kv: int, topk: int) -> torch.Tensor:
    return torch.randint(0, s_kv, (s_q, 1, topk), dtype=torch.int32)


def _approx_attention_flops(
    s_q: int,
    h_q: int,
    topk: int,
    d_qk: int,
    d_v: int,
) -> float:
    return 2.0 * s_q * h_q * topk * (d_qk + d_v)


def _run_sparse_case(
    name: str,
    s_q: int,
    h_q: int,
    s_kv: int,
    d_qk: int,
    d_v: int,
    topk: int,
    *,
    seed: int,
    warmup: int,
    iters: int,
) -> None:
    torch.manual_seed(seed)
    q = torch.randn(s_q, h_q, d_qk).bfloat16()
    kv = torch.randn(s_kv, 1, d_qk).bfloat16()
    if name == "sparse_dense_shared":
        indices = _dense_indices(s_q, topk)
    elif name == "sparse_hybrid_64_shared_tail":
        indices = _hybrid_indices(s_q, s_kv, topk)
    else:
        indices = _random_indices(s_q, s_kv, topk)

    scale = 1.0 / (d_qk**0.5)

    def fn() -> object:
        return flash_mla_sparse_fwd(q, kv, indices, scale, d_v)

    out = fn()
    assert out.shape == (s_q, h_q, d_v)
    median, min_t, max_t = _bench(fn, warmup, iters)
    flops = _approx_attention_flops(s_q, h_q, topk, d_qk, d_v)
    print(
        f"{name},{s_q},{h_q},{s_kv},{d_qk},{d_v},{topk},"
        f"{median * 1e3:.3f},{min_t * 1e3:.3f},{max_t * 1e3:.3f},"
        f"{flops / median / 1e9:.2f}",
        flush=True,
    )


def _run_sdpa_packqkv_case(
    s_q: int,
    h_q: int,
    d_qk: int,
    d_v: int,
    topk: int,
    *,
    seed: int,
    warmup: int,
    iters: int,
) -> None:
    torch.manual_seed(seed)
    q = torch.randn(1, h_q, s_q, d_qk).bfloat16()
    k = torch.randn(1, h_q, topk, d_qk).bfloat16()
    v = torch.randn(1, h_q, topk, d_v).bfloat16()
    scale = 1.0 / (d_qk**0.5)

    def fn() -> object:
        return sdpa_versioned(
            q,
            k,
            v,
            version="flash2_neon_l3kv_packqkv_pbf16pv",
            scale=scale,
        )

    out = fn()
    assert out.shape == (1, h_q, s_q, d_v)
    median, min_t, max_t = _bench(fn, warmup, iters)
    flops = _approx_attention_flops(s_q, h_q, topk, d_qk, d_v)
    print(
        f"sdpa_packqkv_reference,{s_q},{h_q},{topk},{d_qk},{d_v},{topk},"
        f"{median * 1e3:.3f},{min_t * 1e3:.3f},{max_t * 1e3:.3f},"
        f"{flops / median / 1e9:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(f"torch_threads={torch.get_num_threads()}")
    print("case,s_q,h_q,s_kv,d,d_v,topk,median_ms,min_ms,max_ms,approx_gflops")

    sparse_cases = [
        ("sparse_dense_shared", 512, 16, 1024, 128, 128, 128),
        ("sparse_dense_shared", 1024, 16, 2048, 128, 128, 128),
        ("sparse_hybrid_64_shared_tail", 512, 16, 1024, 128, 128, 128),
        ("sparse_indexed_random", 512, 16, 1024, 128, 128, 64),
        ("sparse_indexed_random", 512, 16, 1024, 128, 128, 128),
    ]
    for seed, case in enumerate(sparse_cases, 123):
        _run_sparse_case(*case, seed=seed, warmup=args.warmup, iters=args.iters)

    sdpa_cases = [
        (512, 16, 128, 128, 128),
        (1024, 16, 128, 128, 128),
    ]
    for seed, case in enumerate(sdpa_cases, 1000):
        _run_sdpa_packqkv_case(*case, seed=seed, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
