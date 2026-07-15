# -*- coding: utf-8 -*-
"""Standalone correctness and performance checks for dense MQA attention."""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from dataclasses import dataclass

import torch

from fused_cpp.mqa import multi_query_attention, multi_query_attention_torch
from fused_cpp.sdpa_flops import compute_causal_effective_pairs, gflops


@dataclass(frozen=True)
class Shape:
    B: int
    N: int
    L: int
    S: int
    E: int
    Ev: int

    @classmethod
    def parse(cls, text: str) -> "Shape":
        parts = [int(x) for x in text.split(",")]
        if len(parts) != 6:
            raise argparse.ArgumentTypeError("shape must be B,N,L,S,E,Ev")
        return cls(*parts)

    def label(self) -> str:
        return f"B={self.B} N={self.N} L={self.L} S={self.S} E={self.E} Ev={self.Ev}"


DEFAULT_SHAPES = [
    Shape(1, 16, 128, 128, 192, 128),
    Shape(1, 16, 512, 512, 192, 128),
    Shape(1, 32, 256, 256, 576, 512),
]


def _make_inputs(shape: Shape, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(shape.B, shape.N, shape.L, shape.E, generator=g, dtype=torch.float32).to(dtype)
    k = torch.randn(shape.B, shape.S, shape.E, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(shape.B, shape.S, shape.Ev, generator=g, dtype=torch.float32).to(dtype)
    return q, k, v


def _flops(shape: Shape, is_causal: bool) -> int:
    pairs = compute_causal_effective_pairs(shape.L, shape.S) if is_causal else shape.L * shape.S
    qk = 2 * shape.B * shape.N * pairs * shape.E
    softmax = 5 * shape.B * shape.N * pairs
    pv = 2 * shape.B * shape.N * pairs * shape.Ev
    return qk + softmax + pv


def _bench(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    gc.collect()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": samples[0],
        "max": samples[-1],
    }


def _dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise argparse.ArgumentTypeError("dtype must be fp32 or bf16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--shape", type=Shape.parse, action="append")
    args = parser.parse_args()

    shapes = args.shape or DEFAULT_SHAPES
    print(f"env OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} torch_num_threads={torch.get_num_threads()}")
    print(f"dtype={args.dtype} causal={args.causal} warmup={args.warmup} iters={args.iters}")

    for shape in shapes:
        q, k, v = _make_inputs(shape, args.dtype, args.seed)
        actual = multi_query_attention(q, k, v, is_causal=args.causal)
        expected = multi_query_attention_torch(q, k, v, is_causal=args.causal)
        max_abs = float((actual.float() - expected.float()).abs().max().item())
        denom = expected.float().abs().clamp(min=1e-12)
        max_rel = float(((actual.float() - expected.float()).abs() / denom).max().item())

        custom = _bench(
            lambda: multi_query_attention(q, k, v, is_causal=args.causal),
            args.warmup,
            args.iters,
        )
        torch_ref = _bench(
            lambda: multi_query_attention_torch(q, k, v, is_causal=args.causal),
            args.warmup,
            args.iters,
        )
        total_flops = _flops(shape, args.causal)
        print(
            f"{shape.label()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"custom_mean_ms={custom['mean'] * 1e3:.3f} "
            f"custom_gflops={gflops(total_flops, custom['mean']):.3f} "
            f"torch_mean_ms={torch_ref['mean'] * 1e3:.3f} "
            f"torch_gflops={gflops(total_flops, torch_ref['mean']):.3f} "
            f"speedup={torch_ref['mean'] / custom['mean']:.3f}x"
        )


if __name__ == "__main__":
    main()
