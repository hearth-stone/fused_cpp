# -*- coding: utf-8 -*-
"""Benchmark DeepSeek V4 serial fused attn GEMM with pre-packed weights."""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from fused_cpp.deepseek_v4_attn_gemm_fused import (
    deepseek_v4_attn_gemm_fused_prepacked,
    prepare_deepseek_v4_attn_gemm_weights,
)


def _bf16_randn(rows: int, cols: int, scale: float = 0.1) -> torch.Tensor:
    return (torch.randn(rows, cols, dtype=torch.float32) * scale).to(torch.bfloat16)


def _gflops(m: int, k: int, n: int, seconds: float) -> float:
    return (2.0 * m * k * n) / seconds / 1.0e9


def _parse_cores(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    cores: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid core range: {part}")
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    if any(core < 0 for core in cores):
        raise ValueError(f"core ids must be non-negative: {cores}")
    return cores


def _set_process_affinity(core_ids: list[int]) -> None:
    if core_ids and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(core_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n-fused-wqa-wkv", type=int, default=1536)
    parser.add_argument("--n-compressor-kv-score", type=int, default=2048)
    parser.add_argument("--n-indexer-compressor-kv-score", type=int, default=256)
    parser.add_argument("--n-indexer-weights-proj", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("fused", "torch-mm"),
        default="fused",
        help="fused uses the i8mm fused path; torch-mm runs the 4 GEMMs serially.",
    )
    parser.add_argument(
        "--cores",
        type=str,
        default="",
        help="OpenMP affinity list, e.g. '80-159' or '80,81,82'.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--peak-gflops",
        type=float,
        default=92.0,
        help="Reference single-core peak used for efficiency reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core_ids = _parse_cores(args.cores)
    if args.backend == "torch-mm":
        _set_process_affinity(core_ids)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    hidden_states = _bf16_randn(args.m, args.k).contiguous()
    fused_wqa_wkv = _bf16_randn(args.k, args.n_fused_wqa_wkv).contiguous()
    compressor_kv_score = _bf16_randn(args.k, args.n_compressor_kv_score).contiguous()
    indexer_compressor_kv_score = _bf16_randn(args.k, args.n_indexer_compressor_kv_score).contiguous()
    indexer_weights_proj = _bf16_randn(args.k, args.n_indexer_weights_proj).contiguous()

    if args.backend == "fused":
        pack_t0 = time.perf_counter()
        prepared_weights = prepare_deepseek_v4_attn_gemm_weights(
            fused_wqa_wkv,
            compressor_kv_score,
            indexer_compressor_kv_score,
            indexer_weights_proj,
        )
        pack_t1 = time.perf_counter()
    else:
        prepared_weights = None
        pack_t0 = pack_t1 = time.perf_counter()

    def run_fused() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert prepared_weights is not None
        return deepseek_v4_attn_gemm_fused_prepacked(
            hidden_states,
            prepared_weights,
            cores=core_ids,
        )

    def run_torch_mm() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qr_kv = torch.mm(hidden_states, fused_wqa_wkv)
        kv_score = torch.mm(hidden_states, compressor_kv_score).to(torch.float32)
        indexer_weights = torch.mm(hidden_states, indexer_weights_proj)
        indexer_kv_score = torch.mm(
            hidden_states,
            indexer_compressor_kv_score,
        ).to(torch.float32)
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    run = run_fused if args.backend == "fused" else run_torch_mm

    for _ in range(args.warmup):
        run()

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outputs = run()
        # Touch the outputs so the call cannot be optimized away by future wrappers.
        _ = float(outputs[0].flatten()[0])
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_s = statistics.median(times)
    best_s = min(times)
    mean_s = statistics.mean(times)

    gemms = [
        ("fused_wqa_wkv", args.n_fused_wqa_wkv, "bf16"),
        ("compressor_kv_score", args.n_compressor_kv_score, "fp32"),
        (
            "indexer_compressor_kv_score",
            args.n_indexer_compressor_kv_score,
            "fp32",
        ),
        ("indexer_weights_proj", args.n_indexer_weights_proj, "bf16"),
    ]
    total_flops = 0.0

    print(f"hidden_states=[{args.m},{args.k}]")
    print(f"backend={args.backend}")
    print(f"torch_num_threads={torch.get_num_threads()}")
    print(f"core_ids={core_ids if core_ids else 'serial'}")
    if args.backend == "fused":
        print(f"pack_ms={(pack_t1 - pack_t0) * 1e3:.3f}  # not included below")
    else:
        print("pack_ms=0.000  # not applicable")
    print("")
    for name, n, dtype in gemms:
        flops = 2.0 * args.m * args.k * n
        total_flops += flops
        print(
            f"{name:28s} weight=[{args.k},{n}] out={dtype:4s} "
            f"flops={flops / 1e9:8.3f} GF "
            f"median_contrib={flops / median_s / 1e9:8.3f} GFLOP/s"
        )

    median_gflops = total_flops / median_s / 1.0e9
    best_gflops = total_flops / best_s / 1.0e9
    peak_scale = max(1, len(core_ids))
    peak_ref = args.peak_gflops * peak_scale
    print("")
    print(f"total_flops={total_flops / 1e9:.3f} GF")
    print(f"median_ms={median_s * 1e3:.3f}")
    print(f"best_ms={best_s * 1e3:.3f}")
    print(f"mean_ms={mean_s * 1e3:.3f}")
    print(f"median_gflops={median_gflops:.3f}")
    print(f"best_gflops={best_gflops:.3f}")
    print(f"peak_ref_gflops={peak_ref:.3f}")
    print(f"median_eff_vs_peak={median_gflops / peak_ref * 100.0:.2f}%")
    print(f"times_ms={[round(t * 1e3, 3) for t in times]}")


if __name__ == "__main__":
    main()
