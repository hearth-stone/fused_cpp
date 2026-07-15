# -*- coding: utf-8 -*-
"""Benchmark: N-split single-expert multithreading for the fused SiLU path.

Regime = few active experts, a group of G threads cooperating over one expert
(hierarchical N-split). Compares:
  - nsplit_fused   : fuse_silu weights + N-split enabled
  - nsplit_legacy  : fuse_silu=False (w13 GEMM + separate activation) + N-split
  - default_fused  : fuse_silu weights, N-split OFF (1 thread/expert)

To force single-expert teaming we use E == group_size active experts with one
group covering each; set total_groups so group_size = num_threads / total_groups.

Run pinned:
  OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
    python cpu_moe_schedule_optimization/benchmarks/bench_fused_silu_nsplit.py
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch

from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

_KEYS = (
    "FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT",
    "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
    "FUSED_CPP_MOE_N_SPLIT_CORE_BASES",
)


@contextlib.contextmanager
def nsplit_env(enabled, groups_per_partition=1, core_bases="0"):
    saved = {k: os.environ.get(k) for k in _KEYS}
    if enabled:
        os.environ["FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT"] = "1"
        os.environ["FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION"] = str(groups_per_partition)
        os.environ["FUSED_CPP_MOE_N_SPLIT_CORE_BASES"] = core_bases
    else:
        os.environ.pop("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--ffn-hidden-size", type=int, default=512)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--rows-per-expert", default="1,2,8,128,512,1024")
    ap.add_argument("--group-sizes", default="2,4,8")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H, F, T = args.hidden_size, args.ffn_hidden_size, args.num_threads
    torch.manual_seed(args.seed)

    row_list = [int(x) for x in args.rows_per_expert.split(",")]
    group_list = [int(x) for x in args.group_sizes.split(",")]

    print(f"H={H} F={F} num_threads={T}; median ms; one group of G threads/expert")
    print("E=G active experts, each with `rows` tokens (top_k=1, balanced).")
    print("rows    G   nsplit_fused  nsplit_legacy  default_fused  fused/legacy")
    for G in group_list:
        if T % G != 0:
            continue
        total_groups = T // G  # each group has G threads
        E = total_groups  # one active expert per group
        w13 = _bf16(E, 2 * F, H)
        w2 = _bf16(E, H, F)
        base_w = prepare_fused_moe_bf16_tiled_weights(w13, w2)
        fused_w = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
        for rows in row_list:
            tokens = E * rows
            x = _bf16(tokens, H)
            ids = torch.tensor([[i % E] for i in range(tokens)], dtype=torch.int32)
            w = torch.ones(tokens, 1)

            def run(weights, enabled, degree=5):
                with nsplit_env(
                    enabled,
                    groups_per_partition=total_groups,
                    core_bases="0",
                ):
                    return fused_moe_bf16_tiled(
                        x,
                        weights,
                        w,
                        ids,
                        num_threads=T,
                        activation="silu",
                        silu_poly_degree=degree,
                    )

            nf = _time(lambda: run(fused_w, True), args.warmup, args.runs)
            nl = _time(lambda: run(base_w, True), args.warmup, args.runs)
            df = _time(lambda: run(fused_w, False), args.warmup, args.runs)
            print(f"{rows:5d} {G:3d}  {nf:12.3f}  {nl:13.3f}  {df:13.3f}  {nl / nf:12.3f}")


if __name__ == "__main__":
    main()
