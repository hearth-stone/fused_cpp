#!/usr/bin/env python3
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights  # noqa: E402


H = 4096
F = 512
EXPERTS = 64
WARMUP = EXPERTS * 4
RUNS = EXPERTS * 3
PACKED_B_READ_CEILING_GBS = 40.04
BFMMLA_PEAK_GFLOPS = 403.8


def bf16(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260723)
    w13 = bf16((EXPERTS, 2 * F, H), generator)
    w2 = bf16((EXPERTS, H, F), generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    del w13, w2

    weight_bytes = 2 * F * H * 2
    print(f"H={H} F={F} experts={EXPERTS} packed_b_read_ceiling={PACKED_B_READ_CEILING_GBS:.2f}GB/s")
    print("ranges M median_ms physical_GF weight_GB/s memory_eff")
    for n_ranges in (1, 2):
        for rows in (1, 2):
            a = bf16((rows, H), generator)
            samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                a,
                packed.w13[0],
                packed.w13[1],
                packed.w13[2],
                packed.backend_n_tile,
                n_ranges,
                WARMUP,
                RUNS,
            )
            median_ms = statistics.median(samples)
            seconds = median_ms / 1e3
            compute_rows = 2 * ((rows + 1) // 2)
            physical_flops = 2 * compute_rows * H * (2 * F)
            physical_gflops = physical_flops / seconds / 1e9
            weight_gbs = weight_bytes / seconds / 1e9
            print(
                f"{n_ranges:6d} {rows:1d} {median_ms:9.4f} {physical_gflops:11.2f} "
                f"{weight_gbs:11.3f} {weight_gbs / PACKED_B_READ_CEILING_GBS * 100:9.2f}%"
            )


if __name__ == "__main__":
    main()
