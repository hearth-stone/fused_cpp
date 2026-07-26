#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights  # noqa: E402


BFMMLA_PEAK_GFLOPS = 403.8
PROBE_MODES = {
    "baseline": 0,
    "b-only": 1,
    "ba-only": 2,
    "bfmmla-only": 3,
    "full-no-store": 4,
    "full-with-store": 5,
    "a-only": 6,
    "control-only": 7,
    "ba-fixed-a": 8,
    "full-no-store-fixed-a": 9,
}


def parse_int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or min(result) <= 0:
        raise argparse.ArgumentTypeError(f"expected positive integers, got {value!r}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the exact-M SVE JIT GEMM body with rotating cold B.")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--rows", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--n-ranges", type=parse_int_list, default=parse_int_list("1,2"))
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--runs", type=int, default=192)
    parser.add_argument("--packed-b-read-ceiling-gbs", type=float, default=40.04)
    parser.add_argument("--probe-mode", choices=tuple(PROBE_MODES), default="baseline")
    parser.add_argument("--stop-before-run", action="store_true")
    parser.add_argument(
        "--profile-window",
        action="store_true",
        help="SIGSTOP after native setup and again after all kernel calls",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def bf16(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(args.k, args.n, args.experts, args.runs) <= 0 or args.warmup < 0:
        raise ValueError("K, N, experts, and runs must be positive; warmup must be non-negative")
    if args.k % 8 != 0 or args.n % 2 != 0:
        raise ValueError("K must be divisible by 8 and N must be even")
    if any(args.n % n_ranges != 0 for n_ranges in args.n_ranges):
        raise ValueError("every n-ranges value must divide N")
    if args.stop_before_run and args.profile_window:
        raise ValueError("--stop-before-run and --profile-window are mutually exclusive")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260723)
    intermediate = args.n // 2
    w13 = bf16((args.experts, args.n, args.k), generator)
    w2 = bf16((args.experts, args.k, intermediate), generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    del w13, w2

    if any((args.n // n_ranges) % packed.backend_n_tile != 0 for n_ranges in args.n_ranges):
        raise ValueError(f"every N range must be divisible by n_tile={packed.backend_n_tile}")

    weight_bytes = args.k * args.n * 2
    print(
        f"K={args.k} N={args.n} experts={args.experts} "
        f"packed_b_read_ceiling={args.packed_b_read_ceiling_gbs:.2f}GB/s"
    )
    print("ranges M median_ms physical_GF weight_GB/s memory_eff")
    records: list[dict[str, float | int]] = []
    stopped = False
    for n_ranges in args.n_ranges:
        for rows in args.rows:
            a = bf16((rows, args.k), generator)
            stop_after_setup = (args.stop_before_run or args.profile_window) and not stopped
            if stop_after_setup:
                print("profile_ready", flush=True)
                profile_env = (
                    "FUSED_CPP_MOE_BENCH_PROFILE_WINDOW"
                    if args.profile_window
                    else "FUSED_CPP_MOE_BENCH_STOP_AFTER_SETUP"
                )
                os.environ[profile_env] = "1"
            try:
                samples = _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                    a,
                    packed.w13[0],
                    packed.w13[1],
                    packed.w13[2],
                    packed.backend_n_tile,
                    n_ranges,
                    args.warmup,
                    args.runs,
                    PROBE_MODES[args.probe_mode],
                )
            finally:
                if stop_after_setup:
                    os.environ.pop(profile_env, None)
                    stopped = True
            median_ms = statistics.median(samples)
            seconds = median_ms / 1e3
            compute_rows = 2 * ((rows + 1) // 2)
            physical_flops = 2 * compute_rows * args.k * args.n
            physical_gflops = physical_flops / seconds / 1e9
            weight_gbs = weight_bytes / seconds / 1e9
            record: dict[str, float | int] = {
                "k": args.k,
                "n": args.n,
                "n_ranges": n_ranges,
                "rows": rows,
                "median_ms": median_ms,
                "physical_gflops": physical_gflops,
                "weight_gbs": weight_gbs,
                "memory_efficiency": weight_gbs / args.packed_b_read_ceiling_gbs,
                "probe_mode": args.probe_mode,
            }
            records.append(record)
            print(
                f"{n_ranges:6d} {rows:1d} {median_ms:9.4f} {physical_gflops:11.2f} "
                f"{weight_gbs:11.3f} {weight_gbs / args.packed_b_read_ceiling_gbs * 100:9.2f}%"
            )

    if args.output_json is not None:
        payload = {
            "schema_version": 1,
            "kind": "sve_jit_small_m_cold_b_gemm",
            "config": {
                "k": args.k,
                "n": args.n,
                "experts": args.experts,
                "warmup": args.warmup,
                "runs": args.runs,
                "weight_bytes": weight_bytes,
                "packed_b_read_ceiling_gbs": args.packed_b_read_ceiling_gbs,
                "probe_mode": args.probe_mode,
            },
            "records": records,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
