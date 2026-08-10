#!/usr/bin/env python3
"""Compare exact-M Xbyak SVE kernels with the static assembly fallback."""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


VARIANT_ENV = {
    "asm": "asm",
    "jit": "jit",
}


def parse_int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or min(result) <= 0:
        raise ValueError(f"expected a non-empty positive integer list, got {value!r}")
    return result


def parse_variant_list(value: str) -> tuple[str, ...]:
    result = tuple(item for item in value.split(",") if item)
    if not result:
        raise ValueError("expected at least one benchmark variant")
    unknown = [variant for variant in result if variant not in VARIANT_ENV]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; expected one of {sorted(VARIANT_ENV)}")
    if len(set(result)) != len(result):
        raise ValueError(f"variants must be unique, got {result}")
    return result


def select_variant(variant: str) -> None:
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = VARIANT_ENV[variant]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", default="1,2,3,4,5,6,7,8,9,10,11,12,192,2040")
    parser.add_argument("--threads", default="1,4,8")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--measurement-experts", type=int, default=8)
    parser.add_argument(
        "--experts-per-wave",
        type=int,
        default=1,
        help="experts run concurrently; --threads remains the thread count per expert",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument(
        "--variants",
        default="asm,jit",
        help="comma-separated subset of asm,jit; first variant is the timing baseline",
    )
    parser.add_argument(
        "--switch-period",
        type=int,
        default=5,
        help="timed calls per implementation before switching; 1 reproduces call-by-call interleaving",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-calls",
        type=int,
        default=0,
        help="after timing, stop the process and run this many calls for an externally attached profiler",
    )
    parser.add_argument(
        "--profile-variant",
        choices=tuple(VARIANT_ENV),
        default="jit",
        help="variant used by --profile-calls",
    )
    return parser.parse_args()


def percentile(samples: list[int], fraction: float) -> int:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def bf16_normal(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(mean=0.0, std=0.01, generator=generator)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    routes_values = parse_int_list(args.routes)
    thread_values = parse_int_list(args.threads)
    variants = parse_variant_list(args.variants)
    affinity = sorted(os.sched_getaffinity(0))
    if min(
        args.hidden,
        args.intermediate,
        args.experts,
        args.measurement_experts,
        args.experts_per_wave,
        args.runs,
        args.switch_period,
    ) <= 0:
        raise ValueError("shape, expert, and run counts must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.profile_calls < 0:
        raise ValueError("profile-calls must be non-negative")
    if args.experts < 2 * args.measurement_experts:
        raise ValueError("experts must provide at least two disjoint measurement windows")
    if args.measurement_experts % args.experts_per_wave != 0:
        raise ValueError("measurement-experts must be divisible by experts-per-wave")
    max_total_threads = max(thread_values) * args.experts_per_wave
    if max_total_threads > len(affinity):
        raise ValueError(f"total threads exceed available affinity: {max_total_threads} > {len(affinity)}")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    prewarm_variant = next((variant for variant in variants if variant != "asm"), "asm")
    select_variant(prewarm_variant)
    generator = torch.Generator().manual_seed(args.seed)
    w13 = bf16_normal((args.experts, 2 * args.intermediate, args.hidden), generator)
    w2 = bf16_normal((args.experts, args.hidden, args.intermediate), generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 backend")
    del w13, w2

    records: list[dict[str, object]] = []
    sink = 0
    for routes in routes_values:
        hidden = bf16_normal((routes * args.measurement_experts, args.hidden), generator)
        weights = torch.ones((routes * args.measurement_experts, 1), dtype=torch.float32)
        wave_offsets = torch.arange(
            0,
            args.measurement_experts + 1,
            args.experts_per_wave,
            dtype=torch.int32,
        )

        for threads in thread_values:
            total_threads = threads * args.experts_per_wave
            thread_cpu_ids = torch.tensor(affinity[:total_threads], dtype=torch.int32)
            team_threads = torch.full((args.measurement_experts,), threads, dtype=torch.int32)
            calls = []
            for window in range(2):
                first_expert = window * args.measurement_experts
                expert_ids = torch.arange(
                    first_expert,
                    first_expert + args.measurement_experts,
                    dtype=torch.int32,
                )
                topk_ids = torch.repeat_interleave(expert_ids, routes).reshape(-1, 1)

                def run(
                    expert_ids: torch.Tensor = expert_ids,
                    topk_ids: torch.Tensor = topk_ids,
                ) -> torch.Tensor:
                    return fused_moe_bf16_tiled_scheduled(
                        hidden,
                        packed,
                        weights,
                        topk_ids,
                        wave_offsets,
                        expert_ids,
                        team_threads,
                        thread_cpu_ids=thread_cpu_ids,
                        num_threads=total_threads,
                        global_num_experts=args.experts,
                        skip_weighted=True,
                    )

                calls.append(run)

            select_variant("asm")
            reference = calls[0]()
            for variant in variants:
                if variant == "asm":
                    continue
                select_variant(variant)
                candidate = calls[0]()
                torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

            for variant in variants:
                select_variant(variant)
                for iteration in range(args.warmup):
                    calls[iteration % len(calls)]()

            samples: dict[str, list[int]] = {variant: [] for variant in variants}
            block = 0
            while min(len(values) for values in samples.values()) < args.runs:
                order = variants if block % 2 == 0 else tuple(reversed(variants))
                for variant in order:
                    select_variant(variant)
                    # The production process does not swap implementations on
                    # every call. Keep the transition call out of the samples
                    # so instruction-cache replacement is not charged to JIT.
                    calls[block % len(calls)]()
                    remaining = args.runs - len(samples[variant])
                    for iteration in range(min(args.switch_period, remaining)):
                        # Every variant sees the same weight-window sequence in
                        # a block; otherwise odd sample counts bias a bimodal
                        # pair of disjoint expert windows toward one variant.
                        call = calls[(block * args.switch_period + iteration) % len(calls)]
                        begin = time.perf_counter_ns()
                        output = call()
                        elapsed = time.perf_counter_ns() - begin
                        samples[variant].append(elapsed)
                        sink ^= int(output.view(torch.int16)[0, 0])
                block += 1

            medians = {variant: statistics.median(values) for variant, values in samples.items()}
            baseline = variants[0]
            gains = {variant: 100.0 * (medians[baseline] / median - 1.0) for variant, median in medians.items()}
            record = {
                "routes": routes,
                "threads": threads,
                "total_threads": total_threads,
                "experts_per_wave": args.experts_per_wave,
                "measurement_experts": args.measurement_experts,
                "baseline_variant": baseline,
                "median_ns": {variant: int(value) for variant, value in medians.items()},
                "gain_pct": gains,
                "p10_ns": {variant: percentile(values, 0.10) for variant, values in samples.items()},
                "p90_ns": {variant: percentile(values, 0.90) for variant, values in samples.items()},
                "samples_ns": samples,
            }
            if "asm" in medians and "jit" in medians:
                record.update(
                    {
                        "asm_median_ns": int(medians["asm"]),
                        "jit_median_ns": int(medians["jit"]),
                        "jit_gain_pct": gains["jit"],
                        "asm_p10_ns": percentile(samples["asm"], 0.10),
                        "asm_p90_ns": percentile(samples["asm"], 0.90),
                        "jit_p10_ns": percentile(samples["jit"], 0.10),
                        "jit_p90_ns": percentile(samples["jit"], 0.90),
                        "asm_samples_ns": samples["asm"],
                        "jit_samples_ns": samples["jit"],
                    }
                )
            records.append(record)
            timings = " ".join(f"{variant}={medians[variant] / 1e6:8.3f} ms" for variant in variants)
            relative = " ".join(f"{variant}={gains[variant]:+6.2f}%" for variant in variants[1:])
            print(
                f"M={routes:<4} E/wave={args.experts_per_wave:<3} "
                f"T/expert={threads:<3} total_T={total_threads:<3} "
                f"{timings} gains[{baseline}] {relative}"
            )

            if args.profile_calls:
                if len(routes_values) != 1 or len(thread_values) != 1:
                    raise ValueError("profile-calls requires exactly one route and thread point")
                select_variant(args.profile_variant)
                print(
                    f"profile_ready variant={args.profile_variant} calls={args.profile_calls}",
                    flush=True,
                )
                os.kill(os.getpid(), signal.SIGSTOP)
                for iteration in range(args.profile_calls):
                    output = calls[iteration % len(calls)]()
                    sink ^= int(output.view(torch.int16)[0, 0])

    payload = {
        "schema_version": 1,
        "kind": "sve_xbyak_exact_m_ab",
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "measurement_experts": args.measurement_experts,
            "experts_per_wave": args.experts_per_wave,
        },
        "affinity": affinity,
        "warmup": args.warmup,
        "runs": args.runs,
        "switch_period": args.switch_period,
        "variants": variants,
        "records": records,
        "sink": sink,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
