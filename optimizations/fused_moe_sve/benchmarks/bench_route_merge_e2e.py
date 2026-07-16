#!/usr/bin/env python3
"""Benchmark experimental SVE route-merge variants in fused MoE."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled,
    fused_moe_bf16_tiled_async,
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


MERGE_FLAG = "FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"
VARIANTS = (("baseline", 0), ("sve_u1", 1), ("sve_u2", 2), ("sve_u4", 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--path", choices=("normal", "scheduled", "async"), default="async")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--bf16-route", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.experts, args.top_k, args.threads, args.runs) <= 0:
        raise ValueError("shape, thread, and run arguments must be positive")
    if args.warmup < 0 or args.experts < args.top_k:
        raise ValueError("this benchmark requires warmup >= 0 and experts >= top-k")
    if args.path != "normal" and args.threads % args.experts != 0:
        raise ValueError("scheduled and async paths require threads divisible by experts")
    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    hidden_states = torch.empty((args.tokens, args.hidden), dtype=torch.bfloat16)
    hidden_states.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty((args.experts, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((args.experts, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    topk_ids = torch.tensor(
        [[(token + slot) % args.experts for slot in range(args.top_k)] for token in range(args.tokens)],
        dtype=torch.int32,
    )
    logits = torch.randn((args.tokens, args.top_k), generator=generator)
    topk_weights = torch.softmax(logits, dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if args.bf16_route else "0"
    os.environ["FUSED_CPP_MOE_PIN_THREADS"] = "1"
    os.environ["FUSED_CPP_MOE_PIN_THREAD_CPUS"] = ",".join(str(cpu) for cpu in affinity[: args.threads])
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 fused MoE backend")
    del w13, w2

    thread_cpu_ids = torch.tensor(affinity[: args.threads], dtype=torch.int32)
    if args.path == "normal":

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                num_threads=args.threads,
                activation="silu",
            )

    elif args.path == "scheduled":
        team_threads = args.threads // args.experts
        expert_ids = torch.arange(args.experts, dtype=torch.int32)
        team_widths = torch.full((args.experts,), team_threads, dtype=torch.int32)
        wave_offsets = torch.tensor([0, args.experts], dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_scheduled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                wave_offsets,
                expert_ids,
                team_widths,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=args.threads,
                global_num_experts=args.experts,
                activation="silu",
            )

    else:
        team_threads = args.threads // args.experts
        expert_ids = torch.arange(args.experts, dtype=torch.int32)
        core_begins = torch.arange(args.experts, dtype=torch.int32) * team_threads
        team_widths = torch.full((args.experts,), team_threads, dtype=torch.int32)
        dep_offsets = torch.zeros(args.experts + 1, dtype=torch.int32)
        deps = torch.empty(0, dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_async(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                expert_ids,
                core_begins,
                team_widths,
                dep_offsets,
                deps,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=args.threads,
                global_num_experts=args.experts,
                activation="silu",
                w13_split=True,
            )

    outputs: dict[str, torch.Tensor] = {}
    for name, unroll in VARIANTS:
        os.environ[MERGE_FLAG] = str(unroll)
        outputs[name] = run().clone()
    reference = outputs["baseline"].float()
    errors: dict[str, dict[str, float]] = {}
    for name, output in outputs.items():
        delta = (output.float() - reference).abs()
        errors[name] = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
        }
        torch.testing.assert_close(output.float(), reference, atol=2.0e-3, rtol=2.0e-2)

    for iteration in range(args.warmup):
        offset = iteration % len(VARIANTS)
        for _, unroll in VARIANTS[offset:] + VARIANTS[:offset]:
            os.environ[MERGE_FLAG] = str(unroll)
            run()

    samples = {name: [] for name, _ in VARIANTS}
    sink = 0
    for iteration in range(args.runs):
        offset = iteration % len(VARIANTS)
        for name, unroll in VARIANTS[offset:] + VARIANTS[:offset]:
            os.environ[MERGE_FLAG] = str(unroll)
            begin = time.perf_counter_ns()
            output = run()
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            sink ^= int(output.view(torch.int16)[0, 0])

    baseline_ms = statistics.median(samples["baseline"])
    records: list[dict[str, object]] = []
    for name, unroll in VARIANTS:
        variant_samples = samples[name]
        median_ms = statistics.median(variant_samples)
        record: dict[str, object] = {
            "variant": name,
            "unroll": unroll,
            "median_ms": median_ms,
            "p10_ms": percentile(variant_samples, 0.10),
            "p90_ms": percentile(variant_samples, 0.90),
            "gain_pct": 100.0 * (baseline_ms / median_ms - 1.0),
            "max_abs": errors[name]["max_abs"],
            "mean_abs": errors[name]["mean_abs"],
            "samples": variant_samples,
        }
        records.append(record)

    result = {
        "shape": {
            "path": args.path,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "top_k": args.top_k,
            "threads": args.threads,
            "route_dtype": "bf16" if args.bf16_route else "fp32",
            "threads_per_expert": None if args.path == "normal" else args.threads // args.experts,
        },
        "records": records,
        "sink": sink,
    }
    print("variant      median_ms    gain_pct    max_abs")
    for record in records:
        print(
            f"{record['variant']:<12} {record['median_ms']:>9.3f} "
            f"{record['gain_pct']:>10.2f} {record['max_abs']:>10.6f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
