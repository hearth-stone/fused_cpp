#!/usr/bin/env python3
"""Benchmark single-expert SVE fused-pipeline barrier elision variants."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    PreparedBF16TiledFusedMoEWeights,
    fused_moe_bf16_tiled_async,
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


ZERO_FLAG = "FUSED_CPP_MOE_SVE_ELIDE_INTERMEDIATE_ZERO"
OWNER_FLAG = "FUSED_CPP_MOE_SVE_W2_N_OWNER_SCATTER"
VARIANTS = (
    ("baseline", False, False),
    ("no_zero", True, False),
    ("owner_scatter", False, True),
    ("combined", True, True),
)


def parse_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"invalid positive integer list: {text}")
    return values


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def set_variant(elide_zero: bool, owner_scatter: bool) -> None:
    os.environ[ZERO_FLAG] = "1" if elide_zero else "0"
    os.environ[OWNER_FLAG] = "1" if owner_scatter else "0"


def make_run(
    path: str,
    hidden: torch.Tensor,
    packed: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    threads: int,
    weighted: bool,
) -> Callable[[], torch.Tensor]:
    expert_ids = torch.tensor([0], dtype=torch.int32)
    team_threads = torch.tensor([threads], dtype=torch.int32)
    cpu_ids = torch.tensor(sorted(os.sched_getaffinity(0))[:threads], dtype=torch.int32)
    common = {
        "thread_cpu_ids": cpu_ids,
        "num_threads": threads,
        "global_num_experts": 1,
        "skip_weighted": not weighted,
        "activation": "silu",
    }
    if path == "scheduled":
        wave_offsets = torch.tensor([0, 1], dtype=torch.int32)

        def run() -> torch.Tensor:
            return fused_moe_bf16_tiled_scheduled(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                wave_offsets,
                expert_ids,
                team_threads,
                **common,
            )

        return run

    core_begins = torch.tensor([0], dtype=torch.int32)
    dep_offsets = torch.tensor([0, 0], dtype=torch.int32)
    deps = torch.empty(0, dtype=torch.int32)

    def run() -> torch.Tensor:
        return fused_moe_bf16_tiled_async(
            hidden,
            packed,
            topk_weights,
            topk_ids,
            expert_ids,
            core_begins,
            team_threads,
            dep_offsets,
            deps,
            **common,
        )

    return run


@torch.inference_mode()
def benchmark_case(
    *,
    path: str,
    routes: int,
    threads: int,
    hidden: torch.Tensor,
    packed: PreparedBF16TiledFusedMoEWeights,
    warmup: int,
    runs: int,
    sample_ms: float,
    max_inner_iters: int,
    weighted: bool,
) -> list[dict[str, object]]:
    topk_ids = torch.zeros((routes, 1), dtype=torch.int32)
    topk_weights = torch.ones((routes, 1), dtype=torch.float32)
    run = make_run(path, hidden, packed, topk_weights, topk_ids, threads, weighted)

    outputs: dict[str, torch.Tensor] = {}
    for name, elide_zero, owner_scatter in VARIANTS:
        set_variant(elide_zero, owner_scatter)
        output = run()
        outputs[name] = output.clone()
    reference = outputs["baseline"]
    for name, output in outputs.items():
        torch.testing.assert_close(
            output,
            reference,
            atol=0,
            rtol=0,
            msg=lambda message, name=name: f"path={path} routes={routes} threads={threads} variant={name}: {message}",
        )

    for warmup_index in range(warmup):
        order = VARIANTS[warmup_index % len(VARIANTS) :] + VARIANTS[: warmup_index % len(VARIANTS)]
        for _, elide_zero, owner_scatter in order:
            set_variant(elide_zero, owner_scatter)
            run()

    set_variant(False, False)
    calibration_samples = []
    for _ in range(3):
        begin = time.perf_counter_ns()
        run()
        calibration_samples.append((time.perf_counter_ns() - begin) / 1.0e6)
    calibration_ms = statistics.median(calibration_samples)
    inner_iters = min(
        max_inner_iters,
        max(1, math.ceil(sample_ms / max(calibration_ms, 1.0e-9))),
    )

    samples = {name: [] for name, _, _ in VARIANTS}
    sink = 0
    for iteration in range(runs):
        offset = iteration % len(VARIANTS)
        order = VARIANTS[offset:] + VARIANTS[:offset]
        for name, elide_zero, owner_scatter in order:
            set_variant(elide_zero, owner_scatter)
            begin = time.perf_counter_ns()
            for _ in range(inner_iters):
                output = run()
            elapsed_ms = (time.perf_counter_ns() - begin) / 1.0e6 / inner_iters
            samples[name].append(elapsed_ms)
            sink ^= int(output.view(torch.int16)[0, 0])

    baseline_ms = statistics.median(samples["baseline"])
    records: list[dict[str, object]] = []
    for name, elide_zero, owner_scatter in VARIANTS:
        variant_samples = samples[name]
        median_ms = statistics.median(variant_samples)
        stddev_ms = statistics.stdev(variant_samples) if len(variant_samples) > 1 else 0.0
        record = {
            "path": path,
            "routes": routes,
            "threads": threads,
            "weighted": weighted,
            "variant": name,
            "elide_intermediate_zero": elide_zero,
            "w2_n_owner_scatter": owner_scatter,
            "inner_iters": inner_iters,
            "mean_ms": statistics.fmean(variant_samples),
            "stddev_ms": stddev_ms,
            "median_ms": median_ms,
            "p50_ms": median_ms,
            "p10_ms": percentile(variant_samples, 0.10),
            "p90_ms": percentile(variant_samples, 0.90),
            "p99_ms": percentile(variant_samples, 0.99),
            "gain_pct": 100.0 * (baseline_ms / median_ms - 1.0),
            "samples": variant_samples,
            "sink": sink,
        }
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--routes", type=parse_ints, default=parse_ints("12,48,192,768,2040"))
    parser.add_argument("--threads", type=parse_ints, default=parse_ints("8,16,32,48,64,96"))
    parser.add_argument("--paths", default="scheduled,async")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--runs", type=int, default=17)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--max-inner-iters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [path for path in args.paths.split(",") if path]
    if not paths or any(path not in ("scheduled", "async") for path in paths):
        raise ValueError(f"invalid paths: {args.paths}")
    if args.warmup < 0 or args.runs <= 0 or args.sample_ms <= 0 or args.max_inner_iters <= 0:
        raise ValueError("warmup, runs, sample-ms, or max-inner-iters is invalid")
    if max(args.threads) > len(os.sched_getaffinity(0)):
        raise ValueError("thread count exceeds the process CPU affinity")

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if args.weighted else "0"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "0"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty((1, 2 * args.ffn_hidden_size, args.hidden_size), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    w2 = torch.empty((1, args.hidden_size, args.ffn_hidden_size), dtype=torch.bfloat16).normal_(
        mean=0.0, std=0.01, generator=generator
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 backend")

    hidden_by_routes = {
        routes: torch.empty((routes, args.hidden_size), dtype=torch.bfloat16).normal_(
            mean=0.0, std=0.01, generator=generator
        )
        for routes in args.routes
    }
    records: list[dict[str, object]] = []
    print(
        "path       routes threads inner baseline_ms no_zero% owner% combined%",
        flush=True,
    )
    for path in paths:
        for routes in args.routes:
            for threads in args.threads:
                case_records = benchmark_case(
                    path=path,
                    routes=routes,
                    threads=threads,
                    hidden=hidden_by_routes[routes],
                    packed=packed,
                    warmup=args.warmup,
                    runs=args.runs,
                    sample_ms=args.sample_ms,
                    max_inner_iters=args.max_inner_iters,
                    weighted=args.weighted,
                )
                records.extend(case_records)
                by_name = {record["variant"]: record for record in case_records}
                print(
                    f"{path:<10} {routes:>6} {threads:>7} "
                    f"{by_name['baseline']['inner_iters']:>5} "
                    f"{by_name['baseline']['median_ms']:>11.4f} "
                    f"{by_name['no_zero']['gain_pct']:>8.2f} "
                    f"{by_name['owner_scatter']['gain_pct']:>7.2f} "
                    f"{by_name['combined']['gain_pct']:>9.2f}",
                    flush=True,
                )

    payload = {
        "schema_version": 1,
        "host_logical_cores": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "stage_geometry": "full_n_team_stripes",
        "weighted": args.weighted,
        "warmup": args.warmup,
        "runs": args.runs,
        "sample_ms": args.sample_ms,
        "max_inner_iters": args.max_inner_iters,
        "records": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
