#!/usr/bin/env python3
"""Measure whether a W13 window changes downstream W2 stage latency."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "benchmarks"))
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"))

from bench_analytic_stage_window_tiles import build_plan  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from profile_moe_stage_breakdown import parse_trace, summarize_trace_rows  # noqa: E402
from profile_analytic_services import parse_cpu_ids  # noqa: E402


def parse_pairs(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        left, right = item.split("/", 1)
        result.append((int(left), int(right)))
    if not result or any(min(pair) < 0 for pair in result):
        raise argparse.ArgumentTypeError("pairs must be comma-separated W13/W2 non-negative tile counts")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/stage_window_handoff_traces"))
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=parse_cpu_ids("0-95"))
    parser.add_argument("--routes", type=int, default=216)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--pairs", type=parse_pairs, required=True)
    parser.add_argument("--experts", type=int, default=96)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--trace-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if min(args.routes, args.threads, args.experts, args.hidden_size, args.intermediate_size, args.runs) <= 0:
        raise ValueError("shape and run arguments must be positive")
    if len(args.cpu_ids) % args.threads or args.experts % (len(args.cpu_ids) // args.threads):
        raise ValueError("CPU count and experts must form complete fixed-width waves")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    generator = torch.Generator().manual_seed(args.seed)
    w13 = torch.empty(
        (args.experts, 2 * args.intermediate_size, args.hidden_size), dtype=torch.bfloat16
    ).normal_(0.0, 0.01, generator=generator)
    w2 = torch.empty(
        (args.experts, args.hidden_size, args.intermediate_size), dtype=torch.bfloat16
    ).normal_(0.0, 0.01, generator=generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    del w13, w2

    rows = args.experts * args.routes
    hidden = torch.empty((rows, args.hidden_size), dtype=torch.bfloat16).normal_(
        0.0, 0.01, generator=generator
    )
    topk_ids = torch.arange(args.experts, dtype=torch.int32).repeat_interleave(args.routes).view(-1, 1)
    topk_weights = torch.ones((rows, 1), dtype=torch.float32)
    output = torch.empty_like(hidden)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for point_index, (w13_tiles, w2_tiles) in enumerate(args.pairs):
        plan = build_plan(
            experts=args.experts,
            threads=args.threads,
            cpu_ids=args.cpu_ids,
            w13_window_tiles=w13_tiles,
            w2_window_tiles=w2_tiles,
            start_lane=point_index,
        )

        def run():
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                plan,
                activation="silu",
                global_num_experts=args.experts,
                skip_weighted=True,
                out=output,
            )

        os.environ["FUSED_CPP_MOE_TRACE"] = "0"
        for _ in range(args.warmup):
            _ = int(run().view(torch.int16).flatten()[0])
        samples_ms = []
        for _ in range(args.runs):
            begin = time.perf_counter_ns()
            _ = int(run().view(torch.int16).flatten()[0])
            samples_ms.append((time.perf_counter_ns() - begin) / 1e6)

        trace_file = args.trace_dir / f"m{args.routes}_t{args.threads}_w{w13_tiles}_{w2_tiles}.log"
        trace_file.unlink(missing_ok=True)
        os.environ["FUSED_CPP_MOE_TRACE"] = "1"
        os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_file)
        for _ in range(args.trace_runs):
            _ = int(run().view(torch.int16).flatten()[0])
        os.environ["FUSED_CPP_MOE_TRACE"] = "0"
        trace_rows = parse_trace(trace_file)
        if len(trace_rows) != args.trace_runs:
            raise RuntimeError(f"expected {args.trace_runs} traced calls, found {len(trace_rows)}")
        summary = summarize_trace_rows(trace_rows)
        entry = {
            "w13_window_tiles": w13_tiles,
            "w2_window_tiles": w2_tiles,
            "median_ms": statistics.median(samples_ms),
            "p10_ms": sorted(samples_ms)[round((args.runs - 1) * 0.10)],
            "p90_ms": sorted(samples_ms)[round((args.runs - 1) * 0.90)],
            "samples_ms": samples_ms,
            "trace": summary,
            "trace_file": str(trace_file),
        }
        entries.append(entry)
        print(
            f"w={w13_tiles}/{w2_tiles} e2e={entry['median_ms']:.3f} ms "
            f"trace_w13={summary['w13_ms']:.3f} ms trace_w2={summary['w2_ms']:.3f} ms "
            f"compute={summary['scheduled_compute_ms']:.3f} ms",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "kind": "sve_stage_window_handoff_probe",
        "config": {
            "routes": args.routes,
            "threads": args.threads,
            "experts": args.experts,
            "cpu_ids": args.cpu_ids,
            "pairs": args.pairs,
            "warmup": args.warmup,
            "runs": args.runs,
            "trace_runs": args.trace_runs,
        },
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
