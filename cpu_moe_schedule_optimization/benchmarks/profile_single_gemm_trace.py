#!/usr/bin/env python3
"""Trace per-stage GEMM latency for one scheduled MoE expert."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def parse_trace(path: Path) -> List[Tuple[float, float, float]]:
    pattern = re.compile(r"GEMM call_id=(\d+).* stage=(\w+) .* ms=([0-9.]+)")
    by_call: Dict[int, Dict[str, List[float]]] = {}
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        call_id = int(match.group(1))
        stage = match.group(2)
        ms = float(match.group(3))
        by_call.setdefault(call_id, {}).setdefault(stage, []).append(ms)

    rows: List[Tuple[float, float, float]] = []
    for call_id in sorted(by_call):
        stages = by_call[call_id]
        if "w13" not in stages or "w2" not in stages:
            continue
        w13_ms = max(stages["w13"])
        w2_ms = max(stages["w2"])
        rows.append((w13_ms, w2_ms, w13_ms + w2_ms))
    return rows


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty measurement list")
    return float(statistics.median(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure single-expert scheduled MoE GEMM stage latency. The GEMM "
            "latency for a stage is max(thread_slice_ms) from FUSED_CPP_MOE_TRACE."
        )
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--routes", default="16,64,256,1024,2048")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--trace-file", type=Path, default=Path("/tmp/moe_single_gemm_trace.log"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routes_list = parse_int_list(args.routes)
    threads_list = parse_int_list(args.threads)
    if args.hidden_size <= 0 or args.ffn_hidden_size <= 0:
        raise ValueError("hidden and FFN sizes must be positive")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs positive")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    w13 = bf16_normal(
        (1, 2 * args.ffn_hidden_size, args.hidden_size),
        generator=generator,
        std=args.std,
    )
    w2 = bf16_normal(
        (1, args.hidden_size, args.ffn_hidden_size),
        generator=generator,
        std=args.std,
    )
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2)

    print(f"shape H={args.hidden_size} F={args.ffn_hidden_size}; stage latency=max(thread_slice_ms)")
    print("routes threads split_w13 split_w2 w13_ms w2_ms gemm_sum_ms full_call_ms speedup_vs_t1")

    baseline_by_routes: Dict[int, float] = {}
    for routes in routes_list:
        hidden = bf16_normal(
            (routes, args.hidden_size),
            generator=generator,
            std=args.std,
        )
        topk_ids = torch.zeros((routes, 1), dtype=torch.int32)
        topk_weights = torch.ones((routes, 1), dtype=torch.float32)
        for threads in threads_list:
            wave_offsets = torch.tensor([0, 1], dtype=torch.int32)
            team_expert_ids = torch.tensor([0], dtype=torch.int32)
            team_threads = torch.tensor([threads], dtype=torch.int32)
            thread_cpu_ids = torch.arange(threads, dtype=torch.int32)

            def run() -> torch.Tensor:
                return fused_moe_bf16_tiled_scheduled(
                    hidden,
                    packed,
                    topk_weights,
                    topk_ids,
                    wave_offsets,
                    team_expert_ids,
                    team_threads,
                    thread_cpu_ids=thread_cpu_ids,
                    num_threads=threads,
                    activation="silu",
                    global_num_experts=1,
                    skip_weighted=True,
                )

            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            for _ in range(args.warmup):
                out = run()
                _ = float(out.flatten()[0])

            full_times_ms: List[float] = []
            for _ in range(args.runs):
                begin = time.perf_counter_ns()
                out = run()
                _ = float(out.flatten()[0])
                full_times_ms.append((time.perf_counter_ns() - begin) / 1e6)

            if args.trace_file.exists():
                args.trace_file.unlink()
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(args.trace_file)
            for _ in range(args.runs):
                out = run()
                _ = float(out.flatten()[0])
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

            trace_rows = parse_trace(args.trace_file)
            w13_ms = median([row[0] for row in trace_rows])
            w2_ms = median([row[1] for row in trace_rows])
            gemm_ms = median([row[2] for row in trace_rows])
            full_ms = median(full_times_ms)

            if threads == 1:
                baseline_by_routes[routes] = gemm_ms
            speedup = baseline_by_routes.get(routes, gemm_ms) / gemm_ms
            split_w13 = "M" if routes > 2 * args.ffn_hidden_size else "N"
            split_w2 = "M" if routes > args.hidden_size else "N"
            print(
                f"{routes:<6} {threads:<7} {split_w13:<8} {split_w2:<7} "
                f"{w13_ms:7.3f} {w2_ms:7.3f} {gemm_ms:11.3f} "
                f"{full_ms:12.3f} {speedup:12.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
