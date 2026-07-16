#!/usr/bin/env python3
"""Compare the default SVE W2 FP32 direct-route store with its scatter fallback."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
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


DIRECT_FLAG = "FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"
VARIANTS = (("baseline_scatter", "0"), ("direct_route", "1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--path", choices=("normal", "scheduled", "async"), default="async")
    parser.add_argument("--variant", choices=("both", "baseline_scatter", "direct_route"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument(
        "--stop-before-runs",
        action="store_true",
        help="raise SIGSTOP after warmup so an external profiler can attach to the worker pool",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def parse_trace(path: Path) -> dict[str, object]:
    e2e_ms = 0.0
    stage_groups: dict[str, dict[int, list[float]]] = {}
    for line in path.read_text().splitlines():
        fields = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in line.split() if "=" in part}
        if line.startswith("MOE_CALL "):
            e2e_ms = float(fields["e2e_ms"])
        elif line.startswith("PHASE "):
            stage = fields["stage"]
            group = int(fields["group"])
            stage_groups.setdefault(stage, {}).setdefault(group, []).append(float(fields["ms"]))

    stages: dict[str, object] = {}
    for stage, groups in stage_groups.items():
        worker_samples = [sample for samples in groups.values() for sample in samples]
        team_critical = [max(samples) for samples in groups.values()]
        stages[stage] = {
            "records": len(worker_samples),
            "worker_ms_sum": sum(worker_samples),
            "max_worker_ms": max(worker_samples),
            "concurrent_wall_proxy_ms": max(team_critical),
        }
    return {"e2e_ms": e2e_ms, "stages": stages}


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    positive = (args.tokens, args.hidden, args.intermediate, args.experts, args.top_k, args.threads, args.runs)
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("shapes, threads, and runs must be positive; warmup must be non-negative")
    if args.experts < args.top_k:
        raise ValueError("experts must be at least top-k")
    if args.path != "normal" and args.threads % args.experts != 0:
        raise ValueError("scheduled and async paths require threads divisible by experts")
    variants = VARIANTS if args.variant == "both" else tuple(item for item in VARIANTS if item[0] == args.variant)
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
    topk_weights = torch.softmax(torch.randn((args.tokens, args.top_k), generator=generator), dim=-1)

    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_W13_SPLIT_N"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
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
                w13_split=True,
            )

    outputs: dict[str, torch.Tensor] = {}
    for name, flag in VARIANTS:
        os.environ[DIRECT_FLAG] = flag
        outputs[name] = run().clone()
    for name, output in outputs.items():
        torch.testing.assert_close(
            output.float(),
            outputs["baseline_scatter"].float(),
            atol=0,
            rtol=0,
            msg=lambda message, name=name: f"{name}: {message}",
        )

    for iteration in range(args.warmup):
        rotate = iteration % len(variants)
        ordered = variants[rotate:] + variants[:rotate]
        for _, flag in ordered:
            os.environ[DIRECT_FLAG] = flag
            run()

    if args.stop_before_runs:
        print(f"PROFILE_READY pid={os.getpid()}", flush=True)
        os.kill(os.getpid(), signal.SIGSTOP)

    samples = {name: [] for name, _ in variants}
    minor_faults = {name: [] for name, _ in variants}
    major_faults = {name: [] for name, _ in variants}
    sink = 0
    for iteration in range(args.runs):
        rotate = iteration % len(variants)
        ordered = variants[rotate:] + variants[:rotate]
        for name, flag in ordered:
            os.environ[DIRECT_FLAG] = flag
            usage_before = resource.getrusage(resource.RUSAGE_SELF)
            begin = time.perf_counter_ns()
            output = run()
            samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
            usage_after = resource.getrusage(resource.RUSAGE_SELF)
            minor_faults[name].append(usage_after.ru_minflt - usage_before.ru_minflt)
            major_faults[name].append(usage_after.ru_majflt - usage_before.ru_majflt)
            sink ^= int(output.view(torch.int16)[0, 0])

    trace_results: dict[str, object] = {}
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        for name, flag in variants:
            trace_path = args.trace_dir / f"{name}.log"
            trace_path.unlink(missing_ok=True)
            os.environ[DIRECT_FLAG] = flag
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_path)
            run()
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            trace_results[name] = parse_trace(trace_path)

    baseline_ms = statistics.median(samples["baseline_scatter"]) if "baseline_scatter" in samples else None
    records: list[dict[str, object]] = []
    for name, flag in variants:
        median_ms = statistics.median(samples[name])
        records.append(
            {
                "variant": name,
                "direct_route": flag == "1",
                "median_ms": median_ms,
                "p10_ms": percentile(samples[name], 0.10),
                "p90_ms": percentile(samples[name], 0.90),
                "gain_pct": None if baseline_ms is None else 100.0 * (baseline_ms / median_ms - 1.0),
                "samples": samples[name],
                "minor_faults": minor_faults[name],
                "major_faults": major_faults[name],
            }
        )

    route_bytes = args.tokens * args.top_k * args.hidden * 4
    result = {
        "shape": {
            "path": args.path,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "top_k": args.top_k,
            "threads": args.threads,
            "threads_per_expert": None if args.path == "normal" else args.threads // args.experts,
            "w13_split": True,
        },
        "logical_post_w2_bytes": {
            "baseline_scatter": 4 * route_bytes,
            "direct_route": 2 * route_bytes,
            "saved": 2 * route_bytes,
        },
        "records": records,
        "trace": trace_results,
        "sink": sink,
    }
    print("variant             median_ms    gain_pct     p10_ms     p90_ms  max_minflt")
    for record in records:
        gain = "n/a" if record["gain_pct"] is None else f"{record['gain_pct']:.2f}"
        print(
            f"{record['variant']:<19} {record['median_ms']:>9.3f} "
            f"{gain:>10} {record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f} "
            f"{max(record['minor_faults']):>11}"
        )
    if trace_results:
        print(json.dumps(trace_results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
