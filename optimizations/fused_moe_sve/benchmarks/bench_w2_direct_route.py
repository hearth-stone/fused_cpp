#!/usr/bin/env python3
"""Compare SVE W2 FP32/BF16 route storage with direct and scatter stores."""

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
BF16_FLAG = "FUSED_CPP_MOE_W2_BF16_ROUTE"
VARIANTS = (
    ("fp32_scatter", "0", "0"),
    ("fp32_direct", "0", "1"),
    ("bf16_scatter", "1", "0"),
    ("bf16_direct", "1", "1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--path", choices=("normal", "scheduled", "async"), default="async")
    parser.add_argument("--variant", choices=("all", *(item[0] for item in VARIANTS)), default="all")
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


def error_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_f32 = candidate.float().flatten()
    reference_f32 = reference.float().flatten()
    delta = candidate_f32 - reference_f32
    cosine = torch.nn.functional.cosine_similarity(candidate_f32, reference_f32, dim=0).item()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "cosine": max(-1.0, min(1.0, cosine)),
    }


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
    variants = VARIANTS if args.variant == "all" else tuple(item for item in VARIANTS if item[0] == args.variant)
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
            )

    outputs: dict[str, torch.Tensor] = {}
    for name, bf16_flag, direct_flag in VARIANTS:
        os.environ[BF16_FLAG] = bf16_flag
        os.environ[DIRECT_FLAG] = direct_flag
        outputs[name] = run().clone()
    torch.testing.assert_close(outputs["fp32_direct"].float(), outputs["fp32_scatter"].float(), atol=0, rtol=0)
    torch.testing.assert_close(outputs["bf16_direct"].float(), outputs["bf16_scatter"].float(), atol=0, rtol=0)
    precision = {
        name: error_metrics(output, outputs["fp32_direct"])
        for name, output in outputs.items()
    }

    for iteration in range(args.warmup):
        rotate = iteration % len(variants)
        ordered = variants[rotate:] + variants[:rotate]
        for _, bf16_flag, direct_flag in ordered:
            os.environ[BF16_FLAG] = bf16_flag
            os.environ[DIRECT_FLAG] = direct_flag
            run()

    if args.stop_before_runs:
        print(f"PROFILE_READY pid={os.getpid()}", flush=True)
        os.kill(os.getpid(), signal.SIGSTOP)

    samples = {name: [] for name, _, _ in variants}
    minor_faults = {name: [] for name, _, _ in variants}
    major_faults = {name: [] for name, _, _ in variants}
    sink = 0
    for iteration in range(args.runs):
        rotate = iteration % len(variants)
        ordered = variants[rotate:] + variants[:rotate]
        for name, bf16_flag, direct_flag in ordered:
            os.environ[BF16_FLAG] = bf16_flag
            os.environ[DIRECT_FLAG] = direct_flag
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
        for name, bf16_flag, direct_flag in variants:
            trace_path = args.trace_dir / f"{name}.log"
            trace_path.unlink(missing_ok=True)
            os.environ[BF16_FLAG] = bf16_flag
            os.environ[DIRECT_FLAG] = direct_flag
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_path)
            run()
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            trace_results[name] = parse_trace(trace_path)

    fp32_direct_ms = statistics.median(samples["fp32_direct"]) if "fp32_direct" in samples else None
    scatter_names = {"fp32_direct": "fp32_scatter", "bf16_direct": "bf16_scatter"}
    records: list[dict[str, object]] = []
    for name, bf16_flag, direct_flag in variants:
        median_ms = statistics.median(samples[name])
        scatter_name = scatter_names.get(name)
        scatter_ms = statistics.median(samples[scatter_name]) if scatter_name in samples else None
        records.append(
            {
                "variant": name,
                "bf16_route": bf16_flag == "1",
                "direct_route": direct_flag == "1",
                "median_ms": median_ms,
                "p10_ms": percentile(samples[name], 0.10),
                "p90_ms": percentile(samples[name], 0.90),
                "gain_vs_fp32_direct_pct": (
                    None if fp32_direct_ms is None else 100.0 * (fp32_direct_ms / median_ms - 1.0)
                ),
                "gain_vs_same_dtype_scatter_pct": (
                    None if scatter_ms is None else 100.0 * (scatter_ms / median_ms - 1.0)
                ),
                "samples": samples[name],
                "minor_faults": minor_faults[name],
                "major_faults": major_faults[name],
            }
        )

    fp32_route_bytes = args.tokens * args.top_k * args.hidden * 4
    bf16_route_bytes = fp32_route_bytes // 2
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
            "stage_geometry": "full_n_team_stripes",
        },
        "logical_post_w2_bytes": {
            "fp32_scatter": 4 * fp32_route_bytes,
            "fp32_direct": 2 * fp32_route_bytes,
            "bf16_scatter": 4 * bf16_route_bytes,
            "bf16_direct": 2 * bf16_route_bytes,
            "bf16_direct_saved_vs_fp32_direct": fp32_route_bytes,
        },
        "precision_vs_fp32_direct": precision,
        "records": records,
        "trace": trace_results,
        "sink": sink,
    }
    print("variant         median_ms  vs_fp32_direct  vs_scatter     p10_ms     p90_ms  max_minflt")
    for record in records:
        fp32_gain = (
            "n/a"
            if record["gain_vs_fp32_direct_pct"] is None
            else f"{record['gain_vs_fp32_direct_pct']:.2f}"
        )
        scatter_gain = (
            "n/a"
            if record["gain_vs_same_dtype_scatter_pct"] is None
            else f"{record['gain_vs_same_dtype_scatter_pct']:.2f}"
        )
        print(
            f"{record['variant']:<15} {record['median_ms']:>9.3f} "
            f"{fp32_gain:>15} {scatter_gain:>11} {record['p10_ms']:>10.3f} {record['p90_ms']:>10.3f} "
            f"{max(record['minor_faults']):>11}"
        )
    print("precision_vs_fp32_direct")
    print(json.dumps(precision, indent=2))
    if trace_results:
        print(json.dumps(trace_results, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
