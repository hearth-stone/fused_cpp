#!/usr/bin/env python3
"""Compare fixed-owner early merge with post-barrier merge on paper workloads."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
PLANNER_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"
sys.path[:0] = [str(REPO_ROOT / "src"), str(COST_MODEL_DIR), str(PLANNER_DIR)]

from bench_vllm_staged_schedule import materialize_topk_ids  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from phase_model import ContentionCostModel  # noqa: E402
from planned_moe import PlannedMoE  # noqa: E402
from workload_catalog import default_offline_workloads  # noqa: E402


DEFAULT_PROFILE = (
    REPO_ROOT
    / "cpu_moe_schedule_optimization"
    / "cost_model"
    / "profiles"
    / "contention_async_amazon_c5_192c_numa0_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json"
)


def parse_args() -> argparse.Namespace:
    workloads = default_offline_workloads()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--presets", nargs="*", choices=tuple(workloads), default=tuple(workloads))
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=51)
    parser.add_argument(
        "--policy-block-size",
        type=int,
        default=1,
        help="consecutive timed calls per policy before rotating to the next policy",
    )
    parser.add_argument(
        "--policy-switch-warmup",
        type=int,
        default=0,
        help="untimed calls after each policy switch before timing its block",
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--route-dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--tail-redistribution-factorial",
        action="store_true",
        help="run the full early-merge on/off by strict-tail-redistribution on/off matrix",
    )
    parser.add_argument(
        "--fixed-tail-redistribution",
        choices=("off", "on"),
        default="off",
        help="hold strict-tail redistribution fixed while comparing early merge",
    )
    parser.add_argument(
        "--fixed-early-merge",
        choices=("off", "on"),
        help="run one early-merge policy without switching policy in-process",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def configure_runtime(route_dtype: str) -> None:
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_BATCH"] = "2"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_PREFETCH"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "1" if route_dtype == "bf16" else "0"
    os.environ["FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL"] = "1"
    os.environ["FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS"] = "0"
    os.environ["FUSED_CPP_MOE_STAGE_TIMING"] = "0"
    os.environ["FUSED_CPP_MOE_TRACE"] = "0"
    os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "0"
    os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL_DEPTH"] = "2"
    os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL_MIN_DONOR_TASKS"] = "2"


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if args.threads <= 0 or args.runs <= 0 or args.policy_block_size <= 0:
        raise ValueError("threads, runs, and policy block size must be positive")
    if args.warmup < 0 or args.policy_switch_warmup < 0:
        raise ValueError("warmup counts must be non-negative")
    if not args.presets:
        raise ValueError("at least one preset is required")
    if args.tail_redistribution_factorial and args.fixed_tail_redistribution != "off":
        raise ValueError("factorial mode cannot be combined with fixed tail redistribution")
    if args.tail_redistribution_factorial and args.fixed_early_merge is not None:
        raise ValueError("factorial mode cannot be combined with fixed early merge")
    if not args.profile.is_file():
        raise ValueError(f"profile does not exist: {args.profile}")

    affinity = sorted(os.sched_getaffinity(0))
    if args.threads > len(affinity):
        raise ValueError(f"threads={args.threads} exceeds affinity size {len(affinity)}")
    cpu_ids = affinity[: args.threads]
    torch.set_num_threads(1)
    configure_runtime(args.route_dtype)
    if not args.tail_redistribution_factorial:
        os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = (
            "1" if args.fixed_tail_redistribution == "on" else "0"
        )

    workloads = default_offline_workloads()
    selected = [workloads[name] for name in args.presets]
    model = ContentionCostModel(args.profile)
    policy = model.policy
    if policy is None:
        raise ValueError("early-merge workload comparison requires a schema-v2 profile")
    common_shape = {
        (workload.tokens, workload.top_k, workload.num_experts) for workload in selected
    }
    if len(common_shape) != 1:
        raise ValueError("selected workloads must share tokens, top-k, and expert count")
    tokens, top_k, num_experts = common_shape.pop()
    if policy.local_experts != num_experts or policy.cores_per_rank != args.threads:
        raise ValueError(
            "profile mismatch: "
            f"local_experts={policy.local_experts}, cores={policy.cores_per_rank}; "
            f"benchmark requests experts={num_experts}, cores={args.threads}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((tokens, policy.hidden_size), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    w13 = torch.empty(
        (num_experts, 2 * policy.intermediate_size, policy.hidden_size),
        dtype=torch.bfloat16,
    )
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty(
        (num_experts, policy.hidden_size, policy.intermediate_size),
        dtype=torch.bfloat16,
    )
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed = prepare_fused_moe_bf16_tiled_weights(
        w13,
        w2,
        fuse_silu=True,
        backend="arm_sve_bf16",
    )
    del w13, w2
    if packed.gemm_backend != 1:
        raise RuntimeError("benchmark requires the SVE BF16 fused MoE backend")

    planner = PlannedMoE(model, args.threads, cpu_ids=cpu_ids)
    records: list[dict[str, object]] = []
    sink = 0
    for workload_index, workload in enumerate(selected):
        spec = planner.plan_spec_for(workload.experts)
        base_plan = AsyncMoEPlanV2.from_dict(spec["bridge"])
        if base_plan.execution_mode == "elastic":
            raise ValueError("early merge comparison does not support elastic plans")
        merge_plans = {
            "off": replace(base_plan, early_merge=False),
            "on": replace(base_plan, early_merge=True),
        }
        if args.fixed_early_merge is not None:
            tail_policy = args.fixed_tail_redistribution
            variants = {
                f"early_{args.fixed_early_merge}_tail_{tail_policy}": (
                    args.fixed_early_merge,
                    tail_policy == "on",
                )
            }
        elif args.tail_redistribution_factorial:
            variants = {
                "early_off_tail_off": ("off", False),
                "early_on_tail_off": ("on", False),
                "early_off_tail_on": ("off", True),
                "early_on_tail_on": ("on", True),
            }
        elif args.fixed_tail_redistribution == "on":
            variants = {
                "early_off_tail_on": ("off", True),
                "early_on_tail_on": ("on", True),
            }
        else:
            variants = {
                "early_off_tail_off": ("off", False),
                "early_on_tail_off": ("on", False),
            }
        topk_ids = materialize_topk_ids(
            workload.histogram,
            tokens=workload.tokens,
            top_k=workload.top_k,
            seed=args.seed + workload_index,
        )
        topk_weights = torch.softmax(
            torch.randn((workload.tokens, workload.top_k), generator=generator),
            dim=-1,
        )
        outputs = {name: torch.empty_like(hidden) for name in variants}

        def run(name: str) -> torch.Tensor:
            merge_policy, tail_redistribution = variants[name]
            if args.tail_redistribution_factorial:
                os.environ["FUSED_CPP_MOE_STRICT_TAIL_STEAL"] = "1" if tail_redistribution else "0"
            return fused_moe_bf16_tiled_async_plan(
                hidden,
                packed,
                topk_weights,
                topk_ids,
                merge_plans[merge_policy],
                global_num_experts=workload.num_experts,
                out=outputs[name],
            )

        baseline_name = next(iter(variants))
        reference = run(baseline_name).clone()
        for name in variants:
            if name == baseline_name:
                continue
            candidate = run(name).clone()
            torch.testing.assert_close(candidate.float(), reference.float(), atol=0, rtol=0)

        order_rng = random.Random(args.seed ^ (workload_index * 0x9E3779B1))
        for iteration in range(args.warmup):
            order = list(variants)
            order_rng.shuffle(order)
            for name in order:
                run(name)

        samples = {name: [] for name in variants}
        while any(len(values) < args.runs for values in samples.values()):
            order = list(variants)
            order_rng.shuffle(order)
            for name in order:
                if len(samples[name]) >= args.runs:
                    continue
                for _ in range(args.policy_switch_warmup):
                    run(name)
                block_runs = min(args.policy_block_size, args.runs - len(samples[name]))
                for _ in range(block_runs):
                    begin = time.perf_counter_ns()
                    result = run(name)
                    samples[name].append((time.perf_counter_ns() - begin) / 1.0e6)
                    sink ^= int(result.view(torch.int16)[0, 0])

        summaries = {
            name: {
                "median_ms": statistics.median(values),
                "p10_ms": percentile(values, 0.10),
                "p90_ms": percentile(values, 0.90),
                "samples_ms": values,
            }
            for name, values in samples.items()
        }
        reference_name = next(iter(variants))
        reference_ms = float(summaries[reference_name]["median_ms"])
        paired_speedups: list[float] | None = None
        if len(variants) == 2:
            early_off_name, early_on_name = tuple(variants)
            paired_speedups = [
                100.0 * (off_sample / on_sample - 1.0)
                for off_sample, on_sample in zip(
                    samples[early_off_name],
                    samples[early_on_name],
                    strict=True,
                )
            ]
        total_flops = 6 * workload.routes * policy.hidden_size * policy.intermediate_size
        effects: dict[str, float] = {}
        if "early_off_tail_off" in summaries and "early_on_tail_off" in summaries:
            effects["early_merge_without_tail_pct"] = 100.0 * (
                float(summaries["early_off_tail_off"]["median_ms"])
                / float(summaries["early_on_tail_off"]["median_ms"])
                - 1.0
            )
        if "early_off_tail_on" in summaries and "early_on_tail_on" in summaries:
            effects["early_merge_with_tail_pct"] = 100.0 * (
                float(summaries["early_off_tail_on"]["median_ms"])
                / float(summaries["early_on_tail_on"]["median_ms"])
                - 1.0
            )
        if all(
            name in summaries
            for name in (
                "early_off_tail_off",
                "early_on_tail_off",
                "early_off_tail_on",
                "early_on_tail_on",
            )
        ):
            baseline_ms = float(summaries["early_off_tail_off"]["median_ms"])
            early_ms = float(summaries["early_on_tail_off"]["median_ms"])
            tail_ms = float(summaries["early_off_tail_on"]["median_ms"])
            combined_ms = float(summaries["early_on_tail_on"]["median_ms"])
            effects.update(
                {
                    "tail_without_early_merge_pct": 100.0 * (baseline_ms / tail_ms - 1.0),
                    "combined_vs_baseline_pct": 100.0 * (baseline_ms / combined_ms - 1.0),
                    "tail_with_early_merge_pct": 100.0 * (early_ms / combined_ms - 1.0),
                    "multiplicative_interaction_pct": 100.0
                    * (
                        (baseline_ms / combined_ms)
                        / ((baseline_ms / early_ms) * (baseline_ms / tail_ms))
                        - 1.0
                    ),
                }
            )
        best_variant = min(summaries, key=lambda name: float(summaries[name]["median_ms"]))
        best_ms = float(summaries[best_variant]["median_ms"])
        record = {
            "preset": workload.name,
            "active_experts": workload.observed_active_experts,
            "route_min": min(routes for routes in workload.histogram if routes > 0),
            "route_max": max(workload.histogram),
            "execution_mode": base_plan.execution_mode,
            "shape": list(spec["shape"]),
            "tasks": int(base_plan.task_expert_ids.numel()),
            "planner_early_merge": base_plan.early_merge,
            "variants": summaries,
            "effects": effects,
            "best_variant": best_variant,
            "best_vs_reference_pct": 100.0 * (reference_ms / best_ms - 1.0),
            "paired_speedup_median_pct": (
                statistics.median(paired_speedups) if paired_speedups is not None else None
            ),
            "paired_speedup_p10_pct": (
                percentile(paired_speedups, 0.10) if paired_speedups is not None else None
            ),
            "paired_speedup_p90_pct": (
                percentile(paired_speedups, 0.90) if paired_speedups is not None else None
            ),
            "variant_tflops": {
                name: total_flops / float(summary["median_ms"]) / 1.0e9
                for name, summary in summaries.items()
            },
        }
        records.append(record)
        timings = "  ".join(
            f"{name}={float(summary['median_ms']):.3f}"
            for name, summary in summaries.items()
        )
        print(f"{workload.name:32s} {timings}  best={best_variant}")

    effect_names = list(records[0]["effects"])
    aggregate_effects = {
        name: {
            "geomean_speedup_pct": 100.0
            * (
                statistics.geometric_mean(
                    1.0 + float(record["effects"][name]) / 100.0 for record in records
                )
                - 1.0
            ),
            "wins": sum(float(record["effects"][name]) > 0 for record in records),
            "min_speedup_pct": min(float(record["effects"][name]) for record in records),
            "max_speedup_pct": max(float(record["effects"][name]) for record in records),
        }
        for name in effect_names
    }
    result = {
        "schema_version": 1,
        "machine": "AmazonC5192Cores",
        "numa_node": 0,
        "cpu_ids": cpu_ids,
        "profile": str(args.profile),
        "shape": {
            "tokens": tokens,
            "top_k": top_k,
            "experts": num_experts,
            "hidden": policy.hidden_size,
            "intermediate": policy.intermediate_size,
            "threads": args.threads,
            "route_dtype": args.route_dtype,
        },
        "method": {
            "warmup": args.warmup,
            "runs": args.runs,
            "policy_block_size": args.policy_block_size,
            "policy_switch_warmup": args.policy_switch_warmup,
            "block_order_randomized": True,
            "same_plan": True,
            "same_packed_weights": True,
            "strict_tail_steal": False,
            "ready_token_drain": True,
            "ready_token_batch": 2,
            "ready_token_prefetch": True,
            "tail_redistribution_factorial": args.tail_redistribution_factorial,
            "fixed_tail_redistribution": (
                None if args.tail_redistribution_factorial else args.fixed_tail_redistribution
            ),
            "fixed_early_merge": args.fixed_early_merge,
            "strict_tail_steal_depth": 2,
            "strict_tail_steal_min_donor_tasks": 2,
        },
        "summary": {
            "workloads": len(records),
            "effects": aggregate_effects,
            "best_variant_counts": {
                name: sum(record["best_variant"] == name for record in records)
                for name in variants
            },
        },
        "records": records,
        "sink": sink,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
