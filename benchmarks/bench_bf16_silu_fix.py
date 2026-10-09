"""Compare delivery builds with fixed inputs, CPU placement and a shared profile.

Run each build in a fresh process with its src directory first on PYTHONPATH.
The panel mode isolates the changed epilogue; shared mode includes the DSV4
shared-aware quick planner and uses reduced experts at the TP4 rank shape.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from fused_cpp import moe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("panel", "shared"), required=True)
    parser.add_argument("--impl", choices=("jit", "asm"), default="jit")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "shared" and args.impl == "asm":
        parser.error("the delivery's clamped shared path requires the AOT (jit) selector")
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = args.impl
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    cpus = tuple(sorted(os.sched_getaffinity(0)))
    threads = 1 if args.mode == "panel" else len(cpus)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    generator = torch.Generator().manual_seed(29)
    hidden, intermediate, experts, top_k = (512, 64, 1, 1) if args.mode == "panel" else (4096, 512, 20, 6)

    def bf16(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    w13 = bf16((experts, 2 * intermediate, hidden), 0.05)
    w2 = bf16((experts, hidden, intermediate), 0.05)
    runtime = None
    if args.mode == "panel":
        packed = moe.prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        cases = (9, 12, 24, 128)
    else:
        if args.profile is None:
            parser.error("shared mode requires --profile")
        packed = moe.prepare_routed_shared_moe_bf16_tiled_weights(
            w13,
            w2,
            bf16((2 * intermediate, hidden), 0.05),
            bf16((hidden, intermediate), 0.05),
            backend="arm_sve_bf16",
        )
        runtime = moe.MoePlannerRuntime(
            args.profile,
            hidden_size=hidden,
            intermediate_size=intermediate,
            global_experts=experts,
            local_experts=experts,
            shared_experts=1,
            mode="tp",
            degree=4,
            concurrent_ranks=1,
            cpu_ids=cpus,
            cost_cache_dir=None,
        )
        moe.set_default_moe_planner_runtime(runtime)
        cases = (128, 512, 2048)
    results = []
    for tokens in cases:
        x = bf16((tokens, hidden), 0.5)
        ids = (torch.arange(tokens)[:, None] + torch.arange(top_k)[None, :]) % experts
        ids = ids.to(torch.int32)
        weights = torch.softmax(torch.randn((tokens, top_k), generator=generator), dim=-1)

        def run() -> torch.Tensor:
            if args.mode == "panel":
                return moe.fused_moe_bf16_tiled(
                    x, packed, weights, ids, num_threads=threads, swiglu_limit=10.0 if args.impl == "jit" else None
                )
            return moe.fused_moe_bf16_tiled_with_shared(x, packed, weights, ids, num_threads=threads, swiglu_limit=10.0)

        reference = run().clone()
        assert torch.isfinite(reference).all(), "nonfinite benchmark output"
        for _ in range(args.warmup):
            run()
        times_ms = []
        for _ in range(args.samples):
            start = time.perf_counter_ns()
            result = run()
            times_ms.append((time.perf_counter_ns() - start) / 1e6)
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
        plan = None
        if runtime is not None:
            combined_ids = torch.cat((ids, torch.full((tokens, 1), experts, dtype=torch.int32)), dim=1)
            plan = runtime.plan_for_shared_dispatch(packed, combined_ids, num_threads=threads, activation="silu")
        geometry = []
        n_tile = packed.backend_n_tile if args.mode == "panel" else packed.packed.backend_n_tile
        if plan is not None:
            for width, w13_window, w2_window in zip(
                plan.task_threads.tolist(), plan.task_w13_window_tiles.tolist(), plan.task_w2_window_tiles.tolist()
            ):
                total13, total2 = 2 * intermediate // n_tile, hidden // n_tile
                owner13 = (total13 + width - 1) // width if w13_window == 0 else w13_window
                owner2 = (total2 + width - 1) // width if w2_window == 0 else w2_window
                item = {
                    "threads": width,
                    "w13_window_tiles": w13_window,
                    "w2_window_tiles": w2_window,
                    "w13_owner_bytes": owner13 * n_tile * hidden * 2,
                    "w2_owner_bytes": owner2 * n_tile * intermediate * 2,
                    "R13": (total13 + width * owner13 - 1) // (width * owner13),
                    "R2": (total2 + width * owner2 - 1) // (width * owner2),
                }
                if item not in geometry:
                    geometry.append(item)
        row = {
            "tokens": tokens,
            "median_ms": statistics.median(times_ms),
            "samples_ms": times_ms,
            "planner": dict(runtime._planner.last) if runtime is not None else None,
            "geometry": geometry,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    document = {
        "mode": args.mode,
        "impl": args.impl,
        "module": moe.__file__,
        "torch": torch.__version__,
        "cpu_ids": cpus,
        "threads": threads,
        "hidden": hidden,
        "intermediate": intermediate,
        "routed_experts": experts,
        "shared_experts": int(args.mode == "shared"),
        "top_k": top_k,
        "n_tile": n_tile,
        "w13_stage_bytes_per_expert": 2 * intermediate * hidden * 2,
        "w2_stage_bytes_per_expert": hidden * intermediate * 2,
        "warmup": args.warmup,
        "samples": args.samples,
        "results": results,
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
