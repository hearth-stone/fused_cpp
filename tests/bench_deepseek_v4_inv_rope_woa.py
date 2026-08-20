from __future__ import annotations

import argparse
import statistics
import time

import torch

from fused_cpp.deepseek_v4_inv_rope_woa import (
    deepseek_v4_inv_rope_grouped_woa,
    deepseek_v4_inv_rope_grouped_woa_torch_reference,
    prepare_deepseek_v4_inv_rope_woa,
)


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _measure(function, warmup: int, runs: int) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    samples: list[float] = []
    for _ in range(runs):
        begin = time.perf_counter()
        function()
        samples.append((time.perf_counter() - begin) * 1e3)
    return statistics.median(samples), _percentile(samples, 0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 12, 64, 256, 2048])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--output-rank", type=int, default=1024)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(20260818)
    torch.set_num_threads(args.threads)
    groups, heads_per_group, head_dim, rope_dim = 4, 8, 512, 64
    output_rank = args.output_rank
    heads = groups * heads_per_group
    grouped_dim = heads_per_group * head_dim
    max_position = max(args.tokens) + 17
    weight = (torch.randn(groups * output_rank, grouped_dim) * 0.02).to(torch.bfloat16)
    cache = torch.randn(max_position, rope_dim, dtype=torch.float32).contiguous()
    prepared = prepare_deepseek_v4_inv_rope_woa(
        weight,
        n_groups=groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        rope_dim=rope_dim,
        backend="arm_sve_bf16",
    )
    core_ids = tuple(range(args.threads))

    print("tokens  torch_ms  native_ms  speedup  native_p90  native_tflops  max_abs")
    for tokens in args.tokens:
        o = (torch.randn(tokens, heads, head_dim) * 0.1).to(torch.bfloat16)
        positions = torch.arange(tokens, dtype=torch.int64)
        torch_out = torch.empty(tokens, groups, output_rank, dtype=torch.bfloat16)
        native_out = torch.empty_like(torch_out)

        def run_torch() -> None:
            deepseek_v4_inv_rope_grouped_woa_torch_reference(
                o,
                positions,
                cache,
                weight,
                n_groups=groups,
                heads_per_group=heads_per_group,
                rope_dim=rope_dim,
                out=torch_out,
            )

        def run_native() -> None:
            deepseek_v4_inv_rope_grouped_woa(
                o,
                positions,
                cache,
                prepared,
                out=native_out,
                core_ids=core_ids,
            )

        run_native()
        if args.native_only:
            max_abs = float("nan")
            torch_ms = float("nan")
        else:
            run_torch()
            max_abs = float((torch_out.float() - native_out.float()).abs().max())
            torch_ms, _ = _measure(run_torch, args.warmup, args.runs)
        native_ms, native_p90 = _measure(run_native, args.warmup, args.runs)
        flops = 2.0 * tokens * groups * output_rank * grouped_dim
        tflops = flops / (native_ms * 1e9)
        print(
            f"{tokens:6d} {torch_ms:9.3f} {native_ms:10.3f} "
            f"{torch_ms / native_ms:8.3f} {native_p90:11.3f} {tflops:14.3f} {max_abs:8.4f}"
        )


if __name__ == "__main__":
    main()
