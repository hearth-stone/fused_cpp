#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fused_cpp.moe import (  # noqa: E402
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


H = 4096
F = 512
EXPERTS = 64
CORE = 48
WARMUP_CYCLES = 1
MEASURE_CYCLES = 3
BFMMLA_PEAK_GFLOPS = 403.8
PACKED_B_READ_CEILING_GBS = 40.04
TRACE = Path("/tmp/moe_single_expert_m_efficiency.trace")


def bf16(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, 0.01, generator=generator)


def parse_stage_times() -> list[float]:
    calls: dict[int, dict[str, float]] = {}
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("PHASE "):
            continue
        fields = dict(pattern.findall(line))
        stage = fields.get("stage")
        if stage not in {"w13_fused_silu_packc", "w13", "w2_packed", "w2_direct_route"}:
            continue
        call_id = int(fields["call_id"])
        calls.setdefault(call_id, {})[stage] = calls.setdefault(call_id, {}).get(stage, 0.0) + float(fields["ms"])

    totals: list[float] = []
    for stages in calls.values():
        w13 = stages.get("w13_fused_silu_packc", stages.get("w13", 0.0))
        w2 = stages.get("w2_packed", stages.get("w2_direct_route", 0.0))
        if w13 > 0.0 and w2 > 0.0:
            totals.append(w13 + w2)
    if not totals:
        raise RuntimeError("trace contained no complete W13+W2 calls")
    return totals


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"

    generator = torch.Generator().manual_seed(20260723)
    w13 = bf16((EXPERTS, 2 * F, H), generator)
    w2 = bf16((EXPERTS, H, F), generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    del w13, w2
    if int(packed.gemm_backend) != 1:
        raise RuntimeError(f"expected SVE backend, got {packed.gemm_backend}")

    expert_ids = [torch.tensor([expert], dtype=torch.int32) for expert in range(EXPERTS)]
    wave_offsets = torch.tensor([0, 1], dtype=torch.int32)
    team_threads = torch.tensor([1], dtype=torch.int32)
    thread_cpu_ids = torch.tensor([CORE], dtype=torch.int32)
    records: list[dict[str, float]] = []
    sink = 0

    for rows in range(1, 13):
        hidden = bf16((rows, H), generator)
        route_weights = torch.ones((rows, 1), dtype=torch.float32)
        route_ids = [torch.full((rows, 1), expert, dtype=torch.int32) for expert in range(EXPERTS)]
        output = torch.empty_like(hidden)

        def run(expert: int) -> torch.Tensor:
            return fused_moe_bf16_tiled_scheduled(
                hidden,
                packed,
                route_weights,
                route_ids[expert],
                wave_offsets,
                expert_ids[expert],
                team_threads,
                thread_cpu_ids=thread_cpu_ids,
                num_threads=1,
                global_num_experts=EXPERTS,
                skip_weighted=True,
                out=output,
            )

        os.environ["FUSED_CPP_MOE_TRACE"] = "0"
        for cycle in range(WARMUP_CYCLES):
            for expert in range(EXPERTS):
                sink ^= int(run((expert + cycle) % EXPERTS).view(torch.int16)[0, 0])

        TRACE.unlink(missing_ok=True)
        os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(TRACE)
        os.environ["FUSED_CPP_MOE_TRACE"] = "1"
        for cycle in range(MEASURE_CYCLES):
            for expert in range(EXPERTS):
                sink ^= int(run((expert + cycle * 17) % EXPERTS).view(torch.int16)[0, 0])
        os.environ["FUSED_CPP_MOE_TRACE"] = "0"

        samples_ms = parse_stage_times()
        median_ms = statistics.median(samples_ms)
        seconds = median_ms / 1e3
        compute_rows = 2 * ((rows + 1) // 2)
        useful_flops = 6 * rows * H * F
        physical_flops = 6 * compute_rows * H * F
        weight_bytes = 6 * H * F
        records.append(
            {
                "m": float(rows),
                "samples": float(len(samples_ms)),
                "ms": median_ms,
                "lane_eff": rows / compute_rows,
                "useful_gflops": useful_flops / seconds / 1e9,
                "physical_gflops": physical_flops / seconds / 1e9,
                "useful_compute_eff": useful_flops / seconds / 1e9 / BFMMLA_PEAK_GFLOPS,
                "issue_eff": physical_flops / seconds / 1e9 / BFMMLA_PEAK_GFLOPS,
                "weight_gbs": weight_bytes / seconds / 1e9,
            }
        )

    max_weight_gbs = max(record["weight_gbs"] for record in records)
    print(
        f"host_core={CORE} H={H} F={F} experts={EXPERTS} "
        f"stage_geometry=full_n_team_stripes jit=1 peak={BFMMLA_PEAK_GFLOPS:.1f}GF/s "
        f"packed_b_read_ceiling={PACKED_B_READ_CEILING_GBS:.2f}GB/s "
        f"max_observed_weight_rate={max_weight_gbs:.3f}GB/s sink={sink}"
    )
    print(
        "M samples gemm_ms useful_GF physical_GF lane_eff useful_compute_eff "
        "physical_issue_eff weight_GB/s memory_eff"
    )
    for record in records:
        print(
            f"{int(record['m']):2d} {int(record['samples']):7d} "
            f"{record['ms']:7.4f} {record['useful_gflops']:9.2f} "
            f"{record['physical_gflops']:10.2f} {record['lane_eff'] * 100:8.2f}% "
            f"{record['useful_compute_eff'] * 100:18.2f}% "
            f"{record['issue_eff'] * 100:18.2f}% {record['weight_gbs']:11.3f} "
            f"{record['weight_gbs'] / PACKED_B_READ_CEILING_GBS * 100:9.2f}%"
        )


if __name__ == "__main__":
    main()
