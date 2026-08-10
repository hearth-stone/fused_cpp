#!/usr/bin/env python3
"""Profile scheduled MoE stage-time composition for one active expert."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fused_cpp.moe import (  # noqa: E402
    _HAS_BF16_TILED_FUSED_MOE,
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


PIPELINE_STAGES = [
    "plan_materialize",
    "route_build",
    "plan_validate",
    "scratch_alloc",
    "gather_input",
    "w13",
    "activation",
    "w2",
    "scatter_route_out",
    "compute_gap",
    "merge_routes_total",
    "output_cast",
    "other",
]

GATHER_STAGE_ALIASES = ("gather_input", "gather_pack_a")
W13_STAGE_ALIASES = ("w13", "w13_fused_silu_packc")
ACTIVATION_STAGE_ALIASES = ("activation",)
W2_STAGE_ALIASES = ("w2", "w2_packed")


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def env_enabled(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value[0] != "0")


def bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def choose_moe_gemm_axis(stage: str, rows: int, n_cols: int, threads: int) -> str:
    if threads <= 1:
        return "N"
    if stage == "w13":
        if threads == 8 and rows >= 512:
            return "M"
        if threads == 4 and rows >= 1024:
            return "M"
        if threads == 2 and rows <= 32:
            return "M"
        return "M" if threads > 8 and rows > n_cols else "N"

    if threads == 8 and rows >= 4096:
        return "M"
    if threads == 4 and rows >= 8192:
        return "M"
    return "M" if threads > 8 and rows > n_cols else "N"


def parse_kv_line(line: str) -> Dict[str, str]:
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)", line))


def parse_trace(path: Path) -> List[Dict[str, object]]:
    by_call: Dict[int, Dict[str, object]] = {}
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for line in text.splitlines():
        if line.startswith("MOE_CALL "):
            fields = parse_kv_line(line)
            call_id = int(fields["call_id"])
            by_call.setdefault(call_id, {"gemm": {}, "phase": {}})
            by_call[call_id]["e2e_ms"] = float(fields["e2e_ms"])
            by_call[call_id]["threads"] = int(fields["threads"])
            by_call[call_id]["tokens"] = int(fields["tokens"])
            continue

        if line.startswith("GEMM "):
            fields = parse_kv_line(line)
            call_id = int(fields["call_id"])
            stage = fields["stage"]
            ms = float(fields["ms"])
            call = by_call.setdefault(call_id, {"gemm": {}, "phase": {}})
            call["gemm"].setdefault(stage, []).append(ms)
            continue

        if line.startswith("PHASE "):
            fields = parse_kv_line(line)
            call_id = int(fields["call_id"])
            stage = fields["stage"]
            ms = float(fields["ms"])
            call = by_call.setdefault(call_id, {"gemm": {}, "phase": {}})
            call["phase"].setdefault(stage, []).append(ms)

    rows: List[Dict[str, object]] = []
    for call_id in sorted(by_call):
        call = by_call[call_id]
        if "e2e_ms" not in call:
            continue
        phases = call["phase"]
        gemms = call["gemm"]
        if not phases:
            raise RuntimeError("trace has no PHASE records; rebuild the extension with the MoE phase-trace changes")

        def phase_max(*names: str) -> float:
            values: List[float] = []
            for name in names:
                values.extend(phases.get(name, []))
            return max(values) if values else 0.0

        def gemm_max(*names: str) -> float:
            values: List[float] = []
            for name in names:
                values.extend(gemms.get(name, []))
            return max(values) if values else 0.0

        stage_ms = {
            "plan_materialize": sum(phases.get("plan_materialize", [])),
            "route_build": sum(phases.get("route_build", [])),
            "plan_validate": sum(phases.get("plan_validate", [])),
            "scratch_alloc": sum(phases.get("scratch_alloc", [])),
            "gather_input": phase_max(*GATHER_STAGE_ALIASES),
            "w13": max(
                gemm_max(*W13_STAGE_ALIASES),
                phase_max(*W13_STAGE_ALIASES),
            ),
            "activation": phase_max(*ACTIVATION_STAGE_ALIASES),
            "w2": max(
                gemm_max(*W2_STAGE_ALIASES),
                phase_max(*W2_STAGE_ALIASES),
            ),
            "scatter_route_out": max(phases.get("scatter_route_out", [0.0])),
            "merge_routes_total": sum(phases.get("merge_routes_total", [])),
            "merge_routes_worker_max": max(phases.get("merge_routes", [0.0])),
            "output_cast": sum(phases.get("output_cast", [])),
            "scheduled_compute": sum(phases.get("scheduled_compute", [])),
        }
        inner_compute_ms = (
            stage_ms["gather_input"]
            + stage_ms["w13"]
            + stage_ms["activation"]
            + stage_ms["w2"]
            + stage_ms["scatter_route_out"]
        )
        stage_ms["compute_gap"] = max(
            0.0,
            stage_ms["scheduled_compute"] - inner_compute_ms,
        )
        e2e_ms = float(call["e2e_ms"])
        top_level_accounted_ms = (
            stage_ms["plan_materialize"]
            + stage_ms["route_build"]
            + stage_ms["plan_validate"]
            + stage_ms["scratch_alloc"]
            + stage_ms["scheduled_compute"]
            + stage_ms["merge_routes_total"]
            + stage_ms["output_cast"]
        )
        stage_ms["other"] = max(0.0, e2e_ms - top_level_accounted_ms)
        rows.append(
            {
                "call_id": call_id,
                "e2e_ms": e2e_ms,
                "stage_ms": stage_ms,
            }
        )
    return rows


def summarize_trace_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("empty trace rows")
    summary = {"trace_e2e_ms": median([float(row["e2e_ms"]) for row in rows])}
    for stage in PIPELINE_STAGES:
        summary[f"{stage}_ms"] = median([float(row["stage_ms"].get(stage, 0.0)) for row in rows])
    summary["scheduled_compute_ms"] = median([float(row["stage_ms"].get("scheduled_compute", 0.0)) for row in rows])
    summary["merge_routes_worker_max_ms"] = median(
        [float(row["stage_ms"].get("merge_routes_worker_max", 0.0)) for row in rows]
    )
    gemm_ms = summary["w13_ms"] + summary["w2_ms"]
    summary["gemm_ms"] = gemm_ms
    summary["non_gemm_ms"] = max(0.0, summary["trace_e2e_ms"] - gemm_ms)
    for stage in PIPELINE_STAGES + ["gemm", "non_gemm", "scheduled_compute"]:
        ms = summary.get(f"{stage}_ms", 0.0)
        summary[f"{stage}_pct"] = 100.0 * ms / summary["trace_e2e_ms"] if summary["trace_e2e_ms"] > 0.0 else 0.0
    return summary


def measure_call(run, *, runs: int) -> List[float]:
    times_ms: List[float] = []
    for _ in range(runs):
        begin = time.perf_counter_ns()
        out = run()
        _ = float(out.flatten()[0])
        times_ms.append((time.perf_counter_ns() - begin) / 1e6)
    return times_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure scheduled MoE stage-time composition by enabling "
            "FUSED_CPP_MOE_TRACE and parsing GEMM/PHASE records."
        )
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--routes", default="2048")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--activation", choices=["silu", "swigluoai"], default="silu")
    parser.add_argument(
        "--fuse-silu",
        action="store_true",
        help="Prepare weights with the fused SiLU epilogue layout.",
    )
    parser.add_argument(
        "--weighted-merge",
        action="store_true",
        help="Use weighted top-k merge instead of skip_weighted=True.",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=Path("/tmp/moe_stage_breakdown_trace.log"),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError("BF16 tiled fused MoE backend is unavailable")
    if args.hidden_size <= 0 or args.ffn_hidden_size <= 0:
        raise ValueError("hidden and FFN sizes must be positive")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs positive")

    routes_list = parse_int_list(args.routes)
    threads_list = parse_int_list(args.threads)

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
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=args.fuse_silu)

    print(
        "shape "
        f"H={args.hidden_size} F={args.ffn_hidden_size} "
        f"activation={args.activation}; "
        "pct columns are relative to traced C++ e2e_ms"
    )
    print(
        "routes threads axis_w13 axis_w2 full_ms trace_ms "
        "route% scratch% gather% w13% act% w2% scatter% gap% merge% cast% other%"
    )

    payload_rows: List[Dict[str, object]] = []
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
                    activation=args.activation,
                    global_num_experts=1,
                    skip_weighted=not args.weighted_merge,
                )

            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            for _ in range(args.warmup):
                out = run()
                _ = float(out.flatten()[0])

            full_times_ms = measure_call(run, runs=args.runs)

            if args.trace_file.exists():
                args.trace_file.unlink()
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(args.trace_file)
            for _ in range(args.runs):
                out = run()
                _ = float(out.flatten()[0])
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

            trace_rows = parse_trace(args.trace_file)
            summary = summarize_trace_rows(trace_rows)
            full_ms = median(full_times_ms)
            axis_w13 = choose_moe_gemm_axis("w13", routes, 2 * args.ffn_hidden_size, threads)
            axis_w2 = choose_moe_gemm_axis("w2", routes, args.hidden_size, threads)
            row = {
                "routes": routes,
                "threads": threads,
                "w13_parallel_axis": axis_w13,
                "w2_parallel_axis": axis_w2,
                "full_call_median_ms": full_ms,
                "full_call_times_ms": full_times_ms,
                **summary,
            }
            payload_rows.append(row)
            print(
                f"{routes:<6} {threads:<7} {axis_w13:<8} {axis_w2:<7} "
                f"{full_ms:7.3f} {summary['trace_e2e_ms']:8.3f} "
                f"{summary['route_build_pct']:6.1f} "
                f"{summary['scratch_alloc_pct']:8.1f} "
                f"{summary['gather_input_pct']:7.1f} "
                f"{summary['w13_pct']:5.1f} "
                f"{summary['activation_pct']:5.1f} "
                f"{summary['w2_pct']:4.1f} "
                f"{summary['scatter_route_out_pct']:8.1f} "
                f"{summary['compute_gap_pct']:5.1f} "
                f"{summary['merge_routes_total_pct']:6.1f} "
                f"{summary['output_cast_pct']:5.1f} "
                f"{summary['other_pct']:6.1f}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "kernel": {
            "backend_n_tile": packed.backend_n_tile,
            "parallel_axis": "N",
            "stage_geometry": "full_n_team_stripes",
        },
        "shape": {
            "hidden_size": args.hidden_size,
            "ffn_hidden_size": args.ffn_hidden_size,
            "activation": args.activation,
            "fuse_silu": args.fuse_silu,
            "top_k": 1,
            "skip_weighted": not args.weighted_merge,
        },
        "warmup": args.warmup,
        "runs": args.runs,
        "rows": payload_rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote_json={args.output_json}")
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [key for key in payload_rows[0].keys() if key != "full_call_times_ms"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in payload_rows:
                writer.writerow({key: value for key, value in row.items() if key in fieldnames})
        print(f"wrote_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
