#!/usr/bin/env python3
"""Roofline for the fused BF16 MoE GEMM kernel.

The memory model follows the packed-A BF16 GEMM loop structure instead of the
usual one-pass A+B+C accounting. For an MxK by KxN GEMM with an M-panel kernel:

    A_read = M * K * sizeof(bf16) * n_partitions
    B_read = ceil(M / m_panel) * K * N * sizeof(bf16)

For the current N-split threaded path, A is read once per N partition while the
total B traffic across workers is still one streamed B pass per A panel. This is
the key difference from a naive A+B model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fused_cpp.moe import (  # noqa: E402
    _HAS_BF16_TILED_FUSED_MOE,
    fused_moe_bf16_tiled_scheduled,
    prepare_fused_moe_bf16_tiled_weights,
)


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def bf16_normal(shape: tuple[int, ...], generator: torch.Generator, std: float):
    return torch.empty(shape, dtype=torch.bfloat16).normal_(0.0, std, generator=generator)


def median(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("empty measurement list")
    return float(statistics.median(vals))


def parse_kv_line(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)", line))


def parse_trace(path: Path, stage_aliases: tuple[str, ...]) -> list[float]:
    by_call: dict[int, list[float]] = {}
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    aliases = set(stage_aliases)
    for line in text.splitlines():
        if not line.startswith("GEMM ") and not line.startswith("PHASE "):
            continue
        fields = parse_kv_line(line)
        if fields.get("stage") not in aliases:
            continue
        if "call_id" not in fields or "ms" not in fields:
            continue
        by_call.setdefault(int(fields["call_id"]), []).append(float(fields["ms"]))
    return [max(values) for _, values in sorted(by_call.items()) if values]


def choose_w13_parallel_axis(rows: int, n_cols: int, threads: int) -> str:
    if threads <= 1:
        return "N"
    if threads == 8 and rows >= 512:
        return "M"
    if threads == 4 and rows >= 1024:
        return "M"
    if threads == 2 and rows <= 32:
        return "M"
    return "M" if threads > 8 and rows > n_cols else "N"


@dataclass(frozen=True)
class KernelMemory:
    a_read_bytes: int
    b_read_bytes: int
    c_write_bytes: int
    total_bytes: int
    m_panels: int
    n_partitions: int


def w13_kernel_memory(
    *,
    m: int,
    k: int,
    ffn_hidden_size: int,
    threads: int,
    parallel_axis: str,
    m_panel: int,
    bytes_a: int = 2,
    bytes_b: int = 2,
    bytes_c: int = 2,
    include_c_write: bool = True,
) -> KernelMemory:
    n = 2 * ffn_hidden_size
    m_panels = math.ceil(m / m_panel)
    n_partitions = threads if parallel_axis == "N" else 1
    a_read = m * k * bytes_a * n_partitions
    b_read = m_panels * k * n * bytes_b
    # Fused W13 stores the SiLU(up/gate) product, i.e. F columns, not 2F.
    c_write = m * ffn_hidden_size * bytes_c if include_c_write else 0
    return KernelMemory(
        a_read_bytes=a_read,
        b_read_bytes=b_read,
        c_write_bytes=c_write,
        total_bytes=a_read + b_read + c_write,
        m_panels=m_panels,
        n_partitions=n_partitions,
    )


@dataclass(frozen=True)
class RooflineRow:
    m: int
    n: int
    k: int
    threads: int
    parallel_axis: str
    m_panels: int
    n_partitions: int
    w13_ms: float
    tflops: float
    kernel_bytes: int
    ai_flop_per_byte: float
    bandwidth_gbs: float
    roofline_tflops: float | None
    roofline_util: float | None
    a_read_mib: float
    b_read_mib: float
    c_write_mib: float


def make_row(
    *,
    routes: int,
    hidden_size: int,
    ffn_hidden_size: int,
    threads: int,
    parallel_axis: str,
    m_panel: int,
    w13_ms: float,
    peak_tflops: float | None,
    bandwidth_gbs: float | None,
    include_c_write: bool,
) -> RooflineRow:
    n = 2 * ffn_hidden_size
    flops = 2 * routes * hidden_size * n
    mem = w13_kernel_memory(
        m=routes,
        k=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        threads=threads,
        parallel_axis=parallel_axis,
        m_panel=m_panel,
        include_c_write=include_c_write,
    )
    seconds = w13_ms / 1e3
    tflops = flops / seconds / 1e12
    ai = flops / mem.total_bytes
    measured_bw = mem.total_bytes / seconds / 1e9
    roof = None
    util = None
    if peak_tflops is not None and bandwidth_gbs is not None:
        roof = min(peak_tflops, bandwidth_gbs * ai / 1e3)
        util = tflops / roof if roof > 0 else None
    return RooflineRow(
        m=routes,
        n=n,
        k=hidden_size,
        threads=threads,
        parallel_axis=parallel_axis,
        m_panels=mem.m_panels,
        n_partitions=mem.n_partitions,
        w13_ms=w13_ms,
        tflops=tflops,
        kernel_bytes=mem.total_bytes,
        ai_flop_per_byte=ai,
        bandwidth_gbs=measured_bw,
        roofline_tflops=roof,
        roofline_util=util,
        a_read_mib=mem.a_read_bytes / (1024 * 1024),
        b_read_mib=mem.b_read_bytes / (1024 * 1024),
        c_write_mib=mem.c_write_bytes / (1024 * 1024),
    )


def measure_w13_ms(
    *,
    hidden_size: int,
    ffn_hidden_size: int,
    routes: int,
    threads: int,
    warmup: int,
    runs: int,
    generator: torch.Generator,
    std: float,
    trace_file: Path,
    fuse_silu: bool,
) -> float:
    w13 = bf16_normal((1, 2 * ffn_hidden_size, hidden_size), generator=generator, std=std)
    w2 = bf16_normal((1, hidden_size, ffn_hidden_size), generator=generator, std=std)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=fuse_silu)
    hidden = bf16_normal((routes, hidden_size), generator=generator, std=std)
    topk_ids = torch.zeros((routes, 1), dtype=torch.int32)
    topk_weights = torch.ones((routes, 1), dtype=torch.float32)
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
    for _ in range(warmup):
        out = run()
        _ = float(out.flatten()[0])

    if trace_file.exists():
        trace_file.unlink()
    os.environ["FUSED_CPP_MOE_TRACE"] = "1"
    os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_file)
    for _ in range(runs):
        out = run()
        _ = float(out.flatten()[0])
    os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    stage_aliases = ("w13", "w13_fused_silu_packc")
    return median(parse_trace(trace_file, stage_aliases))


def format_optional(value: float | None, fmt: str) -> str:
    return "-" if value is None else format(value, fmt)


def print_table(rows: list[RooflineRow]) -> None:
    print(
        "M     N     K     T   axis  panels npart  w13_ms  TFLOP/s  "
        "AI(F/B)  BWreq(GB/s)  roof(T) util   A_MiB   B_MiB   C_MiB"
    )
    for r in rows:
        print(
            f"{r.m:<5} {r.n:<5} {r.k:<5} {r.threads:<3} "
            f"{r.parallel_axis:<5} {r.m_panels:<6} {r.n_partitions:<5} "
            f"{r.w13_ms:7.3f} {r.tflops:8.3f} "
            f"{r.ai_flop_per_byte:8.2f} {r.bandwidth_gbs:11.1f} "
            f"{format_optional(r.roofline_tflops, '7.3f')} "
            f"{format_optional(r.roofline_util, '5.2f')} "
            f"{r.a_read_mib:7.2f} {r.b_read_mib:7.2f} {r.c_write_mib:7.2f}"
        )


def row_from_json(row: dict, peak_tflops: float | None, bandwidth_gbs: float | None) -> RooflineRow:
    if peak_tflops is not None and bandwidth_gbs is not None:
        roof = min(peak_tflops, bandwidth_gbs * float(row["ai_flop_per_byte"]) / 1e3)
        util = float(row["tflops"]) / roof if roof > 0 else None
    else:
        roof = row.get("roofline_tflops")
        util = row.get("roofline_util")
    data = dict(row)
    data["roofline_tflops"] = roof
    data["roofline_util"] = util
    return RooflineRow(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--routes", default="48,96,192,384,768,1024")
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--m-panel", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--std", type=float, default=0.01)
    parser.add_argument("--trace-file", type=Path, default=Path("/tmp/bf16_roofline_trace.log"))
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--bandwidth-gbs", type=float, default=None)
    parser.add_argument("--no-c-write", action="store_true")
    parser.add_argument("--no-fuse-silu", action="store_true")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Reprint an existing roofline JSON instead of running benchmarks.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError("BF16 tiled fused MoE backend is unavailable")
    if min(args.hidden_size, args.ffn_hidden_size, args.m_panel) <= 0:
        raise ValueError("hidden size, FFN size, and m-panel must be positive")
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs positive")

    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        rows = [row_from_json(row, args.peak_tflops, args.bandwidth_gbs) for row in payload["rows"]]
        print_table(rows)
        return 0

    routes_list = parse_int_list(args.routes)
    threads_list = parse_int_list(args.threads)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)

    rows: list[RooflineRow] = []
    print(
        f"BF16 W13 roofline H={args.hidden_size} F={args.ffn_hidden_size} "
        f"N={2 * args.ffn_hidden_size} m_panel={args.m_panel} "
        f"kernel_bytes=A*npart + ceil(M/{args.m_panel})*B"
        f"{' + Cwrite' if not args.no_c_write else ''}"
    )
    print(
        "M     N     K     T   axis  panels npart  w13_ms  TFLOP/s  "
        "AI(F/B)  BWreq(GB/s)  roof(T) util   A_MiB   B_MiB   C_MiB"
    )
    for routes in routes_list:
        for threads in threads_list:
            parallel_axis = choose_w13_parallel_axis(routes, 2 * args.ffn_hidden_size, threads)
            w13_ms = measure_w13_ms(
                hidden_size=args.hidden_size,
                ffn_hidden_size=args.ffn_hidden_size,
                routes=routes,
                threads=threads,
                warmup=args.warmup,
                runs=args.runs,
                generator=generator,
                std=args.std,
                trace_file=args.trace_file,
                fuse_silu=not args.no_fuse_silu,
            )
            rows.append(
                make_row(
                    routes=routes,
                    hidden_size=args.hidden_size,
                    ffn_hidden_size=args.ffn_hidden_size,
                    threads=threads,
                    parallel_axis=parallel_axis,
                    m_panel=args.m_panel,
                    w13_ms=w13_ms,
                    peak_tflops=args.peak_tflops,
                    bandwidth_gbs=args.bandwidth_gbs,
                    include_c_write=not args.no_c_write,
                )
            )
            row = rows[-1]
            print(
                f"{row.m:<5} {row.n:<5} {row.k:<5} {row.threads:<3} "
                f"{row.parallel_axis:<5} {row.m_panels:<6} {row.n_partitions:<5} "
                f"{row.w13_ms:7.3f} {row.tflops:8.3f} "
                f"{row.ai_flop_per_byte:8.2f} {row.bandwidth_gbs:11.1f} "
                f"{format_optional(row.roofline_tflops, '7.3f')} "
                f"{format_optional(row.roofline_util, '5.2f')} "
                f"{row.a_read_mib:7.2f} {row.b_read_mib:7.2f} {row.c_write_mib:7.2f}",
                flush=True,
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "shape": {
                "stage": "w13",
                "hidden_size": args.hidden_size,
                "ffn_hidden_size": args.ffn_hidden_size,
                "n": 2 * args.ffn_hidden_size,
                "m_panel": args.m_panel,
                "include_c_write": not args.no_c_write,
                "kernel_memory_model": ("A*M*K duplicated by N partitions; B*K*N streamed once per M panel"),
            },
            "warmup": args.warmup,
            "runs": args.runs,
            "peak_tflops": args.peak_tflops,
            "bandwidth_gbs": args.bandwidth_gbs,
            "rows": [asdict(row) for row in rows],
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
