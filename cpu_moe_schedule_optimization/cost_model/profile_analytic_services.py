#!/usr/bin/env python3
"""Profile route-independent service anchors for the analytical MoE model.

The load probes execute the generated SVE packed-B loop with all arithmetic and
stores disabled. Hot working sets select endpoint-to-register service from L1,
private L2, or shared LLC; the DRAM probe rotates disjoint packed weights whose
aggregate reuse distance is larger than the NUMA-local LLC. These endpoint
ceilings overlap and are composed with max by the model. No routed-expert
latency is used by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytic_probe_geometry import ProbeGeometry, b_only_geometry, m12_gemm_geometry, read_cache_info  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights  # noqa: E402


PROBE_B_ONLY = 1
PROBE_FULL_NO_STORE = 4
PROBE_MATRIX_ONLY = 10
PROBE_FULL_NO_STORE_COLUMN_PIPELINE = 11
M12_ROWS = 12
DEFAULT_N_TILE = 8


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or min(values) <= 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"expected unique positive integers, got {value!r}")
    return values


def parse_cpu_ids(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            if first < 0 or last < first:
                raise argparse.ArgumentTypeError(f"invalid CPU range {item!r}")
            result.extend(range(first, last + 1))
        else:
            result.append(int(item))
    if not result or min(result) < 0 or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"invalid CPU list {value!r}")
    return result


def packed_weights(*, experts: int, k: int, n: int, seed: int):
    if n % 2:
        raise ValueError("packed probe N must be even")
    generator = torch.Generator().manual_seed(seed)
    w13 = torch.empty((experts, n, k), dtype=torch.bfloat16)
    w2 = torch.empty((experts, k, n // 2), dtype=torch.bfloat16)
    w13.normal_(0.0, 0.01, generator=generator)
    w2.normal_(0.0, 0.01, generator=generator)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="sve")
    del w13, w2
    gc.collect()
    return packed


def bind_current_thread(cpu: int) -> None:
    native_id = threading.get_native_id()
    os.sched_setaffinity(native_id, {cpu})


def concurrent_b_probe(
    *,
    packed_b: torch.Tensor,
    k: int,
    n: int,
    n_tile: int,
    width: int,
    cpu_ids: list[int],
    experts_per_thread: int,
    warmup: int,
    runs: int,
) -> dict:
    shared_weight = packed_b.shape[0] == 1 and experts_per_thread == 1
    if not shared_weight and packed_b.shape[0] < width * experts_per_thread:
        raise ValueError("packed-B tensor is too small for disjoint worker windows")
    barrier = threading.Barrier(width)
    a = torch.zeros((1, k), dtype=torch.bfloat16)

    def worker(worker_id: int) -> list[float]:
        bind_current_thread(cpu_ids[worker_id])
        if shared_weight:
            worker_b = packed_b[:1]
        else:
            begin = worker_id * experts_per_thread
            worker_b = packed_b[begin : begin + experts_per_thread]
        barrier.wait()
        return list(
            _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                a,
                worker_b,
                k,
                n,
                n_tile,
                1,
                warmup,
                runs,
                PROBE_B_ONLY,
            )
        )

    with ThreadPoolExecutor(max_workers=width) as executor:
        samples_by_worker = list(executor.map(worker, range(width)))

    bytes_per_scan = k * n * 2
    slowest_timed_ms = max(sum(samples) for samples in samples_by_worker)
    slowest_median_ms = max(statistics.median(samples) for samples in samples_by_worker)
    aggregate_bytes = width * runs * bytes_per_scan
    return {
        "threads": width,
        "bytes_per_worker_scan": bytes_per_scan,
        "experts_per_thread": experts_per_thread,
        "aggregate_rate": aggregate_bytes / (slowest_timed_ms * 1e-3),
        "aggregate_gbytes_per_second": aggregate_bytes / (slowest_timed_ms * 1e6),
        "slowest_worker_median_ms": slowest_median_ms,
        "slowest_worker_timed_ms": slowest_timed_ms,
        "worker_median_ms": [statistics.median(samples) for samples in samples_by_worker],
    }


def profile_load_resource(
    *,
    name: str,
    geometry: ProbeGeometry,
    widths: list[int],
    cpu_ids: list[int],
    warmup: int,
    runs: int,
    seed: int,
    cold: bool,
    minimum_cold_experts: int,
    cold_experts_per_thread: int,
) -> dict:
    k = geometry.k
    n = geometry.n
    if cold:
        max_experts = max(
            width * max(cold_experts_per_thread, math.ceil(minimum_cold_experts / width)) for width in widths
        )
    else:
        max_experts = 1
    packed = packed_weights(experts=max_experts, k=k, n=n, seed=seed)
    packed_b = packed.w13[0]
    n_tile = int(packed.backend_n_tile)

    # Generate the probe before concurrent timing.
    _moe_C.fused_moe_bench_sve_jit_w13_gemm(
        torch.zeros((1, k), dtype=torch.bfloat16), packed_b[:1], k, n, n_tile, 1, 1, 1, PROBE_B_ONLY
    )

    order = list(widths)
    random.Random(seed).shuffle(order)
    rows = []
    for width in order:
        experts_per_thread = max(cold_experts_per_thread, math.ceil(minimum_cold_experts / width)) if cold else 1
        row = concurrent_b_probe(
            packed_b=packed_b,
            k=k,
            n=n,
            n_tile=n_tile,
            width=width,
            cpu_ids=cpu_ids,
            experts_per_thread=experts_per_thread,
            warmup=warmup,
            runs=runs,
        )
        rows.append(row)
        print(
            f"{name:<12} threads={width:>2} rate={row['aggregate_gbytes_per_second']:>8.2f} GB/s "
            f"median={row['slowest_worker_median_ms']:>8.4f} ms",
            flush=True,
        )
    del packed
    gc.collect()
    rows.sort(key=lambda row: int(row["threads"]))
    return {
        "k": k,
        "n": n,
        "working_set_bytes": k * n * 2,
        "target_bytes": geometry.target_bytes,
        "cache_bytes": geometry.cache_bytes,
        "cache_fraction": geometry.cache_fraction,
        "placement": "disjoint_cold_weights" if cold else "shared_read_only_weight",
        "probe_mode": "generated_sve_b_only",
        "point_order": order,
        "rows": rows,
    }


def profile_matrix(
    *,
    name: str,
    widths: list[int],
    cpu_ids: list[int],
    m: int,
    k: int,
    n: int,
    warmup: int,
    runs: int,
    seed: int,
    probe_mode: int,
    cache_level: str | None = None,
    cache_geometry: ProbeGeometry | None = None,
) -> dict:
    if m != M12_ROWS:
        raise ValueError("compute service probes require the M12 kernel")
    if probe_mode not in {PROBE_FULL_NO_STORE, PROBE_MATRIX_ONLY}:
        raise ValueError(f"unsupported compute probe mode {probe_mode}")
    packed = packed_weights(experts=1, k=k, n=n, seed=seed)
    packed_b = packed.w13[0][:1]
    n_tile = int(packed.backend_n_tile)
    a = torch.zeros((m, k), dtype=torch.bfloat16)
    physical_flops = 2 * m * k * n
    _moe_C.fused_moe_bench_sve_jit_w13_gemm(a, packed_b, k, n, n_tile, 1, 1, 1, probe_mode)
    rows = []
    order = list(widths)
    random.Random(seed).shuffle(order)
    for width in order:
        barrier = threading.Barrier(width)

        def worker(worker_id: int) -> list[float]:
            bind_current_thread(cpu_ids[worker_id])
            barrier.wait()
            return list(
                _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                    a,
                    packed_b,
                    k,
                    n,
                    n_tile,
                    1,
                    warmup,
                    runs,
                    probe_mode,
                )
            )

        with ThreadPoolExecutor(max_workers=width) as executor:
            samples_by_worker = list(executor.map(worker, range(width)))
        slowest_timed_ms = max(sum(samples) for samples in samples_by_worker)
        slowest_median_ms = max(statistics.median(samples) for samples in samples_by_worker)
        rate = width * runs * physical_flops / (slowest_timed_ms * 1e-3)
        row = {
            "threads": width,
            "slowest_worker_median_ms": slowest_median_ms,
            "slowest_worker_timed_ms": slowest_timed_ms,
            "aggregate_rate": rate,
            "aggregate_tflops": rate / 1e12,
            "worker_median_ms": [statistics.median(samples) for samples in samples_by_worker],
        }
        rows.append(row)
        print(
            f"{name:<12} threads={width:>2} rate={row['aggregate_tflops']:>8.3f} TFLOP/s "
            f"median={slowest_median_ms:>8.4f} ms",
            flush=True,
        )
    del packed
    gc.collect()
    rows.sort(key=lambda row: int(row["threads"]))
    if probe_mode == PROBE_FULL_NO_STORE:
        resident_level = cache_level or "cache"
        probe_name = f"generated_sve_m12_{resident_level}_hot_full_no_store"
    else:
        probe_name = "generated_sve_m12_register_only_bfmmla"
    payload = {
        "m": m,
        "k": k,
        "n": n,
        "a_bytes": m * k * 2,
        "packed_b_bytes": k * n * 2,
        "working_set_bytes": (m + n) * k * 2 if probe_mode == PROBE_FULL_NO_STORE else 0,
        "probe": probe_name,
        "probe_mode": probe_mode,
        "point_order": order,
        "rows": rows,
    }
    if cache_geometry is not None:
        payload.update(
            {
                "cache_level": cache_level,
                "target_bytes": cache_geometry.target_bytes,
                "cache_bytes": cache_geometry.cache_bytes,
                "cache_fraction": cache_geometry.cache_fraction,
            }
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=parse_cpu_ids("0-95"))
    parser.add_argument("--widths", type=parse_int_list, default=parse_int_list("1,2,4,8,16,24,32,48,64,96"))
    parser.add_argument("--matrix-warmup", type=int, default=16)
    parser.add_argument("--matrix-runs", type=int, default=64)
    parser.add_argument("--core-warmup", type=int, default=64)
    parser.add_argument("--core-runs", type=int, default=4096)
    parser.add_argument("--hot-warmup", type=int, default=32)
    parser.add_argument("--hot-runs", type=int, default=128)
    parser.add_argument("--dram-warmup", type=int, default=4)
    parser.add_argument("--dram-runs", type=int, default=32)
    parser.add_argument("--n-tile", type=int, default=DEFAULT_N_TILE)
    parser.add_argument("--gemm-l1-fraction", type=float, default=0.625)
    parser.add_argument("--gemm-l2-fraction", type=float, default=0.5)
    parser.add_argument("--load-l1-fraction", type=float, default=0.5)
    parser.add_argument("--load-l2-fraction", type=float, default=0.5)
    parser.add_argument("--load-llc-fraction", type=float, default=1.0 / 6.0)
    parser.add_argument("--dram-weight-l2-multiple", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if max(args.widths) > len(args.cpu_ids):
        raise ValueError("largest width exceeds the supplied CPU list")
    if min(args.matrix_warmup, args.core_warmup, args.hot_warmup, args.dram_warmup) < 0:
        raise ValueError("warmup counts must be non-negative")
    if min(args.matrix_runs, args.core_runs, args.hot_runs, args.dram_runs) <= 0:
        raise ValueError("run counts must be positive")
    if args.n_tile <= 0:
        raise ValueError("n_tile must be positive")
    if args.dram_weight_l2_multiple <= 0.0:
        raise ValueError("dram_weight_l2_multiple must be positive")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    caches = read_cache_info(args.cpu_ids[0])
    packed_panel_columns = 2 * args.n_tile
    l1_gemm = m12_gemm_geometry(
        caches["l1d_bytes_per_core"],
        packed_panel_columns,
        cache_fraction=args.gemm_l1_fraction,
    )
    l2_gemm = m12_gemm_geometry(
        caches["l2_bytes_per_core"],
        packed_panel_columns,
        cache_fraction=args.gemm_l2_fraction,
    )
    l1_load = b_only_geometry(
        caches["l1d_bytes_per_core"],
        packed_panel_columns,
        cache_fraction=args.load_l1_fraction,
    )
    l2_load = b_only_geometry(
        caches["l2_bytes_per_core"],
        packed_panel_columns,
        cache_fraction=args.load_l2_fraction,
    )
    llc_load = b_only_geometry(
        caches["llc_bytes_per_rank"],
        packed_panel_columns,
        cache_fraction=args.load_llc_fraction,
    )
    dram_weight = b_only_geometry(
        int(caches["l2_bytes_per_core"] * args.dram_weight_l2_multiple),
        packed_panel_columns,
        cache_fraction=1.0,
    )
    services = {
        "matrix_flops": profile_matrix(
            name="bfmmla",
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            m=M12_ROWS,
            k=4096,
            n=512,
            warmup=args.matrix_warmup,
            runs=args.matrix_runs,
            seed=args.seed,
            probe_mode=PROBE_MATRIX_ONLY,
        ),
        "gemm_core_flops": profile_matrix(
            name="gemm_l1",
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            m=M12_ROWS,
            k=l1_gemm.k,
            n=l1_gemm.n,
            warmup=args.core_warmup,
            runs=args.core_runs,
            seed=args.seed + 1,
            probe_mode=PROBE_FULL_NO_STORE,
            cache_level="l1d",
            cache_geometry=l1_gemm,
        ),
        "gemm_l2_flops": profile_matrix(
            name="gemm_l2",
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            m=M12_ROWS,
            k=l2_gemm.k,
            n=l2_gemm.n,
            warmup=args.hot_warmup,
            runs=args.hot_runs,
            seed=args.seed + 2,
            probe_mode=PROBE_FULL_NO_STORE,
            cache_level="l2",
            cache_geometry=l2_gemm,
        ),
        "l1_bytes": profile_load_resource(
            name="l1_bytes",
            geometry=l1_load,
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            warmup=args.hot_warmup,
            runs=args.hot_runs,
            seed=args.seed + 3,
            cold=False,
            minimum_cold_experts=1,
            cold_experts_per_thread=1,
        ),
        "l2_bytes": profile_load_resource(
            name="l2_bytes",
            geometry=l2_load,
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            warmup=args.hot_warmup,
            runs=args.hot_runs,
            seed=args.seed + 4,
            cold=False,
            minimum_cold_experts=1,
            cold_experts_per_thread=1,
        ),
        "llc_bytes": profile_load_resource(
            name="llc_bytes",
            geometry=llc_load,
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            warmup=args.hot_warmup,
            runs=args.hot_runs,
            seed=args.seed + 5,
            cold=False,
            minimum_cold_experts=1,
            cold_experts_per_thread=1,
        ),
        "dram_bytes": profile_load_resource(
            name="dram_bytes",
            geometry=dram_weight,
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            warmup=args.dram_warmup,
            runs=args.dram_runs,
            seed=args.seed + 6,
            cold=True,
            minimum_cold_experts=64,
            cold_experts_per_thread=4,
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": "moe_analytic_service_probe",
        "machine": {
            "id": platform.node(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "cpu_ids": args.cpu_ids,
            "cores_per_rank": len(args.cpu_ids),
        },
        "kernel": {
            "sve_implementation": "jit",
            "probe_isa": "sve_bf16",
            "gemm_core_probe": "m12_l1_hot_full_no_store",
            "packed_panel_columns": packed_panel_columns,
            "packed_b_element_bytes": 2,
            "bfmmla_flops_per_instruction": 32,
            "bfmmla_instructions_per_cycle": 4,
            "frontend_instructions_per_cycle": 5,
        },
        "caches": caches,
        "measurement": {
            "matrix_warmup": args.matrix_warmup,
            "matrix_runs": args.matrix_runs,
            "core_warmup": args.core_warmup,
            "core_runs": args.core_runs,
            "hot_warmup": args.hot_warmup,
            "hot_runs": args.hot_runs,
            "dram_warmup": args.dram_warmup,
            "dram_runs": args.dram_runs,
            "thread_pinning": "explicit_cpu_ids",
            "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
            "cache_geometry_source": "linux_sysfs",
            "gemm_l1_fraction": args.gemm_l1_fraction,
            "gemm_l2_fraction": args.gemm_l2_fraction,
            "load_l1_fraction": args.load_l1_fraction,
            "load_l2_fraction": args.load_l2_fraction,
            "load_llc_fraction": args.load_llc_fraction,
            "dram_weight_l2_multiple": args.dram_weight_l2_multiple,
        },
        "services": services,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
