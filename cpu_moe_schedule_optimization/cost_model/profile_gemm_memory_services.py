#!/usr/bin/env python3
"""Measure GEMM-consumable bandwidth for three A/B cache states.

The probe uses the production M12 full-no-store JIT body. Each worker owns a
disjoint operand window. A native pre-kernel SIGSTOP lets the parent release all
worker processes together after A packing and output allocation have completed,
so setup traffic is excluded from the concurrent service window.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import random
import signal
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analytic_probe_geometry import m12_gemm_geometry, read_cache_info  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from profile_analytic_services import (  # noqa: E402
    DEFAULT_N_TILE,
    M12_ROWS,
    PROBE_FULL_NO_STORE,
    packed_weights,
    parse_cpu_ids,
    parse_int_list,
)


@dataclass(frozen=True)
class MemoryState:
    name: str
    stream_a: bool
    stream_b: bool


@dataclass(frozen=True)
class StreamWindow:
    copies_per_worker: int
    timed_scans: int
    stream_bytes_per_scan: int
    target_bytes_per_worker: int
    aggregate_window_bytes: int


MEMORY_STATES = (
    MemoryState("a_hot_b_stream", stream_a=False, stream_b=True),
    MemoryState("a_stream_b_hot", stream_a=True, stream_b=False),
    MemoryState("a_b_stream", stream_a=True, stream_b=True),
)


def plan_stream_window(
    *,
    width: int,
    a_bytes: int,
    b_bytes: int,
    state: MemoryState,
    warmup: int,
    minimum_timed_scans: int,
    l2_bytes_per_core: int,
    llc_bytes_per_rank: int,
    private_cache_multiple: float,
    shared_cache_multiple: float,
) -> StreamWindow:
    """Choose a no-reuse window that evicts private L2 and the shared LLC."""
    if min(width, a_bytes, b_bytes, minimum_timed_scans, l2_bytes_per_core, llc_bytes_per_rank) <= 0:
        raise ValueError("stream geometry and cache sizes must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if min(private_cache_multiple, shared_cache_multiple) <= 0.0:
        raise ValueError("cache multiples must be positive")
    stream_bytes = (a_bytes if state.stream_a else 0) + (b_bytes if state.stream_b else 0)
    if stream_bytes <= 0:
        raise ValueError("at least one operand must stream")
    private_target = math.ceil(private_cache_multiple * l2_bytes_per_core)
    shared_target = math.ceil(shared_cache_multiple * llc_bytes_per_rank / width)
    target_per_worker = max(private_target, shared_target)
    copies = max(warmup + minimum_timed_scans, math.ceil(target_per_worker / stream_bytes))
    timed_scans = copies - warmup
    return StreamWindow(
        copies_per_worker=copies,
        timed_scans=timed_scans,
        stream_bytes_per_scan=stream_bytes,
        target_bytes_per_worker=target_per_worker,
        aggregate_window_bytes=width * copies * stream_bytes,
    )


def _worker(
    connection,
    cpu: int,
    a: torch.Tensor,
    packed_b: torch.Tensor,
    k: int,
    n: int,
    n_tile: int,
    warmup: int,
    runs: int,
    probe_mode: int,
    n_ranges: int,
) -> None:
    try:
        os.sched_setaffinity(0, {cpu})
        os.environ["FUSED_CPP_MOE_BENCH_STOP_AFTER_SETUP"] = "1"
        samples = list(
            _moe_C.fused_moe_bench_sve_jit_w13_gemm(
                a,
                packed_b,
                k,
                n,
                n_tile,
                n_ranges,
                warmup,
                runs,
                probe_mode,
            )
        )
        connection.send(("ok", samples))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        os.environ.pop("FUSED_CPP_MOE_BENCH_STOP_AFTER_SETUP", None)
        connection.close()


def _terminate_workers(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def concurrent_probe(
    *,
    a: torch.Tensor,
    packed_b: torch.Tensor,
    state: MemoryState,
    width: int,
    copies_per_worker: int,
    cpu_ids: list[int],
    k: int,
    n: int,
    n_tile: int,
    warmup: int,
    runs: int,
    probe_mode: int = PROBE_FULL_NO_STORE,
    n_ranges: int = 1,
    a_group_size: int | None = None,
) -> list[list[float]]:
    """Run one synchronized process wave and return per-worker scan timings."""
    context = mp.get_context("fork")
    if n_ranges <= 0 or n % n_ranges or (n // n_ranges) % n_tile:
        raise ValueError("n_ranges must divide N into whole N tiles")
    if a_group_size is not None:
        if state.stream_a or a_group_size <= 0 or width % a_group_size:
            raise ValueError("group-shared A requires a positive group size dividing width and a hot-A state")
        if a.dim() != 3 or a.size(0) != width // a_group_size:
            raise ValueError("group-shared A must have shape [width / group_size, M, K]")
    processes: list[mp.Process] = []
    parents = []
    for worker_id in range(width):
        if state.stream_a:
            worker_a = a[worker_id]
        elif a_group_size is not None:
            worker_a = a[worker_id // a_group_size]
        else:
            worker_a = a
        b_begin = worker_id * copies_per_worker if state.stream_b else worker_id
        b_count = copies_per_worker if state.stream_b else 1
        worker_b = packed_b[b_begin : b_begin + b_count]
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker,
            args=(
                child,
                cpu_ids[worker_id],
                worker_a,
                worker_b,
                k,
                n,
                n_tile,
                warmup,
                runs,
                probe_mode,
                n_ranges,
            ),
        )
        process.start()
        child.close()
        processes.append(process)
        parents.append(parent)

    try:
        for worker_id, process in enumerate(processes):
            try:
                pid, status = os.waitpid(process.pid, os.WUNTRACED)
            except ChildProcessError as exc:
                process.join(timeout=0.0)
                detail = f"exitcode={process.exitcode}"
                if parents[worker_id].poll():
                    child_status, payload = parents[worker_id].recv()
                    detail += f" child_status={child_status} payload={payload}"
                raise RuntimeError(
                    f"worker {worker_id} pid={process.pid} exited before SIGSTOP ({detail})"
                ) from exc
            if pid != process.pid or not os.WIFSTOPPED(status):
                raise RuntimeError(
                    f"worker {worker_id} pid={process.pid} exited before the synchronized kernel window: "
                    f"wait_status={status} exitcode={process.exitcode}"
                )
        for process in processes:
            os.kill(process.pid, signal.SIGCONT)

        samples_by_worker: list[list[float]] = []
        for worker_id, connection in enumerate(parents):
            if not connection.poll(120.0):
                raise TimeoutError(f"worker {worker_id} did not return benchmark samples")
            status, payload = connection.recv()
            if status != "ok":
                raise RuntimeError(f"worker {worker_id} failed: {payload}")
            samples_by_worker.append([float(sample) for sample in payload])
        for process in processes:
            process.join(timeout=30.0)
            if process.exitcode != 0:
                raise RuntimeError(f"worker {process.pid} exited with code {process.exitcode}")
        return samples_by_worker
    except BaseException:
        _terminate_workers(processes)
        raise
    finally:
        for connection in parents:
            connection.close()


def _allocate_operands(
    *,
    state: MemoryState,
    width: int,
    copies_per_worker: int,
    m: int,
    k: int,
    n: int,
    seed: int,
):
    a_shape = (width, copies_per_worker, m, k) if state.stream_a else (m, k)
    a = torch.zeros(a_shape, dtype=torch.bfloat16)
    b_experts = width * copies_per_worker if state.stream_b else width
    packed = packed_weights(experts=b_experts, k=k, n=n, seed=seed)
    return a, packed


def profile_state(
    *,
    state: MemoryState,
    widths: list[int],
    cpu_ids: list[int],
    m: int,
    k: int,
    n: int,
    warmup: int,
    minimum_timed_scans: int,
    repeats: int,
    caches: dict[str, int],
    private_cache_multiple: float,
    shared_cache_multiple: float,
    seed: int,
) -> dict[str, object]:
    a_bytes = m * k * 2
    b_bytes = k * n * 2
    physical_flops = 2 * m * k * n
    order = list(widths)
    random.Random(seed).shuffle(order)
    rows = []
    for point_index, width in enumerate(order):
        window = plan_stream_window(
            width=width,
            a_bytes=a_bytes,
            b_bytes=b_bytes,
            state=state,
            warmup=warmup,
            minimum_timed_scans=minimum_timed_scans,
            l2_bytes_per_core=caches["l2_bytes_per_core"],
            llc_bytes_per_rank=caches["llc_bytes_per_rank"],
            private_cache_multiple=private_cache_multiple,
            shared_cache_multiple=shared_cache_multiple,
        )
        print(
            f"allocate state={state.name} threads={width} copies={window.copies_per_worker} "
            f"window={window.aggregate_window_bytes / 2**20:.1f} MiB",
            flush=True,
        )
        a, packed = _allocate_operands(
            state=state,
            width=width,
            copies_per_worker=window.copies_per_worker,
            m=m,
            k=k,
            n=n,
            seed=seed + point_index,
        )
        packed_b = packed.w13[0]
        n_tile = int(packed.backend_n_tile)
        if n % n_tile:
            raise ValueError(f"probe N={n} must be divisible by runtime n_tile={n_tile}")

        # Generate the JIT body in the parent before forking worker waves.
        parent_a = a[0, :1] if state.stream_a else a
        _moe_C.fused_moe_bench_sve_jit_w13_gemm(
            parent_a,
            packed_b[:1],
            k,
            n,
            n_tile,
            1,
            1,
            1,
            PROBE_FULL_NO_STORE,
        )

        samples = []
        for repeat in range(repeats):
            worker_samples = concurrent_probe(
                a=a,
                packed_b=packed_b,
                state=state,
                width=width,
                copies_per_worker=window.copies_per_worker,
                cpu_ids=cpu_ids,
                k=k,
                n=n,
                n_tile=n_tile,
                warmup=warmup,
                runs=window.timed_scans,
            )
            slowest_ms = max(sum(worker) for worker in worker_samples)
            stream_bytes = width * window.timed_scans * window.stream_bytes_per_scan
            flops = width * window.timed_scans * physical_flops
            rate = stream_bytes / (slowest_ms * 1e-3)
            sample = {
                "repeat": repeat,
                "aggregate_rate": rate,
                "aggregate_gbytes_per_second": rate / 1e9,
                "aggregate_tflops": flops / (slowest_ms * 1e9),
                "slowest_worker_timed_ms": slowest_ms,
                "worker_timed_ms": [sum(worker) for worker in worker_samples],
            }
            samples.append(sample)
            print(
                f"{state.name:<16} threads={width:>2} repeat={repeat} "
                f"rate={sample['aggregate_gbytes_per_second']:>8.2f} GB/s "
                f"gemm={sample['aggregate_tflops']:>7.3f} TFLOP/s",
                flush=True,
            )

        rates = [float(sample["aggregate_rate"]) for sample in samples]
        tflops = [float(sample["aggregate_tflops"]) for sample in samples]
        rows.append(
            {
                "threads": width,
                **asdict(window),
                "median_rate": statistics.median(rates),
                "median_gbytes_per_second": statistics.median(rates) / 1e9,
                "min_gbytes_per_second": min(rates) / 1e9,
                "max_gbytes_per_second": max(rates) / 1e9,
                "median_tflops": statistics.median(tflops),
                "samples": samples,
            }
        )
        del packed_b, packed, a
        gc.collect()

    rows.sort(key=lambda row: int(row["threads"]))
    return {
        "state": asdict(state),
        "a_bytes_per_scan": a_bytes,
        "b_bytes_per_scan": b_bytes,
        "physical_flops_per_scan": physical_flops,
        "point_order": order,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", type=parse_cpu_ids, default=parse_cpu_ids("0-95"))
    parser.add_argument("--widths", type=parse_int_list, default=parse_int_list("1,2,4,8,12,16,24,32,48,64,80,96"))
    parser.add_argument("--n-tile", type=int, default=DEFAULT_N_TILE)
    parser.add_argument("--l1-fraction", type=float, default=0.625)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--minimum-timed-scans", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--private-cache-multiple", type=float, default=8.0)
    parser.add_argument("--shared-cache-multiple", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("the synchronized process probe requires Linux SIGSTOP/SIGCONT")
    if max(args.widths) > len(args.cpu_ids):
        raise ValueError("largest width exceeds the supplied CPU list")
    if min(args.n_tile, args.minimum_timed_scans, args.repeats) <= 0 or args.warmup < 0:
        raise ValueError("tile, scan, and repeat counts must be positive; warmup must be non-negative")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    caches = read_cache_info(args.cpu_ids[0])
    geometry = m12_gemm_geometry(
        caches["l1d_bytes_per_core"],
        2 * args.n_tile,
        cache_fraction=args.l1_fraction,
    )
    services = {}
    for state_index, state in enumerate(MEMORY_STATES):
        services[state.name] = profile_state(
            state=state,
            widths=args.widths,
            cpu_ids=args.cpu_ids,
            m=M12_ROWS,
            k=geometry.k,
            n=geometry.n,
            warmup=args.warmup,
            minimum_timed_scans=args.minimum_timed_scans,
            repeats=args.repeats,
            caches=caches,
            private_cache_multiple=args.private_cache_multiple,
            shared_cache_multiple=args.shared_cache_multiple,
            seed=args.seed + 1000 * state_index,
        )

    payload = {
        "schema_version": 1,
        "kind": "moe_gemm_memory_service_probe",
        "machine": {
            "id": platform.node(),
            "architecture": platform.machine(),
            "cpu_ids": args.cpu_ids,
            "cores_per_rank": len(args.cpu_ids),
        },
        "kernel": {
            "implementation": "sve_jit_m12_full_no_store",
            "m": M12_ROWS,
            "k": geometry.k,
            "n": geometry.n,
            "a_plus_b_bytes": geometry.working_set_bytes,
            "l1_fraction": args.l1_fraction,
        },
        "caches": caches,
        "measurement": {
            "warmup": args.warmup,
            "minimum_timed_scans": args.minimum_timed_scans,
            "repeats": args.repeats,
            "private_cache_multiple": args.private_cache_multiple,
            "shared_cache_multiple": args.shared_cache_multiple,
            "worker_model": "forked_processes_native_pre_kernel_sigstop",
            "thread_pinning": "explicit_cpu_ids",
            "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
        },
        "services": services,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
