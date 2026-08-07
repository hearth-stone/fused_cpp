#!/usr/bin/env python3
"""Validate the four-state predictor against synchronized full-M pure GEMMs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fused_cpp import _moe_C  # noqa: E402
from profile_analytic_services import PROBE_FULL_NO_STORE, packed_weights  # noqa: E402
from profile_gemm_memory_services import MemoryState, concurrent_probe  # noqa: E402


REFERENCE_M = 12
REFERENCE_K = 728
REFERENCE_N_TILE = 8


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def row_at_width(payload: dict, service: str, width: int) -> dict:
    rows = payload["services"][service]["rows"]
    for row in rows:
        if int(row["threads"]) == width:
            return row
    raise ValueError(f"service {service!r} has no calibration row for width={width}")


def wave_us(tflops: float, *, width: int, m: int, k: int, n: int) -> float:
    """Convert aggregate throughput to one synchronized worker-wave latency."""
    return width * 2 * m * k * n / (tflops * 1e12) * 1e6


def derive_state_costs(memory_payload: dict, hot_payload: dict, width: int) -> dict[str, float]:
    """Recover per-N8-tile four-state costs from two-tile service probes."""
    state_tflops = {
        name: float(row_at_width(memory_payload, name, width)["median_tflops"])
        for name in ("a_hot_b_stream", "a_stream_b_hot", "a_b_stream")
    }
    hot_l1 = float(row_at_width(hot_payload, "gemm_core_flops", width)["aggregate_tflops"])
    hot_l2 = float(row_at_width(hot_payload, "gemm_l2_flops", width)["aggregate_tflops"])
    full_scan = {
        name: wave_us(
            tflops,
            width=width,
            m=REFERENCE_M,
            k=REFERENCE_K,
            n=2 * REFERENCE_N_TILE,
        )
        for name, tflops in state_tflops.items()
    }
    hot_a_cold_b = full_scan["a_hot_b_stream"] / 2.0
    hot_hot_l1 = wave_us(
        hot_l1,
        width=width,
        m=REFERENCE_M,
        k=REFERENCE_K,
        n=REFERENCE_N_TILE,
    )
    return {
        "cold_a_cold_b": full_scan["a_b_stream"] - hot_a_cold_b,
        "hot_a_cold_b": hot_a_cold_b,
        "cold_a_hot_b": full_scan["a_stream_b_hot"] - hot_hot_l1,
        "hot_a_hot_b_l1": hot_hot_l1,
        "hot_a_hot_b_l2": wave_us(
            hot_l2,
            width=width,
            m=REFERENCE_M,
            k=REFERENCE_K,
            n=REFERENCE_N_TILE,
        ),
    }


def stage_prediction(
    state_us: dict[str, float],
    *,
    m: int,
    k: int,
    q_tiles_per_worker: int,
    windows: int,
) -> dict[str, object]:
    """Predict one stage from its exact M-panel/N-tile state counts."""
    if m <= 0 or m % REFERENCE_M:
        raise ValueError("validation M must be a positive multiple of 12")
    if q_tiles_per_worker <= 0 or windows <= 0:
        raise ValueError("Q tiles and windows must be positive")
    p_panels = m // REFERENCE_M
    counts = {
        "cold_a_cold_b": windows,
        "hot_a_cold_b": windows * (q_tiles_per_worker - 1),
        "cold_a_hot_b": windows * (p_panels - 1),
        "hot_a_hot_b_l2": windows * (p_panels - 1) * (q_tiles_per_worker - 1),
    }
    scale = k / REFERENCE_K
    contributions_us = {name: count * state_us[name] * scale for name, count in counts.items()}
    return {
        "p_panels": p_panels,
        "q_tiles_per_worker": q_tiles_per_worker,
        "windows": windows,
        "counts": counts,
        "contributions_us": contributions_us,
        "predicted_ms": sum(contributions_us.values()) / 1000.0,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def measure_stage(
    *,
    name: str,
    cpu_ids: list[int],
    m_values: list[int],
    k: int,
    n_per_worker: int,
    windows: int,
    waves: int,
    experts: int,
    seed: int,
) -> dict[str, object]:
    """Measure each M with disjoint never-reused packed-B worker stripes."""
    width = len(cpu_ids)
    required_experts = width * waves * len(m_values) + 1
    if experts < required_experts:
        raise ValueError(f"{name} requires at least {required_experts} experts")
    print(f"allocate {name}: E={experts} K={k} N/thread={n_per_worker}", flush=True)
    packed = packed_weights(experts=experts, k=k, n=n_per_worker, seed=seed)
    packed_b = packed.w13[0]
    n_tile = int(packed.backend_n_tile)
    state = MemoryState("full_m_a_reused_b_cold", stream_a=False, stream_b=True)
    results = {}

    for m_index, m in enumerate(m_values):
        a = torch.zeros((m, k), dtype=torch.bfloat16)
        _moe_C.fused_moe_bench_sve_jit_w13_gemm(
            a,
            packed_b[-1:],
            k,
            n_per_worker,
            n_tile,
            1,
            0,
            1,
            PROBE_FULL_NO_STORE,
        )
        wave_ms = []
        for wave in range(waves):
            begin = (m_index * waves + wave) * width
            samples = concurrent_probe(
                a=a,
                packed_b=packed_b[begin : begin + width],
                state=state,
                width=width,
                copies_per_worker=1,
                cpu_ids=cpu_ids,
                k=k,
                n=n_per_worker,
                n_tile=n_tile,
                warmup=0,
                runs=1,
                probe_mode=PROBE_FULL_NO_STORE,
            )
            wave_ms.append(max(worker[0] for worker in samples) * windows)
        median_ms = statistics.median(wave_ms)
        results[str(m)] = {
            "samples_ms": wave_ms,
            "median_ms": median_ms,
            "p10_ms": percentile(wave_ms, 0.10),
            "p90_ms": percentile(wave_ms, 0.90),
            "min_ms": min(wave_ms),
            "max_ms": max(wave_ms),
        }
        print(
            f"{name:<4} M={m:>4} median={median_ms:8.4f} ms "
            f"p10={results[str(m)]['p10_ms']:8.4f} p90={results[str(m)]['p90_ms']:8.4f}",
            flush=True,
        )
        del a

    del packed_b, packed
    gc.collect()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-services", type=Path, required=True)
    parser.add_argument("--hot-services", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-ids", type=parse_int_list, default=parse_int_list("0,1,2,3"))
    parser.add_argument("--m-values", type=parse_int_list, default=parse_int_list("12,24,48,192,768,2040"))
    parser.add_argument("--waves", type=int, default=11)
    parser.add_argument("--experts", type=int, default=320)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("the synchronized process validator requires Linux SIGSTOP/SIGCONT")
    if len(args.cpu_ids) != 4:
        raise ValueError("the TP4 validation geometry requires exactly four CPU IDs")
    if any(m <= 0 or m % REFERENCE_M for m in args.m_values):
        raise ValueError("every M must be a positive multiple of 12")
    if args.waves <= 0 or args.experts <= 0:
        raise ValueError("waves and experts must be positive")

    torch.set_num_threads(1)
    os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
    memory_payload = json.loads(args.memory_services.read_text(encoding="utf-8"))
    hot_payload = json.loads(args.hot_services.read_text(encoding="utf-8"))
    state_us = derive_state_costs(memory_payload, hot_payload, len(args.cpu_ids))
    measured = {
        "w13": measure_stage(
            name="w13",
            cpu_ids=args.cpu_ids,
            m_values=args.m_values,
            k=4096,
            n_per_worker=128,
            windows=2,
            waves=args.waves,
            experts=args.experts,
            seed=20260805,
        ),
        "w2": measure_stage(
            name="w2",
            cpu_ids=args.cpu_ids,
            m_values=args.m_values,
            k=512,
            n_per_worker=1024,
            windows=1,
            waves=args.waves,
            experts=args.experts,
            seed=20260806,
        ),
    }

    rows = []
    for m in args.m_values:
        predicted = {
            "w13": stage_prediction(state_us, m=m, k=4096, q_tiles_per_worker=16, windows=2),
            "w2": stage_prediction(state_us, m=m, k=512, q_tiles_per_worker=128, windows=1),
        }
        row = {"m": m, "stages": {}}
        for stage in ("w13", "w2"):
            predicted_ms = float(predicted[stage]["predicted_ms"])
            measured_ms = float(measured[stage][str(m)]["median_ms"])
            row["stages"][stage] = {
                "prediction": predicted[stage],
                "measurement": measured[stage][str(m)],
                "error_percent": (predicted_ms / measured_ms - 1.0) * 100.0,
            }
        predicted_total = sum(float(predicted[stage]["predicted_ms"]) for stage in ("w13", "w2"))
        measured_total = sum(float(measured[stage][str(m)]["median_ms"]) for stage in ("w13", "w2"))
        row["total"] = {
            "predicted_ms": predicted_total,
            "measured_ms": measured_total,
            "error_percent": (predicted_total / measured_total - 1.0) * 100.0,
        }
        rows.append(row)
        print(
            f"total M={m:>4}: pred={predicted_total:8.4f} ms measured={measured_total:8.4f} ms "
            f"error={row['total']['error_percent']:+7.2f}%",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "kind": "pure_gemm_four_state_validation",
        "machine": {
            "cpu_ids": args.cpu_ids,
            "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
        },
        "reference": {
            "m": REFERENCE_M,
            "k": REFERENCE_K,
            "n_tile": REFERENCE_N_TILE,
            "state_wave_us": state_us,
            "memory_services": str(args.memory_services),
            "hot_services": str(args.hot_services),
        },
        "measurement": {
            "probe_mode": PROBE_FULL_NO_STORE,
            "waves_per_stage_and_m": args.waves,
            "worker_model": "four_forked_processes_native_pre_kernel_sigstop",
            "weight_policy": "one_disjoint_never-reused expert per worker wave",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
