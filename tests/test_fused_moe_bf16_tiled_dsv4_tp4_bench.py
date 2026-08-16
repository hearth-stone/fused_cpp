# -*- coding: utf-8 -*-
"""DeepSeek V4 Flash TP=4 MoE benchmark with vLLM-like CPU binding."""

from __future__ import annotations

import multiprocessing as mp
import json
import os
import platform
import queue
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from fused_cpp.moe import _HAS_BF16_TILED_FUSED_MOE
from fused_cpp.moe import fused_moe_bf16_tiled
from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights


pytestmark = [
    pytest.mark.bench,
    pytest.mark.skipif(
        platform.machine() not in ("aarch64", "arm64")
        or platform.system() != "Linux"
        or not hasattr(os, "sched_setaffinity")
        or not _HAS_BF16_TILED_FUSED_MOE,
        reason=(
            "DeepSeek V4 TP=4 MoE benchmark requires Linux AArch64, sched_setaffinity, and the fused_cpp C++ extension"
        ),
    ),
]


@dataclass(frozen=True)
class DSV4TP4MoEBenchConfig:
    tokens: int = 2048
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    tp_size: int = 4
    experts: int = 256
    top_k: int = 6
    threads_per_rank: int = 64
    cores_per_numa: int = 80
    core_skip: int = 40
    groups_per_partition: int = 4
    warmup: int = 1
    runs: int = 3
    seed: int = 20260624
    std: float = 0.01
    activation: str = "silu"
    barrier_timeout_s: float = 900.0
    prepack_threads: int = 1
    routing_dir: str | None = None
    routing_seq: int = 0

    @property
    def ffn_per_rank(self) -> int:
        if self.moe_intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size must be divisible by tp_size: {self.moe_intermediate_size=} {self.tp_size=}"
            )
        return self.moe_intermediate_size // self.tp_size


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _config_from_env() -> DSV4TP4MoEBenchConfig:
    defaults = DSV4TP4MoEBenchConfig()
    return DSV4TP4MoEBenchConfig(
        tokens=_env_int("FUSED_CPP_DSV4_MOE_BENCH_TOKENS", defaults.tokens),
        hidden_size=_env_int("FUSED_CPP_DSV4_MOE_BENCH_HIDDEN_SIZE", defaults.hidden_size),
        moe_intermediate_size=_env_int(
            "FUSED_CPP_DSV4_MOE_BENCH_MOE_INTERMEDIATE_SIZE",
            defaults.moe_intermediate_size,
        ),
        tp_size=_env_int("FUSED_CPP_DSV4_MOE_BENCH_TP_SIZE", defaults.tp_size),
        experts=_env_int("FUSED_CPP_DSV4_MOE_BENCH_EXPERTS", defaults.experts),
        top_k=_env_int("FUSED_CPP_DSV4_MOE_BENCH_TOP_K", defaults.top_k),
        threads_per_rank=_env_int(
            "FUSED_CPP_DSV4_MOE_BENCH_THREADS_PER_RANK",
            defaults.threads_per_rank,
        ),
        cores_per_numa=_env_int(
            "FUSED_CPP_DSV4_MOE_BENCH_CORES_PER_NUMA",
            defaults.cores_per_numa,
        ),
        core_skip=_env_int("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", defaults.core_skip),
        groups_per_partition=_env_int(
            "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
            defaults.groups_per_partition,
        ),
        warmup=_env_int("FUSED_CPP_DSV4_MOE_BENCH_WARMUP", defaults.warmup),
        runs=_env_int("FUSED_CPP_DSV4_MOE_BENCH_RUNS", defaults.runs),
        seed=_env_int("FUSED_CPP_DSV4_MOE_BENCH_SEED", defaults.seed),
        std=_env_float("FUSED_CPP_DSV4_MOE_BENCH_STD", defaults.std),
        activation=os.environ.get("FUSED_CPP_DSV4_MOE_BENCH_ACTIVATION", defaults.activation),
        barrier_timeout_s=_env_float(
            "FUSED_CPP_DSV4_MOE_BENCH_TIMEOUT_S",
            defaults.barrier_timeout_s,
        ),
        prepack_threads=_env_int("FUSED_CPP_MOE_PREPACK_THREADS", defaults.prepack_threads),
    )


def _rank_cpus(rank: int, config: DSV4TP4MoEBenchConfig) -> list[int]:
    base = rank * config.cores_per_numa
    return [
        *range(base, base + 32),
        *range(base + config.core_skip, base + config.core_skip + 32),
    ]


def _format_cpu_list(cpus: list[int]) -> str:
    if not cpus:
        return ""

    ranges: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = cpu
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _set_vllm_like_rank_env(rank: int, config: DSV4TP4MoEBenchConfig) -> list[int]:
    cpus = _rank_cpus(rank, config)
    os.sched_setaffinity(0, set(cpus))
    cpu_list = ",".join(str(cpu) for cpu in cpus)
    os.environ["OMP_NUM_THREADS"] = str(config.threads_per_rank)
    os.environ["OMP_PLACES"] = "{" + cpu_list + "}"
    os.environ["OMP_PROC_BIND"] = "true"
    os.environ["FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT"] = "1"
    os.environ["FUSED_CPP_MOE_N_SPLIT_CORE_SKIP"] = str(config.core_skip)
    os.environ["FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION"] = str(config.groups_per_partition)
    os.environ["FUSED_CPP_MOE_PREPACK_THREADS"] = str(config.prepack_threads)
    torch.set_num_threads(config.threads_per_rank)
    return cpus


def _bf16_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    std: float,
) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def _make_topk(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.rand(num_tokens, num_experts, generator=generator)
    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1)
    return torch.softmax(topk_weights, dim=-1), topk_ids.to(torch.int32)


def _load_moe_routing_entry(
    routing_dir: str,
    *,
    rank: int,
    seq: int,
) -> dict[str, Any]:
    path = Path(routing_dir) / f"moe_routing_rank{rank}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing routing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if int(entry["seq"]) == seq:
                return entry
    raise ValueError(f"missing routing seq={seq} in {path}")


def _routing_counts_from_entry(entry: dict[str, Any]) -> list[tuple[int, int]]:
    tokens = int(entry["tokens"])
    top_k = int(entry["top_k"])
    routes = int(entry["routes"])
    if routes != tokens * top_k:
        raise ValueError(f"routing routes must equal tokens * top_k, got {routes=} {tokens=} {top_k=}")

    num_experts = int(entry["num_experts"])
    active_experts = int(entry["active_experts"])
    top_counts: list[tuple[int, int]] = []
    seen: set[int] = set()
    for item in entry["top_experts"]:
        expert = int(item["expert"])
        count = int(item["routes"])
        if expert in seen:
            raise ValueError(f"duplicate expert in routing entry: {expert}")
        if expert < 0 or expert >= num_experts:
            raise ValueError(f"expert id out of range in routing entry: {expert}")
        if count < 0:
            raise ValueError(f"negative route count in routing entry: {item}")
        seen.add(expert)
        top_counts.append((expert, count))

    top_sum = sum(count for _, count in top_counts)
    if top_sum == routes:
        return top_counts
    if top_sum > routes:
        raise ValueError(f"routing top_experts route sum exceeds routes={routes}")
    if len(top_counts) > active_experts:
        raise ValueError(f"top_experts has more entries than active_experts: {len(top_counts)=} {active_experts=}")

    tail_active = active_experts - len(top_counts)
    if tail_active <= 0:
        raise ValueError(
            f"routing top_experts route sum mismatch: got {top_sum}, "
            f"expected {routes}, but no active tail experts are available"
        )

    tail_experts = [expert for expert in range(num_experts) if expert not in seen][:tail_active]
    if len(tail_experts) != tail_active:
        raise ValueError(f"not enough experts to synthesize routing tail: {tail_active=} {len(tail_experts)=}")

    tail_total = routes - top_sum
    tail_min = int(entry.get("routes_min", 1))
    if tail_min < 0:
        raise ValueError(f"negative routes_min in routing entry: {tail_min}")
    if tail_active == 1:
        tail_counts = [tail_total]
    else:
        tail_counts = [tail_min for _ in range(tail_active)]
        remaining = tail_total - tail_min * tail_active
        if remaining < 0:
            raise ValueError(
                f"routing tail total is too small for routes_min: {tail_total=} {tail_min=} {tail_active=}"
            )
        tail_cap = min(count for _, count in top_counts) - 1
        tail_cap = max(tail_cap, tail_min)
        # Keep one synthesized expert at routes_min, then balance the rest.
        while remaining > 0:
            progressed = False
            for idx in range(1, tail_active):
                if remaining == 0:
                    break
                if tail_counts[idx] >= tail_cap:
                    continue
                tail_counts[idx] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                raise ValueError(f"routing tail cannot fit below top_experts: {tail_total=} {tail_cap=}")

    return top_counts + list(zip(tail_experts, tail_counts, strict=True))


def _make_topk_from_routing_entry(entry: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = int(entry["tokens"])
    top_k = int(entry["top_k"])
    routes = int(entry["routes"])
    routing_counts = _routing_counts_from_entry(entry)
    topk_ids = torch.empty((tokens, top_k), dtype=torch.int32)
    cursor = 0
    for expert, count in routing_counts:
        for offset in range(count):
            flat = cursor + offset
            token = flat % tokens
            slot = flat // tokens
            if slot >= top_k:
                raise ValueError(f"routing top_experts route sum exceeds routes={routes}")
            topk_ids[token, slot] = expert
        cursor += count
    if cursor != routes:
        raise ValueError(f"routing route sum mismatch: got {cursor}, expected {routes}")
    topk_weights = torch.full(
        (tokens, top_k),
        1.0 / float(top_k),
        dtype=torch.float32,
    )
    return topk_weights, topk_ids


def _rank_flops(config: DSV4TP4MoEBenchConfig) -> float:
    routes = config.tokens * config.top_k
    h = config.hidden_size
    f = config.ffn_per_rank
    w13_flops = 2.0 * routes * h * (2 * f)
    w2_flops = 2.0 * routes * f * h
    return w13_flops + w2_flops


def _bench_rank_worker(
    rank: int,
    config: DSV4TP4MoEBenchConfig,
    barrier: Any,
    results: Any,
) -> None:
    try:
        cpus = _set_vllm_like_rank_env(rank, config)
        data_gen = torch.Generator().manual_seed(config.seed + rank)
        route_gen = torch.Generator().manual_seed(config.seed)

        hidden_states = _bf16_normal(
            (config.tokens, config.hidden_size),
            generator=data_gen,
            std=config.std,
        )
        w13_weight = _bf16_normal(
            (config.experts, 2 * config.ffn_per_rank, config.hidden_size),
            generator=data_gen,
            std=config.std,
        )
        w2_weight = _bf16_normal(
            (config.experts, config.hidden_size, config.ffn_per_rank),
            generator=data_gen,
            std=config.std,
        )
        topk_weights, topk_ids = (
            _make_topk(
                config.tokens,
                config.experts,
                config.top_k,
                generator=route_gen,
            )
            if config.routing_dir is None
            else _make_topk_from_routing_entry(
                _load_moe_routing_entry(
                    config.routing_dir,
                    rank=rank,
                    seq=config.routing_seq,
                )
            )
        )

        pack_t0 = time.perf_counter()
        packed = prepare_fused_moe_bf16_tiled_weights(w13_weight, w2_weight)
        pack_s = time.perf_counter() - pack_t0
        del w13_weight, w2_weight

        counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=config.experts)

        def run_once() -> torch.Tensor:
            return fused_moe_bf16_tiled(
                hidden_states,
                packed,
                topk_weights,
                topk_ids,
                num_threads=config.threads_per_rank,
                activation=config.activation,
            )

        for _ in range(config.warmup):
            barrier.wait(timeout=config.barrier_timeout_s)
            out = run_once()
            _ = float(out.flatten()[0])
            barrier.wait(timeout=config.barrier_timeout_s)

        times: list[float] = []
        for _ in range(config.runs):
            barrier.wait(timeout=config.barrier_timeout_s)
            t0 = time.perf_counter()
            out = run_once()
            _ = float(out.flatten()[0])
            elapsed = time.perf_counter() - t0
            barrier.wait(timeout=config.barrier_timeout_s)
            times.append(elapsed)

        allowed = sorted(os.sched_getaffinity(0))
        results.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "cpu_binding": _format_cpu_list(cpus),
                "final_affinity": _format_cpu_list(allowed),
                "routing_source": config.routing_dir or "",
                "routing_seq": config.routing_seq,
                "pack_ms": pack_s * 1e3,
                "times_s": times,
                "routes_min": int(counts.min().item()),
                "routes_max": int(counts.max().item()),
                "routes_mean": float(counts.float().mean().item()),
            }
        )
    except BaseException:
        results.put(
            {
                "rank": rank,
                "pid": os.getpid(),
                "error": traceback.format_exc(),
            }
        )


def _collect_results(
    processes: list[mp.Process],
    results: Any,
    *,
    timeout_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while len(rows) < len(processes):
        remaining = max(0.1, deadline - time.monotonic())
        try:
            rows.append(results.get(timeout=remaining))
        except queue.Empty:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
            raise TimeoutError(f"timed out waiting for {len(processes) - len(rows)} rank results")
    return sorted(rows, key=lambda row: int(row["rank"]))


def _print_report(rows: list[dict[str, Any]], config: DSV4TP4MoEBenchConfig) -> None:
    rank_flops = _rank_flops(config)
    print("DeepSeek V4 Flash fused_cpp MoE TP=4 benchmark")
    print(
        f"tokens={config.tokens} experts={config.experts} top_k={config.top_k} "
        f"tp={config.tp_size} threads_per_rank={config.threads_per_rank} "
        f"activation={config.activation}"
    )
    if config.routing_dir is not None:
        print(f"routing_dir={config.routing_dir} routing_seq={config.routing_seq}")
    print(
        f"H={config.hidden_size} F_per_rank={config.ffn_per_rank} "
        f"w13=[{config.experts},{2 * config.ffn_per_rank},{config.hidden_size}] "
        f"w2=[{config.experts},{config.hidden_size},{config.ffn_per_rank}]"
    )
    print(
        "binding=vllm_like "
        f"cores_per_numa={config.cores_per_numa} core_skip={config.core_skip} "
        f"groups_per_partition={config.groups_per_partition}"
    )
    print(f"rank_flops={rank_flops / 1e9:.3f} GF")

    all_times: list[list[float]] = []
    for row in rows:
        times = [float(t) for t in row["times_s"]]
        all_times.append(times)
        median_s = statistics.median(times)
        best_s = min(times)
        print(
            f"rank={row['rank']} pid={row['pid']} "
            f"cpus={row['cpu_binding']} final_affinity={row['final_affinity']} "
            f"pack_ms={float(row['pack_ms']):.3f} "
            f"routes_min={row['routes_min']} routes_max={row['routes_max']} "
            f"routes_mean={float(row['routes_mean']):.2f} "
            f"median_ms={median_s * 1e3:.3f} best_ms={best_s * 1e3:.3f} "
            f"median_gflops={rank_flops / median_s / 1e9:.3f} "
            f"best_gflops={rank_flops / best_s / 1e9:.3f} "
            f"times_ms={[round(t * 1e3, 3) for t in times]}"
        )

    step_wall_s = [max(times[i] for times in all_times) for i in range(config.runs)]
    median_step_s = statistics.median(step_wall_s)
    best_step_s = min(step_wall_s)
    aggregate_flops = rank_flops * config.tp_size
    print(
        "aggregate_by_slowest_rank "
        f"median_step_ms={median_step_s * 1e3:.3f} "
        f"best_step_ms={best_step_s * 1e3:.3f} "
        f"median_gflops={aggregate_flops / median_step_s / 1e9:.3f} "
        f"best_gflops={aggregate_flops / best_step_s / 1e9:.3f} "
        f"step_wall_ms={[round(t * 1e3, 3) for t in step_wall_s]}"
    )


def _run_tp4_bound_bench(config: DSV4TP4MoEBenchConfig) -> None:
    if config.tp_size != 4:
        pytest.skip("this benchmark models the vLLM TP=4 CPU binding layout")
    if config.top_k > config.experts:
        pytest.fail(f"top_k={config.top_k} cannot exceed experts={config.experts}")

    required_max_cpu = max(max(_rank_cpus(rank, config)) for rank in range(4))
    if (os.cpu_count() or 0) <= required_max_cpu:
        pytest.skip(f"need CPU id {required_max_cpu}, but os.cpu_count()={os.cpu_count()}")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(config.tp_size, timeout=config.barrier_timeout_s)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_bench_rank_worker,
            args=(rank, config, barrier, results),
            name=f"fused_cpp_moe_tp{rank}",
        )
        for rank in range(config.tp_size)
    ]

    for proc in processes:
        proc.start()

    rows = _collect_results(
        processes,
        results,
        timeout_s=config.barrier_timeout_s + 60.0,
    )

    for proc in processes:
        proc.join(timeout=10.0)

    failures = [row for row in rows if "error" in row]
    if failures:
        joined = "\n".join(str(row["error"]) for row in failures)
        pytest.fail(joined)

    bad_exitcodes = [(proc.name, proc.pid, proc.exitcode) for proc in processes if proc.exitcode not in (0, None)]
    if bad_exitcodes:
        pytest.fail(f"rank process failures: {bad_exitcodes}")

    _print_report(rows, config)
    assert all(len(row["times_s"]) == config.runs for row in rows)


def test_fused_moe_bf16_tiled_deepseek_v4_flash_tp4_bound_gflops() -> None:
    """Measure fused MoE FLOP/s for the 2048-token TP=4 prefill shape."""
    _run_tp4_bound_bench(_config_from_env())


def test_fused_moe_bf16_tiled_deepseek_v4_flash_tp4_2000_tokens_arm_codex_layout() -> None:
    """Arm-codex-internal/Arm-codex TP=4 MoE on rank bases 0/80/160/240."""
    defaults = DSV4TP4MoEBenchConfig()
    config = DSV4TP4MoEBenchConfig(
        tokens=_env_int("FUSED_CPP_DSV4_MOE_BENCH_TOKENS", 2000),
        hidden_size=_env_int("FUSED_CPP_DSV4_MOE_BENCH_HIDDEN_SIZE", defaults.hidden_size),
        moe_intermediate_size=_env_int(
            "FUSED_CPP_DSV4_MOE_BENCH_MOE_INTERMEDIATE_SIZE",
            defaults.moe_intermediate_size,
        ),
        tp_size=4,
        experts=_env_int("FUSED_CPP_DSV4_MOE_BENCH_EXPERTS", defaults.experts),
        top_k=_env_int("FUSED_CPP_DSV4_MOE_BENCH_TOP_K", defaults.top_k),
        threads_per_rank=_env_int("FUSED_CPP_DSV4_MOE_BENCH_THREADS_PER_RANK", 64),
        cores_per_numa=_env_int("FUSED_CPP_DSV4_MOE_BENCH_CORES_PER_NUMA", 80),
        core_skip=_env_int("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", 40),
        groups_per_partition=_env_int(
            "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
            defaults.groups_per_partition,
        ),
        warmup=_env_int("FUSED_CPP_DSV4_MOE_BENCH_WARMUP", defaults.warmup),
        runs=_env_int("FUSED_CPP_DSV4_MOE_BENCH_RUNS", defaults.runs),
        seed=_env_int("FUSED_CPP_DSV4_MOE_BENCH_SEED", defaults.seed),
        std=_env_float("FUSED_CPP_DSV4_MOE_BENCH_STD", defaults.std),
        activation=os.environ.get("FUSED_CPP_DSV4_MOE_BENCH_ACTIVATION", defaults.activation),
        barrier_timeout_s=_env_float(
            "FUSED_CPP_DSV4_MOE_BENCH_TIMEOUT_S",
            defaults.barrier_timeout_s,
        ),
        prepack_threads=_env_int("FUSED_CPP_MOE_PREPACK_THREADS", defaults.prepack_threads),
    )
    _run_tp4_bound_bench(config)


def test_fused_moe_bf16_tiled_deepseek_v4_flash_tp4_profiler_routing_arm_codex_layout() -> None:
    """Arm-codex-internal/Arm-codex TP=4 benchmark using captured routing."""
    routing_dir = os.environ.get(
        "FUSED_CPP_DSV4_MOE_ROUTING_DIR",
        "/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4/tests/v1/e2e/"
        "generation_sampling/profiler_output/deepseek_v4_flash_bf16/"
        "20260628-221103/moe_routing",
    )
    if not Path(routing_dir).exists():
        pytest.skip(f"routing_dir does not exist: {routing_dir}")
    routing_seq = _env_int("FUSED_CPP_DSV4_MOE_ROUTING_SEQ", 0)
    routing_entry = _load_moe_routing_entry(routing_dir, rank=0, seq=routing_seq)

    defaults = DSV4TP4MoEBenchConfig()
    config = DSV4TP4MoEBenchConfig(
        tokens=int(routing_entry["tokens"]),
        hidden_size=int(routing_entry.get("hidden_size", defaults.hidden_size)),
        moe_intermediate_size=_env_int(
            "FUSED_CPP_DSV4_MOE_BENCH_MOE_INTERMEDIATE_SIZE",
            defaults.moe_intermediate_size,
        ),
        tp_size=int(routing_entry.get("tp_world_size", 4)),
        experts=int(routing_entry.get("num_experts", defaults.experts)),
        top_k=int(routing_entry["top_k"]),
        threads_per_rank=_env_int("FUSED_CPP_DSV4_MOE_BENCH_THREADS_PER_RANK", 64),
        cores_per_numa=_env_int("FUSED_CPP_DSV4_MOE_BENCH_CORES_PER_NUMA", 80),
        core_skip=_env_int("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", 40),
        groups_per_partition=_env_int(
            "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
            defaults.groups_per_partition,
        ),
        warmup=_env_int("FUSED_CPP_DSV4_MOE_BENCH_WARMUP", defaults.warmup),
        runs=_env_int("FUSED_CPP_DSV4_MOE_BENCH_RUNS", defaults.runs),
        seed=_env_int("FUSED_CPP_DSV4_MOE_BENCH_SEED", defaults.seed),
        std=_env_float("FUSED_CPP_DSV4_MOE_BENCH_STD", defaults.std),
        activation=os.environ.get("FUSED_CPP_DSV4_MOE_BENCH_ACTIVATION", defaults.activation),
        barrier_timeout_s=_env_float(
            "FUSED_CPP_DSV4_MOE_BENCH_TIMEOUT_S",
            defaults.barrier_timeout_s,
        ),
        prepack_threads=_env_int("FUSED_CPP_MOE_PREPACK_THREADS", defaults.prepack_threads),
        routing_dir=routing_dir,
        routing_seq=routing_seq,
    )
    _run_tp4_bound_bench(config)


def test_fused_moe_bf16_tiled_deepseek_v4_flash_tp4_profiler_routing_2048_arm_codex_layout() -> None:
    """Arm-codex-internal/Arm-codex TP=4 benchmark using 2048-token routing."""
    routing_dir = os.environ.get(
        "FUSED_CPP_DSV4_MOE_ROUTING_DIR",
        "/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4/tests/v1/e2e/"
        "generation_sampling/profiler_output/deepseek_v4_flash_bf16/"
        "20260629-141213/moe_routing",
    )
    if not Path(routing_dir).exists():
        pytest.skip(f"routing_dir does not exist: {routing_dir}")

    target_tokens = _env_int("FUSED_CPP_DSV4_MOE_ROUTING_TOKENS", 2048)
    routing_seq = _env_int("FUSED_CPP_DSV4_MOE_ROUTING_SEQ", 70)
    routing_entry = _load_moe_routing_entry(routing_dir, rank=0, seq=routing_seq)
    if int(routing_entry["tokens"]) != target_tokens:
        pytest.fail(f"routing_seq={routing_seq} has tokens={routing_entry['tokens']}, expected {target_tokens}")

    defaults = DSV4TP4MoEBenchConfig()
    config = DSV4TP4MoEBenchConfig(
        tokens=int(routing_entry["tokens"]),
        hidden_size=int(routing_entry.get("hidden_size", defaults.hidden_size)),
        moe_intermediate_size=_env_int(
            "FUSED_CPP_DSV4_MOE_BENCH_MOE_INTERMEDIATE_SIZE",
            defaults.moe_intermediate_size,
        ),
        tp_size=int(routing_entry.get("tp_world_size", 4)),
        experts=int(routing_entry.get("num_experts", defaults.experts)),
        top_k=int(routing_entry["top_k"]),
        threads_per_rank=_env_int("FUSED_CPP_DSV4_MOE_BENCH_THREADS_PER_RANK", 64),
        cores_per_numa=_env_int("FUSED_CPP_DSV4_MOE_BENCH_CORES_PER_NUMA", 80),
        core_skip=_env_int("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", 40),
        groups_per_partition=_env_int(
            "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION",
            defaults.groups_per_partition,
        ),
        warmup=_env_int("FUSED_CPP_DSV4_MOE_BENCH_WARMUP", defaults.warmup),
        runs=_env_int("FUSED_CPP_DSV4_MOE_BENCH_RUNS", defaults.runs),
        seed=_env_int("FUSED_CPP_DSV4_MOE_BENCH_SEED", defaults.seed),
        std=_env_float("FUSED_CPP_DSV4_MOE_BENCH_STD", defaults.std),
        activation=os.environ.get("FUSED_CPP_DSV4_MOE_BENCH_ACTIVATION", defaults.activation),
        barrier_timeout_s=_env_float(
            "FUSED_CPP_DSV4_MOE_BENCH_TIMEOUT_S",
            defaults.barrier_timeout_s,
        ),
        prepack_threads=_env_int("FUSED_CPP_MOE_PREPACK_THREADS", defaults.prepack_threads),
        routing_dir=routing_dir,
        routing_seq=routing_seq,
    )
    _run_tp4_bound_bench(config)
