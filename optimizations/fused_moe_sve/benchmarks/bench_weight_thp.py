#!/usr/bin/env python3
"""Compare 4 KiB packed weights against verified THP on one contiguous block.

Unified-block was layout-neutral, but torch.empty can silently inherit THP.
This probe copies the same W13+W2 block into mmap+MADV_NOHUGEPAGE versus
mmap+MADV_HUGEPAGE and asks whether 2 MiB pages lower the 16x1T leftover.
It does not add a structure and does not change process-wide FUSED_CPP_PAGES.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COST_MODEL_DIR = REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(COST_MODEL_DIR)]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from fused_cpp import _moe_C  # noqa: E402
from fused_cpp.moe import (  # noqa: E402
    AsyncMoEPlanV2,
    PreparedBF16TiledFusedMoEWeights,
    fused_moe_bf16_tiled_async_plan,
    prepare_fused_moe_bf16_tiled_weights,
)
from optimizations.fused_moe_sve.benchmarks.bench_fill_port_vs_stream import (  # noqa: E402
    BACKGROUND_EXPERTS,
    _build_bridge,
)
from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (  # noqa: E402
    _cpu_mapping,
    _parse_calls,
)
from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (  # noqa: E402
    _paired_delta_stats,
    _sha256,
    _stats,
)
from optimizations.fused_moe_sve.benchmarks.bench_unified_weight_block import (  # noqa: E402
    ALIGN_ELEMS,
    layout_report,
)


OUTPUT_KIND = "moe_weight_thp_probe"
SCHEMA_VERSION = 1
POLICIES = ("small", "thp")
MODES = (
    "isolated_head",
    "wide16_same_head",
    "many16_1t_same_head",
    "many16_1t_cross_head",
)
PAGE_NEUTRAL_MS = 0.04
THP_HELPS_MS = 0.08
REMOTE_NEAR_ZERO_MS = 0.08
OVERLAP_EXPERT_FRACTION = 0.8
THP_ALIGN = 2 << 20
THP_LATCH_FRACTION = 0.8
SMALL_MAX_HUGE_BYTES = THP_ALIGN
MADV_HUGEPAGE = 14
MADV_NOHUGEPAGE = 15
MADV_COLLAPSE = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-calibration", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--threads", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--weight-copies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/moe_weight_thp"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def cell_name(policy: str, mode: str) -> str:
    return f"{policy}:{mode}"


def parse_cell(name: str) -> tuple[str, str]:
    if ":" not in name:
        raise ValueError(f"unsupported weight-thp cell {name!r}")
    policy, mode = name.split(":", 1)
    if policy not in POLICIES or mode not in MODES:
        raise ValueError(f"unsupported weight-thp cell {name!r}")
    return policy, mode


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("alignment must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def read_thp_sysfs() -> dict[str, str]:
    root = Path("/sys/kernel/mm/transparent_hugepage")
    result = {}
    for name in ("enabled", "defrag"):
        path = root / name
        result[name] = path.read_text(encoding="utf-8").strip() if path.exists() else "unavailable"
    return result


def parse_smaps_vma(text: str, address: int) -> dict[str, int]:
    current_start = 0
    current_end = 0
    fields: dict[str, int] = {}
    hit: dict[str, int] | None = None

    def finish() -> None:
        nonlocal hit
        if current_end > current_start and current_start <= address < current_end:
            hit = {"start": current_start, "end": current_end, **fields}

    for raw in text.splitlines():
        if raw and raw[0] in "0123456789abcdef" and "-" in raw.split(None, 1)[0]:
            finish()
            start_s, end_s = raw.split()[0].split("-")
            current_start = int(start_s, 16)
            current_end = int(end_s, 16)
            fields = {}
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        parts = value.split()
        if not parts or not parts[0].isdigit():
            continue
        scale = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
        fields[key.strip()] = int(parts[0]) * scale
    finish()
    if hit is None:
        raise RuntimeError(f"address {address:#x} not present in smaps")
    return hit


def vma_stats(address: int) -> dict[str, int]:
    text = Path("/proc/self/smaps").read_text(encoding="utf-8")
    return parse_smaps_vma(text, address)


def _madvise(address: int, length: int, advice: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(length), ctypes.c_int(advice)) != 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()), "madvise")


@dataclass
class MappedOwner:
    mapping: mmap.mmap
    array: np.ndarray
    tensor: torch.Tensor
    policy: str
    address: int
    map_bytes: int


def allocate_owner(nbytes: int, policy: str) -> MappedOwner:
    if sys.platform != "linux":
        raise RuntimeError("weight THP mappings require Linux")
    if policy not in POLICIES:
        raise ValueError(f"unsupported page policy {policy!r}")
    if nbytes <= 0:
        raise ValueError("owner size must be positive")
    align = THP_ALIGN if policy == "thp" else 4096
    length = round_up(nbytes, align) + align
    flags = mmap.MAP_PRIVATE
    if hasattr(mmap, "MAP_ANONYMOUS"):
        flags |= mmap.MAP_ANONYMOUS
    mapping = mmap.mmap(-1, length, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    array = np.frombuffer(mapping, dtype=np.uint8)
    base = int(array.ctypes.data)
    aligned = round_up(base, align)
    offset = aligned - base
    if offset + nbytes > length:
        mapping.close()
        raise RuntimeError("aligned owner does not fit in the anonymous map")
    advise_len = round_up(nbytes, align)
    if policy == "thp":
        _madvise(aligned, advise_len, MADV_HUGEPAGE)
    else:
        _madvise(base, length, MADV_NOHUGEPAGE)
    tensor = torch.from_numpy(array[offset : offset + nbytes])
    return MappedOwner(
        mapping=mapping,
        array=array,
        tensor=tensor,
        policy=policy,
        address=aligned,
        map_bytes=advise_len,
    )


def page_report(owner: MappedOwner, request_bytes: int) -> dict[str, object]:
    stats = vma_stats(owner.address)
    huge = int(stats.get("AnonHugePages", 0))
    size = int(stats.get("Size", owner.map_bytes))
    return {
        "policy": owner.policy,
        "address": owner.address,
        "request_bytes": request_bytes,
        "map_bytes": owner.map_bytes,
        "vma_start": stats["start"],
        "vma_end": stats["end"],
        "vma_size_bytes": size,
        "anon_huge_bytes": huge,
        "kernel_page_size_bytes": int(stats.get("KernelPageSize", 0)),
        "latch_fraction": huge / request_bytes if request_bytes else 0.0,
        "small_clean": huge <= SMALL_MAX_HUGE_BYTES,
        "thp_latched": huge >= THP_LATCH_FRACTION * request_bytes,
    }


def map_unified(
    prepared: PreparedBF16TiledFusedMoEWeights,
    policy: str,
) -> tuple[PreparedBF16TiledFusedMoEWeights, MappedOwner, dict[str, object]]:
    w13 = prepared.w13[0].contiguous()
    w2 = prepared.w2[0].contiguous()
    w13_n = int(w13.numel())
    pad = (ALIGN_ELEMS - (w13_n % ALIGN_ELEMS)) % ALIGN_ELEMS
    total_n = w13_n + pad + int(w2.numel())
    owner = allocate_owner(total_n * int(w13.element_size()), policy)
    bf16 = owner.tensor.view(torch.bfloat16)
    bf16[:w13_n].copy_(w13.reshape(-1))
    if pad:
        bf16[w13_n : w13_n + pad].zero_()
    bf16[w13_n + pad :].copy_(w2.reshape(-1))
    if policy == "thp":
        try:
            _madvise(owner.address, owner.map_bytes, MADV_COLLAPSE)
        except OSError:
            pass
    w13_u = bf16[:w13_n].view(w13.shape)
    w2_u = bf16[w13_n + pad :].view(w2.shape)
    if not w13_u.is_contiguous() or not w2_u.is_contiguous():
        raise RuntimeError("THP packed views must stay contiguous")
    unified = PreparedBF16TiledFusedMoEWeights(
        w13=(w13_u, prepared.w13[1], prepared.w13[2]),
        w2=(w2_u, prepared.w2[1], prepared.w2[2]),
        fused_silu=prepared.fused_silu,
        gemm_backend=prepared.gemm_backend,
        backend_n_tile=prepared.backend_n_tile,
        backend_name=prepared.backend_name,
    )
    return unified, owner, page_report(owner, int(bf16.nbytes))


def _overlap_ok(mode_calls: dict[str, list[float]], expected: float) -> bool:
    experts = mode_calls.get("peer_overlap_experts")
    if not experts:
        return expected <= 0.0
    return float(_stats(experts)["median_ms"]) >= OVERLAP_EXPERT_FRACTION * expected


def pages_verified(reports: dict[str, dict[str, object]]) -> bool:
    small = reports["small"]
    thp = reports["thp"]
    return bool(small["small_clean"]) and bool(thp["thp_latched"])


def decide_weight_thp(
    comparisons: dict[str, dict[str, object]],
    calls: dict[str, dict[str, list[float]]],
    reports: dict[str, dict[str, object]],
) -> dict[str, object]:
    def delta(policy: str, mode: str) -> float:
        return float(comparisons[f"{cell_name(policy, mode)}_vs_isolated"]["delta"]["median_ms"])

    many_small = delta("small", "many16_1t_same_head")
    many_thp = delta("thp", "many16_1t_same_head")
    wide_small = delta("small", "wide16_same_head")
    wide_thp = delta("thp", "wide16_same_head")
    many_cross_small = delta("small", "many16_1t_cross_head")
    many_cross_thp = delta("thp", "many16_1t_cross_head")
    overlap_ok = (
        _overlap_ok(calls[cell_name("small", "wide16_same_head")], 1.0)
        and _overlap_ok(calls[cell_name("thp", "wide16_same_head")], 1.0)
        and _overlap_ok(calls[cell_name("small", "many16_1t_same_head")], 16.0)
        and _overlap_ok(calls[cell_name("thp", "many16_1t_same_head")], 16.0)
    )
    page_ok = pages_verified(reports)
    page_neutral = abs(many_small - many_thp) <= PAGE_NEUTRAL_MS
    thp_helps = many_small - many_thp >= THP_HELPS_MS
    remote_near_zero = abs(many_cross_small) <= REMOTE_NEAR_ZERO_MS and abs(many_cross_thp) <= REMOTE_NEAR_ZERO_MS
    if not overlap_ok:
        signature = "invalid_overlap"
    elif not page_ok:
        signature = "thp_not_latched"
    elif thp_helps:
        signature = "thp_helps"
    elif page_neutral:
        signature = "page_neutral"
    else:
        signature = "inconclusive"
    return {
        "add_default_off_structure": False,
        "signature": signature,
        "overlap_ok": overlap_ok,
        "pages_verified": page_ok,
        "page_neutral": page_neutral,
        "thp_helps": thp_helps,
        "remote_near_zero": remote_near_zero,
        "many16_small_ms": many_small,
        "many16_thp_ms": many_thp,
        "wide16_small_ms": wide_small,
        "wide16_thp_ms": wide_thp,
        "many16_cross_small_ms": many_cross_small,
        "many16_cross_thp_ms": many_cross_thp,
        "gates": {
            "page_neutral_ms": PAGE_NEUTRAL_MS,
            "thp_helps_ms": THP_HELPS_MS,
            "remote_near_zero_ms": REMOTE_NEAR_ZERO_MS,
            "thp_latch_fraction": THP_LATCH_FRACTION,
            "small_max_huge_bytes": SMALL_MAX_HUGE_BYTES,
        },
        "reason": (
            "Do not add a structure. thp_helps means verified 2 MiB pages lower "
            "the 16x1T leftover versus MADV_NOHUGEPAGE. page_neutral means leftover "
            "does not track weight page size."
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if sys.platform != "linux":
        raise RuntimeError("the weight THP probe requires Linux")
    if args.threads != 80:
        raise ValueError("the Arm weight-THP probe requires exactly 80 threads")
    if min(args.hidden, args.intermediate, args.runs, args.weight_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions/runs/copies must be positive and warmup non-negative")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < args.threads:
        raise ValueError("process affinity does not expose 80 CPUs")
    cpu_ids = tuple(affinity[: args.threads])
    torch.set_num_threads(1)
    expert_count = 2 + BACKGROUND_EXPERTS
    model = AnalyticMoeCostModel(
        args.analytic_calibration,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        global_experts=expert_count,
        local_experts=expert_count,
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    built = {mode: _build_bridge(model, thread_cpu_ids=cpu_ids, mode=mode) for mode in MODES}
    route_map = built[MODES[0]][1]
    if any(built[mode][1] != route_map for mode in MODES):
        raise RuntimeError("weight-THP modes must keep the same expert route histogram")
    plans = {mode: AsyncMoEPlanV2.from_dict(value[0]) for mode, value in built.items()}
    topk_ids = torch.repeat_interleave(
        torch.arange(expert_count, dtype=torch.int32),
        torch.tensor([route_map[expert] for expert in range(expert_count)], dtype=torch.int64),
    ).reshape(-1, 1)
    generator = torch.Generator().manual_seed(args.seed)
    hidden = torch.empty((topk_ids.shape[0], args.hidden), dtype=torch.bfloat16)
    hidden.normal_(mean=0.0, std=0.01, generator=generator)
    topk_weights = torch.ones((topk_ids.shape[0], 1), dtype=torch.float32)
    w13 = torch.empty((expert_count, 2 * args.intermediate, args.hidden), dtype=torch.bfloat16)
    w13.normal_(mean=0.0, std=0.01, generator=generator)
    w2 = torch.empty((expert_count, args.hidden, args.intermediate), dtype=torch.bfloat16)
    w2.normal_(mean=0.0, std=0.01, generator=generator)
    packed: dict[str, list[PreparedBF16TiledFusedMoEWeights]] = {policy: [] for policy in POLICIES}
    owners: list[MappedOwner] = []
    reports: dict[str, dict[str, object]] = {}
    for _ in range(args.weight_copies + 1):
        prepared = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True, backend="arm_sve_bf16")
        for policy in POLICIES:
            unified, owner, report = map_unified(prepared, policy)
            packed[policy].append(unified)
            owners.append(owner)
            reports.setdefault(policy, report)
    scrub_copy = args.weight_copies
    outputs = {cell_name(policy, mode): torch.empty_like(hidden) for policy in POLICIES for mode in MODES}
    os.environ["FUSED_CPP_MOE_SVE"] = "1"
    os.environ["FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE"] = "1"
    os.environ["FUSED_CPP_MOE_W2_BF16_ROUTE"] = "0"
    os.environ["FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE"] = "0"
    cells = [cell_name(policy, mode) for policy in POLICIES for mode in MODES]

    def run(policy: str, mode: str, copy_index: int) -> torch.Tensor:
        name = cell_name(policy, mode)
        return fused_moe_bf16_tiled_async_plan(
            hidden,
            packed[policy][copy_index],
            topk_weights,
            topk_ids,
            plans[mode],
            global_num_experts=expert_count,
            out=outputs[name],
        )

    reference = run("small", MODES[0], 0).clone()
    for policy in POLICIES:
        for mode in MODES:
            torch.testing.assert_close(run(policy, mode, 0).float(), reference.float(), atol=0, rtol=0)
    warmup_order = random.Random(args.seed ^ 0xA5A5)
    for round_index in range(args.warmup):
        names = list(cells)
        warmup_order.shuffle(names)
        for position, name in enumerate(names):
            policy, mode = parse_cell(name)
            run(policy, "isolated_head", scrub_copy)
            run(policy, mode, (round_index + position) % args.weight_copies)

    trace_paths = {name: args.trace_dir / f"{name.replace(':', '_')}.log" for name in cells}
    for trace_path in trace_paths.values():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
    trace_order = random.Random(args.seed ^ 0x5A5A)
    for run_index in range(args.runs):
        names = list(cells)
        trace_order.shuffle(names)
        for position, name in enumerate(names):
            policy, mode = parse_cell(name)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"
            run(policy, "isolated_head", scrub_copy)
            os.environ["FUSED_CPP_MOE_TRACE_FILE"] = str(trace_paths[name])
            os.environ["FUSED_CPP_MOE_TRACE"] = "1"
            run(policy, mode, (run_index + position) % args.weight_copies)
            os.environ["FUSED_CPP_MOE_TRACE"] = "0"

    calls = {name: _parse_calls(path) for name, path in trace_paths.items()}
    comparisons = {}
    isolated_abs = {}
    for policy in POLICIES:
        isolated = calls[cell_name(policy, "isolated_head")]["target_span"]
        isolated_abs[policy] = _stats(isolated)
        for mode in MODES:
            if mode == "isolated_head":
                continue
            name = cell_name(policy, mode)
            comparisons[f"{name}_vs_isolated"] = _paired_delta_stats(calls[name]["target_span"], isolated)
    decision = decide_weight_thp(comparisons, calls, reports)
    result = {
        "kind": OUTPUT_KIND,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "calibration": str(args.analytic_calibration),
            "calibration_sha256": _sha256(args.analytic_calibration),
            "extension": str(_moe_C.__file__),
            "extension_sha256": _sha256(Path(_moe_C.__file__)),
        },
        "shape": {
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": expert_count,
            "threads": args.threads,
            "background_experts": BACKGROUND_EXPERTS,
        },
        "cpu_mapping": _cpu_mapping(cpu_ids),
        "sysfs": read_thp_sysfs(),
        "layouts": {policy: layout_report(packed[policy][0]) for policy in POLICIES},
        "pages": reports,
        "isolated_absolute": isolated_abs,
        "method": {
            "warmup": args.warmup,
            "trace_runs": args.runs,
            "trace_order": "randomized_paired_rounds",
            "weight_copies": args.weight_copies,
            "scrub_policy": "dedicated_disjoint_isolated_mapped_copy_before_every_sample",
            "same_tasks_routes_weights_across_policies": True,
            "metric": "target first-phase start to final-phase end",
            "weight_pages": "mmap MADV_NOHUGEPAGE vs mmap MADV_HUGEPAGE on one W13+W2 block",
            "scratch_pages": "process default FUSED_CPP_PAGES, not switched mid-run",
            "hugetlb": "not reserved; THP only",
        },
        "modes": {
            name: {
                "policy": parse_cell(name)[0],
                "mode": parse_cell(name)[1],
                **{key: _stats(values) for key, values in mode_calls.items()},
            }
            for name, mode_calls in calls.items()
        },
        "comparisons": comparisons,
        "decision": decision,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
