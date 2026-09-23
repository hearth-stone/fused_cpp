"""Explicit quick machine calibration for the analytical MoE planner."""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import torch

try:
    from analytic_model import AnalyticMachineCalibration
    from analytic_probe_geometry import b_only_geometry, m12_gemm_geometry, read_cache_info, read_llc_domains
    from build_analytic_calibration import (
        build_calibration,
        fit_operator_residuals,
        fit_width_overheads,
        median_service_probe,
    )
    from profile_analytic_services import (
        M12_ROWS,
        PROBE_FULL_NO_STORE,
        PROBE_MATRIX_ONLY,
        SERVICE_PROBE_SCHEMA_VERSION,
        packed_weights,
        profile_load_resource,
        profile_matrix,
        profile_panel_range_restart,
    )
except ImportError:  # pragma: no cover - package-style import
    from .analytic_model import AnalyticMachineCalibration
    from .analytic_probe_geometry import b_only_geometry, m12_gemm_geometry, read_cache_info, read_llc_domains
    from .build_analytic_calibration import (
        build_calibration,
        fit_operator_residuals,
        fit_width_overheads,
        median_service_probe,
    )
    from .profile_analytic_services import (
        M12_ROWS,
        PROBE_FULL_NO_STORE,
        PROBE_MATRIX_ONLY,
        SERVICE_PROBE_SCHEMA_VERSION,
        packed_weights,
        profile_load_resource,
        profile_matrix,
        profile_panel_range_restart,
    )


QUICK_CALIBRATION_VERSION = 2
# A single quick service probe is a lottery on C9g: three idle runs gave a
# one-thread isolated error of -16%, -17% and +51% (tmp/c9g_quick_cal_20260923).
QUICK_SERVICE_REPEATS = 3
# Operator-overhead training routes, a subset of the research training profile's
# routes. 12/192/2040 fit the common residual as the research calibration does;
# all of them fit the per-width pairs.
QUICK_TRAIN_ROUTES = (1, 4, 12, 48, 192, 2040)
QUICK_RESIDUAL_ROUTES = (12, 192, 2040)
QUICK_TRAIN_MEASUREMENT_EXPERTS = 8


@dataclass(frozen=True)
class QuickMoeCalibrationResult:
    """Validated quick calibration and the information needed to reproduce it."""

    calibration: AnalyticMachineCalibration
    output_path: Path | None
    cpu_ids: tuple[int, ...]
    service_widths: tuple[int, ...]
    supported_widths: tuple[int, ...]
    elapsed_seconds: float
    fit_report: Mapping[str, object]
    payload: Mapping[str, object] | None = None


def _positive_unique(values: Sequence[int], *, maximum: int, name: str) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or result[0] <= 0 or result[-1] > maximum:
        raise ValueError(f"{name} must contain positive values no larger than {maximum}, got {result}")
    return result


def quick_service_widths(core_count: int, llc_domain_sizes: Sequence[int]) -> tuple[int, ...]:
    """Return sparse service points anchored at core and LLC topology knees."""
    if core_count <= 0:
        raise ValueError(f"core_count must be positive, got {core_count}")
    domains = _positive_unique(llc_domain_sizes, maximum=core_count, name="llc_domain_sizes")
    points = {1, core_count}
    for width in (2, 4, 8, 16):
        if width <= core_count:
            points.add(width)
    for domain_size in domains:
        points.add(domain_size)
        points.add(max(1, domain_size // 2))
    return tuple(sorted(points))


def quick_supported_widths(core_count: int, llc_domain_sizes: Sequence[int]) -> tuple[int, ...]:
    """Return planner-legal widths without requiring each width to be benchmarked."""
    service = set(quick_service_widths(core_count, llc_domain_sizes))
    width = 1
    while width <= core_count:
        service.add(width)
        width *= 2
    return tuple(sorted(service))


def _validate_host() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError("quick MoE calibration requires Linux AArch64 with the native SVE backend")
    from fused_cpp.moe import available_fused_moe_bf16_tiled_backends

    if "arm_sve_bf16" not in available_fused_moe_bf16_tiled_backends():
        raise RuntimeError("quick MoE calibration requires an available arm_sve_bf16 backend")


def _resolve_cpu_ids(cpu_ids: Sequence[int] | None) -> tuple[int, ...]:
    if cpu_ids is None:
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("cpu_ids are required when sched_getaffinity is unavailable")
        cpu_ids = sorted(os.sched_getaffinity(0))
    result = tuple(int(cpu) for cpu in cpu_ids)
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise ValueError("cpu_ids must contain unique non-negative CPU ids")
    return result


def _ordered_topology(cpu_ids: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[dict, ...]]:
    domains = tuple(
        sorted(
            read_llc_domains(cpu_ids),
            key=lambda domain: (-int(domain["capacity_bytes"]), str(domain["id"])),
        )
    )
    ordered = tuple(int(cpu) for domain in domains for cpu in domain["cpu_ids"])
    if set(ordered) != set(cpu_ids) or len(ordered) != len(cpu_ids):
        raise RuntimeError("LLC domains do not form an exact partition of the calibration CPU set")
    return ordered, domains


def _detect_backend_n_tile() -> int:
    packed = packed_weights(experts=1, k=8, n=16, seed=QUICK_CALIBRATION_VERSION)
    try:
        return int(packed.backend_n_tile)
    finally:
        del packed
        gc.collect()


@contextmanager
def _calibration_process_state() -> Iterator[None]:
    old_torch_threads = torch.get_num_threads()
    old_sve_impl = os.environ.get("FUSED_CPP_MOE_SVE_IMPL")
    old_affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    try:
        torch.set_num_threads(1)
        os.environ["FUSED_CPP_MOE_SVE_IMPL"] = "jit"
        yield
    finally:
        torch.set_num_threads(old_torch_threads)
        if old_sve_impl is None:
            os.environ.pop("FUSED_CPP_MOE_SVE_IMPL", None)
        else:
            os.environ["FUSED_CPP_MOE_SVE_IMPL"] = old_sve_impl
        if old_affinity is not None:
            os.sched_setaffinity(0, old_affinity)


def _collect_quick_service_probe(
    cpu_ids: tuple[int, ...],
    *,
    seed: int,
    report: Callable[[str], None] | None,
) -> tuple[dict, dict[str, dict], tuple[int, ...]]:
    ordered_cpu_ids, llc_domains = _ordered_topology(cpu_ids)
    domain_sizes = tuple(len(domain["cpu_ids"]) for domain in llc_domains)
    widths = quick_service_widths(len(ordered_cpu_ids), domain_sizes)
    n_tile = _detect_backend_n_tile()
    caches = read_cache_info(ordered_cpu_ids[0])
    llc_capacity = sum(int(domain["capacity_bytes"]) for domain in llc_domains)
    llc_probe_capacity = max(int(domain["capacity_bytes"]) for domain in llc_domains)
    caches["llc_bytes_per_rank"] = llc_capacity
    caches["llc_bytes_per_domain"] = llc_probe_capacity

    panel_columns = 2 * n_tile
    l1_gemm = m12_gemm_geometry(caches["l1d_bytes_per_core"], panel_columns, cache_fraction=0.625)
    l1_load = b_only_geometry(caches["l1d_bytes_per_core"], panel_columns, cache_fraction=0.5)
    l2_load = b_only_geometry(caches["l2_bytes_per_core"], panel_columns, cache_fraction=0.5)
    llc_load = b_only_geometry(llc_probe_capacity, panel_columns, cache_fraction=1.0 / 6.0)
    dram_weight = b_only_geometry(2 * caches["l2_bytes_per_core"], panel_columns, cache_fraction=1.0)
    minimum_cold_experts = max(2, math.ceil(2 * llc_capacity / dram_weight.working_set_bytes))

    services = {
        "panel_range_restart": profile_panel_range_restart(
            cache_bytes=caches["l1d_bytes_per_core"],
            cache_fraction=0.625,
            n_tile=n_tile,
            cpu=ordered_cpu_ids[0],
            warmup=16,
            runs=512,
            seed=seed,
            repeats=3,
        ),
        "matrix_flops": profile_matrix(
            name="bfmmla",
            widths=list(widths),
            cpu_ids=list(ordered_cpu_ids),
            m=M12_ROWS,
            k=4096,
            n=512,
            warmup=4,
            runs=16,
            seed=seed + 1,
            probe_mode=PROBE_MATRIX_ONLY,
            report=report,
        ),
        "gemm_core_flops": profile_matrix(
            name="gemm_l1",
            widths=list(widths),
            cpu_ids=list(ordered_cpu_ids),
            m=M12_ROWS,
            k=l1_gemm.k,
            n=l1_gemm.n,
            warmup=32,
            runs=512,
            seed=seed + 2,
            probe_mode=PROBE_FULL_NO_STORE,
            cache_level="l1d",
            cache_geometry=l1_gemm,
            report=report,
        ),
        "l1_bytes": profile_load_resource(
            name="l1_bytes",
            geometry=l1_load,
            widths=list(widths),
            cpu_ids=list(ordered_cpu_ids),
            warmup=8,
            runs=32,
            seed=seed + 3,
            cold=False,
            minimum_cold_experts=1,
            cold_experts_per_thread=1,
            report=report,
        ),
        "l2_bytes": profile_load_resource(
            name="l2_bytes",
            geometry=l2_load,
            widths=list(widths),
            cpu_ids=list(ordered_cpu_ids),
            warmup=8,
            runs=16,
            seed=seed + 4,
            cold=False,
            minimum_cold_experts=1,
            cold_experts_per_thread=1,
            report=report,
        ),
        "llc_bytes": profile_load_resource(
            name="llc_bytes",
            geometry=llc_load,
            widths=list(widths),
            cpu_ids=list(ordered_cpu_ids),
            warmup=4,
            runs=8,
            seed=seed + 5,
            cold=False,
            minimum_cold_experts=1,
            cold_experts_per_thread=1,
            report=report,
        ),
        "dram_bytes": profile_load_resource(
            name="dram_bytes",
            geometry=dram_weight,
            widths=list(widths),
            cpu_ids=list(ordered_cpu_ids),
            warmup=1,
            runs=4,
            seed=seed + 6,
            cold=True,
            minimum_cold_experts=minimum_cold_experts,
            cold_experts_per_thread=1,
            report=report,
        ),
    }
    probe = {
        "schema_version": SERVICE_PROBE_SCHEMA_VERSION,
        "kind": "moe_analytic_service_probe",
        "machine": {
            "id": platform.node(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "cpu_ids": list(ordered_cpu_ids),
            "cores_per_rank": len(ordered_cpu_ids),
        },
        "topology": {
            "rank_cpu_ids": list(ordered_cpu_ids),
            "llc_domains": list(llc_domains),
            "dram_scope": "numa_rank",
        },
        "kernel": {
            "sve_implementation": "jit",
            "probe_isa": "sve_bf16",
            "gemm_core_probe": "m12_l1_hot_full_no_store",
            "packed_panel_columns": panel_columns,
            "packed_b_element_bytes": 2,
            "bfmmla_flops_per_instruction": 32,
            "bfmmla_instructions_per_cycle": 4,
            "frontend_instructions_per_cycle": 5,
        },
        "caches": caches,
        "measurement": {
            "mode": "quick",
            "version": QUICK_CALIBRATION_VERSION,
            "service_widths": list(widths),
            "thread_pinning": "explicit_cpu_ids",
            "hugetlbfs_path": os.environ.get("FUSED_CPP_MOE_HUGETLBFS_PATH", ""),
            "cache_geometry_source": "linux_sysfs",
            "minimum_cold_experts": minimum_cold_experts,
        },
        "services": services,
    }

    domain_probes: dict[str, dict] = {}
    measured_signatures = {(domain_sizes[0], int(llc_domains[0]["capacity_bytes"]))}
    for index, domain in enumerate(llc_domains[1:], start=1):
        signature = (len(domain["cpu_ids"]), int(domain["capacity_bytes"]))
        if signature in measured_signatures:
            continue
        domain_cpu_ids = tuple(int(cpu) for cpu in domain["cpu_ids"])
        domain_widths = quick_service_widths(len(domain_cpu_ids), (len(domain_cpu_ids),))
        domain_geometry = b_only_geometry(int(domain["capacity_bytes"]), panel_columns, cache_fraction=1.0 / 6.0)
        domain_probes[str(domain["id"])] = {
            "machine": {"cpu_ids": list(domain_cpu_ids), "cores_per_rank": len(domain_cpu_ids)},
            "services": {
                "llc_bytes": profile_load_resource(
                    name=f"llc_{index}",
                    geometry=domain_geometry,
                    widths=list(domain_widths),
                    cpu_ids=list(domain_cpu_ids),
                    warmup=4,
                    runs=8,
                    seed=seed + 20 + index,
                    cold=False,
                    minimum_cold_experts=1,
                    cold_experts_per_thread=1,
                    report=report,
                )
            },
        }
        measured_signatures.add(signature)
    return probe, domain_probes, widths


def _write_json_atomic(path: Path, payload: Mapping[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"calibration output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def calibrate_moe_planner_quick(
    cpu_ids: Sequence[int] | None = None,
    *,
    output: str | Path | None = None,
    supported_widths: Sequence[int] | None = None,
    machine_id: str | None = None,
    overwrite: bool = False,
    seed: int = 20260814,
    report: Callable[[str], None] | None = None,
    service_repeats: int = QUICK_SERVICE_REPEATS,
) -> QuickMoeCalibrationResult:
    """Measure and return a thin Plan V2 analytical machine calibration.

    This function is intentionally explicit and synchronous. Call it during
    deployment or service setup, before constructing ``PlannedMoE``. It never
    runs at import time or from the fused-MoE execution entrypoints.

    The service probe runs ``service_repeats`` times and each measured point takes
    the median rate. The result is shape-independent and carries no operator
    overheads; ``train_quick_operator_overheads`` adds them for one expert shape.
    """
    _validate_host()
    if int(service_repeats) <= 0:
        raise ValueError(f"service_repeats must be positive, got {service_repeats}")
    resolved_cpu_ids = _resolve_cpu_ids(cpu_ids)
    output_path = Path(output).expanduser() if output is not None else None
    if output_path is not None and output_path.exists() and not overwrite:
        raise FileExistsError(f"calibration output already exists: {output_path}")

    begin = time.perf_counter()
    probes, domain_runs = [], []
    with _calibration_process_state():
        for repeat in range(int(service_repeats)):
            run_probe, run_domains, service_widths = _collect_quick_service_probe(
                resolved_cpu_ids,
                seed=int(seed) + 1000 * repeat,
                report=report,
            )
            probes.append(run_probe)
            domain_runs.append(run_domains)
    probe = median_service_probe(probes)
    domain_probes = {
        domain_id: median_service_probe([run[domain_id] for run in domain_runs]) for domain_id in domain_runs[0]
    }
    ordered_cpu_ids = tuple(int(cpu) for cpu in probe["machine"]["cpu_ids"])
    domain_sizes = tuple(len(domain["cpu_ids"]) for domain in probe["topology"]["llc_domains"])
    planner_widths = (
        quick_supported_widths(len(ordered_cpu_ids), domain_sizes)
        if supported_widths is None
        else _positive_unique(supported_widths, maximum=len(ordered_cpu_ids), name="supported_widths")
    )
    calibration_payload, fit_report = build_calibration(
        probe,
        machine_id=machine_id or f"{platform.node()}-rank-sve-jit-quick-v{QUICK_CALIBRATION_VERSION}",
        l2_effective_fraction=0.75,
        llc_effective_fraction=2.0 / 3.0,
        l2_b_reuse_miss_floor=0.0,
        l2_b_reuse_miss_at_capacity=1.0,
        l2_b_reuse_miss_ceiling=1.0,
        relative_uncertainty=0.20,
        backend_n_tile=int(probe["kernel"]["packed_panel_columns"]) // 2,
        llc_domain_probes=domain_probes,
        supported_widths=planner_widths,
    )
    calibration_payload["provenance"]["quick_calibration"] = {
        "version": QUICK_CALIBRATION_VERSION,
        "service_widths": list(service_widths),
        "supported_widths": list(planner_widths),
        "seed": int(seed),
        "service_repeats": int(service_repeats),
        "operator_residual_training": False,
    }
    calibration = AnalyticMachineCalibration.from_dict(calibration_payload)
    if output_path is not None:
        _write_json_atomic(output_path, calibration_payload, overwrite=overwrite)
    return QuickMoeCalibrationResult(
        calibration=calibration,
        output_path=output_path,
        cpu_ids=ordered_cpu_ids,
        service_widths=service_widths,
        supported_widths=planner_widths,
        elapsed_seconds=time.perf_counter() - begin,
        fit_report=fit_report,
        payload=calibration_payload,
    )


def _measure_isolated_training_rows(
    *,
    cpu_ids: tuple[int, ...],
    hidden_size: int,
    intermediate_size: int,
    routes: Sequence[int],
    widths: Sequence[int],
    measurement_experts: int,
    warmup: int,
    runs: int,
    seed: int,
    report: Callable[[str], None] | None,
) -> tuple[list[dict], int]:
    """Measure per-expert isolated time the way the research training profile does.

    ``measurement_experts`` experts run back to back on one team of each width and
    the call time is divided by their count; the same experts repeat every call.
    """
    try:
        from profile_contention_async import bf16, make_async_run, measure, summarize_times
    except ImportError:  # pragma: no cover - package-style import
        from .profile_contention_async import bf16, make_async_run, measure, summarize_times
    from fused_cpp.moe import prepare_fused_moe_bf16_tiled_weights

    class _NoSync:
        def wait(self) -> None:
            return None

    generator = torch.Generator().manual_seed(int(seed))
    std = 0.01
    w13 = bf16((measurement_experts, 2 * intermediate_size, hidden_size), generator, std)
    w2 = bf16((measurement_experts, hidden_size, intermediate_size), generator, std)
    packed = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)
    del w13, w2
    if int(packed.gemm_backend) != 1:
        raise RuntimeError("operator-overhead training requires the arm_sve_bf16 backend")
    rows = []
    try:
        for route_count in routes:
            for width in widths:
                run, groups, _, _ = make_async_run(
                    packed=packed,
                    hidden_size=hidden_size,
                    routes=int(route_count),
                    shape=[int(width)],
                    measurement_experts=measurement_experts,
                    num_profile_experts=measurement_experts,
                    cpu_ids=list(cpu_ids),
                    generator=generator,
                    std=std,
                )
                times = measure(run, warmup=warmup, runs=runs, sync_client=_NoSync())
                per_expert = [max(1, int(round(value / groups))) for value in times]
                rows.append({"routes": int(route_count), "threads": int(width), **summarize_times(per_expert)})
                if report is not None:
                    report(f"train routes={route_count} threads={width} median={rows[-1]['median_ns'] / 1e6:.3f} ms")
        return rows, int(packed.backend_n_tile)
    finally:
        del packed
        gc.collect()


def train_quick_operator_overheads(
    calibration_payload: Mapping[str, object],
    *,
    hidden_size: int,
    intermediate_size: int,
    global_experts: int,
    local_experts: int,
    cpu_ids: Sequence[int],
    mode: str = "standalone",
    degree: int = 1,
    concurrent_ranks: int = 1,
    routes: Sequence[int] = QUICK_TRAIN_ROUTES,
    residual_routes: Sequence[int] = QUICK_RESIDUAL_ROUTES,
    widths: Sequence[int] | None = None,
    measurement_experts: int = QUICK_TRAIN_MEASUREMENT_EXPERTS,
    warmup: int = 3,
    runs: int = 11,
    seed: int = 20260923,
    report: Callable[[str], None] | None = None,
) -> tuple[dict, dict]:
    """Fit the operator overheads a quick machine calibration leaves at zero.

    The machine probe never reads the expert shape, so the fixed and per-route cost
    around the GEMMs stays zero; that under-predicts every width, and the widest by
    the most (C9g TP4: -54% to -76% at 32-96 threads). This measures isolated
    experts of one shape and fits, as the research calibration does, one common
    ``(expert_fixed, route, stage_scale)`` residual on ``residual_routes`` and then
    one ``(expert_fixed, route)`` pair per width on all ``routes``. The result is
    shape-bound: rerun it when the parallel strategy changes the expert shape.

    Returns the trained calibration payload and a fit report.
    """
    payload = json.loads(json.dumps(calibration_payload))
    AnalyticMachineCalibration.from_dict(payload)
    resolved_cpu_ids = _resolve_cpu_ids(cpu_ids)
    train_widths = tuple(sorted({int(width) for width in (widths or payload["planner"]["supported_widths"])}))
    if not train_widths or train_widths[-1] > len(resolved_cpu_ids):
        raise ValueError(f"training widths {train_widths} exceed the {len(resolved_cpu_ids)} calibration CPUs")
    route_set = tuple(sorted({int(value) for value in routes}))
    residual_set = {int(value) for value in residual_routes}
    if not residual_set <= set(route_set) or min(route_set) <= 0:
        raise ValueError("residual_routes must be a subset of positive routes")

    begin = time.perf_counter()
    with _calibration_process_state():
        rows, n_tile = _measure_isolated_training_rows(
            cpu_ids=resolved_cpu_ids,
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            routes=route_set,
            widths=train_widths,
            measurement_experts=int(measurement_experts),
            warmup=int(warmup),
            runs=int(runs),
            seed=int(seed),
            report=report,
        )
    if n_tile != int(payload["kernel"]["backend_n_tile"]):
        raise RuntimeError(f"training packed n_tile {n_tile} differs from the calibration's")
    profile = {
        "isolated": rows,
        "expert_shape": {
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
            "activation": "silu",
            "dtype": "bf16",
        },
        "kernel": {"m_tail_policy": "xbyak_exact_m", "backend_n_tile": n_tile},
        "parallelism": {
            "mode": str(mode),
            "degree": int(degree),
            "global_experts": int(global_experts),
            "local_experts": int(local_experts),
        },
        "target": {"concurrent_ranks": int(concurrent_ranks)},
    }
    residual = fit_operator_residuals(
        payload, profile, train_routes=residual_set, train_threads=set(train_widths)
    )
    payload["overheads"]["expert_fixed_ns"] = residual["expert_fixed_ns"]
    payload["overheads"]["route_ns"] = residual["route_ns"]
    payload["stage_scales"] = {"w13": residual["stage_scale"], "w2": residual["stage_scale"]}
    by_width = fit_width_overheads(payload, profile)
    payload["overheads"]["by_width"] = by_width["by_width"]
    payload["provenance"]["isolated_residual_training"] = {
        "routes": residual["train_routes"],
        "threads": residual["train_threads"],
        "points": len(residual["rows"]),
        "source": "quick_operator_overhead_training",
    }
    payload["provenance"]["width_specific_overhead"] = {
        "fit_statistic": "non-negative least squares on the isolated rows, one pair per width",
        "stage_scale_source": "the calibration's own operator residual fit",
        "routes": list(route_set),
        "widths": [entry["threads"] for entry in by_width["by_width"]],
    }
    payload["provenance"].setdefault("quick_calibration", {})["operator_residual_training"] = {
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "mode": str(mode),
        "degree": int(degree),
        "measurement_experts": int(measurement_experts),
        "warmup": int(warmup),
        "runs": int(runs),
        "elapsed_seconds": time.perf_counter() - begin,
    }
    AnalyticMachineCalibration.from_dict(payload)
    return payload, {"residual": residual, "by_width": by_width, "rows": rows}


__all__ = [
    "QUICK_CALIBRATION_VERSION",
    "QuickMoeCalibrationResult",
    "calibrate_moe_planner_quick",
    "quick_service_widths",
    "train_quick_operator_overheads",
    "quick_supported_widths",
]
