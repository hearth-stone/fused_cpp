#!/usr/bin/env python3
"""Predict a robust split-W13 expert working-set band from cache metrics.

The model uses a GEMM-free packed-weight scan to identify how many independent
streams are needed to saturate cache bandwidth.  Its upper bound is derived
from the per-owner private-cache budget of the N-split kernel.  ``T_iso`` then
chooses the smallest working set whose isolated makespan is within a configured
headroom of the best candidate inside that band.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gemm_ecm import kernel_panels  # noqa: E402
from iso_formula import IsoFormula  # noqa: E402


@dataclass(frozen=True)
class ScanObservation:
    active_streams: int
    stream_bytes: int
    working_set_bytes: int
    bandwidth_bytes_per_second: float


@dataclass(frozen=True)
class OwnerCacheModel:
    cores: int
    private_cache_bytes_per_core: int
    cache_ways: int
    reserved_ways: int
    bandwidth_limit_bytes_per_second: float
    stream_saturation: float
    target_bandwidth_utilization: float

    @property
    def owner_cache_budget_bytes(self) -> int:
        usable_ways = self.cache_ways - self.reserved_ways
        return (
            self.cores
            * self.private_cache_bytes_per_core
            * usable_ways
            // self.cache_ways
        )

    def resident_bandwidth(self, active_streams: int) -> float:
        if active_streams <= 0:
            raise ValueError("active_streams must be positive")
        utilization = 1.0 - math.exp(-active_streams / self.stream_saturation)
        return self.bandwidth_limit_bytes_per_second * utilization

    def minimum_streams(self) -> int:
        return max(
            1,
            math.ceil(
                -self.stream_saturation
                * math.log(1.0 - self.target_bandwidth_utilization)
            ),
        )

    def maximum_streams(self, stream_bytes: int) -> int:
        if stream_bytes <= 0:
            raise ValueError("stream_bytes must be positive")
        # Require strict headroom. Equality consumes every owner-cache way in
        # the budget and leaves no replacement space for kernel transients.
        return max(1, math.ceil(self.owner_cache_budget_bytes / stream_bytes) - 1)


def load_scan_observations(path: Path) -> list[ScanObservation]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty scan calibration: {path}")
    observations = [
        ScanObservation(
            active_streams=int(row["active_streams"]),
            stream_bytes=int(row["stream_bytes"]),
            working_set_bytes=int(row["working_set_bytes"]),
            bandwidth_bytes_per_second=float(row["bandwidth_gbs"]) * 1e9,
        )
        for row in rows
    ]
    if len({row.active_streams for row in observations}) != len(observations):
        raise ValueError("scan calibration has duplicate active-stream points")
    if len({row.stream_bytes for row in observations}) != 1:
        raise ValueError("scan calibration must use one stream size")
    return sorted(observations, key=lambda row: row.active_streams)


def fit_owner_cache_model(
    observations: list[ScanObservation],
    *,
    cores: int,
    private_cache_bytes_per_core: int,
    cache_ways: int,
    reserved_ways: int,
    target_bandwidth_utilization: float,
) -> OwnerCacheModel:
    if min(cores, private_cache_bytes_per_core, cache_ways) <= 0:
        raise ValueError("cores, private cache bytes, and ways must be positive")
    if not 0 <= reserved_ways < cache_ways:
        raise ValueError("reserved ways must be in [0, cache_ways)")
    if not 0.0 < target_bandwidth_utilization < 1.0:
        raise ValueError("target bandwidth utilization must be in (0, 1)")
    owner_budget = (
        cores
        * private_cache_bytes_per_core
        * (cache_ways - reserved_ways)
        // cache_ways
    )
    stream_bytes = observations[0].stream_bytes
    maximum_streams = max(1, math.ceil(owner_budget / stream_bytes) - 1)
    resident = [row for row in observations if row.working_set_bytes < owner_budget]
    if maximum_streams == 1:
        first = next((row for row in observations if row.active_streams == 1), None)
        if first is None:
            raise ValueError("single-stream owner budget requires a 1-stream point")
        saturation = -1.0 / math.log(1.0 - target_bandwidth_utilization)
        utilization = 1.0 - math.exp(-1.0 / saturation)
        return OwnerCacheModel(
            cores=cores,
            private_cache_bytes_per_core=private_cache_bytes_per_core,
            cache_ways=cache_ways,
            reserved_ways=reserved_ways,
            bandwidth_limit_bytes_per_second=(
                first.bandwidth_bytes_per_second / utilization
            ),
            stream_saturation=saturation,
            target_bandwidth_utilization=target_bandwidth_utilization,
        )
    if len(resident) < 3:
        raise ValueError("need at least three scan points below owner-cache budget")

    best: tuple[float, float, float] | None = None
    for step in range(1, 321):
        saturation = step * 0.025
        log_limits = []
        for row in resident:
            utilization = 1.0 - math.exp(-row.active_streams / saturation)
            log_limits.append(
                math.log(row.bandwidth_bytes_per_second) - math.log(utilization)
            )
        limit = math.exp(statistics.fmean(log_limits))
        error = statistics.fmean(
            math.log(
                limit
                * (1.0 - math.exp(-row.active_streams / saturation))
                / row.bandwidth_bytes_per_second
            )
            ** 2
            for row in resident
        )
        if best is None or error < best[0]:
            best = (error, limit, saturation)
    assert best is not None
    return OwnerCacheModel(
        cores=cores,
        private_cache_bytes_per_core=private_cache_bytes_per_core,
        cache_ways=cache_ways,
        reserved_ways=reserved_ways,
        bandwidth_limit_bytes_per_second=best[1],
        stream_saturation=best[2],
        target_bandwidth_utilization=target_bandwidth_utilization,
    )


def scan_fit_report(
    observations: list[ScanObservation], model: OwnerCacheModel
) -> list[dict]:
    rows = []
    for observation in observations:
        resident = observation.working_set_bytes < model.owner_cache_budget_bytes
        predicted = (
            model.resident_bandwidth(observation.active_streams) if resident else None
        )
        rows.append(
            {
                **asdict(observation),
                "inside_owner_cache_budget": resident,
                "predicted_resident_bandwidth_bytes_per_second": predicted,
                "relative_error": (
                    predicted / observation.bandwidth_bytes_per_second - 1.0
                    if predicted is not None
                    else None
                ),
            }
        )
    return rows


def split_stage_bytes(hidden_size: int, intermediate_size: int) -> int:
    if min(hidden_size, intermediate_size) <= 0:
        raise ValueError("H and F must be positive")
    # W13 is split into two equal N ranges. Each range and W2 contain H*F
    # BF16 values, so all three sequential stages have the same weight size.
    return 2 * hidden_size * intermediate_size


def isolated_baseline_ns(
    routes: int,
    shape: list[int],
    measured_experts: int,
    formula: IsoFormula,
) -> float:
    if not shape or min(shape) <= 0 or measured_experts <= 0:
        raise ValueError("shape and measured_experts must be positive")
    loads = [0.0] * len(shape)
    for _ in range(measured_experts):
        lane = min(
            range(len(shape)),
            key=lambda index: loads[index] + formula.T_iso(routes, shape[index]),
        )
        loads[lane] += formula.T_iso(routes, shape[lane])
    return max(loads)


def validate_split_profile(profile: dict) -> tuple[int, int, int]:
    kernel = profile.get("kernel", {})
    if not kernel.get("w13_split") or int(kernel.get("w13_split_chunks", 0)) != 2:
        raise ValueError("working-set model supports only two-range split-W13")
    shape = profile["expert_shape"]
    hidden_size = int(shape["hidden_size"])
    intermediate_size = int(shape["intermediate_size"])
    stage_bytes = split_stage_bytes(hidden_size, intermediate_size)
    recorded = int(profile["working_set"]["max_weight_stage_bytes_per_expert"])
    if recorded != stage_bytes:
        raise ValueError(f"recorded stage bytes {recorded} != formula {stage_bytes}")
    return hidden_size, intermediate_size, stage_bytes


def profile_candidates(profile: dict, formula: IsoFormula) -> list[dict]:
    shape_info = profile["expert_shape"]
    measured_experts = int(shape_info["measurement_experts"])
    candidates = []
    for entry in profile["entries"]:
        shape = [int(value) for value in entry["shape"]]
        candidates.append(
            {
                "routes": int(entry["routes"]),
                "shape": shape,
                "active_experts": len(shape),
                "isolated_baseline_ns": float(
                    entry.get(
                        "iso_baseline_makespan_ns",
                        isolated_baseline_ns(
                            int(entry["routes"]), shape, measured_experts, formula
                        ),
                    )
                ),
                "measured_ns": float(entry["full_call_median_ns"]),
            }
        )
    return candidates


def search_candidates(search: dict, formula: IsoFormula) -> list[dict]:
    shape_info = search["expert_shape"]
    measured_experts = int(shape_info["measurement_experts"])
    candidates = []
    for entry in search["entries"]:
        if entry["allocation"] != "uniform-cores":
            continue
        shape = [int(value) for value in entry["shape"]]
        candidates.append(
            {
                "routes": int(entry["routes"]),
                "shape": shape,
                "active_experts": len(shape),
                "isolated_baseline_ns": isolated_baseline_ns(
                    int(entry["routes"]), shape, measured_experts, formula
                ),
                "measured_ns": float(entry["full_call_median_ns"]),
            }
        )
    return candidates


def merge_candidates(candidate_sets: list[list[dict]]) -> list[dict]:
    merged: dict[tuple[int, tuple[int, ...]], dict] = {}
    for candidates in candidate_sets:
        for candidate in candidates:
            key = (candidate["routes"], tuple(candidate["shape"]))
            merged[key] = candidate
    return list(merged.values())


def recommend_working_sets(
    candidates: list[dict],
    *,
    stage_bytes: int,
    model: OwnerCacheModel,
    iso_headroom: float,
    minimum_panels: int = 16,
) -> list[dict]:
    if not 0.0 <= iso_headroom < 1.0:
        raise ValueError("iso_headroom must be in [0, 1)")
    if minimum_panels <= 0:
        raise ValueError("minimum_panels must be positive")
    minimum_streams = model.minimum_streams()
    maximum_streams = model.maximum_streams(stage_bytes)
    summaries = []
    for routes in sorted({row["routes"] for row in candidates}):
        rows = [row for row in candidates if row["routes"] == routes]
        in_band = [
            row
            for row in rows
            if minimum_streams <= row["active_experts"] <= maximum_streams
        ]
        if not in_band:
            raise ValueError(f"route {routes} has no candidate in predicted band")
        best_iso = min(row["isolated_baseline_ns"] for row in in_band)
        compute_near = [
            row
            for row in in_band
            if row["isolated_baseline_ns"] <= best_iso * (1.0 + iso_headroom)
        ]
        recommended = min(
            compute_near,
            key=lambda row: (row["active_experts"], row["isolated_baseline_ns"]),
        )
        measured_best = min(rows, key=lambda row: row["measured_ns"])
        physical_panels = len(kernel_panels(routes))
        summaries.append(
            {
                "routes": routes,
                "physical_panels": physical_panels,
                "applicable": physical_panels >= minimum_panels,
                "predicted_min_active_experts": minimum_streams,
                "predicted_max_active_experts": maximum_streams,
                "predicted_min_working_set_bytes": minimum_streams * stage_bytes,
                "predicted_max_working_set_bytes": maximum_streams * stage_bytes,
                "recommended_shape": recommended["shape"],
                "recommended_active_experts": recommended["active_experts"],
                "recommended_working_set_bytes": (
                    recommended["active_experts"] * stage_bytes
                ),
                "recommended_iso_baseline_ns": recommended["isolated_baseline_ns"],
                "best_in_band_iso_baseline_ns": best_iso,
                "measured_best_shape": measured_best["shape"],
                "measured_best_active_experts": measured_best["active_experts"],
                "measured_best_working_set_bytes": (
                    measured_best["active_experts"] * stage_bytes
                ),
                "measured_regret": (
                    recommended["measured_ns"] / measured_best["measured_ns"] - 1.0
                ),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="split schema-v2 profile")
    parser.add_argument("--scan-csv", type=Path, required=True)
    parser.add_argument("--holdout-search", type=Path, action="append", default=[])
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--private-cache-bytes-per-core", type=int, required=True)
    parser.add_argument("--cache-ways", type=int, required=True)
    parser.add_argument("--reserved-ways", type=int, default=2)
    parser.add_argument("--target-bandwidth-utilization", type=float, default=0.95)
    parser.add_argument("--iso-headroom", type=float, default=0.05)
    parser.add_argument("--minimum-panels", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    _, _, stage_bytes = validate_split_profile(profile)
    formula = IsoFormula.from_dict(profile["iso_formula"])
    observations = load_scan_observations(args.scan_csv)
    if observations[0].stream_bytes != stage_bytes:
        raise ValueError(
            f"scan stream is {observations[0].stream_bytes} bytes; "
            f"split stage is {stage_bytes} bytes"
        )
    cores = args.cores or int(profile["target"]["cores_per_rank"])
    model = fit_owner_cache_model(
        observations,
        cores=cores,
        private_cache_bytes_per_core=args.private_cache_bytes_per_core,
        cache_ways=args.cache_ways,
        reserved_ways=args.reserved_ways,
        target_bandwidth_utilization=args.target_bandwidth_utilization,
    )
    fit_rows = scan_fit_report(observations, model)
    base_candidates = profile_candidates(profile, formula)
    base_summary = recommend_working_sets(
        base_candidates,
        stage_bytes=stage_bytes,
        model=model,
        iso_headroom=args.iso_headroom,
        minimum_panels=args.minimum_panels,
    )

    holdout_candidates = []
    for path in args.holdout_search:
        search = json.loads(path.read_text(encoding="utf-8"))
        validate_split_profile(search)
        holdout_candidates.append(search_candidates(search, formula))
    holdout_summary = (
        recommend_working_sets(
            merge_candidates(holdout_candidates),
            stage_bytes=stage_bytes,
            model=model,
            iso_headroom=args.iso_headroom,
            minimum_panels=args.minimum_panels,
        )
        if holdout_candidates
        else []
    )

    resident_errors = [
        abs(row["relative_error"])
        for row in fit_rows
        if row["relative_error"] is not None
    ]
    print(
        "owner-cache model: "
        f"budget={model.owner_cache_budget_bytes / 2**20:.1f} MiB, "
        f"B_limit={model.bandwidth_limit_bytes_per_second / 1e9:.1f} GB/s, "
        f"n_sat={model.stream_saturation:.3f}, "
        f"band={model.minimum_streams()}-"
        f"{model.maximum_streams(stage_bytes)} experts "
        f"({model.minimum_streams() * stage_bytes / 2**20:.1f}-"
        f"{model.maximum_streams(stage_bytes) * stage_bytes / 2**20:.1f} MiB)"
    )
    print(
        f"resident scan fit median={statistics.median(resident_errors):.1%} "
        f"max={max(resident_errors):.1%}"
    )
    for label, summaries in (("profile", base_summary), ("holdout", holdout_summary)):
        if not summaries:
            continue
        print(f"{label}: routes recommended measured_best regret")
        for row in summaries:
            print(
                f"  {row['routes']:<5} "
                f"{row['recommended_working_set_bytes'] / 2**20:6.1f} MiB "
                f"{row['measured_best_working_set_bytes'] / 2**20:6.1f} MiB "
                f"{row['measured_regret']:6.2%} "
                f"{'validated' if row['applicable'] else 'out-of-scope'}"
            )

    payload = {
        "schema_version": 1,
        "kind": "split_working_set_band_validation",
        "source_profile": str(args.profile),
        "scan_calibration": str(args.scan_csv),
        "holdout_searches": [str(path) for path in args.holdout_search],
        "stage_bytes_per_expert": stage_bytes,
        "model": {
            **asdict(model),
            "owner_cache_budget_bytes": model.owner_cache_budget_bytes,
        },
        "scan_fit": fit_rows,
        "profile_summary": base_summary,
        "holdout_summary": holdout_summary,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
