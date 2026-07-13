"""Versioned offline route distributions for planner and cost-model tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKLOAD_DIR = Path(__file__).with_name("workloads")
DSV4_REAL_2048_PATH = WORKLOAD_DIR / "deepseek_v4_flash_2048_seq70.json"


@dataclass(frozen=True)
class RoutingWorkload:
    name: str
    source: dict[str, Any]
    tokens: int
    top_k: int
    num_experts: int
    observed_active_experts: int
    observed_routes_std: float
    histogram: tuple[int, ...]
    tail_reconstructed: bool

    @property
    def experts(self) -> list[tuple[int, int]]:
        return [
            (expert, routes)
            for expert, routes in enumerate(self.histogram)
            if routes > 0
        ]

    @property
    def routes(self) -> int:
        return sum(self.histogram)


def _reconstruct_tail(
    total: int,
    count: int,
    minimum: int,
    maximum: int,
) -> list[int]:
    if count <= 0:
        if total:
            raise ValueError("routing summary has routes but no tail experts")
        return []
    if count == 1:
        if not minimum <= total <= maximum:
            raise ValueError("single routing tail count is outside its bounds")
        return [total]

    values = [minimum] * count
    remaining = total - minimum * count
    if remaining < 0:
        raise ValueError("routing tail total is below routes_min")

    # Preserve one routes_min expert, then balance the unknown tail exactly as
    # the existing TP4 captured-routing benchmark does.
    while remaining:
        progressed = False
        for index in range(1, count):
            if values[index] >= maximum:
                continue
            values[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("routing tail cannot fit below the captured top 16")
    return values


def _tail_counts(payload: dict[str, Any], total: int, count: int) -> list[int]:
    reconstruction = payload.get("tail_reconstruction", {})
    buckets = reconstruction.get("route_count_buckets")
    if buckets is None:
        top_counts = [int(item["routes"]) for item in payload["top_experts"]]
        tail_cap = max(min(top_counts) - 1, int(payload["routes_min"]))
        return _reconstruct_tail(
            total,
            count,
            int(payload["routes_min"]),
            tail_cap,
        )

    values = [
        int(bucket["routes"])
        for bucket in buckets
        for _ in range(int(bucket["experts"]))
    ]
    if len(values) != count or sum(values) != total:
        raise ValueError("moment-matched routing tail has invalid totals")
    return values


def load_routing_workload(path: Path = DSV4_REAL_2048_PATH) -> RoutingWorkload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported routing workload schema: {path}")

    tokens = int(payload["tokens"])
    top_k = int(payload["top_k"])
    routes = int(payload["routes"])
    num_experts = int(payload["num_experts"])
    active_experts = int(payload["active_experts"])
    routes_min = int(payload["routes_min"])
    routes_max = int(payload["routes_max"])
    if routes != tokens * top_k:
        raise ValueError("routing workload routes must equal tokens * top_k")

    histogram = [0] * num_experts
    seen: set[int] = set()
    top_counts: list[int] = []
    for item in payload["top_experts"]:
        expert = int(item["expert"])
        count = int(item["routes"])
        if expert in seen or not 0 <= expert < num_experts or count <= 0:
            raise ValueError(f"invalid top expert entry: {item}")
        seen.add(expert)
        histogram[expert] = count
        top_counts.append(count)
    if not top_counts or max(top_counts) != routes_max:
        raise ValueError("captured top experts do not match routes_max")

    tail_active = active_experts - len(top_counts)
    tail_total = routes - sum(top_counts)
    tail_counts = _tail_counts(payload, tail_total, tail_active)
    tail_experts = [
        expert for expert in range(num_experts) if expert not in seen
    ][:tail_active]
    if len(tail_experts) != tail_active:
        raise ValueError("routing workload does not have enough tail expert ids")
    for expert, count in zip(tail_experts, tail_counts, strict=True):
        histogram[expert] = count

    active = [count for count in histogram if count > 0]
    if len(active) != active_experts or sum(active) != routes:
        raise ValueError("reconstructed routing workload has invalid totals")
    if min(active) != routes_min or max(active) != routes_max:
        raise ValueError("reconstructed routing workload has invalid range")
    if not math.isclose(
        sum(active) / len(active),
        float(payload["routes_mean"]),
        rel_tol=2e-6,
    ):
        raise ValueError("reconstructed routing workload has invalid mean")
    mean = sum(active) / len(active)
    population_std = math.sqrt(
        sum((count - mean) ** 2 for count in active) / len(active)
    )
    if not math.isclose(
        population_std,
        float(payload["routes_std"]),
        abs_tol=1e-4,
    ):
        raise ValueError("reconstructed routing workload has invalid stddev")

    return RoutingWorkload(
        name=str(payload["name"]),
        source=dict(payload["source"]),
        tokens=tokens,
        top_k=top_k,
        num_experts=num_experts,
        observed_active_experts=active_experts,
        observed_routes_std=float(payload["routes_std"]),
        histogram=tuple(histogram),
        tail_reconstructed=tail_active > 0,
    )


def default_offline_workloads() -> dict[str, RoutingWorkload]:
    real_routing = load_routing_workload()
    return {real_routing.name: real_routing}
