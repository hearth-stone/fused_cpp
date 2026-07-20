#!/usr/bin/env python3
"""Evaluate TP versus EP with policy-matched compute profiles and topology."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402
from profile_catalog import (  # noqa: E402
    ProfileCatalog,
    ProfileCompatibilityError,
    ProfileQuery,
)
from interval_planner import PolicyAwarePlanner  # noqa: E402


@dataclass(frozen=True)
class HierarchicalTopology:
    ranks: int
    ranks_per_group: int
    intra_bytes_per_second: float
    inter_bytes_per_second: float
    latency_seconds: float

    def __post_init__(self) -> None:
        if self.ranks <= 0 or self.ranks_per_group <= 0:
            raise ValueError("rank counts must be positive")
        if self.ranks % self.ranks_per_group:
            raise ValueError("ranks must be divisible by ranks_per_group")
        if self.intra_bytes_per_second <= 0 or self.inter_bytes_per_second <= 0:
            raise ValueError("topology bandwidths must be positive")

    @property
    def groups(self) -> int:
        return self.ranks // self.ranks_per_group

    def allreduce_ms(self, message_bytes: int) -> float:
        """Hierarchical reduce-scatter/all-reduce/all-gather bandwidth model."""
        group = self.ranks_per_group
        group_count = self.groups
        intra_bytes = 2.0 * (group - 1) / group * message_bytes if group > 1 else 0.0
        inter_bytes = 2.0 * (group_count - 1) / group_count * message_bytes if group_count > 1 else 0.0
        stages = (2 * (group - 1) if group > 1 else 0) + (2 * (group_count - 1) if group_count > 1 else 0)
        seconds = (
            intra_bytes / self.intra_bytes_per_second
            + inter_bytes / self.inter_bytes_per_second
            + stages * self.latency_seconds
        )
        return seconds * 1e3

    def alltoall_ms(self, outgoing_bytes_per_rank: float) -> float:
        """Dispatch+combine over shared intra-group and inter-group links."""
        group = self.ranks_per_group
        ranks = self.ranks
        intra_load = outgoing_bytes_per_rank * max(group - 1, 0) / ranks
        inter_link_load = outgoing_bytes_per_rank * group * group / ranks if self.groups > 1 else 0.0
        one_way = max(
            intra_load / self.intra_bytes_per_second,
            inter_link_load / self.inter_bytes_per_second,
        )
        peers = max(ranks - 1, 0)
        return 2.0 * (one_way + peers * self.latency_seconds) * 1e3


@dataclass
class RankCompute:
    rank: int
    routes: int
    active_experts: int
    w13_split: bool
    shape: tuple[int, ...]
    predicted_ms: float


@dataclass
class ParallelResult:
    mode: str
    compute_ms: float
    communication_ms: float
    total_ms: float
    rank_compute: list[RankCompute]


def uniform_histogram(total_routes: int, experts: int) -> list[int]:
    base, remainder = divmod(total_routes, experts)
    return [base + (1 if expert < remainder else 0) for expert in range(experts)]


def partition_ep_histogram(global_histogram: list[int], ranks: int) -> list[list[int]]:
    if len(global_histogram) % ranks:
        raise ValueError("global expert count must be divisible by EP degree")
    local_experts = len(global_histogram) // ranks
    return [global_histogram[rank * local_experts : (rank + 1) * local_experts] for rank in range(ranks)]


def load_rank_histograms(path: Path, ranks: int, local_experts: int) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    histograms = payload.get("rank_histograms", payload)
    if len(histograms) != ranks:
        raise ValueError(f"expected {ranks} rank histograms")
    output = [[int(value) for value in histogram] for histogram in histograms]
    if any(len(histogram) != local_experts for histogram in output):
        raise ValueError(f"each rank histogram must contain {local_experts} experts")
    if any(value < 0 for histogram in output for value in histogram):
        raise ValueError("route counts cannot be negative")
    return output


def load_global_histogram(path: Path, experts: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    histogram = payload.get("histogram", payload)
    if not isinstance(histogram, list) or len(histogram) != experts:
        raise ValueError(f"global histogram must contain {experts} experts")
    output = [int(value) for value in histogram]
    if any(value < 0 for value in output):
        raise ValueError("route counts cannot be negative")
    return output


class ParallelLayerEvaluator:
    def __init__(
        self,
        catalog: ProfileCatalog,
        topology: HierarchicalTopology,
        *,
        hidden_size: int,
        full_intermediate_size: int,
        global_experts: int,
        cores_per_rank: int,
        dtype_bytes: int = 2,
        sve_implementation: str = "auto",
        m_tail_policy: str | None = None,
    ):
        self.catalog = catalog
        self.topology = topology
        self.hidden_size = int(hidden_size)
        self.full_intermediate_size = int(full_intermediate_size)
        self.global_experts = int(global_experts)
        self.cores_per_rank = int(cores_per_rank)
        self.dtype_bytes = int(dtype_bytes)
        self.sve_implementation = str(sve_implementation)
        if self.sve_implementation not in {"auto", "jit", "asm"}:
            raise ValueError("sve_implementation must be auto, jit, or asm")
        if self.sve_implementation == "auto" and m_tail_policy is not None:
            raise ValueError("m_tail_policy cannot override automatic SVE profile selection")
        self.m_tail_policy = None if m_tail_policy is None else str(m_tail_policy)

    def _models(self, mode: str, intermediate_size: int, local_experts: int) -> list[ContentionCostModel]:
        variants = (
            (("jit", "xbyak_exact_m"), ("asm", "static_bucketed"))
            if self.sve_implementation == "auto"
            else (
                (
                    self.sve_implementation,
                    self.m_tail_policy
                    or ("xbyak_exact_m" if self.sve_implementation == "jit" else "static_bucketed"),
                ),
            )
        )
        errors: list[str] = []
        for implementation, tail_policy in variants:
            query = ProfileQuery(
                mode=mode,
                degree=self.topology.ranks,
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                global_experts=self.global_experts,
                local_experts=local_experts,
                backend="sve",
                backend_n_tile=8,
                sve_implementation=implementation,
                m_tail_policy=tail_policy,
                activation="silu",
                dtype="bf16",
                measurement_experts=local_experts,
                cores_per_rank=self.cores_per_rank,
                concurrent_ranks=self.topology.ranks,
            )
            try:
                no_split, split = self.catalog.split_pair(query)
            except ProfileCompatibilityError as error:
                errors.append(f"{implementation}/{tail_policy}: {error}")
                continue
            return [
                ContentionCostModel(no_split.path, expected_policy=query),
                ContentionCostModel(split.path, expected_policy=query),
            ]
        raise ProfileCompatibilityError("no complete SVE profile pair matched; " + "; ".join(errors))

    def _compute(
        self,
        mode: str,
        rank_histograms: list[list[int]],
        intermediate_size: int,
    ) -> tuple[float, list[RankCompute]]:
        local_experts = len(rank_histograms[0])
        if len(rank_histograms) != self.topology.ranks:
            raise ValueError("rank histogram count must match topology ranks")
        if any(len(histogram) != local_experts for histogram in rank_histograms):
            raise ValueError("all rank histograms must have the same expert count")
        models = self._models(mode, intermediate_size, local_experts)
        policy = models[0].policy
        if policy is None or len(policy.cpu_ids_by_rank) != self.topology.ranks:
            raise ValueError("profile does not contain one physical CPU set per rank")
        rank_results: list[RankCompute] = []
        for rank, histogram in enumerate(rank_histograms):
            experts = [(expert, routes) for expert, routes in enumerate(histogram) if routes > 0]
            if not experts:
                rank_results.append(RankCompute(rank, 0, 0, False, (), 0.0))
                continue
            cpu_ids = policy.cpu_ids_by_rank[rank]
            if len(cpu_ids) != self.cores_per_rank:
                raise ValueError("profile CPU set does not match cores_per_rank")
            planner = PolicyAwarePlanner(
                models,
                self.cores_per_rank,
                cpu_ids=cpu_ids,
            )
            plan = planner.plan(experts)
            rank_results.append(
                RankCompute(
                    rank=rank,
                    routes=sum(histogram),
                    active_experts=len(experts),
                    w13_split=bool(plan["w13_split"]),
                    shape=tuple(plan["shape"]),
                    predicted_ms=float(plan["makespan_ns"]) / 1e6,
                )
            )
        return max(result.predicted_ms for result in rank_results), rank_results

    def evaluate_tp(
        self,
        tokens: int,
        top_k: int,
        global_histogram: list[int] | None = None,
    ) -> ParallelResult:
        total_routes = int(tokens) * int(top_k)
        histogram = (
            uniform_histogram(total_routes, self.global_experts) if global_histogram is None else list(global_histogram)
        )
        if len(histogram) != self.global_experts or sum(histogram) != total_routes:
            raise ValueError("TP global histogram must match global_experts and tokens * top_k")
        rank_histograms = [list(histogram) for _ in range(self.topology.ranks)]
        sharded_intermediate = self.full_intermediate_size // self.topology.ranks
        if sharded_intermediate * self.topology.ranks != self.full_intermediate_size:
            raise ValueError("full intermediate size must divide TP degree")
        compute, rank_compute = self._compute("tp", rank_histograms, sharded_intermediate)
        message_bytes = int(tokens) * self.hidden_size * self.dtype_bytes
        communication = self.topology.allreduce_ms(message_bytes)
        return ParallelResult("tp", compute, communication, compute + communication, rank_compute)

    def evaluate_ep(
        self,
        tokens: int,
        top_k: int,
        rank_histograms: list[list[int]] | None = None,
        global_histogram: list[int] | None = None,
    ) -> ParallelResult:
        local_experts = self.global_experts // self.topology.ranks
        if local_experts * self.topology.ranks != self.global_experts:
            raise ValueError("global expert count must divide EP degree")
        if rank_histograms is not None and global_histogram is not None:
            raise ValueError("provide rank_histograms or global_histogram, not both")
        total_routes = int(tokens) * int(top_k)
        if rank_histograms is None:
            if global_histogram is None:
                global_histogram = uniform_histogram(total_routes, self.global_experts)
            if len(global_histogram) != self.global_experts or sum(global_histogram) != total_routes:
                raise ValueError("EP global histogram must match global_experts and tokens * top_k")
            rank_histograms = partition_ep_histogram(global_histogram, self.topology.ranks)
        elif sum(sum(histogram) for histogram in rank_histograms) != total_routes:
            raise ValueError("EP rank histograms must sum to tokens * top_k")
        compute, rank_compute = self._compute("ep", rank_histograms, self.full_intermediate_size)
        outgoing_per_rank = int(tokens) / self.topology.ranks * int(top_k) * self.hidden_size * self.dtype_bytes
        communication = self.topology.alltoall_ms(outgoing_per_rank)
        return ParallelResult("ep", compute, communication, compute + communication, rank_compute)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=ROOT / "cpu_moe_schedule_optimization" / "cost_model" / "profiles",
    )
    parser.add_argument("--profile-pattern", default="*_v2_*.json")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--f-full", type=int, default=2048)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--cores-per-rank", type=int, default=32)
    parser.add_argument("--ranks-per-group", type=int, default=1)
    parser.add_argument("--bw-intra", type=float, default=60.0)
    parser.add_argument("--bw-inter", type=float, default=20.0)
    parser.add_argument("--latency-us", type=float, default=1.0)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--sve-implementation", choices=("auto", "jit", "asm"), default="auto")
    parser.add_argument("--ep-routes", type=Path, default=None)
    parser.add_argument("--global-routes", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = ProfileCatalog.from_directory(args.profile_dir, args.profile_pattern)
    topology = HierarchicalTopology(
        ranks=args.ranks,
        ranks_per_group=args.ranks_per_group,
        intra_bytes_per_second=args.bw_intra * 1e9,
        inter_bytes_per_second=args.bw_inter * 1e9,
        latency_seconds=args.latency_us * 1e-6,
    )
    evaluator = ParallelLayerEvaluator(
        catalog,
        topology,
        hidden_size=args.hidden,
        full_intermediate_size=args.f_full,
        global_experts=args.experts,
        cores_per_rank=args.cores_per_rank,
        dtype_bytes=args.dtype_bytes,
        sve_implementation=args.sve_implementation,
    )
    ep_histograms = None
    global_histogram = None
    if args.global_routes is not None:
        global_histogram = load_global_histogram(args.global_routes, args.experts)
    if args.ep_routes is not None:
        if global_histogram is not None:
            raise ValueError("--ep-routes and --global-routes are mutually exclusive")
        ep_histograms = load_rank_histograms(args.ep_routes, args.ranks, args.experts // args.ranks)
    tp = evaluator.evaluate_tp(args.tokens, args.topk, global_histogram)
    ep = evaluator.evaluate_ep(
        args.tokens,
        args.topk,
        ep_histograms,
        global_histogram,
    )
    if args.json:
        print(json.dumps({"tp": asdict(tp), "ep": asdict(ep)}, indent=2))
    else:
        print(f"T={args.tokens} topk={args.topk} E={args.experts} H={args.hidden} F={args.f_full} P={args.ranks}")
        for result in (tp, ep):
            print(
                f"{result.mode.upper():<3} compute={result.compute_ms:8.3f} ms "
                f"comm={result.communication_ms:8.3f} ms "
                f"total={result.total_ms:8.3f} ms"
            )
            for rank in result.rank_compute:
                print(
                    f"  rank{rank.rank}: routes={rank.routes:<6} "
                    f"active={rank.active_experts:<3} split={rank.w13_split!s:<5} "
                    f"shape={rank.shape} time={rank.predicted_ms:.3f} ms"
                )
        winner = "TP" if tp.total_ms < ep.total_ms else "EP"
        print(f"winner={winner} EP/TP={ep.total_ms / tp.total_ms:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
