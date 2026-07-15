"""Runtime integration for cached policy-aware interval-DAG planning."""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402
from interval_planner import IntervalPlanner, PolicyAwarePlanner  # noqa: E402


_BUCKETS = [1, 2, 4, 8, 12, 24, 48, 96, 192, 384, 768, 1536, 2040, 4096, 8192]


def route_counts(topk_ids, num_experts: int) -> List[Tuple[int, int]]:
    import torch

    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
    return [(expert, int(routes)) for expert, routes in enumerate(counts.tolist()) if routes > 0]


def _bucket(value: int) -> int:
    return min(_BUCKETS, key=lambda candidate: abs(candidate - value))


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def signature(counts: List[Tuple[int, int]], policy_identity: tuple[object, ...] = ()) -> Tuple[object, ...]:
    """Policy-bound bucket histogram; expert ids do not affect shape choice."""
    routes = [count for _, count in counts]
    if not routes:
        return (*policy_identity, 0)
    descending = sorted(routes, reverse=True)
    top = descending[:4] + [0] * max(0, 4 - len(descending))
    bucket_histogram = [0] * len(_BUCKETS)
    for routes_for_expert in routes:
        bucket_histogram[_BUCKETS.index(_bucket(routes_for_expert))] += 1
    return (
        *policy_identity,
        _bucket(len(routes)),
        _bucket(sum(routes)),
        _bucket(descending[0]),
        _bucket(_quantile(routes, 0.50)),
        _bucket(_quantile(routes, 0.90)),
        *(_bucket(value) if value else 0 for value in top),
        *bucket_histogram,
    )


class PlannedMoE:
    def __init__(
        self,
        models: ContentionCostModel | Sequence[ContentionCostModel],
        num_cores: int = 8,
        *,
        cpu_ids: Sequence[int] | None = None,
    ):
        if isinstance(models, ContentionCostModel):
            self.models = (models,)
        else:
            self.models = tuple(models)
        if not self.models:
            raise ValueError("at least one cost model is required")
        self.num_cores = int(num_cores)
        self.cpu_ids = tuple(cpu_ids) if cpu_ids is not None else tuple(range(num_cores))
        self.interval_planners = tuple(IntervalPlanner(model, num_cores, cpu_ids=self.cpu_ids) for model in self.models)
        self.policy_planner = (
            PolicyAwarePlanner(self.models, num_cores, cpu_ids=self.cpu_ids) if len(self.models) > 1 else None
        )
        self.policy_identity = tuple(
            (
                model.policy.key_without_split(),
                model.policy.w13_split,
            )
            if model.policy is not None
            else (str(model.profile_path),)
            for model in self.models
        )
        self.shape_cache: Dict[Tuple[object, ...], Tuple[int, Tuple[int, ...]]] = {}
        self.last: dict[str, object] = {}

    def _planner_index(self, result: dict) -> int:
        if len(self.models) == 1:
            return 0
        split = bool(result["w13_split"])
        for index, model in enumerate(self.models):
            if model.policy is not None and model.policy.w13_split == split:
                return index
        raise RuntimeError(f"no planner model for split policy {split}")

    def _build_cached(self, counts, planner_index: int, shape: Tuple[int, ...]):
        planner = self.interval_planners[planner_index]
        lanes = planner._lanes(shape)
        tasks = planner._build_tasks(counts, lanes, planner._assign(counts, lanes))
        model = self.models[planner_index]
        return {
            "shape": shape,
            "w13_split": (model.policy.w13_split if model.policy is not None else None),
            "policy": (
                {
                    "profile": str(model.profile_path),
                    "w13_split": model.policy.w13_split,
                    "w13_split_chunks": model.policy.w13_split_chunks,
                }
                if model.policy is not None
                else None
            ),
            "bridge": planner.to_async_bridge(tasks),
        }

    def plan_spec_for(self, counts) -> Dict[str, object]:
        begin = time.perf_counter_ns()
        cache_key = signature(counts, self.policy_identity)
        after_signature = time.perf_counter_ns()
        hit = cache_key in self.shape_cache
        if hit:
            planner_index, shape = self.shape_cache[cache_key]
            after_search = time.perf_counter_ns()
            result = self._build_cached(counts, planner_index, shape)
        else:
            if self.policy_planner is not None:
                result = self.policy_planner.plan(counts)
            else:
                result = self.interval_planners[0].plan(counts)
            planner_index = self._planner_index(result)
            shape = tuple(result["shape"])
            self.shape_cache[cache_key] = (planner_index, shape)
            after_search = time.perf_counter_ns()
        bridge = result["bridge"]
        after_assign = time.perf_counter_ns()
        self.last = {
            "sig_ns": after_signature - begin,
            "search_ns": (after_search - after_signature) if not hit else 0,
            "lookup_ns": (after_search - after_signature) if hit else 0,
            "assign_ns": after_assign - after_search,
            "cache_hit": hit,
            "planner_overhead_ns": after_assign - begin,
            "shape": tuple(result["shape"]),
            "w13_split": result.get("w13_split"),
            "policy": result.get("policy"),
        }
        return {
            "bridge": bridge,
            "shape": tuple(result["shape"]),
            "w13_split": result.get("w13_split"),
            "operator_options": {"w13_split": result.get("w13_split")},
            "policy": result.get("policy"),
        }

    def plan_for(self, counts) -> Dict[str, object]:
        """Legacy bridge-only API; policy metadata remains available in `last`."""
        return self.plan_spec_for(counts)["bridge"]
