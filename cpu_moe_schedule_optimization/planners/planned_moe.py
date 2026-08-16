"""Runtime integration for cached policy-aware interval-DAG planning."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

try:
    from interval_planner import IntervalPlanner, PlannerCostModel  # noqa: E402
except ImportError:  # pragma: no cover - installed package import
    from .interval_planner import IntervalPlanner, PlannerCostModel


_BUCKETS = [1, 2, 4, 8, 12, 24, 48, 96, 192, 384, 768, 1536, 2040, 4096, 8192]
_TAIL_POOL_THRESHOLDS = (1, 2, 4, 8, 12)


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


def _tail_pool_signature(counts: List[Tuple[int, int]], max_routes: int) -> tuple[tuple[int, int], ...]:
    thresholds = sorted(
        {
            threshold
            for threshold in (*_TAIL_POOL_THRESHOLDS, int(max_routes))
            if 0 < threshold <= max_routes
        }
    )
    return tuple(
        (threshold, sum(routes <= threshold for _, routes in counts))
        for threshold in thresholds
    )


class PlannedMoE:
    def __init__(
        self,
        models: PlannerCostModel | Sequence[PlannerCostModel],
        num_cores: int = 8,
        *,
        cpu_ids: Sequence[int] | None = None,
        tail_repartition_widths: Sequence[int] | None = None,
        search_mode: str = "full",
    ):
        if callable(getattr(models, "T_iso", None)) and callable(getattr(models, "dag_makespan", None)):
            self.models = (models,)
        else:
            self.models = tuple(models)
        if len(self.models) != 1:
            raise ValueError(
                "PlannedMoE requires one calibration model"
            )
        self.num_cores = int(num_cores)
        if search_mode not in {"full", "quick"}:
            raise ValueError("search_mode must be 'full' or 'quick'")
        self.search_mode = search_mode
        self.cpu_ids = tuple(cpu_ids) if cpu_ids is not None else tuple(range(num_cores))
        self.interval_planners = tuple(
            IntervalPlanner(
                model,
                num_cores,
                cpu_ids=self.cpu_ids,
                tail_repartition_widths=tail_repartition_widths,
            )
            for model in self.models
        )
        self.policy_identity = tuple(
            (
                model.policy.identity_key(),
                str(model.profile_path),
            )
            if model.policy is not None
            else (str(model.profile_path),)
            for model in self.models
        )
        self.shape_cache: Dict[
            Tuple[object, ...],
            Tuple[int, Tuple[int, ...], str, str, int | None, int | None, int | None, int, int],
        ] = {}
        self.last: dict[str, object] = {}

    def _planner_index(self, result: dict) -> int:
        del result
        return 0

    def _build_cached(
        self,
        counts,
        planner_index: int,
        shape: Tuple[int, ...],
        execution_mode: str,
        assignment_order: str,
        tail_pool_threads: int | None,
        tail_pool_max_routes: int | None,
        tail_repartition_width: int | None,
        tail_repartition_tasks: int,
        tail_repartition_route_slices: int,
        topk_ids=None,
    ):
        planner = self.interval_planners[planner_index]
        lanes = planner._lanes(shape)
        lpt_assignment = planner._assign_lpt(counts, lanes)
        assignment = planner._assignment_for_order(lpt_assignment, assignment_order)
        tasks = planner._build_tasks(counts, lanes, assignment)
        if tail_repartition_width is not None:
            if (
                execution_mode != "strict"
                or tail_repartition_tasks != 2
                or tail_repartition_route_slices <= 0
            ):
                raise RuntimeError("cached bounded tail metadata is inconsistent")
            tasks = planner._bounded_tail_repartition_tasks(
                tasks,
                tail_repartition_width,
                tail_repartition_route_slices,
            )
            physical_tail_tasks = tail_repartition_tasks * tail_repartition_route_slices
            for _, routes, _, threads, _ in tasks[-physical_tail_tasks:]:
                planner._task_time(routes, threads)
        model = self.models[planner_index]
        if execution_mode == "tail_pool":
            assert tail_pool_threads is not None
            assert tail_pool_max_routes is not None
            bridge = planner.to_tail_pool_bridge(
                tasks,
                pool_threads=tail_pool_threads,
                max_pooled_routes=tail_pool_max_routes,
            )
        else:
            bridge = planner.to_async_bridge(tasks, topk_ids=topk_ids)
        return {
            "shape": shape,
            "assignment_order": assignment_order,
            "execution_mode": bridge["execution_mode"],
            "tail_pool_threads": tail_pool_threads if execution_mode == "tail_pool" else None,
            "tail_pool_max_routes": tail_pool_max_routes if execution_mode == "tail_pool" else None,
            "tail_pool_tasks": (
                sum(routes <= tail_pool_max_routes for _, routes in counts)
                if execution_mode == "tail_pool" and tail_pool_max_routes is not None
                else 0
            ),
            "tail_repartition_width": tail_repartition_width,
            "tail_repartition_tasks": tail_repartition_tasks,
            "tail_repartition_route_slices": tail_repartition_route_slices,
            "policy": (
                {
                    "profile": str(model.profile_path),
                }
                if model.policy is not None
                else None
            ),
            "tasks": tasks,
            "bridge": bridge,
        }

    def plan_spec_for(
        self,
        counts,
        *,
        topk_ids=None,
        dynamic_tail_pool: bool = True,
        tail_pool_threads: int | None = None,
        tail_pool_max_routes: int = 12,
        bounded_tail_repartition: bool | None = None,
    ) -> Dict[str, object]:
        begin = time.perf_counter_ns()
        counts = [(int(expert), int(routes)) for expert, routes in counts if int(routes) > 0]
        if bounded_tail_repartition is None:
            bounded_tail_repartition = dynamic_tail_pool and tail_pool_threads is None
        requested_mode = "forced" if tail_pool_threads is not None else ("auto" if dynamic_tail_pool else "strict")
        tail_pool_cache_signature = (
            _tail_pool_signature(counts, tail_pool_max_routes)
            if requested_mode != "strict"
            else ()
        )
        cache_key = (
            *signature(counts, self.policy_identity),
            requested_mode,
            tail_pool_threads,
            tail_pool_max_routes if requested_mode != "strict" else None,
            tail_pool_cache_signature,
            bounded_tail_repartition,
        )
        after_signature = time.perf_counter_ns()
        cached = self.shape_cache.get(cache_key)
        hit = cached is not None
        if hit:
            assert cached is not None
            (
                planner_index,
                shape,
                execution_mode,
                assignment_order,
                selected_pool_threads,
                selected_max_routes,
                selected_tail_width,
                selected_tail_tasks,
                selected_tail_route_slices,
            ) = cached
            try:
                result = self._build_cached(
                    counts,
                    planner_index,
                    shape,
                    execution_mode,
                    assignment_order,
                    selected_pool_threads,
                    selected_max_routes,
                    selected_tail_width,
                    selected_tail_tasks,
                    selected_tail_route_slices,
                    topk_ids,
                )
                after_search = time.perf_counter_ns()
            except (KeyError, ValueError):
                self.shape_cache.pop(cache_key, None)
                hit = False
        if not hit:
            if self.search_mode == "quick":
                if tail_pool_threads is not None:
                    raise ValueError("quick search does not support a forced tail pool")
                result = self.interval_planners[0].plan_quick(counts, topk_ids=topk_ids)
            else:
                result = self.interval_planners[0].plan(
                    counts,
                    topk_ids=topk_ids,
                    dynamic_tail_pool=dynamic_tail_pool,
                    tail_pool_max_routes=tail_pool_max_routes,
                    forced_tail_pool_threads=tail_pool_threads,
                    bounded_tail_repartition=bounded_tail_repartition,
                )
            planner_index = self._planner_index(result)
            shape = tuple(result["shape"])
            self.shape_cache[cache_key] = (
                planner_index,
                shape,
                str(result["execution_mode"]),
                str(result["assignment_order"]),
                result["tail_pool_threads"],
                result["tail_pool_max_routes"],
                result["tail_repartition_width"],
                result["tail_repartition_tasks"],
                result["tail_repartition_route_slices"],
            )
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
            "plan_version": bridge["plan_version"],
            "execution_mode": bridge["execution_mode"],
            "tail_pool_threads": result.get("tail_pool_threads"),
            "tail_pool_max_routes": result.get("tail_pool_max_routes"),
            "tail_pool_tasks": result.get("tail_pool_tasks", 0),
            "tail_repartition_width": result.get("tail_repartition_width"),
            "tail_repartition_tasks": result.get("tail_repartition_tasks", 0),
            "tail_repartition_route_slices": result.get("tail_repartition_route_slices", 1),
            "shape": tuple(result["shape"]),
            "assignment_order": result.get("assignment_order", "lpt"),
            "policy": result.get("policy"),
            "planner_backend": result.get("planner_backend", "cache"),
            "planner_workers": result.get("planner_workers", 1),
            "strict_candidates": result.get("strict_candidates", 0),
            "dynamic_candidates": result.get("dynamic_candidates", 0),
            "tail_repartition_candidates": result.get("tail_repartition_candidates", 0),
            "early_merge": bridge.get("early_merge"),
            "routing_aware_early_merge": topk_ids is not None and bridge["execution_mode"] == "strict",
        }
        return {
            "plan_version": bridge["plan_version"],
            "execution_mode": bridge["execution_mode"],
            "bridge": bridge,
            "shape": tuple(result["shape"]),
            "assignment_order": result.get("assignment_order", "lpt"),
            "tail_pool_threads": result.get("tail_pool_threads"),
            "tail_pool_max_routes": result.get("tail_pool_max_routes"),
            "tail_pool_tasks": result.get("tail_pool_tasks", 0),
            "tail_repartition_width": result.get("tail_repartition_width"),
            "tail_repartition_tasks": result.get("tail_repartition_tasks", 0),
            "tail_repartition_route_slices": result.get("tail_repartition_route_slices", 1),
            "policy": result.get("policy"),
        }

    def plan_for(
        self,
        counts,
        *,
        topk_ids=None,
        dynamic_tail_pool: bool = True,
        tail_pool_threads: int | None = None,
        tail_pool_max_routes: int = 12,
        bounded_tail_repartition: bool | None = None,
    ) -> Dict[str, object]:
        """Bridge-only API; returns Plan V2 with legacy fixed arrays retained."""
        return self.plan_spec_for(
            counts,
            topk_ids=topk_ids,
            dynamic_tail_pool=dynamic_tail_pool,
            tail_pool_threads=tail_pool_threads,
            tail_pool_max_routes=tail_pool_max_routes,
            bounded_tail_repartition=bounded_tail_repartition,
        )["bridge"]
