"""Runtime integration for cached policy-aware interval-DAG planning."""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

try:
    from hot_wide_planner import HotWidePlanner  # noqa: E402
    from interval_planner import IntervalPlanner, PlannerCostModel  # noqa: E402
    from model_lns import pack_lanes, planner_tasks  # noqa: E402
except ImportError:  # pragma: no cover - installed package import
    from .hot_wide_planner import HotWidePlanner
    from .interval_planner import IntervalPlanner, PlannerCostModel
    from .model_lns import pack_lanes, planner_tasks


def _domain_cores(model, num_cores: int) -> tuple[int, ...]:
    """Cores per LLC domain of the calibrated rank, or one domain when none is calibrated."""
    domains = getattr(getattr(model, "calibration", None), "llc_domains", ())
    sizes = tuple(len(getattr(domain, "cpu_ids", ())) for domain in domains)
    return sizes if sizes and sum(sizes) == int(num_cores) else (int(num_cores),)


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
        cache_plans: bool = True,
        fixed_threads: int | None = None,
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
        if search_mode not in {"full", "quick", "hot_wide"}:
            raise ValueError("search_mode must be 'full', 'quick', or 'hot_wide'")
        self.search_mode = search_mode
        self.cache_plans = bool(cache_plans)
        self.fixed_threads = None if fixed_threads is None else int(fixed_threads)
        if self.fixed_threads is not None and self.search_mode != "quick":
            raise ValueError("fixed_threads requires quick search mode")
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
        self.hot_wide_planners = tuple(
            HotWidePlanner(
                model,
                num_cores=num_cores,
                domain_cores=_domain_cores(model, num_cores),
                window_policy=planner._stage_window_policy(),
            )
            for model, planner in zip(self.models, self.interval_planners)
        ) if self.search_mode == "hot_wide" else ()
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
        shared_expert_id: int | None = None,
    ):
        planner = self.interval_planners[planner_index]
        if self.search_mode == "hot_wide" and shared_expert_id is None:
            lanes, _ = self.hot_wide_planners[planner_index].assign(
                sorted(((int(e), int(r)) for e, r in counts if int(r) > 0), key=lambda item: (-item[1], item[0])),
                shape,
            )
            used = tuple(lane for lane in lanes if lane.experts)
            begins = pack_lanes(used, self.hot_wide_planners[planner_index].domain_cores)
            if begins is None:
                raise ValueError("cached hot-wide template no longer packs")
            tasks = planner_tasks(used, begins)
        elif self.search_mode == "quick" and shared_expert_id is not None:
            tasks = planner.quick_tasks_for_shared_shape(counts, shape, shared_expert_id)
        elif self.search_mode == "quick" and len(set(shape)) == 1:
            tasks = planner.quick_tasks_for_shape(counts, shape)
        else:
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
            "shared_expert_id": shared_expert_id,
            "shared_width": shape[0] if shared_expert_id is not None else None,
            "routed_width": (
                shape[1] if shared_expert_id is not None and len(shape) > 1 else shape[0]
                if shared_expert_id is not None
                else None
            ),
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
        shared_expert_id: int | None = None,
        shared_allowed_widths: Sequence[int] | None = None,
        shared_homogeneous_only: bool = False,
    ) -> Dict[str, object]:
        begin = time.perf_counter_ns()
        counts = [(int(expert), int(routes)) for expert, routes in counts if int(routes) > 0]
        if bounded_tail_repartition is None:
            bounded_tail_repartition = dynamic_tail_pool and tail_pool_threads is None
        if shared_expert_id is not None:
            if self.search_mode != "quick":
                raise ValueError("synthetic shared expert planning currently requires quick search")
            if tail_pool_threads is not None:
                raise ValueError("synthetic shared expert planning does not support a forced tail pool")
            dynamic_tail_pool = False
            bounded_tail_repartition = False
        normalized_shared_widths = (
            None
            if shared_allowed_widths is None
            else tuple(sorted({int(width) for width in shared_allowed_widths}))
        )
        requested_mode = (
            "shared"
            if shared_expert_id is not None
            else "forced"
            if tail_pool_threads is not None
            else "auto"
            if dynamic_tail_pool
            else "strict"
        )
        tail_pool_cache_signature = (
            _tail_pool_signature(counts, tail_pool_max_routes)
            if self.cache_plans and requested_mode in {"auto", "forced"}
            else ()
        )
        cache_key = None
        if self.cache_plans:
            cache_key = (
                *signature(counts, self.policy_identity),
                requested_mode,
                tail_pool_threads,
                tail_pool_max_routes if requested_mode != "strict" else None,
                tail_pool_cache_signature,
                bounded_tail_repartition,
                shared_expert_id,
                normalized_shared_widths,
                bool(shared_homogeneous_only),
            )
        after_signature = time.perf_counter_ns()
        cached = self.shape_cache.get(cache_key) if cache_key is not None else None
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
                    topk_ids=topk_ids,
                    shared_expert_id=shared_expert_id,
                )
                after_search = time.perf_counter_ns()
            except (KeyError, ValueError):
                assert cache_key is not None
                self.shape_cache.pop(cache_key, None)
                hit = False
        if not hit:
            if self.search_mode == "hot_wide" and shared_expert_id is None:
                if tail_pool_threads is not None:
                    raise ValueError("hot-wide search does not support a forced tail pool")
                fast = self.hot_wide_planners[0].plan(counts)
                tasks = fast.tasks()
                result = {
                    "shape": tuple(fast.shape),
                    "execution_mode": "strict",
                    "assignment_order": "hot_wide",
                    "tail_pool_threads": None,
                    "tail_pool_max_routes": None,
                    "tail_repartition_width": None,
                    "tail_repartition_tasks": 0,
                    "tail_repartition_route_slices": 1,
                    "makespan_ns": fast.score_ns,
                    "uncertainty_ns": 0.0,
                    "tasks": tasks,
                    "bridge": self.interval_planners[0].to_async_bridge(tasks, topk_ids=topk_ids),
                    "policy": None,
                    "shared_expert_id": None,
                    "shared_width": None,
                    "routed_width": None,
                }
            elif self.search_mode == "quick" or (self.search_mode == "hot_wide" and shared_expert_id is not None):
                if tail_pool_threads is not None:
                    raise ValueError("quick search does not support a forced tail pool")
                if shared_expert_id is None:
                    if self.fixed_threads is None:
                        result = self.interval_planners[0].plan_quick(counts, topk_ids=topk_ids)
                    else:
                        result = self.interval_planners[0].plan_quick_fixed(
                            counts,
                            self.fixed_threads,
                            topk_ids=topk_ids,
                        )
                else:
                    result = self.interval_planners[0].plan_quick_with_shared(
                        counts,
                        shared_expert_id=shared_expert_id,
                        topk_ids=topk_ids,
                        allowed_widths=normalized_shared_widths,
                        homogeneous_only=shared_homogeneous_only,
                    )
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
            if cache_key is not None:
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
            "cache_enabled": self.cache_plans,
            "fixed_threads": self.fixed_threads,
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
            "routing_aware_early_merge": False,
            "shared_expert_id": result.get("shared_expert_id"),
            "shared_width": result.get("shared_width"),
            "routed_width": result.get("routed_width"),
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
            "shared_expert_id": result.get("shared_expert_id"),
            "shared_width": result.get("shared_width"),
            "routed_width": result.get("routed_width"),
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
