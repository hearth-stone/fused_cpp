"""Runtime integration for the static interval-DAG planner.

Turns a live MoE call into a planned async schedule:

    routing histogram  ->  (cached) core-partition shape  ->  LPT-assign current
    experts to lanes   ->  fused_moe_bf16_tiled_async bridge

The expensive part (searching shapes, scoring each with dag_makespan) is cached
by a bucketized routing signature; the cheap part (LPT assignment + bridge
tensors) runs every call. The planner's own overhead is instrumented so it can
be counted against compute time (critical in decode, where compute is sub-ms).

This is a Python prototype for concept + overhead validation; the production hot
path would reimplement the cached lookup + assignment natively.
"""
from __future__ import annotations
import os, sys, time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402
from interval_planner import IntervalPlanner  # noqa: E402

_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def route_counts(topk_ids, num_experts: int) -> List[Tuple[int, int]]:
    """Per-expert route counts (active experts only) from topk_ids [tokens,top_k]."""
    import torch
    c = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=num_experts)
    return [(e, int(n)) for e, n in enumerate(c.tolist()) if n > 0]


def signature(counts: List[Tuple[int, int]]) -> Tuple[int, ...]:
    """Coarse, order-independent routing signature for plan caching. The shape
    choice is regime-level (few-big vs many-small vs one-dominant), so keying on
    (active count, max count, total routes) -- each bucketized -- is stable across
    decode steps with similar distributions while still separating regimes."""
    def b(x): return min(_BUCKETS, key=lambda v: abs(v - x))
    n_active = len(counts)
    total = sum(n for _, n in counts)
    mx = max(n for _, n in counts)
    return (b(n_active), b(mx), b(total))


class PlannedMoE:
    def __init__(self, model: ContentionCostModel, num_cores: int = 8):
        self.planner = IntervalPlanner(model, num_cores)
        self.shape_cache: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
        self.last = {}  # timing of the most recent plan_for() call (ns)

    def plan_for(self, counts) -> Dict[str, object]:
        """Return the async bridge for the given route counts. Times each stage
        into self.last: sig_ns, search_ns (0 on cache hit), assign_ns, cache_hit."""
        t0 = time.perf_counter_ns()
        sig = signature(counts)
        t1 = time.perf_counter_ns()
        hit = sig in self.shape_cache
        if hit:
            shape = self.shape_cache[sig]
        else:
            shape = self.planner.plan(counts)["shape"]  # full cost-model search
            self.shape_cache[sig] = shape
        t2 = time.perf_counter_ns()
        lanes = self.planner._lanes(shape)
        tasks = self.planner._build_tasks(counts, lanes, self.planner._assign(counts, lanes))
        bridge = self.planner.to_async_bridge(tasks)
        t3 = time.perf_counter_ns()
        self.last = {"sig_ns": t1 - t0, "search_ns": (t2 - t1) if not hit else 0,
                     "lookup_ns": 0 if not hit else (t2 - t1),
                     "assign_ns": t3 - t2, "cache_hit": hit,
                     "planner_overhead_ns": t3 - t0, "shape": shape}
        return bridge
