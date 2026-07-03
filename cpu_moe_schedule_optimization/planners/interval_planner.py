"""Static interval-DAG planner for CPU MoE expert scheduling (non-wave).

Given a routing histogram (per-expert route counts) and a core count, search
core-partition "shapes" (e.g. [8], [4,4], [2,2,2,2], [4,2,1,1], ...); for each
shape lay out contiguous core lanes, LPT-assign experts to lanes (each expert
runs at its lane's width), build the interval-DAG (experts on a lane form a
sequential chain; lanes run concurrently), and score the whole plan with the
validated ContentionCostModel.dag_makespan (which accounts for cross-lane
contention). Return the min-makespan plan.

This replaces the deprecated wave planners: no global barrier, cost-driven, and
emits the fused_moe_bf16_tiled_async bridge directly.
"""
from __future__ import annotations
import os, sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402


def _partitions(n: int, parts: Sequence[int]) -> List[Tuple[int, ...]]:
    """Non-increasing integer partitions of n using the allowed part sizes."""
    parts = sorted(parts, reverse=True)
    out: List[Tuple[int, ...]] = []

    def rec(remaining: int, max_part: int, cur: List[int]) -> None:
        if remaining == 0:
            out.append(tuple(cur))
            return
        for p in parts:
            if p <= max_part and p <= remaining:
                rec(remaining - p, p, cur + [p])
    rec(n, max(parts), [])
    return out


class IntervalPlanner:
    def __init__(self, model: ContentionCostModel, num_cores: int = 8,
                 widths: Sequence[int] = (1, 2, 4, 8)):
        self.model = model
        self.num_cores = num_cores
        self.widths = tuple(widths)
        self.shapes = _partitions(num_cores, widths)

    # ---- one shape ---------------------------------------------------------
    def _lanes(self, shape: Tuple[int, ...]):
        begins, c = [], 0
        for w in shape:
            begins.append(c); c += w
        return list(zip(begins, shape))  # [(core_begin, width), ...]

    def _assign(self, experts, lanes):
        """LPT: place each expert on the lane with the earliest resulting finish
        (using that lane's width for T_iso)."""
        L = len(lanes)
        load = [0.0] * L
        lane_exp: List[List[int]] = [[] for _ in range(L)]
        order = sorted(range(len(experts)), key=lambda i: -experts[i][1])
        for i in order:
            _, routes = experts[i]
            best = min(range(L),
                       key=lambda l: load[l] + self.model.T_iso(routes, lanes[l][1]))
            lane_exp[best].append(i)
            load[best] += self.model.T_iso(routes, lanes[best][1])
        return lane_exp

    def _build_tasks(self, experts, lanes, lane_exp):
        """tasks[k] = (expert_id, routes, core_begin, threads, deps)."""
        tasks = []
        for l, exps in enumerate(lane_exp):
            cb, w = lanes[l]
            prev = None
            for i in exps:  # LPT append order = sequential chain on this lane
                rid, routes = experts[i]
                deps = [prev] if prev is not None else []
                tasks.append((rid, routes, cb, w, deps))
                prev = len(tasks) - 1
        return tasks

    def _score(self, tasks) -> float:
        return self.model.dag_makespan([(r, t, d) for (_, r, _, t, d) in tasks])

    def score_shape(self, experts, shape):
        lanes = self._lanes(shape)
        tasks = self._build_tasks(experts, lanes, self._assign(experts, lanes))
        return self._score(tasks), tasks

    # ---- full search -------------------------------------------------------
    def plan(self, experts: List[Tuple[int, int]]) -> Dict[str, object]:
        experts = [(e, r) for (e, r) in experts if r > 0]
        ranked = []
        for shape in self.shapes:
            ms, tasks = self.score_shape(experts, shape)
            ranked.append((ms, shape, tasks))
        ranked.sort(key=lambda x: x[0])
        # Tie-break: within TIE_EPS of the best predicted makespan, prefer the
        # most cooperative shape (fewest lanes -> widest, best locality, and the
        # model is most confident there: derate(1)=1 exact, no contention guess).
        TIE_EPS = 0.03
        best_ms = ranked[0][0]
        near = [r for r in ranked if r[0] <= best_ms * (1 + TIE_EPS)]
        near.sort(key=lambda x: (len(x[1]), x[0]))  # fewest lanes, then makespan
        best_ms, best_shape, best_tasks = near[0]
        return {
            "shape": best_shape,
            "makespan_ns": best_ms,
            "tasks": best_tasks,
            "bridge": self.to_async_bridge(best_tasks),
            "ranking": [(s, round(m / 1e6, 3)) for (m, s, _) in ranked],
        }

    def to_async_bridge(self, tasks) -> Dict[str, object]:
        dep_off, dep_flat = [0], []
        for (_, _, _, _, deps) in tasks:
            dep_flat.extend(deps); dep_off.append(len(dep_flat))
        return {
            "num_threads": self.num_cores,
            "thread_cpu_ids": list(range(self.num_cores)),
            "task_expert_ids": [e for (e, _, _, _, _) in tasks],
            "task_core_begins": [c for (_, _, c, _, _) in tasks],
            "task_threads": [t for (_, _, _, t, _) in tasks],
            "task_dep_offsets": dep_off,
            "task_deps": dep_flat,
        }


if __name__ == "__main__":
    import random
    model = ContentionCostModel(sys.argv[1])
    pl = IntervalPlanner(model, num_cores=8)

    def show(name, experts):
        r = pl.plan(experts)
        coop = pl.score_shape(experts, (8,))[0] / 1e6
        ep = pl.score_shape(experts, tuple([1] * 8))[0] / 1e6
        print("\n%s  experts=%d routes=%s" % (name, len(experts), [r for _, r in experts]))
        print("  best shape=%s  pred=%.3f ms" % (r["shape"], r["makespan_ns"] / 1e6))
        print("  vs coop[8]=%.3f  expert-parallel[1x8]=%.3f" % (coop, ep))
        print("  top3:", r["ranking"][:3])

    show("balanced-large", [(i, 512) for i in range(4)])
    show("hotspot", [(0, 1536), (1, 256), (2, 128), (3, 64), (4, 64)])
    show("decode-many-small", [(i, 8) for i in range(16)])
    show("one-dominant", [(0, 2048), (1, 64), (2, 64)])
