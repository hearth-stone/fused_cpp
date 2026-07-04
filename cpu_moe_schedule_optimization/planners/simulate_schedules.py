#!/usr/bin/env python3
"""Offline schedule simulator/comparator.

Given a workload (per-expert route histogram) and a core count, score MANY
scheduling algorithms with the *validated* contention cost model
(ContentionCostModel.dag_makespan — event-driven, per-task contention derate +
overhead-split; AWS-validated to median ~2% on interval-DAGs) and print a
side-by-side comparison. Pure cost-model: runs anywhere, no kernel/AWS needed.

Each "algorithm" maps (experts, planner) -> a task-DAG in IntervalPlanner form
  tasks[k] = (expert_id, routes, core_begin, threads, deps)
which is then scored by dag_makespan. Add your own by registering in ALGORITHMS.

Usage:
  python simulate_schedules.py PROFILE.json --experts 512,512,512,512
  python simulate_schedules.py PROFILE.json --preset hotspot
  python simulate_schedules.py PROFILE.json --preset decode --cores 8 --shapes
"""
from __future__ import annotations
import argparse, os, sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel          # noqa: E402
from interval_planner import IntervalPlanner, _partitions  # noqa: E402

Experts = List[Tuple[int, int]]      # [(expert_id, routes), ...]
Tasks = List[Tuple[int, int, int, int, List[int]]]

PRESETS: Dict[str, Experts] = {
    "balanced-large":   [(i, 512) for i in range(4)],
    "balanced-8":       [(i, 256) for i in range(8)],
    "hotspot":          [(0, 1536), (1, 256), (2, 128), (3, 64), (4, 64)],
    "one-dominant":     [(0, 2048), (1, 64), (2, 64)],
    "decode-many-small":[(i, 8) for i in range(16)],
    "moe256-uniform":   [(i, 48) for i in range(256)],
}


# ---- algorithms: (experts, planner) -> (label, tasks) ----------------------
def alg_coop(experts: Experts, pl: IntervalPlanner):
    return "coop[%d]" % pl.num_cores, pl.score_shape(experts, (pl.num_cores,))[1]


def alg_expert_parallel(experts: Experts, pl: IntervalPlanner):
    shape = tuple([1] * pl.num_cores)
    return "expert-parallel[1x%d]" % pl.num_cores, pl.score_shape(experts, shape)[1]


def alg_planner_best(experts: Experts, pl: IntervalPlanner):
    r = pl.plan(experts)
    return "planner:%s" % (r["shape"],), r["tasks"]


def alg_greedy_listsched(experts: Experts, pl: IntervalPlanner):
    """List-schedule (isolated-driven placement): assign each expert (largest
    first) to the (threads, contiguous interval) that minimizes the resulting
    isolated makespan; deps = last task on each touched core. Then the tool
    scores the resulting DAG under contention. Mirrors the old offline_simulator
    greedy, but scored with the validated model."""
    ordered = sorted(experts, key=lambda e: -e[1])
    N = pl.num_cores
    best_all = None
    for cap in pl.widths:
        widths = [w for w in pl.widths if w <= cap]
        core_ready = [0.0] * N
        core_last: List[int] = [-1] * N
        tasks: Tasks = []
        for rid, routes in ordered:
            pick = None
            for th in widths:
                t_iso = pl.model.T_iso(routes, th)
                for cb in range(0, N - th + 1):
                    start = max(core_ready[cb:cb + th])
                    fin = start + t_iso
                    key = (fin, th, cb)
                    if pick is None or key < pick[0]:
                        pick = (key, th, cb, fin)
            _, th, cb, fin = pick
            deps = sorted({core_last[c] for c in range(cb, cb + th) if core_last[c] >= 0})
            tasks.append((rid, routes, cb, th, deps))
            for c in range(cb, cb + th):
                core_ready[c] = fin
                core_last[c] = len(tasks) - 1
        ms = pl._score(tasks)
        if best_all is None or ms < best_all[0]:
            best_all = (ms, cap, tasks)
    return "greedy(cap=%d)" % best_all[1], best_all[2]


ALGORITHMS = {
    "coop": alg_coop,
    "expert_parallel": alg_expert_parallel,
    "greedy": alg_greedy_listsched,
    "planner": alg_planner_best,
}


def ideal_lower_bound_ns(experts: Experts, pl: IntervalPlanner) -> float:
    """Perfect-balance, no-overhead, no-contention floor: total single-core work
    (sum of 1-thread T_iso) divided by cores. A sanity floor, not achievable."""
    total = sum(pl.model.T_iso(r, 1) for _, r in experts)
    return total / pl.num_cores


def run(experts: Experts, cores: int, model: ContentionCostModel, all_shapes: bool):
    pl = IntervalPlanner(model, cores)
    rows: List[Tuple[str, float, Tasks]] = []
    for name, fn in ALGORITHMS.items():
        label, tasks = fn(experts, pl)
        rows.append(("%s = %s" % (name, label), pl._score(tasks), tasks))
    if all_shapes:
        for shape in pl.shapes:
            ms, tasks = pl.score_shape(experts, shape)
            rows.append(("shape %s" % (shape,), ms, tasks))
    rows.sort(key=lambda r: r[1])
    best = rows[0][1]
    floor = ideal_lower_bound_ns(experts, pl)
    print("workload: %d experts, routes=%s  cores=%d" %
          (len(experts), [r for _, r in experts] if len(experts) <= 16 else
           "min/med/max=%d/%d/%d" % (min(r for _, r in experts),
                                     sorted(r for _, r in experts)[len(experts)//2],
                                     max(r for _, r in experts)), cores))
    print("ideal floor (balanced, no contention/overhead) = %.3f ms\n" % (floor / 1e6))
    print("  %-34s %10s %8s %8s" % ("algorithm", "makespan", "vs_best", "vs_floor"))
    for label, ms, _ in rows:
        print("  %-34s %8.3f ms %7.2fx %7.2fx" %
              (label, ms / 1e6, ms / best, ms / floor))


def parse_experts(s: str) -> Experts:
    return [(i, int(x)) for i, x in enumerate(s.split(",")) if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", help="contention derate profile JSON (block-2)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--experts", type=parse_experts, help="comma-separated route counts")
    g.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--shapes", action="store_true", help="also score every core-partition shape")
    a = ap.parse_args()
    experts = a.experts if a.experts is not None else PRESETS[a.preset]
    run(experts, a.cores, ContentionCostModel(a.profile), a.shapes)


if __name__ == "__main__":
    main()
