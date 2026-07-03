"""Phase-based contention cost model for concurrent MoE expert groups.

Predicts the makespan of a set of experts running concurrently on disjoint
core intervals, from two calibrated tables:

  * T_iso(routes, threads)   -- isolated per-expert time  (block 1 / block-2 iso)
  * derate(n)                -- slowdown when n distinct experts contend (block 2)

The scalar baseline assumes all teams contend for the full duration:
    makespan ~= max_i T_iso_i * derate(K)
which over-predicts heterogeneous groups (small teams finish early and stop
contending). The phase model instead steps through completion events: while n
teams are active each drains at rate 1/derate(n); as teams finish, survivors
speed up.

No free parameters are fit to heterogeneous data: derate(n) comes only from the
homogeneous block-2 table. This keeps the model parameter-light (overfit-
resistant) and makes heterogeneous prediction a genuine out-of-sample test.
"""
from __future__ import annotations
import json
from bisect import bisect_left


class ContentionCostModel:
    def __init__(self, derate_profile_path: str):
        prof = json.load(open(derate_profile_path))
        # T_iso from the block-2 isolated baselines (same async path / pinning).
        self._iso = {(e["routes"], e["threads"]): float(e["median_ns"])
                     for e in prof["isolated"]}
        self._iso_routes = sorted({r for (r, _) in self._iso})
        self._iso_threads = sorted({t for (_, t) in self._iso})
        # derate(n): average over all shapes/routes with that distinct-count.
        by_n: dict[int, list[float]] = {}
        for e in prof["entries"]:
            by_n.setdefault(int(e["distinct_experts"]), []).append(float(e["derate"]))
        self._derate = {n: sum(v) / len(v) for n, v in by_n.items()}
        self._derate[1] = 1.0
        self._derate_ns = sorted(self._derate)

    # ---- table lookups -----------------------------------------------------
    def T_iso(self, routes: int, threads: int) -> float:
        if (routes, threads) in self._iso:
            return self._iso[(routes, threads)]
        # linear interpolation over routes at the given thread count
        rs = sorted(r for (r, t) in self._iso if t == threads)
        if not rs:
            raise KeyError(f"no isolated data for threads={threads}")
        if routes <= rs[0]:
            return self._iso[(rs[0], threads)] * routes / rs[0]
        if routes >= rs[-1]:
            return self._iso[(rs[-1], threads)] * routes / rs[-1]
        i = bisect_left(rs, routes)
        r0, r1 = rs[i - 1], rs[i]
        y0, y1 = self._iso[(r0, threads)], self._iso[(r1, threads)]
        return y0 + (y1 - y0) * (routes - r0) / (r1 - r0)

    def derate(self, n: int) -> float:
        if n in self._derate:
            return self._derate[n]
        ns = self._derate_ns
        if n <= ns[0]:
            return self._derate[ns[0]]
        if n >= ns[-1]:
            return self._derate[ns[-1]]
        i = bisect_left(ns, n)
        n0, n1 = ns[i - 1], ns[i]
        d0, d1 = self._derate[n0], self._derate[n1]
        return d0 + (d1 - d0) * (n - n0) / (n1 - n0)

    # ---- predictions -------------------------------------------------------
    def scalar_makespan(self, tasks) -> float:
        """Baseline: max isolated time * derate(#tasks)."""
        w = [self.T_iso(r, t) for (r, t) in tasks]
        return max(w) * self.derate(len(tasks))

    def phase_makespan(self, tasks) -> float:
        """Event-driven phase simulation. tasks = [(routes, threads), ...]."""
        w = [self.T_iso(r, t) for (r, t) in tasks]
        active = [i for i in range(len(tasks)) if w[i] > 0]
        wall = 0.0
        guard = 0
        while active:
            guard += 1
            assert guard <= len(tasks) + 2, "phase loop did not converge"
            n = len(active)
            rate = 1.0 / self.derate(n)          # isolated-ns drained per wall-ns
            dt = min(w[i] / rate for i in active)  # next completion
            wall += dt
            drained = dt * rate
            for i in active:
                w[i] -= drained
            active = [i for i in active if w[i] > 1e-6]
        return wall


    def dag_makespan(self, tasks) -> float:
        """Interval-DAG makespan. tasks[i] = (routes, threads, deps) where deps
        is a list of task indices that must finish before task i starts (the
        async bridge derives these from overlapping core-interval reuse).

        Event-driven over completions: between events the active set is constant,
        so every active task drains at 1/derate(#active); a completion frees a
        successor which then starts at full work. Contention uses #active only
        (derate is calibrated at full occupancy -- partial-occupancy ramp phases
        are approximated by the same curve; validate before trusting).
        """
        K = len(tasks)
        w = [self.T_iso(r, t) for (r, t, _) in tasks]
        dep_rem = [len(d) for (_, _, d) in tasks]
        succ = [[] for _ in range(K)]
        for i, (_, _, deps) in enumerate(tasks):
            for d in deps:
                succ[d].append(i)
        finish = [None] * K
        started = [dep_rem[i] == 0 for i in range(K)]
        t = 0.0
        guard = 0
        while any(f is None for f in finish):
            guard += 1
            assert guard <= 2 * K + 2, "dag loop did not converge"
            active = [i for i in range(K) if started[i] and finish[i] is None]
            if not active:
                raise ValueError("DAG deadlock (cycle or unreachable task)")
            n = len(active)
            rate = 1.0 / self.derate(n)
            dt = min(w[i] / rate for i in active)
            t += dt
            drained = dt * rate
            for i in active:
                w[i] -= drained
            for i in active:
                if w[i] <= 1e-6:
                    finish[i] = t
                    for s in succ[i]:
                        dep_rem[s] -= 1
                        if dep_rem[s] == 0:
                            started[s] = True
        return max(finish)


if __name__ == "__main__":
    import sys
    m = ContentionCostModel(sys.argv[1])
    print("derate(n):", {n: round(m.derate(n), 3) for n in (1, 2, 3, 4, 5, 6, 7, 8)})
    demo = [(2048, 4), (64, 2), (64, 2)]
    print("demo", demo, "phase=%.3fms scalar=%.3fms"
          % (m.phase_makespan(demo) / 1e6, m.scalar_makespan(demo) / 1e6))
