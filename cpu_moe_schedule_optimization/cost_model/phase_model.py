"""Phase-based contention cost model for concurrent MoE expert groups.

Predicts the makespan of experts running concurrently on disjoint core intervals,
from two calibrated tables:

  * T_iso(routes, threads)   -- isolated per-expert time  (block-2 iso baselines)
  * derate(n, routes)        -- slowdown when n distinct experts contend

derate is ~route-independent for routes>=16 (a plateau) but drops toward ~1.0 for
tiny routes (decode), where fixed per-call overhead dominates the makespan and
there is little sustained bandwidth to contend for. We therefore key derate on
BOTH the active count n AND a representative route size; per phase we use the MAX
routes among active tasks (a large active task drives steady-state contention;
an all-tiny phase gets the low decode derate).

Scalar baseline: max_i T_iso_i * derate(K, maxR). Phase model: event-driven, while
n tasks active each drains at 1/derate(n, maxR_active); completions free
successors (interval-DAG deps). No parameters are fit to validation data.
"""
from __future__ import annotations
import json
import statistics
from bisect import bisect_left


class ContentionCostModel:
    def __init__(self, derate_profile_path: str):
        prof = json.load(open(derate_profile_path))
        self._iso = {(e["routes"], e["threads"]): float(e["median_ns"])
                     for e in prof["isolated"]}
        # derate2d[n][routes] = median derate over shapes with that (n, routes)
        d2: dict[int, dict[int, list[float]]] = {}
        for e in prof["entries"]:
            n = int(e["distinct_experts"]); r = int(e["routes"])
            d2.setdefault(n, {}).setdefault(r, []).append(float(e["derate"]))
        self._derate2d = {n: {r: statistics.median(v) for r, v in rd.items()}
                          for n, rd in d2.items()}
        self._dn = sorted(self._derate2d)                       # measured counts
        self._dr = sorted({r for rd in self._derate2d.values() for r in rd})

    # ---- table lookups -----------------------------------------------------
    def T_iso(self, routes: int, threads: int) -> float:
        if (routes, threads) in self._iso:
            return self._iso[(routes, threads)]
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

    def derate(self, n: int, routes: int | None = None) -> float:
        if n <= 1:
            return 1.0
        r = self._dr[-1] if routes is None else min(self._dr, key=lambda x: abs(x - routes))
        if n in self._derate2d:
            return self._derate2d[n][r]
        # interpolate over n at the (nearest) route
        if n <= self._dn[0]:
            return self._derate2d[self._dn[0]][r]
        if n >= self._dn[-1]:
            return self._derate2d[self._dn[-1]][r]
        i = bisect_left(self._dn, n)
        n0, n1 = self._dn[i - 1], self._dn[i]
        d0, d1 = self._derate2d[n0][r], self._derate2d[n1][r]
        return d0 + (d1 - d0) * (n - n0) / (n1 - n0)

    # ---- predictions -------------------------------------------------------
    def scalar_makespan(self, tasks) -> float:
        w = [self.T_iso(r, t) for (r, t) in tasks]
        maxR = max(r for (r, _) in tasks)
        return max(w) * self.derate(len(tasks), maxR)

    def phase_makespan(self, tasks) -> float:
        rts = [r for (r, _) in tasks]
        w = [self.T_iso(r, t) for (r, t) in tasks]
        active = [i for i in range(len(tasks)) if w[i] > 0]
        wall = 0.0
        guard = 0
        while active:
            guard += 1
            assert guard <= len(tasks) + 2, "phase loop did not converge"
            n = len(active)
            rate = 1.0 / self.derate(n, max(rts[i] for i in active))
            dt = min(w[i] / rate for i in active)
            wall += dt
            drained = dt * rate
            for i in active:
                w[i] -= drained
            active = [i for i in active if w[i] > 1e-6]
        return wall

    def dag_makespan(self, tasks) -> float:
        """tasks[i] = (routes, threads, deps). Event-driven over completions;
        each phase uses derate(#active, max routes among active)."""
        K = len(tasks)
        rts = [r for (r, _, _) in tasks]
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
            rate = 1.0 / self.derate(n, max(rts[i] for i in active))
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
    print("derate(n, plateau):", {n: round(m.derate(n), 3) for n in (2, 3, 4, 8)})
    print("derate(8, R):", {r: round(m.derate(8, r), 3) for r in (8, 16, 64, 512, 2048)})
    for demo in ([(2048, 4), (64, 2), (64, 2)], [(8, 1)] * 8):
        print(demo[:2], "... phase=%.3fms" % (m.phase_makespan(demo) / 1e6))
