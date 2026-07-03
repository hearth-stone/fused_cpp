"""Phase-based contention cost model for concurrent MoE expert groups.

Tables (from the block-2 async profile):
  * T_iso(routes, threads)  -- isolated per-expert time
  * derate(n, routes)       -- slowdown when n distinct experts contend
      (~plateau for routes>=16; drops toward ~1.0 for tiny routes / decode)

Per-task overhead split: T_iso(R,t) = O(t) + G(R,t), where O(t) is the fixed
per-task overhead (dispatch/barrier/gather-scatter setup, ~0.12ms; the R->0
intercept of T_iso vs routes). Contention scales only the compute part, so a
task's contended slowdown factor is  fo + (1-fo)*derate,  fo = O/T_iso. This keeps
the tiny-R/decode regime (overhead-dominated) from being over-penalized.

Sims are event-driven with per-task effective rates; each phase uses
derate(#active, max routes among active). No parameters fit to validation data.
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
        d2: dict[int, dict[int, list[float]]] = {}
        for e in prof["entries"]:
            n = int(e["distinct_experts"]); r = int(e["routes"])
            d2.setdefault(n, {}).setdefault(r, []).append(float(e["derate"]))
        self._derate2d = {n: {r: statistics.median(v) for r, v in rd.items()}
                          for n, rd in d2.items()}
        self._dn = sorted(self._derate2d)
        self._dr = sorted({r for rd in self._derate2d.values() for r in rd})
        # per-thread fixed overhead O(t): intercept of T_iso vs routes (small R)
        self._O: dict[int, float] = {}
        for t in sorted({th for (_, th) in self._iso}):
            pts = sorted((r, v) for (r, th), v in self._iso.items() if th == t and r <= 64)
            if len(pts) >= 2:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                m = len(xs); sx = sum(xs); sy = sum(ys)
                sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
                den = m * sxx - sx * sx
                slope = (m * sxy - sx * sy) / den if den else 0.0
                self._O[t] = max((sy - slope * sx) / m, 0.0)
            else:
                self._O[t] = 0.0

    # ---- lookups -----------------------------------------------------------
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
        if n <= self._dn[0]:
            return self._derate2d[self._dn[0]][r]
        if n >= self._dn[-1]:
            return self._derate2d[self._dn[-1]][r]
        i = bisect_left(self._dn, n)
        n0, n1 = self._dn[i - 1], self._dn[i]
        return (self._derate2d[n0][r]
                + (self._derate2d[n1][r] - self._derate2d[n0][r]) * (n - n0) / (n1 - n0))

    def _fo(self, routes: int, threads: int) -> float:
        ti = self.T_iso(routes, threads)
        return min(self._O.get(threads, 0.0) / ti, 0.9) if ti > 0 else 0.0

    # ---- predictions -------------------------------------------------------
    def scalar_makespan(self, tasks) -> float:
        d = self.derate(len(tasks), max(r for (r, _) in tasks))
        return max(self.T_iso(r, t) * (self._fo(r, t) + (1 - self._fo(r, t)) * d)
                   for (r, t) in tasks)

    def _sim(self, tasks) -> float:
        """tasks[i]=(routes,threads,deps). Event-driven; per-task effective
        derate fo+(1-fo)*derate(#active, max active routes)."""
        K = len(tasks)
        rts = [r for (r, _, _) in tasks]
        thr = [t for (_, t, _) in tasks]
        rem = [self.T_iso(r, t) for (r, t, _) in tasks]
        dep_rem = [len(d) for (_, _, d) in tasks]
        succ = [[] for _ in range(K)]
        for i, (_, _, deps) in enumerate(tasks):
            for dp in deps:
                succ[dp].append(i)
        started = [dep_rem[i] == 0 for i in range(K)]
        finish = [None] * K
        wall = 0.0
        guard = 0
        while any(f is None for f in finish):
            guard += 1
            assert guard <= 2 * K + 2, "sim did not converge"
            active = [i for i in range(K) if started[i] and finish[i] is None]
            if not active:
                raise ValueError("DAG deadlock (cycle or unreachable task)")
            n = len(active)
            d = self.derate(n, max(rts[i] for i in active))
            de = {i: self._fo(rts[i], thr[i]) + (1 - self._fo(rts[i], thr[i])) * d
                  for i in active}
            dt = min(rem[i] * de[i] for i in active)
            wall += dt
            for i in active:
                rem[i] -= dt / de[i]
            for i in active:
                if rem[i] <= 1e-6:
                    finish[i] = wall
                    for s in succ[i]:
                        dep_rem[s] -= 1
                        if dep_rem[s] == 0:
                            started[s] = True
        return wall

    def phase_makespan(self, tasks) -> float:
        return self._sim([(r, t, []) for (r, t) in tasks])

    def dag_makespan(self, tasks) -> float:
        return self._sim(tasks)


if __name__ == "__main__":
    import sys
    m = ContentionCostModel(sys.argv[1])
    print("O(t) ms:", {t: round(v / 1e6, 3) for t, v in m._O.items()})
    print("derate(8,R):", {r: round(m.derate(8, r), 3) for r in (8, 64, 2048)})
    for demo in ([(2048, 4), (64, 2), (64, 2)], [(8, 1)] * 8):
        print(demo[:2], "phase=%.3fms" % (m.phase_makespan(demo) / 1e6))
