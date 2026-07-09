"""Phase-based contention cost model for concurrent MoE expert groups.

Tables (from the block-2 async profile):
  * T_iso(routes, threads)  -- isolated per-expert time
  * derate(n, routes, max_team_threads)
                              -- slowdown when n distinct experts contend
      (~plateau for routes>=16; drops toward ~1.0 for tiny routes / decode)

Per-task overhead split: T_iso(R,t) = O(t) + G(R,t), where O(t) is the fixed
per-task overhead (dispatch/barrier/gather-scatter setup, ~0.12ms; the R->0
intercept of T_iso vs routes). Contention scales only the compute part, so a
task's contended slowdown factor is  fo + (1-fo)*derate,  fo = O/T_iso. This keeps
the tiny-R/decode regime (overhead-dominated) from being over-penalized.

Sims are event-driven with per-task effective rates; each phase uses
derate(#active, max routes among active, max team width among active). No
parameters fit to validation data.
"""
from __future__ import annotations
import json
import math
import os
import statistics
from bisect import bisect_left


class ContentionCostModel:
    def __init__(self, derate_profile_path: str,
                 use_max_team_derate: bool | None = None,
                 use_shape_derate: bool | None = None):
        if use_max_team_derate is None:
            use_max_team_derate = os.environ.get(
                "FUSED_CPP_COST_MODEL_MAX_TEAM_DERATE", "0"
            ) == "1"
        if use_shape_derate is None:
            use_shape_derate = os.environ.get(
                "FUSED_CPP_COST_MODEL_SHAPE_DERATE", "0"
            ) == "1"
        self.use_max_team_derate = use_max_team_derate
        self.use_shape_derate = use_shape_derate
        prof = json.load(open(derate_profile_path))
        self._iso = {(e["routes"], e["threads"]): float(e["median_ns"])
                     for e in prof["isolated"]}
        d2: dict[int, dict[int, list[float]]] = {}
        d3: dict[int, dict[int, dict[int, list[float]]]] = {}
        ds: dict[tuple[int, ...], dict[int, list[float]]] = {}
        for e in prof["entries"]:
            n = int(e["distinct_experts"]); r = int(e["routes"])
            derate = float(e["derate"])
            d2.setdefault(n, {}).setdefault(r, []).append(derate)
            shape = e.get("shape")
            if shape:
                sig = self._shape_signature(int(v) for v in shape)
                mt = sig[0]
                d3.setdefault(n, {}).setdefault(r, {}).setdefault(mt, []).append(derate)
                ds.setdefault(sig, {}).setdefault(r, []).append(derate)
        self._derate2d = {n: {r: statistics.median(v) for r, v in rd.items()}
                          for n, rd in d2.items()}
        self._derate3d = {
            n: {
                r: {mt: statistics.median(v) for mt, v in by_mt.items()}
                for r, by_mt in rd.items()
            }
            for n, rd in d3.items()
        }
        self._derate_shape = {
            sig: {r: statistics.median(v) for r, v in by_route.items()}
            for sig, by_route in ds.items()
        }
        self._dn = sorted(self._derate2d)
        self._dr = sorted({r for rd in self._derate2d.values() for r in rd})
        self._shape_keys = sorted(self._derate_shape)
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
    @staticmethod
    def _shape_signature(threads) -> tuple[int, ...]:
        return tuple(sorted((int(t) for t in threads if int(t) > 0), reverse=True))

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

    def _nearest_route(self, routes: int | None) -> int:
        return self._dr[-1] if routes is None else min(self._dr, key=lambda x: abs(x - routes))

    @staticmethod
    def _interp_log_route(curve: dict[int, float], routes: int | None) -> float:
        rs = sorted(curve)
        if routes is None:
            return curve[rs[-1]]
        if routes <= rs[0]:
            return curve[rs[0]]
        if routes >= rs[-1]:
            return curve[rs[-1]]
        i = bisect_left(rs, routes)
        r0, r1 = rs[i - 1], rs[i]
        x = math.log2(routes)
        x0, x1 = math.log2(r0), math.log2(r1)
        y0, y1 = curve[r0], curve[r1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    @staticmethod
    def _shape_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        width = max(len(a), len(b))
        aa = list(a) + [0] * (width - len(a))
        bb = list(b) + [0] * (width - len(b))
        return sum(abs(x - y) for x, y in zip(aa, bb))

    def _nearest_shape(self, shape: tuple[int, ...]) -> tuple[int, ...] | None:
        if not self._shape_keys:
            return None
        same_n = [key for key in self._shape_keys if len(key) == len(shape)]
        keys = same_n if same_n else self._shape_keys
        return min(keys, key=lambda key: (self._shape_distance(shape, key), key))

    def _shape_derate(self, routes: int | None,
                      shape: tuple[int, ...] | None) -> float | None:
        if not (self.use_shape_derate and shape):
            return None
        sig = self._shape_signature(shape)
        if len(sig) <= 1:
            return 1.0
        key = sig if sig in self._derate_shape else self._nearest_shape(sig)
        if key is None:
            return None
        return self._interp_log_route(self._derate_shape[key], routes)

    def _derate_at_n_route(self, n: int, r: int, max_threads: int | None) -> float:
        if (self.use_max_team_derate and max_threads is not None
                and n in self._derate3d and r in self._derate3d[n]):
            by_mt = self._derate3d[n][r]
            mt = min(by_mt, key=lambda x: abs(x - max_threads))
            return by_mt[mt]
        return self._derate2d[n][r]

    def derate(self, n: int, routes: int | None = None,
               max_threads: int | None = None,
               shape: tuple[int, ...] | None = None) -> float:
        if n <= 1:
            return 1.0
        shape_derate = self._shape_derate(routes, shape)
        if shape_derate is not None:
            return shape_derate
        r = self._nearest_route(routes)
        if n in self._derate2d:
            return self._derate_at_n_route(n, r, max_threads)
        if n <= self._dn[0]:
            return self._derate_at_n_route(self._dn[0], r, max_threads)
        if n >= self._dn[-1]:
            return self._derate_at_n_route(self._dn[-1], r, max_threads)
        i = bisect_left(self._dn, n)
        n0, n1 = self._dn[i - 1], self._dn[i]
        d0 = self._derate_at_n_route(n0, r, max_threads)
        d1 = self._derate_at_n_route(n1, r, max_threads)
        return d0 + (d1 - d0) * (n - n0) / (n1 - n0)

    def _fo(self, routes: int, threads: int) -> float:
        ti = self.T_iso(routes, threads)
        return min(self._O.get(threads, 0.0) / ti, 0.9) if ti > 0 else 0.0

    # ---- predictions -------------------------------------------------------
    def scalar_makespan(self, tasks) -> float:
        d = self.derate(
            len(tasks),
            max(r for (r, _) in tasks),
            max(t for (_, t) in tasks),
            self._shape_signature(t for (_, t) in tasks),
        )
        return max(self.T_iso(r, t) * (self._fo(r, t) + (1 - self._fo(r, t)) * d)
                   for (r, t) in tasks)

    def _sim(self, tasks) -> float:
        """tasks[i]=(routes,threads,deps). Event-driven; per-task effective
        derate fo+(1-fo)*derate(#active, max active routes, max team width)."""
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
            d = self.derate(
                n,
                max(rts[i] for i in active),
                max(thr[i] for i in active),
                self._shape_signature(thr[i] for i in active),
            )
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
