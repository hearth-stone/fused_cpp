"""Parametric T_iso model + fitter.

Formula (separable compute-x-thread, additive overhead):

    T_iso(R, t) = O(t)  +  C(R) * phi(t)

    phi(t) = (1 + alpha*(t-1) + beta*t*(t-1)) / t      # USL, phi(1)=1
    O(t)   = o0 + o1 / t                                # fixed-overhead curve
    C(R)   = single-thread compute-time route shape (1D interp of the
             measured G(R,1) = T_iso(R,1) - O(1); route curvature is real
             and non-linear, so it stays tabular rather than forced to k*R)

phi(t) gives ANY thread count from 2 params {alpha, beta} (validated ~0.3-3%
on 8-core profiles). alpha = serial/contention, beta = coherency/BW-turnover
term (~0 at 8 cores; turns positive at the high-thread N-split knee -> must be
anchored by high-t measurements, cannot be extrapolated from {1,2,4,8}).

Params are fit either from a profile's isolated[] table (`fit_from_profile`)
or from live (R, t, ns) probes (`fit_from_measurements`).
"""
from __future__ import annotations
import json
import statistics
from bisect import bisect_left


def _solve2(a11, a12, a22, b1, b2):
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-30:
        return 0.0, 0.0
    return (b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det


class IsoFormula:
    def __init__(self, o0, o1, alpha, beta, c_pts, phi_pts):
        self.o0, self.o1 = o0, o1
        self.alpha, self.beta = alpha, beta
        self._cr = sorted(c_pts)                       # [(R, C(R))]
        self._rs = [r for r, _ in self._cr]
        self._cv = [c for _, c in self._cr]
        self._phi = sorted(phi_pts)                    # [(t, phi_meas(t))], t=1 -> 1
        self._pt = [t for t, _ in self._phi]
        self._pv = [v for _, v in self._phi]

    def O(self, t: float) -> float:
        return max(self.o0 + self.o1 / t, 0.0)

    def _usl(self, t: float) -> float:
        return (1.0 + self.alpha * (t - 1) + self.beta * t * (t - 1)) / t

    def phi(self, t: float) -> float:
        """Measured-point interpolation within [1, t_max]; USL-shape extrapolation
        beyond t_max (anchored at t_max for continuity)."""
        pt, pv = self._pt, self._pv
        if t <= pt[0]:
            return pv[0]
        if t <= pt[-1]:
            i = bisect_left(pt, t)
            if pt[i] == t:
                return pv[i]
            t0, t1 = pt[i - 1], pt[i]
            return pv[i - 1] + (pv[i] - pv[i - 1]) * (t - t0) / (t1 - t0)
        return pv[-1] * self._usl(t) / self._usl(pt[-1])   # extrapolate

    def C(self, R: float) -> float:
        rs, cv = self._rs, self._cv
        if R <= rs[0]:
            return cv[0] * R / rs[0]                    # linear-to-origin
        if R >= rs[-1]:
            return cv[-1] * R / rs[-1]
        i = bisect_left(rs, R)
        r0, r1 = rs[i - 1], rs[i]
        return cv[i - 1] + (cv[i] - cv[i - 1]) * (R - r0) / (r1 - r0)

    def T_iso(self, R: float, t: float) -> float:
        return self.O(t) + self.C(R) * self.phi(t)

    def to_dict(self):
        return {"o0": self.o0, "o1": self.o1, "alpha": self.alpha,
                "beta": self.beta, "c_pts": self._cr, "phi_pts": self._phi}

    @classmethod
    def from_dict(cls, d):
        return cls(d["o0"], d["o1"], d["alpha"], d["beta"],
                   [tuple(p) for p in d["c_pts"]], [tuple(p) for p in d["phi_pts"]])


def fit_from_measurements(points, phi_route_min: int = 256):
    """points: iterable of (routes, threads, ns). Returns IsoFormula.

    phi_route_min: phi(t) is fit from ratios at routes >= this threshold, i.e.
    the compute-efficient / makespan-dominant regime. Small-R is overhead-
    dominated (handled by O(t)) and scales worse per-thread, so including it
    would bias phi and over-predict the large-R prefill case."""
    iso = {(int(r), int(t)): float(v) for r, t, v in points}
    threads = sorted({t for _, t in iso})
    routes = sorted({r for r, _ in iso})
    if 1 not in threads:
        raise ValueError("need t=1 reference points to fit phi(1)=1")

    # --- O(t): small-R (R<=64) linear-fit intercept per thread ---
    O_by_t = {}
    for t in threads:
        pts = sorted((r, iso[(r, t)]) for r in routes if (r, t) in iso and r <= 64)
        if len(pts) >= 2:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            m = len(xs); sx = sum(xs); sy = sum(ys)
            sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
            den = m * sxx - sx * sx
            k = (m * sxy - sx * sy) / den if den else 0.0
            O_by_t[t] = max((sy - k * sx) / m, 0.0)
        else:
            O_by_t[t] = 0.0

    # --- O(t) = o0 + o1/t  (LS on 1/t) ---
    xs = [1.0 / t for t in threads]; ys = [O_by_t[t] for t in threads]
    m = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = m * sxx - sx * sx
    o1 = (m * sxy - sx * sy) / den if den else 0.0
    o0 = (sy - o1 * sx) / m

    # --- compute part G(R,t) = T_iso - O(t);  C(R) = G(R,1) ---
    def G(r, t):
        return max(iso[(r, t)] - O_by_t[t], 1e-9)
    c_pts = [(r, G(r, 1)) for r in routes if (r, 1) in iso]

    # --- phi_meas(t) = median_R G(R,t)/G(R,1) over the large-R (efficient) regime ---
    hi = [r for r in routes if r >= phi_route_min] or routes
    phi_meas, phi_spread = {}, {}
    for t in threads:
        ratios = [G(r, t) / G(r, 1) for r in hi if (r, t) in iso and (r, 1) in iso]
        phi_meas[t] = statistics.median(ratios)
        allr = [G(r, t) / G(r, 1) for r in routes if (r, t) in iso and (r, 1) in iso]
        phi_spread[t] = (min(allr), max(allr))

    # --- USL fit: phi(t)*t - 1 = alpha*(t-1) + beta*t*(t-1)  (LS, no intercept) ---
    a11 = a12 = a22 = b1 = b2 = 0.0
    for t in threads:
        if t == 1:
            continue
        x1 = (t - 1); x2 = t * (t - 1); y = phi_meas[t] * t - 1.0
        a11 += x1 * x1; a12 += x1 * x2; a22 += x2 * x2
        b1 += x1 * y; b2 += x2 * y
    alpha, beta = _solve2(a11, a12, a22, b1, b2)
    alpha = max(alpha, 0.0); beta = max(beta, 0.0)

    f = IsoFormula(o0, o1, alpha, beta, c_pts, [(t, phi_meas[t]) for t in threads])
    f._diag = {"O_by_t": O_by_t, "phi_meas": phi_meas, "phi_spread": phi_spread,
               "threads": threads, "routes": routes, "iso": iso}
    return f


def fit_from_profile(path):
    prof = json.load(open(path))
    return fit_from_measurements(
        (e["routes"], e["threads"], e["median_ns"]) for e in prof["isolated"])


if __name__ == "__main__":
    import sys, os
    for path in sys.argv[1:]:
        f = fit_from_profile(path)
        d = f._diag
        print("\n===", os.path.basename(path), "===")
        print("  O(t) = %.4f + %.4f/t  (ms)" % (f.o0 / 1e6, f.o1 / 1e6))
        print("  phi(t) USL: alpha=%.4f  beta=%.5f" % (f.alpha, f.beta))
        print("  %-4s %10s %10s %8s   %-18s" % ("t", "O_meas(ms)", "phi_meas", "phi_fit", "phi ratio spread"))
        for t in d["threads"]:
            lo, hi = d["phi_spread"][t]
            print("  %-4d %10.4f %10.4f %8.4f   [%.3f, %.3f]"
                  % (t, d["O_by_t"][t] / 1e6, d["phi_meas"][t], f.phi(t), lo, hi))
        # reconstruction error vs the measured table
        errs = [abs(f.T_iso(r, t) - v) / v for (r, t), v in d["iso"].items()]
        big = sorted((abs(f.T_iso(r, t) - v) / v, r, t) for (r, t), v in d["iso"].items())[-3:]
        print("  reconstruction err vs table: median=%.1f%% max=%.1f%%  (n=%d)"
              % (100 * statistics.median(errs), 100 * max(errs), len(errs)))
        print("  worst points (err,R,t):", [(round(100 * e, 1), r, t) for e, r, t in big])
