"""Parametric isolated-expert latency model.

The active formula separates route work from thread scaling::

    T_iso(R, t) = O(t) + C(R) * phi_usl(t) * k_phi(t)

    O(t)   = o0 + o1 / t
    phi_usl(t) = (1 + alpha * (t - 1) + beta * t * (t - 1)) / t

``C(R)`` is a one-dimensional calibration curve derived from single-thread
measurements.  This is intentionally much smaller than the old two-dimensional
``routes x threads`` lookup table.  The generalized-USL term represents ideal
``1/t`` speedup plus serial/coherency effects. ``k_phi(t)`` is a small measured
one-dimensional correction curve; it does not depend on routes.

The fitted parameters are valid only inside the calibrated thread domain.
In particular, a negative fitted ``beta`` can describe cache-assisted scaling
inside that domain, but must not be extrapolated to a larger machine.
"""

from __future__ import annotations

import json
import os
import statistics
from bisect import bisect_left
from pathlib import Path
from typing import Iterable


FORMULA_VERSION = 3


def _solve2(
    a11: float,
    a12: float,
    a22: float,
    b1: float,
    b2: float,
) -> tuple[float, float]:
    determinant = a11 * a22 - a12 * a12
    if abs(determinant) < 1e-30:
        return 0.0, 0.0
    return (
        (b1 * a22 - b2 * a12) / determinant,
        (a11 * b2 - a12 * b1) / determinant,
    )


def _fit_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return ``intercept, slope`` for an ordinary least-squares line."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("affine fit requires equally sized, non-empty inputs")
    count = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(value * value for value in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denominator = count * sxx - sx * sx
    slope = (count * sxy - sx * sy) / denominator if denominator else 0.0
    return (sy - slope * sx) / count, slope


class IsoFormula:
    """Fitted ``T_iso`` formula and its calibration domain."""

    def __init__(
        self,
        o0: float,
        o1: float,
        alpha: float,
        beta: float,
        c_pts: Iterable[tuple[int, float]],
        phi_pts: Iterable[tuple[int, float]],
    ):
        self.o0 = float(o0)
        self.o1 = float(o1)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self._cr = sorted((int(route), float(value)) for route, value in c_pts)
        if not self._cr or any(route <= 0 for route, _ in self._cr):
            raise ValueError("C(R) requires at least one positive route anchor")
        self._rs = [route for route, _ in self._cr]
        self._cv = [value for _, value in self._cr]

        self._phi = sorted(
            (int(threads), float(value)) for threads, value in phi_pts
        )
        if not self._phi or self._phi[0][0] != 1:
            raise ValueError("phi calibration must include threads=1")
        self.min_threads = self._phi[0][0]
        self.max_threads = self._phi[-1][0]
        self._pt = [threads for threads, _ in self._phi]
        self._pk = []
        for threads, measured in self._phi:
            baseline = self._phi_usl(threads)
            if baseline <= 0.0:
                raise ValueError(
                    f"fitted USL baseline phi({threads})={baseline} is non-positive"
                )
            self._pk.append(measured / baseline)

    def O(self, threads: float) -> float:
        if threads <= 0:
            raise ValueError(f"threads must be positive, got {threads}")
        return max(self.o0 + self.o1 / threads, 0.0)

    def _phi_usl(self, threads: float) -> float:
        return (
            1.0
            + self.alpha * (threads - 1.0)
            + self.beta * threads * (threads - 1.0)
        ) / threads

    def phi_baseline(self, threads: float) -> float:
        if threads < self.min_threads or threads > self.max_threads:
            raise ValueError(
                f"threads={threads} is outside calibrated domain "
                f"[{self.min_threads}, {self.max_threads}]"
            )
        value = self._phi_usl(threads)
        if value <= 0.0:
            raise ValueError(
                f"fitted phi({threads})={value} is non-positive inside the "
                "calibration domain"
            )
        return value

    def _phi_correction(self, threads: float) -> float:
        if threads <= self._pt[0]:
            return self._pk[0]
        if threads >= self._pt[-1]:
            return self._pk[-1]
        index = bisect_left(self._pt, threads)
        if self._pt[index] == threads:
            return self._pk[index]
        t0, t1 = self._pt[index - 1], self._pt[index]
        return self._pk[index - 1] + (
            (self._pk[index] - self._pk[index - 1])
            * (threads - t0)
            / (t1 - t0)
        )

    def phi(self, threads: float) -> float:
        """Return USL scaling with a route-independent measured correction."""
        return self.phi_baseline(threads) * self._phi_correction(threads)

    def measured_phi(self, threads: int) -> float:
        """Return/interpolate the measured thread-scaling calibration."""
        points = [team for team, _ in self._phi]
        values = [value for _, value in self._phi]
        if threads <= points[0]:
            return values[0]
        if threads >= points[-1]:
            return values[-1]
        index = bisect_left(points, threads)
        if points[index] == threads:
            return values[index]
        t0, t1 = points[index - 1], points[index]
        return values[index - 1] + (
            (values[index] - values[index - 1])
            * (threads - t0)
            / (t1 - t0)
        )

    def C(self, routes: float) -> float:
        """Interpolate single-thread route work; extrapolate linearly at ends."""
        if routes <= 0:
            return 0.0
        if routes <= self._rs[0]:
            return self._cv[0] * routes / self._rs[0]
        if routes >= self._rs[-1]:
            return self._cv[-1] * routes / self._rs[-1]
        index = bisect_left(self._rs, routes)
        r0, r1 = self._rs[index - 1], self._rs[index]
        c0, c1 = self._cv[index - 1], self._cv[index]
        return c0 + (c1 - c0) * (routes - r0) / (r1 - r0)

    def T_iso(self, routes: float, threads: float) -> float:
        if routes <= 0:
            return 0.0
        return self.O(threads) + self.C(routes) * self.phi(threads)

    def to_dict(self) -> dict:
        return {
            "version": FORMULA_VERSION,
            "kind": "separable_route_usl_calibrated",
            "equation": "O(t) + C(R) * phi_usl(t) * k_phi(t)",
            "o0": self.o0,
            "o1": self.o1,
            "alpha": self.alpha,
            "beta": self.beta,
            "c_pts": self._cr,
            "phi_pts": self._phi,
            "thread_domain": [self.min_threads, self.max_threads],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IsoFormula":
        return cls(
            payload["o0"],
            payload["o1"],
            payload["alpha"],
            payload["beta"],
            [tuple(point) for point in payload["c_pts"]],
            [tuple(point) for point in payload["phi_pts"]],
        )


def fit_from_measurements(
    points: Iterable[tuple[int, int, float]],
    phi_route_min: int = 256,
) -> IsoFormula:
    """Fit an :class:`IsoFormula` from ``(routes, threads, ns)`` points.

    ``C(R)`` consumes only the ``threads=1`` measurements. Other thread points
    fit the scalar USL/overhead parameters and the route-independent
    ``k_phi(t)`` curve, so they do not create a two-dimensional runtime table.
    """
    iso = {(int(route), int(team)): float(value) for route, team, value in points}
    if not iso:
        raise ValueError("cannot fit T_iso without measurements")
    threads = sorted({team for _, team in iso})
    routes = sorted({route for route, _ in iso})
    if 1 not in threads:
        raise ValueError("need threads=1 reference points to fit phi(1)=1")

    overhead_by_thread: dict[int, float] = {}
    for team in threads:
        small = sorted(
            (route, iso[(route, team)])
            for route in routes
            if (route, team) in iso and route <= 64
        )
        overhead_by_thread[team] = (
            max(_fit_affine(
                [float(route) for route, _ in small],
                [value for _, value in small],
            )[0], 0.0)
            if len(small) >= 2
            else 0.0
        )

    o0, o1 = _fit_affine(
        [1.0 / team for team in threads],
        [overhead_by_thread[team] for team in threads],
    )
    # _fit_affine returns intercept + slope*x, and x is 1/t.

    def fitted_overhead(team: int) -> float:
        return max(o0 + o1 / team, 0.0)

    def compute_part(route: int, team: int) -> float:
        return max(iso[(route, team)] - fitted_overhead(team), 1e-9)

    c_pts = [
        (route, compute_part(route, 1))
        for route in routes
        if (route, 1) in iso
    ]

    bulk_routes = [route for route in routes if route >= phi_route_min]
    if not bulk_routes:
        bulk_routes = routes
    phi_measured: dict[int, float] = {}
    phi_spread: dict[int, tuple[float, float]] = {}
    for team in threads:
        ratios = [
            compute_part(route, team) / compute_part(route, 1)
            for route in bulk_routes
            if (route, team) in iso and (route, 1) in iso
        ]
        if not ratios:
            raise ValueError(
                f"no route shared by threads=1 and threads={team} for phi fit"
            )
        phi_measured[team] = statistics.median(ratios)
        all_ratios = [
            compute_part(route, team) / compute_part(route, 1)
            for route in routes
            if (route, team) in iso and (route, 1) in iso
        ]
        phi_spread[team] = min(all_ratios), max(all_ratios)

    a11 = a12 = a22 = b1 = b2 = 0.0
    for team in threads:
        if team == 1:
            continue
        x1 = team - 1.0
        x2 = team * (team - 1.0)
        y = phi_measured[team] * team - 1.0
        a11 += x1 * x1
        a12 += x1 * x2
        a22 += x2 * x2
        b1 += x1 * y
        b2 += x2 * y
    alpha, beta = _solve2(a11, a12, a22, b1, b2)

    formula = IsoFormula(
        o0,
        o1,
        alpha,
        beta,
        c_pts,
        [(team, phi_measured[team]) for team in threads],
    )
    # Reject a fit that is invalid at one of its own calibration points.
    for team in threads:
        formula.phi(team)
    formula._diag = {
        "O_by_t": overhead_by_thread,
        "phi_meas": phi_measured,
        "phi_spread": phi_spread,
        "threads": threads,
        "routes": routes,
        "iso": iso,
    }
    return formula


def fit_from_profile(path: str | Path) -> IsoFormula:
    with Path(path).open(encoding="utf-8") as handle:
        profile = json.load(handle)
    return fit_from_measurements(
        (entry["routes"], entry["threads"], entry["median_ns"])
        for entry in profile["isolated"]
    )


def _main(paths: list[str]) -> int:
    for path in paths:
        formula = fit_from_profile(path)
        diag = formula._diag
        print(f"\n=== {os.path.basename(path)} ===")
        print(
            "  O(t) = %.4f + %.4f/t  (ms)"
            % (formula.o0 / 1e6, formula.o1 / 1e6)
        )
        print(
            "  phi_usl(t): alpha=%.6f beta=%.7f, domain=[%d,%d]"
            % (
                formula.alpha,
                formula.beta,
                formula.min_threads,
                formula.max_threads,
            )
        )
        print("  %-4s %10s %10s %10s %10s %18s" % (
            "t", "O_meas", "phi_meas", "phi_usl", "phi_cal", "phi ratio spread"
        ))
        for team in diag["threads"]:
            low, high = diag["phi_spread"][team]
            print(
                "  %-4d %10.4f %10.4f %10.4f %10.4f [%7.3f, %7.3f]"
                % (
                    team,
                    diag["O_by_t"][team] / 1e6,
                    diag["phi_meas"][team],
                    formula.phi_baseline(team),
                    formula.phi(team),
                    low,
                    high,
                )
            )
        errors = [
            abs(formula.T_iso(route, team) - value) / value
            for (route, team), value in diag["iso"].items()
        ]
        worst = sorted(
            (
                abs(formula.T_iso(route, team) - value) / value,
                route,
                team,
            )
            for (route, team), value in diag["iso"].items()
        )[-3:]
        print(
            "  reconstruction: median=%.1f%% max=%.1f%% (n=%d)"
            % (100 * statistics.median(errors), 100 * max(errors), len(errors))
        )
        print(
            "  worst (err%%, R, t):",
            [(round(100 * error, 1), route, team) for error, route, team in worst],
        )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
