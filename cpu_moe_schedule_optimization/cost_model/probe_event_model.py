"""Probe-calibrated fused event model (v10) for placed CPU MoE plans.

Formulas: ``cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`` ("探针校准的融合
event 模型（v10）"). The model keeps the phase skeleton of
:class:`analytic_model.AnalyticMoeCostModel` (per-task setup, cold-B and steady-B
phases of W13/W2) and replaces its shared-resource service with measured
dilation curves:

* isolated phase time ``tau = (1 + eps) * (fixed + max(local, compulsory / R) + epilogue)``;
* per-expert overhead ``O(t) = o0 + o1 * t`` for teams of at least four threads;
* a phase is *loading* (cold-B) or *steady*; in an event with ``n_L`` other cores
  loading and ``n_S`` other cores steady a phase of kind ``k`` runs
  ``1 + (D_kL(t, n_L) - 1) + (D_kS(t, n_S) - 1)`` times its isolated time;
* windowed tasks run ``(1 - g0)`` of their isolated time alone and ``r_cal`` of
  their full-stripe time at the window table's calibration load, linear in the
  modeled excess in between;
* a whole call adds ``t_over`` to the event makespan.

The curves and scalars come from a calibration JSON (schema
``moe_probe_event_model``) that names the analytic calibration supplying the
phase skeleton. All planner-facing methods not defined here are delegated to
the wrapped analytic model, except native planner exports, which would bypass
this objective.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

try:
    from analytic_model import AnalyticMoeCostModel
except ImportError:  # pragma: no cover - package-style import
    from .analytic_model import AnalyticMoeCostModel


PROBE_EVENT_MODEL_SCHEMA = "moe_probe_event_model"
PROBE_EVENT_MODEL_NAME = "probe_calibrated_fused_event_v10"
_CURVES = ("LL", "LS", "SL", "SS")
_MAX_CORES = 1024
_RESOLUTION = 4  # curve tables per quarter core (fractional loading-equivalent core counts)


@lru_cache(maxsize=None)
def _tabulate(points: tuple) -> tuple[float, ...]:
    return tuple(_piecewise(points, index / _RESOLUTION) for index in range(_MAX_CORES * _RESOLUTION + 1))


def _log_interp(points: Sequence[Sequence[float]], x: float) -> float:
    """Linear in log(x) between points [(x_k, y_k)] sorted by x; clamped outside."""
    if x <= points[0][0]:
        return float(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return float(y0 + f * (y1 - y0))
    return float(points[-1][1])


def _piecewise(points: Sequence[Sequence[float]], cores: float) -> float:
    if cores <= points[0][0]:
        return float(points[0][1])
    for (n0, d0), (n1, d1) in zip(points, points[1:]):
        if cores <= n1:
            return float(d0 + (d1 - d0) * (cores - n0) / (n1 - n0))
    return float(points[-1][1])


class ProbeCurves:
    """Measured dilation curves D_xy(t, n), tabulated on integer core counts."""

    def __init__(self, curves: Mapping[str, Mapping[str, Sequence[Sequence[float]]]], max_cores: int):
        if set(curves) != set(_CURVES):
            raise ValueError(f"probe curves must be exactly {_CURVES}, got {sorted(curves)}")
        self.points = {
            name: {int(width): tuple(tuple(map(float, point)) for point in points) for width, points in table.items()}
            for name, table in curves.items()
        }
        for name, table in self.points.items():
            for width, points in table.items():
                if width <= 0 or not points or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
                    raise ValueError(f"curve {name} width {width} needs increasing core counts")
                # Measured points may sit slightly below one (session noise); larger gains are errors.
                if any(point[1] < 0.9 for point in points):
                    raise ValueError(f"curve {name} width {width} has a dilation below 0.9")
        self.max_cores = int(max_cores)
        self._tables: dict[tuple[str, int], tuple[float, ...]] = {}

    def value(self, name: str, width: int, cores: float) -> float:
        """Piecewise linear in cores, linear in log2(width) between measured widths, clamped outside."""
        table = self.points[name]
        widths = sorted(table)
        width = min(max(int(width), widths[0]), widths[-1])
        lower = max(w for w in widths if w <= width)
        upper = min(w for w in widths if w >= width)
        low = _piecewise(table[lower], cores)
        if lower == upper:
            return low
        fraction = (math.log2(width) - math.log2(lower)) / (math.log2(upper) - math.log2(lower))
        return low + fraction * (_piecewise(table[upper], cores) - low)

    def table(self, name: str, width: int) -> tuple[float, ...]:
        """D at cores = index / _RESOLUTION."""
        key = (name, int(width))
        cached = self._tables.get(key)
        if cached is None:
            cached = tuple(self.value(name, width, index / _RESOLUTION)
                           for index in range(self.max_cores * _RESOLUTION + 1))
            self._tables[key] = cached
        return cached


class ProbeEventModel:
    """Planner cost model scoring placed plans with the v10 probe-calibrated event simulation."""

    iso_mode = "analytic"
    model_name = PROBE_EVENT_MODEL_NAME
    # Native planners carry their own (v8/v9) service model; keep them off this objective.
    native_interval_planner_payload = None
    native_quick_planner_payload = None

    def __init__(self, base: AnalyticMoeCostModel, calibration: Mapping[str, object] | str | Path, *,
                 window_policy=None):
        if isinstance(calibration, (str, Path)):
            self.calibration_path = Path(calibration)
            payload = json.loads(self.calibration_path.read_text())
        else:
            self.calibration_path = None
            payload = dict(calibration)
        if payload.get("schema") != PROBE_EVENT_MODEL_SCHEMA:
            raise ValueError(f"expected schema {PROBE_EVENT_MODEL_SCHEMA!r}, got {payload.get('schema')!r}")
        self.base = base
        self.eps = float(payload["eps"])
        self.o0_ns = float(payload["o0_ns"])
        self.o1_ns = float(payload["o1_ns"])
        self.g0 = float(payload["g0"])
        self.t_over_ns = float(payload["t_over_ns"])
        self.overhead_min_threads = int(payload.get("overhead_min_threads", 4))
        if min(self.eps, self.o0_ns, self.o1_ns, self.g0, self.t_over_ns) < 0.0 or self.g0 >= 1.0:
            raise ValueError("probe event scalars must be non-negative and g0 < 1")
        self.curves = ProbeCurves(payload["curves"], max_cores=_MAX_CORES)
        # v11 extensions (optional; absent in the v10 asset, where they reduce to v10 exactly):
        # isolated_correction {width: [[M, c], ...]}: measured / modeled isolated per-expert time;
        # steady_curves_by_m {"SL"|"SS": {width: [[M_mid, [[n, D], ...]], ...]}}: M-indexed steady curves;
        # steady_loading_weight [[M, w], ...]: share of a steady phase's cores counted as loading.
        self.isolated_correction = {int(w): tuple(map(tuple, pts))
                                    for w, pts in payload.get("isolated_correction", {}).items()}
        self.steady_by_m = {
            name: {int(w): tuple((float(m), tuple(map(tuple, pts))) for m, pts in sorted(entries, key=lambda e: e[0]))
                   for w, entries in table.items()}
            for name, table in payload.get("steady_curves_by_m", {}).items()
        }
        self.loading_weight = tuple(map(tuple, payload.get("steady_loading_weight", ())))
        # v12: background lane width factors on the loading-neighbour excess,
        # {"L"|"S": {target width: {background width: [[n, f], ...]}}}; absent -> factor 1.
        # v13: lanes this narrow load their neighbours like a loading phase in every phase (P8).
        self.loading_weight_by_width = {int(w): float(v) for w, v in payload.get("loading_weight_by_width", {}).items()}
        self.width_factors = {
            kind: {int(t): {int(w): tuple(map(tuple, points)) for w, points in table.items()}
                   for t, table in per_target.items()}
            for kind, per_target in payload.get("background_width_factors", {}).items()
        }
        self.payload = payload
        self.window_policy = window_policy
        self._t_iso_cache: dict[tuple[int, int], float] = {}

    @classmethod
    def from_calibration(cls, calibration: str | Path, *, window_policy=None, **analytic_kwargs) -> "ProbeEventModel":
        """Build the wrapped analytic model from the calibration's ``base_calibration`` path."""
        path = Path(calibration)
        payload = json.loads(path.read_text())
        base_path = Path(payload["base_calibration"])
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        return cls(AnalyticMoeCostModel(str(base_path), **analytic_kwargs), payload, window_policy=window_policy)

    def __getattr__(self, name: str):
        # Only reached for attributes not defined here: shapes, widths, bytes, uncertainty, ...
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    # ---- isolated task time

    def correction(self, routes: int, threads: int) -> float:
        points = self.isolated_correction.get(int(threads))
        return 1.0 if not points or routes > points[-1][0] * 1.5 else _log_interp(points, routes)

    @lru_cache(maxsize=None)
    def phase_table(self, routes: int, threads: int) -> tuple[tuple[float, bool], ...]:
        """Per phase of one expert: (isolated ns, is loading phase)."""
        scale = self.correction(int(routes), int(threads))
        rows = []
        for phase in self.base.predict_expert(int(routes), int(threads)).phases:
            local = max(phase.gemm_core_ns, phase.l2_ns, phase.llc_ns)
            budget = phase.compulsory_dram_bytes
            transfer = budget / phase.dram_rate * 1e9 if budget > 0 else 0.0
            tau = (1.0 + self.eps) * (phase.fixed_ns + phase.epilogue_ns + max(local, transfer)) * scale
            rows.append((tau, phase.kind == "cold_b"))
        return tuple(rows)

    def overhead_ns(self, threads: int, routes: int | None = None) -> float:
        base = self.o0_ns + self.o1_ns * threads if threads >= self.overhead_min_threads else 0.0
        return base if routes is None else base * self.correction(int(routes), int(threads))

    def T_iso(self, routes: int, threads: int) -> float:
        key = (int(routes), int(threads))
        value = self._t_iso_cache.get(key)
        if value is None:
            value = self.overhead_ns(key[1], key[0]) + sum(tau for tau, _ in self.phase_table(*key))
            self._t_iso_cache[key] = value
        return value

    def t_iso_cache_identity(self) -> dict[str, object]:
        """Every stable input needed to validate persisted T_iso values of this model."""
        payload = json.dumps(self.payload, sort_keys=True).encode()
        return {
            "model": PROBE_EVENT_MODEL_NAME,
            "calibration_sha256": hashlib.sha256(payload).hexdigest(),
            "base": self.base.t_iso_cache_identity(),
        }

    def import_t_iso_cache(self, entries: Mapping[tuple[int, int], float]) -> int:
        loaded = 0
        for (routes, threads), value in entries.items():
            routes, threads, value = int(routes), int(threads), float(value)
            if routes <= 0 or threads not in self.supported_widths or not math.isfinite(value) or value <= 0.0:
                continue
            self._t_iso_cache[(routes, threads)] = value
            loaded += 1
        return loaded

    def export_t_iso_cache(self) -> dict[tuple[int, int], float]:
        return dict(self._t_iso_cache)

    @lru_cache(maxsize=None)
    def steady_tables(self, routes: int, threads: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """(D_SL, D_SS) tables for a steady phase of a task with ``routes`` on ``threads``."""
        out = []
        for name in ("SL", "SS"):
            entries = self.steady_by_m.get(name, {}).get(int(threads))
            if not entries:
                out.append(self.curves.table(name, threads))
                continue
            tables = [(m, _tabulate(points)) for m, points in entries]
            if routes <= tables[0][0]:
                out.append(tables[0][1])
                continue
            if routes >= tables[-1][0]:
                out.append(tables[-1][1])
                continue
            for (m0, t0), (m1, t1) in zip(tables, tables[1:]):
                if routes <= m1:
                    f = (math.log(routes) - math.log(m0)) / (math.log(m1) - math.log(m0))
                    out.append(tuple(a + f * (b - a) for a, b in zip(t0, t1)))
                    break
        return tuple(out)

    @lru_cache(maxsize=None)
    def width_factor(self, kind: str, threads: int, cores: float, background_width: float) -> float:
        """Excess scale of a phase whose loading neighbours run on ``background_width`` lanes."""
        table = self.width_factors.get(kind)
        if not table:
            return 1.0
        targets = sorted(table)
        target = min(targets, key=lambda t: (abs(math.log2(t) - math.log2(max(threads, 1))), t))
        rows = table[target]
        widths = sorted(rows)
        value = min(max(background_width, widths[0]), widths[-1])
        lower = max(w for w in widths if w <= value)
        upper = min(w for w in widths if w >= value)
        low = _piecewise(rows[lower], cores)
        if lower == upper:
            return low
        fraction = (math.log2(value) - math.log2(lower)) / (math.log2(upper) - math.log2(lower))
        return low + fraction * (_piecewise(rows[upper], cores) - low)

    @lru_cache(maxsize=None)
    def steady_weight(self, routes: int) -> float:
        """Share of a steady phase's cores counted as loading cores for other tasks."""
        if not self.loading_weight:
            return 0.0
        return min(max(_log_interp(self.loading_weight, routes), 0.0), 1.0)

    # ---- windows

    @lru_cache(maxsize=None)
    def window_terms(self, routes: int, threads: int, r_cal: float, cores: int) -> tuple[float, float] | None:
        """(r_cal, e_cal): e_cal is the task's modeled excess at the table's calibration load.

        The calibration load is every other core running the same (routes, threads) task,
        split between loading and steady phases in proportion to the task's isolated time.
        """
        if r_cal >= 1.0:
            return None
        rows = self.phase_table(routes, threads)
        total = sum(tau for tau, _ in rows)
        loading = sum(tau for tau, is_load in rows if is_load) / total
        others = cores - threads
        weight = self.steady_weight(routes)
        n_load = loading * others + (1.0 - loading) * others * weight
        n_steady = (1.0 - loading) * others * (1.0 - weight)
        excess = 0.0
        sl, ss = self.steady_tables(routes, threads)
        index_l, index_s = int(round(n_load * _RESOLUTION)), int(round(n_steady * _RESOLUTION))
        for tau, is_load in rows:
            if is_load:
                dilation = (1.0 + (self.curves.value("LL", threads, n_load) - 1.0)
                            + (self.curves.value("LS", threads, n_steady) - 1.0))
            elif self.steady_by_m:
                dilation = sl[index_l] + ss[index_s] - 1.0
            else:
                dilation = (1.0 + (self.curves.value("SL", threads, n_load) - 1.0)
                            + (self.curves.value("SS", threads, n_steady) - 1.0))
            excess += tau * (dilation - 1.0)
        return (float(r_cal), excess / total)

    def task_windows(self, tasks, cores: int | None = None) -> list[tuple[float, float] | None] | None:
        """Window terms of placed tasks under ``window_policy`` (None: full stripe everywhere)."""
        policy = self.window_policy
        if policy is None:
            return None
        cores = int(cores if cores is not None else self.base.calibration.cores_per_rank)
        windows = []
        for routes, threads, *_ in tasks:
            selected = policy.select(int(routes), int(threads))
            windowed = any(int(value) != 0 for value in selected)
            windows.append(self.window_terms(int(routes), int(threads), float(policy.time_scale(int(routes), int(threads))), cores)
                           if windowed else None)
        return windows if any(window is not None for window in windows) else None

    # ---- event simulation

    def simulate(self, tasks, windows=None, *, event_log: list | None = None) -> dict:
        """Placed DAG simulation.

        ``tasks``: (routes, threads, cpu_ids, dependencies) with dependencies on earlier tasks.
        ``windows``: per task None or (r_cal, e_cal) from :meth:`window_terms`; default from
        ``window_policy``.
        """
        tasks = [(int(r), int(w), tuple(c), tuple(int(d) for d in deps)) for r, w, c, deps in tasks]
        if windows is None:
            windows = self.task_windows(tasks)
        n = len(tasks)
        res = _RESOLUTION
        tables = [self.phase_table(r, w) for r, w, _, _ in tasks]
        widths = [w for _, w, _, _ in tasks]
        overhead = [self.overhead_ns(w, r) for r, w, _, _ in tasks]
        loading_curves = {w: (self.curves.table("LL", w), self.curves.table("LS", w)) for w in set(widths)}
        steady_curves = [self.steady_tables(r, w) for r, w, _, _ in tasks]
        # Cores a task adds to the (loading, steady) counters while in a steady phase.
        # A lane narrow enough to stream weights for its whole life counts as loading throughout (P8).
        weight = [self.loading_weight_by_width.get(w, self.steady_weight(r)) for r, w, _, _ in tasks]
        hook = getattr(self, "_scale_hook", None)
        if hook is not None:  # calibration inversion only
            weight = [min(v * hook(i), 1.0) for i, v in enumerate(weight)]
        steady_share = [(weight[i] * w, (1.0 - weight[i]) * w) for i, (_, w, _, _) in enumerate(tasks)]
        pending = [len(deps) for _, _, _, deps in tasks]
        successors: list[list[int]] = [[] for _ in range(n)]
        for index, (_, _, _, deps) in enumerate(tasks):
            for dep in deps:
                if not 0 <= dep < index:
                    raise ValueError("placed task dependencies must refer to earlier tasks")
                successors[dep].append(index)
        phase = [0] * n
        remaining = [1.0] * n
        active: set[int] = set()
        counters = [0.0, 0.0, 0.0]  # loading cores, steady cores, sum of loading cores x ln(width)
        log_width = [math.log(w) for w in widths]

        def share(index: int) -> tuple[float, float]:
            p = phase[index]
            if p < 0 or p >= len(tables[index]):
                return (0.0, 0.0)
            return (float(widths[index]), 0.0) if tables[index][p][1] else steady_share[index]

        def enter(index: int) -> None:
            while phase[index] < len(tables[index]) and tables[index][phase[index]][0] <= 0.0:
                phase[index] += 1
            dl, ds = share(index)
            counters[0] += dl
            counters[1] += ds
            counters[2] += dl * log_width[index]

        def leave(index: int) -> None:
            dl, ds = share(index)
            counters[0] -= dl
            counters[1] -= ds
            counters[2] -= dl * log_width[index]

        def start(index: int) -> None:
            remaining[index] = 1.0
            active.add(index)
            if overhead[index] > 0.0:
                phase[index] = -1
                return
            phase[index] = 0
            enter(index)

        for index in range(n):
            if pending[index] == 0:
                start(index)
        now = 0.0
        finish = [0.0] * n
        while active:
            duration = {}
            load_cores, steady_cores, load_log = counters
            for index in active:
                p = phase[index]
                if p < 0:
                    duration[index] = overhead[index]
                    continue
                tau, is_load = tables[index][p]
                if is_load:
                    own_load, own_steady = float(widths[index]), 0.0
                    ll, ls = loading_curves[widths[index]]
                    others = load_cores - own_load
                    kind, loading_excess = "L", ll[max(int(round(others * res)), 0)] - 1.0
                    steady_excess = ls[max(int(round(steady_cores * res)), 0)] - 1.0
                else:
                    own_load, own_steady = steady_share[index]
                    sl, ss = steady_curves[index]
                    others = load_cores - own_load
                    kind, loading_excess = "S", sl[max(int(round(others * res)), 0)] - 1.0
                    steady_excess = ss[max(int(round((steady_cores - own_steady) * res)), 0)] - 1.0
                if self.width_factors and loading_excess > 0.0 and others > 1e-9:
                    # Effective background lane width: the core-weighted geometric mean of the
                    # widths of the other tasks counted as loading.
                    background = math.exp((load_log - own_load * log_width[index]) / others)
                    loading_excess *= self.width_factor(kind, widths[index], others, background)
                dilation = 1.0 + loading_excess + steady_excess
                value = tau * dilation
                window = windows[index] if windows else None
                if window is not None:
                    r_cal, e_cal = window
                    if e_cal < 0.01 or dilation - 1.0 >= e_cal:
                        value *= r_cal
                    else:
                        level = (dilation - 1.0) / e_cal
                        value = tau * ((1.0 - self.g0) + level * (r_cal * (1.0 + e_cal) - (1.0 - self.g0)))
                duration[index] = value
            step = min(remaining[index] * duration[index] for index in active)
            if event_log is not None:
                event_log.append({
                    "start_ns": now,
                    "duration_ns": step,
                    "active_tasks": sorted(active),
                    "phase_dilation": {
                        str(index): (duration[index] / (tables[index][phase[index]][0] if phase[index] >= 0 else overhead[index]))
                        for index in active
                    },
                })
            now += step
            done = []
            for index in active:
                remaining[index] -= step / duration[index] if duration[index] > 0 else 1.0
                if remaining[index] <= 1e-12:
                    done.append(index)
            for index in done:
                leave(index)
                phase[index] += 1
                remaining[index] = 1.0
                enter(index)
                if phase[index] >= len(tables[index]):
                    active.discard(index)
                    finish[index] = now
                    for successor in successors[index]:
                        pending[successor] -= 1
                        if pending[successor] == 0:
                            start(successor)
        if any(pending):
            raise ValueError("placed task dependencies contain a cycle")
        return {"makespan_ns": now, "task_finish_ns": finish, "call_ns": now + self.t_over_ns}

    # ---- planner-facing API

    def dag_makespan_placed(self, tasks) -> float:
        return self.simulate(tasks)["makespan_ns"]

    def call_time_placed(self, tasks) -> float:
        """Predicted whole-call time: event makespan plus the measured call overhead."""
        return self.simulate(tasks)["call_ns"]

    def explain_dag_placed(self, tasks) -> dict:
        events: list[dict] = []
        result = self.simulate(tasks, event_log=events)
        return {
            "model": PROBE_EVENT_MODEL_NAME,
            "machine_id": self.base.calibration.machine_id,
            "makespan_ns": result["makespan_ns"],
            "task_finish_ns": list(result["task_finish_ns"]),
            "events": events,
        }

    def _unplaced(self, tasks):
        # The probe response reads only width and concurrency, not physical cores.
        placed, cursor = [], 0
        for routes, threads, deps in tasks:
            placed.append((routes, threads, tuple(range(cursor, cursor + int(threads))), tuple(deps)))
            cursor += int(threads)
        return placed

    def dag_makespan(self, tasks) -> float:
        return self.simulate(self._unplaced(tasks))["makespan_ns"]

    def dag_task_finish_times(self, tasks) -> tuple[float, ...]:
        return tuple(self.simulate(self._unplaced(tasks))["task_finish_ns"])

    @property
    def calibrated_widths(self) -> tuple[int, ...]:
        """Lane widths the probes actually measured; anything else is an extrapolation.

        The curves clamp outside this range, so a plan that puts work on an uncalibrated width
        is scored with the nearest measured one. Measured consequence (tmp/search_reliability_20260920,
        E8): tail-pool plans that pool onto one thread run 3.2-4.3x the predicted isolated time
        of those tasks, which is why the planner must not choose widths outside this set.
        """
        widths = [set(int(w) for w in table) for table in self.curves.points.values()]
        return tuple(sorted(set.intersection(*widths))) if widths else ()

    @property
    def reliable_widths(self) -> tuple[int, ...]:
        """Calibrated widths minus the ones whole-plan measurements refuted (``unreliable_widths``).

        Lanes of two threads are measured by the probes but whole plans built from them run about
        24% above the prediction (E2, E5), so the planner must not choose them by itself.
        """
        blocked = {int(w) for w in self.payload.get("unreliable_widths", ())}
        return tuple(w for w in self.calibrated_widths if w not in blocked)

    def supports_width(self, width: int) -> bool:
        reliable = self.reliable_widths
        return not reliable or int(width) in reliable

    def quick_homogeneous_scale(self, width: int, occupied: int, num_cores: int) -> float:
        """Width-ranking scale for quick: none; quick ranks on isolated v10 task times."""
        return 1.0


__all__ = ["PROBE_EVENT_MODEL_NAME", "PROBE_EVENT_MODEL_SCHEMA", "ProbeCurves", "ProbeEventModel"]
