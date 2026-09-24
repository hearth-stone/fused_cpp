"""Native scorer for plans under the probe-calibrated event model.

``NativeProbeEventScorer`` wraps a :class:`probe_event_model.ProbeEventModel` and runs its
``simulate`` in C++ (``NativeProbeEventSim``), so a search can score many candidate plans on the
request path. Every per-task input (phase times, loading flags, overhead, steady and loading curve
tables, loading weight, window terms) is taken from the wrapped model, and every other attribute
is delegated to it. Background width factors, event logs, explicit windows and calibration hooks
stay on the Python model.
"""

from __future__ import annotations

import importlib

try:
    from probe_event_model import _RESOLUTION
except ImportError:  # pragma: no cover - package-style import
    from .probe_event_model import _RESOLUTION


def _native_type():
    for module_name in ("fused_cpp._moe_C", "fused_cpp._C"):
        try:
            return importlib.import_module(module_name).NativeProbeEventSim
        except (ImportError, AttributeError):
            continue
    return None


class NativeProbeEventScorer:
    """Drop-in ``simulate`` for a ProbeEventModel, backed by the native event loop."""

    def __init__(self, model) -> None:
        native_type = _native_type()
        if native_type is None:
            raise RuntimeError("the native probe event simulator is not built")
        self.model = model
        self._sim = native_type(_RESOLUTION, float(model.g0))
        self._tables: dict[int, tuple[object, int]] = {}
        self._profiles: dict[tuple[int, int], int] = {}

    def __getattr__(self, name: str):
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def _table(self, values) -> int:
        # Curve tables are cached tuples owned by the model; keep a reference so ids stay unique.
        entry = self._tables.get(id(values))
        if entry is None:
            entry = (values, self._sim.add_table(list(values)))
            self._tables[id(values)] = entry
        return entry[1]

    def _profile(self, routes: int, width: int) -> int:
        key = (routes, width)
        profile = self._profiles.get(key)
        if profile is not None:
            return profile
        model = self.model
        rows = model.phase_table(routes, width)
        steady_l, steady_s = model.steady_tables(routes, width)
        weight = model.loading_weight_by_width.get(width, model.steady_weight(routes))
        window = None
        policy = model.window_policy
        if policy is not None and any(int(value) != 0 for value in policy.select(routes, width)):
            window = model.window_terms(routes, width, float(policy.time_scale(routes, width)),
                                        int(model.base.calibration.cores_per_rank))
        profile = self._sim.add_profile(
            [float(tau) for tau, _ in rows], [bool(is_load) for _, is_load in rows],
            float(model.overhead_ns(width, routes)), width, weight * width, (1.0 - weight) * width,
            self._table(model.curves.table("LL", width)), self._table(model.curves.table("LS", width)),
            self._table(steady_l), self._table(steady_s),
            window is not None, float(window[0]) if window else 1.0, float(window[1]) if window else 0.0,
        )
        self._profiles[key] = profile
        return profile

    def _native_ready(self, windows, event_log) -> bool:
        model = self.model
        return (windows is None and event_log is None and not model.width_factors
                and getattr(model, "_scale_hook", None) is None)

    def simulate(self, tasks, windows=None, *, event_log: list | None = None) -> dict:
        if not self._native_ready(windows, event_log):
            return self.model.simulate(tasks, windows, event_log=event_log)
        profiles, dependencies = [], []
        for routes, width, _cpus, deps in tasks:
            profiles.append(self._profile(int(routes), int(width)))
            dependencies.append([int(dep) for dep in deps])
        makespan, finish = self._sim.simulate(profiles, dependencies)
        return {"makespan_ns": makespan, "task_finish_ns": list(finish), "call_ns": makespan + self.model.t_over_ns}

    def dag_makespan_placed(self, tasks) -> float:
        return self.simulate(tasks)["makespan_ns"]

    def call_time_placed(self, tasks) -> float:
        return self.simulate(tasks)["call_ns"]


__all__ = ["NativeProbeEventScorer"]
