"""Policy-bound, exact-shape and stage-aware MoE contention model.

Schema-v2 profiles bind measurements to a sharded expert shape, split-W13
policy, kernel binary, NUMA topology, and concurrent-rank count.  The model
keeps the validated event-driven DAG mechanics, but models W13 chunks and W2 as
separate working-set phases.  Legacy schema-v1 profiles retain the old flat-task
behavior for reproducibility.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from bisect import bisect_left
from pathlib import Path

try:
    from iso_formula import IsoFormula, fit_from_measurements
    from profile_catalog import (
        ProfileCompatibilityError,
        ProfilePolicy,
        ProfileQuery,
    )
except ImportError:  # pragma: no cover - package-style import
    from .iso_formula import IsoFormula, fit_from_measurements
    from .profile_catalog import (
        ProfileCompatibilityError,
        ProfilePolicy,
        ProfileQuery,
    )


class ContentionCostModel:
    def __init__(
        self,
        derate_profile_path: str | Path,
        use_max_team_derate: bool | None = None,
        use_shape_derate: bool | None = None,
        *,
        expected_policy: ProfileQuery | None = None,
        use_stage_model: bool | None = None,
        iso_mode: str | None = None,
    ):
        self.profile_path = Path(derate_profile_path)
        prof = json.loads(self.profile_path.read_text(encoding="utf-8"))
        self.profile = prof
        self.schema_version = int(prof.get("schema_version", 1))
        self.m_tail_policy = str(prof.get("kernel", {}).get("m_tail_policy", "static_bucketed"))
        if self.m_tail_policy not in {"static_bucketed", "xbyak_exact_m"}:
            raise ValueError(f"unsupported kernel.m_tail_policy={self.m_tail_policy!r}")
        formula_payload = prof.get("iso_formula")
        self.policy = ProfilePolicy.from_payload(prof) if self.schema_version >= 2 else None
        if expected_policy is not None:
            if self.policy is None:
                raise ProfileCompatibilityError("cannot apply a policy query to a legacy profile")
            mismatch = self.policy.mismatch(expected_policy)
            if mismatch:
                raise ProfileCompatibilityError(f"profile {self.profile_path.name} is incompatible: {mismatch}")

        if use_max_team_derate is None:
            use_max_team_derate = os.environ.get("FUSED_CPP_COST_MODEL_MAX_TEAM_DERATE", "0") == "1"
        if use_shape_derate is None:
            configured = os.environ.get("FUSED_CPP_COST_MODEL_SHAPE_DERATE")
            use_shape_derate = self.schema_version >= 2 if configured is None else configured == "1"
        if use_stage_model is None:
            configured = os.environ.get("FUSED_CPP_COST_MODEL_STAGE_AWARE")
            use_stage_model = self.schema_version >= 2 if configured is None else configured != "0"
        self.use_max_team_derate = bool(use_max_team_derate)
        self.use_shape_derate = bool(use_shape_derate)
        self.use_stage_model = bool(use_stage_model)

        if iso_mode is None:
            iso_mode = os.environ.get("FUSED_CPP_COST_MODEL_ISO_MODE")
        if iso_mode is None:
            # Existing profiles remain table-backed. A serialized formula is
            # the profile's explicit opt-in marker for the compact model.
            iso_mode = "formula" if formula_payload is not None else "table"
        if iso_mode not in {"formula", "table"}:
            raise ValueError(f"iso_mode must be 'formula' or 'table', got {iso_mode!r}")
        self.iso_mode = iso_mode

        self._iso = {
            (int(entry["routes"]), int(entry["threads"])): float(entry["median_ns"]) for entry in prof["isolated"]
        }
        self._iso_routes = {
            threads: sorted(routes for routes, t in self._iso if t == threads)
            for threads in sorted({threads for _, threads in self._iso})
        }
        self.iso_formula: IsoFormula | None = None
        if self.iso_mode == "formula":
            self.iso_formula = (
                IsoFormula.from_dict(formula_payload)
                if formula_payload is not None
                else fit_from_measurements((route, team, value) for (route, team), value in self._iso.items())
            )

        d2: dict[int, dict[int, list[float]]] = {}
        d3: dict[int, dict[int, dict[int, list[float]]]] = {}
        ds: dict[tuple[int, ...], dict[int, list[float]]] = {}
        group_curves: dict[tuple[int, ...], dict[int, float]] = {}
        full_call_curves: dict[tuple[int, ...], dict[int, float]] = {}
        p10_curves: dict[tuple[int, ...], dict[int, float]] = {}
        p90_curves: dict[tuple[int, ...], dict[int, float]] = {}
        full_call_p10_curves: dict[tuple[int, ...], dict[int, float]] = {}
        full_call_p90_curves: dict[tuple[int, ...], dict[int, float]] = {}
        for entry in prof["entries"]:
            n = int(entry["distinct_experts"])
            routes = int(entry["routes"])
            derate = float(entry["derate"])
            d2.setdefault(n, {}).setdefault(routes, []).append(derate)
            shape = entry.get("shape")
            if shape:
                signature = self._shape_signature(int(value) for value in shape)
                max_team = signature[0]
                d3.setdefault(n, {}).setdefault(routes, {}).setdefault(max_team, []).append(derate)
                ds.setdefault(signature, {}).setdefault(routes, []).append(derate)
                group_curves.setdefault(signature, {})[routes] = float(entry["makespan_ns"])
                full_call_curves.setdefault(signature, {})[routes] = float(
                    entry.get("full_call_median_ns", entry["makespan_ns"])
                )
                p10_curves.setdefault(signature, {})[routes] = float(entry.get("p10_ns", entry["makespan_ns"]))
                p90_curves.setdefault(signature, {})[routes] = float(entry.get("p90_ns", entry["makespan_ns"]))
                full_call_p10_curves.setdefault(signature, {})[routes] = float(
                    entry.get(
                        "full_call_p10_ns",
                        entry.get("full_call_median_ns", entry["makespan_ns"]),
                    )
                )
                full_call_p90_curves.setdefault(signature, {})[routes] = float(
                    entry.get(
                        "full_call_p90_ns",
                        entry.get("full_call_median_ns", entry["makespan_ns"]),
                    )
                )
        self._derate2d = {
            n: {routes: statistics.median(values) for routes, values in curve.items()} for n, curve in d2.items()
        }
        self._derate3d = {
            n: {
                routes: {max_team: statistics.median(values) for max_team, values in by_team.items()}
                for routes, by_team in curve.items()
            }
            for n, curve in d3.items()
        }
        self._derate_shape = {
            signature: {routes: statistics.median(values) for routes, values in curve.items()}
            for signature, curve in ds.items()
        }
        self._group_curves = group_curves
        self._full_call_curves = full_call_curves
        self._p10_curves = p10_curves
        self._p90_curves = p90_curves
        self._full_call_p10_curves = full_call_p10_curves
        self._full_call_p90_curves = full_call_p90_curves
        self._dn = sorted(self._derate2d)
        self._dr = sorted({routes for curve in self._derate2d.values() for routes in curve})
        self._shape_keys = tuple(sorted(self._derate_shape))

        self._O: dict[int, float] = {}
        for threads in self._iso_routes:
            if self.iso_formula is not None:
                self._O[threads] = self.iso_formula.O(threads)
            else:
                points = sorted(
                    (routes, value) for (routes, team), value in self._iso.items() if team == threads and routes <= 64
                )
                self._O[threads] = self._linear_intercept(points)

        measurement = prof.get("measurement", {})
        self.call_setup_ns = float(measurement.get("call_setup_ns", 0.0))
        self.profile_runs = max(int(measurement.get("runs", 1)), 1)
        working_set = prof.get("working_set", {})
        self.w13_chunk_bytes = int(working_set.get("w13_chunk_bytes_per_expert", 0))
        self.w2_bytes = int(working_set.get("w2_packed_bytes_per_expert", 0))
        self.max_stage_bytes = int(working_set.get("max_weight_stage_bytes_per_expert", 0))
        self.w13_split_chunks = int(self.policy.w13_split_chunks) if self.policy is not None else 1
        expert_shape = prof.get("expert_shape", {})
        self.measurement_experts = int(expert_shape.get("measurement_experts", 0))
        self.local_experts = int(self.policy.local_experts) if self.policy is not None else 0
        self.has_full_workload_anchors = (
            self.schema_version >= 2 and self.local_experts > 0 and self.measurement_experts == self.local_experts
        )

    @staticmethod
    def _linear_intercept(points: list[tuple[int, float]]) -> float:
        if len(points) < 2:
            return 0.0
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        count = len(points)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(value * value for value in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denominator = count * sxx - sx * sx
        slope = (count * sxy - sx * sy) / denominator if denominator else 0.0
        return max((sy - slope * sx) / count, 0.0)

    @staticmethod
    def _shape_signature(threads) -> tuple[int, ...]:
        return tuple(sorted((int(value) for value in threads if int(value) > 0), reverse=True))

    @property
    def supported_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self._shape_keys

    def supports_shape(self, shape) -> bool:
        return self._shape_signature(shape) in self._derate_shape

    def m12_tail_capacity(self, remainder: int) -> int:
        if remainder <= 0:
            return 0
        if self.m_tail_policy == "xbyak_exact_m":
            return remainder
        if remainder <= 2:
            return remainder
        if remainder <= 4:
            return 4
        if remainder <= 8:
            return 8
        return 12

    def m12_effective_rows(self, routes: int) -> int:
        blocks, remainder = divmod(max(int(routes), 0), 12)
        return blocks * 12 + self.m12_tail_capacity(remainder)

    @staticmethod
    def _interp_linear(curve: dict[int, float], x: int) -> float:
        xs = sorted(curve)
        if x <= xs[0]:
            return curve[xs[0]] * x / xs[0]
        if x >= xs[-1]:
            return curve[xs[-1]] * x / xs[-1]
        index = bisect_left(xs, x)
        x0, x1 = xs[index - 1], xs[index]
        y0, y1 = curve[x0], curve[x1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    @staticmethod
    def _interp_log_route(curve: dict[int, float], routes: int | None) -> float:
        points = sorted(curve)
        if routes is None:
            return curve[points[-1]]
        routes = max(int(routes), 1)
        if routes <= points[0]:
            return curve[points[0]]
        if routes >= points[-1]:
            return curve[points[-1]]
        index = bisect_left(points, routes)
        r0, r1 = points[index - 1], points[index]
        x, x0, x1 = math.log2(routes), math.log2(r0), math.log2(r1)
        y0, y1 = curve[r0], curve[r1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _raw_iso_interp(self, routes: int, threads: int) -> float:
        if (routes, threads) in self._iso:
            return self._iso[(routes, threads)]
        route_points = self._iso_routes.get(threads)
        if not route_points:
            raise KeyError(f"no isolated data for threads={threads}")
        curve = {route: self._iso[(route, threads)] for route in route_points}
        return self._interp_linear(curve, routes)

    def _bulk_iso(self, routes: int, threads: int) -> float:
        bulk_curve = {
            route: value
            for (route, team), value in self._iso.items()
            if team == threads and route >= 12 and route % 12 == 0
        }
        if not bulk_curve:
            return self._raw_iso_interp(routes, threads)
        return self._interp_linear(bulk_curve, routes)

    def T_iso(self, routes: int, threads: int) -> float:
        routes = int(routes)
        threads = int(threads)
        if routes <= 0:
            return 0.0
        if threads not in self._iso_routes:
            raise KeyError(f"no isolated calibration for threads={threads}")
        if self.iso_formula is not None:
            return self._formula_iso(routes, threads)
        if (routes, threads) in self._iso:
            return self._iso[(routes, threads)]
        if self.schema_version < 2:
            return self._raw_iso_interp(routes, threads)

        blocks, remainder = divmod(routes, 12)
        tail = self.m12_tail_capacity(remainder)
        overhead = self._O.get(threads, 0.0)
        if blocks == 0:
            return self._raw_iso_interp(tail, threads)
        base = self._bulk_iso(blocks * 12, threads)
        if tail == 0:
            return base
        tail_time = self._raw_iso_interp(tail, threads)
        return overhead + max(base - overhead, 0.0) + max(tail_time - overhead, 0.0)

    def _formula_iso(self, routes: int, threads: int) -> float:
        assert self.iso_formula is not None
        if self.schema_version < 2:
            return self.iso_formula.T_iso(routes, threads)

        blocks, remainder = divmod(routes, 12)
        tail = self.m12_tail_capacity(remainder)
        overhead = self.iso_formula.O(threads)
        if blocks == 0:
            return self._raw_iso_interp(tail, threads)
        # Exact-M JIT tails (or the legacy M1/M2/M4/M8 buckets) and the first
        # two M12 panels have materially different startup/thread efficiency
        # from steady-state M12 bulk. Keep this bounded measured residual and
        # use the formula for the scalable bulk region.
        if blocks <= 2:
            bulk = self._raw_iso_interp(blocks * 12, threads)
        else:
            bulk = self.iso_formula.T_iso(blocks * 12, threads)
        if tail == 0:
            return bulk
        tail_time = self._raw_iso_interp(tail, threads)
        return overhead + max(bulk - overhead, 0.0) + max(tail_time - overhead, 0.0)

    def profiled_group_time(self, routes: int, shape) -> float:
        signature = self._shape_signature(shape)
        curve = self._group_curves.get(signature)
        if curve is None:
            raise ProfileCompatibilityError(f"shape {signature} was not measured in {self.profile_path.name}")
        effective = self.m12_effective_rows(routes)
        return self._interp_linear(curve, effective)

    def profiled_full_call_time(self, routes: int, shape) -> float:
        signature = self._shape_signature(shape)
        curve = self._full_call_curves.get(signature)
        if curve is None:
            raise ProfileCompatibilityError(f"shape {signature} was not measured in {self.profile_path.name}")
        effective = self.m12_effective_rows(routes)
        return self._interp_linear(curve, effective)

    def profiled_group_interval(self, routes: int, shape) -> tuple[float, float]:
        signature = self._shape_signature(shape)
        p10_curve = self._p10_curves.get(signature)
        p90_curve = self._p90_curves.get(signature)
        if p10_curve is None or p90_curve is None:
            raise ProfileCompatibilityError(f"shape {signature} was not measured in {self.profile_path.name}")
        effective = self.m12_effective_rows(routes)
        return (
            self._interp_linear(p10_curve, effective),
            self._interp_linear(p90_curve, effective),
        )

    def profiled_full_call_interval(self, routes: int, shape) -> tuple[float, float]:
        signature = self._shape_signature(shape)
        p10_curve = self._full_call_p10_curves.get(signature)
        p90_curve = self._full_call_p90_curves.get(signature)
        if p10_curve is None or p90_curve is None:
            raise ProfileCompatibilityError(f"shape {signature} was not measured in {self.profile_path.name}")
        effective = self.m12_effective_rows(routes)
        return (
            self._interp_linear(p10_curve, effective),
            self._interp_linear(p90_curve, effective),
        )

    def relative_uncertainty(self, routes: int, shape) -> float:
        median = self.profiled_group_time(routes, shape)
        p10, p90 = self.profiled_group_interval(routes, shape)
        if median <= 0:
            return 0.0
        return max(median - p10, p90 - median, 0.0) / median

    def relative_full_call_uncertainty(self, routes: int, shape) -> float:
        median = self.profiled_full_call_time(routes, shape)
        p10, p90 = self.profiled_full_call_interval(routes, shape)
        if median <= 0:
            return 0.0
        return max(median - p10, p90 - median, 0.0) / median

    def _nearest_route(self, routes: int | None) -> int:
        if routes is None:
            return self._dr[-1]
        effective = self.m12_effective_rows(routes)
        return min(self._dr, key=lambda point: abs(point - effective))

    def _shape_derate(self, routes: int | None, shape) -> float | None:
        if not (self.use_shape_derate and shape):
            return None
        signature = self._shape_signature(shape)
        if len(signature) <= 1:
            return 1.0
        curve = self._derate_shape.get(signature)
        if curve is None:
            return None
        effective = None if routes is None else self.m12_effective_rows(routes)
        return self._interp_log_route(curve, effective)

    def _derate_at_n_route(self, n: int, route_point: int, max_threads: int | None) -> float:
        if (
            self.use_max_team_derate
            and max_threads is not None
            and n in self._derate3d
            and route_point in self._derate3d[n]
        ):
            by_team = self._derate3d[n][route_point]
            team = min(by_team, key=lambda value: abs(value - max_threads))
            return by_team[team]
        return self._derate2d[n][route_point]

    def _derate_float(
        self,
        equivalent_experts: float,
        routes: int | None,
        max_threads: int | None,
    ) -> float:
        if equivalent_experts <= 1.0:
            return 1.0
        route_point = self._nearest_route(routes)
        if equivalent_experts <= self._dn[0]:
            return self._derate_at_n_route(self._dn[0], route_point, max_threads)
        if equivalent_experts >= self._dn[-1]:
            return self._derate_at_n_route(self._dn[-1], route_point, max_threads)
        upper_index = bisect_left(self._dn, equivalent_experts)
        n0, n1 = self._dn[upper_index - 1], self._dn[upper_index]
        d0 = self._derate_at_n_route(n0, route_point, max_threads)
        d1 = self._derate_at_n_route(n1, route_point, max_threads)
        fraction = (equivalent_experts - n0) / (n1 - n0)
        return d0 + (d1 - d0) * fraction

    def derate(
        self,
        n: int,
        routes: int | None = None,
        max_threads: int | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> float:
        if n <= 1:
            return 1.0
        shape_derate = self._shape_derate(routes, shape)
        if shape_derate is not None:
            return shape_derate
        return self._derate_float(float(n), routes, max_threads)

    def _fo(self, routes: int, threads: int) -> float:
        isolated = self.T_iso(routes, threads)
        if isolated <= 0:
            return 0.0
        return min(self._O.get(threads, 0.0) / isolated, 0.9)

    def scalar_makespan(self, tasks) -> float:
        derate = self.derate(
            len(tasks),
            max(routes for routes, _ in tasks),
            max(threads for _, threads in tasks),
            self._shape_signature(threads for _, threads in tasks),
        )
        return self.call_setup_ns + max(
            self.T_iso(routes, threads) * (self._fo(routes, threads) + (1 - self._fo(routes, threads)) * derate)
            for routes, threads in tasks
        )

    @staticmethod
    def _dag_state(tasks):
        count = len(tasks)
        dependency_count = [len(deps) for _, _, deps in tasks]
        successors = [[] for _ in range(count)]
        for task, (_, _, dependencies) in enumerate(tasks):
            for dependency in dependencies:
                successors[dependency].append(task)
        started = [dependency_count[index] == 0 for index in range(count)]
        return dependency_count, successors, started

    def _sim_flat(self, tasks) -> float:
        count = len(tasks)
        routes = [value for value, _, _ in tasks]
        threads = [value for _, value, _ in tasks]
        remaining = [self.T_iso(route_count, team) for route_count, team, _ in tasks]
        dependency_count, successors, started = self._dag_state(tasks)
        finished = [False] * count
        wall = self.call_setup_ns
        guard = 0
        while not all(finished):
            guard += 1
            if guard > 2 * count + 2:
                raise RuntimeError("flat DAG simulation did not converge")
            active = [index for index in range(count) if started[index] and not finished[index]]
            if not active:
                raise ValueError("DAG deadlock (cycle or unreachable task)")
            slowdown = self.derate(
                len(active),
                max(routes[index] for index in active),
                max(threads[index] for index in active),
                self._shape_signature(threads[index] for index in active),
            )
            effective = {
                index: self._fo(routes[index], threads[index])
                + (1 - self._fo(routes[index], threads[index])) * slowdown
                for index in active
            }
            elapsed = min(remaining[index] * effective[index] for index in active)
            wall += elapsed
            for index in active:
                remaining[index] -= elapsed / effective[index]
            for index in active:
                if remaining[index] <= 1e-6:
                    finished[index] = True
                    for successor in successors[index]:
                        dependency_count[successor] -= 1
                        if dependency_count[successor] == 0:
                            started[successor] = True
        return wall

    def _task_phases(self, routes: int, threads: int) -> list[tuple[float, int]]:
        isolated = self.T_iso(routes, threads)
        overhead = min(self._O.get(threads, 0.0), isolated * 0.9)
        compute = max(isolated - overhead, 0.0)
        phases: list[tuple[float, int]] = []
        if overhead > 0:
            phases.append((overhead, 0))
        chunks = max(self.w13_split_chunks, 1)
        w13_phase = compute * (2.0 / 3.0) / chunks
        for _ in range(chunks):
            phases.append((w13_phase, self.w13_chunk_bytes))
        phases.append((compute / 3.0, self.w2_bytes))
        return [(duration, workset) for duration, workset in phases if duration > 0]

    def _working_set_derate(
        self,
        active: list[int],
        routes: list[int],
        threads: list[int],
        worksets: list[int],
    ) -> float:
        compute_active = [index for index in active if worksets[index] > 0]
        if len(compute_active) <= 1 or self.max_stage_bytes <= 0:
            return 1.0
        total_bytes = sum(worksets[index] for index in compute_active)
        equivalent = total_bytes / self.max_stage_bytes
        shape = self._shape_signature(threads[index] for index in compute_active)
        homogeneous_max_stage = all(worksets[index] == self.max_stage_bytes for index in compute_active)
        if homogeneous_max_stage:
            exact = self._shape_derate(max(routes[index] for index in compute_active), shape)
            if exact is not None:
                return exact
        return self._derate_float(
            equivalent,
            max(routes[index] for index in compute_active),
            max(threads[index] for index in compute_active),
        )

    def _sim_staged(self, tasks) -> float:
        count = len(tasks)
        routes = [int(value) for value, _, _ in tasks]
        threads = [int(value) for _, value, _ in tasks]
        phases = [self._task_phases(route_count, team) for route_count, team, _ in tasks]
        phase_index = [0] * count
        remaining = [task_phases[0][0] for task_phases in phases]
        dependency_count, successors, started = self._dag_state(tasks)
        finished = [False] * count
        wall = self.call_setup_ns
        guard = 0
        max_events = sum(len(task_phases) for task_phases in phases) + count + 2
        while not all(finished):
            guard += 1
            if guard > 2 * max_events:
                raise RuntimeError("stage-aware DAG simulation did not converge")
            active = [index for index in range(count) if started[index] and not finished[index]]
            if not active:
                raise ValueError("DAG deadlock (cycle or unreachable task)")
            worksets = [0] * count
            for index in active:
                worksets[index] = phases[index][phase_index[index]][1]
            slowdown = self._working_set_derate(active, routes, threads, worksets)
            effective = {index: (1.0 if worksets[index] == 0 else slowdown) for index in active}
            elapsed = min(remaining[index] * effective[index] for index in active)
            wall += elapsed
            for index in active:
                remaining[index] -= elapsed / effective[index]
            completed_phases = [index for index in active if remaining[index] <= 1e-6]
            for index in completed_phases:
                phase_index[index] += 1
                if phase_index[index] < len(phases[index]):
                    remaining[index] = phases[index][phase_index[index]][0]
                    continue
                finished[index] = True
                for successor in successors[index]:
                    dependency_count[successor] -= 1
                    if dependency_count[successor] == 0:
                        started[successor] = True
        return wall

    def flat_dag_makespan(self, tasks) -> float:
        return self._sim_flat(tasks)

    def phase_makespan(self, tasks) -> float:
        dag = [(routes, threads, []) for routes, threads in tasks]
        return self._sim_staged(dag) if self.use_stage_model else self._sim_flat(dag)

    def dag_makespan(self, tasks) -> float:
        if self.use_stage_model and self.max_stage_bytes > 0:
            return self._sim_staged(tasks)
        return self._sim_flat(tasks)


if __name__ == "__main__":
    import sys

    model = ContentionCostModel(sys.argv[1])
    print("policy:", model.policy)
    print("O(t) ms:", {t: round(value / 1e6, 3) for t, value in model._O.items()})
    print("supported shapes:", model.supported_shapes)
