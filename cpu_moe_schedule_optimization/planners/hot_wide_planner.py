"""Hot-wide template planner: a fast planner built from features of model-searched plans.

The near-optimal plans found by the model-objective LNS (``model_lns.py``) under the
probe-calibrated event model share three features, which this planner hard-codes:

1. shape: a few wide lanes (8/16/32 threads) that hold the hottest experts, and all
   remaining cores in narrow bulk lanes (4 threads);
2. load balance: lanes finish together once each width's isolated load is scaled by the
   event/isolated ratio the searched plans show for that width (``lane_scale``);
3. order: inside a lane the largest expert starts first and the rest follow in ascending
   route order.

For every template (wide multiset plus bulk lanes) that packs into the LLC domains, the
planner assigns experts by LPT on heterogeneous lanes (hot experts land on wide lanes when
that lowers the scaled load) and keeps the template with the smallest scaled maximum lane
load. No event simulation runs on the request path.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import combinations_with_replacement
from typing import Mapping, Sequence

try:
    from .model_lns import Lane, pack_lanes, planner_tasks
except ImportError:  # pragma: no cover - direct script import
    from model_lns import Lane, pack_lanes, planner_tasks

# Event makespan / isolated (windowed) load of lanes in the v10-searched, 2T-free plans of the
# 18 E1 workloads, scored with the v11 model (tmp/fast_planner_20260920, median per width).
DEFAULT_LANE_SCALE = {4: 1.119, 8: 1.065, 16: 1.045, 32: 1.032}


_UNSET = object()


def _begins(tasks) -> tuple[int, ...]:
    """Core offset of each lane, in the order the tasks introduce them."""
    seen: dict[tuple[int, int], None] = {}
    for _, _, core, threads, _ in tasks:
        seen.setdefault((int(core), int(threads)), None)
    return tuple(core for core, _ in seen)


def _lanes_from_tasks(tasks) -> tuple[Lane, ...]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for expert, routes, core, threads, _ in tasks:
        groups.setdefault((int(core), int(threads)), []).append((int(expert), int(routes)))
    return tuple(Lane(threads, tuple(members)) for (_, threads), members in groups.items())


@dataclass(frozen=True)
class HotWidePlan:
    lanes: tuple[Lane, ...]
    begins: tuple[int, ...]
    shape: tuple[int, ...]
    score_ns: float
    templates: int

    def tasks(self):
        """IntervalPlanner task tuples (expert, routes, core_begin, threads, dependencies)."""
        return planner_tasks(self.lanes, self.begins)


class HotWidePlanner:
    def __init__(
        self,
        model,
        *,
        num_cores: int = 80,
        domain_cores: Sequence[int] = (40, 40),
        bulk_width: int = 4,
        wide_widths: Sequence[int] = (8, 16, 32),
        max_wide_lanes: int = 3,
        max_wide_cores: int = 48,
        lane_scale: Mapping[int, float] | None = None,
        window_policy=None,
        use_native: bool = True,
    ):
        if sum(domain_cores) != num_cores:
            raise ValueError("domain cores must cover num_cores")
        self.model = model
        self.num_cores = int(num_cores)
        self.domain_cores = tuple(int(c) for c in domain_cores)
        self.bulk_width = int(bulk_width)
        self.lane_scale = dict(DEFAULT_LANE_SCALE if lane_scale is None else lane_scale)
        self.window_policy = window_policy
        self.use_native = bool(use_native)
        self.wide_widths = tuple(sorted(wide_widths, reverse=True))
        self.max_wide_lanes = int(max_wide_lanes)
        self.max_wide_cores = int(max_wide_cores)
        self._native_planner = _UNSET
        self._cost: dict[tuple[int, int], float] = {}
        self._raw: dict[tuple[int, int], float] = {}
        self._last_shape: tuple[int, ...] | None = None
        shapes = []
        for count in range(0, max_wide_lanes + 1):
            for combo in combinations_with_replacement(sorted(wide_widths, reverse=True), count):
                if sum(combo) > max_wide_cores:
                    continue
                bulk = (self.num_cores - sum(combo)) // self.bulk_width
                shape = tuple(combo) + (self.bulk_width,) * bulk
                if pack_lanes([Lane(w, ()) for w in shape], self.domain_cores) is not None:
                    shapes.append(shape)
        if not shapes:
            raise ValueError("no template packs into the LLC domains")
        self.shapes = tuple(dict.fromkeys(shapes))

    def raw_cost(self, routes: int, width: int) -> float:
        """Isolated time scaled by the window table, before the per-width lane scale.

        The native planner applies the lane scale itself, so it takes these values; caching them
        here keeps both paths paying the same price for the model lookups.
        """
        key = (routes, width)
        value = self._raw.get(key)
        if value is None:
            scale = 1.0 if self.window_policy is None else float(self.window_policy.time_scale(routes, width))
            value = float(self.model.T_iso(routes, width)) * scale
            self._raw[key] = value
        return value

    def cost(self, routes: int, width: int) -> float:
        key = (routes, width)
        value = self._cost.get(key)
        if value is None:
            scale = 1.0 if self.window_policy is None else float(self.window_policy.time_scale(routes, width))
            value = float(self.model.T_iso(routes, width)) * scale * self.lane_scale.get(width, 1.0)
            self._cost[key] = value
        return value

    def assign(self, experts: Sequence[tuple[int, int]], shape: Sequence[int],
               costs: Mapping[int, Sequence[float]] | None = None,
               limit: float = float("inf")) -> tuple[tuple[Lane, ...], float] | tuple[None, float]:
        """LPT over heterogeneous lanes, ties by lane index; lanes come back in hot-first order.

        ``experts`` must be sorted by descending routes when ``costs`` is given (the cost rows are
        indexed by position); :meth:`plan` does that once for all templates. With ``limit`` the
        assignment stops as soon as a lane passes that load (LPT loads only grow), and returns
        ``(None, load)``: the template cannot beat the incumbent.
        """
        if costs is None:
            experts = sorted(experts, key=lambda item: (-item[1], item[0]))
            costs = {width: [self.cost(routes, width) for _, routes in experts] for width in set(shape)}
        heaps: dict[int, list[list[float]]] = {}
        members: list[list[tuple[int, int]]] = [[] for _ in shape]
        loads = [0.0] * len(shape)
        for lane, width in enumerate(shape):
            heaps.setdefault(width, []).append([0.0, lane])
        for heap in heaps.values():
            heapify(heap)
        for index, expert in enumerate(experts):
            best = None
            for width, heap in heaps.items():
                candidate = (heap[0][0] + costs[width][index], heap[0][1], width)
                if best is None or candidate < best:
                    best = candidate
            load, lane, width = best
            heappop(heaps[width])
            heappush(heaps[width], [load, lane])
            members[lane].append(expert)
            loads[lane] = load
            if load >= limit:
                return None, load
        lanes = tuple(Lane(width, tuple(group[:1] + group[1:][::-1])) for width, group in zip(shape, members))
        return lanes, max(loads, default=0.0)

    def _native(self):
        """The C++ port of this planner, or None when the extension does not carry it."""
        if self._native_planner is not _UNSET:
            return self._native_planner
        self._native_planner = None
        if self.use_native:
            try:
                from fused_cpp import _moe_C
            except Exception:  # pragma: no cover - no extension in this environment
                _moe_C = None
            factory = getattr(_moe_C, "NativeHotWidePlanner", None) if _moe_C is not None else None
            if factory is not None:
                self._native_planner = factory(
                    num_cores=self.num_cores, domain_cores=list(self.domain_cores), bulk_width=self.bulk_width,
                    wide_widths=list(self.wide_widths), max_wide_lanes=self.max_wide_lanes,
                    max_wide_cores=self.max_wide_cores,
                    lane_scale=[(int(w), float(v)) for w, v in sorted(self.lane_scale.items())],
                )
        return self._native_planner

    def plan_native(self, experts: Sequence[tuple[int, int]]) -> HotWidePlan | None:
        """The same plan from the native planner, or None when it is unavailable.

        The costs cross the boundary as plain doubles - one row per width, before the lane
        scale - so the native side holds no model state and the two paths score identically.
        """
        native = self._native()
        if native is None:
            return None
        active = [(int(e), int(r)) for e, r in experts if int(r) > 0]
        if not active:
            raise ValueError("at least one active expert is required")
        widths = sorted({w for shape in self.shapes for w in shape})
        rows = [[self.raw_cost(routes, width) for _, routes in active] for width in widths]
        result = native.plan([e for e, _ in active], [r for _, r in active], widths, rows)
        return HotWidePlan(lanes=_lanes_from_tasks(result["tasks"]), begins=_begins(result["tasks"]),
                           shape=tuple(result["shape"]), score_ns=float(result["score_ns"]),
                           templates=int(result["templates"]))

    def plan(self, experts: Sequence[tuple[int, int]]) -> HotWidePlan:
        experts = sorted(((int(e), int(r)) for e, r in experts if int(r) > 0), key=lambda item: (-item[1], item[0]))
        if not experts:
            raise ValueError("at least one active expert is required")
        costs = {width: [self.cost(routes, width) for _, routes in experts]
                 for width in {w for shape in self.shapes for w in shape}}
        best = None
        # The template that won the previous call first: it usually survives and then prunes the rest.
        order = self.shapes if self._last_shape is None else (
            (self._last_shape,) + tuple(s for s in self.shapes if s != self._last_shape))
        for shape in order:
            lanes, score = self.assign(experts, shape, costs, float("inf") if best is None else best[1])
            if lanes is not None and (best is None or score < best[1]):
                best = (lanes, score, shape)
        lanes, score, shape = best
        self._last_shape = shape
        used = tuple(lane for lane in lanes if lane.experts)
        begins = pack_lanes(used, self.domain_cores)
        assert begins is not None  # a subset of a packable template packs
        return HotWidePlan(lanes=used, begins=tuple(begins), shape=tuple(shape), score_ns=score,
                           templates=len(self.shapes))


__all__ = ["DEFAULT_LANE_SCALE", "HotWidePlan", "HotWidePlanner"]
