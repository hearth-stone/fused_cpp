"""Model-objective large-neighborhood search over strict lane plans.

Offline reference search for the CPU MoE planner (TODO "Track P"): it minimizes
the cost model's placed event makespan directly, unlike the hardware-diagnostic
template LNS driver. A plan is a set of serial lanes; each lane has a width, a
contiguous core interval inside one LLC domain, and an ordered list of whole
experts. The search is anytime: it keeps the best plan seen and returns it at
the time limit or when every restart stalls.

Moves, all guided by the simulated lane finish times:

* ``relocate`` / ``swap``: an expert of a late lane moves to (or swaps with a
  smaller expert of) an early lane;
* ``split`` / ``merge``: a lane of width 2t becomes two lanes of width t with
  isolated-LPT redistribution, or two lanes of width t become one of 2t;
* ``repack``: the compound move. The experts of the latest lane and a few other
  lanes are pooled and re-assigned by isolated LPT onto a new width template
  covering the same cores (homogeneous, or one wide lane plus narrow lanes);
* ``rebalance``: the experts of a late and an early lane are re-split between
  the two lanes by branch and bound on the larger isolated lane load (reaches
  multi-expert exchanges that relocate/swap cannot);
* ``recreate``: ruin and recreate of a late lane plus a few others with a
  randomized greedy fill (lane widths kept);
* ``reorder``: a late lane's experts in descending, ascending, hot-first then
  ascending, hot-last, or random order; ``reorder_all`` puts every lane in
  hot-first then ascending order;
* ``slice`` / ``unslice`` (only when ``allow_route_slicing``): one expert is cut
  into equal route ranges on separate lanes, or its ranges are put back
  together. The runtime lowers an expert that appears more than once as equal
  fixed ranges, so the slices must be equal, and the model scores each of them
  as a task of its own route count.

Each iteration evaluates a batch of sampled neighbors (optionally in worker
processes) and moves to the best one when it improves; with probability 0.3 it
also accepts a worse best neighbor within a threshold of the best plan that
decays linearly to zero over the time slice. After ``patience`` non-improving
batches the incumbent restarts from the slice's best plan with a random
perturbation. The time limit is split over the best few distinct starts, each
descended separately (multi-start), and the best plan over all slices is kept. The model must provide ``dag_makespan_placed``; when it also
provides ``simulate`` its task finish times guide the moves.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

Expert = tuple[int, int]  # (expert id, routes)


@dataclass(frozen=True)
class Lane:
    width: int
    experts: tuple[Expert, ...]


@dataclass
class SearchResult:
    lanes: tuple[Lane, ...]
    makespan_ns: float
    start_makespan_ns: float
    evaluations: int
    iterations: int
    elapsed_s: float
    trace: list[tuple[float, float]] = field(default_factory=list)  # (elapsed s, best makespan ns)
    accepted_moves: dict[str, int] = field(default_factory=dict)


def pack_lanes(lanes: Sequence[Lane], domain_cores: Sequence[int]) -> list[int] | None:
    """Core offset of every lane with each lane inside one LLC domain, or None.

    First-fit decreasing by width into the domain with the most free cores; the
    event model reads placement only through domain membership, so any valid
    packing is equivalent for it.
    """
    free = list(domain_cores)
    starts = [sum(domain_cores[:index]) for index in range(len(domain_cores))]
    cursor = list(starts)
    begins = [0] * len(lanes)
    for index in sorted(range(len(lanes)), key=lambda k: (-lanes[k].width, k)):
        width = lanes[index].width
        domain = max(range(len(free)), key=lambda d: (free[d], -d))
        if free[domain] < width:
            return None
        begins[index] = cursor[domain]
        cursor[domain] += width
        free[domain] -= width
    return begins


def planner_tasks(lanes: Sequence[Lane], begins: Sequence[int]):
    """IntervalPlanner task tuples: (expert, routes, core_begin, threads, dependencies)."""
    tasks = []
    for lane, begin in zip(lanes, begins):
        previous = None
        for expert, routes in lane.experts:
            tasks.append((expert, routes, begin, lane.width, [previous] if previous is not None else []))
            previous = len(tasks) - 1
    return tasks


def placed_from_lanes(lanes: Sequence[Lane], begins: Sequence[int], cpu_ids: Sequence[int]):
    return [
        (routes, threads, tuple(cpu_ids[core : core + threads]), tuple(deps))
        for _, routes, core, threads, deps in planner_tasks(lanes, begins)
    ]


def canonical(lanes: Sequence[Lane]) -> tuple:
    return tuple(sorted((lane.width, lane.experts) for lane in lanes if lane.experts or lane.width))


_WORKER_MODEL = None


def _worker_init(factory: Callable[[], object]) -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = factory()


def _worker_score(placed) -> tuple[float, list[float]]:
    return _score(_WORKER_MODEL, placed)


def _score(model, placed) -> tuple[float, list[float]]:
    simulate = getattr(model, "simulate", None)
    if callable(simulate):
        result = simulate(placed)
        return float(result["makespan_ns"]), list(result["task_finish_ns"])
    return float(model.dag_makespan_placed(placed)), []


class ModelLnsSearch:
    def __init__(
        self,
        model,
        cpu_ids: Sequence[int],
        *,
        widths: Sequence[int] | None = None,
        domain_cores: Sequence[int] = (40, 40),
        isolated_cost: Callable[[int, int], float] | None = None,
        batch: int = 32,
        patience: int = 8,
        threshold: float = 0.004,
        allow_route_slicing: bool = False,
        min_sliceable_routes: int = 64,
        seed: int = 20260920,
        pool=None,
    ):
        if sum(domain_cores) != len(cpu_ids):
            raise ValueError("domain cores must cover cpu_ids")
        self.model = model
        self.cpu_ids = tuple(int(cpu) for cpu in cpu_ids)
        if widths is None:  # the widths the model is trusted on, else its calibrated set
            widths = getattr(model, "reliable_widths", None) or (4, 8, 16, 32)
        self.widths = tuple(sorted(int(width) for width in widths))
        self.domain_cores = tuple(int(value) for value in domain_cores)
        self.isolated_cost = isolated_cost or model.T_iso
        self.batch = int(batch)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.allow_route_slicing = bool(allow_route_slicing)
        self.min_sliceable_routes = int(min_sliceable_routes)
        self.rng = random.Random(seed)
        self.pool = pool
        self.cache: dict[tuple, tuple[float, list[float]]] = {}
        self.evaluations = 0
        self._cost_cache: dict[tuple[int, int], float] = {}

    # ---- evaluation

    def cost(self, routes: int, width: int) -> float:
        key = (routes, width)
        value = self._cost_cache.get(key)
        if value is None:
            value = float(self.isolated_cost(routes, width))
            self._cost_cache[key] = value
        return value

    def evaluate_many(self, plans: Sequence[tuple[Lane, ...]]) -> list[tuple[float, list[float]] | None]:
        keys = [canonical(plan) for plan in plans]
        missing: dict[tuple, tuple[tuple[Lane, ...], list]] = {}
        for key, plan in zip(keys, plans):
            if key in self.cache or key in missing:
                continue
            begins = pack_lanes(plan, self.domain_cores)
            if begins is None or sum(lane.width for lane in plan) > len(self.cpu_ids):
                self.cache[key] = None
                continue
            missing[key] = (plan, placed_from_lanes(plan, begins, self.cpu_ids))
        if missing:
            placed = [value[1] for value in missing.values()]
            if self.pool is not None and len(placed) > 1:
                scores = self.pool.map(_worker_score, placed, chunksize=max(1, len(placed) // 16))
            else:
                scores = [_score(self.model, item) for item in placed]
            self.evaluations += len(placed)
            for key, score in zip(missing, scores):
                self.cache[key] = score
        results = []
        for key, plan in zip(keys, plans):
            score = self.cache[key]
            if score is None:
                results.append(None)
                continue
            results.append(score)
        return results

    @staticmethod
    def lane_finish(plan: Sequence[Lane], finish: Sequence[float]) -> list[float]:
        """Finish time of every lane in plan order (tasks are laid out lane by lane)."""
        out, cursor = [], 0
        for lane in plan:
            count = len(lane.experts)
            out.append(finish[cursor + count - 1] if count and finish else 0.0)
            cursor += count
        return out

    # ---- construction

    def lpt(self, experts: Sequence[Expert], widths: Sequence[int]) -> tuple[Lane, ...]:
        """Isolated LPT onto lanes of the given widths; lanes keep descending route order."""
        loads = [0.0] * len(widths)
        members: list[list[Expert]] = [[] for _ in widths]
        for expert in sorted(experts, key=lambda item: (-item[1], item[0])):
            lane = min(range(len(widths)), key=lambda k: (loads[k] + self.cost(expert[1], widths[k]), k))
            members[lane].append(expert)
            loads[lane] += self.cost(expert[1], widths[lane])
        return tuple(Lane(width, tuple(group)) for width, group in zip(widths, members))

    def templates(self, cores: int, heaviest_routes: int) -> list[tuple[int, ...]]:
        """Width templates covering ``cores``: homogeneous, and one wide lane plus narrow ones."""
        out = set()
        for width in self.widths:
            if cores % width == 0:
                out.add((width,) * (cores // width))
        for wide in self.widths:
            for narrow in self.widths:
                if narrow < wide <= cores and (cores - wide) % narrow == 0:
                    out.add((wide,) + (narrow,) * ((cores - wide) // narrow))
                    if 2 * wide <= cores and (cores - 2 * wide) % narrow == 0:
                        out.add((wide, wide) + (narrow,) * ((cores - 2 * wide) // narrow))
        return sorted(out)

    # ---- moves

    @staticmethod
    def _order(experts: Sequence[Expert]) -> tuple[Expert, ...]:
        return tuple(sorted(experts, key=lambda item: (-item[1], item[0])))

    def balance(self, pooled: Sequence[Expert], width_a: int, width_b: int,
                node_limit: int = 20000) -> tuple[tuple[Expert, ...], tuple[Expert, ...]]:
        """Split ``pooled`` over two lanes minimizing the larger isolated lane load.

        Depth-first branch and bound over experts in descending cost, seeded with the greedy
        split; stops after ``node_limit`` nodes (then the best split found is returned).
        """
        items = sorted(pooled, key=lambda e: -self.cost(e[1], min(width_a, width_b)))
        cost_a = [self.cost(e[1], width_a) for e in items]
        cost_b = [self.cost(e[1], width_b) for e in items]
        greedy_a, greedy_b, load_a, load_b = [], [], 0.0, 0.0
        for index in range(len(items)):
            if load_a + cost_a[index] <= load_b + cost_b[index]:
                greedy_a.append(index)
                load_a += cost_a[index]
            else:
                greedy_b.append(index)
                load_b += cost_b[index]
        best = [max(load_a, load_b), greedy_a, greedy_b]
        rest_b = [0.0] * (len(items) + 1)
        for index in range(len(items) - 1, -1, -1):
            rest_b[index] = rest_b[index + 1] + min(cost_a[index], cost_b[index])
        nodes = 0
        chosen: list[int] = []

        def visit(index: int, a: float, b: float) -> None:
            nonlocal nodes
            nodes += 1
            if nodes > node_limit or max(a, b) >= best[0]:
                return
            # Lower bound: the remaining work split perfectly across both lanes.
            if (a + b + rest_b[index]) / 2.0 >= best[0]:
                return
            if index == len(items):
                in_a = set(chosen)
                best[:] = [max(a, b), sorted(in_a), [k for k in range(len(items)) if k not in in_a]]
                return
            first = (True, False) if a + cost_a[index] <= b + cost_b[index] else (False, True)
            for to_a in first:
                if to_a:
                    chosen.append(index)
                    visit(index + 1, a + cost_a[index], b)
                    chosen.pop()
                else:
                    visit(index + 1, a, b + cost_b[index])

        visit(0, 0.0, 0.0)
        return tuple(items[k] for k in best[1]), tuple(items[k] for k in best[2])

    def neighbors(self, plan: tuple[Lane, ...], finish: Sequence[float]) -> list[tuple[str, tuple[Lane, ...]]]:
        rng = self.rng
        lanes = list(plan)
        ends = self.lane_finish(plan, finish) if finish else [
            sum(self.cost(r, lane.width) for _, r in lane.experts) for lane in lanes
        ]
        by_end = sorted(range(len(lanes)), key=lambda k: -ends[k])
        late = [k for k in by_end if lanes[k].experts][: max(1, min(4, len(lanes)))]
        early = [k for k in reversed(by_end)][: max(1, min(8, len(lanes)))]
        out: list[tuple[str, tuple[Lane, ...]]] = []

        def replace(changes: dict[int, Lane | None], extra: Sequence[Lane] = ()) -> tuple[Lane, ...]:
            new = [changes.get(k, lane) for k, lane in enumerate(lanes)]
            return tuple(lane for lane in new if lane is not None) + tuple(extra)

        kinds = ("relocate", "relocate", "swap", "rebalance", "rebalance", "repack", "recreate", "split", "merge",
                 "reorder", "reorder_all")
        if self.allow_route_slicing:
            kinds = kinds + ("slice", "slice", "unslice")
        attempts = 0
        while len(out) < self.batch and attempts < 8 * self.batch:
            attempts += 1
            kind = rng.choice(kinds)
            source = late[0] if rng.random() < 0.6 else rng.choice(late)
            if kind == "relocate" and len(lanes) > 1:
                target = rng.choice([k for k in early if k != source] or [k for k in range(len(lanes)) if k != source])
                if not lanes[source].experts:
                    continue
                expert = rng.choice(lanes[source].experts)
                src = Lane(lanes[source].width, tuple(e for e in lanes[source].experts if e != expert))
                dst = Lane(lanes[target].width, self._order(lanes[target].experts + (expert,)))
                out.append((kind, replace({source: src, target: dst})))
            elif kind == "swap" and len(lanes) > 1:
                target = rng.choice([k for k in early if k != source] or [k for k in range(len(lanes)) if k != source])
                if not lanes[source].experts or not lanes[target].experts:
                    continue
                a = rng.choice(lanes[source].experts)
                smaller = [e for e in lanes[target].experts if e[1] < a[1]]
                if not smaller:
                    continue
                b = rng.choice(smaller)
                src = Lane(lanes[source].width, self._order(tuple(e for e in lanes[source].experts if e != a) + (b,)))
                dst = Lane(lanes[target].width, self._order(tuple(e for e in lanes[target].experts if e != b) + (a,)))
                out.append((kind, replace({source: src, target: dst})))
            elif kind == "repack":
                region = {source}
                others = [k for k in range(len(lanes)) if k != source]
                rng.shuffle(others)
                region.update(others[: rng.choice((1, 2, 3, 5, 7))])
                cores = sum(lanes[k].width for k in region)
                pooled = [e for k in region for e in lanes[k].experts]
                if not pooled:
                    continue
                options = self.templates(cores, max(r for _, r in pooled))
                if not options:
                    continue
                template = rng.choice(options)
                if len(template) > len(pooled) + 2:
                    continue
                new = self.lpt(pooled, template)
                out.append((kind, replace({k: None for k in region}, [lane for lane in new if lane.experts]
                                          + [Lane(lane.width, ()) for lane in new if not lane.experts])))
            elif kind == "split":
                lane = lanes[source]
                half = lane.width // 2
                if half not in self.widths or len(lane.experts) < 2:
                    continue
                out.append((kind, replace({source: None}, self.lpt(lane.experts, (half, half)))))
            elif kind == "merge":
                width = lanes[source].width
                if 2 * width not in self.widths:
                    continue
                peers = [k for k in range(len(lanes)) if k != source and lanes[k].width == width]
                if not peers:
                    continue
                other = min(peers, key=lambda k: ends[k]) if rng.random() < 0.5 else rng.choice(peers)
                merged = Lane(2 * width, self._order(lanes[source].experts + lanes[other].experts))
                out.append((kind, replace({source: merged, other: None})))
            elif kind == "rebalance" and len(lanes) > 1:
                target = rng.choice([k for k in early if k != source] or [k for k in range(len(lanes)) if k != source])
                pooled = lanes[source].experts + lanes[target].experts
                if len(pooled) < 2:
                    continue
                left, right = self.balance(pooled, lanes[source].width, lanes[target].width)
                out.append((kind, replace({source: Lane(lanes[source].width, self._order(left)),
                                           target: Lane(lanes[target].width, self._order(right))})))
            elif kind == "recreate":
                region = {source}
                others = [k for k in range(len(lanes)) if k != source]
                rng.shuffle(others)
                region.update(others[: rng.choice((1, 2, 3, 5))])
                keys = sorted(region)
                pooled = [e for k in keys for e in lanes[k].experts]
                if not pooled:
                    continue
                widths = [lanes[k].width for k in keys]
                loads = [0.0] * len(keys)
                members: list[list[Expert]] = [[] for _ in keys]
                noisy = sorted(pooled, key=lambda e: -self.cost(e[1], 4) * rng.uniform(0.7, 1.3))
                for expert in noisy:
                    slot = min(range(len(keys)),
                               key=lambda j: (loads[j] + self.cost(expert[1], widths[j])) * rng.uniform(0.97, 1.03))
                    members[slot].append(expert)
                    loads[slot] += self.cost(expert[1], widths[slot])
                out.append((kind, replace({k: Lane(w, self._order(m)) for k, w, m in zip(keys, widths, members)})))
            elif kind == "slice":
                lane = lanes[source]
                sliceable = [e for e in lane.experts
                             if e[1] >= self.min_sliceable_routes and e[1] % 2 == 0
                             and sum(1 for other in lanes for x in other.experts if x[0] == e[0]) == 1]
                if not sliceable or len(lanes) < 2:
                    continue
                expert = max(sliceable, key=lambda e: e[1])
                target = rng.choice([k for k in early if k != source] or [k for k in range(len(lanes)) if k != source])
                half = (expert[0], expert[1] // 2)
                src = Lane(lanes[source].width, self._order(tuple(e for e in lane.experts if e != expert) + (half,)))
                dst = Lane(lanes[target].width, self._order(lanes[target].experts + (half,)))
                out.append((kind, replace({source: src, target: dst})))
            elif kind == "unslice":
                counts: dict[int, list[int]] = {}
                for k, lane in enumerate(lanes):
                    for expert, _ in lane.experts:
                        counts.setdefault(expert, []).append(k)
                sliced = [e for e, where in counts.items() if len(where) > 1]
                if not sliced:
                    continue
                expert = rng.choice(sliced)
                pieces = [(k, e) for k in set(counts[expert]) for e in lanes[k].experts if e[0] == expert]
                total = sum(routes for _, (_, routes) in pieces)
                keep = min(pieces, key=lambda item: item[0])[0]
                changes = {}
                for k, piece in pieces:
                    changes[k] = Lane(lanes[k].width, tuple(e for e in changes.get(k, lanes[k]).experts if e != piece))
                changes[keep] = Lane(lanes[keep].width, self._order(changes[keep].experts + ((expert, total),)))
                out.append((kind, replace(changes)))
            elif kind == "reorder_all":
                # Hot-first then ascending in every lane (the order feature of the searched plans).
                changed = {}
                for k, lane in enumerate(lanes):
                    ordered = self._order(lane.experts)
                    experts = ordered[:1] + tuple(reversed(ordered[1:]))
                    if experts != lane.experts:
                        changed[k] = Lane(lane.width, experts)
                if not changed:
                    continue
                out.append((kind, replace(changed)))
            elif kind == "reorder":
                lane = lanes[source]
                if len(lane.experts) < 2:
                    continue
                ordered = self._order(lane.experts)
                variant = rng.choice(("desc", "asc", "hot_first_asc", "hot_last", "random"))
                if variant == "desc":
                    experts = ordered
                elif variant == "asc":
                    experts = tuple(reversed(ordered))
                elif variant == "hot_first_asc":
                    experts = (ordered[0],) + tuple(reversed(ordered[1:]))
                elif variant == "hot_last":
                    experts = ordered[1:] + (ordered[0],)
                else:
                    shuffled = list(lane.experts)
                    rng.shuffle(shuffled)
                    experts = tuple(shuffled)
                if experts == lane.experts:
                    continue
                out.append((kind, replace({source: Lane(lane.width, experts)})))
        return out

    def perturb(self, plan: tuple[Lane, ...]) -> tuple[Lane, ...]:
        lanes = list(plan)
        for _ in range(3):
            k = self.rng.randrange(len(lanes))
            j = self.rng.randrange(len(lanes))
            if k == j or not lanes[k].experts:
                continue
            expert = self.rng.choice(lanes[k].experts)
            lanes[k] = Lane(lanes[k].width, tuple(e for e in lanes[k].experts if e != expert))
            lanes[j] = Lane(lanes[j].width, self._order(lanes[j].experts + (expert,)))
        return tuple(lanes)

    # ---- driver

    def search(self, starts: Sequence[tuple[Lane, ...]], *, time_limit_s: float,
               max_evaluations: int | None = None, max_starts: int = 4) -> SearchResult:
        """Descend from each of the best ``max_starts`` distinct starts in turn (equal time slices).

        A single descent from the best start stays in that start's basin; separate descents from
        structurally different starts (e.g. homogeneous shapes of different widths) do not.
        """
        begin = time.perf_counter()
        scored = [(score, plan) for plan, score in zip(starts, self.evaluate_many(starts)) if score is not None]
        if not scored:
            raise ValueError("no valid start plan")
        unique: dict[tuple, tuple] = {}
        for score, plan in sorted(scored, key=lambda item: item[0][0]):
            unique.setdefault(canonical(plan), (score, plan))
        chosen = list(unique.values())[: max(1, int(max_starts))]
        start_value = chosen[0][0][0]
        state = {"best": chosen[0][1], "best_value": chosen[0][0][0], "trace": [(0.0, chosen[0][0][0])],
                 "accepted": {}, "iterations": 0}
        for index, (score, plan) in enumerate(chosen):
            deadline = begin + time_limit_s * (index + 1) / len(chosen)
            self._descend(plan, score, begin, deadline, time_limit_s / len(chosen), state, max_evaluations)
        return SearchResult(
            lanes=tuple(lane for lane in state["best"] if lane.experts),
            makespan_ns=state["best_value"],
            start_makespan_ns=start_value,
            evaluations=self.evaluations,
            iterations=state["iterations"],
            elapsed_s=time.perf_counter() - begin,
            trace=state["trace"],
            accepted_moves=state["accepted"],
        )

    def _descend(self, start, start_score, begin, deadline, span_s, state, max_evaluations) -> None:
        slice_begin = time.perf_counter()
        incumbent, (value, finish) = start, start_score
        local_best, local_value, local_finish = incumbent, value, finish
        stall = 0
        while time.perf_counter() < deadline:
            if max_evaluations is not None and self.evaluations >= max_evaluations:
                break
            state["iterations"] += 1
            moves = self.neighbors(incumbent, finish)
            if not moves:
                break
            scores = self.evaluate_many([plan for _, plan in moves])
            options = [(score, kind, plan) for (kind, plan), score in zip(moves, scores) if score is not None]
            if not options:
                stall += 1
                continue
            (candidate, candidate_finish), kind, plan = min(options, key=lambda item: item[0][0])
            # Threshold accepting: early in the slice, a slightly worse best neighbor may replace the incumbent.
            progress = min((time.perf_counter() - slice_begin) / max(span_s, 1e-9), 1.0)
            threshold = self.threshold * (1.0 - progress)
            if candidate < value * (1.0 - 1e-9) or (candidate != value and candidate < local_value * (1.0 + threshold)
                                                    and self.rng.random() < 0.3):
                incumbent, value, finish = plan, candidate, candidate_finish
                state["accepted"][kind] = state["accepted"].get(kind, 0) + 1
                stall = 0
                if value < local_value * (1.0 - 1e-9):
                    local_best, local_value, local_finish = incumbent, value, finish
                if value < state["best_value"] * (1.0 - 1e-9):
                    state["best"], state["best_value"] = incumbent, value
                    state["trace"].append((time.perf_counter() - begin, value))
                continue
            stall += 1
            if stall >= self.patience:
                stall = 0
                incumbent = self.perturb(local_best)
                score = self.evaluate_many([incumbent])[0]
                if score is None:
                    incumbent, value, finish = local_best, local_value, local_finish
                else:
                    value, finish = score


def lanes_from_planner_tasks(tasks) -> tuple[Lane, ...]:
    """Convert IntervalPlanner tasks (expert, routes, core_begin, threads, deps) into lanes."""
    lanes: dict[tuple[int, int], list[Expert]] = {}
    for expert, routes, core_begin, threads, _ in tasks:
        lanes.setdefault((int(core_begin), int(threads)), []).append((int(expert), int(routes)))
    return tuple(Lane(width, tuple(experts)) for (_, width), experts in sorted(lanes.items()))


__all__ = [
    "Lane",
    "ModelLnsSearch",
    "SearchResult",
    "canonical",
    "lanes_from_planner_tasks",
    "pack_lanes",
    "placed_from_lanes",
    "planner_tasks",
]
