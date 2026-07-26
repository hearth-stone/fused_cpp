"""Static interval-DAG planner with exact-profile and policy-aware search."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cost_model"))
from phase_model import ContentionCostModel  # noqa: E402
from profile_catalog import ProfileCompatibilityError  # noqa: E402


def _partitions(n: int, parts: Sequence[int]) -> List[Tuple[int, ...]]:
    parts = sorted(parts, reverse=True)
    output: List[Tuple[int, ...]] = []

    def recurse(remaining: int, max_part: int, current: List[int]) -> None:
        if remaining == 0:
            output.append(tuple(current))
            return
        for part in parts:
            if part <= max_part and part <= remaining:
                recurse(remaining - part, part, current + [part])

    recurse(n, max(parts), [])
    return output


def _default_widths(num_cores: int) -> tuple[int, ...]:
    widths: list[int] = []
    value = 1
    while value <= num_cores:
        widths.append(value)
        value *= 2
    return tuple(widths)


class IntervalPlanner:
    def __init__(
        self,
        model: ContentionCostModel,
        num_cores: int = 8,
        widths: Sequence[int] | None = None,
        *,
        cpu_ids: Sequence[int] | None = None,
        shapes: Sequence[Sequence[int]] | None = None,
    ):
        self.model = model
        self.num_cores = int(num_cores)
        self.widths = tuple(widths or _default_widths(self.num_cores))
        self.cpu_ids = tuple(int(cpu) for cpu in (cpu_ids if cpu_ids is not None else range(num_cores)))
        if len(self.cpu_ids) != self.num_cores or len(set(self.cpu_ids)) != num_cores:
            raise ValueError("cpu_ids must contain num_cores unique physical CPUs")

        if shapes is not None:
            candidates = [tuple(int(value) for value in shape) for shape in shapes]
        elif model.schema_version >= 2:
            candidates = list(model.supported_shapes)
        else:
            candidates = _partitions(self.num_cores, self.widths)
        self.shapes = tuple(
            shape
            for shape in candidates
            if sum(shape) == self.num_cores and all(width in self.widths for width in shape)
        )
        if not self.shapes:
            raise ProfileCompatibilityError(
                f"no measured shape covers {self.num_cores} cores with widths {self.widths}"
            )

    def _lanes(self, shape: Tuple[int, ...]):
        begins, core = [], 0
        for width in shape:
            begins.append(core)
            core += width
        return list(zip(begins, shape))

    def _assign(self, experts, lanes):
        lane_count = len(lanes)
        load = [0.0] * lane_count
        lane_experts: List[List[int]] = [[] for _ in range(lane_count)]
        order = sorted(range(len(experts)), key=lambda index: -experts[index][1])
        for index in order:
            _, routes = experts[index]
            lane = min(
                range(lane_count),
                key=lambda candidate: load[candidate] + self.model.T_iso(routes, lanes[candidate][1]),
            )
            lane_experts[lane].append(index)
            load[lane] += self.model.T_iso(routes, lanes[lane][1])
        return lane_experts

    def _build_tasks(self, experts, lanes, lane_experts):
        tasks = []
        for lane, expert_indices in enumerate(lane_experts):
            core_begin, width = lanes[lane]
            previous = None
            for index in expert_indices:
                expert_id, routes = experts[index]
                dependencies = [previous] if previous is not None else []
                tasks.append((expert_id, routes, core_begin, width, dependencies))
                previous = len(tasks) - 1
        return tasks

    def _score(self, tasks) -> float:
        return self.model.dag_makespan([(routes, threads, deps) for _, routes, _, threads, deps in tasks])

    def _uncertainty(self, experts, shape, makespan: float) -> float:
        if self.model.schema_version < 2:
            return 0.0
        if self._uses_full_workload_anchor(experts):
            relative = self.model.relative_full_call_uncertainty(experts[0][1], shape)
            return makespan * relative / math.sqrt(self.model.profile_runs)
        relative = max(self.model.relative_uncertainty(routes, shape) for _, routes in experts)
        waves = max(1, math.ceil(len(experts) / len(shape)))
        return makespan * relative / math.sqrt(waves * self.model.profile_runs)

    def active_working_set_bytes(self, shape, tasks=None) -> int:
        if tasks is None:
            active_lanes = len(shape)
        else:
            active_lanes = len({(core, threads) for _, _, core, threads, _ in tasks})
        return active_lanes * self.model.max_stage_bytes

    def window_bytes_per_worker(self, shape, tasks=None) -> tuple[int, ...]:
        if tasks is None:
            active_widths = tuple(int(width) for width in shape)
        else:
            active_widths = tuple(
                threads
                for _, threads in sorted(
                    {(core, threads) for _, _, core, threads, _ in tasks},
                )
            )
        return tuple(self.model.window_bytes_per_worker(width) for width in active_widths)

    def _uses_full_workload_anchor(self, experts) -> bool:
        return (
            self.model.has_full_workload_anchors
            and len(experts) == self.model.local_experts
            and bool(experts)
            and all(routes == experts[0][1] for _, routes in experts)
        )

    def score_shape(self, experts, shape):
        signature = tuple(int(value) for value in shape)
        if self.model.schema_version >= 2 and not self.model.supports_shape(signature):
            raise ProfileCompatibilityError(f"shape {signature} is not present in {self.model.profile_path.name}")
        lanes = self._lanes(signature)
        tasks = self._build_tasks(experts, lanes, self._assign(experts, lanes))
        if self._uses_full_workload_anchor(experts):
            return self.model.profiled_full_call_time(experts[0][1], signature), tasks
        return self._score(tasks), tasks

    def _candidate(self, experts, shape) -> dict:
        makespan, tasks = self.score_shape(experts, shape)
        uncertainty = self._uncertainty(experts, shape, makespan)
        return {
            "shape": tuple(shape),
            "makespan_ns": makespan,
            "uncertainty_ns": uncertainty,
            "pessimistic_ns": makespan + uncertainty,
            "tasks": tasks,
            "active_working_set_bytes": self.active_working_set_bytes(shape, tasks),
            "window_bytes_per_worker": self.window_bytes_per_worker(shape, tasks),
        }

    @staticmethod
    def _select(candidates: list[dict]) -> dict:
        candidates.sort(key=lambda candidate: candidate["makespan_ns"])
        fastest = candidates[0]
        fastest_lower = fastest["makespan_ns"] - fastest["uncertainty_ns"]
        fastest_upper = fastest["pessimistic_ns"]
        overlapping = [
            candidate
            for candidate in candidates
            if candidate["makespan_ns"] - candidate["uncertainty_ns"] <= fastest_upper
            and candidate["pessimistic_ns"] >= fastest_lower
        ]
        return min(
            overlapping,
            key=lambda candidate: (
                candidate["active_working_set_bytes"],
                len(candidate["shape"]),
                candidate["pessimistic_ns"],
                candidate["makespan_ns"],
            ),
        )

    def plan(self, experts: List[Tuple[int, int]]) -> Dict[str, object]:
        experts = [(expert, routes) for expert, routes in experts if routes > 0]
        if not experts:
            raise ValueError("at least one active expert is required")
        candidates = [self._candidate(experts, shape) for shape in self.shapes]
        selected = self._select(candidates)
        policy = None
        if self.model.policy is not None:
            policy = {
                "profile": str(self.model.profile_path),
                "w13_split": self.model.policy.w13_split,
                "w13_split_chunks": self.model.policy.w13_split_chunks,
                "weight_window_bytes": self.model.policy.weight_window_bytes,
                "w13_window_ranges": self.model.policy.w13_window_ranges,
                "w2_window_ranges": self.model.policy.w2_window_ranges,
                "intermediate_size": self.model.policy.intermediate_size,
                "mode": self.model.policy.mode,
                "degree": self.model.policy.degree,
            }
        return {
            "shape": selected["shape"],
            "makespan_ns": selected["makespan_ns"],
            "uncertainty_ns": selected["uncertainty_ns"],
            "active_working_set_bytes": selected["active_working_set_bytes"],
            "w13_split": (self.model.policy.w13_split if self.model.policy is not None else None),
            "weight_window_bytes": (self.model.policy.weight_window_bytes if self.model.policy is not None else None),
            "w13_window_ranges": (self.model.policy.w13_window_ranges if self.model.policy is not None else None),
            "w2_window_ranges": (self.model.policy.w2_window_ranges if self.model.policy is not None else None),
            "window_bytes_per_worker": selected["window_bytes_per_worker"],
            "policy": policy,
            "tasks": selected["tasks"],
            "bridge": self.to_async_bridge(selected["tasks"]),
            "ranking": [
                {
                    "shape": candidate["shape"],
                    "makespan_ms": round(candidate["makespan_ns"] / 1e6, 6),
                    "pessimistic_ms": round(candidate["pessimistic_ns"] / 1e6, 6),
                    "active_working_set_bytes": candidate["active_working_set_bytes"],
                    "window_bytes_per_worker": candidate["window_bytes_per_worker"],
                }
                for candidate in sorted(candidates, key=lambda candidate: candidate["makespan_ns"])
            ],
        }

    def to_async_bridge(self, tasks) -> Dict[str, object]:
        dependency_offsets, flat_dependencies = [0], []
        for _, _, _, _, dependencies in tasks:
            flat_dependencies.extend(dependencies)
            dependency_offsets.append(len(flat_dependencies))
        return {
            "num_threads": self.num_cores,
            "thread_cpu_ids": list(self.cpu_ids),
            "task_expert_ids": [expert for expert, _, _, _, _ in tasks],
            "task_core_begins": [core for _, _, core, _, _ in tasks],
            "task_threads": [threads for _, _, _, threads, _ in tasks],
            "task_dep_offsets": dependency_offsets,
            "task_deps": flat_dependencies,
        }


class PolicyAwarePlanner:
    """Search packed-B window policy and an exact profiled core shape together."""

    def __init__(
        self,
        models: Sequence[ContentionCostModel],
        num_cores: int,
        *,
        cpu_ids: Sequence[int] | None = None,
        working_set_target_fraction: float = 2.0 / 3.0,
    ):
        if not models:
            raise ValueError("at least one policy model is required")
        policies = [model.policy for model in models]
        if any(policy is None for policy in policies):
            raise ProfileCompatibilityError("joint policy search requires schema-v2 profiles")
        base_key = policies[0].key_without_kernel_policy()
        if any(policy.key_without_kernel_policy() != base_key for policy in policies[1:]):
            raise ProfileCompatibilityError("joint planner profiles differ outside the packed-B window policy")
        variant_keys = [policy.kernel_policy_key() for policy in policies]
        if len(variant_keys) != len(set(variant_keys)):
            raise ProfileCompatibilityError("joint planner received duplicate packed-B window policies")
        self.models = tuple(models)
        self.num_cores = int(num_cores)
        self.cpu_ids = cpu_ids
        self.working_set_target_fraction = float(working_set_target_fraction)
        self.planners = tuple(
            IntervalPlanner(
                model,
                num_cores,
                cpu_ids=cpu_ids,
                shapes=self._pruned_shapes(model),
            )
            for model in self.models
        )

    def _pruned_shapes(self, model: ContentionCostModel):
        policy = model.policy
        assert policy is not None
        target = policy.llc_bytes_per_rank * self.working_set_target_fraction
        shapes = [shape for shape in model.supported_shapes if sum(shape) == self.num_cores]
        by_distance = sorted(
            shapes,
            key=lambda shape: abs(len(shape) * model.max_stage_bytes - target),
        )
        keep = {shape for shape in shapes if len(shape) == 1 or len(shape) * model.max_stage_bytes <= target * 1.25}
        keep.update(by_distance[:2])
        lane_counts = {len(shape) for shape in keep}
        for shape in shapes:
            if len(shape) in lane_counts:
                keep.add(shape)
        return tuple(sorted(keep, key=lambda shape: (len(shape), shape), reverse=False))

    def plan(self, experts: List[Tuple[int, int]]) -> Dict[str, object]:
        policy_results = [planner.plan(experts) for planner in self.planners]
        candidates = []
        for result in policy_results:
            candidates.append(
                {
                    "result": result,
                    "makespan_ns": result["makespan_ns"],
                    "uncertainty_ns": result["uncertainty_ns"],
                    "pessimistic_ns": result["makespan_ns"] + result["uncertainty_ns"],
                    "active_working_set_bytes": result["active_working_set_bytes"],
                    "shape": result["shape"],
                }
            )
        selected = IntervalPlanner._select(candidates)["result"]
        selected["policy_ranking"] = [
            {
                "w13_split": result["w13_split"],
                "weight_window_bytes": result["weight_window_bytes"],
                "w13_window_ranges": result["w13_window_ranges"],
                "w2_window_ranges": result["w2_window_ranges"],
                "shape": result["shape"],
                "makespan_ms": result["makespan_ns"] / 1e6,
                "uncertainty_ms": result["uncertainty_ns"] / 1e6,
                "active_working_set_bytes": result["active_working_set_bytes"],
                "window_bytes_per_worker": result["window_bytes_per_worker"],
            }
            for result in sorted(policy_results, key=lambda result: result["makespan_ns"])
        ]
        return selected


if __name__ == "__main__":
    model = ContentionCostModel(sys.argv[1])
    planner = IntervalPlanner(model, num_cores=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    result = planner.plan([(index, 192) for index in range(8)])
    print(result)
