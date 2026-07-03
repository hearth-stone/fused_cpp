#!/usr/bin/env python3
# ⚠ DEPRECATED (wave, 后续不考虑) — see cpu_moe_schedule_optimization/DEPRECATED_WAVE.md
"""Twin implementation of KARMARKAR_KARP planner for Python equivalence testing.

This implements the IDENTICAL multiway Largest Differencing Method (LDM)
algorithm as the C++ plan_karmarkar_karp.cpp, ensuring byte-identical
deterministic queue assignments.
"""

from __future__ import annotations

from typing import List, Sequence

# Import from the reference simulator module.
import offline_simulator as sim


def plan_karmarkar_karp(
    active: Sequence[sim.ExpertWork],
    num_cores: int,
    cost_model: sim.ExpertCostModel,
    planner_cost_model: sim.PlannerCostModel,
) -> sim.Plan:
    """KARMARKAR_KARP planner: multiway LDM for 1-thread C-way balancing.

    This is the twin of csrc/moe_planner/plan_karmarkar_karp.cpp and must
    implement the IDENTICAL algorithm for cross-language equivalence testing.

    Args:
        active: Sequence of ExpertWork objects (already sorted by -routes, expert_id)
        num_cores: Number of logical core queues (C)
        cost_model: Expert cost model for estimating execution time
        planner_cost_model: Planner overhead cost model

    Returns:
        Plan with waves built from LDM-balanced queues
    """
    import time

    start_ns = time.perf_counter_ns()

    C = num_cores
    A = len(active)

    # Handle empty workload
    if A == 0:
        measured_plan_cost_ns = time.perf_counter_ns() - start_ns
        return sim.Plan(
            kind=sim.PlanKind.SORTED_TOKEN_BALANCED_1T,
            num_cores=num_cores,
            active_experts=list(active),
            waves=[],
            measured_plan_cost_ns=measured_plan_cost_ns,
            estimated_plan_cost_ns=0,
            exactness_scope="heuristic",
            metadata={},
        )

    # Build active index mapping: active index -> routes, expert_id
    routes = [work.routes for work in active]
    expert_ids = [work.expert_id for work in active]

    # LDM Node representation
    class LdmNode:
        __slots__ = ("loads", "buckets", "insertion_index")

        def __init__(self, C: int, insertion_index: int):
            self.loads: List[int] = [0] * C  # C loads, sorted DESCENDING
            self.buckets: List[List[int]] = [[] for _ in range(C)]  # C buckets of active indices
            self.insertion_index = insertion_index

    def compute_spread(node: LdmNode) -> int:
        """Compute spread (max - min) for a node."""
        if C == 0:
            return 0
        return node.loads[0] - node.loads[C - 1]

    def node_key(node: LdmNode):
        """Return a key tuple for deterministic node ordering.

        Ordering: spread desc, loads[0] desc, full loads lexicographically desc,
        then insertion_index asc (earlier node wins ties).
        """
        # Use negative for descending, positive for ascending
        return (
            -compute_spread(node),
            -node.loads[0],
            tuple(-load for load in node.loads),
            node.insertion_index,
        )

    # Initialize: one node per active expert
    nodes: List[LdmNode] = []
    for a in range(A):
        node = LdmNode(C, a)  # insertion_index = a
        node.loads[0] = routes[a]
        node.buckets[0].append(a)
        # loads already sorted: [routes[a], 0, ..., 0]
        nodes.append(node)

    # LDM pairing loop
    while len(nodes) > 1:
        # Find top two nodes: X (highest rank), Y (second highest)
        # Sort by key to find top two deterministically
        sorted_indices = sorted(range(len(nodes)), key=lambda i: node_key(nodes[i]))
        idx_x = sorted_indices[0]  # highest rank (smallest key)
        idx_y = sorted_indices[1]  # second highest rank

        # Extract X and Y (remove higher index first to preserve lower)
        if idx_x > idx_y:
            idx_x, idx_y = idx_y, idx_x

        Y = nodes.pop(idx_y)
        X = nodes.pop(idx_x)

        # COMBINE: X.loads already descending.
        # Take Y in ASCENDING load order (reverse of its descending loads).
        y_order = list(range(C - 1, -1, -1))  # [C-1, C-2, ..., 0] for ascending

        next_ins_idx = len(nodes)  # will be appended
        merged = LdmNode(C, next_ins_idx)

        for i in range(C):
            y_i = y_order[i]  # Y's index for ascending position i
            merged.loads[i] = X.loads[i] + Y.loads[y_i]
            merged.buckets[i] = X.buckets[i] + Y.buckets[y_i]  # concatenate

        # Stable sort the C (load, bucket) pairs by load DESCENDING.
        # Ties keep current relative order.
        pairs = list(zip(merged.loads, merged.buckets))
        pairs.sort(key=lambda p: -p[0])  # descending by load (stable sort in Python)

        for i in range(C):
            merged.loads[i] = pairs[i][0]
            merged.buckets[i] = pairs[i][1]

        # Append merged node
        nodes.append(merged)

    # Final node contains the C queues
    queues = nodes[0].buckets

    # Within each queue, sort by descending routes then ascending expert_id
    for q in range(C):
        queues[q].sort(
            key=lambda a: (-routes[a], expert_ids[a])
        )

    # Emit waves by depth: wave d gathers queue[q][d] for q = 0..C-1
    # This mirrors build_waves_from_sorted_token_balanced_queues
    waves: List[sim.Wave] = []
    max_depth = max((len(queue) for queue in queues), default=0)

    for depth in range(max_depth):
        teams = []
        for q in range(C):
            if depth < len(queues[q]):
                a = queues[q][depth]
                team = sim.make_team(active[a], 1, cost_model)
                teams.append(team)

        if teams:
            wave = sim.Wave(
                wave_id=len(waves),
                teams=teams,
                estimated_wave_time_ns=max(team.estimated_time_ns for team in teams),
            )
            waves.append(wave)

    measured_plan_cost_ns = time.perf_counter_ns() - start_ns
    estimated_plan_cost_ns = planner_cost_model.estimate_ns(
        sim.PlanKind.SORTED_TOKEN_BALANCED_1T,
        measured_plan_cost_ns,
        active_count=len(active),
        num_cores=num_cores,
    )

    metadata = {
        "planner": "twin_karmarkar_karp.py",
        "cost_model": cost_model.source,
        "cost_metric": cost_model.metric,
        "model_note": (
            "multiway Karmarkar-Karp (LDM): one-thread experts assigned to "
            "C logical queues via differencing method"
        ),
        "assignment": "karmarkar_karp_ldm",
    }
    metadata.update(
        planner_cost_model.metadata_for(
            sim.PlanKind.SORTED_TOKEN_BALANCED_1T,
            active_count=len(active),
            num_cores=num_cores,
        )
    )

    return sim.Plan(
        kind=sim.PlanKind.SORTED_TOKEN_BALANCED_1T,
        num_cores=num_cores,
        active_experts=list(active),
        waves=waves,
        measured_plan_cost_ns=measured_plan_cost_ns,
        estimated_plan_cost_ns=estimated_plan_cost_ns,
        exactness_scope="heuristic",
        metadata=metadata,
    )
