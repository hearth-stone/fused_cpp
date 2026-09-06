#!/usr/bin/env python3
"""Build the Task 6A full-parent cross-domain d4 hardware-reference pool.

Lab diagnostic only. Uses shipped ``enumerate_template_lns_neighbors`` and
global canonical-hash ownership before filtering to one operator. Does not
change production sampling, selector v1, or K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "cost_model"),
    str(REPO_ROOT / "cpu_moe_schedule_optimization" / "planners"),
]

from analytic_model import AnalyticMoeCostModel  # noqa: E402
from executable_plan_neighborhood import (  # noqa: E402
    TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
    TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
    ExecutablePlanNeighbor,
    enumerate_template_lns_neighbors,
)
from interval_planner import IntervalPlanner  # noqa: E402

from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import (  # noqa: E402
    compact_structure,
    state_from_canonical_payload,
    width_histogram,
)

TARGET_OPERATOR = "critical_window_template_repartition_cross_domain_d4_b16"
FULL_PARENT_HASH = "98a32da5c3ee195802fbc4291ecf0227f57af8f94dcd4330653f3b6244c2d10e"
PREVIOUS_SELECTED = "0418b88445c2f88488ca10a1eeae3aeb7b6080e806e241eacfed17ec10904bf6"
KNOWN_ELITE = "2ab43572d14e11060e4fc07b3aa7e0c1780553233dfda61abe74f08abc51e973"
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "bench_assets/moe_paper/arm_codex_numa3_80c_temporal"
    / "analytic_machine_numa3_80c_narrow_merge_v8_20260903.json"
)
SCALE_PLANS = 85
SCALE_WALL_S = 100.89228965


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_sorted_hashes(hashes: Sequence[str]) -> str:
    payload = json.dumps(sorted(set(hashes)), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_registered_artifact(path: Path | None, expected_sha256: str, *, label: str) -> Path:
    digest = str(expected_sha256)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"unavailable {label} artifact: {digest} has no registered path")
    actual = sha256_file(path)
    if actual != digest:
        raise ValueError(f"{label} sha256 mismatch: expected {digest} got {actual}")
    return path


def unique_then_operator_filter(
    neighbors: Iterable[ExecutablePlanNeighbor],
    *,
    operator: str,
    baseline_hash: str,
) -> list[ExecutablePlanNeighbor]:
    seen = {str(baseline_hash)}
    owned: list[ExecutablePlanNeighbor] = []
    for neighbor in neighbors:
        hashed = neighbor.state.canonical_hash()
        if hashed in seen:
            continue
        seen.add(hashed)
        if neighbor.operator == operator:
            owned.append(neighbor)
    return owned


def order_signature(state: Any) -> str:
    payload = [[int(task.expert_id) for task in lane.tasks] for lane in state.lanes]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def member_provenance(neighbor: ExecutablePlanNeighbor) -> dict[str, Any]:
    state = neighbor.state
    structure = compact_structure(state)
    payload = state.canonical_payload()
    restored = state_from_canonical_payload(payload)
    bridge = state.to_bridge()
    if restored.canonical_hash() != state.canonical_hash():
        raise ValueError(f"canonical round trip failed for {state.canonical_hash()}")
    if bridge.get("execution_mode") != "strict":
        raise ValueError(f"PlanV2 bridge is not strict for {state.canonical_hash()}")
    return {
        "actual_closure_size": len(neighbor.moved_experts),
        "canonical_state": payload,
        "domain_assignment_signature": structure["domain_assignment_signature"],
        "operator": neighbor.operator,
        "order_signature": order_signature(state),
        "plan_v2_bridge": bridge,
        "shape": list(state.shape),
        "state_hash": state.canonical_hash(),
        "width_histogram": [list(pair) for pair in width_histogram(state)],
    }


def pooled_selected_keys(model: Mapping[str, Any], *, pooling_enabled: bool) -> list[str]:
    if pooling_enabled:
        pooled = model.get("parent_pooled_shortlists") or {}
        if pooled:
            keys: list[str] = []
            for record in pooled.values():
                shortlist = record.get("shortlist") or {}
                keys.extend(str(key) for key in shortlist.get("selected_keys") or [])
            return sorted(set(keys))
    keys = []
    for run in (model.get("runs") or {}).values():
        shortlist = run["iterations"][0]["lns_diverse_shortlist"]
        keys.extend(str(key) for key in shortlist.get("selected_keys") or [])
    return sorted(set(keys))


def per_start_selected_keys(model: Mapping[str, Any]) -> list[str]:
    return pooled_selected_keys(model, pooling_enabled=False)


def account_reference_roles(
    pool_hashes: Sequence[str],
    *,
    controls: Mapping[str, str],
    elites: Mapping[str, str],
    pooled_selected: Sequence[str],
    per_start_selected: Sequence[str],
    pooling_enabled: bool,
) -> dict[str, Any]:
    pool = set(pool_hashes)
    records: dict[str, dict[str, Any]] = {}

    def add(state_hash: str, role: str, *, proposal: bool) -> None:
        record = records.setdefault(
            state_hash,
            {
                "in_operator_pool": state_hash in pool,
                "is_proposal": False,
                "roles": [],
                "state_hash": state_hash,
            },
        )
        if role not in record["roles"]:
            record["roles"].append(role)
        if proposal:
            record["is_proposal"] = True

    for role, state_hash in controls.items():
        add(str(state_hash), f"anchor:{role}", proposal=False)
    for role, state_hash in elites.items():
        add(str(state_hash), str(role), proposal=False)
    final_selected = pooled_selected if pooling_enabled else per_start_selected
    for state_hash in final_selected:
        add(str(state_hash), "pooled_selected" if pooling_enabled else "per_start_selected", proposal=True)
    for state_hash in per_start_selected:
        if pooling_enabled:
            add(str(state_hash), "per_start_selected_intermediate", proposal=False)
    return {
        "elites_in_operator_pool": {
            role: str(state_hash) in pool for role, state_hash in elites.items()
        },
        "final_selected_keys": sorted(set(final_selected)),
        "pooling_enabled": pooling_enabled,
        "records": records,
    }


def join_session_pair(
    name: str,
    frontier: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    pool_hashes: Sequence[str],
    full_hash: str,
) -> dict[str, Any]:
    pool = set(pool_hashes)
    stats1 = first["plan_stats"]
    stats2 = second["plan_stats"]
    measured = sorted(hash_ for hash_ in stats1 if hash_ in pool)
    repeated = sorted(hash_ for hash_ in measured if hash_ in stats2)
    unmeasured = sorted(pool - set(stats1) - set(stats2))
    labels: dict[str, str] = {}
    if full_hash in stats1 and full_hash in stats2:
        full1 = float(stats1[full_hash]["median_ms"])
        full2 = float(stats2[full_hash]["median_ms"])
        for hash_ in repeated:
            gain1 = 100.0 * (full1 / float(stats1[hash_]["median_ms"]) - 1.0)
            gain2 = 100.0 * (full2 / float(stats2[hash_]["median_ms"]) - 1.0)
            if gain1 > 2.0 and gain2 > 2.0:
                labels[hash_] = "stable_fast_vs_full"
            elif gain1 > 2.0 or gain2 > 2.0:
                labels[hash_] = "single_session_positive_vs_full"
            else:
                labels[hash_] = "not_fast_vs_full"
    roles = {
        hash_: list(frontier.get("records", {}).get(hash_, {}).get("roles") or [])
        for hash_ in measured
    }
    return {
        "frontier_sha256": first.get("identity", {}).get("frontier_sha256"),
        "measured": measured,
        "measured_count": len(measured),
        "measurement_labels": labels,
        "name": name,
        "repeated": repeated,
        "repeated_count": len(repeated),
        "roles": roles,
        "unmeasured": unmeasured,
        "unmeasured_count": len(unmeasured),
    }


def estimate_budget(*, unique_plans: int, reference_plans: int = SCALE_PLANS, reference_wall_s: float = SCALE_WALL_S) -> dict[str, Any]:
    per_plan_s = float(reference_wall_s) / float(reference_plans)
    wall_s = unique_plans * per_plan_s
    timed_calls = (5 + 31) * unique_plans
    return {
        "comparisons_note": "session JSON comparisons are versus recorded anchors, not all pairs",
        "expected_two_session_wall_s": round(2.0 * wall_s, 3),
        "expected_wall_s_per_session": round(wall_s, 3),
        "label": "estimate",
        "memory_estimate": (
            "Four rotating packed-weight copies dominate RSS; plan objects add "
            f"on the order of {unique_plans} executable states. Search-model RSS "
            "on this protocol was ~340 MiB; treat extra plan serialization as "
            "additional, not a measured hardware RSS."
        ),
        "proposed_command": (
            "numactl --physcpubind=240-319 --membind=3 .venv/bin/python "
            "optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py "
            "--frontier <task6a_frontier.json> --warmup 5 --runs 31 --weight-copies 4"
        ),
        "reference_plans": reference_plans,
        "reference_session_wall_s": reference_wall_s,
        "scale": "linear in timed plans from two-restart session 1 (85 plans, 100.892 s)",
        "timed_calls_per_session": timed_calls,
        "unique_plans_after_dedup": unique_plans,
        "warmup": 5,
        "weight_copies": 4,
        "runs": 31,
    }


def window_selector_for_model(model: Mapping[str, Any], calibration: Path):
    analytic = AnalyticMoeCostModel(
        calibration,
        hidden_size=int(model["shape"]["hidden"]),
        intermediate_size=int(model["shape"]["intermediate"]),
        global_experts=int(model["shape"]["experts"]),
        local_experts=int(model["shape"]["experts"]),
        mode="tp",
        degree=4,
        concurrent_ranks=1,
        down_output_element_bytes=4,
    )
    interval = IntervalPlanner(
        analytic,
        num_cores=int(model["shape"]["threads"]),
        cpu_ids=list(range(int(model["shape"]["threads"]))),
    )
    policy = interval._stage_window_policy()

    def window_selector(routes: int, threads: int) -> tuple[int, int]:
        return (0, 0) if policy is None else policy.select(routes, threads)

    return analytic, interval, window_selector


def enumerate_operator_pool(
    model: Mapping[str, Any],
    start_name: str,
    *,
    calibration: Path,
    operator: str = TARGET_OPERATOR,
) -> dict[str, Any]:
    run = model["runs"][start_name]
    parent_hash = str(run["initial_state_hash"])
    if parent_hash != FULL_PARENT_HASH:
        raise ValueError(f"{start_name} is not the frozen full parent")
    state = state_from_canonical_payload(run["initial_canonical_state"])
    if state.canonical_hash() != parent_hash:
        raise ValueError(f"{start_name} canonical hash mismatch")
    analytic, interval, window_selector = window_selector_for_model(model, calibration)
    critical_ids = list(run["iterations"][0]["critical_expert_ids"])
    neighbors = list(
        enumerate_template_lns_neighbors(
            state,
            allowed_widths=interval.widths,
            isolated_cost=analytic.T_iso,
            window_selector=window_selector,
            critical_expert_ids=critical_ids,
            destroy_sizes=TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
            repair_beam_widths=TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
            templates_per_block=int(model["method"]["templates_per_block"]),
        )
    )
    owned = unique_then_operator_filter(
        neighbors,
        operator=operator,
        baseline_hash=parent_hash,
    )
    members = [member_provenance(neighbor) for neighbor in owned]
    hashes = [row["state_hash"] for row in members]
    return {
        "critical_expert_ids": critical_ids,
        "emitted": len(neighbors),
        "hashes": hashes,
        "members": members,
        "parent_state_hash": parent_hash,
        "start": start_name,
        "unique_operator_count": len(hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-20261010", type=Path, required=True)
    parser.add_argument("--model-20261011", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--join-dir", type=Path, required=True)
    parser.add_argument("--pool-output", type=Path, required=True)
    parser.add_argument("--join-output", type=Path, required=True)
    parser.add_argument("--budget-output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    model_10 = _load_json(
        require_registered_artifact(
            args.model_20261010,
            "9b334b784e80c8376aa059084875a75d4f323c2a4c8f4192e60288f4590595b8",
            label="seed20261010 model",
        )
    )
    model_11 = _load_json(
        require_registered_artifact(
            args.model_20261011,
            "cc53d43d5827a353ba4325c8a787e0302fcff20a0ebc3840752acfa824d3238a",
            label="seed20261011 model",
        )
    )
    print("enumerate 20261010 lns_00_r00", flush=True)
    pool_10_r00 = enumerate_operator_pool(model_10, "lns_00_r00", calibration=args.calibration)
    print("enumerate 20261010 lns_00_r01", flush=True)
    pool_10_r01 = enumerate_operator_pool(model_10, "lns_00_r01", calibration=args.calibration)
    print("enumerate 20261011 lns_00_r00", flush=True)
    pool_11_r00 = enumerate_operator_pool(model_11, "lns_00_r00", calibration=args.calibration)
    print("enumerate 20261011 lns_00_r01", flush=True)
    pool_11_r01 = enumerate_operator_pool(model_11, "lns_00_r01", calibration=args.calibration)
    sets = {
        "20261010_lns_00_r00": set(pool_10_r00["hashes"]),
        "20261010_lns_00_r01": set(pool_10_r01["hashes"]),
        "20261011_lns_00_r00": set(pool_11_r00["hashes"]),
        "20261011_lns_00_r01": set(pool_11_r01["hashes"]),
    }
    primary = pool_10_r00
    emission_order_hashes = list(primary["hashes"])
    hashes = sorted(emission_order_hashes)
    digest = sha256_sorted_hashes(hashes)
    controls = {
        "full": str(model_10["parents"]["lns_00_r00"]["state_hash"]),
        "one_step": str(model_10["parents"]["lns_01_r00"]["state_hash"]),
        "greedy": str(model_10["parents"]["lns_02_r00"]["state_hash"]),
        "fixed_width": str(model_10["parents"]["lns_03_r00"]["state_hash"]),
    }
    elites = {
        "known_hardware_elite": KNOWN_ELITE,
        "previous_seed_selected": PREVIOUS_SELECTED,
    }
    pooling_enabled = bool(model_10.get("method", {}).get("pool_restarts_by_parent"))
    roles = account_reference_roles(
        hashes,
        controls=controls,
        elites=elites,
        pooled_selected=pooled_selected_keys(model_10, pooling_enabled=pooling_enabled),
        per_start_selected=per_start_selected_keys(model_10),
        pooling_enabled=pooling_enabled,
    )
    unique_hashes = sorted(set(hashes) | set(controls.values()) | set(elites.values()))
    pool_payload = {
        "kind": "lns_task6a_reference_pool",
        "operator": TARGET_OPERATOR,
        "parent_state_hash": FULL_PARENT_HASH,
        "reported_unique_count": 452,
        "unique_count": len(hashes),
        "sorted_hash_sha256": digest,
        "hashes": hashes,
        "emission_order_hashes": emission_order_hashes,
        "seed_hash_sets_equal": all(value == sets["20261010_lns_00_r00"] for value in sets.values()),
        "seed_set_sizes": {name: len(value) for name, value in sets.items()},
        "seed_set_digests": {name: sha256_sorted_hashes(value) for name, value in sets.items()},
        "elites_in_operator_pool": {
            "known_hardware_elite": KNOWN_ELITE in hashes,
            "previous_seed_selected": PREVIOUS_SELECTED in hashes,
        },
        "members": primary["members"],
        "roles": roles,
        "unique_plans_with_reference_roles": len(unique_hashes),
        "critical_expert_ids": primary["critical_expert_ids"],
        "emitted": primary["emitted"],
        "model_sha256": {
            "20261010": sha256_file(args.model_20261010),
            "20261011": sha256_file(args.model_20261011),
        },
    }
    args.pool_output.parent.mkdir(parents=True, exist_ok=True)
    args.pool_output.write_text(json.dumps(pool_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.pool_output, sha256_file(args.pool_output), "n", len(hashes), flush=True)

    join_dir = args.join_dir
    artifacts = [
        (
            "two_restart",
            _load_json(join_dir / "two_restart_frontier.json"),
            _load_json(join_dir / "two_restart_session1.json"),
            _load_json(join_dir / "two_restart_session2.json"),
        ),
        (
            "one_restart",
            _load_json(join_dir / "one_restart_frontier.json"),
            _load_json(join_dir / "one_restart_session1.json"),
            _load_json(join_dir / "one_restart_session2.json"),
        ),
        (
            "second_seed",
            _load_json(join_dir / "second_seed_frontier.json"),
            _load_json(join_dir / "second_seed_session1.json"),
            _load_json(join_dir / "second_seed_session2.json"),
        ),
        (
            "independent_median",
            _load_json(join_dir / "median_lns_diverse_frontier.json"),
            _load_json(join_dir / "median_lns_diverse_session1.json"),
            _load_json(join_dir / "median_lns_diverse_session2.json"),
        ),
    ]
    joins = [
        join_session_pair(name, frontier, first, second, hashes, FULL_PARENT_HASH)
        for name, frontier, first, second in artifacts
    ]
    union_measured = sorted({hash_ for row in joins for hash_ in row["measured"]})
    join_payload = {
        "kind": "lns_task6a_evidence_join",
        "not_a_density_oracle": True,
        "not_a_common_timing_scale": True,
        "pool_hash_sha256": digest,
        "pool_unique_count": len(hashes),
        "sessions": joins,
        "union_measured_count": len(union_measured),
        "union_measured": union_measured,
        "unmeasured_in_all_sessions": sorted(set(hashes) - set(union_measured)),
    }
    args.join_output.write_text(json.dumps(join_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.join_output, sha256_file(args.join_output), flush=True)

    budget = estimate_budget(unique_plans=len(unique_hashes))
    budget_payload = {
        "kind": "lns_task6a_diagnostic_budget",
        "authorization_required": True,
        "ceiling_before_dedup": 458,
        "exceeds_existing_80_lns_candidate_slots": True,
        "not_a_change_to_default_n25_k16_search": True,
        "task5_closed": True,
        "task6b_not_authorized_by_6a": True,
        "unique_plans_after_dedup": len(unique_hashes),
        "operator_pool_count": len(hashes),
        "control_count": len(set(controls.values())),
        "elite_roles": elites,
        **budget,
    }
    args.budget_output.write_text(json.dumps(budget_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", args.budget_output, sha256_file(args.budget_output), flush=True)
    print(json.dumps({"unique_count": len(hashes), "digest": digest, "sets_equal": pool_payload["seed_hash_sets_equal"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
