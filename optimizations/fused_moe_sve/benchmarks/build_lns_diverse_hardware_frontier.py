#!/usr/bin/env python3
"""Build a nested LNS hardware frontier from a diverse-shortlist model replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import (  # noqa: E402
    LnsDiverseShortlist,
    select_lns_global_shortlist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lns-artifact", type=Path, required=True)
    parser.add_argument("--known-elite-plan", type=Path)
    parser.add_argument("--reference-plan", type=Path)
    parser.add_argument("--include-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-stratified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_control_plan(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    state_hash = str(payload["state_hash"])
    canonical_state = payload["canonical_state"]
    bridge = payload["plan_v2_bridge"]
    if not isinstance(canonical_state, dict) or not isinstance(bridge, dict):
        raise ValueError(f"{label} plan is missing canonical state or PlanV2 bridge")
    if _canonical_hash(canonical_state) != state_hash:
        raise ValueError(f"{label} canonical hash mismatch: {state_hash}")
    return {
        "state_hash": state_hash,
        "canonical_state": canonical_state,
        "plan_v2_bridge": bridge,
        "source": str(path),
        "sha256": _sha256(path),
    }


def _load_known_elite(path: Path) -> dict[str, object]:
    return _load_control_plan(path, label="known elite")


def _full_parent_hash(payload: dict[str, object], runs: dict[str, object]) -> str | None:
    full_hashes = [
        str(run["initial_state_hash"])
        for start_name, run in runs.items()
        if "full" in payload.get("parents", {}).get(start_name, {}).get("roles", [])
        or "full" in payload.get("parents", {}).get(run.get("start"), {}).get("roles", [])
    ]
    if full_hashes:
        return full_hashes[0]
    named = payload.get("identity", {}).get("parent_source", {}).get("named_state_hashes", {})
    if isinstance(named, dict) and named.get("full"):
        return str(named["full"])
    return None


def build_lns_diverse_frontier(
    payload: dict[str, object],
    *,
    source_path: Path,
    known_elite: dict[str, object] | None = None,
    reference_plan: dict[str, object] | None = None,
    include_audit: bool = True,
    include_stratified: bool = True,
) -> dict[str, object]:
    if payload.get("kind") != "executable_partial_order_template_lns_model_replay":
        raise ValueError("input must be a template-LNS model replay")
    runs = payload.get("runs")
    expected_starts = tuple(payload.get("method", {}).get("start_names", ()))
    if not isinstance(runs, dict) or not expected_starts or set(runs) != set(expected_starts):
        raise ValueError("search replay runs do not match method.start_names")

    plans: dict[str, dict[str, object]] = {}
    records: dict[str, dict[str, object]] = {}
    per_start: dict[str, LnsDiverseShortlist] = {}

    def register(
        state_hash: str,
        canonical_state: dict[str, object],
        bridge: dict[str, object],
        source: dict[str, object],
    ) -> None:
        if _canonical_hash(canonical_state) != state_hash:
            raise ValueError(f"canonical payload does not match state hash {state_hash}")
        existing = plans.setdefault(
            state_hash,
            {
                "canonical_state": canonical_state,
                "plan_v2_bridge": bridge,
            },
        )
        if existing["canonical_state"] != canonical_state or existing["plan_v2_bridge"] != bridge:
            raise ValueError(f"state hash collision or inconsistent plan for {state_hash}")
        record = records.setdefault(
            state_hash,
            {
                "state_hash": state_hash,
                "roles": [],
                "comparisons": [],
            },
        )
        role = str(source["role"])
        if role not in record["roles"]:
            record["roles"].append(role)
        record["comparisons"].append(source)

    shortlist_budget = int(payload["method"].get("shortlist_budget", 16))
    audit_budget = int(payload["method"].get("audit_budget", 32))
    pooled = payload.get("parent_pooled_shortlists") or {}
    pooled_rows = payload.get("pooled_frontier_rows") or {}
    for start_name, run in runs.items():
        iterations = run.get("iterations", [])
        if len(iterations) != 1:
            raise ValueError(f"{start_name} must stop after the audited first iteration")
        iteration = iterations[0]
        if iteration["partial_order"].get("automatic_acceptance_enabled") is not False:
            raise ValueError(f"{start_name} must disable automatic acceptance")
        if iteration["partial_order"].get("dominance_pruning_enabled") is not False:
            raise ValueError(f"{start_name} must disable dominance pruning")
        if iteration.get("worse_sentinels"):
            raise ValueError(f"{start_name} must not emit dominated or sentinel LNS categories")
        shortlist_payload = iteration.get("lns_diverse_shortlist")
        if not isinstance(shortlist_payload, dict):
            raise ValueError(f"{start_name} is missing lns_diverse_shortlist")
        shortlist = LnsDiverseShortlist.from_dict(shortlist_payload)
        if shortlist.selected_keys != shortlist.audit_keys[: len(shortlist.selected_keys)]:
            raise ValueError(f"{start_name} selected_keys are not a prefix of audit_keys")
        per_start[start_name] = shortlist
        anchor_hash = str(run["initial_state_hash"])
        register(
            anchor_hash,
            run["initial_canonical_state"],
            run["initial_plan_v2_bridge"],
            {
                "role": "anchor",
                "start": start_name,
                "anchor_state_hash": anchor_hash,
            },
        )
        if pooled:
            continue
        audit_rows = {
            str(row["state_hash"]): row
            for row in iteration.get("audit_frontier") or iteration.get("selected_frontier") or []
        }
        measured_keys = shortlist.audit_keys if include_audit else shortlist.selected_keys
        if set(measured_keys) - set(audit_rows):
            raise ValueError(f"{start_name} measured keys are missing canonical states")
        selected = set(shortlist.selected_keys)
        for key in measured_keys:
            row = audit_rows[key]
            if row.get("canonical_state") is None or row.get("plan_v2_bridge") is None:
                raise ValueError(f"{start_name} plan is missing state or PlanV2: {key}")
            register(
                key,
                row["canonical_state"],
                row["plan_v2_bridge"],
                {
                    "role": "lns_diverse_top16" if key in selected else "lns_diverse_audit_top32",
                    "start": start_name,
                    "anchor_state_hash": anchor_hash,
                    "operator": row["operator"],
                    "moved_experts": row["moved_experts"],
                    "event_gain_pct": row["event_gain_pct"],
                    "robust_gain_pct": row["robust_gain_pct"],
                    "partial_order": row.get("partial_order"),
                },
            )

    if pooled:
        parent_shortlists = {
            parent_hash: LnsDiverseShortlist.from_dict(spec["shortlist"])
            for parent_hash, spec in pooled.items()
        }
        global_shortlist = select_lns_global_shortlist(
            parent_shortlists,
            shortlist_budget=shortlist_budget,
            audit_budget=audit_budget,
        )
        for parent_hash, spec in pooled.items():
            shortlist = parent_shortlists[parent_hash]
            selected = set(shortlist.selected_keys)
            measured = list(shortlist.selected_keys)
            if include_audit:
                measured = list(shortlist.audit_keys)
            if include_stratified:
                measured = list(dict.fromkeys([*measured, *spec.get("stratified_keys", ())]))
            for key in measured:
                row = pooled_rows.get(key)
                if not isinstance(row, dict) or row.get("canonical_state") is None:
                    raise ValueError(f"pooled parent {parent_hash} is missing canonical state for {key}")
                if key in selected:
                    role = "lns_diverse_top16"
                elif key in set(shortlist.audit_keys):
                    role = "lns_diverse_audit_top32"
                else:
                    role = "lns_stratified_outside_top32"
                register(
                    key,
                    row["canonical_state"],
                    row["plan_v2_bridge"],
                    {
                        "role": role,
                        "start": str(row.get("start", spec.get("starts", ["pooled"])[0])),
                        "anchor_state_hash": parent_hash,
                        "operator": row.get("operator"),
                        "moved_experts": row.get("moved_experts"),
                        "event_gain_pct": row.get("event_gain_pct"),
                        "robust_gain_pct": row.get("robust_gain_pct"),
                        "partial_order": row.get("partial_order"),
                    },
                )
    else:
        global_shortlist = select_lns_global_shortlist(
            per_start,
            shortlist_budget=shortlist_budget,
            audit_budget=audit_budget,
        )

    def register_control(plan: dict[str, object], *, role: str, start: str) -> None:
        state_hash = str(plan["state_hash"])
        register(
            state_hash,
            plan["canonical_state"],
            plan["plan_v2_bridge"],
            {
                "role": role,
                "start": start,
                "anchor_state_hash": _full_parent_hash(payload, runs) or state_hash,
                "operator": None,
                "moved_experts": [],
            },
        )

    if known_elite is not None:
        register_control(known_elite, role="known_hardware_elite", start="known_median_elite")
    if reference_plan is not None:
        register_control(
            reference_plan,
            role="previous_seed_selected",
            start="previous_seed_selected",
        )

    for record in records.values():
        record["roles"].sort()
        record["comparisons"].sort(
            key=lambda item: (str(item["role"]), str(item["start"]), str(item.get("operator", "")))
        )
    role_names = (
        "anchor",
        "lns_diverse_top16",
        "lns_diverse_audit_top32",
        "lns_stratified_outside_top32",
        "known_hardware_elite",
        "previous_seed_selected",
    )
    role_counts = {
        role: sum(role in record["roles"] for record in records.values())
        for role in role_names
    }
    method = {
        "starts": list(expected_starts),
        "shortlist_budget": shortlist_budget,
        "audit_budget": audit_budget,
        "policy": "relation_agnostic_categorical_farthest_first_v1",
        "policy_sha256": payload.get("method", {}).get("lns_shortlist_policy_sha256"),
        "deduplication": "canonical_state_hash",
        "pool_restarts_by_parent": bool(pooled),
        "include_audit": bool(include_audit),
        "include_stratified": bool(include_stratified and pooled),
        "known_elite_state_hash": None if known_elite is None else known_elite["state_hash"],
        "reference_state_hash": None if reference_plan is None else reference_plan["state_hash"],
    }
    return {
        "kind": "partial_order_hardware_frontier",
        "source": {
            "artifact": str(source_path),
            "sha256": _sha256(source_path),
        },
        "route": payload["route"],
        "shape": payload["shape"],
        "identity": payload["identity"],
        "method": method,
        "lns_diverse_shortlist": global_shortlist,
        "role_counts": role_counts,
        "unique_plans": len(plans),
        "plans": dict(sorted(plans.items())),
        "records": dict(sorted(records.items())),
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.lns_artifact.read_text(encoding="utf-8"))
    elite = None if args.known_elite_plan is None else _load_known_elite(args.known_elite_plan)
    reference = (
        None
        if args.reference_plan is None
        else _load_control_plan(args.reference_plan, label="reference")
    )
    result = build_lns_diverse_frontier(
        payload,
        source_path=args.lns_artifact,
        known_elite=elite,
        reference_plan=reference,
        include_audit=args.include_audit,
        include_stratified=args.include_stratified,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"unique_plans": result["unique_plans"], **result["role_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
