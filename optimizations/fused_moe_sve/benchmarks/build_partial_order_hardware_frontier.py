#!/usr/bin/env python3
"""Build a deduplicated hardware frontier from a self-contained VND replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vnd-artifact", type=Path, required=True)
    parser.add_argument("--carry-frontier", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SELECTED_ROLE_BY_RELATION = {
    "candidate_better": "candidate_better_frontier",
    "incomparable": "incomparable_frontier",
}
DIAGNOSTIC_SELECTED_ROLE_BY_RELATION = {
    "candidate_better": "model_better_frontier",
    "incomparable": "incomparable_frontier",
}


def build_frontier(
    payload: dict[str, object],
    *,
    source_path: Path,
    carry: dict[str, object] | None = None,
) -> dict[str, object]:
    allowed_kinds = {
        "executable_partial_order_vnd_model_replay",
        "executable_partial_order_beam_layer_model_replay",
        "executable_partial_order_template_lns_model_replay",
    }
    if payload.get("kind") not in allowed_kinds:
        raise ValueError("input must be an executable partial-order search replay")
    if payload.get("lns_diverse_shortlist") or any(
        (run.get("iterations") or [{}])[0].get("lns_diverse_shortlist")
        for run in (payload.get("runs") or {}).values()
        if isinstance(run, dict)
    ):
        raise ValueError(
            "template-LNS diverse shortlists must use build_lns_diverse_hardware_frontier.py"
        )
    runs = payload.get("runs")
    expected_starts = tuple(payload.get("method", {}).get("start_names", ()))
    if not isinstance(runs, dict) or not expected_starts or set(runs) != set(expected_starts):
        raise ValueError("search replay runs do not match method.start_names")

    plans: dict[str, dict[str, object]] = {}
    records: dict[str, dict[str, object]] = {}

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

    if carry is not None:
        if carry.get("kind") != "partial_order_hardware_frontier":
            raise ValueError("carry input must be a partial-order hardware frontier")
        if carry["route"]["sha256"] != payload["route"]["sha256"]:
            raise ValueError("carry frontier route does not match search replay")
        for state_hash, record in carry["records"].items():
            if not {"anchor", "carried_anchor"}.intersection(record["roles"]):
                continue
            plan = carry["plans"][state_hash]
            register(
                state_hash,
                plan["canonical_state"],
                plan["plan_v2_bridge"],
                {
                    "role": "carried_anchor",
                    "start": "depth_1",
                    "anchor_state_hash": state_hash,
                },
            )

    forced_greedy_hash = None
    shortlist_budget = int(payload["method"]["partial_order_shortlist_budget"])
    for start_name, run in runs.items():
        iterations = run.get("iterations", [])
        if len(iterations) != 1:
            raise ValueError(f"{start_name} must stop after the audited first iteration")
        iteration = iterations[0]
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
        selected = iteration.get("selected_frontier", [])
        if len(selected) != shortlist_budget:
            raise ValueError(
                f"{start_name} must contribute exactly top-{shortlist_budget} candidates"
            )
        for candidate in selected:
            state_hash = str(candidate["state_hash"])
            relation = str(candidate["partial_order"]["relation"])
            role_by_relation = (
                DIAGNOSTIC_SELECTED_ROLE_BY_RELATION
                if iteration["partial_order"].get("automatic_acceptance_enabled") is False
                else SELECTED_ROLE_BY_RELATION
            )
            try:
                role = role_by_relation[relation]
            except KeyError as error:
                raise ValueError(
                    f"{start_name} selected candidate has invalid relation: {relation}"
                ) from error
            register(
                state_hash,
                candidate["canonical_state"],
                candidate["plan_v2_bridge"],
                {
                    "role": role,
                    "start": start_name,
                    "anchor_state_hash": anchor_hash,
                    "operator": candidate["operator"],
                    "moved_experts": candidate["moved_experts"],
                    "event_gain_pct": candidate["event_gain_pct"],
                    "robust_gain_pct": candidate["robust_gain_pct"],
                    "partial_order": candidate["partial_order"],
                },
            )
        for sentinel in iteration.get("worse_sentinels", []):
            state_hash = str(sentinel["state_hash"])
            sentinel_role = (
                "model_worse_spectrum"
                if iteration["partial_order"].get("dominance_pruning_enabled") is False
                else "candidate_worse_sentinel"
            )
            register(
                state_hash,
                sentinel["canonical_state"],
                sentinel["plan_v2_bridge"],
                {
                    "role": sentinel_role,
                    "start": start_name,
                    "anchor_state_hash": anchor_hash,
                    "operator": sentinel["operator"],
                    "moved_experts": sentinel["moved_experts"],
                    "event_gain_pct": sentinel["event_gain_pct"],
                    "robust_gain_pct": sentinel["robust_gain_pct"],
                    "partial_order": sentinel["partial_order"],
                },
            )
        if payload.get("kind") == "executable_partial_order_vnd_model_replay" and start_name == "greedy":
            evidence = iteration["partial_order"]["evidence"]
            forced_greedy_hash = max(
                evidence,
                key=lambda key: (
                    float(evidence[key]["lower_gain_pct"]),
                    float(evidence[key]["predicted_gain_pct"]),
                    key,
                ),
            )
            if forced_greedy_hash not in {str(item["state_hash"]) for item in selected}:
                raise ValueError("the best greedy lower-bound candidate is missing from top-16")

    for record in records.values():
        record["roles"].sort()
        record["comparisons"].sort(
            key=lambda item: (str(item["role"]), str(item["start"]), str(item.get("operator", "")))
        )
    role_counts = {
        role: sum(role in record["roles"] for record in records.values())
        for role in (
            "anchor",
            "carried_anchor",
            "candidate_better_frontier",
            "model_better_frontier",
            "incomparable_frontier",
            "candidate_worse_sentinel",
            "model_worse_spectrum",
        )
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
        "method": {
            "starts": list(expected_starts),
            "per_start_incomparable": shortlist_budget,
            "worse_sentinels_per_start_max": 2,
            "deduplication": "canonical_state_hash",
            "forced_greedy_best_lower_bound_state_hash": forced_greedy_hash,
            "carry_frontier_sha256": (
                carry.get("source", {}).get("sha256") if carry is not None else None
            ),
        },
        "role_counts": role_counts,
        "unique_plans": len(plans),
        "plans": dict(sorted(plans.items())),
        "records": dict(sorted(records.items())),
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.vnd_artifact.read_text(encoding="utf-8"))
    carry = (
        json.loads(args.carry_frontier.read_text(encoding="utf-8"))
        if args.carry_frontier is not None
        else None
    )
    result = build_frontier(payload, source_path=args.vnd_artifact, carry=carry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"unique_plans": result["unique_plans"], **result["role_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
