#!/usr/bin/env python3
"""Replay LNS farthest-first ranking from saved frozen feature pools.

This does not regenerate the neighborhood or run a hardware session. It ranks
saved per-start and parent-pooled ``candidate_features`` with the current fill
and the naive reference fill, then reports wall time, peak RSS, and key identity.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "optimizations" / "fused_moe_sve" / "benchmarks")]

from lns_diverse_shortlist import (  # noqa: E402
    RANKING_FILL_NAME,
    RANKING_FILL_REFERENCE_NAME,
    LnsCandidateFeature,
    rank_lns_diverse_candidates,
    recover_run_features,
    select_lns_diverse_shortlist,
    select_lns_global_shortlist,
    select_lns_parent_pooled_shortlists,
    state_from_canonical_payload,
    stratified_keys_outside_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--rss-only-fill")
    return parser.parse_args()


def _peak_rss_kb() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(round(usage / 1024.0))
    return int(usage)


def _features_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[LnsCandidateFeature]:
    return [LnsCandidateFeature.from_dict(row) for row in rows]


def recovered_per_start_features(
    model: Mapping[str, Any],
) -> tuple[dict[str, list[LnsCandidateFeature]], dict[str, dict[str, Any]]]:
    features: dict[str, list[LnsCandidateFeature]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for start, run in model["runs"].items():
        recovered, record = recover_run_features(run, iteration_index=0)
        features[str(start)] = recovered
        provenance[str(start)] = record
    return features, provenance


def _plan_library(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    library: dict[str, dict[str, Any]] = {}
    for run in (model.get("runs") or {}).values():
        iterations = run.get("iterations") or []
        if not iterations:
            continue
        iteration = iterations[0]
        for row in (*(iteration.get("selected_frontier") or []), *(iteration.get("audit_frontier") or [])):
            if not isinstance(row, Mapping):
                continue
            library.setdefault(str(row["state_hash"]), dict(row))
    for key, row in (model.get("pooled_frontier_rows") or {}).items():
        if isinstance(row, Mapping):
            library.setdefault(str(key), dict(row))
    return library


def require_plan_row(library: Mapping[str, Mapping[str, Any]], state_hash: str, *, role: str) -> dict[str, Any]:
    if state_hash not in library:
        raise KeyError(f"unavailable {role} state: {state_hash}")
    row = dict(library[state_hash])
    canonical = row.get("canonical_state")
    bridge = row.get("plan_v2_bridge")
    if not isinstance(canonical, Mapping):
        raise KeyError(f"unavailable {role} state: {state_hash} has no canonical_state")
    restored = state_from_canonical_payload(canonical)
    if restored.canonical_hash() != state_hash:
        raise ValueError(f"canonical hash mismatch for {role} {state_hash}")
    if restored.to_bridge() != bridge:
        raise ValueError(f"plan_v2_bridge mismatch for {role} {state_hash}")
    return row


def _recompute_outside_audit(
    per_start: Mapping[str, Sequence[LnsCandidateFeature]],
    pooled: Mapping[str, Any],
    parent_of_start: Mapping[str, str],
    *,
    sample_size: int,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    starts_by_parent: dict[str, list[str]] = {}
    for start, parent in parent_of_start.items():
        starts_by_parent.setdefault(parent, []).append(start)
    all_features = [feature for rows in per_start.values() for feature in rows]
    audit_union = {key for shortlist in pooled.values() for key in shortlist.audit_keys}
    ranked_outside: list[str] = []
    seen: set[str] = set()
    for shortlist in pooled.values():
        for key in shortlist.ranked_keys:
            if key in audit_union or key in seen:
                continue
            ranked_outside.append(key)
            seen.add(key)
    sampled = stratified_keys_outside_audit(
        all_features,
        ranked_outside,
        (),
        sample_size=sample_size,
        seed=seed,
    )
    hashes_by_parent = {
        parent: {feature.state_hash for start in starts for feature in per_start[start]}
        for parent, starts in starts_by_parent.items()
    }
    return {
        parent: tuple(key for key in sampled if key in hashes_by_parent.get(parent, set()))
        for parent in pooled
    }


def _rank_keys(features: Sequence[LnsCandidateFeature], fill: str) -> list[str]:
    return [item.state_hash for item in rank_lns_diverse_candidates(features, fill=fill)]


def replay_frozen_ranking(
    model: Mapping[str, Any],
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    per_start, recovery_provenance = recovered_per_start_features(model)
    parent_of_start = {
        start: str(item["state_hash"]) for start, item in (model.get("parents") or {}).items()
    }
    method = model.get("method") or {}
    shortlist_budget = int(method.get("shortlist_budget", 16))
    audit_budget = int(method.get("audit_budget", 32))
    stratified_n = int(method.get("stratified_outside_audit", 0))
    stratified_seed = int(method.get("stratified_seed", 0))
    stored_per_start = {
        start: list(run["iterations"][0]["lns_diverse_shortlist"]["ranked_keys"])
        for start, run in model["runs"].items()
        if run.get("iterations") and run["iterations"][0].get("lns_diverse_shortlist")
    }

    def one_pass(fill: str) -> dict[str, Any]:
        rss_before = _peak_rss_kb()
        begin = time.perf_counter_ns()
        per_start_keys = {start: _rank_keys(rows, fill) for start, rows in per_start.items()}
        pooled_keys: dict[str, list[str]] = {}
        if parent_of_start:
            pooled = select_lns_parent_pooled_shortlists(
                per_start,
                parent_of_start,
                shortlist_budget=shortlist_budget,
                audit_budget=audit_budget,
                fill=fill,
            )
            pooled_keys = {
                parent: list(shortlist.ranked_keys) for parent, shortlist in pooled.items()
            }
        wall_s = (time.perf_counter_ns() - begin) / 1.0e9
        return {
            "fill": fill,
            "parent_pooled_ranked_keys": pooled_keys,
            "per_start_ranked_keys": per_start_keys,
            "ru_maxrss_kb": max(rss_before, _peak_rss_kb()),
            "wall_s": wall_s,
        }

    reference_runs = [one_pass(RANKING_FILL_REFERENCE_NAME) for _ in range(repeats)]
    incremental_runs = [one_pass(RANKING_FILL_NAME) for _ in range(repeats)]

    def _agree(runs: list[dict[str, Any]]) -> bool:
        first = runs[0]
        return all(
            run["per_start_ranked_keys"] == first["per_start_ranked_keys"]
            and run["parent_pooled_ranked_keys"] == first["parent_pooled_ranked_keys"]
            for run in runs[1:]
        )

    matches_original = all(
        incremental_runs[0]["per_start_ranked_keys"].get(start) == keys
        for start, keys in stored_per_start.items()
    )
    fills_equal = (
        reference_runs[0]["per_start_ranked_keys"] == incremental_runs[0]["per_start_ranked_keys"]
        and reference_runs[0]["parent_pooled_ranked_keys"]
        == incremental_runs[0]["parent_pooled_ranked_keys"]
    )
    repeat_ok = _agree(reference_runs) and _agree(incremental_runs)
    outside_keys: dict[str, list[str]] = {}
    if parent_of_start and stratified_n:
        pooled = select_lns_parent_pooled_shortlists(
            per_start,
            parent_of_start,
            shortlist_budget=shortlist_budget,
            audit_budget=audit_budget,
        )
        by_parent = _recompute_outside_audit(
            per_start,
            pooled,
            parent_of_start,
            sample_size=stratified_n,
            seed=stratified_seed,
        )
        outside_keys = {parent: list(keys) for parent, keys in by_parent.items()}

    return {
        "after": {
            "fill": RANKING_FILL_NAME,
            "mean_wall_s": sum(run["wall_s"] for run in incremental_runs) / repeats,
            "repeat_agreement": _agree(incremental_runs),
            "ru_maxrss_kb": max(run["ru_maxrss_kb"] for run in incremental_runs),
            "runs": incremental_runs,
        },
        "before": {
            "fill": RANKING_FILL_REFERENCE_NAME,
            "mean_wall_s": sum(run["wall_s"] for run in reference_runs) / repeats,
            "repeat_agreement": _agree(reference_runs),
            "ru_maxrss_kb": max(run["ru_maxrss_kb"] for run in reference_runs),
            "runs": reference_runs,
        },
        "equal": bool(fills_equal and repeat_ok and matches_original),
        "fills_equal": fills_equal,
        "matches_original_ranked_keys": matches_original,
        "outside_audit_keys": outside_keys,
        "recovery_provenance": recovery_provenance,
        "repeat_agreement": repeat_ok,
        "repeats": repeats,
    }


def replayed_model_payload(model: Mapping[str, Any], fill: str = RANKING_FILL_NAME) -> dict[str, Any]:
    """Copy a frozen model and replace ranking artifacts from recovered inputs."""

    payload = json.loads(json.dumps(model))
    per_start, recovery_provenance = recovered_per_start_features(payload)
    method = payload.get("method") or {}
    shortlist_budget = int(method.get("shortlist_budget", 16))
    audit_budget = int(method.get("audit_budget", 32))
    stratified_n = int(method.get("stratified_outside_audit", 0))
    stratified_seed = int(method.get("stratified_seed", 0))
    parent_of_start = {
        start: str(item["state_hash"]) for start, item in (payload.get("parents") or {}).items()
    }
    library = _plan_library(payload)
    per_start_shortlists = {}
    for start, rows in per_start.items():
        shortlist = select_lns_diverse_shortlist(
            rows,
            shortlist_budget=shortlist_budget,
            audit_budget=audit_budget,
            fill=fill,
        )
        per_start_shortlists[start] = shortlist
        iteration = payload["runs"][start]["iterations"][0]
        iteration["lns_diverse_shortlist"] = shortlist.to_dict()
        iteration["candidate_features"] = [item.to_dict() for item in rows]
        iteration["anchor_recovery"] = recovery_provenance[start]
        iteration["selected_frontier"] = [
            require_plan_row(library, key, role="selected") for key in shortlist.selected_keys
        ]
        iteration["audit_frontier"] = [
            require_plan_row(library, key, role="audit") for key in shortlist.audit_keys
        ]
    if parent_of_start:
        pooled = select_lns_parent_pooled_shortlists(
            per_start,
            parent_of_start,
            shortlist_budget=shortlist_budget,
            audit_budget=audit_budget,
            fill=fill,
        )
        stratified_by_parent = _recompute_outside_audit(
            per_start,
            pooled,
            parent_of_start,
            sample_size=stratified_n,
            seed=stratified_seed,
        )
        existing = payload.get("parent_pooled_shortlists") or {}
        payload["parent_pooled_shortlists"] = {
            parent: {
                "starts": (existing.get(parent) or {}).get("starts")
                or sorted(start for start, item in parent_of_start.items() if item == parent),
                "shortlist": shortlist.to_dict(),
                "stratified_keys": list(stratified_by_parent.get(parent, ())),
            }
            for parent, shortlist in pooled.items()
        }
        needed = {
            *{key for shortlist in pooled.values() for key in shortlist.audit_keys},
            *{key for keys in stratified_by_parent.values() for key in keys},
        }
        payload["pooled_frontier_rows"] = {
            key: require_plan_row(library, key, role="pooled") for key in sorted(needed)
        }
        source = pooled
    else:
        source = per_start_shortlists
    payload["lns_diverse_shortlist"] = select_lns_global_shortlist(
        source,
        shortlist_budget=shortlist_budget,
        audit_budget=audit_budget,
    )
    method["lns_ranking_fill"] = fill
    method["anchor_recovery"] = {
        "fields": ["anchor_domain_assignment_signature"],
        "source": "iteration_canonical_state",
    }
    payload["method"] = method
    payload["recovery_provenance"] = recovery_provenance
    return payload


def main() -> int:
    args = parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    if args.rss_only_fill:
        per_start, _provenance = recovered_per_start_features(model)
        for rows in per_start.values():
            _rank_keys(rows, args.rss_only_fill)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fill": args.rss_only_fill, "ru_maxrss_kb": _peak_rss_kb()}
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        return 0
    result = replay_frozen_ranking(model, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.candidate_model is not None:
        candidate = replayed_model_payload(model, fill=RANKING_FILL_NAME)
        args.candidate_model.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_model.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"equal": result["equal"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
