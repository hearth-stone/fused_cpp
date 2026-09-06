from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cpu_moe_schedule_optimization" / "planners"))

from executable_plan_neighborhood import (  # noqa: E402
    TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
    TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
    ExecutablePlanNeighbor,
    enumerate_template_lns_neighbors,
)
from executable_plan_state import (  # noqa: E402
    ExecutableExpertTask,
    ExecutableLane,
    ExecutableLlcDomain,
    ExecutablePlanState,
)
from optimizations.fused_moe_sve.benchmarks.build_lns_reference_pool import (  # noqa: E402
    TARGET_OPERATOR,
    account_reference_roles,
    member_provenance,
    pooled_selected_keys,
    require_registered_artifact,
    sha256_sorted_hashes,
    unique_then_operator_filter,
)
from optimizations.fused_moe_sve.benchmarks.lns_diverse_shortlist import state_from_canonical_payload  # noqa: E402


def _state() -> ExecutablePlanState:
    task = ExecutableExpertTask
    return ExecutablePlanState(
        num_threads=12,
        thread_cpu_ids=tuple(range(100, 112)),
        lanes=(
            ExecutableLane(0, 4, (task(0, 80), task(1, 32), task(2, 8))),
            ExecutableLane(4, 4, (task(3, 64), task(4, 24), task(5, 12), task(6, 4))),
            ExecutableLane(8, 2, (task(7, 20), task(8, 6))),
            ExecutableLane(10, 2, (task(9, 18), task(10, 5))),
        ),
        llc_domains=(
            ExecutableLlcDomain("left", 0, 6),
            ExecutableLlcDomain("right", 6, 6),
        ),
        early_merge=True,
    )


def _windows(routes: int, threads: int) -> tuple[int, int]:
    return (routes + threads, routes + 2 * threads)


def _isolated_cost(routes: int, threads: int) -> float:
    return float(routes) / float(threads)


def test_unique_then_operator_filter_uses_global_first_owner() -> None:
    state = _state()
    first = ExecutablePlanNeighbor(TARGET_OPERATOR, (0, 1, 2, 3), state)
    later = ExecutablePlanNeighbor(
        "critical_window_template_repartition_cross_domain_d8_b32",
        (0, 1, 2, 3),
        state,
    )
    owned_d8 = unique_then_operator_filter(
        [first, later],
        operator="critical_window_template_repartition_cross_domain_d8_b32",
        baseline_hash="0" * 64,
    )
    owned_d4 = unique_then_operator_filter(
        [first, later],
        operator=TARGET_OPERATOR,
        baseline_hash="0" * 64,
    )
    assert owned_d8 == []
    assert len(owned_d4) == 1
    assert owned_d4[0].operator == TARGET_OPERATOR


def test_member_round_trips_canonical_hash_and_strict_bridge() -> None:
    state = _state()
    neighbors = list(
        enumerate_template_lns_neighbors(
            state,
            allowed_widths=(1, 2, 4, 6),
            isolated_cost=_isolated_cost,
            window_selector=_windows,
            critical_expert_ids=(3, 9),
            destroy_sizes=TEMPLATE_LNS_DEFAULT_DESTROY_SIZES,
            repair_beam_widths=TEMPLATE_LNS_DEFAULT_REPAIR_BEAM_WIDTHS,
            templates_per_block=3,
        )
    )
    owned = unique_then_operator_filter(
        neighbors,
        operator=TARGET_OPERATOR,
        baseline_hash=state.canonical_hash(),
    )
    assert owned
    row = member_provenance(owned[0])
    restored = state_from_canonical_payload(row["canonical_state"])
    assert restored.canonical_hash() == row["state_hash"]
    assert row["plan_v2_bridge"]["execution_mode"] == "strict"
    assert row["operator"] == TARGET_OPERATOR


def test_reference_roles_are_not_proposals_and_pooling_uses_global_keys() -> None:
    roles = account_reference_roles(
        ["pool_a"],
        controls={"full": "full_hash"},
        elites={"known_hardware_elite": "pool_a", "previous_seed_selected": "elite_b"},
        pooled_selected=["pool_c"],
        per_start_selected=["pool_a"],
        pooling_enabled=True,
    )
    assert roles["elites_in_operator_pool"]["known_hardware_elite"] is True
    assert roles["elites_in_operator_pool"]["previous_seed_selected"] is False
    assert roles["records"]["pool_a"]["is_proposal"] is False
    assert "known_hardware_elite" in roles["records"]["pool_a"]["roles"]
    assert roles["final_selected_keys"] == ["pool_c"]
    assert "per_start_selected_intermediate" in roles["records"]["pool_a"]["roles"]
    model = {
        "parent_pooled_shortlists": {
            "parent": {"shortlist": {"selected_keys": ["pool_c", "pool_d"]}}
        },
        "runs": {
            "lns_00_r00": {
                "iterations": [{"lns_diverse_shortlist": {"selected_keys": ["pool_a"]}}]
            }
        },
    }
    assert pooled_selected_keys(model, pooling_enabled=True) == ["pool_c", "pool_d"]
    assert pooled_selected_keys(model, pooling_enabled=False) == ["pool_a"]


def test_pool_identity_digest_is_sha256_of_sorted_unique_hashes() -> None:
    hashes = ["c" * 64, "a" * 64, "b" * 64, "a" * 64]
    unique = sorted(set(hashes))
    expected = hashlib.sha256(json.dumps(unique, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert sha256_sorted_hashes(hashes) == expected
    assert sha256_sorted_hashes(unique) == expected
    assert sha256_sorted_hashes(list(reversed(unique))) == expected


def test_missing_raw_audit_recovery_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="unavailable task3 audit artifact"):
        require_registered_artifact(None, "abc123", label="task3 audit")
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="unavailable task4 replay artifact"):
        require_registered_artifact(missing, "def456", label="task4 replay")
    path = tmp_path / "wrong.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        require_registered_artifact(path, "0" * 64, label="task2a replay")
