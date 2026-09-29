from __future__ import annotations

import json
from pathlib import Path

from fused_cpp.moe.cost_cache import MoeCostDiskCache


def _identity(machine: str = "test-machine") -> dict[str, object]:
    return {
        "model": {"schema": 7, "name": "phase_ecm_llc_domain_v4"},
        "machine": machine,
        "shape": [4096, 1024, 256],
        "widths": [1, 2, 4, 8, 16],
    }


def test_cost_disk_cache_round_trip_and_merge(tmp_path: Path) -> None:
    cache = MoeCostDiskCache(tmp_path, _identity())
    first = cache.store({(12, 1): 100.0, (24, 2): 200.0})

    assert first.status == "stored"
    assert cache.load().entries == {(12, 1): 100.0, (24, 2): 200.0}

    second = MoeCostDiskCache(tmp_path, _identity()).store({(48, 4): 300.0})
    assert second.status == "stored"
    assert second.entries == {(12, 1): 100.0, (24, 2): 200.0, (48, 4): 300.0}


def test_cost_disk_cache_identity_selects_a_different_file(tmp_path: Path) -> None:
    first = MoeCostDiskCache(tmp_path, _identity("first"))
    second = MoeCostDiskCache(tmp_path, _identity("second"))
    first.store({(12, 1): 100.0})

    assert first.path != second.path
    assert second.load().status == "miss"


def test_cost_disk_cache_rejects_corruption_and_repairs_on_store(tmp_path: Path) -> None:
    cache = MoeCostDiskCache(tmp_path, _identity())
    cache.directory.mkdir(parents=True, exist_ok=True)
    cache.path.write_text("not-json")

    failed = cache.load()
    assert failed.status == "error"
    assert failed.entries == {}

    repaired = cache.store({(12, 1): 100.0})
    assert repaired.status == "stored"
    assert cache.load().entries == {(12, 1): 100.0}


def test_cost_disk_cache_rejects_tampered_identity(tmp_path: Path) -> None:
    cache = MoeCostDiskCache(tmp_path, _identity())
    cache.store({(12, 1): 100.0})
    payload = json.loads(cache.path.read_text())
    payload["identity"]["machine"] = "tampered"
    cache.path.write_text(json.dumps(payload))

    loaded = cache.load()
    assert loaded.status == "error"
    assert "identity mismatch" in str(loaded.error)


def test_cost_disk_cache_write_failure_is_non_fatal(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied")
    cache = MoeCostDiskCache(blocked, _identity())

    result = cache.store({(12, 1): 100.0})

    assert result.status == "error"
    assert result.entries == {}
