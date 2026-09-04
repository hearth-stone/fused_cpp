from __future__ import annotations

from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_gather_injection_overlap import (
    BACKGROUND_EXPERTS,
    LEGACY_MODES,
    MODES,
    OUTPUT_KIND,
    SCHEMA_VERSION,
    _build_bridge,
    _comparisons,
    _cpu_mapping,
    _parse_calls,
    _parse_cpu_list,
    _task_specs,
    enumerate_probe_modes,
    mode_name,
    parse_aggressor_counts,
    parse_mode,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, threads: int) -> tuple[int, int]:
        return (routes + threads, routes + 2 * threads)


class _BridgeModel:
    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


def _dependencies(bridge: dict[str, object], task: int) -> list[int]:
    offsets = bridge["task_dep_offsets"]
    dependencies = bridge["task_deps"]
    return dependencies[offsets[task] : offsets[task + 1]]


def _routes_and_bridge(mode: str) -> tuple[dict[str, object], dict[int, int]]:
    return _build_bridge(_BridgeModel(), thread_cpu_ids=tuple(range(80)), mode=mode)


@pytest.mark.parametrize("raw", ["", "0,1,1", "1,2,4", "0,16", "-1", "0,foo", "0,2,1"])
def test_aggressor_counts_reject_illegal_values(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_aggressor_counts(raw)


def test_aggressor_counts_accept_required_sweep() -> None:
    assert parse_aggressor_counts("0,1,2,4,8,15") == (0, 1, 2, 4, 8, 15)


def test_probe_modes_cover_count_placement_and_split_boundary() -> None:
    modes = enumerate_probe_modes((0, 1, 8, 15))

    assert modes[:2] == ("isolated_head", "isolated_after_1")
    assert "same_llc_head_n1" in modes
    assert "cross_llc_after_1_n1" in modes
    assert "split_head_n1" not in modes
    assert "split_head_n8" in modes
    assert "split_after_1_n15" in modes
    assert "local_head" not in modes
    assert parse_mode("split_head_n8") == ("split", "head", 8)
    with pytest.raises(ValueError, match="split placement"):
        mode_name("split", "head", 4)


@pytest.mark.parametrize("mode", tuple(dict.fromkeys((*MODES, *LEGACY_MODES))))
def test_overlap_bridge_keeps_work_and_target_lane_constant(mode: str) -> None:
    bridge, routes = _routes_and_bridge(mode)

    assert sorted(routes) == list(range(17))
    assert routes[0] == routes[1] == 1
    assert all(routes[expert] == 68 for expert in range(2, 17))
    assert sorted(bridge["task_expert_ids"][:2]) == [0, 1]
    assert bridge["task_core_begins"][:2] == [64, 64]
    assert _dependencies(bridge, 1) == [0]
    assert len(bridge["task_expert_ids"]) == 17
    assert len(_task_specs(mode)) == 17


def test_isolated_control_does_not_invent_locality_difference() -> None:
    isolated_head, routes_head = _routes_and_bridge("isolated_head")
    isolated_after, routes_after = _routes_and_bridge("isolated_after_1")

    assert routes_head == routes_after
    assert isolated_head["task_core_begins"][2:] == list(range(65, 80))
    assert isolated_after["task_core_begins"][2:] == list(range(65, 80))
    assert all(_dependencies(isolated_head, task) == [1] for task in range(2, 17))
    assert all(_dependencies(isolated_after, task) == [1] for task in range(2, 17))
    assert isolated_head["task_expert_ids"][:2] == [0, 1]
    assert isolated_after["task_expert_ids"][:2] == [1, 0]


def test_background_placement_and_delayed_unused_aggressors() -> None:
    same = _routes_and_bridge("same_llc_head_n2")[0]
    cross = _routes_and_bridge("cross_llc_head_n2")[0]
    split = _routes_and_bridge("split_head_n8")[0]
    split_15 = _routes_and_bridge("split_after_1_n15")[0]
    legacy_local = _routes_and_bridge("local_head")[0]
    legacy_remote = _routes_and_bridge("remote_head")[0]

    assert same["task_core_begins"][2:] == list(range(65, 80))
    assert cross["task_core_begins"][2:] == list(range(15))
    assert all(_dependencies(same, task) == [] for task in range(2, 4))
    assert all(_dependencies(same, task) == [1] for task in range(4, 17))
    assert all(_dependencies(cross, task) == [] for task in range(2, 4))
    assert split["task_core_begins"][2:10] == [0, 1, 2, 3, 65, 66, 67, 68]
    assert all(_dependencies(split, task) == [] for task in range(2, 10))
    assert all(_dependencies(split, task) == [1] for task in range(10, 17))
    assert split_15["task_core_begins"][2:] == list(range(7)) + list(range(65, 73))
    assert all(_dependencies(split_15, task) == [] for task in range(2, 17))
    assert legacy_local["task_core_begins"][2:] == list(range(65, 80))
    assert legacy_remote["task_core_begins"][2:] == list(range(15))
    assert all(_dependencies(legacy_local, task) == [] for task in range(2, 17))


def test_parse_calls_reports_target_envelope_and_peer_overlap(tmp_path: Path) -> None:
    trace = tmp_path / "trace.log"
    trace.write_text(
        "\n".join(
            (
                "MOE_CALL call_id=0",
                "PHASE expert=0 stage=gather_pack_a start_ms=1.0 end_ms=1.1 ms=0.1",
                "PHASE expert=2 stage=gather_pack_a start_ms=0.5 end_ms=1.2 ms=0.7",
                "PHASE expert=0 stage=w13_fused_silu_packc start_ms=1.1 end_ms=2.0 ms=0.9",
                "PHASE expert=2 stage=w13_fused_silu_packc start_ms=1.2 end_ms=3.0 ms=1.8",
                "PHASE expert=0 stage=w2_direct_route start_ms=2.0 end_ms=2.3 ms=0.3",
                "MOE_CALL_END call_id=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    calls = _parse_calls(trace)

    assert calls["target_span"] == pytest.approx([1.3])
    assert calls["target.w13_fused_silu_packc"] == pytest.approx([0.9])
    assert calls["peer_overlap_core_ms.gather_pack_a"] == pytest.approx([0.2])
    assert calls["peer_overlap_core_ms.w13_fused_silu_packc"] == pytest.approx([1.1])
    assert calls["peer_active_at_target_start.gather_pack_a"] == pytest.approx([1.0])
    assert calls["peer_overlap_experts"] == pytest.approx([1.0])


def test_comparisons_report_absolute_and_locality_contrast() -> None:
    spans = {
        "isolated_head": [1.0, 1.0],
        "isolated_after_1": [1.5, 1.5],
        "same_llc_head_n1": [2.0, 2.0],
        "cross_llc_head_n1": [1.2, 1.2],
        "same_llc_after_1_n1": [2.5, 2.5],
        "cross_llc_after_1_n1": [1.8, 1.8],
        "local_head": [3.0, 3.0],
        "remote_head": [2.0, 2.0],
        "local_after_1": [4.0, 4.0],
        "remote_after_1": [3.5, 3.5],
    }

    report = _comparisons(spans)

    assert report["same_llc_head_n1_vs_isolated"]["delta"]["median_ms"] == pytest.approx(1.0)
    assert report["cross_llc_head_n1_vs_isolated"]["delta"]["median_ms"] == pytest.approx(0.2)
    assert report["same_minus_cross_head_n1"]["delta"]["median_ms"] == pytest.approx(0.8)
    assert report["local_head_vs_isolated"]["delta"]["median_ms"] == pytest.approx(2.0)
    assert report["remote_head_vs_isolated"]["delta"]["median_ms"] == pytest.approx(1.0)
    assert report["local_vs_remote_after_1"]["delta"]["median_ms"] == pytest.approx(0.5)


def _write_llc_cache(cpu_dir: Path, *, cache_id: str | None, shared: str) -> None:
    cache = cpu_dir / "cache" / "index3"
    cache.mkdir(parents=True)
    (cache / "level").write_text("3\n", encoding="utf-8")
    (cache / "type").write_text("Unified\n", encoding="utf-8")
    (cache / "shared_cpu_list").write_text(shared + "\n", encoding="utf-8")
    if cache_id is not None:
        (cache / "id").write_text(cache_id + "\n", encoding="utf-8")


def test_cpu_mapping_reads_numa_and_explicit_llc_id(tmp_path: Path) -> None:
    cpu_dir = tmp_path / "cpu304"
    (cpu_dir / "node3").mkdir(parents=True)
    _write_llc_cache(cpu_dir, cache_id="7", shared="280-319")

    mapping = _cpu_mapping((304,), sysfs_root=tmp_path)

    assert mapping == [
        {
            "logical_index": 0,
            "os_cpu": 304,
            "physical_cpu": 304,
            "numa_node": 3,
            "llc_domain": 7,
            "llc_shared_cpus": "280-319",
        }
    ]


def test_cpu_mapping_numbers_shared_llc_lists_when_id_is_missing(tmp_path: Path) -> None:
    for cpu, shared in ((240, "240-279"), (304, "280-319")):
        cpu_dir = tmp_path / f"cpu{cpu}"
        (cpu_dir / "node3").mkdir(parents=True)
        _write_llc_cache(cpu_dir, cache_id=None, shared=shared)

    mapping = _cpu_mapping((240, 304), sysfs_root=tmp_path)

    assert _parse_cpu_list("240-279,304")[-1] == 304
    assert [row["llc_domain"] for row in mapping] == [0, 1]
    assert [row["llc_shared_cpus"] for row in mapping] == ["240-279", "280-319"]


def test_output_schema_constants_are_locked() -> None:
    assert OUTPUT_KIND == "moe_gather_absolute_pressure_probe"
    assert SCHEMA_VERSION == 1
    assert BACKGROUND_EXPERTS == 15
    assert len(MODES) == 26
