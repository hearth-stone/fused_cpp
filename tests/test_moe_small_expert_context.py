from __future__ import annotations

from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_small_expert_context import (
    MODES,
    _build_bridge,
    _decomposition,
    _parse_target_calls,
    _run_scrubbed,
    _task_specs,
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


@pytest.mark.parametrize("mode", MODES)
def test_context_bridge_keeps_work_constant_and_changes_only_context(mode: str) -> None:
    bridge, routes = _build_bridge(
        _BridgeModel(),
        thread_cpu_ids=tuple(range(80)),
        mode=mode,
    )

    assert sorted(routes) == list(range(55))
    assert routes[0] == 1
    assert routes[1] == 68
    assert sorted(bridge["task_expert_ids"][:2]) == [0, 1]
    assert bridge["task_core_begins"][:2] == [64, 64]
    assert bridge["task_threads"][:2] == [1, 1]
    assert bridge["early_merge"] is False


def test_context_bridge_delays_only_disabled_background_families() -> None:
    bridges = {
        mode: _build_bridge(_BridgeModel(), thread_cpu_ids=tuple(range(80)), mode=mode)[0]
        for mode in MODES
    }

    assert bridges["full_head"]["task_expert_ids"][:2] == [0, 1]
    assert bridges["full_after_68"]["task_expert_ids"][:2] == [1, 0]
    assert _dependencies(bridges["full_head"], 2) == []
    assert _dependencies(bridges["full_head"], 10) == []
    assert _dependencies(bridges["wide_only_head"], 2) == []
    assert _dependencies(bridges["wide_only_head"], 10) == [1]
    assert _dependencies(bridges["narrow_only_head"], 2) == [1]
    assert _dependencies(bridges["narrow_only_head"], 10) == []
    assert _dependencies(bridges["isolated_head"], 2) == [1]
    assert _dependencies(bridges["isolated_head"], 10) == [1]


def test_context_bridge_accepts_a_target_route_sweep_without_changing_background() -> None:
    default = _task_specs()
    swept = _task_specs(12)
    bridge, routes = _build_bridge(
        _BridgeModel(),
        thread_cpu_ids=tuple(range(80)),
        mode="full_head",
        target_routes=12,
    )

    assert swept[0][1] == 12
    assert swept[1:] == default[1:]
    assert routes[0] == 12
    assert bridge["task_expert_ids"] == [task[0] for task in swept]


def test_context_route_sweep_rejects_nonpositive_target_routes() -> None:
    with pytest.raises(ValueError, match="target_routes must be positive"):
        _task_specs(0)


def test_target_trace_parser_excludes_non_target_and_merge_records(tmp_path: Path) -> None:
    trace = tmp_path / "target.log"
    trace.write_text(
        "\n".join(
            (
                "MOE_CALL call_id=0",
                "PHASE expert=0 stage=gather_pack_a start_ms=1.0 end_ms=1.2 ms=0.2",
                "PHASE expert=4 stage=w13_fused_silu_packc start_ms=1.0 end_ms=2.0 ms=1.0",
                "PHASE expert=0 stage=w13_fused_silu_packc start_ms=1.2 end_ms=2.0 ms=0.8",
                "PHASE expert=-1 stage=merge_ready_token start_ms=3.0 end_ms=4.0 ms=1.0",
                "MOE_CALL_END call_id=0",
                "MOE_CALL call_id=1",
                "PHASE expert=0 stage=gather_pack_a start_ms=2.0 end_ms=2.3 ms=0.3",
                "PHASE expert=0 stage=w13_fused_silu_packc start_ms=2.3 end_ms=3.2 ms=0.9",
                "MOE_CALL_END call_id=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    calls = _parse_target_calls(trace)

    assert calls["span"] == pytest.approx([1.0, 1.2])
    assert calls["gather_pack_a"] == pytest.approx([0.2, 0.3])
    assert calls["w13_fused_silu_packc"] == pytest.approx([0.8, 0.9])


def test_decomposition_separates_background_main_effects_and_interaction() -> None:
    spans = {
        "isolated_head": [1.0, 1.0],
        "wide_only_head": [2.0, 2.0],
        "narrow_only_head": [3.0, 3.0],
        "full_head": [5.5, 5.5],
        "full_after_68": [4.5, 4.5],
    }

    report = _decomposition(spans)

    assert report["wide_only_vs_isolated"]["delta"]["median_ms"] == pytest.approx(1.0)
    assert report["narrow_only_vs_isolated"]["delta"]["median_ms"] == pytest.approx(2.0)
    assert report["full_vs_isolated"]["delta"]["median_ms"] == pytest.approx(4.5)
    assert report["wide_narrow_interaction_ms"]["median_ms"] == pytest.approx(1.5)
    assert report["after_68_vs_full_head"]["delta"]["median_ms"] == pytest.approx(-1.0)


def test_run_scrubbed_uses_dedicated_copy_before_measured_copy() -> None:
    calls = []

    def run(mode: str, copy_index: int) -> tuple[str, int]:
        calls.append((mode, copy_index))
        return (mode, copy_index)

    result = _run_scrubbed(
        run,
        mode="narrow_only_head",
        measured_copy=2,
        scrub_copy=4,
        scrub_passes=2,
    )

    assert calls == [("full_head", 4), ("full_head", 4), ("narrow_only_head", 2)]
    assert result == ("narrow_only_head", 2)


def test_run_scrubbed_rejects_nonpositive_pass_count() -> None:
    with pytest.raises(ValueError, match="scrub_passes must be positive"):
        _run_scrubbed(
            lambda mode, copy_index: (mode, copy_index),
            mode="full_head",
            measured_copy=0,
            scrub_copy=1,
            scrub_passes=0,
        )
