from __future__ import annotations

from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_mixed_width_victim_pressure import (
    SMALL_PEER_ROUTES,
    VICTIM_ROUTES,
    WIDE_PEER_ROUTES,
    _build_bridge,
    _fit_width_overhead,
    _parse_victim_spans,
    _task_specs,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, width: int) -> tuple[int, int]:
        return routes % 3, width


class _Model:
    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


def _dependencies(bridge: dict[str, object]) -> list[list[int]]:
    offsets = bridge["task_dep_offsets"]
    dependencies = bridge["task_deps"]
    return [dependencies[offsets[index] : offsets[index + 1]] for index in range(len(offsets) - 1)]


def test_bridge_changes_only_peer_release_dependencies() -> None:
    common = {
        "thread_cpu_ids": tuple(range(80)),
        "peer_width": 16,
        "victim_core": 77,
    }
    concurrent = _build_bridge(_Model(), peer_mode="all_concurrent", **common)
    wide_only = _build_bridge(_Model(), peer_mode="wide_only", **common)
    delayed = _build_bridge(_Model(), peer_mode="all_delayed", **common)
    victim_count = len(VICTIM_ROUTES)
    concurrent_dependencies = _dependencies(concurrent)
    delayed_dependencies = _dependencies(delayed)

    assert concurrent["task_expert_ids"] == delayed["task_expert_ids"]
    assert concurrent["task_core_begins"] == delayed["task_core_begins"]
    assert concurrent["task_threads"] == delayed["task_threads"]
    assert concurrent["early_merge"] is False
    assert concurrent_dependencies[:victim_count] == delayed_dependencies[:victim_count]
    assert concurrent_dependencies[victim_count] == []
    assert delayed_dependencies[victim_count] == [victim_count - 1]
    for lane in range(len(WIDE_PEER_ROUTES)):
        first = victim_count + lane * len(WIDE_PEER_ROUTES[lane])
        second = first + 1
        assert concurrent_dependencies[second] == [first]
        assert delayed_dependencies[second] == [first]
    first_small = victim_count + sum(len(routes) for routes in WIDE_PEER_ROUTES)
    assert concurrent_dependencies[first_small] == []
    assert _dependencies(wide_only)[first_small] == [victim_count - 1]
    assert delayed_dependencies[first_small] == [victim_count - 1]
    assert concurrent_dependencies[first_small + 1] == [first_small]
    assert delayed_dependencies[first_small + 1] == [first_small]


def test_task_specs_keep_victim_and_peer_teams_disjoint() -> None:
    tasks = _task_specs(peer_width=16, victim_core=77)

    victim = tasks[: len(VICTIM_ROUTES)]
    wide_count = sum(len(routes) for routes in WIDE_PEER_ROUTES)
    wide_peers = tasks[len(VICTIM_ROUTES) : len(VICTIM_ROUTES) + wide_count]
    small_peers = tasks[len(VICTIM_ROUTES) + wide_count :]
    assert all(core == 77 and width == 1 for _, _, core, width in victim)
    assert {core for _, _, core, _ in wide_peers} == {0, 16, 32, 48}
    assert all(width == 16 for _, _, _, width in wide_peers)
    assert len(small_peers) == sum(len(routes) for routes in SMALL_PEER_ROUTES)
    assert all(core != 77 and width == 1 for _, _, core, width in small_peers)


def test_width_overhead_fit_recovers_linear_residual() -> None:
    class _IsoModel:
        @staticmethod
        def T_iso(routes: int, threads: int) -> float:
            assert threads == 1
            return routes * 1_000.0

    calls = []
    for jitter in (-1.0, 0.0, 1.0):
        call = {}
        cursor = 0.0
        for task, routes in enumerate(VICTIM_ROUTES):
            duration_ms = (100_000.0 + routes * 5_000.0 + routes * 1_000.0) / 1.0e6
            call[task] = (cursor, cursor + duration_ms + jitter * 1.0e-6)
            cursor += duration_ms + jitter * 1.0e-6
        calls.append(call)

    expert_fixed_ns, route_ns = _fit_width_overhead(_IsoModel(), calls)

    assert expert_fixed_ns == pytest.approx(100_000.0)
    assert route_ns == pytest.approx(5_000.0)


def test_trace_parser_returns_one_complete_victim_span(tmp_path: Path) -> None:
    trace = tmp_path / "trace.log"
    lines = [
        "MOE_CALL call_id=0 e2e_ms=1.0",
        "PHASE call_id=0 tid=77 group=-1 stage=worker start_ms=0 end_ms=20",
    ]
    for task in range(len(VICTIM_ROUTES)):
        lines.append(f"PHASE call_id=0 tid=77 group={task} stage=w2_direct_route start_ms={task}.0 end_ms={task + 1}.0")
    lines.append("MOE_CALL_END call_id=0")
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert _parse_victim_spans(trace) == [pytest.approx(float(len(VICTIM_ROUTES)))]


def test_trace_parser_rejects_incomplete_victim_call(tmp_path: Path) -> None:
    trace = tmp_path / "trace.log"
    trace.write_text(
        "MOE_CALL call_id=0 e2e_ms=1.0\n"
        "PHASE call_id=0 tid=77 group=0 stage=w2 start_ms=0 end_ms=1\n"
        "MOE_CALL_END call_id=0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="victim tasks"):
        _parse_victim_spans(trace)
