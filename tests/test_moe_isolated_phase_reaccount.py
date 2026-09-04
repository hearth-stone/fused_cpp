from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from optimizations.fused_moe_sve.benchmarks.bench_isolated_phase_reaccount import (
    EXPERT_COUNT,
    FORBIDDEN_HOLDOUT_ROUTES,
    SCRUB_ROUTES,
    _measured_bridge,
    _positive_int_list,
    _parse_stage_envelopes,
    _scrub_bridge,
    _validate_fit_domain,
)


class _WindowPolicy:
    @staticmethod
    def select(routes: int, threads: int) -> tuple[int, int]:
        return (routes + threads, routes + 2 * threads)


class _BridgeModel:
    @staticmethod
    def shadow_stage_window_policy() -> _WindowPolicy:
        return _WindowPolicy()


def test_positive_int_list_normalizes_and_rejects_invalid_values() -> None:
    assert _positive_int_list("8,3,8") == (3, 8)
    with pytest.raises(argparse.ArgumentTypeError, match="positive integers"):
        _positive_int_list("0,3")


def test_fit_domain_rejects_locked_holdout_routes_and_wide_teams() -> None:
    with pytest.raises(ValueError, match="overlap locked holdout"):
        _validate_fit_domain((3, min(FORBIDDEN_HOLDOUT_ROUTES)), (1,), 80)
    with pytest.raises(ValueError, match="available threads"):
        _validate_fit_domain((3,), (1, 81), 80)


def test_measured_bridge_contains_one_dependency_free_target() -> None:
    bridge = _measured_bridge(
        _BridgeModel(),
        thread_cpu_ids=tuple(range(80)),
        routes=24,
        width=8,
    )

    assert bridge["task_expert_ids"] == [0]
    assert bridge["task_core_begins"] == [64]
    assert bridge["task_threads"] == [8]
    assert bridge["task_deps"] == []
    assert bridge["task_dep_offsets"] == [0, 0]
    assert bridge["task_w13_window_tiles"] == [32]
    assert bridge["task_w2_window_tiles"] == [40]


@pytest.mark.parametrize(
    ("width", "core_begin"),
    ((32, 48), (40, 40), (80, 0)),
)
def test_wide_measured_bridge_uses_one_llc_or_the_full_rank(width: int, core_begin: int) -> None:
    bridge = _measured_bridge(
        _BridgeModel(),
        thread_cpu_ids=tuple(range(80)),
        routes=24,
        width=width,
    )

    assert bridge["task_core_begins"] == [core_begin]
    assert bridge["task_threads"] == [width]


def test_scrub_bridge_touches_every_expert_on_disjoint_1t_lanes() -> None:
    bridge = _scrub_bridge(_BridgeModel(), thread_cpu_ids=tuple(range(80)))

    assert bridge["task_expert_ids"] == list(range(EXPERT_COUNT))
    assert bridge["task_core_begins"] == list(range(EXPERT_COUNT))
    assert bridge["task_threads"] == [1] * EXPERT_COUNT
    assert bridge["task_w13_window_tiles"] == [SCRUB_ROUTES + 1] * EXPERT_COUNT
    assert bridge["task_deps"] == []


def test_stage_parser_uses_multithread_envelopes_instead_of_core_ms(tmp_path: Path) -> None:
    trace = tmp_path / "trace.log"
    trace.write_text(
        "\n".join(
            (
                "MOE_CALL call_id=0",
                "PHASE expert=0 stage=gather_pack_a start_ms=1.0 end_ms=1.2 ms=0.2",
                "PHASE expert=0 stage=gather_pack_a start_ms=1.1 end_ms=1.3 ms=0.2",
                "PHASE expert=0 stage=w13_fused_silu_packc start_ms=1.3 end_ms=2.0 ms=0.7",
                "PHASE expert=0 stage=w13_fused_silu_packc start_ms=1.3 end_ms=2.1 ms=0.8",
                "PHASE expert=4 stage=w13_fused_silu_packc start_ms=1.0 end_ms=3.0 ms=2.0",
                "MOE_CALL_END call_id=0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    calls = _parse_stage_envelopes(trace)

    assert calls["gather_pack_a"] == pytest.approx([0.3])
    assert calls["w13_fused_silu_packc"] == pytest.approx([0.8])
    assert calls["span"] == pytest.approx([1.1])
