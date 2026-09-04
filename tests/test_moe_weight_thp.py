from __future__ import annotations

from optimizations.fused_moe_sve.benchmarks.bench_weight_thp import (
    MODES,
    POLICIES,
    cell_name,
    decide_weight_thp,
    pages_verified,
    parse_cell,
    parse_smaps_vma,
    round_up,
)


SAMPLE_SMAPS = """\
7f0000000000-7f00000d8000 rw-p 00000000 00:00 0
Size:               864 kB
KernelPageSize:        4 kB
AnonHugePages:         0 kB
7f1000000000-7f1000e00000 rw-p 00000000 00:00 0
Size:            143360 kB
KernelPageSize:        4 kB
AnonHugePages:     143360 kB
"""


def _delta(median_ms: float) -> dict[str, dict[str, float]]:
    return {"delta": {"median_ms": median_ms}}


def _calls(overlap: float) -> dict[str, list[float]]:
    return {"peer_overlap_experts": [overlap]}


def _reports(*, small_huge: int, thp_huge: int, request: int = 216 << 20) -> dict[str, dict[str, object]]:
    return {
        "small": {
            "small_clean": small_huge <= (2 << 20),
            "thp_latched": small_huge >= 0.8 * request,
            "anon_huge_bytes": small_huge,
            "request_bytes": request,
        },
        "thp": {
            "small_clean": thp_huge <= (2 << 20),
            "thp_latched": thp_huge >= 0.8 * request,
            "anon_huge_bytes": thp_huge,
            "request_bytes": request,
        },
    }


def _live_calls() -> dict[str, dict[str, list[float]]]:
    return {
        "small:wide16_same_head": _calls(1.0),
        "thp:wide16_same_head": _calls(1.0),
        "small:many16_1t_same_head": _calls(16.0),
        "thp:many16_1t_same_head": _calls(16.0),
    }


def _comparisons(many_small: float, many_thp: float) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "small:wide16_same_head_vs_isolated": _delta(0.04),
        "thp:wide16_same_head_vs_isolated": _delta(0.04),
        "small:many16_1t_same_head_vs_isolated": _delta(many_small),
        "thp:many16_1t_same_head_vs_isolated": _delta(many_thp),
        "small:many16_1t_cross_head_vs_isolated": _delta(0.03),
        "thp:many16_1t_cross_head_vs_isolated": _delta(0.03),
    }


def test_smaps_parser_reads_anon_huge_pages() -> None:
    small = parse_smaps_vma(SAMPLE_SMAPS, 0x7F0000001000)
    thp = parse_smaps_vma(SAMPLE_SMAPS, 0x7F1000001000)

    assert small["AnonHugePages"] == 0
    assert thp["AnonHugePages"] == 143360 * 1024
    assert thp["Size"] == 143360 * 1024


def test_round_up_and_cell_names() -> None:
    assert round_up(1, 4096) == 4096
    assert round_up(4096, 4096) == 4096
    assert cell_name("thp", "many16_1t_same_head") == "thp:many16_1t_same_head"
    assert parse_cell("small:wide16_same_head") == ("small", "wide16_same_head")
    assert POLICIES == ("small", "thp")
    assert "isolated_head" in MODES


def test_pages_verified_requires_clean_small_and_latched_thp() -> None:
    request = 216 << 20
    assert pages_verified(_reports(small_huge=0, thp_huge=request, request=request)) is True
    assert pages_verified(_reports(small_huge=request, thp_huge=request, request=request)) is False
    assert pages_verified(_reports(small_huge=0, thp_huge=0, request=request)) is False


def test_decide_thp_helps_when_verified_pages_drop_many16() -> None:
    decision = decide_weight_thp(
        _comparisons(0.16, 0.04),
        _live_calls(),
        _reports(small_huge=0, thp_huge=216 << 20),
    )

    assert decision["signature"] == "thp_helps"
    assert decision["add_default_off_structure"] is False


def test_decide_page_neutral_when_leftover_matches() -> None:
    decision = decide_weight_thp(
        _comparisons(0.16, 0.15),
        _live_calls(),
        _reports(small_huge=0, thp_huge=216 << 20),
    )

    assert decision["signature"] == "page_neutral"
    assert decision["thp_helps"] is False
    assert decision["add_default_off_structure"] is False


def test_decide_thp_not_latched_when_smaps_misses() -> None:
    decision = decide_weight_thp(
        _comparisons(0.16, 0.04),
        _live_calls(),
        _reports(small_huge=0, thp_huge=0),
    )

    assert decision["signature"] == "thp_not_latched"
    assert decision["pages_verified"] is False
    assert decision["add_default_off_structure"] is False
