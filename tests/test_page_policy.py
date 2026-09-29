"""The shared page policy: one environment surface for every large buffer.

The policy latches on the first allocation, so each case runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


REPORT = "from fused_cpp import _moe_C; import json; print('R' + json.dumps(_moe_C.page_policy_info()))"

MIB = 1024 * 1024


def resolve(**overrides: str) -> dict:
    """Resolve the policy in a fresh interpreter with only these FUSED_CPP_ vars."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("FUSED_CPP_")}
    env.update(overrides)
    env.setdefault("PYTHONPATH", "src")
    done = subprocess.run([sys.executable, "-c", REPORT], capture_output=True, text=True, env=env)
    if done.returncode != 0:
        pytest.skip(f"page policy probe unavailable: {done.stderr.strip().splitlines()[-1:]}")
    payload = next((line[1:] for line in done.stdout.splitlines() if line.startswith("R{")), None)
    assert payload is not None, done.stdout
    return json.loads(payload)


def test_default_policy_is_transparent_huge_pages() -> None:
    info = resolve()
    assert info["policy"] == "thp"
    assert info["thp_align_bytes"] == 2 * MIB


@pytest.mark.parametrize(
    ("value", "expected"),
    (("small", "small"), ("4k", "small"), ("thp", "thp"), ("hugetlb", "hugetlb")),
)
def test_pages_variable_selects_the_backing(value: str, expected: str) -> None:
    assert resolve(FUSED_CPP_PAGES=value)["policy"] == expected


def test_page_size_is_honoured_and_must_be_a_power_of_two() -> None:
    assert resolve(FUSED_CPP_PAGES="hugetlb", FUSED_CPP_PAGE_SIZE_MB="2")["hugetlb_bytes"] == 2 * MIB
    # 3 MiB is not a valid MAP_HUGE_SHIFT size, so it falls back to the default.
    assert resolve(FUSED_CPP_PAGES="hugetlb", FUSED_CPP_PAGE_SIZE_MB="3")["hugetlb_bytes"] == 32 * MIB


def test_small_requests_avoid_whole_huge_pages_by_default() -> None:
    """Without this gate 40 scratch buffers needing 58 MiB would reserve 1280 MiB."""
    info = resolve(FUSED_CPP_PAGES="hugetlb", FUSED_CPP_PAGE_SIZE_MB="32")
    assert info["hugetlb_min_bytes"] == info["hugetlb_bytes"]
    override = resolve(FUSED_CPP_PAGES="hugetlb", FUSED_CPP_PAGE_SIZE_MB="32", FUSED_CPP_PAGE_MIN_KB="64")
    assert override["hugetlb_min_bytes"] == 64 * 1024


@pytest.mark.parametrize(
    ("legacy", "expected"),
    (
        ({"FUSED_CPP_MOE_THP": "0"}, "small"),
        ({"FUSED_CPP_MOE_THP": "1"}, "thp"),
        ({"FUSED_CPP_MOE_HUGETLB": "1"}, "hugetlb"),
    ),
)
def test_deprecated_variables_still_select_the_same_backing(legacy: dict, expected: str) -> None:
    assert resolve(**legacy)["policy"] == expected


def test_new_variable_wins_over_the_deprecated_one() -> None:
    info = resolve(FUSED_CPP_PAGES="small", FUSED_CPP_MOE_HUGETLB="1")
    assert info["policy"] == "small"


def test_legacy_hugetlb_megabytes_alias() -> None:
    info = resolve(FUSED_CPP_MOE_HUGETLB="1", FUSED_CPP_MOE_HUGETLB_MB="2")
    assert (info["policy"], info["hugetlb_bytes"]) == ("hugetlb", 2 * MIB)


def test_counters_are_exposed_for_diagnostics() -> None:
    info = resolve()
    for key in ("live_bytes", "live_mappings", "peak_bytes", "total_allocations", "hugetlb_fallbacks"):
        assert isinstance(info[key], int)
        assert info[key] >= 0
