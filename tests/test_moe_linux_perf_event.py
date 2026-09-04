from __future__ import annotations

import struct

import pytest

from optimizations.fused_moe_sve.benchmarks.linux_perf_event import (
    PERF_ATTR_FLAG_DISABLED,
    PERF_ATTR_FLAG_EXCLUDE_GUEST,
    PERF_ATTR_FLAG_INHERIT,
    PERF_ATTR_SIZE,
    PerfCounterSet,
    PerfEventSpec,
    _event_config,
    perf_event_attr,
)


def test_perf_event_attr_encodes_type_config_and_read_format() -> None:
    payload = perf_event_attr(PerfEventSpec("event", 284, 0x41, 280))
    assert len(payload) == PERF_ATTR_SIZE
    assert struct.unpack_from("II", payload, 0) == (284, PERF_ATTR_SIZE)
    assert struct.unpack_from("Q", payload, 8)[0] == 0x41
    assert struct.unpack_from("Q", payload, 32)[0] == 3
    assert struct.unpack_from("Q", payload, 40)[0] == (
        PERF_ATTR_FLAG_DISABLED | PERF_ATTR_FLAG_INHERIT | PERF_ATTR_FLAG_EXCLUDE_GUEST
    )

    uncore = perf_event_attr(PerfEventSpec("uncore", 305, 0xB8, 280, exclude_guest=False))
    assert struct.unpack_from("Q", uncore, 40)[0] == PERF_ATTR_FLAG_DISABLED | PERF_ATTR_FLAG_INHERIT


def test_perf_counter_set_rejects_duplicate_names() -> None:
    spec = PerfEventSpec("duplicate", 8, 0x11, 304)
    with pytest.raises(ValueError, match="must be unique"):
        PerfCounterSet([spec, spec], opener=lambda _spec: -1)


@pytest.mark.parametrize("encoded", ["event=0x41", "config=0xb8"])
def test_event_config_accepts_core_and_hisilicon_sysfs_fields(tmp_path, encoded: str) -> None:
    device = tmp_path / "pmu"
    (device / "events").mkdir(parents=True)
    (device / "events" / "sample").write_text(encoded, encoding="utf-8")
    assert _event_config(device, "sample") == int(encoded.split("=")[1], 0)
