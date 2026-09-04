"""Minimal Linux perf_event_open support for the Arm MoE Lab probes."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import platform
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403
PERF_ATTR_SIZE = 120
PERF_ATTR_FLAG_DISABLED = 1 << 0
PERF_ATTR_FLAG_INHERIT = 1 << 1
PERF_ATTR_FLAG_EXCLUDE_GUEST = 1 << 20
DDRC_EVENT_CONFIG = {
    "flux_rd": 0x84,
    "flux_wr": 0x83,
    "read_cmd": 0x41,
    "read_cmd_occupancy": 0x80,
}


@dataclass(frozen=True)
class PerfEventSpec:
    name: str
    pmu_type: int
    config: int
    cpu: int
    exclude_guest: bool = True


@dataclass(frozen=True)
class PerfEventValue:
    count: int
    time_enabled_ns: int
    time_running_ns: int

    @property
    def running_ratio(self) -> float:
        if self.time_enabled_ns <= 0:
            return 0.0
        return self.time_running_ns / self.time_enabled_ns


def perf_event_attr(spec: PerfEventSpec) -> bytes:
    payload = bytearray(PERF_ATTR_SIZE)
    struct.pack_into("II", payload, 0, spec.pmu_type, PERF_ATTR_SIZE)
    struct.pack_into("Q", payload, 8, spec.config)
    struct.pack_into(
        "Q",
        payload,
        32,
        PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
    )
    flags = PERF_ATTR_FLAG_DISABLED | PERF_ATTR_FLAG_INHERIT
    if spec.exclude_guest:
        flags |= PERF_ATTR_FLAG_EXCLUDE_GUEST
    struct.pack_into("Q", payload, 40, flags)
    return bytes(payload)


def _syscall_number() -> int:
    if platform.system() != "Linux":
        raise OSError(errno.ENOSYS, "perf_event_open requires Linux")
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return 241
    if machine in {"x86_64", "amd64"}:
        return 298
    raise OSError(errno.ENOSYS, f"unsupported perf_event_open architecture {machine!r}")


def _open_event(spec: PerfEventSpec) -> int:
    payload = ctypes.create_string_buffer(perf_event_attr(spec))
    libc = ctypes.CDLL(None, use_errno=True)
    fd = int(libc.syscall(_syscall_number(), ctypes.byref(payload), -1, spec.cpu, -1, 0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"perf_event_open failed for {spec.name}: {os.strerror(error)}")
    return fd


class PerfCounterSet:
    def __init__(
        self,
        specs: list[PerfEventSpec],
        *,
        opener: Callable[[PerfEventSpec], int] = _open_event,
    ) -> None:
        if not specs:
            raise ValueError("at least one perf event is required")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("perf event names must be unique")
        self.specs = specs
        self._fds: list[int] = []
        try:
            for spec in specs:
                self._fds.append(opener(spec))
        except BaseException:
            self.close()
            raise

    def reset_enable(self) -> None:
        for fd in self._fds:
            fcntl.ioctl(fd, PERF_EVENT_IOC_RESET, 0)
        for fd in self._fds:
            fcntl.ioctl(fd, PERF_EVENT_IOC_ENABLE, 0)

    def disable_read(self) -> dict[str, PerfEventValue]:
        for fd in reversed(self._fds):
            fcntl.ioctl(fd, PERF_EVENT_IOC_DISABLE, 0)
        values = {}
        for spec, fd in zip(self.specs, self._fds, strict=True):
            raw = os.read(fd, 24)
            if len(raw) != 24:
                raise OSError(f"short perf read for {spec.name}: {len(raw)} bytes")
            values[spec.name] = PerfEventValue(*struct.unpack("QQQ", raw))
        return values

    def close(self) -> None:
        for fd in self._fds:
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> PerfCounterSet:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _read_int(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip(), 0)


def _event_config(device: Path, event: str) -> int:
    text = (device / "events" / event).read_text(encoding="utf-8").strip()
    for prefix in ("event=", "config="):
        if text.startswith(prefix):
            return int(text[len(prefix) :], 0)
    raise ValueError(f"unsupported event encoding in {device.name}/{event}: {text!r}")


def stream_pressure_specs(
    sysfs_root: Path = Path("/sys/bus/event_source/devices"),
) -> list[PerfEventSpec]:
    core = sysfs_root / "armv8_pmuv3_0"
    core_events = {
        "cycles": "cpu_cycles",
        "instructions": "inst_retired",
        "stall_backend": "stall_backend",
        "mem_access": "mem_access",
        "l2d_cache_refill": "l2d_cache_refill",
        "ll_cache_rd": "ll_cache_rd",
        "ll_cache_miss_rd": "ll_cache_miss_rd",
        "dtlb_walk": "dtlb_walk",
    }
    specs = [
        PerfEventSpec(f"core.{name}", _read_int(core / "type"), _event_config(core, event), 304)
        for name, event in core_events.items()
    ]
    for index in range(10):
        device = sysfs_root / f"hisi_sccl25_l3c{index}"
        cpu = _read_int(device / "cpumask")
        for event in ("l3c_ref", "l3c_hit"):
            specs.append(
                PerfEventSpec(
                    f"l3c{index}.{event}",
                    _read_int(device / "type"),
                    _event_config(device, event),
                    cpu,
                    exclude_guest=False,
                )
            )
    for controller in ("0_0", "0_1", "2_0", "2_1", "3_0", "3_1", "5_0", "5_1"):
        device = sysfs_root / f"hisi_sccl25_ddrc{controller}"
        cpu = _read_int(device / "cpumask")
        for event, config in DDRC_EVENT_CONFIG.items():
            specs.append(
                PerfEventSpec(
                    f"ddrc{controller}.{event}",
                    _read_int(device / "type"),
                    config,
                    cpu,
                    exclude_guest=False,
                )
            )
    return specs
