"""Hardware-derived cache-resident geometries for analytical service probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BF16_BYTES = 2
M12_ROWS = 12
K_ALIGNMENT = 8


def parse_cache_size(value: str) -> int:
    text = value.strip().upper()
    if not text:
        raise ValueError("cache size must be non-empty")
    scale = 1
    if text[-1] in {"K", "M", "G"}:
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[text[-1]]
        text = text[:-1]
    size = int(text) * scale
    if size <= 0:
        raise ValueError(f"cache size must be positive, got {value!r}")
    return size


def parse_linux_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux sysfs CPU-list syntax such as ``0-3,8,10-11``."""
    cpus: list[int] = []
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range {item!r}")
            cpus.extend(range(first, last + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU id {cpu}")
            cpus.append(cpu)
    if not cpus or len(set(cpus)) != len(cpus):
        raise ValueError(f"invalid Linux CPU list {value!r}")
    return tuple(cpus)


def format_linux_cpu_list(cpu_ids: Iterable[int]) -> str:
    """Return a stable compact Linux CPU-list representation."""
    cpus = sorted({int(cpu) for cpu in cpu_ids})
    if not cpus or cpus[0] < 0:
        raise ValueError("cpu_ids must contain non-negative values")
    ranges: list[str] = []
    first = last = cpus[0]
    for cpu in cpus[1:]:
        if cpu == last + 1:
            last = cpu
            continue
        ranges.append(str(first) if first == last else f"{first}-{last}")
        first = last = cpu
    ranges.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(ranges)


def read_cache_info(cpu: int, *, sysfs_cpu_root: Path = Path("/sys/devices/system/cpu")) -> dict[str, int]:
    """Read the cache hierarchy visible to one Linux CPU from sysfs."""
    if cpu < 0:
        raise ValueError(f"cpu must be non-negative, got {cpu}")
    result: dict[str, int] = {}
    cache_root = sysfs_cpu_root / f"cpu{cpu}" / "cache"
    for index in sorted(cache_root.glob("index*")):
        try:
            level = int((index / "level").read_text(encoding="utf-8").strip())
            cache_type = (index / "type").read_text(encoding="utf-8").strip().lower()
            size = parse_cache_size((index / "size").read_text(encoding="utf-8"))
            line_size = int((index / "coherency_line_size").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if level == 1 and cache_type in {"data", "unified"}:
            result["l1d_bytes_per_core"] = max(result.get("l1d_bytes_per_core", 0), size)
        elif level == 2:
            result["l2_bytes_per_core"] = max(result.get("l2_bytes_per_core", 0), size)
        elif level == 3:
            result["llc_bytes_per_rank"] = max(result.get("llc_bytes_per_rank", 0), size)
        result["cache_line_bytes"] = max(result.get("cache_line_bytes", 0), line_size)
    required = {"l1d_bytes_per_core", "l2_bytes_per_core", "llc_bytes_per_rank"}
    missing = required - result.keys()
    if missing:
        raise RuntimeError(f"failed to detect cache fields for CPU {cpu}: {sorted(missing)}")
    return result


def read_llc_domains(
    cpu_ids: Iterable[int],
    *,
    sysfs_cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> tuple[dict, ...]:
    """Return the LLC domains intersecting an explicitly pinned CPU rank.

    Domain membership comes from Linux ``shared_cpu_list`` rather than CPU-id
    adjacency. The returned CPU sets contain only CPUs selected for this rank.
    """
    selected = tuple(int(cpu) for cpu in cpu_ids)
    if not selected or min(selected) < 0 or len(set(selected)) != len(selected):
        raise ValueError("cpu_ids must be unique and non-negative")
    selected_set = set(selected)
    domains: dict[tuple[str, tuple[int, ...]], dict] = {}
    covered: set[int] = set()
    for cpu in selected:
        cache_root = sysfs_cpu_root / f"cpu{cpu}" / "cache"
        candidates = []
        for index in sorted(cache_root.glob("index*")):
            try:
                level = int((index / "level").read_text(encoding="utf-8").strip())
                cache_type = (index / "type").read_text(encoding="utf-8").strip().lower()
                size = parse_cache_size((index / "size").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if level < 3 or cache_type not in {"data", "unified"}:
                continue
            try:
                shared = parse_linux_cpu_list((index / "shared_cpu_list").read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise RuntimeError(f"failed to read LLC shared_cpu_list for CPU {cpu}") from error
            try:
                cache_id = (index / "id").read_text(encoding="utf-8").strip()
            except OSError:
                cache_id = format_linux_cpu_list(shared)
            candidates.append((level, size, cache_id, shared))
        if not candidates:
            raise RuntimeError(f"failed to detect an LLC domain for CPU {cpu}")
        level, size, cache_id, shared = max(candidates, key=lambda item: (item[0], item[1]))
        del level
        rank_cpus = tuple(value for value in selected if value in set(shared))
        if cpu not in rank_cpus:
            raise RuntimeError(f"CPU {cpu} is absent from its LLC shared_cpu_list")
        key = (cache_id, tuple(shared))
        existing = domains.get(key)
        if existing is not None and existing["capacity_bytes"] != size:
            raise RuntimeError(f"inconsistent capacity for LLC domain {cache_id}")
        domains[key] = {
            "id": cache_id,
            "cpu_ids": list(rank_cpus),
            "capacity_bytes": size,
        }
        covered.update(rank_cpus)
    if covered != selected_set:
        raise RuntimeError(f"LLC domains do not cover selected CPUs: {sorted(selected_set - covered)}")
    return tuple(
        sorted(domains.values(), key=lambda domain: min(domain["cpu_ids"]))
    )


@dataclass(frozen=True)
class ProbeGeometry:
    k: int
    n: int
    working_set_bytes: int
    target_bytes: int
    cache_bytes: int
    cache_fraction: float

    def __post_init__(self) -> None:
        if min(self.k, self.n, self.working_set_bytes, self.target_bytes, self.cache_bytes) <= 0:
            raise ValueError("probe geometry values must be positive")
        if self.k % K_ALIGNMENT:
            raise ValueError("probe K must be aligned to eight")
        if not 0.0 < self.cache_fraction <= 1.0:
            raise ValueError("cache_fraction must be in (0, 1]")
        if self.working_set_bytes > self.target_bytes:
            raise ValueError("probe working set exceeds its cache target")


def _cache_budget(cache_bytes: int, cache_fraction: float) -> int:
    if cache_bytes <= 0:
        raise ValueError(f"cache_bytes must be positive, got {cache_bytes}")
    if not 0.0 < cache_fraction <= 1.0:
        raise ValueError("cache_fraction must be in (0, 1]")
    return max(int(cache_bytes * cache_fraction), 1)


def m12_gemm_geometry(
    cache_bytes: int,
    n_columns: int,
    *,
    cache_fraction: float,
    rows: int = M12_ROWS,
    element_bytes: int = BF16_BYTES,
) -> ProbeGeometry:
    """Select a one-panel M12 A+B footprint that fits the cache budget."""
    if min(n_columns, rows, element_bytes) <= 0:
        raise ValueError("M12 probe dimensions must be positive")
    budget = _cache_budget(cache_bytes, cache_fraction)
    bytes_per_k = (rows + n_columns) * element_bytes
    k = budget // bytes_per_k // K_ALIGNMENT * K_ALIGNMENT
    if k < K_ALIGNMENT:
        raise ValueError(f"cache budget {budget} bytes cannot hold one aligned M12/N{n_columns} GEMM slice")
    working_set = bytes_per_k * k
    return ProbeGeometry(
        k=k,
        n=n_columns,
        working_set_bytes=working_set,
        target_bytes=budget,
        cache_bytes=cache_bytes,
        cache_fraction=cache_fraction,
    )


def b_only_geometry(
    cache_bytes: int,
    n_quantum: int,
    *,
    cache_fraction: float,
    preferred_k: int = 4096,
    element_bytes: int = BF16_BYTES,
) -> ProbeGeometry:
    """Select a tile-aligned packed-B footprint close to a cache budget."""
    if min(n_quantum, preferred_k, element_bytes) <= 0:
        raise ValueError("B-only probe dimensions must be positive")
    budget = _cache_budget(cache_bytes, cache_fraction)
    bytes_per_min_tile_k = n_quantum * element_bytes
    maximum_k = budget // bytes_per_min_tile_k // K_ALIGNMENT * K_ALIGNMENT
    if maximum_k < K_ALIGNMENT:
        raise ValueError(f"cache budget {budget} bytes cannot hold one packed-B tile")
    k = min(preferred_k // K_ALIGNMENT * K_ALIGNMENT, maximum_k)
    n_tiles = max(budget // (k * bytes_per_min_tile_k), 1)
    n = n_tiles * n_quantum
    working_set = k * n * element_bytes
    return ProbeGeometry(
        k=k,
        n=n,
        working_set_bytes=working_set,
        target_bytes=budget,
        cache_bytes=cache_bytes,
        cache_fraction=cache_fraction,
    )
