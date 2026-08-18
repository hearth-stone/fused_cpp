"""Versioned disk cache for analytical MoE isolated-cost scalars."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux/AArch64 production always has fcntl
    fcntl = None


MOE_COST_CACHE_SCHEMA_VERSION = 1
DEFAULT_MOE_COST_CACHE_DIR = Path.home() / ".fused_cpp" / "cache" / "moe_costs"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class MoeCostCacheLoad:
    status: str
    path: Path
    entries: dict[tuple[int, int], float]
    error: str | None = None


class MoeCostDiskCache:
    """Read-through cache bound to one complete analytical model identity."""

    def __init__(self, directory: str | Path, identity: Mapping[str, object]) -> None:
        self.directory = Path(directory).expanduser()
        self.identity = json.loads(_canonical_json(dict(identity)))
        digest = hashlib.sha256(_canonical_json(self.identity).encode("ascii")).hexdigest()[:24]
        self.path = self.directory / f"t_iso_{digest}.json"
        self.lock_path = self.directory / f"t_iso_{digest}.lock"

    def load(self) -> MoeCostCacheLoad:
        if not self.path.is_file():
            return MoeCostCacheLoad("miss", self.path, {})
        try:
            payload = json.loads(self.path.read_text())
            entries = self._validated_entries(payload)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            return MoeCostCacheLoad("error", self.path, {}, str(error))
        return MoeCostCacheLoad("hit", self.path, entries)

    def store(self, entries: Mapping[tuple[int, int], float]) -> MoeCostCacheLoad:
        validated = self._validate_mapping(entries)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._exclusive_lock():
                merged = {}
                if self.path.is_file():
                    try:
                        merged = self._validated_entries(json.loads(self.path.read_text()))
                    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                        merged = {}
                merged.update(validated)
                payload = {
                    "schema_version": MOE_COST_CACHE_SCHEMA_VERSION,
                    "identity": self.identity,
                    "entries": [
                        {"routes": routes, "threads": threads, "time_ns": time_ns}
                        for (routes, threads), time_ns in sorted(merged.items())
                    ],
                }
                fd, temporary_name = tempfile.mkstemp(
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    dir=self.directory,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(fd, "w", encoding="ascii") as stream:
                        stream.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True))
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.path)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError as error:
            return MoeCostCacheLoad("error", self.path, {}, str(error))
        return MoeCostCacheLoad("stored", self.path, merged)

    def _validated_entries(self, payload: object) -> dict[tuple[int, int], float]:
        if not isinstance(payload, dict):
            raise ValueError("MoE cost cache root must be an object")
        if payload.get("schema_version") != MOE_COST_CACHE_SCHEMA_VERSION:
            raise ValueError("MoE cost cache schema mismatch")
        if payload.get("identity") != self.identity:
            raise ValueError("MoE cost cache identity mismatch")
        rows = payload.get("entries")
        if not isinstance(rows, list):
            raise ValueError("MoE cost cache entries must be a list")
        entries: dict[tuple[int, int], float] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("MoE cost cache entry must be an object")
            routes = int(row["routes"])
            threads = int(row["threads"])
            time_ns = float(row["time_ns"])
            key = (routes, threads)
            if key in entries:
                raise ValueError(f"duplicate MoE cost cache entry {key}")
            entries[key] = self._validate_entry(routes, threads, time_ns)
        return entries

    def _validate_mapping(self, entries: Mapping[tuple[int, int], float]) -> dict[tuple[int, int], float]:
        validated = {}
        for (routes, threads), time_ns in entries.items():
            routes = int(routes)
            threads = int(threads)
            validated[(routes, threads)] = self._validate_entry(routes, threads, float(time_ns))
        return validated

    @staticmethod
    def _validate_entry(routes: int, threads: int, time_ns: float) -> float:
        if routes <= 0 or threads <= 0 or not math.isfinite(time_ns) or time_ns <= 0.0:
            raise ValueError(
                f"invalid MoE cost cache entry routes={routes}, threads={threads}, time_ns={time_ns}"
            )
        return time_ns

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self.lock_path.open("a+b") as stream:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "DEFAULT_MOE_COST_CACHE_DIR",
    "MOE_COST_CACHE_SCHEMA_VERSION",
    "MoeCostCacheLoad",
    "MoeCostDiskCache",
]
