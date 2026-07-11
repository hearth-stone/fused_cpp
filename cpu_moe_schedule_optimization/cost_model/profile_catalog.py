"""Strict schema-v2 profile identity and selection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable


class ProfileCompatibilityError(ValueError):
    """Raised when no exact calibration profile matches a requested policy."""


@dataclass(frozen=True)
class ProfilePolicy:
    mode: str
    degree: int
    hidden_size: int
    intermediate_size: int
    global_experts: int
    local_experts: int
    backend: str
    backend_n_tile: int
    activation: str
    dtype: str
    w13_split: bool
    w13_split_chunks: int
    measurement_experts: int
    cores_per_rank: int
    concurrent_ranks: int
    llc_bytes_per_rank: int
    numa_nodes: tuple[int, ...]
    cpu_ids_by_rank: tuple[tuple[int, ...], ...]
    source_sha256: str
    extension_sha256: str

    @classmethod
    def from_payload(cls, payload: dict) -> "ProfilePolicy":
        if int(payload.get("schema_version", 0)) != 2:
            raise ProfileCompatibilityError("policy identity requires schema v2")
        target = payload["target"]
        kernel = payload["kernel"]
        parallelism = payload["parallelism"]
        expert = payload["expert_shape"]
        llc_values = target.get("llc_bytes_by_rank")
        if llc_values is None:
            llc_values = [target.get("llc_bytes_per_rank")]
        if not llc_values or any(value is None for value in llc_values):
            raise ProfileCompatibilityError("profile is missing LLC identity")
        if len({int(value) for value in llc_values}) != 1:
            raise ProfileCompatibilityError(
                "heterogeneous per-rank LLC profiles are not supported"
            )
        cpu_ids_by_rank = target.get("cpu_ids_by_rank")
        if cpu_ids_by_rank is None:
            cpu_ids_by_rank = [target.get("cpu_ids", [])]
        numa_nodes = target.get("numa_nodes")
        if numa_nodes is None:
            numa_nodes = [target.get("numa_node", -1)]
        source_sha = kernel.get("source_sha256")
        extension_sha = kernel.get("extension_sha256")
        if not source_sha or not extension_sha:
            raise ProfileCompatibilityError(
                "schema-v2 profile requires source and extension hashes"
            )
        return cls(
            mode=str(parallelism["mode"]),
            degree=int(parallelism["degree"]),
            hidden_size=int(expert["hidden_size"]),
            intermediate_size=int(expert["intermediate_size"]),
            global_experts=int(parallelism["global_experts"]),
            local_experts=int(parallelism["local_experts"]),
            backend=str(kernel["backend"]),
            backend_n_tile=int(kernel["backend_n_tile"]),
            activation=str(expert["activation"]),
            dtype=str(expert["dtype"]),
            w13_split=bool(kernel["w13_split"]),
            w13_split_chunks=int(kernel["w13_split_chunks"]),
            measurement_experts=int(expert["measurement_experts"]),
            cores_per_rank=int(target["cores_per_rank"]),
            concurrent_ranks=int(target["concurrent_ranks"]),
            llc_bytes_per_rank=int(llc_values[0]),
            numa_nodes=tuple(int(value) for value in numa_nodes),
            cpu_ids_by_rank=tuple(
                tuple(int(cpu) for cpu in cpu_ids) for cpu_ids in cpu_ids_by_rank
            ),
            source_sha256=str(source_sha),
            extension_sha256=str(extension_sha),
        )

    def mismatch(self, query: "ProfileQuery") -> dict[str, tuple[object, object]]:
        mismatches: dict[str, tuple[object, object]] = {}
        for field in fields(query):
            expected = getattr(query, field.name)
            if expected is None:
                continue
            actual = getattr(self, field.name)
            if actual != expected:
                mismatches[field.name] = (actual, expected)
        return mismatches

    def key_without_split(self) -> tuple[object, ...]:
        return (
            self.mode,
            self.degree,
            self.hidden_size,
            self.intermediate_size,
            self.global_experts,
            self.local_experts,
            self.backend,
            self.backend_n_tile,
            self.activation,
            self.dtype,
            self.measurement_experts,
            self.cores_per_rank,
            self.concurrent_ranks,
            self.llc_bytes_per_rank,
            self.numa_nodes,
            self.cpu_ids_by_rank,
            self.source_sha256,
            self.extension_sha256,
        )


@dataclass(frozen=True)
class ProfileQuery:
    mode: str | None = None
    degree: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    global_experts: int | None = None
    local_experts: int | None = None
    backend: str | None = None
    backend_n_tile: int | None = None
    activation: str | None = None
    dtype: str | None = None
    w13_split: bool | None = None
    w13_split_chunks: int | None = None
    measurement_experts: int | None = None
    cores_per_rank: int | None = None
    concurrent_ranks: int | None = None
    llc_bytes_per_rank: int | None = None
    numa_nodes: tuple[int, ...] | None = None
    cpu_ids_by_rank: tuple[tuple[int, ...], ...] | None = None
    source_sha256: str | None = None
    extension_sha256: str | None = None


@dataclass(frozen=True)
class ProfileRecord:
    path: Path
    payload: dict
    policy: ProfilePolicy


class ProfileCatalog:
    def __init__(self, records: Iterable[ProfileRecord]):
        self.records = tuple(records)
        if not self.records:
            raise ProfileCompatibilityError("profile catalog is empty")

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "ProfileCatalog":
        records: list[ProfileRecord] = []
        for raw_path in paths:
            path = Path(raw_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                ProfileRecord(path, payload, ProfilePolicy.from_payload(payload))
            )
        return cls(records)

    @classmethod
    def from_directory(
        cls, directory: str | Path, pattern: str = "*_v2_*.json"
    ) -> "ProfileCatalog":
        return cls.from_paths(sorted(Path(directory).glob(pattern)))

    def select(self, query: ProfileQuery) -> ProfileRecord:
        matches = [record for record in self.records if not record.policy.mismatch(query)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            details = {
                record.path.name: record.policy.mismatch(query)
                for record in self.records
            }
            raise ProfileCompatibilityError(
                f"no exact profile matches {query}; mismatches={details}"
            )
        raise ProfileCompatibilityError(
            "profile query is ambiguous: "
            + ", ".join(record.path.name for record in matches)
        )

    def split_pair(self, query: ProfileQuery) -> tuple[ProfileRecord, ProfileRecord]:
        if query.w13_split is not None:
            raise ValueError("split_pair query must leave w13_split unspecified")
        no_split = self.select(
            ProfileQuery(**{**query.__dict__, "w13_split": False})
        )
        split = self.select(ProfileQuery(**{**query.__dict__, "w13_split": True}))
        if no_split.policy.key_without_split() != split.policy.key_without_split():
            raise ProfileCompatibilityError("split/no-split profiles are not a pair")
        if self._grid_signature(no_split.payload) != self._grid_signature(split.payload):
            raise ProfileCompatibilityError(
                "split/no-split profiles use different route/thread/shape grids"
            )
        return no_split, split

    @staticmethod
    def _grid_signature(payload: dict) -> tuple[object, ...]:
        isolated = tuple(
            sorted(
                (int(entry["routes"]), int(entry["threads"]))
                for entry in payload["isolated"]
            )
        )
        contention = tuple(
            sorted(
                (
                    tuple(sorted(map(int, entry["shape"]), reverse=True)),
                    int(entry["routes"]),
                )
                for entry in payload["entries"]
            )
        )
        return isolated, contention
