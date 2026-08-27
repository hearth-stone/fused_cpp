#!/usr/bin/env python3
"""Run a declared fused-expert experiment suite on one configured SSH host."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a JSON object: {path}")
    return payload


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must match {IDENTIFIER.pattern!r}, got {value!r}")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return value


def validate_machine(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("machine schema_version must be 1")
    _require_identifier(payload.get("id"), field="machine.id")
    _require_string(payload.get("ssh_host"), field="machine.ssh_host")
    project_root = _require_string(payload.get("project_root"), field="machine.project_root")
    results_root = _require_string(payload.get("remote_results_root"), field="machine.remote_results_root")
    if not PurePosixPath(project_root).is_absolute() or not PurePosixPath(results_root).is_absolute():
        raise ValueError("machine project_root and remote_results_root must be absolute POSIX paths")
    _require_string(payload.get("python"), field="machine.python")
    _require_string_list(payload.get("affinity_argv", []), field="machine.affinity_argv")
    _require_string_list(payload.get("snapshot_submodules", []), field="machine.snapshot_submodules")
    _require_string_list(payload.get("snapshot_external_paths", []), field="machine.snapshot_external_paths")
    for field in ("environment", "variables"):
        values = payload.get(field, {})
        if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
            raise ValueError(f"machine.{field} must be a string-keyed object")
        if any(not isinstance(value, (str, int, float)) for value in values.values()):
            raise ValueError(f"machine.{field} values must be strings or numbers")
    return payload


def validate_suite(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("suite schema_version must be 1")
    _require_identifier(payload.get("id"), field="suite.id")
    for section in ("setup", "cases"):
        entries = payload.get(section)
        if not isinstance(entries, list):
            raise ValueError(f"suite.{section} must be a list")
        seen: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"suite.{section}[{index}] must be an object")
            entry_id = _require_identifier(entry.get("id"), field=f"suite.{section}[{index}].id")
            if entry_id in seen:
                raise ValueError(f"duplicate {section} id {entry_id!r}")
            seen.add(entry_id)
            _require_string_list(entry.get("argv"), field=f"suite.{section}[{index}].argv")
            if "timeout_seconds" in entry and (
                not isinstance(entry["timeout_seconds"], int) or entry["timeout_seconds"] <= 0
            ):
                raise ValueError(f"suite.{section}[{index}].timeout_seconds must be positive")
            if entry.get("output_json", False) and "{remote_output}" not in entry["argv"]:
                raise ValueError(f"suite.{section}[{index}] enables output_json without {{remote_output}}")
    return payload


def load_machine(path: Path) -> dict[str, Any]:
    return validate_machine(_load_object(path))


def load_suite(path: Path) -> dict[str, Any]:
    return validate_suite(_load_object(path))


def render_argv(argv: list[str], context: dict[str, object]) -> list[str]:
    try:
        return [argument.format_map(context) for argument in argv]
    except KeyError as error:
        raise ValueError(f"unknown command placeholder {error.args[0]!r}") from error


def build_remote_command(
    *,
    project_root: str,
    argv: list[str],
    environment: dict[str, str],
    affinity_argv: list[str],
) -> str:
    command = ["env", *(f"{key}={value}" for key, value in sorted(environment.items())), *affinity_argv, *argv]
    return f"cd {shlex.quote(project_root)} && {shlex.join(command)}"


def parse_prefixed_json(stdout: str, prefix: str) -> list[object]:
    return [json.loads(line.removeprefix(prefix)) for line in stdout.splitlines() if line.startswith(prefix)]


def _run(command: list[str], *, timeout_seconds: int, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _resolve_revision(revision: str) -> str:
    result = _run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        timeout_seconds=30,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot resolve revision {revision!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def _submodule_commit(revision: str, path: str) -> str:
    result = _run(["git", "ls-tree", revision, "--", path], timeout_seconds=30, cwd=REPO_ROOT)
    fields = result.stdout.strip().split()
    if result.returncode != 0 or len(fields) < 3 or fields[1] != "commit":
        raise RuntimeError(f"cannot resolve submodule {path!r} at revision {revision}")
    return fields[2]


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _populate_submodule(snapshot_root: Path, revision: str, relative_path: str) -> dict[str, str]:
    source = REPO_ROOT / relative_path
    if not source.is_dir():
        raise RuntimeError(f"required submodule is not initialized: {relative_path}")
    expected = _submodule_commit(revision, relative_path)
    actual = _run(["git", "rev-parse", "HEAD"], timeout_seconds=30, cwd=source)
    dirty = _run(["git", "status", "--porcelain"], timeout_seconds=30, cwd=source)
    if actual.returncode != 0 or actual.stdout.strip() != expected:
        raise RuntimeError(f"submodule {relative_path} is not at pinned commit {expected}")
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise RuntimeError(f"submodule {relative_path} has uncommitted changes")
    destination = snapshot_root / relative_path
    destination.mkdir(parents=True, exist_ok=True)
    copied = _run(
        ["rsync", "-a", "--exclude=.git/", f"{source}/", f"{destination}/"],
        timeout_seconds=300,
    )
    if copied.returncode != 0:
        raise RuntimeError(f"failed to copy submodule {relative_path}: {copied.stderr.strip()}")
    return {"path": relative_path, "commit": expected, "sha256": _tree_digest(destination)}


def _populate_external_path(snapshot_root: Path, relative_path: str) -> dict[str, str]:
    source = (REPO_ROOT / relative_path).resolve()
    if not source.is_dir():
        raise RuntimeError(f"required external source directory is unavailable: {relative_path}")
    destination = snapshot_root / relative_path
    destination.mkdir(parents=True, exist_ok=True)
    copied = _run(
        ["rsync", "-a", "--copy-links", "--exclude=.git/", f"{source}/", f"{destination}/"],
        timeout_seconds=300,
    )
    if copied.returncode != 0:
        raise RuntimeError(f"failed to copy external source {relative_path}: {copied.stderr.strip()}")
    return {"path": relative_path, "sha256": _tree_digest(destination)}


def sync_snapshot(machine: dict[str, Any], revision: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="moe-paper-snapshot-") as directory:
        temporary = Path(directory)
        archive = temporary / "source.tar"
        snapshot = temporary / "source"
        snapshot.mkdir()
        archived = _run(
            ["git", "archive", "--format=tar", "-o", str(archive), revision],
            timeout_seconds=120,
            cwd=REPO_ROOT,
        )
        if archived.returncode != 0:
            raise RuntimeError(f"git archive failed: {archived.stderr.strip()}")
        extracted = _run(["tar", "-xf", str(archive), "-C", str(snapshot)], timeout_seconds=120)
        if extracted.returncode != 0:
            raise RuntimeError(f"snapshot extraction failed: {extracted.stderr.strip()}")
        submodules = [
            _populate_submodule(snapshot, revision, submodule) for submodule in machine.get("snapshot_submodules", [])
        ]
        external_paths = [
            _populate_external_path(snapshot, external_path)
            for external_path in machine.get("snapshot_external_paths", [])
        ]
        created = _run(
            ["ssh", machine["ssh_host"], f"mkdir -p {shlex.quote(machine['project_root'])}"],
            timeout_seconds=60,
        )
        if created.returncode != 0:
            raise RuntimeError(f"remote mkdir failed: {created.stderr.strip()}")
        synced = _run(
            [
                "rsync",
                "-az",
                "--copy-links",
                f"{snapshot}/",
                f"{machine['ssh_host']}:{machine['project_root']}/",
            ],
            timeout_seconds=900,
        )
        if synced.returncode != 0:
            raise RuntimeError(f"snapshot sync failed: {synced.stderr.strip()}")
        return {"revision": revision, "submodules": submodules, "external_paths": external_paths}


def _context(machine: dict[str, Any], remote_output: str) -> dict[str, object]:
    return {
        "machine_id": machine["id"],
        "project_root": machine["project_root"],
        "python": machine["python"],
        "remote_output": remote_output,
        **machine.get("variables", {}),
    }


def _execute_entry(
    *,
    machine: dict[str, Any],
    entry: dict[str, Any],
    remote_output: str,
    local_directory: Path,
) -> dict[str, Any]:
    argv = render_argv(entry["argv"], _context(machine, remote_output))
    environment = {str(key): str(value) for key, value in machine.get("environment", {}).items()}
    affinity = machine["affinity_argv"] if entry.get("use_affinity", True) else []
    remote_command = build_remote_command(
        project_root=machine["project_root"],
        argv=argv,
        environment=environment,
        affinity_argv=affinity,
    )
    begin = time.monotonic()
    result = _run(
        ["ssh", machine["ssh_host"], remote_command],
        timeout_seconds=entry.get("timeout_seconds", 1800),
    )
    duration_seconds = time.monotonic() - begin
    local_directory.mkdir(parents=True, exist_ok=True)
    (local_directory / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (local_directory / "stderr.log").write_text(result.stderr, encoding="utf-8")
    record: dict[str, Any] = {
        "id": entry["id"],
        "argv": argv,
        "remote_command": remote_command,
        "returncode": result.returncode,
        "duration_seconds": duration_seconds,
    }
    prefix = entry.get("result_prefix")
    if prefix:
        record["prefixed_results"] = parse_prefixed_json(result.stdout, prefix)
    if result.returncode == 0 and entry.get("output_json", False):
        copied = _run(
            ["scp", f"{machine['ssh_host']}:{remote_output}", str(local_directory / "result.json")],
            timeout_seconds=120,
        )
        record["result_copy_returncode"] = copied.returncode
        if copied.returncode != 0:
            record["result_copy_error"] = copied.stderr.strip()
            record["returncode"] = copied.returncode
    (local_directory / "command.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def _remote_metadata(machine: dict[str, Any], revision: str, output_path: Path) -> None:
    argv = [
        machine["python"],
        "optimizations/fused_moe_sve/paper_experiments/collect_machine_metadata.py",
        "--machine-id",
        machine["id"],
        "--source-revision",
        revision,
    ]
    command = build_remote_command(
        project_root=machine["project_root"],
        argv=argv,
        environment={str(key): str(value) for key, value in machine.get("environment", {}).items()},
        affinity_argv=machine["affinity_argv"],
    )
    result = _run(["ssh", machine["ssh_host"], command], timeout_seconds=120)
    if result.returncode != 0:
        raise RuntimeError(f"metadata collection failed: {result.stderr.strip()}")
    output_path.write_text(
        json.dumps(json.loads(result.stdout), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selected_cases(suite: dict[str, Any], requested: set[str]) -> list[dict[str, Any]]:
    cases = suite["cases"]
    if not requested:
        return cases
    unknown = requested - {case["id"] for case in cases}
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    return [case for case in cases if case["id"] in requested]


def _dry_run_plan(machine: dict[str, Any], suite: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    remote_directory = str(PurePosixPath(machine["remote_results_root"]) / "DRY_RUN")
    entries = []
    for entry in [*suite["setup"], *cases]:
        remote_output = str(PurePosixPath(remote_directory) / f"{entry['id']}.json")
        argv = render_argv(entry["argv"], _context(machine, remote_output))
        entries.append(
            {
                "id": entry["id"],
                "command": build_remote_command(
                    project_root=machine["project_root"],
                    argv=argv,
                    environment={str(key): str(value) for key, value in machine.get("environment", {}).items()},
                    affinity_argv=machine["affinity_argv"] if entry.get("use_affinity", True) else [],
                ),
            }
        )
    return {"machine": machine["id"], "suite": suite["id"], "entries": entries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "tmp" / "moe_paper_runs")
    parser.add_argument("--case", action="append", default=[], help="run only the named case; repeatable")
    parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    machine = load_machine(args.machine)
    suite = load_suite(args.suite)
    cases = _selected_cases(suite, set(args.case))
    if args.dry_run:
        print(json.dumps(_dry_run_plan(machine, suite, cases), indent=2))
        return 0

    revision = _resolve_revision(args.revision)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{machine['id']}-{suite['id']}-{revision[:12]}"
    local_directory = args.output_root / run_id
    local_directory.mkdir(parents=True, exist_ok=False)
    snapshot_provenance = sync_snapshot(machine, revision) if args.sync else None
    remote_directory = str(PurePosixPath(machine["remote_results_root"]) / run_id)
    created = _run(
        ["ssh", machine["ssh_host"], f"mkdir -p {shlex.quote(remote_directory)}"],
        timeout_seconds=60,
    )
    if created.returncode != 0:
        raise RuntimeError(f"cannot create remote result directory: {created.stderr.strip()}")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "source_revision": revision,
        "snapshot_sync": bool(args.sync),
        "snapshot_provenance": snapshot_provenance,
        "machine": machine["id"],
        "suite": suite["id"],
        "remote_directory": remote_directory,
        "setup": [],
        "cases": [],
    }
    for entry in suite["setup"] if args.build else []:
        record = _execute_entry(
            machine=machine,
            entry=entry,
            remote_output=str(PurePosixPath(remote_directory) / f"{entry['id']}.json"),
            local_directory=local_directory / "setup" / entry["id"],
        )
        summary["setup"].append(record)
        if record["returncode"] != 0:
            (local_directory / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return record["returncode"]

    _remote_metadata(machine, revision, local_directory / "machine.json")
    for case in cases:
        record = _execute_entry(
            machine=machine,
            entry=case,
            remote_output=str(PurePosixPath(remote_directory) / f"{case['id']}.json"),
            local_directory=local_directory / "cases" / case["id"],
        )
        summary["cases"].append(record)
        if record["returncode"] != 0 and not args.continue_on_error:
            break
    summary["success"] = all(record["returncode"] == 0 for record in [*summary["setup"], *summary["cases"]])
    (local_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(local_directory)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
