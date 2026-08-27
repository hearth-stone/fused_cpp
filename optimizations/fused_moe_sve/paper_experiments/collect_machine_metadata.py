#!/usr/bin/env python3
"""Emit reproducibility metadata for one remote CPU MoE experiment run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


RECORDED_ENVIRONMENT = (
    "FUSED_CPP_MOE_HUGETLBFS_PATH",
    "FUSED_CPP_MOE_SVE",
    "FUSED_CPP_MOE_SVE_IMPL",
    "FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE",
    "FUSED_CPP_MOE_W2_BF16_ROUTE",
    "FUSED_CPP_SVE_VECTOR_BITS",
    "MKL_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "PYTHONPATH",
    "VECLIB_MAXIMUM_THREADS",
)


def _command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": command, "error": str(error)}
    return {
        "argv": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_metadata(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 - the metadata must retain arbitrary import failures.
        return {"module": name, "error": repr(error)}
    module_file = Path(module.__file__).resolve()
    payload: dict[str, Any] = {
        "module": name,
        "path": str(module_file),
        "sha256": _sha256(module_file),
    }
    page_policy = getattr(module, "page_policy_info", None)
    if callable(page_policy):
        payload["page_policy"] = page_policy()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    payload = {
        "schema_version": 1,
        "machine_id": args.machine_id,
        "source_revision": args.source_revision,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {"executable": sys.executable, "version": sys.version},
        "torch": {"version": torch.__version__, "num_threads": torch.get_num_threads()},
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {name: os.environ[name] for name in RECORDED_ENVIRONMENT if name in os.environ},
        "commands": {
            "lscpu": _command(["lscpu", "-J"]),
            "numactl": _command(["numactl", "--hardware"]),
            "cxx": _command(["c++", "--version"]),
        },
        "extensions": [_extension_metadata("fused_cpp._moe_C"), _extension_metadata("fused_cpp._C")],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
