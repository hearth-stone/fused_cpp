"""Focused tests for the build-time fixed SVE vector-length selection."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


def _load_setup_sve_helpers() -> dict[str, object]:
    """Load only the pure VL helpers without executing setuptools setup()."""
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    selected_names = {
        "_SUPPORTED_FIXED_SVE_VECTOR_BITS",
        "_detect_max_sve_vector_bits_for_build",
        "_sve_vector_bits_for_build",
    }
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            names = {target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)}
            if isinstance(node, ast.FunctionDef):
                names.add(node.name)
            if names & selected_names:
                selected_nodes.append(node)
    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace: dict[str, object] = {"ctypes": __import__("ctypes"), "os": os}
    exec(compile(module, str(setup_path), "exec"), namespace)
    return namespace


_HELPERS = _load_setup_sve_helpers()
_DETECT_MAX = _HELPERS["_detect_max_sve_vector_bits_for_build"]
_SELECT_FOR_BUILD = _HELPERS["_sve_vector_bits_for_build"]


def test_detect_max_sve_vector_bits_restores_original_configuration() -> None:
    calls: list[tuple[int, int]] = []
    responses = iter((16, 32, 16))

    def fake_prctl(option: int, value: int, _arg3: int, _arg4: int, _arg5: int) -> int:
        calls.append((option, value))
        return next(responses)

    assert _DETECT_MAX(fake_prctl) == 256
    assert calls == [(51, 0), (50, 256), (50, 16)]


def test_sve_vector_bits_explicit_override_skips_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUSED_CPP_SVE_VECTOR_BITS", "512")
    assert _SELECT_FOR_BUILD() == 512


@pytest.mark.parametrize("value", ["", "abc", "384", "4096"])
def test_sve_vector_bits_rejects_invalid_override(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("FUSED_CPP_SVE_VECTOR_BITS", value)
    with pytest.raises(RuntimeError, match="FUSED_CPP_SVE_VECTOR_BITS"):
        _SELECT_FOR_BUILD()


def test_sve_vector_bits_unset_uses_detected_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUSED_CPP_SVE_VECTOR_BITS", raising=False)
    monkeypatch.setitem(_SELECT_FOR_BUILD.__globals__, "_detect_max_sve_vector_bits_for_build", lambda: 256)
    assert _SELECT_FOR_BUILD() == 256


def test_sve_vector_bits_without_host_sve_keeps_build_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUSED_CPP_SVE_VECTOR_BITS", raising=False)
    assert _SELECT_FOR_BUILD(detect_host_max=False) == 128


def test_detect_max_sve_vector_bits_reports_probe_failure() -> None:
    responses = iter((32, -1))

    def fake_prctl(_option: int, _value: int, _arg3: int, _arg4: int, _arg5: int) -> int:
        return next(responses)

    with pytest.raises(RuntimeError, match="set FUSED_CPP_SVE_VECTOR_BITS explicitly"):
        _DETECT_MAX(fake_prctl)


def test_detect_max_sve_vector_bits_reports_restore_failure() -> None:
    responses = iter((32, 32, -1))

    def fake_prctl(_option: int, _value: int, _arg3: int, _arg4: int, _arg5: int) -> int:
        return next(responses)

    with pytest.raises(RuntimeError, match="restore the build thread"):
        _DETECT_MAX(fake_prctl)
