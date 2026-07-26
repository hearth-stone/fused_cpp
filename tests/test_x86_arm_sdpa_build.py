"""x86 extension surface when ARM-only SDPA translation units are omitted."""

from __future__ import annotations

import ctypes
import importlib
import platform

import pytest

pytestmark = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"),
    reason="x86 architecture-specific build contract",
)

pytest.importorskip("fused_cpp._C")

from fused_cpp import _C  # noqa: E402

mqa_module = importlib.import_module("fused_cpp.mqa")
sparse_mla_module = importlib.import_module("fused_cpp.sparse_mla")


ARM_ONLY_BINDINGS = (
    "multi_query_attention",
    "validate_sdpa_flash2_neon_cache_microkernels",
    "benchmark_sdpa_flash2_neon_cache_microkernels",
    "list_microkernel_impls",
    "validate_microkernel",
    "benchmark_microkernel",
    "flash_mla_sparse_fwd",
)


def test_x86_extension_omits_arm_only_sdpa_bindings() -> None:
    missing = [name for name in ARM_ONLY_BINDINGS if hasattr(_C, name)]
    assert not missing
    assert not mqa_module._HAS_CPP_MQA
    assert not sparse_mla_module._HAS_CPP_SPARSE_MLA


def test_x86_extension_keeps_portable_sdpa_versions() -> None:
    versions = set(_C.list_sdpa_versions())
    assert {"naive", "flash1", "flash2", "flash2_neon"}.issubset(versions)
    assert not any(name.startswith("flash2_neon_cache") for name in versions)
    assert not any(name.startswith("flash2_neon_l3kv") for name in versions)

    from fused_cpp.sdpa import available_sdpa_versions

    python_versions = {info.name for info in available_sdpa_versions() if info.source == "cpp"}
    assert python_versions == versions


def test_x86_extension_omits_arm_llamacpp_bridge() -> None:
    library = ctypes.CDLL(_C.__file__)
    with pytest.raises(AttributeError):
        getattr(library, "fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp")
