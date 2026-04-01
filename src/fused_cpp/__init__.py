# -*- coding: utf-8 -*-
"""fused-cpp: CPU Fused MLA and MoE implementations (pure PyTorch)."""
from fused_cpp.mla.impl import _HAS_CPP
from fused_cpp.mla import CPUFusedMLAImpl
from fused_cpp.moe import FusedMoEImpl


def has_cpp_kernels() -> bool:
    """Return True if C++ extension is available."""
    return _HAS_CPP


BACKEND: str = "cpp" if _HAS_CPP else "pytorch"

__all__ = ["CPUFusedMLAImpl", "FusedMoEImpl", "has_cpp_kernels", "BACKEND"]
