"""CPU Fused Multi-head Latent Attention (pure PyTorch)."""

from fused_mla_cpp.core import CPUFusedMLAImpl, _HAS_CPP


def has_cpp_kernels() -> bool:
    """Return True if C++ extension is available."""
    return _HAS_CPP


BACKEND: str = "cpp" if _HAS_CPP else "pytorch"

__all__ = ["CPUFusedMLAImpl", "has_cpp_kernels", "BACKEND"]
