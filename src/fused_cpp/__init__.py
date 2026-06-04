# -*- coding: utf-8 -*-
"""fused-cpp: CPU Fused MLA and MoE implementations (pure PyTorch)."""
from fused_cpp.acl_gemm import (
    ACLGEMMHandler,
    _supports_acl,
    acl_gemm,
    create_acl_gemm,
    get_acl_affinity,
    set_acl_affinity,
)
from fused_cpp.kai_gemm import (
    KAIGEMMHandler,
    KAIThreadPool,
    _supports_kai,
    create_kai_gemm,
    kai_gemm,
    kai_gemm_prepare,
)
from fused_cpp.mla import CPUFusedMLAImpl
from fused_cpp.mla.impl import _HAS_CPP
from fused_cpp.moe import AWQFusedMoEImpl, FusedMoEImpl
from fused_cpp.sdpa import (
    VersionInfo,
    available_sdpa_versions,
    get_sdpa_version,
    register_sdpa_version,
    scaled_dot_product_attention,
    sdpa_versioned,
)
from fused_cpp.sparse_mla import (
    flash_mla_sparse_fwd_naive,
    sparse_mla,
    sparse_mla_naive,
)
from fused_cpp.w4a8_linear import (
    unpack_awq_qweight,
    unpack_awq_qzeros,
    w4a8_linear,
)


def has_cpp_kernels() -> bool:
    """Return True if C++ extension is available."""
    return _HAS_CPP


BACKEND: str = "cpp" if _HAS_CPP else "pytorch"

__all__ = [
    "ACLGEMMHandler",
    "AWQFusedMoEImpl",
    "BACKEND",
    "CPUFusedMLAImpl",
    "FusedMoEImpl",
    "KAIGEMMHandler",
    "KAIThreadPool",
    "VersionInfo",
    "_supports_acl",
    "_supports_kai",
    "acl_gemm",
    "available_sdpa_versions",
    "create_acl_gemm",
    "create_kai_gemm",
    "flash_mla_sparse_fwd_naive",
    "get_acl_affinity",
    "get_sdpa_version",
    "has_cpp_kernels",
    "kai_gemm",
    "kai_gemm_prepare",
    "register_sdpa_version",
    "scaled_dot_product_attention",
    "sdpa_versioned",
    "set_acl_affinity",
    "sparse_mla",
    "sparse_mla_naive",
    "unpack_awq_qweight",
    "unpack_awq_qzeros",
    "w4a8_linear",
]
