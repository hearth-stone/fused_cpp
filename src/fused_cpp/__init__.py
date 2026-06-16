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
from . import i8gemm
from fused_cpp.i8gemm import PreparedI8GEMMWeight, _supports_i8gemm
from fused_cpp.mla import CPUFusedMLAImpl
from fused_cpp.mla.impl import _HAS_CPP
from fused_cpp.moe import (
    AWQFusedMoEImpl,
    FusedMoEImpl,
    PreparedBF16TiledFusedMoEWeights,
    _HAS_BF16_TILED_FUSED_MOE,
    fused_moe_bf16_tiled,
    fused_moe_naive,
    bf16_tiled_fused_moe,
    naive_fused_moe,
    prepare_fused_moe_bf16_tiled_weights,
    prepare_bf16_tiled_fused_moe_weights,
)
from fused_cpp.mqa import multi_query_attention, multi_query_attention_torch
from fused_cpp.deepseek_v4_attn_gemm_fused import (
    PreparedDeepSeekV4AttnGemmWeights,
    PreparedWeight,
    _HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED,
    deepseek_v4_attn_gemm_fused_prepacked,
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused,
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt,
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare,
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare_weights,
    fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepacked,
    prepare_deepseek_v4_attn_gemm_weights,
)
from fused_cpp.sdpa import (
    VersionInfo,
    available_sdpa_versions,
    get_sdpa_version,
    register_sdpa_version,
    scaled_dot_product_attention,
    sdpa_versioned,
)
from fused_cpp.sparse_mla import (
    flash_mla_sparse_fwd,
    flash_mla_sparse_fwd_naive,
    sparse_mla,
    sparse_mla_naive,
)
from fused_cpp.sparse_attn_indexer import cpu_sparse_attn_indexer_op
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
    "PreparedDeepSeekV4AttnGemmWeights",
    "PreparedBF16TiledFusedMoEWeights",
    "PreparedI8GEMMWeight",
    "PreparedWeight",
    "VersionInfo",
    "_HAS_DEEPSEEK_V4_ATTN_GEMM_FUSED",
    "_HAS_BF16_TILED_FUSED_MOE",
    "_supports_acl",
    "_supports_i8gemm",
    "_supports_kai",
    "acl_gemm",
    "available_sdpa_versions",
    "create_acl_gemm",
    "create_kai_gemm",
    "cpu_sparse_attn_indexer_op",
    "deepseek_v4_attn_gemm_fused_prepacked",
    "flash_mla_sparse_fwd",
    "flash_mla_sparse_fwd_naive",
    "fused_moe_bf16_tiled",
    "fused_moe_naive",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare_weights",
    "fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepacked",
    "get_acl_affinity",
    "get_sdpa_version",
    "bf16_tiled_fused_moe",
    "has_cpp_kernels",
    "i8gemm",
    "kai_gemm",
    "kai_gemm_prepare",
    "multi_query_attention",
    "multi_query_attention_torch",
    "naive_fused_moe",
    "prepare_deepseek_v4_attn_gemm_weights",
    "prepare_fused_moe_bf16_tiled_weights",
    "prepare_bf16_tiled_fused_moe_weights",
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
