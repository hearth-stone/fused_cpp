# -*- coding: utf-8 -*-
"""数值正确性测试：自定义 SDPA vs PyTorch F.scaled_dot_product_attention。"""

import pytest
import torch
import torch.nn.functional as F

from fused_cpp.sdpa import scaled_dot_product_attention, _HAS_CPP_SDPA


# ── 辅助函数 ──────────────────────────────────────────────────────────────


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个张量的余弦相似度（展平后）。"""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))


def _error_metrics(a: torch.Tensor, b: torch.Tensor):
    """计算最大绝对误差和最大相对误差。"""
    a_f = a.float()
    b_f = b.float()
    abs_err = (a_f - b_f).abs()
    max_abs_err = float(abs_err.max())
    # 相对误差：避免除零
    denom = b_f.abs().clamp(min=1e-12)
    max_rel_err = float((abs_err / denom).max())
    return max_abs_err, max_rel_err


# ── 参数化配置 ────────────────────────────────────────────────────────────

BATCH_SIZES = [1, 4, 8]
SEQ_LENS = [1, 128, 512, 2048]
NUM_HEADS_LIST = [8, 16]
DTYPES = [torch.float32, torch.bfloat16]
IS_CAUSAL_LIST = [True, False]


# ── 标准 SDPA 测试（qk_head_dim == v_head_dim） ──────────────────────────


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS_LIST)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("is_causal", IS_CAUSAL_LIST, ids=["causal", "noncausal"])
def test_sdpa_standard(batch_size, seq_len, num_heads, dtype, is_causal):
    """标准 SDPA：qk_head_dim == v_head_dim == 64。"""
    head_dim = 64
    torch.manual_seed(42)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # 参考实现
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    # 自定义实现
    out = scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    cos_sim = _cosine_similarity(out, ref)
    max_abs, max_rel = _error_metrics(out, ref)

    # 打印辅助信息
    print(f"\n[标准 SDPA] B={batch_size}, L={seq_len}, N={num_heads}, dtype={dtype}, causal={is_causal}")
    print(f"  余弦相似度: {cos_sim:.6f}")
    print(f"  最大绝对误差: {max_abs:.6e}")
    print(f"  最大相对误差: {max_rel:.6e}")

    threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert cos_sim >= threshold, (
        f"余弦相似度 {cos_sim:.6f} < {threshold} "
        f"(B={batch_size}, L={seq_len}, N={num_heads}, "
        f"dtype={dtype}, causal={is_causal})"
    )


# ── MLA 场景测试（qk_head_dim != v_head_dim） ────────────────────────────


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
@pytest.mark.parametrize("is_causal", IS_CAUSAL_LIST, ids=["causal", "noncausal"])
def test_sdpa_mla(batch_size, seq_len, dtype, is_causal):
    """MLA 场景：qk_head_dim=192, v_head_dim=128，Q/K 和 V 的 head_dim 不同。"""
    num_heads = 16
    qk_head_dim = 192
    v_head_dim = 128
    torch.manual_seed(42)

    q = torch.randn(batch_size, num_heads, seq_len, qk_head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, qk_head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, v_head_dim, dtype=dtype)

    # 参考实现
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    # 自定义实现
    out = scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    cos_sim = _cosine_similarity(out, ref)
    max_abs, max_rel = _error_metrics(out, ref)

    print(
        f"\n[MLA SDPA] B={batch_size}, L={seq_len}, N={num_heads}, "
        f"qk_dim={qk_head_dim}, v_dim={v_head_dim}, "
        f"dtype={dtype}, causal={is_causal}"
    )
    print(f"  余弦相似度: {cos_sim:.6f}")
    print(f"  最大绝对误差: {max_abs:.6e}")
    print(f"  最大相对误差: {max_rel:.6e}")

    threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert cos_sim >= threshold, (
        f"余弦相似度 {cos_sim:.6f} < {threshold} (MLA, B={batch_size}, L={seq_len}, dtype={dtype}, causal={is_causal})"
    )


# ── attn_mask 测试 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
def test_sdpa_attn_mask(dtype):
    """测试 additive attention mask 参数。"""
    batch_size = 2
    num_heads = 8
    seq_len = 64
    head_dim = 64
    torch.manual_seed(42)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # 创建 additive mask（随机的负值掩码）
    mask = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=dtype)

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)

    cos_sim = _cosine_similarity(out, ref)
    max_abs, max_rel = _error_metrics(out, ref)

    print(f"\n[attn_mask] dtype={dtype}")
    print(f"  余弦相似度: {cos_sim:.6f}")
    print(f"  最大绝对误差: {max_abs:.6e}")
    print(f"  最大相对误差: {max_rel:.6e}")

    threshold = 0.9999 if dtype == torch.float32 else 0.999
    assert cos_sim >= threshold, f"余弦相似度 {cos_sim:.6f} < {threshold} (attn_mask, dtype={dtype})"


# ── scale=None 自动计算测试 ───────────────────────────────────────────────


def test_sdpa_auto_scale():
    """测试 scale=None 时自动计算缩放因子的行为。"""
    batch_size = 2
    num_heads = 8
    seq_len = 32
    head_dim = 64
    torch.manual_seed(42)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # scale=None（自动计算）
    out_auto = scaled_dot_product_attention(q, k, v, scale=None)
    # 手动指定 scale
    manual_scale = 1.0 / (head_dim**0.5)
    out_manual = scaled_dot_product_attention(q, k, v, scale=manual_scale)

    cos_sim = _cosine_similarity(out_auto, out_manual)
    print(f"\n[auto_scale] 余弦相似度: {cos_sim:.6f}")
    assert cos_sim >= 0.99999, f"scale=None 与手动 scale 结果不一致，余弦相似度 {cos_sim:.6f}"


# ── enable_gqa=True 报错测试 ─────────────────────────────────────────────


def test_sdpa_enable_gqa_error():
    """测试 enable_gqa=True 时抛出错误。"""
    q = torch.randn(1, 8, 16, 64)
    k = torch.randn(1, 8, 16, 64)
    v = torch.randn(1, 8, 16, 64)

    with pytest.raises(RuntimeError, match="enable_gqa"):
        scaled_dot_product_attention(q, k, v, enable_gqa=True)


# ── C++ 扩展可用性检查 ───────────────────────────────────────────────────


def test_cpp_sdpa_available():
    """验证 C++ SDPA 扩展已正确加载。"""
    assert _HAS_CPP_SDPA, "C++ SDPA 扩展未加载"
