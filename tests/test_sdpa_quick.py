# -*- coding: utf-8 -*-
"""快速验证脚本：FlashAttention 分块 SDPA 数值正确性。

用法：
    python fused_cpp/tests/test_sdpa_quick.py
"""
import torch
import torch.nn.functional as F

from fused_cpp.sdpa import scaled_dot_product_attention, _HAS_CPP_SDPA


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个张量的余弦相似度。"""
    return float(
        F.cosine_similarity(
            a.float().flatten().unsqueeze(0),
            b.float().flatten().unsqueeze(0),
        )
    )


def run_test(name, q, k, v, threshold=0.9999, **kwargs):
    """运行单个测试并打印结果。"""
    ref = F.scaled_dot_product_attention(q, k, v, **kwargs)
    out = scaled_dot_product_attention(q, k, v, **kwargs)
    cos = cosine_sim(out, ref)
    max_abs = float((out.float() - ref.float()).abs().max())
    status = "✅ PASS" if cos >= threshold else "❌ FAIL"
    print(f"  {status} {name}: cos={cos:.8f}, max_abs_err={max_abs:.6e} (threshold={threshold})")
    return cos >= threshold


def main():
    print(f"C++ SDPA 可用: {_HAS_CPP_SDPA}")
    if not _HAS_CPP_SDPA:
        print("⚠️  C++ 扩展未加载，测试将使用 PyTorch fallback（无意义）")
        return

    torch.manual_seed(42)
    all_passed = True

    print("\n=== 基本功能测试 ===")

    # 测试1: fp32 non-causal
    q = torch.randn(1, 8, 32, 64)
    k = torch.randn(1, 8, 32, 64)
    v = torch.randn(1, 8, 32, 64)
    all_passed &= run_test("fp32 non-causal", q, k, v)

    # 测试2: fp32 causal
    all_passed &= run_test("fp32 causal", q, k, v, is_causal=True)

    # 测试3: bf16 non-causal
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    all_passed &= run_test("bf16 non-causal", qb, kb, vb, threshold=0.999)

    # 测试4: bf16 causal
    all_passed &= run_test("bf16 causal", qb, kb, vb, threshold=0.999, is_causal=True)

    print("\n=== MLA 场景测试 (qk_dim=192, v_dim=128) ===")

    # 测试5: MLA fp32
    q5 = torch.randn(1, 16, 32, 192)
    k5 = torch.randn(1, 16, 32, 192)
    v5 = torch.randn(1, 16, 32, 128)
    all_passed &= run_test("MLA fp32 non-causal", q5, k5, v5)
    all_passed &= run_test("MLA fp32 causal", q5, k5, v5, is_causal=True)

    # 测试6: MLA bf16
    q5b, k5b, v5b = q5.bfloat16(), k5.bfloat16(), v5.bfloat16()
    all_passed &= run_test("MLA bf16 non-causal", q5b, k5b, v5b, threshold=0.999)

    print("\n=== Additive Attention Mask 测试 ===")

    # 测试7: attn_mask fp32
    mask = torch.randn(1, 8, 32, 32)
    all_passed &= run_test("attn_mask fp32", q, k, v, attn_mask=mask)

    # 测试8: attn_mask bf16
    all_passed &= run_test("attn_mask bf16", qb, kb, vb, threshold=0.999, attn_mask=mask.bfloat16())

    print("\n=== 边界情况测试 ===")

    # 测试9: S=1 (单 token)
    q9 = torch.randn(1, 8, 1, 64)
    k9 = torch.randn(1, 8, 1, 64)
    v9 = torch.randn(1, 8, 1, 64)
    all_passed &= run_test("S=1 单token", q9, k9, v9)

    # 测试10: L=1, S=512 (典型解码场景, causal)
    q10 = torch.randn(1, 8, 1, 64)
    k10 = torch.randn(1, 8, 512, 64)
    v10 = torch.randn(1, 8, 512, 64)
    all_passed &= run_test("L=1 S=512 causal (解码)", q10, k10, v10, is_causal=True)

    # 测试11: S < BLOCK_S (S=16 < 64)
    q11 = torch.randn(2, 4, 16, 64)
    k11 = torch.randn(2, 4, 16, 64)
    v11 = torch.randn(2, 4, 16, 64)
    all_passed &= run_test("S=16 < BLOCK_S", q11, k11, v11)

    # 测试12: S 不能被 BLOCK_S 整除 (S=100)
    q12 = torch.randn(1, 4, 100, 64)
    k12 = torch.randn(1, 4, 100, 64)
    v12 = torch.randn(1, 4, 100, 64)
    all_passed &= run_test("S=100 remainder block", q12, k12, v12, is_causal=True)

    # 测试13: 大 batch
    q13 = torch.randn(8, 16, 128, 64)
    k13 = torch.randn(8, 16, 128, 64)
    v13 = torch.randn(8, 16, 128, 64)
    all_passed &= run_test("B=8 N=16 L=128 fp32", q13, k13, v13, is_causal=True)

    # 测试14: 大 seq_len (S=2048)
    q14 = torch.randn(1, 8, 128, 64)
    k14 = torch.randn(1, 8, 2048, 64)
    v14 = torch.randn(1, 8, 2048, 64)
    all_passed &= run_test("L=128 S=2048 causal", q14, k14, v14, is_causal=True)

    print("\n=== scale 参数测试 ===")

    # 测试15: scale=None vs 手动 scale
    manual_scale = 1.0 / (64 ** 0.5)
    out_auto = scaled_dot_product_attention(q, k, v, scale=None)
    out_manual = scaled_dot_product_attention(q, k, v, scale=manual_scale)
    cos15 = cosine_sim(out_auto, out_manual)
    status15 = "✅ PASS" if cos15 >= 0.99999 else "❌ FAIL"
    print(f"  {status15} scale=None vs manual: cos={cos15:.8f}")
    all_passed &= (cos15 >= 0.99999)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！FlashAttention 分块实现数值正确。")
    else:
        print("⚠️  部分测试失败，请检查实现。")


if __name__ == "__main__":
    main()
