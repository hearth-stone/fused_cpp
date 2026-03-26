#include <torch/extension.h>
#include "utils.h"

// Forward declaration — gather_kv_cache is implemented in kv_cache.cpp
extern at::Tensor gather_kv_cache(at::Tensor kv_cache, at::Tensor block_table,
                                  int64_t seq_len, int64_t block_size);

py::object varlen_attention(at::Tensor q, at::Tensor k, at::Tensor v,
                            at::Tensor cu_seqlens_q, at::Tensor cu_seqlens_k,
                            int64_t max_seqlen_q, int64_t max_seqlen_k,
                            double scale, bool causal, bool return_softmax_lse) {
    torch::NoGradGuard no_grad;

    // Input validation
    TORCH_CHECK(q.dim() == 3,
                "varlen_attention: q must be 3-D [total_tokens, num_heads, qk_head_dim], got ",
                q.dim(), "-D");
    TORCH_CHECK(k.dim() == 3,
                "varlen_attention: k must be 3-D [total_tokens, num_heads, qk_head_dim], got ",
                k.dim(), "-D");
    TORCH_CHECK(v.dim() == 3,
                "varlen_attention: v must be 3-D [total_tokens, num_heads, v_head_dim], got ",
                v.dim(), "-D");
    TORCH_CHECK(cu_seqlens_q.dim() == 1,
                "varlen_attention: cu_seqlens_q must be 1-D");
    TORCH_CHECK(cu_seqlens_k.dim() == 1,
                "varlen_attention: cu_seqlens_k must be 1-D");
    TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0),
                "varlen_attention: cu_seqlens_q and cu_seqlens_k must have same length");
    TORCH_CHECK(cu_seqlens_q.size(0) >= 2,
                "varlen_attention: cu_seqlens must have at least 2 elements (batch_size >= 1)");
    TORCH_CHECK(q.size(1) == k.size(1),
                "varlen_attention: q and k num_heads mismatch (",
                q.size(1), " vs ", k.size(1), ")");
    TORCH_CHECK(q.size(2) == k.size(2),
                "varlen_attention: q and k head_dim mismatch (",
                q.size(2), " vs ", k.size(2), ")");
    TORCH_CHECK(v.size(1) == k.size(1),
                "varlen_attention: v and k num_heads mismatch (",
                v.size(1), " vs ", k.size(1), ")");

    auto batch_size = cu_seqlens_q.size(0) - 1;
    auto total_q_tokens = q.size(0);
    auto num_heads = q.size(1);
    auto v_head_dim = v.size(2);
    auto qk_head_dim = q.size(2);

    // Allocate output: [total_q_tokens, num_heads, v_head_dim]
    auto output = at::zeros({total_q_tokens, num_heads, v_head_dim},
                            q.options());

    // Optionally allocate LSE: [num_heads, total_q_tokens]
    at::Tensor lse;
    if (return_softmax_lse) {
        lse = at::full({num_heads, total_q_tokens},
                       -std::numeric_limits<float>::infinity(),
                       q.options().dtype(at::kFloat));
    }

    // Access cu_seqlens on CPU — normalize dtype
    auto cu_q_i32 = ensure_i32(cu_seqlens_q);
    auto cu_k_i32 = ensure_i32(cu_seqlens_k);
    auto cu_q_acc = cu_q_i32.accessor<int32_t, 1>();
    auto cu_k_acc = cu_k_i32.accessor<int32_t, 1>();

    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t q_start = cu_q_acc[i];
        int64_t q_end = cu_q_acc[i + 1];
        int64_t k_start = cu_k_acc[i];
        int64_t k_end = cu_k_acc[i + 1];

        if (q_end <= q_start || k_end <= k_start) {
            continue;
        }

        // q_i: [seq_q, num_heads, qk_head_dim] -> transpose(0,1) -> [num_heads, seq_q, qk_head_dim] -> unsqueeze(0) -> [1, num_heads, seq_q, qk_head_dim]
        auto q_i = q.slice(0, q_start, q_end).transpose(0, 1).unsqueeze(0);
        auto k_i = k.slice(0, k_start, k_end).transpose(0, 1).unsqueeze(0);
        auto v_i = v.slice(0, k_start, k_end).transpose(0, 1).unsqueeze(0);

        // If qk_head_dim != v_head_dim, pad v to match qk_head_dim
        if (qk_head_dim != v_head_dim) {
            v_i = at::constant_pad_nd(v_i, {0, qk_head_dim - v_head_dim}, 0.0);
        }

        at::Tensor output_i;

        if (return_softmax_lse) {
            // Manual attention with LSE computation
            // attn_scores: [1, num_heads, seq_q, seq_k]
            auto attn_scores = at::matmul(q_i, k_i.transpose(-2, -1)) * scale;

            if (causal) {
                int64_t seq_q = q_end - q_start;
                int64_t seq_k = k_end - k_start;
                // q_idx: [seq_q, 1], k_idx: [1, seq_k]
                auto q_idx = at::arange(seq_q, q.options().dtype(at::kLong)).unsqueeze(1);
                auto k_idx = at::arange(seq_k, q.options().dtype(at::kLong)).unsqueeze(0);
                // causal_mask: [seq_q, seq_k] -> [1, 1, seq_q, seq_k]
                auto causal_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0);
                // Fill masked positions with -inf
                attn_scores = at::where(causal_mask, attn_scores,
                                        at::full_like(attn_scores, -std::numeric_limits<float>::infinity()));
            }

            // lse_i: [1, num_heads, seq_q]
            auto lse_i = at::logsumexp(attn_scores, -1);
            // lse_i.squeeze(0) -> [num_heads, seq_q], write to lse[:, q_start:q_end]
            lse.slice(1, q_start, q_end) = lse_i.squeeze(0);

            auto attn_weights = at::softmax(attn_scores, -1);
            output_i = at::matmul(attn_weights, v_i);
        } else {
            // Use scaled_dot_product_attention
            // Signature: at::scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)
            output_i = at::scaled_dot_product_attention(
                q_i, k_i, v_i,
                /*attn_mask=*/{},
                /*dropout_p=*/0.0,
                /*is_causal=*/causal,
                /*scale=*/scale);
        }

        // output_i: [1, num_heads, seq_q, qk_head_dim] -> [0, :, :, :v_head_dim] -> [num_heads, seq_q, v_head_dim]
        // then transpose(0,1) -> [seq_q, num_heads, v_head_dim]
        output.slice(0, q_start, q_end) =
            output_i[0].slice(-1, 0, v_head_dim).transpose(0, 1);
    }

    if (return_softmax_lse) {
        return py::make_tuple(output, lse);
    }
    return py::cast(output);
}


at::Tensor forward_decode(at::Tensor q_nope_proj, at::Tensor q_pe,
                          at::Tensor kv_cache, at::Tensor block_table,
                          at::Tensor seq_lens, double scale,
                          int64_t kv_lora_rank, int64_t qk_rope_head_dim) {
    torch::NoGradGuard no_grad;

    // Input validation
    TORCH_CHECK(q_nope_proj.dim() == 3,
                "forward_decode: q_nope_proj must be 3-D [B, N, kv_lora_rank], got ",
                q_nope_proj.dim(), "-D");
    TORCH_CHECK(q_pe.dim() == 3,
                "forward_decode: q_pe must be 3-D [B, N, qk_rope_head_dim], got ",
                q_pe.dim(), "-D");
    TORCH_CHECK(kv_cache.dim() == 3,
                "forward_decode: kv_cache must be 3-D [num_blocks, block_size, head_size]");
    TORCH_CHECK(block_table.dim() == 2,
                "forward_decode: block_table must be 2-D [B, max_blocks]");
    TORCH_CHECK(seq_lens.dim() == 1,
                "forward_decode: seq_lens must be 1-D [B]");
    TORCH_CHECK(q_nope_proj.size(0) == q_pe.size(0),
                "forward_decode: q_nope_proj and q_pe batch size mismatch");
    TORCH_CHECK(q_nope_proj.size(1) == q_pe.size(1),
                "forward_decode: q_nope_proj and q_pe num_heads mismatch");

    auto batch_size = q_nope_proj.size(0);
    auto num_heads = q_nope_proj.size(1);
    auto block_size = kv_cache.size(1);

    // Allocate output zeros [batch_size, num_heads, kv_lora_rank]
    auto output = at::zeros({batch_size, num_heads, kv_lora_rank},
                            q_nope_proj.options());

    // Move seq_lens to CPU for iteration, normalize dtype
    auto seq_lens_i64 = ensure_i64(seq_lens);
    auto seq_lens_cpu = seq_lens_i64.cpu();
    auto seq_lens_acc = seq_lens_cpu.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch_size; ++b) {
        auto seq_len = seq_lens_acc[b];
        if (seq_len == 0) {
            continue;
        }

        // Gather KV cache for this sequence
        auto gathered_kv = gather_kv_cache(kv_cache, block_table[b],
                                           seq_len, block_size);

        // Split into kv_c [seq_len, kv_lora_rank] and k_pe_seq [seq_len, qk_rope_head_dim]
        auto kv_c = gathered_kv.slice(1, 0, kv_lora_rank);
        auto k_pe_seq = gathered_kv.slice(1, kv_lora_rank);

        // attn_scores = (q_nope_proj[b] @ kv_c.T + q_pe[b] @ k_pe_seq.T) * scale
        // q_nope_proj[b]: [num_heads, kv_lora_rank], kv_c.T: [kv_lora_rank, seq_len]
        // q_pe[b]: [num_heads, qk_rope_head_dim], k_pe_seq.T: [qk_rope_head_dim, seq_len]
        auto attn_scores = (at::matmul(q_nope_proj[b], kv_c.t())
                            + at::matmul(q_pe[b], k_pe_seq.t())) * scale;

        // attn_weights = softmax(attn_scores, dim=-1)
        auto attn_weights = at::softmax(attn_scores, /*dim=*/-1);

        // output[b] = attn_weights @ kv_c
        output[b] = at::matmul(attn_weights, kv_c);
    }

    return output;
}
