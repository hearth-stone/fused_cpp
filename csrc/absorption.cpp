#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> build_absorption_matrices(
    at::Tensor kv_b_proj_weight,
    int64_t num_heads,
    int64_t qk_nope_head_dim,
    int64_t v_head_dim,
    int64_t kv_lora_rank,
    at::ScalarType dtype) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(kv_b_proj_weight.dim() == 2,
                "build_absorption_matrices: kv_b_proj_weight must be 2-D "
                "[out_features, kv_lora_rank]");
    TORCH_CHECK(num_heads > 0, "build_absorption_matrices: num_heads must be positive");
    TORCH_CHECK(qk_nope_head_dim > 0, "build_absorption_matrices: qk_nope_head_dim must be positive");
    TORCH_CHECK(v_head_dim > 0, "build_absorption_matrices: v_head_dim must be positive");
    TORCH_CHECK(kv_lora_rank > 0, "build_absorption_matrices: kv_lora_rank must be positive");

    int64_t expected_out = num_heads * (qk_nope_head_dim + v_head_dim);
    TORCH_CHECK(kv_b_proj_weight.size(0) == expected_out,
                "build_absorption_matrices: kv_b_proj_weight rows (", kv_b_proj_weight.size(0),
                ") must equal num_heads * (qk_nope_head_dim + v_head_dim) = ", expected_out);
    TORCH_CHECK(kv_b_proj_weight.size(1) == kv_lora_rank,
                "build_absorption_matrices: kv_b_proj_weight cols (", kv_b_proj_weight.size(1),
                ") must equal kv_lora_rank (", kv_lora_rank, ")");

    // kv_b_proj_weight: [out_features, kv_lora_rank]
    // Transpose to: [kv_lora_rank, out_features], then cast to dtype
    auto w = kv_b_proj_weight.to(dtype).t();
    // w: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]

    // Reshape: [kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim]
    w = w.view({kv_lora_rank, num_heads, qk_nope_head_dim + v_head_dim});

    // Split along last dim
    auto splits = w.split({qk_nope_head_dim, v_head_dim}, /*dim=*/-1);
    auto w_uk = splits[0];  // [kv_lora_rank, num_heads, qk_nope_head_dim]
    auto w_uv = splits[1];  // [kv_lora_rank, num_heads, v_head_dim]

    // W_UK_T: [num_heads, qk_nope_head_dim, kv_lora_rank]
    auto W_UK_T = w_uk.permute({1, 2, 0}).contiguous();
    // W_UV: [num_heads, kv_lora_rank, v_head_dim]
    auto W_UV = w_uv.transpose(0, 1).contiguous();

    return std::make_tuple(W_UK_T, W_UV);
}
