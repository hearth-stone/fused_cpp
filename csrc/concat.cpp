#include <torch/extension.h>

at::Tensor concat_k_nope_k_pe(at::Tensor k_nope, at::Tensor k_pe) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(k_nope.dim() == 3,
                "concat_k_nope_k_pe: k_nope must be 3-D [num_tokens, num_heads, qk_nope_head_dim]");
    TORCH_CHECK(k_pe.dim() == 3,
                "concat_k_nope_k_pe: k_pe must be 3-D [num_tokens, 1, qk_rope_head_dim]");
    TORCH_CHECK(k_nope.size(0) == k_pe.size(0),
                "concat_k_nope_k_pe: num_tokens mismatch");

    auto num_heads = k_nope.size(1);
    auto k_pe_expanded = k_pe.expand({-1, num_heads, -1});
    return at::cat({k_nope, k_pe_expanded}, /*dim=*/-1);
}
