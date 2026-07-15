#include <torch/extension.h>

at::Tensor merge_attn_states(at::Tensor prefix_output, at::Tensor prefix_lse, at::Tensor suffix_output,
                             at::Tensor suffix_lse) {
  torch::NoGradGuard no_grad;

  // Input validation
  TORCH_CHECK(prefix_output.dim() == 3, "merge_attn_states: prefix_output must be 3-D [T, H, D], got ",
              prefix_output.dim(), "-D");
  TORCH_CHECK(suffix_output.dim() == 3, "merge_attn_states: suffix_output must be 3-D [T, H, D], got ",
              suffix_output.dim(), "-D");
  TORCH_CHECK(prefix_lse.dim() == 2, "merge_attn_states: prefix_lse must be 2-D [H, T], got ", prefix_lse.dim(), "-D");
  TORCH_CHECK(suffix_lse.dim() == 2, "merge_attn_states: suffix_lse must be 2-D [H, T], got ", suffix_lse.dim(), "-D");

  auto T = prefix_output.size(0);
  auto H = prefix_output.size(1);
  auto D = prefix_output.size(2);

  TORCH_CHECK(suffix_output.size(0) == T && suffix_output.size(1) == H && suffix_output.size(2) == D,
              "merge_attn_states: suffix_output shape must match prefix_output");
  TORCH_CHECK(prefix_lse.size(0) == H && prefix_lse.size(1) == T, "merge_attn_states: prefix_lse shape must be [H, T]");
  TORCH_CHECK(suffix_lse.size(0) == H && suffix_lse.size(1) == T, "merge_attn_states: suffix_lse shape must be [H, T]");

  auto orig_dtype = prefix_output.scalar_type();

  // Transpose LSE from [H, T] to [T, H, 1] and convert to float32
  auto p_lse = prefix_lse.transpose(0, 1).unsqueeze(-1).to(at::kFloat);
  auto s_lse = suffix_lse.transpose(0, 1).unsqueeze(-1).to(at::kFloat);

  // Numerically stable LSE merge
  auto max_lse = at::maximum(p_lse, s_lse);
  auto p_se = at::exp(p_lse - max_lse);
  auto s_se = at::exp(s_lse - max_lse);
  auto out_se = p_se + s_se;

  auto p_scale = p_se / out_se;
  auto s_scale = s_se / out_se;

  // Merge in float32, cast back to input dtype
  auto merged = (p_scale * prefix_output.to(at::kFloat) + s_scale * suffix_output.to(at::kFloat)).to(orig_dtype);
  return merged;
}
