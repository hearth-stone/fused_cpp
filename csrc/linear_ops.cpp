#include <torch/extension.h>

at::Tensor fused_mla_linear(at::Tensor x, at::Tensor weight, c10::optional<at::Tensor> bias) {
  torch::NoGradGuard no_grad;

  TORCH_CHECK(x.dim() >= 1, "linear: x must have at least 1 dimension");
  TORCH_CHECK(weight.dim() == 2, "linear: weight must be 2-D");
  TORCH_CHECK(x.size(-1) == weight.size(1), "linear: x last dim (", x.size(-1), ") must match weight columns (",
              weight.size(1), ")");

  return at::linear(x, weight, bias);
}

// kv_b_proj_forward is semantically identical to linear — keep as
// a separate binding for API clarity, but delegate to avoid duplication.
at::Tensor kv_b_proj_forward(at::Tensor x, at::Tensor weight, c10::optional<at::Tensor> bias) {
  return fused_mla_linear(x, weight, bias);
}
