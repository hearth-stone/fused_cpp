#include <torch/extension.h>

at::Tensor rms_norm(at::Tensor x, at::Tensor weight, double eps) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(x.dim() >= 1, "rms_norm: x must have at least 1 dimension");
    TORCH_CHECK(weight.dim() == 1, "rms_norm: weight must be 1-D");
    TORCH_CHECK(x.size(-1) == weight.size(0),
                "rms_norm: x last dim (", x.size(-1),
                ") must match weight size (", weight.size(0), ")");
    TORCH_CHECK(eps > 0, "rms_norm: eps must be positive");

    auto orig_dtype = x.scalar_type();
    auto x_f32 = x.to(at::kFloat);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    x_f32 = x_f32 * at::rsqrt(variance + eps);
    return x_f32.to(orig_dtype) * weight;
}
