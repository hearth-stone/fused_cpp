#include <torch/extension.h>

at::Tensor apply_rope(at::Tensor x, at::Tensor cos_sin_cache, at::Tensor positions, bool is_neox_style) {
  torch::NoGradGuard no_grad;

  TORCH_CHECK(x.dim() == 3, "apply_rope: x must be 3-D [num_tokens, num_heads, rot_dim], got dim=", x.dim());
  TORCH_CHECK(cos_sin_cache.dim() == 2, "apply_rope: cos_sin_cache must be 2-D [max_pos, rot_dim]");
  TORCH_CHECK(positions.dim() == 1, "apply_rope: positions must be 1-D [num_tokens]");
  TORCH_CHECK(x.size(0) == positions.size(0), "apply_rope: x num_tokens (", x.size(0), ") must match positions size (",
              positions.size(0), ")");

  auto rot_dim = x.size(-1);
  TORCH_CHECK(rot_dim % 2 == 0, "apply_rope: rot_dim must be even, got ", rot_dim);
  TORCH_CHECK(cos_sin_cache.size(-1) == rot_dim, "apply_rope: cos_sin_cache last dim (", cos_sin_cache.size(-1),
              ") must match x rot_dim (", rot_dim, ")");

  // cos_sin = cos_sin_cache[positions]  -> [num_tokens, rot_dim]
  auto cos_sin = cos_sin_cache.index_select(0, positions);
  // chunk into cos and sin, each [num_tokens, rot_dim/2]
  auto chunks = cos_sin.chunk(2, /*dim=*/-1);
  auto cos_vals = chunks[0].unsqueeze(-2).to(x.scalar_type());
  auto sin_vals = chunks[1].unsqueeze(-2).to(x.scalar_type());

  if (is_neox_style) {
    // NeoX style: first/second half split
    auto x_chunks = x.chunk(2, /*dim=*/-1);
    auto x1 = x_chunks[0];
    auto x2 = x_chunks[1];
    auto o1 = x1 * cos_vals - x2 * sin_vals;
    auto o2 = x2 * cos_vals + x1 * sin_vals;
    return at::cat({o1, o2}, /*dim=*/-1);
  } else {
    // GPT-J style: even/odd interleaved split
    auto x1 = x.slice(/*dim=*/-1, /*start=*/0, /*end=*/c10::nullopt, /*step=*/2);
    auto x2 = x.slice(/*dim=*/-1, /*start=*/1, /*end=*/c10::nullopt, /*step=*/2);
    auto o1 = x1 * cos_vals - x2 * sin_vals;
    auto o2 = x2 * cos_vals + x1 * sin_vals;
    return at::stack({o1, o2}, /*dim=*/-1).flatten(-2);
  }
}
