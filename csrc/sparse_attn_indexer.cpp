#include <torch/extension.h>

#include <cstdint>

namespace py = pybind11;

at::Tensor sparse_attn_indexer_prefill_cpp_v0(at::Tensor /*q_quant*/, at::Tensor /*weights*/, at::Tensor /*kv_cache*/,
                                              at::Tensor /*topk_indices_buffer*/, int64_t /*topk_tokens*/,
                                              py::object /*attn_metadata*/) {
  TORCH_CHECK(false,
              "sparse_attn_indexer_prefill_cpp_v0 is deprecated and must not be used. "
              "Use the Python/Torch sparse indexer baseline or "
              "deepseek_v4_post_gemm_parallel_stage for the migrated native path.");
  return at::Tensor();
}
