#pragma once

#include <torch/extension.h>

namespace fused_cpp::deepseek_v4 {

bool q_norm_rope_fused_sve(const at::Tensor& q,
                           const at::Tensor& positions_long,
                           const at::Tensor& cos_sin_f,
                           double eps);

bool kv_rope_fused_sve(const at::Tensor& kv,
                       const at::Tensor& positions_long,
                       const at::Tensor& cos_sin_f);

bool kv_rope_cache_insert_fused_sve(const at::Tensor& kv,
                                    const at::Tensor& swa_kv_cache,
                                    const at::Tensor& slot_mapping_long,
                                    const at::Tensor& positions_long,
                                    const at::Tensor& cos_sin_f);

bool indexer_q_rope_fused_sve(const at::Tensor& q,
                              const at::Tensor& positions_long,
                              const at::Tensor& cos_sin_f);

}  // namespace fused_cpp::deepseek_v4
