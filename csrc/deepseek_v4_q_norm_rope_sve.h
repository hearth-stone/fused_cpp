/*
Copyright (c) Huawei Technologies Co., Ltd. 2023-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <torch/extension.h>

namespace fused_cpp::deepseek_v4 {

bool q_norm_rope_fused_sve(const at::Tensor& q, const at::Tensor& positions_long, const at::Tensor& cos_sin_f,
                           double eps);

bool kv_rope_fused_sve(const at::Tensor& kv, const at::Tensor& positions_long, const at::Tensor& cos_sin_f);

bool kv_rope_cache_insert_fused_sve(const at::Tensor& kv, const at::Tensor& swa_kv_cache,
                                    const at::Tensor& slot_mapping_long, const at::Tensor& positions_long,
                                    const at::Tensor& cos_sin_f);

bool indexer_q_rope_fused_sve(const at::Tensor& q, const at::Tensor& positions_long, const at::Tensor& cos_sin_f);

}  // namespace fused_cpp::deepseek_v4
