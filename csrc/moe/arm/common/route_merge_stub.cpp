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

// SPDX-License-Identifier: Apache-2.0
#include "../../common/route_merge.h"

#include <stdexcept>

#if !defined(FUSED_CPP_MOE_HAS_ARM_SVE)
namespace fused_cpp::moe_route_merge {

bool sve_available() { return false; }

void merge_f32_sve(const float*, const float*, uint16_t*, int64_t, int64_t, int64_t, int64_t) {
  throw std::runtime_error("SVE route merge is unavailable in this build");
}

void merge_bf16_sve(const uint16_t*, const float*, uint16_t*, int64_t, int64_t, int64_t, int64_t) {
  throw std::runtime_error("SVE route merge is unavailable in this build");
}

}  // namespace fused_cpp::moe_route_merge
#endif
