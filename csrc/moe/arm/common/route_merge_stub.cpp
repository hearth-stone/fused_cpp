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
