// ── flash2_neon_cache 微内核 validate / benchmark 历史入口 ─────────────────
//
// 历史接口（保留以兼容现有 tests/test_sdpa_flash2_neon_cache_microkernels.py
// 与 src/fused_cpp 侧潜在调用者）：
//
//   _C.validate_sdpa_flash2_neon_cache_microkernels(dtype, E, Sk)
//   _C.benchmark_sdpa_flash2_neon_cache_microkernels(dtype, E, Sk, iters, warmup)
//
// 行为：原先这两个入口直接走 `gemm_qkt_8x8 / gemm_pv_8x8` 全局函数。框架
// 升级后这些全局函数仍然被 `MK_Baseline` 包装，所以这里改为转发到
// 新的 `mk_validate_dispatch / mk_benchmark_dispatch`，固定 impl 名为
// "baseline"。新代码请直接使用 `_C.validate_microkernel(impl, dtype, ...)`
// 与 `_C.benchmark_microkernel(impl, dtype, ...)`，以选择不同 impl。

#include <torch/extension.h>

#include <map>
#include <string>

#include "mk_traits.h"

std::map<std::string, double> validate_sdpa_flash2_neon_cache_microkernels(std::string dtype, int64_t E, int64_t Sk) {
  return ::fused_cpp::sdpa_microkernels::mk_validate_dispatch("baseline", dtype, E, Sk);
}

std::map<std::string, double> benchmark_sdpa_flash2_neon_cache_microkernels(std::string dtype, int64_t E, int64_t Sk,
                                                                            int64_t iterations, int64_t warmup) {
  return ::fused_cpp::sdpa_microkernels::mk_benchmark_dispatch("baseline", dtype, E, Sk, iterations, warmup);
}
