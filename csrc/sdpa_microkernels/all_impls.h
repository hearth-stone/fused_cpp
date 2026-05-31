#pragma once
// ── 所有微内核 impl 的总入口 ───────────────────────────────────────────
//
// 这个头文件不包含任何代码逻辑，只是把所有 impl 的头文件一起 include。
// 上层 SDPA 生成器（sdpa_flash2_neon_cache.cpp）和 Python 侧的 registry
// （mk_registry.cpp）都用它来获取所有 enabled impl 的类型。
//
// 新增 impl 的工作流（5 步以内）：
//
//   1. 在 csrc/sdpa_microkernels/impls/ 下新建 `mk_<your_name>.h`，
//      定义 `struct MK_<YourName>`，包含：
//        - constexpr const char* kName = "<your_name>";
//        - constexpr bool kEnabled = ... (建议靠 FUSED_CPP_MK_ENABLE_<NAME>
//          编译期宏控制，默认 1)；
//        - 5 组（×2 dtype）静态 inline 方法 qkt_8x8 / qkt_8x4 /
//          qkt_tail / pv_8x8 / pv_tail。
//      可参考 mk_baseline.h / mk_scalar.h 的现成签名。
//
//   2. 在本文件末尾添加 `#include "impls/mk_<your_name>.h"`。
//
//   3. 在 csrc/sdpa_microkernels/mk_registry.cpp 末尾追加：
//        REGISTER_MK_IF_ENABLED(MK_<YourName>);
//      该宏会自动调用 REGISTER_MICROKERNEL_IMPL，并把 SDPA 主路径绑定到
//      `flash2_neon_cache_<your_name>` 名字下注册到全局 SDPA 表。
//
//   4. （可选）在 src/fused_cpp/sdpa.py 的 _CPP_VERSION_META 里加一行
//      `"flash2_neon_cache_<your_name>": {...}`；不加也能用，仅影响
//      Python registry 的 description / tags 字段。
//
//   5. 重编 + 跑 tests/test_microkernel_framework.py：
//      新 impl 自动出现在测试矩阵里，对每个 op 跑正确性 + GFLOPS。

#include "impls/mk_baseline.h"
#include "impls/mk_scalar.h"
#include "impls/mk_pquad.h"
#include "impls/mk_qk_ublock4.h"
#include "impls/mk_qk_packk_full.h"
#include "impls/mk_qk_packk_inner.h"
#include "impls/mk_qk_packk_seq.h"
#include "impls/mk_qk_unroll2.h"
#include "impls/mk_qk_packqk_seq.h"
#include "impls/mk_qk_packqk_seq4.h"
#include "impls/mk_qk_packqk_seq4_ptr.h"
#include "impls/mk_qk_packqk_seq4_bmajor.h"
#include "impls/mk_qk_packqk_seq4_bmajor_pv_pquad.h"
#include "impls/mk_qk_packqk_seq4_pipe_a.h"
#include "impls/mk_qk_packqk_seq4_pipe_b.h"
