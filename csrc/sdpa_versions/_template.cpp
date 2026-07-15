// ── SDPA 多版本内核新增模板 ─────────────────────────────────────────────
//
// 用法：拷贝本文件改名为 `csrc/sdpa_<your_name>.cpp`，按以下 5 步操作即可
//      让新内核自动接入：
//        (1) 等价性测试矩阵（tests/test_sdpa_versions_equiv.py）；
//        (2) pytest-benchmark 基准（tests/bench_sdpa_versions.py）；
//        (3) GFLOP/s 报告。
//
// 注意：本文件自身**不**会被 setuptools 编译（setup.py 仅 glob csrc/*.cpp，
//       本文件位于子目录 csrc/sdpa_versions/），仅作为模板存在。
//       要启用新内核，请把 sdpa_<your_name>.cpp 直接放在 csrc/ 一级目录下。
//
// =============== 5 步清单（≤ 5 步） ===============
//
// (a) 拷贝模板文件并改名为 `csrc/sdpa_<your_name>.cpp`。
//
// (b) 实现 `void sdpa_<your_name>_impl(const SdpaParams& p);`：
//       - 必须严格遵循 `sdpa_common.h` 中的统一签名；
//       - 内部按 `p.dtype` 在 fp32 / bf16 模板特化间分发；
//       - 输出累加缓冲必须是 fp32（写入 `p.out_ptr`）。
//
// (c) 在文件末尾加一行：
//       REGISTER_SDPA_VERSION("<your_name>", sdpa_<your_name>_impl);
//     该宏在静态初始化阶段把内核注册到全局表，dispatcher 自动识别。
//
// (d) （**自动**）`fused_cpp/sdpa.py` 的 `_register_cpp_versions()` 在 import
//     时会调用 `_C.list_sdpa_versions()`，把全部 C++ 内核名注册到 Python
//     注册表，标记 `source="cpp"`。**无需**手动改 Python 代码。
//     若需要为新版本声明特殊的能力位（例如 supports_attn_mask=False），
//     在 `_CPP_VERSION_META` 字典中加一条即可。
//
// (e) （**自动**）`tests/test_sdpa_versions_equiv.py` 与
//     `tests/bench_sdpa_versions.py` 的 `version` 维度通过
//     `pytest_generate_tests` 钩子从注册表读取，新版本会**自动**纳入测试
//     与基准矩阵。**无需**修改任何测试文件。
//
// =============== 模板代码 ===============

#include <torch/extension.h>
#include <cmath>
#include <limits>
#include "../sdpa_common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

template <typename scalar_t>
inline void sdpa_template_kernel_tmpl(const scalar_t* q_ptr, const scalar_t* k_ptr, const scalar_t* v_ptr,
                                      const SdpaParams& p) {
  // 你的算法实现：以 (b, n, l) 为单位计算 attention，将结果写入 p.out_ptr。
  //
  // 注意：
  //   - 必须支持 p.is_causal（causal_offset = S - L 语义）；
  //   - 必须支持 MLA 形状（p.E != p.Ev）；
  //   - 严格在 fp32 累加；最终 cast 由调度层处理（在 sdpa_versions.cpp）。
  //
  // 如果你的内核**不**支持某些能力（例如不支持 causal），请在
  // src/fused_cpp/sdpa.py 的 _CPP_VERSION_META 中给该 version 设置相应
  // 能力位为 False；测试 / benchmark 会自动 skip 不兼容的组合。

  (void)q_ptr;
  (void)k_ptr;
  (void)v_ptr;
  (void)p;  // suppress unused warnings
}

}  // anonymous namespace

void sdpa_template_impl(const SdpaParams& p) {
  if (p.dtype == SdpaDtype::kBFloat16) {
    sdpa_template_kernel_tmpl<at::BFloat16>(static_cast<const at::BFloat16*>(p.q_ptr),
                                            static_cast<const at::BFloat16*>(p.k_ptr),
                                            static_cast<const at::BFloat16*>(p.v_ptr), p);
  } else {
    sdpa_template_kernel_tmpl<float>(static_cast<const float*>(p.q_ptr), static_cast<const float*>(p.k_ptr),
                                     static_cast<const float*>(p.v_ptr), p);
  }
}

// 取消下面这一行注释后，本模板会被注册（请改为你的真实版本名）。
// REGISTER_SDPA_VERSION("template", sdpa_template_impl);
