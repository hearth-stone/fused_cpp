// ── 微内核管理框架：registry 实现 ───────────────────────────────────────
//
// 持有 enabled MK impl 的 dispatch 表，对外暴露：
//   * mk_list_impls()                            — 列出所有 enabled impl 名
//   * mk_validate_dispatch(impl, dtype, E, Sk)   — 单 impl 的正确性单测
//   * mk_benchmark_dispatch(impl, dtype, E, Sk, iters, warmup) — 单 impl 的
//     GFLOPS 基准
//
// 该路径只供 Python 侧的 list/validate/benchmark API 使用。SDPA 主路径
// 通过 `flash2_neon_cache_<impl>` 名字走 SdpaRegistry，不会经过本表。
//
// REGISTER_MK_IF_ENABLED(MK_X) 宏负责：
//   * 在 MK_X::kEnabled 为 true 时，显式实例化 fp32/bf16 的 validate +
//     benchmark 模板（已在 mk_registry_helpers.h 中定义）；
//   * 调用 mk_register_impl(...) 把 entry 注册进表。

#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mk_traits.h"
#include "mk_registry_helpers.h"
#include "all_impls.h"

namespace fused_cpp::sdpa_microkernels {

namespace {

struct Registry {
  std::unordered_map<std::string, MicrokernelEntry> map;
  std::vector<std::string> names;
};

Registry& registry() {
  static Registry inst;
  return inst;
}

}  // anonymous namespace

int mk_register_impl(const MicrokernelEntry& entry) {
  auto& reg = registry();
  std::string key(entry.name);
  if (reg.map.find(key) != reg.map.end()) {
    throw std::runtime_error(
        std::string("mk_register_impl: duplicate impl name '") + key + "'");
  }
  reg.map.emplace(key, entry);
  reg.names.emplace_back(std::move(key));
  return 0;
}

std::vector<std::string> mk_list_impls() {
  return registry().names;
}

const MicrokernelEntry* mk_find_impl(const std::string& name) {
  auto& reg = registry();
  auto it = reg.map.find(name);
  return (it == reg.map.end()) ? nullptr : &it->second;
}

namespace {

const MicrokernelEntry& must_find(const std::string& name) {
  const MicrokernelEntry* e = mk_find_impl(name);
  if (e == nullptr) {
    std::string msg = "Microkernel impl '" + name +
                      "' is not registered; available: [";
    auto& reg = registry();
    for (size_t i = 0; i < reg.names.size(); ++i) {
      if (i) msg += ", ";
      msg += "'" + reg.names[i] + "'";
    }
    msg += "]";
    throw std::runtime_error(msg);
  }
  return *e;
}

bool dtype_is_bf16(const std::string& dtype) {
  return dtype == "bf16" || dtype == "bfloat16";
}
bool dtype_is_fp32(const std::string& dtype) {
  return dtype == "fp32" || dtype == "float32" || dtype == "float";
}

}  // anonymous namespace

std::map<std::string, double> mk_validate_dispatch(
    const std::string& impl_name,
    const std::string& dtype,
    int64_t E,
    int64_t Sk) {
  TORCH_CHECK(E > 0, "mk_validate: E must be > 0");
  TORCH_CHECK(Sk > 0, "mk_validate: Sk must be > 0");
  const MicrokernelEntry& entry = must_find(impl_name);
  if (dtype_is_bf16(dtype)) {
    return entry.validate_bf16(E, Sk);
  }
  if (dtype_is_fp32(dtype)) {
    return entry.validate_fp32(E, Sk);
  }
  TORCH_CHECK(false,
              "mk_validate: dtype must be 'bf16'/'bfloat16' or "
              "'fp32'/'float32'/'float', got '", dtype, "'");
}

std::map<std::string, double> mk_benchmark_dispatch(
    const std::string& impl_name,
    const std::string& dtype,
    int64_t E,
    int64_t Sk,
    int64_t iterations,
    int64_t warmup) {
  TORCH_CHECK(E > 0, "mk_benchmark: E must be > 0");
  TORCH_CHECK(Sk > 0, "mk_benchmark: Sk must be > 0");
  TORCH_CHECK(iterations > 0, "mk_benchmark: iterations must be > 0");
  TORCH_CHECK(warmup >= 0, "mk_benchmark: warmup must be >= 0");
  const MicrokernelEntry& entry = must_find(impl_name);
  if (dtype_is_bf16(dtype)) {
    return entry.benchmark_bf16(E, Sk, iterations, warmup);
  }
  if (dtype_is_fp32(dtype)) {
    return entry.benchmark_fp32(E, Sk, iterations, warmup);
  }
  TORCH_CHECK(false,
              "mk_benchmark: dtype must be 'bf16'/'bfloat16' or "
              "'fp32'/'float32'/'float', got '", dtype, "'");
}

// ── enabled impl 的注册（由本文件唯一持有，避免分散）──────────────────
//
// REGISTER_MK_IF_ENABLED 在 MK::kEnabled 为 true 时调用 REGISTER_MICROKERNEL_IMPL；
// 否则什么都不做。SFINAE 比较干净；因为 if constexpr 在静态初始化器里也能用，
// 但 REGISTER_MICROKERNEL_IMPL 宏展开是文件作用域的 static 变量，所以这里
// 用预处理判断 + 直接调用更简单：每个 impl 的 .h 已经在 disabled 时
// 不定义 MK_X 类，因此本文件只在 enabled 时才 #include 到那个类型。
//
// 不过为了让「禁用某个 impl 时本文件依然能编译」，我们用 has_enabled SFINAE
// 探测：若 MK 类型存在 + kEnabled=true 才注册。

#define REGISTER_MK_IF_ENABLED(MK_TYPE)                                     \
  namespace { static int MK_REG_CONCAT(_mk_cond_reg_, __COUNTER__) = []() { \
    if constexpr (mk_is_enabled_v<MK_TYPE>) {                               \
      mk_register_impl(MicrokernelEntry{                                    \
          MK_TYPE::kName,                                                   \
          &validate_microkernels_tmpl<MK_TYPE, float>,                      \
          &validate_microkernels_tmpl<MK_TYPE, at::BFloat16>,               \
          &benchmark_microkernels_tmpl<MK_TYPE, float>,                     \
          &benchmark_microkernels_tmpl<MK_TYPE, at::BFloat16>,              \
      });                                                                   \
    }                                                                       \
    return 0;                                                               \
  }(); }

#if FUSED_CPP_MK_ENABLE_BASELINE
REGISTER_MK_IF_ENABLED(MK_Baseline);
#endif
#if FUSED_CPP_MK_ENABLE_SCALAR
REGISTER_MK_IF_ENABLED(MK_Scalar);
#endif
#if FUSED_CPP_MK_ENABLE_PQUAD
REGISTER_MK_IF_ENABLED(MK_PQuad);
#endif
#if FUSED_CPP_MK_ENABLE_QK_UBLOCK4
REGISTER_MK_IF_ENABLED(MK_QkUblock4);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKK_FULL
REGISTER_MK_IF_ENABLED(MK_QkPackkFull);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKK_INNER
REGISTER_MK_IF_ENABLED(MK_QkPackkInner);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKK_SEQ
REGISTER_MK_IF_ENABLED(MK_QkPackkSeq);
#endif
#if FUSED_CPP_MK_ENABLE_QK_UNROLL2
REGISTER_MK_IF_ENABLED(MK_QkUnroll2);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq4);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PTR
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq4Ptr);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq4Bmajor);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq4BmajorPvPquad);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PIPE_A
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq4PipeA);
#endif
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_PIPE_B
REGISTER_MK_IF_ENABLED(MK_QkPackqkSeq4PipeB);
#endif

}  // namespace fused_cpp::sdpa_microkernels

// ── 给 module.cpp 暴露的 C++ 函数（dtype-erased，按字符串）──────────────

std::vector<std::string> list_microkernel_impls() {
  return ::fused_cpp::sdpa_microkernels::mk_list_impls();
}

std::map<std::string, double> validate_microkernel(
    std::string impl, std::string dtype, int64_t E, int64_t Sk) {
  return ::fused_cpp::sdpa_microkernels::mk_validate_dispatch(
      impl, dtype, E, Sk);
}

std::map<std::string, double> benchmark_microkernel(
    std::string impl, std::string dtype, int64_t E, int64_t Sk,
    int64_t iterations, int64_t warmup) {
  return ::fused_cpp::sdpa_microkernels::mk_benchmark_dispatch(
      impl, dtype, E, Sk, iterations, warmup);
}
