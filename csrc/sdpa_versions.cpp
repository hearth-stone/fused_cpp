// ── SDPA 多版本调度层 ────────────────────────────────────────────────────
//
// 本文件提供：
//   1. 全局版本注册表（实现 sdpa_register_version / sdpa_list_versions /
//      sdpa_find_kernel / sdpa_dispatch）。
//   2. 对外 PyTorch 接口 scaled_dot_product_attention_versioned，按版本名查表
//      分发；非法版本名抛出包含可用集合的 RuntimeError。
//   3. list_sdpa_versions() 供 Python 侧自动发现。
//
// 内核（naive / flash1 / flash2）通过 REGISTER_SDPA_VERSION 宏在各自的
// .cpp 文件中自注册，本文件不感知具体实现。

#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <string>

#include "utils.h"
#include "sdpa_common.h"

namespace {

// ── 注册表单例 ──
// 注：按注册顺序保留 names_，避免遍历 unordered_map 的不稳定顺序。
struct SdpaRegistry {
  std::unordered_map<std::string, SdpaKernelFn> map;
  std::vector<std::string> names;
};

SdpaRegistry& registry() {
  // Meyers singleton：保证全局静态初始化的线程安全
  static SdpaRegistry inst;
  return inst;
}

}  // anonymous namespace

// ── 注册接口实现 ───────────────────────────────────────────────────────
int sdpa_register_version(const char* name, SdpaKernelFn fn) {
  if (name == nullptr || fn == nullptr) {
    throw std::runtime_error("sdpa_register_version: name and fn must be non-null");
  }
  auto& reg = registry();
  std::string key(name);
  auto it = reg.map.find(key);
  if (it != reg.map.end()) {
    // 同 .so 中第二次注册同名内核 → 视为编程错误。
    throw std::runtime_error("sdpa_register_version: duplicate version name '" + key + "'");
  }
  reg.map.emplace(key, fn);
  reg.names.emplace_back(std::move(key));
  return 0;
}

std::vector<std::string> sdpa_list_versions() { return registry().names; }

SdpaKernelFn sdpa_find_kernel(const std::string& name) {
  auto& reg = registry();
  auto it = reg.map.find(name);
  if (it == reg.map.end()) return nullptr;
  return it->second;
}

void sdpa_dispatch(const std::string& name, const SdpaParams& p) {
  auto fn = sdpa_find_kernel(name);
  if (fn == nullptr) {
    std::string msg = "SDPA version '" + name + "' is not registered; available: [";
    const auto& names = registry().names;
    for (size_t i = 0; i < names.size(); ++i) {
      if (i) msg += ", ";
      msg += "'" + names[i] + "'";
    }
    msg += "]";
    throw std::runtime_error(msg);
  }
  fn(p);
}

// ── PyTorch 对外接口 ───────────────────────────────────────────────────

at::Tensor scaled_dot_product_attention_versioned(at::Tensor query, at::Tensor key, at::Tensor value,
                                                  c10::optional<at::Tensor> attn_mask, double dropout_p, bool is_causal,
                                                  c10::optional<double> scale, bool enable_gqa, std::string version) {
  torch::NoGradGuard no_grad;

  // ── 输入校验（与 sdpa.cpp 中默认入口保持一致） ──────────────────────
  TORCH_CHECK(query.dim() == 4, "sdpa_versioned: query must be 4-D [B, N, L, E], got ", query.dim(), "-D");
  TORCH_CHECK(key.dim() == 4, "sdpa_versioned: key must be 4-D [B, N, S, E], got ", key.dim(), "-D");
  TORCH_CHECK(value.dim() == 4, "sdpa_versioned: value must be 4-D [B, N, S, Ev], got ", value.dim(), "-D");
  TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0),
              "sdpa_versioned: batch size mismatch among Q/K/V (", query.size(0), ", ", key.size(0), ", ",
              value.size(0), ")");
  TORCH_CHECK(query.size(1) == key.size(1) && query.size(1) == value.size(1),
              "sdpa_versioned: num_heads mismatch among Q/K/V (", query.size(1), ", ", key.size(1), ", ", value.size(1),
              ")");
  TORCH_CHECK(key.size(2) == value.size(2), "sdpa_versioned: K and V seq_len mismatch (", key.size(2), " vs ",
              value.size(2), ")");
  TORCH_CHECK(query.size(3) == key.size(3), "sdpa_versioned: Q and K head_dim mismatch (", query.size(3), " vs ",
              key.size(3), ")");
  TORCH_CHECK(!enable_gqa, "sdpa_versioned: enable_gqa=true is not supported");

  if (dropout_p != 0.0) {
    TORCH_WARN("sdpa_versioned: dropout_p=", dropout_p, " is ignored (inference only)");
  }

  // ── 提前校验 version 合法性，给出清晰错误 ──
  auto kernel_fn = sdpa_find_kernel(version);
  if (kernel_fn == nullptr) {
    const auto names = sdpa_list_versions();
    std::string available;
    for (size_t i = 0; i < names.size(); ++i) {
      if (i) available += ", ";
      available += "'" + names[i] + "'";
    }
    TORCH_CHECK(false, "sdpa_versioned: version='", version, "' is not registered; available versions: [", available,
                "]");
  }

  // ── 维度信息 ──
  auto orig_dtype = query.scalar_type();
  auto q = ensure_contiguous(query);
  auto k = ensure_contiguous(key);
  auto v = ensure_contiguous(value);

  const int64_t B = q.size(0);
  const int64_t N = q.size(1);
  const int64_t L = q.size(2);
  const int64_t S = k.size(2);
  const int64_t E = q.size(3);
  const int64_t Ev = v.size(3);

  double scale_val = scale.has_value() ? scale.value() : 1.0 / std::sqrt(static_cast<double>(E));

  // ── 处理 dtype；fp32 / bf16 之外的输入提升到 fp32 ──
  SdpaDtype kernel_dtype;
  if (orig_dtype == at::kBFloat16) {
    kernel_dtype = SdpaDtype::kBFloat16;
  } else if (orig_dtype == at::kFloat) {
    kernel_dtype = SdpaDtype::kFloat32;
  } else {
    // 其它类型统一 cast 到 fp32
    q = q.to(at::kFloat);
    k = k.to(at::kFloat);
    v = v.to(at::kFloat);
    kernel_dtype = SdpaDtype::kFloat32;
  }

  // ── 处理可选 additive attention mask（统一 cast 到 fp32）──
  at::Tensor mask_fp32;
  const float* mask_ptr = nullptr;
  if (attn_mask.has_value()) {
    mask_fp32 = ensure_contiguous(attn_mask.value().to(at::kFloat));
    mask_ptr = mask_fp32.data_ptr<float>();
  }

  // ── 分配输出 fp32 缓冲 ──
  auto output_fp32 = at::empty({B, N, L, Ev}, q.options().dtype(at::kFloat));

  // ── 构建 SdpaParams ──
  SdpaParams p{};
  p.B = B;
  p.N = N;
  p.L = L;
  p.S = S;
  p.E = E;
  p.Ev = Ev;
  p.scale_f = static_cast<float>(scale_val);
  p.neg_inf = -std::numeric_limits<float>::infinity();
  p.causal_offset = S - L;
  p.is_causal = is_causal;
  p.dtype = kernel_dtype;
  p.q_ptr = q.data_ptr();
  p.k_ptr = k.data_ptr();
  p.v_ptr = v.data_ptr();
  p.mask_ptr = mask_ptr;
  p.out_ptr = output_fp32.data_ptr<float>();

  // ── 调度内核 ──
  kernel_fn(p);

  // ── 还原 dtype ──
  if (orig_dtype != at::kFloat) {
    return output_fp32.to(orig_dtype);
  }
  return output_fp32;
}

// ── 暴露给 Python：列出已注册版本 ───────────────────────────────────────
std::vector<std::string> list_sdpa_versions() { return sdpa_list_versions(); }
