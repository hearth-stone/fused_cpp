#pragma once
// ── SDPA 多版本框架公共头 ──────────────────────────────────────────────
//
// 本头文件定义所有 SDPA 内核共享的接口与注册机制：
//   * SdpaParams      —— 所有内核共用的参数结构体
//   * SdpaKernelFn    —— dtype-erased 内核函数指针类型
//   * REGISTER_SDPA_VERSION(name, fn) —— 静态注册宏
//   * sdpa_dispatch_*(...)            —— 内核分发与枚举接口
//
// 设计要点：
//   1. 所有内核拥有 **完全一致的 C++ 签名**：通过 dtype-erased 的
//      `SdpaKernelFn` 类型擦除 scalar_t，将 dtype 分发交给内核内部
//      （内核可在内部按 SdpaParams::dtype 再做模板特化）。
//   2. 新增内核时只需：
//        (a) 实现 `void my_kernel(const SdpaParams& p);`
//        (b) 在某个 .cpp 中加一行 `REGISTER_SDPA_VERSION("name", my_kernel);`
//      调度层无需修改任何 if-else 分支。
//   3. SdpaParams 中的 q/k/v 指针为 `const void*`；mask/out 指针为 `const float*`
//      / `float*`，因为输出始终在 fp32 累加，mask 始终先 cast 到 fp32。
//   4. 本头文件不依赖 PyTorch；具体的 at::Tensor 解析在
//      `scaled_dot_product_attention_versioned` 中完成后填充 SdpaParams。
//      这样能让单元测试在脱离 libtorch 时也能直接调用内核。

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ── SDPA dtype 标签 ──────────────────────────────────────────────────────
//
// 直接以整型 enum 表示，避免在头文件中引入 ATen 依赖。
// 取值与常用 ScalarType 一一对应，转换在 sdpa_versions.cpp 完成。
enum class SdpaDtype : int32_t {
    kFloat32  = 0,
    kBFloat16 = 1,
};

// ── SDPA 参数结构体 ──────────────────────────────────────────────────────
struct SdpaParams {
    // ── 形状 ──
    int64_t B   = 0;   ///< batch_size
    int64_t N   = 0;   ///< num_heads
    int64_t L   = 0;   ///< query seq_len
    int64_t S   = 0;   ///< key/value seq_len
    int64_t E   = 0;   ///< qk_head_dim
    int64_t Ev  = 0;   ///< v_head_dim

    // ── 标量/掩码控制 ──
    float scale_f       = 0.0f;             ///< 缩放因子
    float neg_inf       = 0.0f;             ///< -infinity 占位
    int64_t causal_offset = 0;              ///< S - L
    bool is_causal      = false;
    SdpaDtype dtype     = SdpaDtype::kFloat32;

    // ── 数据指针（dtype-erased）──
    // q_ptr / k_ptr / v_ptr 的实际类型由 dtype 决定：
    //   dtype == kFloat32  -> const float*
    //   dtype == kBFloat16 -> const at::BFloat16* (二进制等价于 uint16_t)
    const void* q_ptr   = nullptr;
    const void* k_ptr   = nullptr;
    const void* v_ptr   = nullptr;

    // mask 指针始终是 fp32（或 nullptr）
    const float* mask_ptr = nullptr;

    // 输出指针始终为 fp32 累加缓冲
    float* out_ptr      = nullptr;
};

// ── 内核函数指针 ──────────────────────────────────────────────────────────
using SdpaKernelFn = void (*)(const SdpaParams& p);

// ── 注册表对外接口 ────────────────────────────────────────────────────────

/// 注册一个 SDPA 内核。重名会触发 std::runtime_error；返回值为占位 int，
/// 仅用于让 REGISTER_SDPA_VERSION 宏能在文件作用域调用。
int sdpa_register_version(const char* name, SdpaKernelFn fn);

/// 列出所有已注册的版本名（按注册顺序）。
std::vector<std::string> sdpa_list_versions();

/// 按名查找内核；找不到时返回 nullptr。
SdpaKernelFn sdpa_find_kernel(const std::string& name);

/// 调用内核：根据 name 查表 + 参数有效性检查；查不到时抛 std::runtime_error。
void sdpa_dispatch(const std::string& name, const SdpaParams& p);

// ── 注册宏 ───────────────────────────────────────────────────────────────
//
// 用法：
//   void sdpa_my_impl(const SdpaParams& p) { ... }
//   REGISTER_SDPA_VERSION("my", sdpa_my_impl);
//
// 实现说明：依靠静态对象的初始化器在 .so 加载时自动调用
// sdpa_register_version()。`__COUNTER__` 保证同一文件内可多次注册不冲突。

#define SDPA_REG_CONCAT_INNER(a, b) a##b
#define SDPA_REG_CONCAT(a, b) SDPA_REG_CONCAT_INNER(a, b)

#define REGISTER_SDPA_VERSION(name, fn) \
    static int SDPA_REG_CONCAT(_sdpa_reg_, __COUNTER__) = \
        sdpa_register_version((name), (fn))
