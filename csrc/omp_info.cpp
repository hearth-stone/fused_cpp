// ── OpenMP 运行时探测 ───────────────────────────────────────────────────
//
// 提供给 Python 侧用于：
//   1. 在 benchmark / 测试中**真实记录**当前 OMP 配置（避免 Python 与 C++
//      视角不一致：例如 `os.environ["OMP_NUM_THREADS"]=N` 必须在 PyTorch
//      libomp 加载前设置才能生效，事后查询 env 不可靠 → 直接读 omp_* API）。
//   2. 跳过 Thread Sweep 等需要 OpenMP 的用例（macOS Apple Silicon 默认
//      Clang 不带 libomp）。
//
// 设计原则：
//   * **绝不**在 C++ 侧调用 omp_set_num_threads()，多线程数量完全由
//     OMP_NUM_THREADS 环境变量控制（需求 12.2）。
//   * 不实现 CPU 核心绑定（需求 12.1）。

#include <torch/extension.h>
#include <cstdlib>
#include <map>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline std::string env_or_null(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string("null");
}

}  // anonymous namespace

/// 返回当前 OpenMP 运行时配置快照。
///
/// 输出 dict 字段（与 Python 注册表一致）：
///   max_threads      omp_get_max_threads() 返回值（int 转 str）；编译期未
///                    启用 OpenMP 时记 ``"1"``。
///   num_procs        omp_get_num_procs() 返回值；同上。
///   has_openmp       ``"true"`` / ``"false"``，是否实际链接了 libomp。
///   OMP_NUM_THREADS  环境变量原始值，未设置记 ``"null"``。
///   OMP_SCHEDULE     同上。
///   OMP_PROC_BIND    同上。
///   OMP_PLACES       同上。
///   OMP_DYNAMIC      同上。
///   OMP_NESTED       同上。
///   OMP_WAIT_POLICY  同上。
std::map<std::string, std::string> get_omp_runtime_info() {
  std::map<std::string, std::string> out;
#ifdef _OPENMP
  out["max_threads"] = std::to_string(omp_get_max_threads());
  out["num_procs"] = std::to_string(omp_get_num_procs());
  out["has_openmp"] = "true";
#else
  out["max_threads"] = "1";
  out["num_procs"] = "1";
  out["has_openmp"] = "false";
#endif
  out["OMP_NUM_THREADS"] = env_or_null("OMP_NUM_THREADS");
  out["OMP_SCHEDULE"] = env_or_null("OMP_SCHEDULE");
  out["OMP_PROC_BIND"] = env_or_null("OMP_PROC_BIND");
  out["OMP_PLACES"] = env_or_null("OMP_PLACES");
  out["OMP_DYNAMIC"] = env_or_null("OMP_DYNAMIC");
  out["OMP_NESTED"] = env_or_null("OMP_NESTED");
  out["OMP_WAIT_POLICY"] = env_or_null("OMP_WAIT_POLICY");
  return out;
}

/// 是否实际链接了 OpenMP（编译期 ``_OPENMP`` 宏可见）。
bool has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}
