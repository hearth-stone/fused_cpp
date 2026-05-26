// ── L1-Resident SDPA Microkernel Throughput Benchmark (standalone) ─────────
//
// 与 fused_cpp/benchmarks/bench_microkernel_l1.py 等价的独立 C++ binary：
//   * 直接调 MK_Baseline / MK_Scalar 的 5 个静态方法
//   * L1 探测复用 csrc/sdpa_tile_sizes.h::effective_cache_bytes()
//   * 计时 / barrier / checksum 复用 csrc/sdpa_microkernels/mk_registry_helpers.h
//
// 与 PyTorch extension 入口的差异：
//   * 没有 Python 解释器启动开销，对 perf record / objdump / lldb 友好
//   * 但仍然链接 libtorch（c10 + torch_cpu）以拿到 at::BFloat16 类型
//     —— 微内核头声明就是 at::BFloat16，绕开成本太高（需要复制类型 shim）
//
// 用法：
//   make -C benchmarks
//   ./benchmarks/bench_microkernel_l1
//   ./benchmarks/bench_microkernel_l1 --dtype bf16 --target-frac 0.5
//   ./benchmarks/bench_microkernel_l1 --E 1024 --Sk 1024 --iters 200000

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "sdpa_microkernels/impls/mk_baseline.h"
#include "sdpa_microkernels/impls/mk_scalar.h"
#include "sdpa_microkernels/mk_registry_helpers.h"
#include "sdpa_tile_sizes.h"

namespace {

using ::fused_cpp::sdpa_microkernels::benchmark_microkernels_tmpl;
using ::fused_cpp::sdpa_tile_sizes::effective_cache_bytes;

struct CliOptions {
  std::string impl{"baseline"};
  std::string dtype{"bf16"};
  int64_t E{0};
  int64_t Sk{0};
  int64_t iters{100000};
  int64_t warmup{2000};
  double target_frac{0.5};
  int64_t l1_bytes_override{0};
};

void print_usage(const char* prog) {
  std::fprintf(
      stderr,
      "Usage: %s [options]\n"
      "  --impl <name>          baseline | scalar (default: baseline)\n"
      "  --dtype <name>         bf16 | fp32 (default: bf16)\n"
      "  --E <int>              head_dim E (0 = auto from L1)\n"
      "  --Sk <int>             Sk for pv_* (0 = auto from L1)\n"
      "  --iters <int>          benchmark iterations (default: 100000)\n"
      "  --warmup <int>         warmup iterations (default: 2000)\n"
      "  --target-frac <float>  working_set / L1d ratio (default: 0.5)\n"
      "  --l1-bytes <int>       override detected L1d size (0 = auto)\n"
      "  -h, --help             show this help message\n",
      prog);
}

bool parse_int64(const char* s, int64_t* out) {
  if (s == nullptr || *s == '\0') return false;
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (end == s || *end != '\0') return false;
  *out = static_cast<int64_t>(v);
  return true;
}

bool parse_double(const char* s, double* out) {
  if (s == nullptr || *s == '\0') return false;
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (end == s || *end != '\0') return false;
  *out = v;
  return true;
}

bool parse_args(int argc, char** argv, CliOptions* opts) {
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        return nullptr;
      }
      return argv[++i];
    };
    if (flag == "-h" || flag == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (flag == "--impl") {
      const char* v = need_value("--impl");
      if (!v) return false;
      opts->impl = v;
    } else if (flag == "--dtype") {
      const char* v = need_value("--dtype");
      if (!v) return false;
      opts->dtype = v;
    } else if (flag == "--E") {
      const char* v = need_value("--E");
      if (!v || !parse_int64(v, &opts->E)) return false;
    } else if (flag == "--Sk") {
      const char* v = need_value("--Sk");
      if (!v || !parse_int64(v, &opts->Sk)) return false;
    } else if (flag == "--iters") {
      const char* v = need_value("--iters");
      if (!v || !parse_int64(v, &opts->iters)) return false;
    } else if (flag == "--warmup") {
      const char* v = need_value("--warmup");
      if (!v || !parse_int64(v, &opts->warmup)) return false;
    } else if (flag == "--target-frac") {
      const char* v = need_value("--target-frac");
      if (!v || !parse_double(v, &opts->target_frac)) return false;
    } else if (flag == "--l1-bytes") {
      const char* v = need_value("--l1-bytes");
      if (!v || !parse_int64(v, &opts->l1_bytes_override)) return false;
    } else {
      std::fprintf(stderr, "unknown flag: %s\n", flag.c_str());
      return false;
    }
  }
  return true;
}

bool dtype_is_bf16(const std::string& s) {
  return s == "bf16" || s == "bfloat16";
}
bool dtype_is_fp32(const std::string& s) {
  return s == "fp32" || s == "float32" || s == "float";
}

int64_t derive_E(int64_t l1_bytes, int sizeof_elt, double target_frac) {
  // Working set: 16 · E · sizeof(elt). Align E down to multiple of 4
  // (BFMMLA stride along head_dim = 4).
  int64_t e_max = static_cast<int64_t>(
      target_frac * static_cast<double>(l1_bytes)
      / 16.0 / static_cast<double>(sizeof_elt));
  int64_t aligned = (e_max / 4) * 4;
  return aligned >= 8 ? aligned : 8;
}

int64_t derive_Sk(int64_t l1_bytes, int sizeof_elt, double target_frac) {
  // Working set: P̂[8 · Sk · 4] + V[Sk · 8 · sizeof(elt)]
  //             = Sk · (32 + 8 · sizeof(elt))
  const int64_t per_sk = 32 + 8 * sizeof_elt;
  int64_t sk_max = static_cast<int64_t>(
      target_frac * static_cast<double>(l1_bytes)
      / static_cast<double>(per_sk));
  int64_t aligned = (sk_max / 8) * 8;
  return aligned >= 8 ? aligned : 8;
}

void format_bytes(int64_t n, char* buf, size_t buflen) {
  if (n >= 1024 * 1024) {
    std::snprintf(buf, buflen, "%.2f MiB",
                  static_cast<double>(n) / (1024.0 * 1024.0));
  } else if (n >= 1024) {
    std::snprintf(buf, buflen, "%.2f KiB",
                  static_cast<double>(n) / 1024.0);
  } else {
    std::snprintf(buf, buflen, "%lld B", static_cast<long long>(n));
  }
}

template <class MK>
std::map<std::string, double> dispatch_one(
    const std::string& dtype, int64_t E, int64_t Sk,
    int64_t iters, int64_t warmup) {
  if (dtype_is_bf16(dtype)) {
    return benchmark_microkernels_tmpl<MK, at::BFloat16>(E, Sk, iters, warmup);
  }
  if (dtype_is_fp32(dtype)) {
    return benchmark_microkernels_tmpl<MK, float>(E, Sk, iters, warmup);
  }
  std::fprintf(stderr, "unknown dtype: %s\n", dtype.c_str());
  std::exit(2);
}

std::map<std::string, double> dispatch(
    const std::string& impl, const std::string& dtype,
    int64_t E, int64_t Sk, int64_t iters, int64_t warmup) {
#if FUSED_CPP_MK_ENABLE_BASELINE
  if (impl == "baseline") {
    return dispatch_one<::fused_cpp::sdpa_microkernels::MK_Baseline>(
        dtype, E, Sk, iters, warmup);
  }
#endif
#if FUSED_CPP_MK_ENABLE_SCALAR
  if (impl == "scalar") {
    return dispatch_one<::fused_cpp::sdpa_microkernels::MK_Scalar>(
        dtype, E, Sk, iters, warmup);
  }
#endif
  std::fprintf(stderr, "unknown impl: %s\n", impl.c_str());
  std::exit(2);
}

void print_table(const std::string& dtype, const std::string& impl,
                 int64_t E, int64_t Sk, int sizeof_elt, int64_t l1_bytes,
                 const std::map<std::string, double>& r) {
  const int64_t ws_qkt = 16 * E * sizeof_elt + 256;
  const int64_t ws_pv = (32 + 8 * sizeof_elt) * Sk + 256;
  char buf_q[32], buf_p[32];
  format_bytes(ws_qkt, buf_q, sizeof(buf_q));
  format_bytes(ws_pv, buf_p, sizeof(buf_p));
  std::printf(
      "=== dtype=%s  impl=%s  E=%lld  Sk=%lld\n"
      "    qkt_* working set = %s (%.1f%% of L1d)\n"
      "    pv_*  working set = %s (%.1f%% of L1d)\n",
      dtype.c_str(), impl.c_str(),
      static_cast<long long>(E), static_cast<long long>(Sk),
      buf_q, 100.0 * static_cast<double>(ws_qkt) / static_cast<double>(l1_bytes),
      buf_p, 100.0 * static_cast<double>(ws_pv) / static_cast<double>(l1_bytes));

  struct Row {
    const char* name;
    int M;
    int N;
    int64_t K;
    double us;
    double gflops;
  };
  auto get = [&](const std::string& k) -> double {
    auto it = r.find(k);
    return it == r.end() ? 0.0 : it->second;
  };
  Row rows[5] = {
      {"qkt_8x8",  8, 8, E,  get("qkt_8x8_us"),  get("qkt_8x8_gflops")},
      {"qkt_8x4",  8, 4, E,  get("qkt_8x4_us"),  get("qkt_8x4_gflops")},
      {"qkt_tail", 5, 3, E,  get("qkt_tail_us"), get("qkt_tail_gflops")},
      {"pv_8x8",   8, 8, Sk, get("pv_8x8_us"),   get("pv_8x8_gflops")},
      {"pv_tail",  5, 3, Sk, get("pv_tail_us"),  get("pv_tail_gflops")},
  };
  std::printf(
      "  %-10s%4s%4s%8s%14s%12s\n",
      "op", "M", "N", "K", "us/iter", "GFLOPS");
  for (const auto& row : rows) {
    std::printf("  %-10s%4d%4d%8lld%14.4f%12.2f\n",
                row.name, row.M, row.N,
                static_cast<long long>(row.K),
                row.us, row.gflops);
  }
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  CliOptions opts;
  if (!parse_args(argc, argv, &opts)) {
    print_usage(argv[0]);
    return 2;
  }

  const auto& cache_bytes = effective_cache_bytes();
  const int64_t l1_bytes = opts.l1_bytes_override > 0
                               ? opts.l1_bytes_override
                               : cache_bytes[0];
  const int sizeof_elt = dtype_is_bf16(opts.dtype) ? 2
                          : (dtype_is_fp32(opts.dtype) ? 4 : 0);
  if (sizeof_elt == 0) {
    std::fprintf(stderr, "unknown dtype '%s'; expected bf16 or fp32\n",
                 opts.dtype.c_str());
    return 2;
  }

  const int64_t E = opts.E > 0
                        ? opts.E
                        : derive_E(l1_bytes, sizeof_elt, opts.target_frac);
  const int64_t Sk = opts.Sk > 0
                         ? opts.Sk
                         : derive_Sk(l1_bytes, sizeof_elt, opts.target_frac);

  char l1_str[32];
  format_bytes(l1_bytes, l1_str, sizeof(l1_str));
  std::printf(
      "detected L1d = %lld bytes (%s)\n"
      "target_frac = %.2f\n"
      "iters = %lld, warmup = %lld\n\n",
      static_cast<long long>(l1_bytes), l1_str,
      opts.target_frac,
      static_cast<long long>(opts.iters),
      static_cast<long long>(opts.warmup));

  auto r = dispatch(opts.impl, opts.dtype, E, Sk, opts.iters, opts.warmup);
  print_table(opts.dtype, opts.impl, E, Sk, sizeof_elt, l1_bytes, r);
  return 0;
}
