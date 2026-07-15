// main.cpp
// CLI：./microkernel_min [test|bench|all] [--E=N] [--Sk=N] [--iters=N] [--warmup=N]
//
// test  正确性测试：5 个 (E, Sk) 形状，每个跑 QKᵀ 和 PV，对比 reference
// bench 性能测试：默认 E=192 Sk=128 iters=20000 warmup=1000，单线程，hot L1
// all   先 test 后 bench
//
// 平台亲和性：
//   * macOS：用 QOS_CLASS_USER_INTERACTIVE 倾向 P-core 调度（macOS 没有
//     真线程亲和性 API，QoS 是最接近的手段）。
//   * Linux：用 sched_setaffinity 绑定到 CPU 0（命令行可用 taskset 覆盖）。

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#endif

// ── 平台亲和性辅助 ──────────────────────────────────────────────────────
// 返回一段描述实际生效内容的字符串（用于打印）。
static std::string apply_affinity() {
#if defined(__APPLE__)
  // QOS_CLASS_USER_INTERACTIVE 强烈倾向 P-core；不保证，但比默认调度稳。
  int rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  if (rc == 0) return "macOS QoS=USER_INTERACTIVE (P-core preferred)";
  return std::string("macOS QoS set failed (rc=") + std::to_string(rc) + ")";
#elif defined(__linux__)
  // 默认绑 CPU 0；如果用户用 taskset 启动，本调用会覆盖 taskset，需要
  // 通过 MICROKERNEL_NO_PIN=1 环境变量关掉这里以让 taskset 生效。
  if (std::getenv("MICROKERNEL_NO_PIN") != nullptr) {
    return "Linux pinning skipped (MICROKERNEL_NO_PIN set)";
  }
  cpu_set_t cs;
  CPU_ZERO(&cs);
  CPU_SET(0, &cs);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
  if (rc == 0) return "Linux pinned to CPU 0";
  return std::string("Linux affinity set failed (rc=") + std::to_string(rc) + ")";
#else
  return "no affinity (unknown platform)";
#endif
}

#include "microkernels.h"
#include "reference.h"

// ── 工具：fp32 → bf16（高 16 位截断，与 PyTorch round-to-nearest 略有差，
//    但作为输入 fixture 完全够用）──────────────────────────────────────────
static inline uint16_t fp32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // 简单截断；不做 round-to-nearest-even（输入数据用，舍入误差 <1 ulp）
  return static_cast<uint16_t>(u >> 16);
}

// ── 工具：填充随机 bf16 矩阵 [-1, 1] ──
static void fill_random_bf16(uint16_t* dst, size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) dst[i] = fp32_to_bf16(dist(rng));
}
static void fill_random_fp32(float* dst, size_t n, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) dst[i] = dist(rng);
}

// ── 工具：max abs 误差与 max rel 误差 ──
struct DiffStat {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  int first_bad_i = -1;
  int first_bad_j = -1;
  float first_bad_actual = 0.0f;
  float first_bad_expect = 0.0f;
};

static DiffStat compare(const float* actual, const float* expect, int rows, int cols, float atol, float rtol) {
  DiffStat s;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float a = actual[i * cols + j];
      float e = expect[i * cols + j];
      float abs_err = std::fabs(a - e);
      float rel_err = abs_err / (std::fabs(e) + 1e-8f);
      if (abs_err > s.max_abs) s.max_abs = abs_err;
      if (rel_err > s.max_rel) s.max_rel = rel_err;
      if (s.first_bad_i < 0 && abs_err > atol && rel_err > rtol) {
        s.first_bad_i = i;
        s.first_bad_j = j;
        s.first_bad_actual = a;
        s.first_bad_expect = e;
      }
    }
  }
  return s;
}

// ── 测试单个 (E, Sk) 的 QKᵀ + PV ──
static int test_one(int64_t E, int64_t Sk, std::mt19937& rng, const char* label) {
  // QKᵀ：Q [8][E] bf16，K [8][E] bf16，scores [8][8] fp32
  std::vector<uint16_t> Q(8 * E), K(8 * E);
  std::vector<uint16_t> Q_seq((E / 4) * 32 + 32, 0);  // 多分配防越界
  std::vector<uint16_t> K_seq((E / 4) * 32 + 32, 0);
  std::vector<float> scores_actual(8 * 8, 0.0f);
  std::vector<float> scores_expect(8 * 8, 0.0f);
  fill_random_bf16(Q.data(), Q.size(), rng);
  fill_random_bf16(K.data(), K.size(), rng);
  pack_q_8rows_to_seq_bf16(Q.data(), E, E, Q_seq.data());
  pack_k_8rows_to_seq_bf16(K.data(), E, E, K_seq.data());

  const float scale = 1.0f / std::sqrt(static_cast<float>(E));
  gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner(Q_seq.data(), Q.data(), E, K_seq.data(), K.data(), E, E, scale,
                                                         scores_actual.data());
  qkt_ref(Q.data(), E, K.data(), E, E, scale, scores_expect.data());

  // bf16 精度：abs ~ 0.01 * sqrt(E)（每元素相对 ulp 1/256）
  // 用宽松阈值，重点是发现量级错误而非 ULP 精度
  const float atol_qkt = 0.05f;
  const float rtol_qkt = 0.05f;
  DiffStat ds_qkt = compare(scores_actual.data(), scores_expect.data(), 8, 8, atol_qkt, rtol_qkt);

  bool qkt_ok = (ds_qkt.first_bad_i < 0);
  std::printf("  %s QKᵀ E=%4lld         max_abs=%.4f max_rel=%.4f  %s\n", label, (long long)E, ds_qkt.max_abs,
              ds_qkt.max_rel, qkt_ok ? "PASS" : "FAIL");
  if (!qkt_ok) {
    std::printf("    first bad [%d,%d]: actual=%.6f expect=%.6f\n", ds_qkt.first_bad_i, ds_qkt.first_bad_j,
                ds_qkt.first_bad_actual, ds_qkt.first_bad_expect);
  }

  // PV：P_hat [8][Sk] fp32，V [Sk][8] bf16，O [8][8] fp32
  // 测两次：一次 O=0 起累加；一次 O 已有非零值确认是 += 而非 =
  std::vector<float> P(8 * Sk);
  std::vector<uint16_t> V(Sk * 8);
  fill_random_fp32(P.data(), P.size(), rng);
  fill_random_bf16(V.data(), V.size(), rng);

  std::vector<float> O_init(8 * 8);
  fill_random_fp32(O_init.data(), O_init.size(), rng);

  std::vector<float> O_actual = O_init;
  std::vector<float> O_expect = O_init;
  gemm_pv_microkernel_8x8_bf16_pquad(P.data(), Sk, V.data(), 8, Sk, O_actual.data(), 8);
  pv_ref(P.data(), Sk, V.data(), 8, Sk, O_expect.data(), 8);

  // PV 累加 Sk 项；bf16 V 引入 ~0.01 * Sk 累积误差（最坏估计）
  const float atol_pv = std::max(0.05f, 0.005f * static_cast<float>(Sk));
  const float rtol_pv = 0.05f;
  DiffStat ds_pv = compare(O_actual.data(), O_expect.data(), 8, 8, atol_pv, rtol_pv);
  bool pv_ok = (ds_pv.first_bad_i < 0);
  std::printf("  %s PV  Sk=%4lld        max_abs=%.4f max_rel=%.4f  %s\n", label, (long long)Sk, ds_pv.max_abs,
              ds_pv.max_rel, pv_ok ? "PASS" : "FAIL");
  if (!pv_ok) {
    std::printf("    first bad [%d,%d]: actual=%.6f expect=%.6f\n", ds_pv.first_bad_i, ds_pv.first_bad_j,
                ds_pv.first_bad_actual, ds_pv.first_bad_expect);
  }

  return (qkt_ok && pv_ok) ? 0 : 1;
}

static int cmd_test() {
  std::printf("=== Correctness test ===\n");
  std::mt19937 rng(42);
  struct Shape {
    int64_t E, Sk;
    const char* tag;
  };
  Shape shapes[] = {
      {64, 128, "[E=64  Sk=128]"},  {128, 128, "[E=128 Sk=128]"}, {192, 128, "[E=192 Sk=128]"},
      {192, 512, "[E=192 Sk=512]"}, {64, 8, "[E=64  Sk=8  ]"},
  };
  int fails = 0;
  for (const auto& s : shapes) {
    fails += test_one(s.E, s.Sk, rng, s.tag);
  }
  std::printf("=== %d failure(s) ===\n", fails);
  return fails == 0 ? 0 : 2;
}

// ── Bench ────────────────────────────────────────────────────────────────
// 用 asm volatile barrier 而不是 volatile sink：
//   * volatile sink += scores[0] 会引入额外 load + fp add，每次 iter ~3 LSU op，
//     在 hot L1 下显著扰动测量（实测 PV 偏低 ~16%）
//   * asm volatile("" : : "r"(p) : "memory") 是 0 cost barrier，仅阻止编译器
//     把 microkernel 调用 hoist 出循环；与仓库 bench (mk_bench_barrier)
//     完全一致

static inline void bench_barrier(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r"(p) : "memory");
#else
  (void)p;
#endif
}

static double bench_qkt(int64_t E, int iters, int warmup) {
  std::mt19937 rng(123);
  std::vector<uint16_t> Q(8 * E), K(8 * E);
  std::vector<uint16_t> Q_seq((E / 4) * 32 + 32, 0);
  std::vector<uint16_t> K_seq((E / 4) * 32 + 32, 0);
  std::vector<float> scores(8 * 8);
  fill_random_bf16(Q.data(), Q.size(), rng);
  fill_random_bf16(K.data(), K.size(), rng);
  pack_q_8rows_to_seq_bf16(Q.data(), E, E, Q_seq.data());
  pack_k_8rows_to_seq_bf16(K.data(), E, E, K_seq.data());
  const float scale = 1.0f / std::sqrt(static_cast<float>(E));

  // warmup
  for (int i = 0; i < warmup; ++i) {
    gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner(Q_seq.data(), Q.data(), E, K_seq.data(), K.data(), E, E,
                                                           scale, scores.data());
  }
  bench_barrier(scores.data());

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner(Q_seq.data(), Q.data(), E, K_seq.data(), K.data(), E, E,
                                                           scale, scores.data());
    bench_barrier(scores.data());
  }
  auto t1 = std::chrono::steady_clock::now();
  bench_barrier(scores.data());

  double secs = std::chrono::duration<double>(t1 - t0).count();
  // 每次 8x8 输出，内积长 E：FMA = 64 * E，FLOPS = 2 * 64 * E
  double flops_per_iter = 2.0 * 8.0 * 8.0 * static_cast<double>(E);
  return flops_per_iter * iters / secs / 1e9;
}

static double bench_pv(int64_t Sk, int iters, int warmup) {
  // 数据生成方式与仓库 mk_registry_helpers.h::benchmark_microkernels_tmpl 对齐：
  // p_hat / v 都用小幅值 deterministic 取模值，让 O 累加在 fp32 normal range
  // 内不会进入 denormal slow path。bench 的 GFLOPS 对输入数据敏感（denormal
  // 处理可能拖慢 FMA 数倍），保持一致才能公平对比。
  std::vector<float> P(8 * Sk);
  std::vector<uint16_t> V(Sk * 8);
  std::vector<float> O(8 * 8, 0.0f);
  for (int64_t i = 0; i < static_cast<int64_t>(V.size()); ++i)
    V[i] = fp32_to_bf16(0.015f * static_cast<float>((i % 9) + 1));
  for (int64_t i = 0; i < static_cast<int64_t>(P.size()); ++i) P[i] = 0.005f * static_cast<float>((i % 5) + 1);

  for (int i = 0; i < warmup; ++i) {
    gemm_pv_microkernel_8x8_bf16_pquad(P.data(), Sk, V.data(), 8, Sk, O.data(), 8);
  }
  bench_barrier(O.data());

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    gemm_pv_microkernel_8x8_bf16_pquad(P.data(), Sk, V.data(), 8, Sk, O.data(), 8);
    bench_barrier(O.data());
  }
  auto t1 = std::chrono::steady_clock::now();
  bench_barrier(O.data());

  double secs = std::chrono::duration<double>(t1 - t0).count();
  double flops_per_iter = 2.0 * 8.0 * 8.0 * static_cast<double>(Sk);
  return flops_per_iter * iters / secs / 1e9;
}

static int cmd_bench(int64_t E, int64_t Sk, int iters, int warmup) {
  std::printf("=== Benchmark (single-threaded, hot L1) ===\n");
  std::printf("  E=%lld  Sk=%lld  iters=%d  warmup=%d\n", (long long)E, (long long)Sk, iters, warmup);
  std::printf("  -------------------------------------------\n");

  double gflops_qkt = bench_qkt(E, iters, warmup);
  std::printf("  QKᵀ packqk_seq4_bmajor (E=%lld)         %8.2f GFLOPS\n", (long long)E, gflops_qkt);

  double gflops_pv = bench_pv(Sk, iters, warmup);
  std::printf("  PV  bf16_pquad           (Sk=%lld)        %8.2f GFLOPS\n", (long long)Sk, gflops_pv);

  std::printf("=== Done ===\n");
  return 0;
}

// ── CLI 解析 ─────────────────────────────────────────────────────────────
static bool parse_kv(const std::string& arg, const char* key, int64_t* out) {
  std::string p = std::string("--") + key + "=";
  if (arg.compare(0, p.size(), p) == 0) {
    *out = std::stoll(arg.substr(p.size()));
    return true;
  }
  return false;
}

int main(int argc, char** argv) {
  std::string mode = (argc >= 2) ? argv[1] : "all";

  int64_t E = 192, Sk = 128;
  int iters = 20000, warmup = 1000;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    int64_t v;
    if (parse_kv(arg, "E", &v)) {
      E = v;
      continue;
    }
    if (parse_kv(arg, "Sk", &v)) {
      Sk = v;
      continue;
    }
    if (parse_kv(arg, "iters", &v)) {
      iters = static_cast<int>(v);
      continue;
    }
    if (parse_kv(arg, "warmup", &v)) {
      warmup = static_cast<int>(v);
      continue;
    }
    std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str());
    return 1;
  }

  std::printf("microkernel_min  (HAS_BFMMLA=%d)\n", HAS_BFMMLA);
  std::printf("  affinity: %s\n", apply_affinity().c_str());

  if (mode == "test") return cmd_test();
  if (mode == "bench") return cmd_bench(E, Sk, iters, warmup);
  if (mode == "all") {
    int rc = cmd_test();
    if (rc != 0) return rc;
    return cmd_bench(E, Sk, iters, warmup);
  }
  std::fprintf(stderr, "Unknown mode: %s (use test/bench/all)\n", mode.c_str());
  return 1;
}
