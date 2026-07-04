#include <arm_sve.h>

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

namespace {

enum class DivMode {
  kFdiv,
  kRecip1,
  kRecip2,
  kRecip3,
};

const char* ModeName(DivMode mode) {
  switch (mode) {
    case DivMode::kFdiv: return "fdiv";
    case DivMode::kRecip1: return "frecpe_1";
    case DivMode::kRecip2: return "frecpe_2";
    case DivMode::kRecip3: return "frecpe_3";
  }
  return "unknown";
}

float* AllocFloats(std::size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
    std::fprintf(stderr, "posix_memalign failed\n");
    std::exit(2);
  }
  return static_cast<float*>(p);
}

uint16_t F32ToBf16Bits(float x) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

svfloat32_t ExpNegPoly(svbool_t pg, svfloat32_t gate) {
  const svfloat32_t invln2 = svdup_f32(1.4426950408889634f);
  const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
  const svfloat32_t c3 = svdup_f32(0.16666666f);
  const svfloat32_t c4 = svdup_f32(0.04166666f);
  const svfloat32_t c5 = svdup_f32(0.00833333f);
  const svfloat32_t c6 = svdup_f32(0.0013888889f);
  const svfloat32_t clamp_hi = svdup_f32(87.0f);
  const svfloat32_t clamp_lo = svdup_f32(-87.0f);

  svfloat32_t x = svneg_f32_x(pg, gate);
  x = svmin_f32_x(pg, x, clamp_hi);
  x = svmax_f32_x(pg, x, clamp_lo);
  svfloat32_t fn = svmul_f32_x(pg, x, invln2);
  fn = svrintn_f32_x(pg, fn);
  svint32_t n = svcvt_s32_f32_x(pg, fn);
  svfloat32_t r = svmls_f32_x(pg, x, fn, ln2);

  // Degree-6 polynomial, matching the highest-degree fused kernel variant.
  svfloat32_t t = svmla_f32_x(pg, c5, c6, r);
  svfloat32_t poly = svmla_f32_x(pg, c4, t, r);
  t = svmla_f32_x(pg, c3, poly, r);
  poly = svmla_f32_x(pg, svdup_f32(0.5f), t, r);
  t = svmla_f32_x(pg, svdup_f32(1.0f), poly, r);
  poly = svmla_f32_x(pg, svdup_f32(1.0f), t, r);

  svint32_t pow_bits = svlsl_n_s32_x(pg, svadd_s32_x(pg, n, svdup_s32(127)), 23);
  svfloat32_t pow2n = svreinterpret_f32_s32(pow_bits);
  return svmul_f32_x(pg, poly, pow2n);
}

template <DivMode mode>
svfloat32_t SiluMul(svbool_t pg, svfloat32_t gate, svfloat32_t up) {
  svfloat32_t e = ExpNegPoly(pg, gate);
  svfloat32_t den = svadd_f32_x(pg, e, svdup_f32(1.0f));
  svfloat32_t num = svmul_f32_x(pg, gate, up);
  if constexpr (mode == DivMode::kFdiv) {
    return svdiv_f32_x(pg, num, den);
  } else {
    svfloat32_t r = svrecpe_f32(den);
    r = svmul_f32_x(pg, r, svrecps_f32(den, r));
    if constexpr (mode == DivMode::kRecip2 || mode == DivMode::kRecip3) {
      r = svmul_f32_x(pg, r, svrecps_f32(den, r));
    }
    if constexpr (mode == DivMode::kRecip3) {
      r = svmul_f32_x(pg, r, svrecps_f32(den, r));
    }
    return svmul_f32_x(pg, num, r);
  }
}

template <DivMode mode>
void KernelUnroll4(const float* gate, const float* up, float* out, std::size_t n) {
  const std::size_t vl = svcntw();
  std::size_t i = 0;
  const svbool_t pg_full = svptrue_b32();
  for (; i + 4 * vl <= n; i += 4 * vl) {
    svfloat32_t g0 = svld1_f32(pg_full, gate + i + 0 * vl);
    svfloat32_t g1 = svld1_f32(pg_full, gate + i + 1 * vl);
    svfloat32_t g2 = svld1_f32(pg_full, gate + i + 2 * vl);
    svfloat32_t g3 = svld1_f32(pg_full, gate + i + 3 * vl);
    svfloat32_t u0 = svld1_f32(pg_full, up + i + 0 * vl);
    svfloat32_t u1 = svld1_f32(pg_full, up + i + 1 * vl);
    svfloat32_t u2 = svld1_f32(pg_full, up + i + 2 * vl);
    svfloat32_t u3 = svld1_f32(pg_full, up + i + 3 * vl);
    svfloat32_t y0 = SiluMul<mode>(pg_full, g0, u0);
    svfloat32_t y1 = SiluMul<mode>(pg_full, g1, u1);
    svfloat32_t y2 = SiluMul<mode>(pg_full, g2, u2);
    svfloat32_t y3 = SiluMul<mode>(pg_full, g3, u3);
    svst1_f32(pg_full, out + i + 0 * vl, y0);
    svst1_f32(pg_full, out + i + 1 * vl, y1);
    svst1_f32(pg_full, out + i + 2 * vl, y2);
    svst1_f32(pg_full, out + i + 3 * vl, y3);
  }
  for (; i < n;) {
    svbool_t pg = svwhilelt_b32(i, n);
    svfloat32_t g = svld1_f32(pg, gate + i);
    svfloat32_t u = svld1_f32(pg, up + i);
    svfloat32_t y = SiluMul<mode>(pg, g, u);
    svst1_f32(pg, out + i, y);
    i += vl;
  }
}

using KernelFn = void (*)(const float*, const float*, float*, std::size_t);

KernelFn FnForMode(DivMode mode) {
  switch (mode) {
    case DivMode::kFdiv: return &KernelUnroll4<DivMode::kFdiv>;
    case DivMode::kRecip1: return &KernelUnroll4<DivMode::kRecip1>;
    case DivMode::kRecip2: return &KernelUnroll4<DivMode::kRecip2>;
    case DivMode::kRecip3: return &KernelUnroll4<DivMode::kRecip3>;
  }
  return nullptr;
}

double NowSeconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Stats {
  double median_ms = 0.0;
  double min_ms = 0.0;
  double elems_per_s = 0.0;
};

Stats Bench(KernelFn fn, const float* gate, const float* up, float* out,
            std::size_t n, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn(gate, up, out, n);
  }
  std::vector<double> ms;
  ms.reserve(iters);
  volatile float sink = 0.0f;
  for (int i = 0; i < iters; ++i) {
    double t0 = NowSeconds();
    fn(gate, up, out, n);
    double t1 = NowSeconds();
    sink += out[(static_cast<std::size_t>(i) * 104729u) % n];
    ms.push_back((t1 - t0) * 1000.0);
  }
  (void)sink;
  std::sort(ms.begin(), ms.end());
  Stats s;
  s.median_ms = ms[ms.size() / 2];
  s.min_ms = ms.front();
  s.elems_per_s = static_cast<double>(n) / (s.median_ms * 1e-3);
  return s;
}

struct Diff {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double rms = 0.0;
  std::size_t bf16_mismatch = 0;
};

Diff Compare(const float* ref, const float* got, std::size_t n) {
  long double sum_sq = 0.0;
  Diff d;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = ref[i];
    const double b = got[i];
    const double abs = std::fabs(a - b);
    const double rel = abs / std::max(1e-12, std::fabs(a));
    d.max_abs = std::max(d.max_abs, abs);
    d.max_rel = std::max(d.max_rel, rel);
    sum_sq += abs * abs;
    d.bf16_mismatch += F32ToBf16Bits(ref[i]) != F32ToBf16Bits(got[i]);
  }
  d.rms = std::sqrt(static_cast<double>(sum_sq / n));
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t elems = 4096ull * 4096ull;
  int warmup = 3;
  int iters = 20;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--elems") {
      elems = std::strtoull(need("--elems"), nullptr, 10);
    } else if (arg == "--m") {
      std::size_t m = std::strtoull(need("--m"), nullptr, 10);
      if (i + 1 >= argc || std::string(argv[i + 1]) != "--n") {
        std::fprintf(stderr, "--m must be followed by --n\n");
        return 2;
      }
      ++i;
      std::size_t n = std::strtoull(need("--n"), nullptr, 10);
      elems = m * n;
    } else if (arg == "--warmup") {
      warmup = std::atoi(need("--warmup"));
    } else if (arg == "--iters") {
      iters = std::atoi(need("--iters"));
    } else {
      std::fprintf(stderr, "unknown arg %s\n", arg.c_str());
      return 2;
    }
  }

  float* gate = AllocFloats(elems);
  float* up = AllocFloats(elems);
  float* ref = AllocFloats(elems);
  float* out = AllocFloats(elems);

  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (std::size_t i = 0; i < elems; ++i) {
    gate[i] = std::max(-8.0f, std::min(8.0f, dist(rng)));
    up[i] = std::max(-8.0f, std::min(8.0f, dist(rng)));
  }

  std::printf("elems=%zu vl_f32=%zu matrix_like=%.1fM values\n",
              elems, svcntw(), elems / 1e6);
  std::printf("%-10s %10s %10s %12s %12s %12s %12s %12s\n",
              "mode", "median_ms", "min_ms", "Gelem/s",
              "max_abs", "max_rel", "rms", "bf16_mis%");

  KernelUnroll4<DivMode::kFdiv>(gate, up, ref, elems);
  const DivMode modes[] = {
      DivMode::kFdiv, DivMode::kRecip1, DivMode::kRecip2, DivMode::kRecip3};
  for (DivMode mode : modes) {
    KernelFn fn = FnForMode(mode);
    Stats s = Bench(fn, gate, up, out, elems, warmup, iters);
    Diff d;
    if (mode == DivMode::kFdiv) {
      d = Compare(ref, out, elems);
    } else {
      fn(gate, up, out, elems);
      d = Compare(ref, out, elems);
    }
    std::printf("%-10s %10.3f %10.3f %12.3f %12.6g %12.6g %12.6g %12.4f\n",
                ModeName(mode), s.median_ms, s.min_ms, s.elems_per_s / 1e9,
                d.max_abs, d.max_rel, d.rms,
                100.0 * static_cast<double>(d.bf16_mismatch) /
                    static_cast<double>(elems));
  }

  std::free(gate);
  std::free(up);
  std::free(ref);
  std::free(out);
  return 0;
}
