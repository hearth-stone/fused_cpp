#include <arm_neon.h>
#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static inline float32x4_t current_vexpq_f32(float32x4_t x) {
  const float32x4_t kLn2 = vdupq_n_f32(0.6931471805599453f);
  const float32x4_t kInvLn2 = vdupq_n_f32(1.4426950408889634f);
  const float32x4_t c1 = vdupq_n_f32(1.0f);
  const float32x4_t c2 = vdupq_n_f32(0.5f);
  const float32x4_t c3 = vdupq_n_f32(0.16666666f);
  const float32x4_t c4 = vdupq_n_f32(0.04166666f);
  const float32x4_t c5 = vdupq_n_f32(0.00833333f);

  const float32x4_t kHi = vdupq_n_f32(87.0f);
  const float32x4_t kLo = vdupq_n_f32(-87.0f);
  x = vminq_f32(x, kHi);
  x = vmaxq_f32(x, kLo);

  float32x4_t fn = vrndnq_f32(vmulq_f32(x, kInvLn2));
  int32x4_t n = vcvtq_s32_f32(fn);
  float32x4_t r = vfmsq_f32(x, fn, kLn2);

  float32x4_t poly = c5;
  poly = vfmaq_f32(c4, poly, r);
  poly = vfmaq_f32(c3, poly, r);
  poly = vfmaq_f32(c2, poly, r);
  poly = vfmaq_f32(c1, poly, r);
  poly = vfmaq_f32(c1, poly, r);

  int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(exp_bits);
  return vmulq_f32(poly, pow2n);
}

NOINLINE void current_neon_exp(const float* in, float* out, size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t x = vld1q_f32(in + i);
    vst1q_f32(out + i, current_vexpq_f32(x));
  }
  for (; i < n; ++i) out[i] = std::exp(in[i]);
}

NOINLINE void refs_sve_poly4_exp(const float* in, float* out, size_t n) {
  const svfloat32_t inv_ln2 = svdup_f32(1.4426950408889634f);
  const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
  const svfloat32_t one = svdup_f32(1.0f);
  const svfloat32_t half = svdup_f32(0.5f);
  const svfloat32_t c3 = svdup_f32(1.0f / 6.0f);
  const svfloat32_t c4 = svdup_f32(1.0f / 24.0f);

  size_t i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(i),
                                static_cast<uint64_t>(n));
    svfloat32_t x = svld1_f32(pg, in + i);
    svfloat32_t nf = svrinta_f32_x(pg, svmul_f32_x(pg, x, inv_ln2));
    svfloat32_t r = svmls_f32_x(pg, x, nf, ln2);
    svint32_t ni = svcvt_s32_f32_x(pg, nf);

    svfloat32_t r2 = svmul_f32_x(pg, r, r);
    svfloat32_t r3 = svmul_f32_x(pg, r2, r);
    svfloat32_t r4 = svmul_f32_x(pg, r2, r2);
    svfloat32_t poly = one;
    poly = svmla_f32_x(pg, poly, r, one);
    poly = svmla_f32_x(pg, poly, r2, half);
    poly = svmla_f32_x(pg, poly, r3, c3);
    poly = svmla_f32_x(pg, poly, r4, c4);

    svst1_f32(pg, out + i, svscale_f32_x(pg, poly, ni));
    i += svcntw();
  }
}

NOINLINE void refs_sve_poly6_exp(const float* in, float* out, size_t n) {
  const svfloat32_t inv_ln2 = svdup_f32(1.4426950408889634f);
  const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
  const svfloat32_t c0 = svdup_f32(1.0f);
  const svfloat32_t c1 = svdup_f32(1.0f);
  const svfloat32_t c2 = svdup_f32(0.5f);
  const svfloat32_t c3 = svdup_f32(1.0f / 6.0f);
  const svfloat32_t c4 = svdup_f32(1.0f / 24.0f);
  const svfloat32_t c5 = svdup_f32(1.0f / 120.0f);
  const svfloat32_t c6 = svdup_f32(1.0f / 720.0f);

  size_t i = 0;
  while (i < n) {
    svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(i),
                                static_cast<uint64_t>(n));
    svfloat32_t x = svld1_f32(pg, in + i);
    svfloat32_t nf = svrinta_f32_x(pg, svmul_f32_x(pg, x, inv_ln2));
    svfloat32_t r = svmls_f32_x(pg, x, nf, ln2);
    svint32_t ni = svcvt_s32_f32_x(pg, nf);

    svfloat32_t poly = c6;
    poly = svmla_f32_x(pg, c5, poly, r);
    poly = svmla_f32_x(pg, c4, poly, r);
    poly = svmla_f32_x(pg, c3, poly, r);
    poly = svmla_f32_x(pg, c2, poly, r);
    poly = svmla_f32_x(pg, c1, poly, r);
    poly = svmla_f32_x(pg, c0, poly, r);

    svst1_f32(pg, out + i, svscale_f32_x(pg, poly, ni));
    i += svcntw();
  }
}

NOINLINE void refs_scalar_poly6_exp(const float* in, float* out, size_t n) {
  constexpr float kInvLn2 = 1.4426950408889634f;
  constexpr float kLn2 = 0.6931471805599453f;
  for (size_t i = 0; i < n; ++i) {
    float x = in[i];
    if (x > 88.0f) {
      out[i] = std::numeric_limits<float>::infinity();
      continue;
    }
    if (x < -87.0f) {
      out[i] = 0.0f;
      continue;
    }
    float nf = std::round(x * kInvLn2);
    float r = x - nf * kLn2;
    int ni = static_cast<int>(nf);
    float r2 = r * r;
    float r3 = r2 * r;
    float r4 = r2 * r2;
    float r5 = r4 * r;
    float r6 = r4 * r2;
    float poly = 1.0f + r + 0.5f * r2 + (1.0f / 6.0f) * r3
                 + (1.0f / 24.0f) * r4 + (1.0f / 120.0f) * r5
                 + (1.0f / 720.0f) * r6;
    union {
      float f;
      int32_t i;
    } scale;
    scale.i = (127 + ni) << 23;
    out[i] = poly * scale.f;
  }
}

NOINLINE void libm_exp(const float* in, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = std::exp(in[i]);
}

static uint32_t ordered_bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct ErrStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double avg_abs = 0.0;
  double avg_rel = 0.0;
  uint32_t max_ulp = 0;
};

static ErrStats error_stats(const std::vector<float>& got,
                            const std::vector<float>& ref) {
  ErrStats s;
  for (size_t i = 0; i < got.size(); ++i) {
    double a = got[i];
    double b = ref[i];
    double abs_err = std::abs(a - b);
    double rel_err = b != 0.0 ? abs_err / std::abs(b) : 0.0;
    s.max_abs = std::max(s.max_abs, abs_err);
    s.max_rel = std::max(s.max_rel, rel_err);
    s.avg_abs += abs_err;
    s.avg_rel += rel_err;
    if (std::isfinite(got[i]) && std::isfinite(ref[i])) {
      uint32_t ug = ordered_bits(got[i]);
      uint32_t ur = ordered_bits(ref[i]);
      s.max_ulp = std::max(s.max_ulp, ug > ur ? ug - ur : ur - ug);
    }
  }
  s.avg_abs /= static_cast<double>(got.size());
  s.avg_rel /= static_cast<double>(got.size());
  return s;
}

using ExpFn = void (*)(const float*, float*, size_t);

static double bench_ms(ExpFn fn, const std::vector<float>& in,
                       std::vector<float>& out, int repeats) {
  fn(in.data(), out.data(), in.size());
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeats; ++r) {
    fn(in.data(), out.data(), in.size());
  }
  auto t1 = std::chrono::steady_clock::now();
  volatile float sink = out[(repeats * 9973) % out.size()];
  (void)sink;
  return std::chrono::duration<double, std::milli>(t1 - t0).count()
         / static_cast<double>(repeats);
}

static void fill_inputs(std::vector<float>& x, float lo, float hi,
                        uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& v : x) v = dist(rng);
}

int main() {
  struct Method {
    const char* name;
    ExpFn fn;
  };
  const Method methods[] = {
      {"libm_expf", libm_exp},
      {"current_neon_poly5", current_neon_exp},
      {"refs_sve_poly4", refs_sve_poly4_exp},
      {"refs_sve_poly6", refs_sve_poly6_exp},
      {"refs_scalar_poly6", refs_scalar_poly6_exp},
  };
  struct Range {
    const char* name;
    float lo;
    float hi;
  };
  const Range ranges[] = {
      {"softmax_typical", -16.0f, 0.0f},
      {"softmax_wide", -80.0f, 0.0f},
      {"symmetric", -10.0f, 10.0f},
  };
  struct SizeCase {
    const char* name;
    size_t n;
    int repeats;
  };
  const SizeCase sizes[] = {
      {"l1_like", 8192, 20000},
      {"stream", 1 << 20, 200},
  };

  std::printf("sve_vl_f32=%zu lanes\n", static_cast<size_t>(svcntw()));
  for (const auto& range : ranges) {
    std::printf("\nrange=%s [%.1f, %.1f]\n", range.name, range.lo, range.hi);
    for (const auto& sz : sizes) {
      std::vector<float> in(sz.n), ref(sz.n), out(sz.n);
      fill_inputs(in, range.lo, range.hi, 12345);
      libm_exp(in.data(), ref.data(), ref.size());

      double current_ms = 0.0;
      std::printf("case=%s n=%zu repeats=%d\n", sz.name, sz.n, sz.repeats);
      std::printf("%-20s %10s %12s %10s %10s %10s %10s %10s\n",
                  "method", "ms/pass", "Melement/s", "speedup",
                  "max_abs", "max_rel", "avg_rel", "max_ulp");
      for (const auto& m : methods) {
        double ms = bench_ms(m.fn, in, out, sz.repeats);
        if (std::strcmp(m.name, "current_neon_poly5") == 0) current_ms = ms;
        ErrStats e = error_stats(out, ref);
        double melems = static_cast<double>(sz.n) / ms / 1000.0;
        double speedup = current_ms > 0.0 ? current_ms / ms : 0.0;
        std::printf("%-20s %10.4f %12.2f %10.3fx %10.2e %10.2e %10.2e %10u\n",
                    m.name, ms, melems, speedup, e.max_abs, e.max_rel,
                    e.avg_rel, e.max_ulp);
      }
    }
  }
  return 0;
}
