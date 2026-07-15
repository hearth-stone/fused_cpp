#include <arm_neon.h>
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

float* AllocFloats(std::size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
    std::fprintf(stderr, "posix_memalign failed\n");
    std::exit(2);
  }
  return static_cast<float*>(p);
}

double NowSeconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void KernelScalarTail(const float* gate, const float* up, float* out, std::size_t begin, std::size_t n) {
  const std::size_t vl = svcntw();
  const svfloat32_t invln2 = svdup_f32(1.4426950408889634f);
  const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);
  const svfloat32_t c3 = svdup_f32(0.16666666f);
  const svfloat32_t c4 = svdup_f32(0.04166666f);
  const svfloat32_t c5 = svdup_f32(0.00833333f);
  const svfloat32_t c6 = svdup_f32(0.0013888889f);
  const svfloat32_t clamp_hi = svdup_f32(87.0f);
  const svfloat32_t clamp_lo = svdup_f32(-87.0f);
  std::size_t i = begin;
  while (i < n) {
    svbool_t pg = svwhilelt_b32(i, n);
    svfloat32_t g = svld1_f32(pg, gate + i);
    svfloat32_t u = svld1_f32(pg, up + i);
    svfloat32_t x = svneg_f32_x(pg, g);
    x = svmin_f32_x(pg, x, clamp_hi);
    x = svmax_f32_x(pg, x, clamp_lo);
    svfloat32_t fn = svmul_f32_x(pg, x, invln2);
    fn = svrintn_f32_x(pg, fn);
    svint32_t ni = svcvt_s32_f32_x(pg, fn);
    svfloat32_t r = svmls_f32_x(pg, x, fn, ln2);
    svfloat32_t t = svmla_f32_x(pg, c5, c6, r);
    svfloat32_t poly = svmla_f32_x(pg, c4, t, r);
    t = svmla_f32_x(pg, c3, poly, r);
    poly = svmla_f32_x(pg, svdup_f32(0.5f), t, r);
    t = svmla_f32_x(pg, svdup_f32(1.0f), poly, r);
    poly = svmla_f32_x(pg, svdup_f32(1.0f), t, r);
    svint32_t pow_bits = svlsl_n_s32_x(pg, svadd_s32_x(pg, ni, svdup_s32(127)), 23);
    svfloat32_t e = svmul_f32_x(pg, poly, svreinterpret_f32_s32(pow_bits));
    svfloat32_t den = svadd_f32_x(pg, e, svdup_f32(1.0f));
    svfloat32_t num = svmul_f32_x(pg, g, u);
    svst1_f32(pg, out + i, svdiv_f32_x(pg, num, den));
    i += vl;
  }
}

#define DO2(M) M(0) M(1)
#define DO4(M) M(0) M(1) M(2) M(3)
#define DO8(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

#define LOAD_STAGE(i)                                                              \
  svfloat32_t g##i = svld1_f32(pg, gate + idx + static_cast<std::size_t>(i) * vl); \
  svfloat32_t u##i = svld1_f32(pg, up + idx + static_cast<std::size_t>(i) * vl);
#define NEG_STAGE(i) svfloat32_t x##i = svneg_f32_x(pg, g##i);
#define CLAMP_STAGE(i)                    \
  x##i = svmin_f32_x(pg, x##i, clamp_hi); \
  x##i = svmax_f32_x(pg, x##i, clamp_lo);
#define FN_STAGE(i)                                  \
  svfloat32_t fn##i = svmul_f32_x(pg, x##i, invln2); \
  fn##i = svrintn_f32_x(pg, fn##i);
#define INT_STAGE(i) svint32_t ni##i = svcvt_s32_f32_x(pg, fn##i);
#define R_STAGE(i) svfloat32_t r##i = svmls_f32_x(pg, x##i, fn##i, ln2);
#define POLY0_STAGE(i) svfloat32_t t##i = svmla_f32_x(pg, c5, c6, r##i);
#define POLY1_STAGE(i) svfloat32_t poly##i = svmla_f32_x(pg, c4, t##i, r##i);
#define POLY2_STAGE(i) t##i = svmla_f32_x(pg, c3, poly##i, r##i);
#define POLY3_STAGE(i) poly##i = svmla_f32_x(pg, half, t##i, r##i);
#define POLY4_STAGE(i) t##i = svmla_f32_x(pg, one, poly##i, r##i);
#define POLY5_STAGE(i) poly##i = svmla_f32_x(pg, one, t##i, r##i);
#define POW_STAGE(i)                                                              \
  svint32_t pow_bits##i = svlsl_n_s32_x(pg, svadd_s32_x(pg, ni##i, bias127), 23); \
  svfloat32_t e##i = svmul_f32_x(pg, poly##i, svreinterpret_f32_s32(pow_bits##i));
#define DEN_STAGE(i) svfloat32_t den##i = svadd_f32_x(pg, e##i, one);
#define NUM_STAGE(i) svfloat32_t num##i = svmul_f32_x(pg, g##i, u##i);
#define DIV_STAGE(i) svfloat32_t y##i = svdiv_f32_x(pg, num##i, den##i);
#define STORE_STAGE(i) svst1_f32(pg, out + idx + static_cast<std::size_t>(i) * vl, y##i);

#define DEFINE_STAGED_KERNEL(U, DO)                                                     \
  void KernelStaged##U(const float* gate, const float* up, float* out, std::size_t n) { \
    const std::size_t vl = svcntw();                                                    \
    const std::size_t step = static_cast<std::size_t>(U) * vl;                          \
    const svbool_t pg = svptrue_b32();                                                  \
    const svfloat32_t invln2 = svdup_f32(1.4426950408889634f);                          \
    const svfloat32_t ln2 = svdup_f32(0.6931471805599453f);                             \
    const svfloat32_t c3 = svdup_f32(0.16666666f);                                      \
    const svfloat32_t c4 = svdup_f32(0.04166666f);                                      \
    const svfloat32_t c5 = svdup_f32(0.00833333f);                                      \
    const svfloat32_t c6 = svdup_f32(0.0013888889f);                                    \
    const svfloat32_t clamp_hi = svdup_f32(87.0f);                                      \
    const svfloat32_t clamp_lo = svdup_f32(-87.0f);                                     \
    const svfloat32_t half = svdup_f32(0.5f);                                           \
    const svfloat32_t one = svdup_f32(1.0f);                                            \
    const svint32_t bias127 = svdup_s32(127);                                           \
    std::size_t idx = 0;                                                                \
    for (; idx + step <= n; idx += step) {                                              \
      DO(LOAD_STAGE)                                                                    \
      DO(NEG_STAGE)                                                                     \
      DO(CLAMP_STAGE)                                                                   \
      DO(FN_STAGE)                                                                      \
      DO(INT_STAGE)                                                                     \
      DO(R_STAGE)                                                                       \
      DO(POLY0_STAGE)                                                                   \
      DO(POLY1_STAGE)                                                                   \
      DO(POLY2_STAGE)                                                                   \
      DO(POLY3_STAGE)                                                                   \
      DO(POLY4_STAGE)                                                                   \
      DO(POLY5_STAGE)                                                                   \
      DO(POW_STAGE)                                                                     \
      DO(DEN_STAGE)                                                                     \
      DO(NUM_STAGE)                                                                     \
      DO(DIV_STAGE)                                                                     \
      DO(STORE_STAGE)                                                                   \
    }                                                                                   \
    KernelScalarTail(gate, up, out, idx, n);                                            \
  }

DEFINE_STAGED_KERNEL(2, DO2)
DEFINE_STAGED_KERNEL(4, DO4)
DEFINE_STAGED_KERNEL(8, DO8)

#undef DEFINE_STAGED_KERNEL
#undef STORE_STAGE
#undef DIV_STAGE
#undef NUM_STAGE
#undef DEN_STAGE
#undef POW_STAGE
#undef POLY5_STAGE
#undef POLY4_STAGE
#undef POLY3_STAGE
#undef POLY2_STAGE
#undef POLY1_STAGE
#undef POLY0_STAGE
#undef R_STAGE
#undef INT_STAGE
#undef FN_STAGE
#undef CLAMP_STAGE
#undef NEG_STAGE
#undef LOAD_STAGE

void KernelNeonTail(const float* gate, const float* up, float* out, std::size_t begin, std::size_t n) {
  for (std::size_t i = begin; i < n; ++i) {
    float x = -gate[i];
    x = std::min(87.0f, std::max(-87.0f, x));
    const float fn = std::nearbyint(x * 1.4426950408889634f);
    const int ni = static_cast<int>(fn);
    const float r = x - fn * 0.6931471805599453f;
    float t = 0.00833333f + 0.0013888889f * r;
    float poly = 0.04166666f + t * r;
    t = 0.16666666f + poly * r;
    poly = 0.5f + t * r;
    t = 1.0f + poly * r;
    poly = 1.0f + t * r;
    std::uint32_t bits = static_cast<std::uint32_t>(ni + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));
    const float e = poly * scale;
    out[i] = gate[i] * up[i] / (e + 1.0f);
  }
}

#define NLOAD_STAGE(i)                                                          \
  float32x4_t ng##i = vld1q_f32(gate + idx + static_cast<std::size_t>(i) * vl); \
  float32x4_t nu##i = vld1q_f32(up + idx + static_cast<std::size_t>(i) * vl);
#define NNEG_STAGE(i) float32x4_t nx##i = vnegq_f32(ng##i);
#define NCLAMP_STAGE(i)                \
  nx##i = vminq_f32(nx##i, nclamp_hi); \
  nx##i = vmaxq_f32(nx##i, nclamp_lo);
#define NFN_STAGE(i)                              \
  float32x4_t nfn##i = vmulq_f32(nx##i, ninvln2); \
  nfn##i = vrndnq_f32(nfn##i);
#define NINT_STAGE(i) int32x4_t nni##i = vcvtq_s32_f32(nfn##i);
#define NR_STAGE(i) float32x4_t nr##i = vmlsq_f32(nx##i, nfn##i, nln2);
#define NPOLY0_STAGE(i) float32x4_t nt##i = vmlaq_f32(nc5, nc6, nr##i);
#define NPOLY1_STAGE(i) float32x4_t npoly##i = vmlaq_f32(nc4, nt##i, nr##i);
#define NPOLY2_STAGE(i) nt##i = vmlaq_f32(nc3, npoly##i, nr##i);
#define NPOLY3_STAGE(i) npoly##i = vmlaq_f32(nhalf, nt##i, nr##i);
#define NPOLY4_STAGE(i) nt##i = vmlaq_f32(none, npoly##i, nr##i);
#define NPOLY5_STAGE(i) npoly##i = vmlaq_f32(none, nt##i, nr##i);
#define NPOW_STAGE(i)                                                    \
  int32x4_t npow_bits##i = vshlq_n_s32(vaddq_s32(nni##i, nbias127), 23); \
  float32x4_t ne##i = vmulq_f32(npoly##i, vreinterpretq_f32_s32(npow_bits##i));
#define NDEN_STAGE(i) float32x4_t nden##i = vaddq_f32(ne##i, none);
#define NNUM_STAGE(i) float32x4_t nnum##i = vmulq_f32(ng##i, nu##i);
#define NDIV_STAGE(i) float32x4_t ny##i = vdivq_f32(nnum##i, nden##i);
#define NSTORE_STAGE(i) vst1q_f32(out + idx + static_cast<std::size_t>(i) * vl, ny##i);

#define DEFINE_NEON_STAGED_KERNEL(U, DO)                                                    \
  void KernelNeonStaged##U(const float* gate, const float* up, float* out, std::size_t n) { \
    constexpr std::size_t vl = 4;                                                           \
    constexpr std::size_t step = static_cast<std::size_t>(U) * vl;                          \
    const float32x4_t ninvln2 = vdupq_n_f32(1.4426950408889634f);                           \
    const float32x4_t nln2 = vdupq_n_f32(0.6931471805599453f);                              \
    const float32x4_t nc3 = vdupq_n_f32(0.16666666f);                                       \
    const float32x4_t nc4 = vdupq_n_f32(0.04166666f);                                       \
    const float32x4_t nc5 = vdupq_n_f32(0.00833333f);                                       \
    const float32x4_t nc6 = vdupq_n_f32(0.0013888889f);                                     \
    const float32x4_t nclamp_hi = vdupq_n_f32(87.0f);                                       \
    const float32x4_t nclamp_lo = vdupq_n_f32(-87.0f);                                      \
    const float32x4_t nhalf = vdupq_n_f32(0.5f);                                            \
    const float32x4_t none = vdupq_n_f32(1.0f);                                             \
    const int32x4_t nbias127 = vdupq_n_s32(127);                                            \
    std::size_t idx = 0;                                                                    \
    for (; idx + step <= n; idx += step) {                                                  \
      DO(NLOAD_STAGE)                                                                       \
      DO(NNEG_STAGE)                                                                        \
      DO(NCLAMP_STAGE)                                                                      \
      DO(NFN_STAGE)                                                                         \
      DO(NINT_STAGE)                                                                        \
      DO(NR_STAGE)                                                                          \
      DO(NPOLY0_STAGE)                                                                      \
      DO(NPOLY1_STAGE)                                                                      \
      DO(NPOLY2_STAGE)                                                                      \
      DO(NPOLY3_STAGE)                                                                      \
      DO(NPOLY4_STAGE)                                                                      \
      DO(NPOLY5_STAGE)                                                                      \
      DO(NPOW_STAGE)                                                                        \
      DO(NDEN_STAGE)                                                                        \
      DO(NNUM_STAGE)                                                                        \
      DO(NDIV_STAGE)                                                                        \
      DO(NSTORE_STAGE)                                                                      \
    }                                                                                       \
    KernelNeonTail(gate, up, out, idx, n);                                                  \
  }

DEFINE_NEON_STAGED_KERNEL(2, DO2)
DEFINE_NEON_STAGED_KERNEL(4, DO4)
DEFINE_NEON_STAGED_KERNEL(8, DO8)

#undef DEFINE_NEON_STAGED_KERNEL
#undef NSTORE_STAGE
#undef NDIV_STAGE
#undef NNUM_STAGE
#undef NDEN_STAGE
#undef NPOW_STAGE
#undef NPOLY5_STAGE
#undef NPOLY4_STAGE
#undef NPOLY3_STAGE
#undef NPOLY2_STAGE
#undef NPOLY1_STAGE
#undef NPOLY0_STAGE
#undef NR_STAGE
#undef NINT_STAGE
#undef NFN_STAGE
#undef NCLAMP_STAGE
#undef NNEG_STAGE
#undef NLOAD_STAGE

using KernelFn = void (*)(const float*, const float*, float*, std::size_t);

struct Stats {
  double median_ms = 0.0;
  double min_ms = 0.0;
  double elems_per_s = 0.0;
};

Stats Bench(KernelFn fn, const float* gate, const float* up, float* out, std::size_t n, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn(gate, up, out, n);
  }
  std::vector<double> ms;
  ms.reserve(iters);
  volatile float sink = 0.0f;
  for (int i = 0; i < iters; ++i) {
    const double t0 = NowSeconds();
    fn(gate, up, out, n);
    const double t1 = NowSeconds();
    sink += out[(static_cast<std::size_t>(i) * 131071u) % n];
    ms.push_back((t1 - t0) * 1000.0);
  }
  (void)sink;
  std::sort(ms.begin(), ms.end());
  return Stats{ms[ms.size() / 2], ms.front(), static_cast<double>(n) / (ms[ms.size() / 2] * 1e-3)};
}

double MaxAbsDiff(const float* a, const float* b, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
  }
  return m;
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
      const std::size_t m = std::strtoull(need("--m"), nullptr, 10);
      if (i + 1 >= argc || std::string(argv[i + 1]) != "--n") {
        std::fprintf(stderr, "--m must be followed by --n\n");
        return 2;
      }
      ++i;
      const std::size_t n = std::strtoull(need("--n"), nullptr, 10);
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

  std::printf("elems=%zu sve_vl_f32=%zu neon_vl_f32=4 matrix_like=%.1fM values staged=fdiv/poly6\n", elems, svcntw(),
              elems / 1e6);
  std::printf("%-10s %10s %10s %12s %12s\n", "kernel", "median_ms", "min_ms", "Gelem/s", "max_abs");

  KernelStaged4(gate, up, ref, elems);

  struct Case {
    const char* name;
    KernelFn fn;
  };
  const Case cases[] = {
      {"sve_u2", KernelStaged2},      {"sve_u4", KernelStaged4},      {"sve_u8", KernelStaged8},
      {"neon_u2", KernelNeonStaged2}, {"neon_u4", KernelNeonStaged4}, {"neon_u8", KernelNeonStaged8},
  };
  for (const Case& c : cases) {
    Stats s = Bench(c.fn, gate, up, out, elems, warmup, iters);
    const double diff = MaxAbsDiff(ref, out, elems);
    std::printf("%-10s %10.3f %10.3f %12.3f %12.6g\n", c.name, s.median_ms, s.min_ms, s.elems_per_s / 1e9, diff);
  }

  std::free(gate);
  std::free(up);
  std::free(ref);
  std::free(out);
  return 0;
}
