// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <tuple>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_neon.h>
#include <arm_sve.h>
#endif

namespace {

#if defined(__ARM_FEATURE_SVE)

constexpr int64_t kMhcProjectionN = 24;
constexpr int64_t kKWindowAlignment = 256;

#if defined(FUSED_CPP_DEEPSEEK_V4_MHC_SVE_ASM)
extern "C" void deepseek_v4_mhc_m7n24_sve(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output,
                                          int64_t k, bool load_c);
extern "C" void deepseek_v4_mhc_m8n24_sve(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output,
                                          int64_t k, bool load_c);
extern "C" void deepseek_v4_mhc_m4n24_sve(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output,
                                          int64_t k, bool load_c);
#endif

int EffectiveThreads(int64_t requested, int64_t work_items) {
  if (work_items <= 1) {
    return 1;
  }
#if defined(_OPENMP)
  const int64_t limit = requested > 0 ? requested : omp_get_max_threads();
  return static_cast<int>(std::max<int64_t>(1, std::min(limit, work_items)));
#else
  (void)requested;
  return 1;
#endif
}

int64_t KWindowElements(int64_t k, int64_t window_bytes) {
  int64_t target_kc = window_bytes / (kMhcProjectionN * static_cast<int64_t>(sizeof(float)));
  if (target_kc >= kKWindowAlignment) {
    target_kc = target_kc / kKWindowAlignment * kKWindowAlignment;
  }
  target_kc = std::max<int64_t>(1, std::min(k, target_kc));

  // Capacity determines the minimum number of windows. Balance K across that
  // count so a power-of-two K does not become one near-capacity window plus a
  // short remainder (for example, 16384 -> 8192 + 8192 at a 1 MiB budget).
  const int64_t windows = (k + target_kc - 1) / target_kc;
  int64_t kc = (k + windows - 1) / windows;
  if (kc >= kKWindowAlignment) {
    kc = (kc + kKWindowAlignment - 1) / kKWindowAlignment * kKWindowAlignment;
  }
  return std::min(k, kc);
}

struct ProjectionBlockGeometry {
  int64_t m7_blocks = 0;
  int64_t secondary_blocks = 0;
  int64_t tail_rows = 0;
  int64_t secondary_rows = 8;

  int64_t blocks() const { return m7_blocks + secondary_blocks + (tail_rows != 0 ? 1 : 0); }

  std::pair<int64_t, int64_t> block(int64_t index) const {
    if (index < m7_blocks) {
      return {index * 7, 7};
    }
    const int64_t m7_rows = m7_blocks * 7;
    index -= m7_blocks;
    if (index < secondary_blocks) {
      return {m7_rows + index * secondary_rows, secondary_rows};
    }
    return {m7_rows + secondary_blocks * secondary_rows, tail_rows};
  }
};

ProjectionBlockGeometry MakeProjectionBlockGeometry(int64_t m, int64_t mr, bool pad_m7_tail) {
  if (mr != 7) {
    return ProjectionBlockGeometry{0, m / mr, m % mr, mr};
  }
  if (pad_m7_tail) {
    return ProjectionBlockGeometry{(m + 6) / 7, 0, 0, 8};
  }
  const int64_t remainder = m % 7;
  if (m >= remainder * 8) {
    return ProjectionBlockGeometry{(m - remainder * 8) / 7, remainder, 0, 8};
  }
  return ProjectionBlockGeometry{m / 7, 0, remainder, 8};
}

float SquaredSum(const c10::BFloat16* input, int64_t k) {
  const int64_t lanes = svcntw();
  const auto* bits = reinterpret_cast<const uint16_t*>(input);
  svfloat32_t sum0 = svdup_f32(0.0f);
  svfloat32_t sum1 = svdup_f32(0.0f);
  int64_t offset = 0;
  for (; offset + 2 * lanes <= k; offset += 2 * lanes) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t value0 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, bits + offset), 16));
    const svfloat32_t value1 = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, bits + offset + lanes), 16));
    sum0 = svmla_f32_x(pg, sum0, value0, value0);
    sum1 = svmla_f32_x(pg, sum1, value1, value1);
  }
  for (; offset < k; offset += lanes) {
    const svbool_t pg = svwhilelt_b32(offset, k);
    const svfloat32_t value = svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, bits + offset), 16));
    sum0 = svmla_f32_m(pg, sum0, value, value);
  }
  return svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sum0, sum1));
}

#define DECLARE_ACCUMULATORS_M8()                                         \
  svfloat32_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32; \
  svfloat32_t c40, c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72

#define INIT_ROW_M8(row)                                                           \
  c##row##0 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN) : zero;     \
  c##row##1 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 8) : zero; \
  c##row##2 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 16) : zero

#define FMLA_B_M8(column)                            \
  c0##column = svmla_n_f32_x(pg, c0##column, b, a0); \
  c1##column = svmla_n_f32_x(pg, c1##column, b, a1); \
  c2##column = svmla_n_f32_x(pg, c2##column, b, a2); \
  c3##column = svmla_n_f32_x(pg, c3##column, b, a3); \
  c4##column = svmla_n_f32_x(pg, c4##column, b, a4); \
  c5##column = svmla_n_f32_x(pg, c5##column, b, a5); \
  c6##column = svmla_n_f32_x(pg, c6##column, b, a6); \
  c7##column = svmla_n_f32_x(pg, c7##column, b, a7)

#define STORE_ROW_M8(row)                                         \
  svst1_f32(pg, output + (row) * kMhcProjectionN, c##row##0);     \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 8, c##row##1); \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 16, c##row##2)

// SVE256 maps N=24 to three vectors. Keeping eight rows gives 24 independent
// accumulators, while A is streamed as BF16 scalars and the K-major FP32 B
// window is reused across this thread's M blocks from private L2.
void KernelM8N24(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output, int64_t k,
                 bool load_c) {
#if defined(FUSED_CPP_DEEPSEEK_V4_MHC_SVE_ASM)
  deepseek_v4_mhc_m8n24_sve(input, lda, packed_b, output, k, load_c);
#else
  const svbool_t pg = svptrue_b32();
  const svfloat32_t zero = svdup_f32(0.0f);
  DECLARE_ACCUMULATORS_M8();
  INIT_ROW_M8(0);
  INIT_ROW_M8(1);
  INIT_ROW_M8(2);
  INIT_ROW_M8(3);
  INIT_ROW_M8(4);
  INIT_ROW_M8(5);
  INIT_ROW_M8(6);
  INIT_ROW_M8(7);
  for (int64_t index = 0; index < k; ++index) {
    const float a0 = static_cast<float>(input[index]);
    const float a1 = static_cast<float>(input[lda + index]);
    const float a2 = static_cast<float>(input[2 * lda + index]);
    const float a3 = static_cast<float>(input[3 * lda + index]);
    const float a4 = static_cast<float>(input[4 * lda + index]);
    const float a5 = static_cast<float>(input[5 * lda + index]);
    const float a6 = static_cast<float>(input[6 * lda + index]);
    const float a7 = static_cast<float>(input[7 * lda + index]);
    const float* weight = packed_b + index * kMhcProjectionN;
    svfloat32_t b = svld1_f32(pg, weight);
    FMLA_B_M8(0);
    b = svld1_f32(pg, weight + 8);
    FMLA_B_M8(1);
    b = svld1_f32(pg, weight + 16);
    FMLA_B_M8(2);
  }
  STORE_ROW_M8(0);
  STORE_ROW_M8(1);
  STORE_ROW_M8(2);
  STORE_ROW_M8(3);
  STORE_ROW_M8(4);
  STORE_ROW_M8(5);
  STORE_ROW_M8(6);
  STORE_ROW_M8(7);
#endif
}

#undef DECLARE_ACCUMULATORS_M8
#undef INIT_ROW_M8
#undef FMLA_B_M8
#undef STORE_ROW_M8

#if defined(FUSED_CPP_DEEPSEEK_V4_MHC_SVE_ASM)
void KernelM7N24(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output, int64_t k,
                 bool load_c) {
  deepseek_v4_mhc_m7n24_sve(input, lda, packed_b, output, k, load_c);
}
#endif

void HeadProjectionM12N4Neon(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output, int64_t k) {
  const float32x4_t initial_weight = vld1q_f32(packed_b);
#define INIT_HEAD_ACCUMULATOR(row) \
  float32x4_t accumulator##row = vmulq_n_f32(initial_weight, static_cast<float>(input[(row) * lda]))
  INIT_HEAD_ACCUMULATOR(0);
  INIT_HEAD_ACCUMULATOR(1);
  INIT_HEAD_ACCUMULATOR(2);
  INIT_HEAD_ACCUMULATOR(3);
  INIT_HEAD_ACCUMULATOR(4);
  INIT_HEAD_ACCUMULATOR(5);
  INIT_HEAD_ACCUMULATOR(6);
  INIT_HEAD_ACCUMULATOR(7);
  INIT_HEAD_ACCUMULATOR(8);
  INIT_HEAD_ACCUMULATOR(9);
  INIT_HEAD_ACCUMULATOR(10);
  INIT_HEAD_ACCUMULATOR(11);
#undef INIT_HEAD_ACCUMULATOR

  for (int64_t index = 1; index < k; ++index) {
    const float32x4_t weight = vld1q_f32(packed_b + index * 4);
#define UPDATE_HEAD_ACCUMULATOR(row) \
  accumulator##row = vfmaq_n_f32(accumulator##row, weight, static_cast<float>(input[(row) * lda + index]))
    UPDATE_HEAD_ACCUMULATOR(0);
    UPDATE_HEAD_ACCUMULATOR(1);
    UPDATE_HEAD_ACCUMULATOR(2);
    UPDATE_HEAD_ACCUMULATOR(3);
    UPDATE_HEAD_ACCUMULATOR(4);
    UPDATE_HEAD_ACCUMULATOR(5);
    UPDATE_HEAD_ACCUMULATOR(6);
    UPDATE_HEAD_ACCUMULATOR(7);
    UPDATE_HEAD_ACCUMULATOR(8);
    UPDATE_HEAD_ACCUMULATOR(9);
    UPDATE_HEAD_ACCUMULATOR(10);
    UPDATE_HEAD_ACCUMULATOR(11);
#undef UPDATE_HEAD_ACCUMULATOR
  }

#define STORE_HEAD_ACCUMULATOR(row) vst1q_f32(output + (row) * 4, accumulator##row)
  STORE_HEAD_ACCUMULATOR(0);
  STORE_HEAD_ACCUMULATOR(1);
  STORE_HEAD_ACCUMULATOR(2);
  STORE_HEAD_ACCUMULATOR(3);
  STORE_HEAD_ACCUMULATOR(4);
  STORE_HEAD_ACCUMULATOR(5);
  STORE_HEAD_ACCUMULATOR(6);
  STORE_HEAD_ACCUMULATOR(7);
  STORE_HEAD_ACCUMULATOR(8);
  STORE_HEAD_ACCUMULATOR(9);
  STORE_HEAD_ACCUMULATOR(10);
  STORE_HEAD_ACCUMULATOR(11);
#undef STORE_HEAD_ACCUMULATOR
}

#define DECLARE_ACCUMULATORS_M4()                                         \
  svfloat32_t c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15; \
  svfloat32_t c20, c21, c22, c23, c24, c25, c30, c31, c32, c33, c34, c35

#define INIT_ROW_M4(row)                                                            \
  c##row##0 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN) : zero;      \
  c##row##1 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 4) : zero;  \
  c##row##2 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 8) : zero;  \
  c##row##3 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 12) : zero; \
  c##row##4 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 16) : zero; \
  c##row##5 = load_c ? svld1_f32(pg, output + (row) * kMhcProjectionN + 20) : zero

#define FMLA_B_M4(column)                            \
  c0##column = svmla_n_f32_x(pg, c0##column, b, a0); \
  c1##column = svmla_n_f32_x(pg, c1##column, b, a1); \
  c2##column = svmla_n_f32_x(pg, c2##column, b, a2); \
  c3##column = svmla_n_f32_x(pg, c3##column, b, a3)

#define STORE_ROW_M4(row)                                          \
  svst1_f32(pg, output + (row) * kMhcProjectionN, c##row##0);      \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 4, c##row##1);  \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 8, c##row##2);  \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 12, c##row##3); \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 16, c##row##4); \
  svst1_f32(pg, output + (row) * kMhcProjectionN + 20, c##row##5)

// SVE128 maps the same N=24 projection to six vectors. Four rows preserve the
// same 24-accumulator register budget as the SVE256 kernel.
void KernelM4N24(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output, int64_t k,
                 bool load_c) {
#if defined(FUSED_CPP_DEEPSEEK_V4_MHC_SVE_ASM)
  deepseek_v4_mhc_m4n24_sve(input, lda, packed_b, output, k, load_c);
#else
  const svbool_t pg = svptrue_b32();
  const svfloat32_t zero = svdup_f32(0.0f);
  DECLARE_ACCUMULATORS_M4();
  INIT_ROW_M4(0);
  INIT_ROW_M4(1);
  INIT_ROW_M4(2);
  INIT_ROW_M4(3);
  for (int64_t index = 0; index < k; ++index) {
    const float a0 = static_cast<float>(input[index]);
    const float a1 = static_cast<float>(input[lda + index]);
    const float a2 = static_cast<float>(input[2 * lda + index]);
    const float a3 = static_cast<float>(input[3 * lda + index]);
    const float* weight = packed_b + index * kMhcProjectionN;
    svfloat32_t b = svld1_f32(pg, weight);
    FMLA_B_M4(0);
    b = svld1_f32(pg, weight + 4);
    FMLA_B_M4(1);
    b = svld1_f32(pg, weight + 8);
    FMLA_B_M4(2);
    b = svld1_f32(pg, weight + 12);
    FMLA_B_M4(3);
    b = svld1_f32(pg, weight + 16);
    FMLA_B_M4(4);
    b = svld1_f32(pg, weight + 20);
    FMLA_B_M4(5);
  }
  STORE_ROW_M4(0);
  STORE_ROW_M4(1);
  STORE_ROW_M4(2);
  STORE_ROW_M4(3);
#endif
}

#undef DECLARE_ACCUMULATORS_M4
#undef INIT_ROW_M4
#undef FMLA_B_M4
#undef STORE_ROW_M4

void KernelTail(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output, int64_t rows, int64_t k,
                bool load_c) {
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < kMhcProjectionN; ++column) {
      float sum = load_c ? output[row * kMhcProjectionN + column] : 0.0f;
      for (int64_t index = 0; index < k; ++index) {
        sum += static_cast<float>(input[row * lda + index]) * packed_b[index * kMhcProjectionN + column];
      }
      output[row * kMhcProjectionN + column] = sum;
    }
  }
}

void RunBlock(const c10::BFloat16* input, int64_t lda, const float* packed_b, float* output, int64_t rows, int64_t k,
              int64_t mr, bool load_c) {
#if defined(FUSED_CPP_DEEPSEEK_V4_MHC_SVE_ASM)
  if (rows == 7) {
    KernelM7N24(input, lda, packed_b, output, k, load_c);
    return;
  }
#endif
  if (rows == 8) {
    KernelM8N24(input, lda, packed_b, output, k, load_c);
    return;
  }
  if (rows == 4 && mr == 4) {
    KernelM4N24(input, lda, packed_b, output, k, load_c);
    return;
  }
  KernelTail(input, lda, packed_b, output, rows, k, load_c);
}

[[gnu::always_inline]] inline svfloat32_t ExpFexpa(svbool_t pg, svfloat32_t value) {
  const svfloat32_t x = svmax_n_f32_x(pg, svmin_n_f32_x(pg, value, 87.0f), -87.0f);
  const svfloat32_t encoded = svmla_n_f32_x(pg, svdup_f32(196735.0f), x, 1.4426950216293335f);
  const svfloat32_t exponent = svsub_n_f32_x(pg, encoded, 196735.0f);
  svfloat32_t residual = svmls_n_f32_x(pg, x, exponent, 0.693145751953125f);
  residual = svmls_n_f32_x(pg, residual, exponent, 1.428606765330187e-06f);
  const svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(encoded));
  const svfloat32_t polynomial = svmla_n_f32_x(pg, svdup_f32(1.000003695487976f), residual, 0.5000003576278687f);
  return svmla_f32_x(pg, scale, scale, svmul_f32_x(pg, polynomial, residual));
}

[[gnu::always_inline]] inline svfloat32_t SigmoidFexpa(svbool_t pg, svfloat32_t value) {
  const svfloat32_t exponential = ExpFexpa(pg, svneg_f32_x(pg, svabs_f32_x(pg, value)));
  const svfloat32_t denominator = svadd_n_f32_x(pg, exponential, 1.0f);
  const svbool_t nonnegative = svcmpge_n_f32(pg, value, 0.0f);
  const svfloat32_t numerator = svsel_f32(nonnegative, svdup_f32(1.0f), exponential);
  return svdiv_f32_x(pg, numerator, denominator);
}

[[gnu::always_inline]] inline svfloat32_t LoadBf16AsF32(svbool_t pg, const c10::BFloat16* source) {
  const auto* bits = reinterpret_cast<const uint16_t*>(source);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, bits), 16));
}

[[gnu::always_inline]] inline svfloat32_t StoreRoundedBf16(svbool_t pg, c10::BFloat16* destination, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const svuint32_t rounded = svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fff)), 16);
  svst1h_u32(pg, reinterpret_cast<uint16_t*>(destination), rounded);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, rounded, 16));
}

[[gnu::always_inline]] inline svfloat32_t MixResidual4(svbool_t pg, const c10::BFloat16* residual, int64_t h,
                                                       int64_t column, const float* pre) {
  svfloat32_t value = svmul_n_f32_x(pg, LoadBf16AsF32(pg, residual + column), pre[0]);
  value = svmla_n_f32_x(pg, value, LoadBf16AsF32(pg, residual + h + column), pre[1]);
  value = svmla_n_f32_x(pg, value, LoadBf16AsF32(pg, residual + 2 * h + column), pre[2]);
  return svmla_n_f32_x(pg, value, LoadBf16AsF32(pg, residual + 3 * h + column), pre[3]);
}

void ApplyPreMixRmsNormRow(const c10::BFloat16* residual, const float* pre, const c10::BFloat16* norm_weight,
                           c10::BFloat16* output, int64_t h, float norm_eps) {
  const int64_t lanes = svcntw();
  svfloat32_t sum0 = svdup_f32(0.0f);
  svfloat32_t sum1 = svdup_f32(0.0f);
  int64_t column = 0;
  for (; column + 2 * lanes <= h; column += 2 * lanes) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t raw0 = StoreRoundedBf16(pg, output + column, MixResidual4(pg, residual, h, column, pre));
    const svfloat32_t raw1 =
        StoreRoundedBf16(pg, output + column + lanes, MixResidual4(pg, residual, h, column + lanes, pre));
    sum0 = svmla_f32_x(pg, sum0, raw0, raw0);
    sum1 = svmla_f32_x(pg, sum1, raw1, raw1);
  }
  for (; column < h; column += lanes) {
    const svbool_t pg = svwhilelt_b32(column, h);
    const svfloat32_t raw = StoreRoundedBf16(pg, output + column, MixResidual4(pg, residual, h, column, pre));
    sum0 = svmla_f32_m(pg, sum0, raw, raw);
  }
  const float squared_sum = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sum0, sum1));
  const float rms_scale = 1.0f / std::sqrt(squared_sum / static_cast<float>(h) + norm_eps);

  column = 0;
  for (; column + 2 * lanes <= h; column += 2 * lanes) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t value0 = svmul_n_f32_x(
        pg, svmul_f32_x(pg, LoadBf16AsF32(pg, output + column), LoadBf16AsF32(pg, norm_weight + column)), rms_scale);
    const svfloat32_t value1 = svmul_n_f32_x(
        pg,
        svmul_f32_x(pg, LoadBf16AsF32(pg, output + column + lanes), LoadBf16AsF32(pg, norm_weight + column + lanes)),
        rms_scale);
    StoreRoundedBf16(pg, output + column, value0);
    StoreRoundedBf16(pg, output + column + lanes, value1);
  }
  for (; column < h; column += lanes) {
    const svbool_t pg = svwhilelt_b32(column, h);
    const svfloat32_t value = svmul_n_f32_x(
        pg, svmul_f32_x(pg, LoadBf16AsF32(pg, output + column), LoadBf16AsF32(pg, norm_weight + column)), rms_scale);
    StoreRoundedBf16(pg, output + column, value);
  }
}

void ApplyPostRowU2(const c10::BFloat16* residual, const c10::BFloat16* layer_output, const float* post,
                    const float* comb, c10::BFloat16* output, int64_t h) {
  const int64_t lanes = svcntw();
  int64_t column = 0;
  for (; column + 2 * lanes <= h; column += 2 * lanes) {
    const svbool_t pg = svptrue_b32();
    const svfloat32_t source0a = LoadBf16AsF32(pg, residual + column);
    const svfloat32_t source0b = LoadBf16AsF32(pg, residual + column + lanes);
    svfloat32_t output0a = svmul_n_f32_x(pg, source0a, comb[0]);
    svfloat32_t output0b = svmul_n_f32_x(pg, source0b, comb[0]);
    svfloat32_t output1a = svmul_n_f32_x(pg, source0a, comb[1]);
    svfloat32_t output1b = svmul_n_f32_x(pg, source0b, comb[1]);
    svfloat32_t output2a = svmul_n_f32_x(pg, source0a, comb[2]);
    svfloat32_t output2b = svmul_n_f32_x(pg, source0b, comb[2]);
    svfloat32_t output3a = svmul_n_f32_x(pg, source0a, comb[3]);
    svfloat32_t output3b = svmul_n_f32_x(pg, source0b, comb[3]);

#define ACCUMULATE_RESIDUAL_STREAM(stream)                                                    \
  do {                                                                                        \
    const svfloat32_t source_a = LoadBf16AsF32(pg, residual + (stream) * h + column);         \
    const svfloat32_t source_b = LoadBf16AsF32(pg, residual + (stream) * h + column + lanes); \
    output0a = svmla_n_f32_x(pg, output0a, source_a, comb[(stream) * 4]);                     \
    output0b = svmla_n_f32_x(pg, output0b, source_b, comb[(stream) * 4]);                     \
    output1a = svmla_n_f32_x(pg, output1a, source_a, comb[(stream) * 4 + 1]);                 \
    output1b = svmla_n_f32_x(pg, output1b, source_b, comb[(stream) * 4 + 1]);                 \
    output2a = svmla_n_f32_x(pg, output2a, source_a, comb[(stream) * 4 + 2]);                 \
    output2b = svmla_n_f32_x(pg, output2b, source_b, comb[(stream) * 4 + 2]);                 \
    output3a = svmla_n_f32_x(pg, output3a, source_a, comb[(stream) * 4 + 3]);                 \
    output3b = svmla_n_f32_x(pg, output3b, source_b, comb[(stream) * 4 + 3]);                 \
  } while (false)
    ACCUMULATE_RESIDUAL_STREAM(1);
    ACCUMULATE_RESIDUAL_STREAM(2);
    ACCUMULATE_RESIDUAL_STREAM(3);
#undef ACCUMULATE_RESIDUAL_STREAM

    const svfloat32_t layer_a = LoadBf16AsF32(pg, layer_output + column);
    const svfloat32_t layer_b = LoadBf16AsF32(pg, layer_output + column + lanes);
    output0a = svmla_n_f32_x(pg, output0a, layer_a, post[0]);
    output0b = svmla_n_f32_x(pg, output0b, layer_b, post[0]);
    output1a = svmla_n_f32_x(pg, output1a, layer_a, post[1]);
    output1b = svmla_n_f32_x(pg, output1b, layer_b, post[1]);
    output2a = svmla_n_f32_x(pg, output2a, layer_a, post[2]);
    output2b = svmla_n_f32_x(pg, output2b, layer_b, post[2]);
    output3a = svmla_n_f32_x(pg, output3a, layer_a, post[3]);
    output3b = svmla_n_f32_x(pg, output3b, layer_b, post[3]);
    StoreRoundedBf16(pg, output + column, output0a);
    StoreRoundedBf16(pg, output + column + lanes, output0b);
    StoreRoundedBf16(pg, output + h + column, output1a);
    StoreRoundedBf16(pg, output + h + column + lanes, output1b);
    StoreRoundedBf16(pg, output + 2 * h + column, output2a);
    StoreRoundedBf16(pg, output + 2 * h + column + lanes, output2b);
    StoreRoundedBf16(pg, output + 3 * h + column, output3a);
    StoreRoundedBf16(pg, output + 3 * h + column + lanes, output3b);
  }

  for (; column < h; column += lanes) {
    const svbool_t pg = svwhilelt_b32(column, h);
    const svfloat32_t source0 = LoadBf16AsF32(pg, residual + column);
    svfloat32_t output0 = svmul_n_f32_x(pg, source0, comb[0]);
    svfloat32_t output1 = svmul_n_f32_x(pg, source0, comb[1]);
    svfloat32_t output2 = svmul_n_f32_x(pg, source0, comb[2]);
    svfloat32_t output3 = svmul_n_f32_x(pg, source0, comb[3]);
    for (int64_t stream = 1; stream < 4; ++stream) {
      const svfloat32_t source = LoadBf16AsF32(pg, residual + stream * h + column);
      output0 = svmla_n_f32_x(pg, output0, source, comb[stream * 4]);
      output1 = svmla_n_f32_x(pg, output1, source, comb[stream * 4 + 1]);
      output2 = svmla_n_f32_x(pg, output2, source, comb[stream * 4 + 2]);
      output3 = svmla_n_f32_x(pg, output3, source, comb[stream * 4 + 3]);
    }
    const svfloat32_t layer = LoadBf16AsF32(pg, layer_output + column);
    output0 = svmla_n_f32_x(pg, output0, layer, post[0]);
    output1 = svmla_n_f32_x(pg, output1, layer, post[1]);
    output2 = svmla_n_f32_x(pg, output2, layer, post[2]);
    output3 = svmla_n_f32_x(pg, output3, layer, post[3]);
    StoreRoundedBf16(pg, output + column, output0);
    StoreRoundedBf16(pg, output + h + column, output1);
    StoreRoundedBf16(pg, output + 2 * h + column, output2);
    StoreRoundedBf16(pg, output + 3 * h + column, output3);
  }
}

[[gnu::always_inline]] inline void ProcessHeadControl(const float* mixes, float rms_scale, float head_scale,
                                                      const float* head_base, float eps, float* pre) {
  const svbool_t pg4 = svwhilelt_b32(uint64_t{0}, uint64_t{4});
  svfloat32_t value = svld1_f32(pg4, mixes);
  value = svadd_f32_x(pg4, svmul_n_f32_x(pg4, value, rms_scale * head_scale), svld1_f32(pg4, head_base));
  svst1_f32(pg4, pre, svadd_n_f32_x(pg4, SigmoidFexpa(pg4, value), eps));
}

[[gnu::always_inline]] inline void ProcessControlRow(const float* mixes, float rms_scale, const float* hc_scale,
                                                     const float* hc_base, float pre_eps, float post_multiplier,
                                                     float* pre, float* post, float* comb) {
  const int64_t lanes = svcntw();
  const svbool_t pg4 = svwhilelt_b32(uint64_t{0}, uint64_t{4});
  if (lanes == 8) {
    const svbool_t pg8 = svptrue_b32();
    const svfloat32_t group_scale =
        svsel_f32(pg4, svdup_f32(rms_scale * hc_scale[0]), svdup_f32(rms_scale * hc_scale[1]));
    svfloat32_t value = svld1_f32(pg8, mixes);
    value = svadd_f32_x(pg8, svmul_f32_x(pg8, value, group_scale), svld1_f32(pg8, hc_base));
    value = SigmoidFexpa(pg8, value);
    svst1_f32(pg4, pre, svadd_n_f32_x(pg4, value, pre_eps));
    const svfloat32_t post_value = svext_f32(value, value, 4);
    svst1_f32(pg4, post, svmul_n_f32_x(pg4, post_value, post_multiplier));
  } else {
    svfloat32_t pre_value = svld1_f32(pg4, mixes);
    pre_value = svadd_f32_x(pg4, svmul_n_f32_x(pg4, pre_value, rms_scale * hc_scale[0]), svld1_f32(pg4, hc_base));
    svst1_f32(pg4, pre, svadd_n_f32_x(pg4, SigmoidFexpa(pg4, pre_value), pre_eps));

    svfloat32_t post_value = svld1_f32(pg4, mixes + 4);
    post_value = svadd_f32_x(pg4, svmul_n_f32_x(pg4, post_value, rms_scale * hc_scale[1]), svld1_f32(pg4, hc_base + 4));
    svst1_f32(pg4, post, svmul_n_f32_x(pg4, SigmoidFexpa(pg4, post_value), post_multiplier));
  }

  const svbool_t pg = svptrue_b32();
  const float comb_scale = rms_scale * hc_scale[2];
  for (int64_t column = 0; column < 16; column += lanes) {
    svfloat32_t value = svld1_f32(pg, mixes + 8 + column);
    value = svadd_f32_x(pg, svmul_n_f32_x(pg, value, comb_scale), svld1_f32(pg, hc_base + 8 + column));
    svst1_f32(pg, comb + column, value);
  }
}

[[gnu::always_inline]] inline void RunSinkhornBlock(float* comb, int64_t token, int64_t tokens, int repeat, float eps) {
  const int64_t active = std::min<int64_t>(svcntw(), tokens - token);
  const svbool_t pg = svwhilelt_b32(int64_t{0}, active);
  const svuint32_t offsets = svindex_u32(0, 16);
  float* base = comb + token * 16;
#define LOAD_COMB(row, column) svld1_gather_u32index_f32(pg, base + (row) * 4 + (column), offsets)
  svfloat32_t value00 = LOAD_COMB(0, 0);
  svfloat32_t value01 = LOAD_COMB(0, 1);
  svfloat32_t value02 = LOAD_COMB(0, 2);
  svfloat32_t value03 = LOAD_COMB(0, 3);
  svfloat32_t value10 = LOAD_COMB(1, 0);
  svfloat32_t value11 = LOAD_COMB(1, 1);
  svfloat32_t value12 = LOAD_COMB(1, 2);
  svfloat32_t value13 = LOAD_COMB(1, 3);
  svfloat32_t value20 = LOAD_COMB(2, 0);
  svfloat32_t value21 = LOAD_COMB(2, 1);
  svfloat32_t value22 = LOAD_COMB(2, 2);
  svfloat32_t value23 = LOAD_COMB(2, 3);
  svfloat32_t value30 = LOAD_COMB(3, 0);
  svfloat32_t value31 = LOAD_COMB(3, 1);
  svfloat32_t value32 = LOAD_COMB(3, 2);
  svfloat32_t value33 = LOAD_COMB(3, 3);
#undef LOAD_COMB

  // SVE sizeless tuple helpers make GCC spill live matrix vectors. These local
  // macros keep all 16 values named in this scope across the normalization loop.
#define SOFTMAX_VALUES(value0, value1, value2, value3)                                                                 \
  do {                                                                                                                 \
    const svfloat32_t maximum =                                                                                        \
        svmax_f32_x(pg, svmax_f32_x(pg, (value0), (value1)), svmax_f32_x(pg, (value2), (value3)));                     \
    (value0) = ExpFexpa(pg, svsub_f32_x(pg, (value0), maximum));                                                       \
    (value1) = ExpFexpa(pg, svsub_f32_x(pg, (value1), maximum));                                                       \
    (value2) = ExpFexpa(pg, svsub_f32_x(pg, (value2), maximum));                                                       \
    (value3) = ExpFexpa(pg, svsub_f32_x(pg, (value3), maximum));                                                       \
    const svfloat32_t sum = svadd_f32_x(pg, svadd_f32_x(pg, (value0), (value1)), svadd_f32_x(pg, (value2), (value3))); \
    const svfloat32_t reciprocal = svdiv_f32_x(pg, svdup_f32(1.0f), sum);                                              \
    (value0) = svadd_n_f32_x(pg, svmul_f32_x(pg, (value0), reciprocal), eps);                                          \
    (value1) = svadd_n_f32_x(pg, svmul_f32_x(pg, (value1), reciprocal), eps);                                          \
    (value2) = svadd_n_f32_x(pg, svmul_f32_x(pg, (value2), reciprocal), eps);                                          \
    (value3) = svadd_n_f32_x(pg, svmul_f32_x(pg, (value3), reciprocal), eps);                                          \
  } while (false)
#define NORMALIZE_VALUES(value0, value1, value2, value3)                                                     \
  do {                                                                                                       \
    const svfloat32_t sum = svadd_n_f32_x(                                                                   \
        pg, svadd_f32_x(pg, svadd_f32_x(pg, (value0), (value1)), svadd_f32_x(pg, (value2), (value3))), eps); \
    const svfloat32_t reciprocal = svdiv_f32_x(pg, svdup_f32(1.0f), sum);                                    \
    (value0) = svmul_f32_x(pg, (value0), reciprocal);                                                        \
    (value1) = svmul_f32_x(pg, (value1), reciprocal);                                                        \
    (value2) = svmul_f32_x(pg, (value2), reciprocal);                                                        \
    (value3) = svmul_f32_x(pg, (value3), reciprocal);                                                        \
  } while (false)

  SOFTMAX_VALUES(value00, value01, value02, value03);
  SOFTMAX_VALUES(value10, value11, value12, value13);
  SOFTMAX_VALUES(value20, value21, value22, value23);
  SOFTMAX_VALUES(value30, value31, value32, value33);
  NORMALIZE_VALUES(value00, value10, value20, value30);
  NORMALIZE_VALUES(value01, value11, value21, value31);
  NORMALIZE_VALUES(value02, value12, value22, value32);
  NORMALIZE_VALUES(value03, value13, value23, value33);
  for (int iteration = 1; iteration < repeat; ++iteration) {
    NORMALIZE_VALUES(value00, value01, value02, value03);
    NORMALIZE_VALUES(value10, value11, value12, value13);
    NORMALIZE_VALUES(value20, value21, value22, value23);
    NORMALIZE_VALUES(value30, value31, value32, value33);
    NORMALIZE_VALUES(value00, value10, value20, value30);
    NORMALIZE_VALUES(value01, value11, value21, value31);
    NORMALIZE_VALUES(value02, value12, value22, value32);
    NORMALIZE_VALUES(value03, value13, value23, value33);
  }

#undef NORMALIZE_VALUES
#undef SOFTMAX_VALUES
#define STORE_COMB(row, column, value) svst1_scatter_u32index_f32(pg, base + (row) * 4 + (column), offsets, (value))
  STORE_COMB(0, 0, value00);
  STORE_COMB(0, 1, value01);
  STORE_COMB(0, 2, value02);
  STORE_COMB(0, 3, value03);
  STORE_COMB(1, 0, value10);
  STORE_COMB(1, 1, value11);
  STORE_COMB(1, 2, value12);
  STORE_COMB(1, 3, value13);
  STORE_COMB(2, 0, value20);
  STORE_COMB(2, 1, value21);
  STORE_COMB(2, 2, value22);
  STORE_COMB(2, 3, value23);
  STORE_COMB(3, 0, value30);
  STORE_COMB(3, 1, value31);
  STORE_COMB(3, 2, value32);
  STORE_COMB(3, 3, value33);
#undef STORE_COMB
}

#endif  // __ARM_FEATURE_SVE

}  // namespace

bool deepseek_v4_mhc_sve_projection_available() {
#if defined(__ARM_FEATURE_SVE)
  return svcntw() == 4 || svcntw() == 8;
#else
  return false;
#endif
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> deepseek_v4_mhc_sve_projection(at::Tensor residual,
                                                                                    at::Tensor packed_b,
                                                                                    int64_t num_threads,
                                                                                    int64_t b_window_bytes) {
#if defined(__ARM_FEATURE_SVE)
  TORCH_CHECK(deepseek_v4_mhc_sve_projection_available(), "mHC SVE projection requires SVE VL=128 or 256 bits");
  TORCH_CHECK(residual.device().is_cpu() && residual.scalar_type() == at::kBFloat16 && residual.dim() == 3 &&
                  residual.is_contiguous(),
              "mHC SVE residual must be contiguous CPU BF16 [M,C,H]");
  TORCH_CHECK(packed_b.device().is_cpu() && packed_b.scalar_type() == at::kFloat && packed_b.dim() == 2 &&
                  packed_b.is_contiguous(),
              "mHC SVE packed-B must be contiguous CPU FP32 [K,24]");
  TORCH_CHECK(num_threads >= 0, "mHC SVE num_threads must be non-negative");
  TORCH_CHECK(b_window_bytes > 0, "mHC SVE b_window_bytes must be positive");
  const int64_t m = residual.size(0);
  const int64_t k = residual.size(1) * residual.size(2);
  TORCH_CHECK(packed_b.size(0) == k && packed_b.size(1) == kMhcProjectionN, "mHC SVE packed-B must have shape [", k,
              ",24], got ", packed_b.sizes());
  at::Tensor output = at::empty({m, kMhcProjectionN}, at::TensorOptions().dtype(at::kFloat));
  at::Tensor sqrsum = at::empty({m}, at::TensorOptions().dtype(at::kFloat));
  const int requested = num_threads > 0 ? static_cast<int>(num_threads) : 0;
  const int max_threads = requested > 0 ? requested
#if defined(_OPENMP)
                                        : omp_get_max_threads();
#else
                                        : 1;
#endif
  const int64_t mr = svcntw() == 8
#if defined(FUSED_CPP_DEEPSEEK_V4_MHC_SVE_ASM)
                         ? 7
#else
                         ? 8
#endif
                         : 4;
  const int64_t kc = KWindowElements(k, b_window_bytes);
  if (m == 0) {
    return std::make_tuple(output, sqrsum, mr, kc);
  }

  const auto* input = residual.data_ptr<c10::BFloat16>();
  const auto* weight = packed_b.data_ptr<float>();
  auto* result = output.data_ptr<float>();
  auto* sums = sqrsum.data_ptr<float>();
  const int64_t m7_tail_rows = mr == 7 ? m % 7 : 0;
  const bool pad_m7_tail = m7_tail_rows != 0 && (m + 6) / 7 >= max_threads;
  at::Tensor padded_a;
  at::Tensor padded_output;
  const c10::BFloat16* padded_input = nullptr;
  float* padded_result = nullptr;
  if (pad_m7_tail) {
    padded_a = at::empty({7, k}, residual.options());
    padded_output = at::empty({7, kMhcProjectionN}, output.options());
    auto* padded_data = padded_a.data_ptr<c10::BFloat16>();
    std::memcpy(padded_data, input + (m - m7_tail_rows) * k,
                static_cast<size_t>(m7_tail_rows * k) * sizeof(c10::BFloat16));
    std::memset(padded_data + m7_tail_rows * k, 0, static_cast<size_t>((7 - m7_tail_rows) * k) * sizeof(c10::BFloat16));
    padded_input = padded_data;
    padded_result = padded_output.data_ptr<float>();
  }
  const ProjectionBlockGeometry geometry = MakeProjectionBlockGeometry(m, mr, pad_m7_tail);
  const int64_t blocks = geometry.blocks();
  const int64_t windows = (k + kc - 1) / kc;

  const int sum_threads = EffectiveThreads(requested, m);
  if (blocks >= max_threads || windows == 1) {
    const int projection_threads = EffectiveThreads(max_threads, blocks);
    const int threads = std::max(sum_threads, projection_threads);
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static) nowait
      for (int64_t row = 0; row < m; ++row) {
        sums[row] = SquaredSum(input + row * k, k);
      }
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
      const int team = omp_get_num_threads();
#else
      const int tid = 0;
      const int team = 1;
#endif
      const int64_t block_begin = blocks * tid / team;
      const int64_t block_end = blocks * (tid + 1) / team;
      for (int64_t k_begin = 0; k_begin < k; k_begin += kc) {
        const int64_t k_length = std::min(kc, k - k_begin);
        for (int64_t block = block_begin; block < block_end; ++block) {
          const auto [m_begin, rows] = geometry.block(block);
          const bool padded_block = pad_m7_tail && block == blocks - 1;
          const auto* block_input = padded_block ? padded_input : input + m_begin * k;
          auto* block_output = padded_block ? padded_result : result + m_begin * kMhcProjectionN;
          RunBlock(block_input + k_begin, k, weight + k_begin * kMhcProjectionN, block_output, rows, k_length, mr,
                   k_begin != 0);
        }
      }
    }
  } else {
    const int64_t k_parts = std::min<int64_t>(windows, (max_threads + blocks - 1) / blocks);
    at::Tensor partial = at::empty({k_parts, m, kMhcProjectionN}, at::TensorOptions().dtype(at::kFloat));
    auto* partial_data = partial.data_ptr<float>();
    const int64_t tasks = blocks * k_parts;
    const int task_threads = EffectiveThreads(max_threads, tasks);
    const int threads = std::max(sum_threads, task_threads);
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static) nowait
      for (int64_t row = 0; row < m; ++row) {
        sums[row] = SquaredSum(input + row * k, k);
      }
#pragma omp for schedule(static)
      for (int64_t task = 0; task < tasks; ++task) {
        const int64_t part = task / blocks;
        const int64_t block = task % blocks;
        const int64_t k_begin = k * part / k_parts;
        const int64_t k_end = k * (part + 1) / k_parts;
        const auto [m_begin, rows] = geometry.block(block);
        RunBlock(input + m_begin * k + k_begin, k, weight + k_begin * kMhcProjectionN,
                 partial_data + (part * m + m_begin) * kMhcProjectionN, rows, k_end - k_begin, mr, false);
      }
    }
    const int reduce_threads = EffectiveThreads(max_threads, m);
#pragma omp parallel for num_threads(reduce_threads) schedule(static)
    for (int64_t row = 0; row < m; ++row) {
      for (int64_t column = 0; column < kMhcProjectionN; ++column) {
        float value = 0.0f;
        for (int64_t part = 0; part < k_parts; ++part) {
          value += partial_data[(part * m + row) * kMhcProjectionN + column];
        }
        result[row * kMhcProjectionN + column] = value;
      }
    }
  }
  if (pad_m7_tail) {
    std::memcpy(result + (m - m7_tail_rows) * kMhcProjectionN, padded_result,
                static_cast<size_t>(m7_tail_rows * kMhcProjectionN) * sizeof(float));
  }
  return std::make_tuple(output, sqrsum, mr, kc);
#else
  (void)residual;
  (void)packed_b;
  (void)num_threads;
  (void)b_window_bytes;
  TORCH_CHECK(false, "mHC SVE projection is unavailable in this build");
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> deepseek_v4_mhc_sve_control_postprocess(
    at::Tensor mixes, at::Tensor sqrsum, at::Tensor hc_scale, at::Tensor hc_base, int64_t rms_elements, double rms_eps,
    double pre_eps, double post_multiplier, double sinkhorn_eps, int64_t sinkhorn_repeat, int64_t num_threads) {
#if defined(__ARM_FEATURE_SVE)
  TORCH_CHECK(deepseek_v4_mhc_sve_projection_available(),
              "mHC SVE control postprocess requires SVE VL=128 or 256 bits");
  TORCH_CHECK(mixes.device().is_cpu() && mixes.scalar_type() == at::kFloat && mixes.dim() == 2 &&
                  mixes.size(1) == kMhcProjectionN && mixes.is_contiguous(),
              "mHC SVE mixes must be contiguous CPU FP32 [T,24]");
  const int64_t tokens = mixes.size(0);
  TORCH_CHECK(sqrsum.device().is_cpu() && sqrsum.scalar_type() == at::kFloat && sqrsum.dim() == 1 &&
                  sqrsum.size(0) == tokens && sqrsum.is_contiguous(),
              "mHC SVE sqrsum must be contiguous CPU FP32 [T]");
  TORCH_CHECK(hc_scale.device().is_cpu() && hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 3 &&
                  hc_scale.is_contiguous(),
              "mHC SVE hc_scale must be contiguous CPU FP32 [3]");
  TORCH_CHECK(hc_base.device().is_cpu() && hc_base.scalar_type() == at::kFloat && hc_base.numel() == kMhcProjectionN &&
                  hc_base.is_contiguous(),
              "mHC SVE hc_base must be contiguous CPU FP32 [24]");
  TORCH_CHECK(rms_elements > 0, "mHC SVE rms_elements must be positive");
  TORCH_CHECK(rms_eps >= 0.0 && pre_eps >= 0.0 && sinkhorn_eps >= 0.0, "mHC SVE eps values must be non-negative");
  TORCH_CHECK(post_multiplier > 0.0, "mHC SVE post_multiplier must be positive");
  TORCH_CHECK(sinkhorn_repeat >= 1, "mHC SVE sinkhorn_repeat must be at least one");
  TORCH_CHECK(num_threads >= 0, "mHC SVE num_threads must be non-negative");

  at::Tensor pre_mix = at::empty({tokens, 4}, mixes.options());
  at::Tensor post_mix = at::empty({tokens, 4, 1}, mixes.options());
  at::Tensor comb_mix = at::empty({tokens, 4, 4}, mixes.options());
  if (tokens == 0) {
    return std::make_tuple(pre_mix, post_mix, comb_mix);
  }

  const auto* mixes_data = mixes.data_ptr<float>();
  const auto* sums = sqrsum.data_ptr<float>();
  const auto* scale = hc_scale.data_ptr<float>();
  const auto* base = hc_base.data_ptr<float>();
  auto* pre = pre_mix.data_ptr<float>();
  auto* post = post_mix.data_ptr<float>();
  auto* comb = comb_mix.data_ptr<float>();
  const int64_t vector_blocks = (tokens + svcntw() - 1) / svcntw();
  const int requested = num_threads > 0 ? static_cast<int>(num_threads) : 0;
  const int threads = EffectiveThreads(requested, std::max(tokens / 2 + tokens % 2, vector_blocks));

#pragma omp parallel num_threads(threads)
  {
#pragma omp for schedule(static)
    for (int64_t pair = 0; pair < (tokens + 1) / 2; ++pair) {
      const int64_t row0 = pair * 2;
      const float rms0 = 1.0f / std::sqrt(sums[row0] / static_cast<float>(rms_elements) + static_cast<float>(rms_eps));
      ProcessControlRow(mixes_data + row0 * 24, rms0, scale, base, static_cast<float>(pre_eps),
                        static_cast<float>(post_multiplier), pre + row0 * 4, post + row0 * 4, comb + row0 * 16);
      const int64_t row1 = row0 + 1;
      if (row1 < tokens) {
        const float rms1 =
            1.0f / std::sqrt(sums[row1] / static_cast<float>(rms_elements) + static_cast<float>(rms_eps));
        ProcessControlRow(mixes_data + row1 * 24, rms1, scale, base, static_cast<float>(pre_eps),
                          static_cast<float>(post_multiplier), pre + row1 * 4, post + row1 * 4, comb + row1 * 16);
      }
    }

#pragma omp for schedule(static)
    for (int64_t block = 0; block < vector_blocks; ++block) {
      RunSinkhornBlock(comb, block * svcntw(), tokens, static_cast<int>(sinkhorn_repeat),
                       static_cast<float>(sinkhorn_eps));
    }
  }
  return std::make_tuple(pre_mix, post_mix, comb_mix);
#else
  (void)mixes;
  (void)sqrsum;
  (void)hc_scale;
  (void)hc_base;
  (void)rms_elements;
  (void)rms_eps;
  (void)pre_eps;
  (void)post_multiplier;
  (void)sinkhorn_eps;
  (void)sinkhorn_repeat;
  (void)num_threads;
  TORCH_CHECK(false, "mHC SVE control postprocess is unavailable in this build");
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t> deepseek_v4_mhc_sve_projection_control(
    at::Tensor residual, at::Tensor packed_b, at::Tensor hc_scale, at::Tensor hc_base, double rms_eps, double pre_eps,
    double post_multiplier, double sinkhorn_eps, int64_t sinkhorn_repeat, int64_t num_threads, int64_t b_window_bytes) {
  auto [mixes, sqrsum, mr, kc] = deepseek_v4_mhc_sve_projection(residual, packed_b, num_threads, b_window_bytes);
  const int64_t rms_elements = residual.size(1) * residual.size(2);
  auto [pre_mix, post_mix, comb_mix] =
      deepseek_v4_mhc_sve_control_postprocess(mixes, sqrsum, hc_scale, hc_base, rms_elements, rms_eps, pre_eps,
                                              post_multiplier, sinkhorn_eps, sinkhorn_repeat, num_threads);
  return std::make_tuple(pre_mix, post_mix, comb_mix, mr, kc);
}

at::Tensor deepseek_v4_mhc_sve_pre_apply_rmsnorm(at::Tensor residual, at::Tensor pre_mix, at::Tensor norm_weight,
                                                 double norm_eps, int64_t num_threads) {
#if defined(__ARM_FEATURE_SVE)
  TORCH_CHECK(deepseek_v4_mhc_sve_projection_available(), "mHC SVE pre-apply RMSNorm requires SVE VL=128 or 256 bits");
  TORCH_CHECK(residual.device().is_cpu() && residual.scalar_type() == at::kBFloat16 && residual.dim() == 3 &&
                  residual.size(1) == 4 && residual.is_contiguous(),
              "mHC SVE pre-apply residual must be contiguous CPU BF16 [T,4,H]");
  const int64_t tokens = residual.size(0);
  const int64_t h = residual.size(2);
  TORCH_CHECK(h > 0, "mHC SVE pre-apply H must be positive");
  TORCH_CHECK(pre_mix.device().is_cpu() && pre_mix.scalar_type() == at::kFloat && pre_mix.dim() == 2 &&
                  pre_mix.size(0) == tokens && pre_mix.size(1) == 4 && pre_mix.is_contiguous(),
              "mHC SVE pre_mix must be contiguous CPU FP32 [T,4]");
  TORCH_CHECK(norm_weight.device().is_cpu() && norm_weight.scalar_type() == at::kBFloat16 && norm_weight.dim() == 1 &&
                  norm_weight.size(0) == h && norm_weight.is_contiguous(),
              "mHC SVE norm_weight must be contiguous CPU BF16 [H]");
  TORCH_CHECK(norm_eps >= 0.0, "mHC SVE norm_eps must be non-negative");
  TORCH_CHECK(num_threads >= 0, "mHC SVE num_threads must be non-negative");

  at::Tensor output = at::empty({tokens, h}, residual.options());
  if (tokens == 0) {
    return output;
  }
  const auto* residual_data = residual.data_ptr<c10::BFloat16>();
  const auto* pre_data = pre_mix.data_ptr<float>();
  const auto* weight_data = norm_weight.data_ptr<c10::BFloat16>();
  auto* output_data = output.data_ptr<c10::BFloat16>();
  const int requested = num_threads > 0 ? static_cast<int>(num_threads) : 0;
  const int threads = EffectiveThreads(requested, tokens);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t token = 0; token < tokens; ++token) {
    ApplyPreMixRmsNormRow(residual_data + token * 4 * h, pre_data + token * 4, weight_data, output_data + token * h, h,
                          static_cast<float>(norm_eps));
  }
  return output;
#else
  (void)residual;
  (void)pre_mix;
  (void)norm_weight;
  (void)norm_eps;
  (void)num_threads;
  TORCH_CHECK(false, "mHC SVE pre-apply RMSNorm is unavailable in this build");
#endif
}

at::Tensor deepseek_v4_mhc_sve_post(at::Tensor layer_output, at::Tensor residual, at::Tensor post_mix,
                                    at::Tensor comb_mix, int64_t num_threads) {
#if defined(__ARM_FEATURE_SVE)
  TORCH_CHECK(deepseek_v4_mhc_sve_projection_available(), "mHC SVE post requires SVE VL=128 or 256 bits");
  TORCH_CHECK(residual.device().is_cpu() && residual.scalar_type() == at::kBFloat16 && residual.dim() == 3 &&
                  residual.size(1) == 4 && residual.is_contiguous(),
              "mHC SVE post residual must be contiguous CPU BF16 [T,4,H]");
  const int64_t tokens = residual.size(0);
  const int64_t h = residual.size(2);
  TORCH_CHECK(h > 0, "mHC SVE post H must be positive");
  TORCH_CHECK(layer_output.device().is_cpu() && layer_output.scalar_type() == at::kBFloat16 &&
                  layer_output.dim() == 2 && layer_output.size(0) == tokens && layer_output.size(1) == h &&
                  layer_output.is_contiguous(),
              "mHC SVE layer_output must be contiguous CPU BF16 [T,H]");
  TORCH_CHECK(post_mix.device().is_cpu() && post_mix.scalar_type() == at::kFloat && post_mix.dim() == 3 &&
                  post_mix.size(0) == tokens && post_mix.size(1) == 4 && post_mix.size(2) == 1 &&
                  post_mix.is_contiguous(),
              "mHC SVE post_mix must be contiguous CPU FP32 [T,4,1]");
  TORCH_CHECK(comb_mix.device().is_cpu() && comb_mix.scalar_type() == at::kFloat && comb_mix.dim() == 3 &&
                  comb_mix.size(0) == tokens && comb_mix.size(1) == 4 && comb_mix.size(2) == 4 &&
                  comb_mix.is_contiguous(),
              "mHC SVE comb_mix must be contiguous CPU FP32 [T,4,4]");
  TORCH_CHECK(num_threads >= 0, "mHC SVE num_threads must be non-negative");

  at::Tensor output = at::empty_like(residual);
  if (tokens == 0) {
    return output;
  }
  const auto* layer_data = layer_output.data_ptr<c10::BFloat16>();
  const auto* residual_data = residual.data_ptr<c10::BFloat16>();
  const auto* post_data = post_mix.data_ptr<float>();
  const auto* comb_data = comb_mix.data_ptr<float>();
  auto* output_data = output.data_ptr<c10::BFloat16>();
  const int requested = num_threads > 0 ? static_cast<int>(num_threads) : 0;
  const int threads = EffectiveThreads(requested, tokens);
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t token = 0; token < tokens; ++token) {
    ApplyPostRowU2(residual_data + token * 4 * h, layer_data + token * h, post_data + token * 4, comb_data + token * 16,
                   output_data + token * 4 * h, h);
  }
  return output;
#else
  (void)layer_output;
  (void)residual;
  (void)post_mix;
  (void)comb_mix;
  (void)num_threads;
  TORCH_CHECK(false, "mHC SVE post is unavailable in this build");
#endif
}

std::tuple<at::Tensor, at::Tensor> deepseek_v4_mhc_sve_post_head_rmsnorm(at::Tensor layer_output, at::Tensor residual,
                                                                         at::Tensor post_mix, at::Tensor comb_mix,
                                                                         at::Tensor packed_head, at::Tensor head_scale,
                                                                         at::Tensor head_base, at::Tensor norm_weight,
                                                                         double rms_eps, double head_eps,
                                                                         double norm_eps, int64_t num_threads) {
#if defined(__ARM_FEATURE_SVE)
  TORCH_CHECK(deepseek_v4_mhc_sve_projection_available(), "mHC SVE post-head requires SVE VL=128 or 256 bits");
  TORCH_CHECK(residual.device().is_cpu() && residual.scalar_type() == at::kBFloat16 && residual.dim() == 3 &&
                  residual.size(1) == 4 && residual.is_contiguous(),
              "mHC SVE post-head residual must be contiguous CPU BF16 [T,4,H]");
  const int64_t tokens = residual.size(0);
  const int64_t h = residual.size(2);
  const int64_t k = 4 * h;
  TORCH_CHECK(h > 0, "mHC SVE post-head H must be positive");
  TORCH_CHECK(packed_head.device().is_cpu() && packed_head.scalar_type() == at::kFloat && packed_head.dim() == 2 &&
                  packed_head.size(0) == k && packed_head.size(1) == 4 && packed_head.is_contiguous(),
              "mHC SVE packed head must be contiguous CPU FP32 [4H,4]");
  TORCH_CHECK(head_scale.device().is_cpu() && head_scale.scalar_type() == at::kFloat && head_scale.dim() == 1 &&
                  head_scale.size(0) == 1 && head_scale.is_contiguous(),
              "mHC SVE head_scale must be contiguous CPU FP32 [1]");
  TORCH_CHECK(head_base.device().is_cpu() && head_base.scalar_type() == at::kFloat && head_base.dim() == 1 &&
                  head_base.size(0) == 4 && head_base.is_contiguous(),
              "mHC SVE head_base must be contiguous CPU FP32 [4]");
  TORCH_CHECK(norm_weight.device().is_cpu() && norm_weight.scalar_type() == at::kBFloat16 && norm_weight.dim() == 1 &&
                  norm_weight.size(0) == h && norm_weight.is_contiguous(),
              "mHC SVE post-head norm_weight must be contiguous CPU BF16 [H]");
  TORCH_CHECK(rms_eps >= 0.0 && head_eps >= 0.0 && norm_eps >= 0.0,
              "mHC SVE post-head eps values must be non-negative");
  TORCH_CHECK(num_threads >= 0, "mHC SVE post-head num_threads must be non-negative");

  at::Tensor final_residual = deepseek_v4_mhc_sve_post(layer_output, residual, post_mix, comb_mix, num_threads);
  at::Tensor hidden_states = at::empty({tokens, h}, residual.options());
  if (tokens == 0) {
    return std::make_tuple(hidden_states, final_residual);
  }

  at::Tensor mixes = at::empty({tokens, 4}, at::TensorOptions().dtype(at::kFloat));
  at::Tensor sqrsum = at::empty({tokens}, at::TensorOptions().dtype(at::kFloat));
  const int64_t full_blocks = tokens / 12;
  const int64_t tail_rows = tokens % 12;
  const int64_t blocks = full_blocks + (tail_rows != 0 ? 1 : 0);
  const auto* residual_data = final_residual.data_ptr<c10::BFloat16>();
  const auto* weight_data = packed_head.data_ptr<float>();
  auto* mixes_data = mixes.data_ptr<float>();
  auto* sums = sqrsum.data_ptr<float>();
  at::Tensor padded_input;
  at::Tensor padded_output;
  const c10::BFloat16* tail_input = nullptr;
  float* tail_output = nullptr;
  if (tail_rows != 0) {
    padded_input = at::empty({12, k}, residual.options());
    padded_output = at::empty({12, 4}, mixes.options());
    auto* padded_data = padded_input.data_ptr<c10::BFloat16>();
    std::memcpy(padded_data, residual_data + full_blocks * 12 * k,
                static_cast<size_t>(tail_rows * k) * sizeof(c10::BFloat16));
    std::memset(padded_data + tail_rows * k, 0, static_cast<size_t>((12 - tail_rows) * k) * sizeof(c10::BFloat16));
    tail_input = padded_data;
    tail_output = padded_output.data_ptr<float>();
  }

  const int requested = num_threads > 0 ? static_cast<int>(num_threads) : 0;
  const int threads = EffectiveThreads(requested, std::max(tokens, blocks));
#pragma omp parallel num_threads(threads)
  {
#pragma omp for schedule(static) nowait
    for (int64_t token = 0; token < tokens; ++token) {
      sums[token] = SquaredSum(residual_data + token * k, k);
    }

#pragma omp for schedule(static)
    for (int64_t block = 0; block < blocks; ++block) {
      const bool tail = block == full_blocks && tail_rows != 0;
      const auto* block_input = tail ? tail_input : residual_data + block * 12 * k;
      auto* block_output = tail ? tail_output : mixes_data + block * 12 * 4;
      HeadProjectionM12N4Neon(block_input, k, weight_data, block_output, k);
    }
  }
  if (tail_rows != 0) {
    std::memcpy(mixes_data + full_blocks * 12 * 4, tail_output, static_cast<size_t>(tail_rows * 4) * sizeof(float));
  }

  const float scale = head_scale.data_ptr<float>()[0];
  const auto* base = head_base.data_ptr<float>();
  const auto* norm = norm_weight.data_ptr<c10::BFloat16>();
  auto* hidden = hidden_states.data_ptr<c10::BFloat16>();
  const int output_threads = EffectiveThreads(requested, tokens);
#pragma omp parallel for num_threads(output_threads) schedule(static)
  for (int64_t token = 0; token < tokens; ++token) {
    alignas(16) float pre[4];
    const float rms_scale = 1.0f / std::sqrt(sums[token] / static_cast<float>(k) + static_cast<float>(rms_eps));
    ProcessHeadControl(mixes_data + token * 4, rms_scale, scale, base, static_cast<float>(head_eps), pre);
    ApplyPreMixRmsNormRow(residual_data + token * k, pre, norm, hidden + token * h, h, static_cast<float>(norm_eps));
  }
  return std::make_tuple(hidden_states, final_residual);
#else
  (void)layer_output;
  (void)residual;
  (void)post_mix;
  (void)comb_mix;
  (void)packed_head;
  (void)head_scale;
  (void)head_base;
  (void)norm_weight;
  (void)rms_eps;
  (void)head_eps;
  (void)norm_eps;
  (void)num_threads;
  TORCH_CHECK(false, "mHC SVE post-head is unavailable in this build");
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t> deepseek_v4_mhc_sve_pre_rmsnorm(
    at::Tensor residual, at::Tensor packed_b, at::Tensor hc_scale, at::Tensor hc_base, at::Tensor norm_weight,
    double rms_eps, double pre_eps, double post_multiplier, double sinkhorn_eps, int64_t sinkhorn_repeat,
    double norm_eps, int64_t num_threads, int64_t b_window_bytes) {
#if defined(__ARM_FEATURE_SVE)
  auto [mixes, sqrsum, mr, kc] = deepseek_v4_mhc_sve_projection(residual, packed_b, num_threads, b_window_bytes);
  const int64_t tokens = residual.size(0);
  const int64_t channels = residual.size(1);
  const int64_t h = residual.size(2);
  TORCH_CHECK(channels == 4 && h > 0, "mHC SVE team-reuse pre requires residual shape [T,4,H]");
  TORCH_CHECK(hc_scale.device().is_cpu() && hc_scale.scalar_type() == at::kFloat && hc_scale.numel() == 3 &&
                  hc_scale.is_contiguous(),
              "mHC SVE hc_scale must be contiguous CPU FP32 [3]");
  TORCH_CHECK(hc_base.device().is_cpu() && hc_base.scalar_type() == at::kFloat && hc_base.numel() == kMhcProjectionN &&
                  hc_base.is_contiguous(),
              "mHC SVE hc_base must be contiguous CPU FP32 [24]");
  TORCH_CHECK(norm_weight.device().is_cpu() && norm_weight.scalar_type() == at::kBFloat16 && norm_weight.dim() == 1 &&
                  norm_weight.size(0) == h && norm_weight.is_contiguous(),
              "mHC SVE norm_weight must be contiguous CPU BF16 [H]");
  TORCH_CHECK(rms_eps >= 0.0 && pre_eps >= 0.0 && sinkhorn_eps >= 0.0 && norm_eps >= 0.0,
              "mHC SVE eps values must be non-negative");
  TORCH_CHECK(post_multiplier > 0.0, "mHC SVE post_multiplier must be positive");
  TORCH_CHECK(sinkhorn_repeat >= 1, "mHC SVE sinkhorn_repeat must be at least one");
  TORCH_CHECK(num_threads >= 0, "mHC SVE num_threads must be non-negative");

  at::Tensor pre_mix = at::empty({tokens, 4}, mixes.options());
  at::Tensor post_mix = at::empty({tokens, 4, 1}, mixes.options());
  at::Tensor comb_mix = at::empty({tokens, 4, 4}, mixes.options());
  at::Tensor normed_input = at::empty({tokens, h}, residual.options());
  if (tokens == 0) {
    return std::make_tuple(post_mix, comb_mix, normed_input, mr, kc);
  }

  const auto* residual_data = residual.data_ptr<c10::BFloat16>();
  const auto* mixes_data = mixes.data_ptr<float>();
  const auto* sums = sqrsum.data_ptr<float>();
  const auto* scale = hc_scale.data_ptr<float>();
  const auto* base = hc_base.data_ptr<float>();
  const auto* weight = norm_weight.data_ptr<c10::BFloat16>();
  auto* pre = pre_mix.data_ptr<float>();
  auto* post = post_mix.data_ptr<float>();
  auto* comb = comb_mix.data_ptr<float>();
  auto* output = normed_input.data_ptr<c10::BFloat16>();
  const int64_t pairs = (tokens + 1) / 2;
  const int64_t vector_blocks = (tokens + svcntw() - 1) / svcntw();
  const int requested = num_threads > 0 ? static_cast<int>(num_threads) : 0;
  const int threads = EffectiveThreads(requested, std::max({pairs, vector_blocks, tokens}));

#pragma omp parallel num_threads(threads)
  {
#pragma omp for schedule(static)
    for (int64_t pair = 0; pair < pairs; ++pair) {
      const int64_t row0 = pair * 2;
      const float rms0 = 1.0f / std::sqrt(sums[row0] / static_cast<float>(channels * h) + static_cast<float>(rms_eps));
      ProcessControlRow(mixes_data + row0 * 24, rms0, scale, base, static_cast<float>(pre_eps),
                        static_cast<float>(post_multiplier), pre + row0 * 4, post + row0 * 4, comb + row0 * 16);
      const int64_t row1 = row0 + 1;
      if (row1 < tokens) {
        const float rms1 =
            1.0f / std::sqrt(sums[row1] / static_cast<float>(channels * h) + static_cast<float>(rms_eps));
        ProcessControlRow(mixes_data + row1 * 24, rms1, scale, base, static_cast<float>(pre_eps),
                          static_cast<float>(post_multiplier), pre + row1 * 4, post + row1 * 4, comb + row1 * 16);
      }
    }

#pragma omp for schedule(static)
    for (int64_t block = 0; block < vector_blocks; ++block) {
      RunSinkhornBlock(comb, block * svcntw(), tokens, static_cast<int>(sinkhorn_repeat),
                       static_cast<float>(sinkhorn_eps));
    }

#pragma omp for schedule(static)
    for (int64_t token = 0; token < tokens; ++token) {
      ApplyPreMixRmsNormRow(residual_data + token * channels * h, pre + token * 4, weight, output + token * h, h,
                            static_cast<float>(norm_eps));
    }
  }
  return std::make_tuple(post_mix, comb_mix, normed_input, mr, kc);
#else
  (void)residual;
  (void)packed_b;
  (void)hc_scale;
  (void)hc_base;
  (void)norm_weight;
  (void)rms_eps;
  (void)pre_eps;
  (void)post_multiplier;
  (void)sinkhorn_eps;
  (void)sinkhorn_repeat;
  (void)norm_eps;
  (void)num_threads;
  (void)b_window_bytes;
  TORCH_CHECK(false, "mHC SVE team-reuse pre is unavailable in this build");
#endif
}
