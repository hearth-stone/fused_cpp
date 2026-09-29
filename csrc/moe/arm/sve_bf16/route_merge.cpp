#include "../../common/route_merge.h"
#include "vector_length.h"

#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace fused_cpp::moe_route_merge {
namespace {

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)

#if defined(__GNUC__) || defined(__clang__)
#define FUSED_CPP_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define FUSED_CPP_ALWAYS_INLINE inline
#endif

FUSED_CPP_ALWAYS_INLINE svfloat32_t load_f32(svbool_t pg, const float* ptr) { return svld1_f32(pg, ptr); }

FUSED_CPP_ALWAYS_INLINE svfloat32_t load_f32(svbool_t pg, const uint16_t* ptr) {
  const svuint32_t bits = svld1uh_u32(pg, ptr);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, bits, 16));
}

FUSED_CPP_ALWAYS_INLINE svuint32_t f32_to_bf16_bits(svbool_t pg, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  const svuint32_t bias = svadd_n_u32_x(pg, lsb, 0x7fff);
  return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, bias), 16);
}

FUSED_CPP_ALWAYS_INLINE void store_bf16(svbool_t pg, uint16_t* ptr, svfloat32_t value) {
  svst1h_u32(pg, ptr, f32_to_bf16_bits(pg, value));
}

constexpr int largest_power_of_two_less_than(int value) {
  int power = 1;
  while ((power << 1) < value) {
    power <<= 1;
  }
  return power;
}

template <typename Src, int Slot>
FUSED_CPP_ALWAYS_INLINE svfloat32_t weighted_leaf(svbool_t pg, const Src* token_src, const float* weights,
                                                  int64_t hidden_size, int64_t offset) {
  const Src* src = token_src + static_cast<int64_t>(Slot) * hidden_size + offset;
  return svmul_n_f32_x(pg, load_f32(pg, src), weights[Slot]);
}

// Splitting each range at its largest power-of-two prefix creates adjacent
// pairs first. TopK=6 therefore becomes ((0+1)+(2+3))+(4+5).
template <typename Src, int Begin, int Count>
FUSED_CPP_ALWAYS_INLINE svfloat32_t reduce_fixed(svbool_t pg, const Src* token_src, const float* weights,
                                                 int64_t hidden_size, int64_t offset) {
  static_assert(Count > 0);
  if constexpr (Count == 1) {
    return weighted_leaf<Src, Begin>(pg, token_src, weights, hidden_size, offset);
  } else {
    constexpr int kLeftCount = largest_power_of_two_less_than(Count);
    const svfloat32_t lhs = reduce_fixed<Src, Begin, kLeftCount>(pg, token_src, weights, hidden_size, offset);
    const svfloat32_t rhs =
        reduce_fixed<Src, Begin + kLeftCount, Count - kLeftCount>(pg, token_src, weights, hidden_size, offset);
    return svadd_f32_x(pg, lhs, rhs);
  }
}

template <typename Src, int TopK>
FUSED_CPP_ALWAYS_INLINE void merge_fixed_block(svbool_t pg, const Src* token_src, const float* weights, uint16_t* dst,
                                               int64_t hidden_size, int64_t offset) {
  static_assert(TopK == 2 || TopK == 4 || TopK == 6 || TopK == 8);
  store_bf16(pg, dst + offset, reduce_fixed<Src, 0, TopK>(pg, token_src, weights, hidden_size, offset));
}

template <typename Src>
FUSED_CPP_ALWAYS_INLINE void merge_dynamic_block(svbool_t pg, const Src* token_src, const float* weights, uint16_t* dst,
                                                 int64_t top_k, int64_t hidden_size, int64_t offset) {
  svfloat32_t acc = svdup_f32(0.0f);
  for (int64_t slot = 0; slot < top_k; ++slot) {
    acc = svmla_n_f32_x(pg, acc, load_f32(pg, token_src + slot * hidden_size + offset), weights[slot]);
  }
  store_bf16(pg, dst + offset, acc);
}

template <typename Src, int TopK>
void merge_fixed_range(const Src* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                       int64_t token_end, int64_t hidden_size) {
  constexpr int64_t vl = fused_cpp::moe_sve::kF32Lanes;
  const svbool_t all = svptrue_b32();
  for (int64_t token = token_begin; token < token_end; ++token) {
    const Src* token_src = route_output + token * TopK * hidden_size;
    const float* token_weights = weights + token * TopK;
    uint16_t* dst = output + token * hidden_size;
    int64_t offset = 0;
    for (; offset + vl <= hidden_size; offset += vl) {
      merge_fixed_block<Src, TopK>(all, token_src, token_weights, dst, hidden_size, offset);
    }
    for (; offset < hidden_size; offset += vl) {
      const svbool_t pg = svwhilelt_b32(offset, hidden_size);
      merge_fixed_block<Src, TopK>(pg, token_src, token_weights, dst, hidden_size, offset);
    }
  }
}

template <typename Src>
void merge_dynamic_range(const Src* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                         int64_t token_end, int64_t top_k, int64_t hidden_size) {
  constexpr int64_t vl = fused_cpp::moe_sve::kF32Lanes;
  const svbool_t all = svptrue_b32();
  for (int64_t token = token_begin; token < token_end; ++token) {
    const Src* token_src = route_output + token * top_k * hidden_size;
    const float* token_weights = weights + token * top_k;
    uint16_t* dst = output + token * hidden_size;
    int64_t offset = 0;
    for (; offset + vl <= hidden_size; offset += vl) {
      merge_dynamic_block<Src>(all, token_src, token_weights, dst, top_k, hidden_size, offset);
    }
    for (; offset < hidden_size; offset += vl) {
      const svbool_t pg = svwhilelt_b32(offset, hidden_size);
      merge_dynamic_block<Src>(pg, token_src, token_weights, dst, top_k, hidden_size, offset);
    }
  }
}

template <typename Src>
void dispatch_top_k(const Src* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                    int64_t token_end, int64_t top_k, int64_t hidden_size) {
  switch (top_k) {
    case 2:
      merge_fixed_range<Src, 2>(route_output, weights, output, token_begin, token_end, hidden_size);
      return;
    case 4:
      merge_fixed_range<Src, 4>(route_output, weights, output, token_begin, token_end, hidden_size);
      return;
    case 6:
      merge_fixed_range<Src, 6>(route_output, weights, output, token_begin, token_end, hidden_size);
      return;
    case 8:
      merge_fixed_range<Src, 8>(route_output, weights, output, token_begin, token_end, hidden_size);
      return;
    default:
      merge_dynamic_range<Src>(route_output, weights, output, token_begin, token_end, top_k, hidden_size);
  }
}

template <typename Src>
void dispatch(const Src* route_output, const float* weights, uint16_t* output, int64_t token_begin, int64_t token_end,
              int64_t top_k, int64_t hidden_size) {
  if (route_output == nullptr || weights == nullptr || output == nullptr) {
    throw std::invalid_argument("SVE route merge received a null pointer");
  }
  if (token_begin < 0 || token_end < 0 || top_k <= 0 || hidden_size <= 0) {
    throw std::invalid_argument("SVE route merge received an invalid range or shape");
  }
  if (token_begin >= token_end) {
    return;
  }
  dispatch_top_k<Src>(route_output, weights, output, token_begin, token_end, top_k, hidden_size);
}

#endif

}  // namespace

bool sve_available() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
  return true;
#else
  return false;
#endif
}

void merge_f32_sve(const float* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                   int64_t token_end, int64_t top_k, int64_t hidden_size) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
  dispatch(route_output, weights, output, token_begin, token_end, top_k, hidden_size);
#else
  (void)route_output;
  (void)weights;
  (void)output;
  (void)token_begin;
  (void)token_end;
  (void)top_k;
  (void)hidden_size;
  throw std::runtime_error("SVE route merge is unavailable in this build");
#endif
}

void merge_bf16_sve(const uint16_t* route_output, const float* weights, uint16_t* output, int64_t token_begin,
                    int64_t token_end, int64_t top_k, int64_t hidden_size) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
  dispatch(route_output, weights, output, token_begin, token_end, top_k, hidden_size);
#else
  (void)route_output;
  (void)weights;
  (void)output;
  (void)token_begin;
  (void)token_end;
  (void)top_k;
  (void)hidden_size;
  throw std::runtime_error("SVE route merge is unavailable in this build");
#endif
}

}  // namespace fused_cpp::moe_route_merge
