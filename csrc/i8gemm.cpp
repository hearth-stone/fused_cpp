#include <torch/extension.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#if defined(FUSED_CPP_HAS_I8GEMM)
extern "C" {
#include "i8gemm.h"
}
#endif

namespace {

#if defined(FUSED_CPP_I8GEMM_BACKEND)
constexpr const char* kCompiledBackend = FUSED_CPP_I8GEMM_BACKEND;
#else
constexpr const char* kCompiledBackend = "fallback";
#endif

int64_t round_up_int64(int64_t value, int64_t quantum) { return ((value + quantum - 1) / quantum) * quantum; }

template <typename Function>
void openmp_parallel_for_rows(int64_t rows, int64_t requested_threads, const Function& function) {
  if (rows <= 0) {
    return;
  }
#if defined(_OPENMP)
  const int max_threads = requested_threads > 0
                              ? static_cast<int>(std::min<int64_t>(requested_threads, std::numeric_limits<int>::max()))
                              : omp_get_max_threads();
  const int num_threads = static_cast<int>(std::min<int64_t>(rows, std::max(max_threads, 1)));
#pragma omp parallel num_threads(num_threads)
  {
    const int thread_id = omp_get_thread_num();
    const int team_size = omp_get_num_threads();
    const int64_t begin = rows * thread_id / team_size;
    const int64_t end = rows * (thread_id + 1) / team_size;
    function(begin, end);
  }
#else
  (void)requested_threads;
  function(0, rows);
#endif
}

float read_scalar_as_float(const at::Tensor& tensor, int64_t index) {
  switch (tensor.scalar_type()) {
    case at::kFloat:
      return tensor.data_ptr<float>()[index];
    case at::kBFloat16:
      return static_cast<float>(tensor.data_ptr<c10::BFloat16>()[index]);
    case at::kHalf:
      return static_cast<float>(tensor.data_ptr<c10::Half>()[index]);
    default:
      TORCH_CHECK(false, "i8gemm: unsupported scalar dtype");
  }
}

void check_cpu_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), "i8gemm: ", name, " must be a CPU tensor");
}

void check_backend_request(const std::string& backend) {
  TORCH_CHECK(backend == "auto" || backend == "neon" || backend == "sve",
              "i8gemm.prepare: backend must be one of auto/neon/sve, got ", backend);
#if defined(FUSED_CPP_HAS_I8GEMM)
  if (backend != "auto") {
    TORCH_CHECK(backend == kCompiledBackend, "i8gemm.prepare: requested backend=", backend,
                " but this extension was built with backend=", kCompiledBackend);
  }
#else
  TORCH_CHECK(backend == "auto", "i8gemm.prepare: explicit backend=", backend,
              " requires an AArch64 i8gemm build; current backend=fallback");
#endif
}

at::Tensor prepare_weight_scale(const at::Tensor& weight_scale, int64_t N) {
  TORCH_CHECK(weight_scale.defined(), "i8gemm.prepare: weight_scale must be defined");
  check_cpu_tensor(weight_scale, "weight_scale");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat || weight_scale.scalar_type() == at::kBFloat16 ||
                  weight_scale.scalar_type() == at::kHalf,
              "i8gemm.prepare: weight_scale dtype must be float32/bfloat16/float16");
  TORCH_CHECK(weight_scale.numel() == 1 || weight_scale.numel() == N,
              "i8gemm.prepare: weight_scale must be scalar or shape [N], got numel=", weight_scale.numel(),
              " for N=", N);

  at::Tensor scale_contig = weight_scale.contiguous();
  at::Tensor scale_f32 = at::empty({N}, at::TensorOptions().dtype(at::kFloat));
  float* dst = scale_f32.data_ptr<float>();
  if (scale_contig.numel() == 1) {
    const float s = read_scalar_as_float(scale_contig, 0);
    std::fill(dst, dst + N, s);
  } else {
    for (int64_t i = 0; i < N; ++i) {
      dst[i] = read_scalar_as_float(scale_contig, i);
    }
  }
  return scale_f32;
}

#if defined(FUSED_CPP_HAS_I8GEMM)
at::Tensor pack_weight_i8gemm(const at::Tensor& weight, int64_t Kp, int64_t Np) {
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  at::Tensor weight_kn = at::zeros({Kp, Np}, at::TensorOptions().dtype(at::kChar));
  const auto* src = weight.data_ptr<int8_t>();
  auto* b_pad = weight_kn.data_ptr<int8_t>();
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      b_pad[k * Np + n] = src[n * K + k];
    }
  }

  at::Tensor packed = at::empty({Kp * Np}, at::TensorOptions().dtype(at::kChar));
  i8_pack_B(b_pad, packed.data_ptr<int8_t>(), static_cast<int>(Kp), static_cast<int>(Np));
  return packed;
}
#else
at::Tensor pack_weight_fallback(const at::Tensor& weight, int64_t Kp, int64_t Np) {
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  at::Tensor packed = at::zeros({Kp, Np}, at::TensorOptions().dtype(at::kChar));
  const auto* src = weight.data_ptr<int8_t>();
  auto* dst = packed.data_ptr<int8_t>();
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      dst[k * Np + n] = src[n * K + k];
    }
  }
  return packed.reshape({Kp * Np});
}
#endif

template <typename scalar_t>
void write_scaled_output(scalar_t* out, const float* acc, const float* x_scale, const float* w_scale, const float* bias,
                         int64_t M, int64_t N, int64_t Np, int64_t nthreads) {
  openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      const float row_scale = x_scale[m];
      const float* acc_row = acc + m * Np;
      scalar_t* out_row = out + m * N;
      for (int64_t n = 0; n < N; ++n) {
        float value = acc_row[n] * row_scale * w_scale[n];
        if (bias != nullptr) {
          value += bias[n];
        }
        out_row[n] = static_cast<scalar_t>(value);
      }
    }
  });
}

#if defined(__ARM_FEATURE_SVE)
svfloat32_t load_bf16_sve(svbool_t pg, const c10::BFloat16* source) {
  const auto* bits = reinterpret_cast<const uint16_t*>(source);
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, bits), 16));
}

void quantize_bf16_rows_sve(const c10::BFloat16* input, int8_t* output, float* scales, int64_t M, int64_t K, int64_t Kp,
                            int64_t nthreads) {
  openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
    const int64_t vl = svcntw();
    for (int64_t row = begin; row < end; ++row) {
      const c10::BFloat16* source = input + row * K;
      svfloat32_t maximum = svdup_f32(0.0f);
      for (int64_t column = 0; column < K; column += vl) {
        const svbool_t pg = svwhilelt_b32(column, K);
        maximum = svmax_f32_m(pg, maximum, svabs_f32_x(pg, load_bf16_sve(pg, source + column)));
      }
      const float max_value = svmaxv_f32(svptrue_b32(), maximum);
      const float scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
      scales[row] = scale;
      int8_t* destination = output + row * Kp;
      const float inverse_scale = 1.0f / scale;
      for (int64_t column = 0; column < K; column += vl) {
        const svbool_t pg = svwhilelt_b32(column, K);
        svfloat32_t value = svmul_n_f32_x(pg, load_bf16_sve(pg, source + column), inverse_scale);
        value = svmax_n_f32_x(pg, svmin_n_f32_x(pg, value, 127.0f), -127.0f);
        svst1b_s32(pg, destination + column, svcvt_s32_f32_x(pg, svrintn_f32_x(pg, value)));
      }
    }
  });
}

void write_scaled_output_bf16_sve(c10::BFloat16* output, const float* acc, const float* x_scale, const float* w_scale,
                                  const float* bias, int64_t M, int64_t N, int64_t Np, int64_t nthreads) {
  openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
    const int64_t vl = svcntw();
    auto* output_bits = reinterpret_cast<uint16_t*>(output);
    for (int64_t row = begin; row < end; ++row) {
      const svfloat32_t activation = svdup_f32(x_scale[row]);
      for (int64_t column = 0; column < N; column += vl) {
        const svbool_t pg = svwhilelt_b32(column, N);
        svfloat32_t value = svmul_f32_x(pg, svld1_f32(pg, acc + row * Np + column),
                                        svmul_f32_x(pg, activation, svld1_f32(pg, w_scale + column)));
        if (bias != nullptr) {
          value = svadd_f32_x(pg, value, svld1_f32(pg, bias + column));
        }
        const svuint32_t bits = svreinterpret_u32_f32(value);
        const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
        svst1h_u32(pg, output_bits + row * N + column,
                   svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fff)), 16));
      }
    }
  });
}
#endif

#if !defined(FUSED_CPP_HAS_I8GEMM)
void fallback_int8_accum_f32(const int8_t* a, const int8_t* b_kn, float* c, int64_t M, int64_t K, int64_t N, int64_t Kp,
                             int64_t Np, int64_t nthreads) {
  openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      const int8_t* a_row = a + m * Kp;
      float* c_row = c + m * Np;
      for (int64_t n = 0; n < N; ++n) {
        int32_t sum = 0;
        for (int64_t k = 0; k < K; ++k) {
          sum += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(b_kn[k * Np + n]);
        }
        c_row[n] = static_cast<float>(sum);
      }
    }
  });
}
#endif

at::Tensor prepare_bias(c10::optional<at::Tensor> bias, int64_t N) {
  if (!bias.has_value() || !bias.value().defined()) {
    return at::Tensor();
  }
  at::Tensor b = bias.value();
  check_cpu_tensor(b, "bias");
  TORCH_CHECK(b.dim() == 1, "i8gemm.dynamic_scaled_mm: bias must be 1D [N], got dim=", b.dim());
  TORCH_CHECK(b.size(0) == N, "i8gemm.dynamic_scaled_mm: bias length must equal N=", N, ", got ", b.size(0));
  TORCH_CHECK(b.scalar_type() == at::kFloat || b.scalar_type() == at::kBFloat16 || b.scalar_type() == at::kHalf,
              "i8gemm.dynamic_scaled_mm: bias dtype must be float32/bfloat16/float16");
  return b.to(at::kFloat).contiguous();
}

#if defined(FUSED_CPP_HAS_I8GEMM) && defined(__ARM_FEATURE_SVE)
at::Tensor pad_f32_vector(const at::Tensor& source, int64_t padded_size) {
  if (!source.defined() || source.numel() == padded_size) {
    return source;
  }
  at::Tensor padded = at::zeros({padded_size}, source.options().dtype(at::kFloat));
  std::copy_n(source.data_ptr<float>(), source.numel(), padded.data_ptr<float>());
  return padded;
}

template <typename scalar_t>
void copy_logical_columns(const at::Tensor& padded, at::Tensor& output, int64_t M, int64_t N, int64_t Np,
                          int64_t nthreads) {
  const auto* source = padded.data_ptr<scalar_t>();
  auto* destination = output.data_ptr<scalar_t>();
  openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      std::copy_n(source + row * Np, N, destination + row * N);
    }
  });
}
#endif

}  // namespace

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, std::string> i8gemm_prepare(
    at::Tensor weight, at::Tensor weight_scale, std::string backend) {
  check_backend_request(backend);
  TORCH_CHECK(weight.dim() == 2, "i8gemm.prepare: weight must be 2D [N, K], got dim=", weight.dim());
  check_cpu_tensor(weight, "weight");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "i8gemm.prepare: weight dtype must be torch.int8");
  TORCH_CHECK(weight.is_contiguous(), "i8gemm.prepare: weight must be contiguous in [N, K] layout");

  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(N > 0 && K > 0, "i8gemm.prepare: N and K must be positive, got N=", N, " K=", K);
  TORCH_CHECK(N <= std::numeric_limits<int>::max() && K <= std::numeric_limits<int>::max(),
              "i8gemm.prepare: N/K exceed int32 kernel limits");

#if defined(FUSED_CPP_HAS_I8GEMM) && defined(__ARM_FEATURE_SVE)
  const int64_t n_tile = static_cast<int64_t>(svcntb() / 16) * 8;
#else
  const int64_t n_tile = 8;
#endif
  const int64_t Kp = std::max<int64_t>(16, round_up_int64(K, 16));
  const int64_t Np = std::max<int64_t>(n_tile, round_up_int64(N, n_tile));
  TORCH_CHECK(Kp <= std::numeric_limits<int>::max() && Np <= std::numeric_limits<int>::max(),
              "i8gemm.prepare: padded N/K exceed int32 kernel limits");

  at::Tensor weight_contig = weight.contiguous();
  at::Tensor packed;
#if defined(FUSED_CPP_HAS_I8GEMM)
  packed = pack_weight_i8gemm(weight_contig, Kp, Np);
  const std::string selected_backend = kCompiledBackend;
#else
  packed = pack_weight_fallback(weight_contig, Kp, Np);
  const std::string selected_backend = "fallback";
#endif
  at::Tensor scale_f32 = prepare_weight_scale(weight_scale, N);

  return std::make_tuple(packed, scale_f32, K, N, Kp, Np, selected_backend);
}

void i8gemm_dynamic_scaled_mm(at::Tensor output, at::Tensor input, at::Tensor packed_weight, at::Tensor weight_scale,
                              c10::optional<at::Tensor> bias, int64_t K, int64_t N, int64_t Kp, int64_t Np,
                              int64_t nthreads) {
  TORCH_CHECK(input.dim() == 2, "i8gemm.dynamic_scaled_mm: input must be 2D [M, K], got dim=", input.dim());
  check_cpu_tensor(input, "input");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16,
              "i8gemm.dynamic_scaled_mm: input dtype must be float32/bfloat16");
  TORCH_CHECK(input.is_contiguous(), "i8gemm.dynamic_scaled_mm: input must be contiguous");
  TORCH_CHECK(output.dim() == 2, "i8gemm.dynamic_scaled_mm: output must be 2D [M, N], got dim=", output.dim());
  check_cpu_tensor(output, "output");
  check_cpu_tensor(packed_weight, "packed_weight");
  check_cpu_tensor(weight_scale, "weight_scale");
  TORCH_CHECK(output.scalar_type() == at::kFloat || output.scalar_type() == at::kBFloat16,
              "i8gemm.dynamic_scaled_mm: output dtype must be float32/bfloat16");
  TORCH_CHECK(output.is_contiguous(), "i8gemm.dynamic_scaled_mm: output must be contiguous");
  TORCH_CHECK(
      packed_weight.scalar_type() == at::kChar && packed_weight.is_contiguous() && packed_weight.numel() == Kp * Np,
      "i8gemm.dynamic_scaled_mm: invalid packed_weight");
  TORCH_CHECK(weight_scale.scalar_type() == at::kFloat && weight_scale.is_contiguous() && weight_scale.numel() == N,
              "i8gemm.dynamic_scaled_mm: weight_scale must be contiguous float32 [N]");

  const int64_t M = input.size(0);
  TORCH_CHECK(input.size(1) == K, "i8gemm.dynamic_scaled_mm: input K mismatch, expected ", K, ", got ", input.size(1));
  TORCH_CHECK(output.size(0) == M && output.size(1) == N, "i8gemm.dynamic_scaled_mm: output shape must be [", M, ", ",
              N, "], got [", output.size(0), ", ", output.size(1), "]");
  TORCH_CHECK(K > 0 && N > 0 && Kp >= K && Np >= N, "i8gemm.dynamic_scaled_mm: invalid K/N metadata");
  TORCH_CHECK(Kp % 16 == 0 && Np % 8 == 0, "i8gemm.dynamic_scaled_mm: Kp must be multiple of 16 and Np multiple of 8");
  TORCH_CHECK(Kp <= std::numeric_limits<int>::max() && Np <= std::numeric_limits<int>::max() &&
                  M <= std::numeric_limits<int>::max(),
              "i8gemm.dynamic_scaled_mm: dimensions exceed int32 kernel limits");

  if (M == 0) {
    return;
  }

  at::Tensor bias_f32 = prepare_bias(bias, N);
  const float* bias_ptr = bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr;

  at::Tensor a_q = at::zeros({M, Kp}, at::TensorOptions().dtype(at::kChar));
  at::Tensor x_scale = at::empty({M}, at::TensorOptions().dtype(at::kFloat));
  auto* a_ptr = a_q.data_ptr<int8_t>();
  auto* x_scale_ptr = x_scale.data_ptr<float>();

  if (input.scalar_type() == at::kFloat) {
    const float* x = input.data_ptr<float>();
    openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        const float* x_row = x + m * K;
        float max_abs = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          max_abs = std::max(max_abs, std::fabs(x_row[k]));
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        x_scale_ptr[m] = scale;
        int8_t* a_row = a_ptr + m * Kp;
        for (int64_t k = 0; k < K; ++k) {
          const long q = static_cast<long>(std::nearbyint(x_row[k] / scale));
          a_row[k] = static_cast<int8_t>(std::max<long>(-128, std::min<long>(127, q)));
        }
      }
    });
  } else {
    const auto* x = input.data_ptr<c10::BFloat16>();
#if defined(__ARM_FEATURE_SVE)
    quantize_bf16_rows_sve(x, a_ptr, x_scale_ptr, M, K, Kp, nthreads);
#else
    openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
      for (int64_t m = begin; m < end; ++m) {
        const auto* x_row = x + m * K;
        float max_abs = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          max_abs = std::max(max_abs, std::fabs(static_cast<float>(x_row[k])));
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        x_scale_ptr[m] = scale;
        int8_t* a_row = a_ptr + m * Kp;
        for (int64_t k = 0; k < K; ++k) {
          const long q = static_cast<long>(std::nearbyint(static_cast<float>(x_row[k]) / scale));
          a_row[k] = static_cast<int8_t>(std::max<long>(-128, std::min<long>(127, q)));
        }
      }
    });
#endif
  }

#if defined(FUSED_CPP_HAS_I8GEMM) && defined(__ARM_FEATURE_SVE)
  {
    at::Tensor padded_weight_scale = pad_f32_vector(weight_scale, Np);
    at::Tensor padded_bias = pad_f32_vector(bias_f32, Np);
    const float* direct_bias = padded_bias.defined() ? padded_bias.data_ptr<float>() : nullptr;
    at::Tensor direct_output = N == Np ? output : at::empty({M, Np}, output.options().dtype(output.scalar_type()));
    if (output.scalar_type() == at::kFloat) {
      i8gemm_mt_dispatch_scaled_f(a_ptr, packed_weight.data_ptr<int8_t>(), direct_output.data_ptr<float>(),
                                  static_cast<int>(M), static_cast<int>(Kp), static_cast<int>(Np),
                                  static_cast<int>(nthreads), x_scale_ptr, padded_weight_scale.data_ptr<float>(),
                                  direct_bias);
      if (N != Np) {
        copy_logical_columns<float>(direct_output, output, M, N, Np, nthreads);
      }
    } else {
      i8gemm_mt_dispatch_scaled_b(
          a_ptr, packed_weight.data_ptr<int8_t>(), reinterpret_cast<uint16_t*>(direct_output.data_ptr<c10::BFloat16>()),
          static_cast<int>(M), static_cast<int>(Kp), static_cast<int>(Np), static_cast<int>(nthreads), x_scale_ptr,
          padded_weight_scale.data_ptr<float>(), direct_bias);
      if (N != Np) {
        copy_logical_columns<c10::BFloat16>(direct_output, output, M, N, Np, nthreads);
      }
    }
    return;
  }
#endif

  at::Tensor acc = at::zeros({M, Np}, at::TensorOptions().dtype(at::kFloat));
  float* acc_ptr = acc.data_ptr<float>();
  const auto* packed_ptr = packed_weight.data_ptr<int8_t>();

#if defined(FUSED_CPP_HAS_I8GEMM)
  i8gemm_mt_dispatch_f(a_ptr, packed_ptr, acc_ptr, static_cast<int>(M), static_cast<int>(Kp), static_cast<int>(Np),
                       static_cast<int>(nthreads));
#else
  fallback_int8_accum_f32(a_ptr, packed_ptr, acc_ptr, M, K, N, Kp, Np, nthreads);
#endif

  const float* w_scale = weight_scale.data_ptr<float>();
  if (output.scalar_type() == at::kFloat) {
    write_scaled_output(output.data_ptr<float>(), acc_ptr, x_scale_ptr, w_scale, bias_ptr, M, N, Np, nthreads);
  } else {
#if defined(__ARM_FEATURE_SVE)
    write_scaled_output_bf16_sve(output.data_ptr<c10::BFloat16>(), acc_ptr, x_scale_ptr, w_scale, bias_ptr, M, N, Np,
                                 nthreads);
#else
    write_scaled_output(output.data_ptr<c10::BFloat16>(), acc_ptr, x_scale_ptr, w_scale, bias_ptr, M, N, Np, nthreads);
#endif
  }
}

void i8gemm_dynamic_scaled_mm_pair(at::Tensor first_output, at::Tensor second_output, at::Tensor input,
                                   at::Tensor first_packed_weight, at::Tensor first_weight_scale, int64_t K,
                                   int64_t first_N, int64_t Kp, int64_t first_Np, at::Tensor second_packed_weight,
                                   at::Tensor second_weight_scale, int64_t second_N, int64_t second_Np,
                                   int64_t nthreads) {
  TORCH_CHECK(
      input.device().is_cpu() && input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
      "i8gemm.dynamic_scaled_mm_pair: input must be contiguous CPU BF16 [M, K]");
  const int64_t M = input.size(0);
  TORCH_CHECK(input.size(1) == K && K > 0 && Kp >= K && Kp % 16 == 0,
              "i8gemm.dynamic_scaled_mm_pair: invalid K metadata");
  auto check_output = [&](const at::Tensor& output, const at::Tensor& packed, const at::Tensor& scale, int64_t N,
                          int64_t Np, const char* name) {
    TORCH_CHECK(output.device().is_cpu() && output.scalar_type() == at::kBFloat16 && output.is_contiguous() &&
                    output.sizes() == at::IntArrayRef({M, N}),
                name, " output must be contiguous CPU BF16 [M, N]");
    TORCH_CHECK(packed.device().is_cpu() && packed.scalar_type() == at::kChar && packed.is_contiguous() &&
                    packed.numel() == Kp * Np,
                name, " packed weight is invalid");
    TORCH_CHECK(
        scale.device().is_cpu() && scale.scalar_type() == at::kFloat && scale.is_contiguous() && scale.numel() == N,
        name, " scale must be contiguous CPU float32 [N]");
    TORCH_CHECK(N > 0 && Np >= N && Np % 8 == 0, name, " N metadata is invalid");
  };
  check_output(first_output, first_packed_weight, first_weight_scale, first_N, first_Np, "first");
  check_output(second_output, second_packed_weight, second_weight_scale, second_N, second_Np, "second");
  if (M == 0) {
    return;
  }

  at::Tensor a_q = at::zeros({M, Kp}, at::TensorOptions().dtype(at::kChar));
  at::Tensor x_scale = at::empty({M}, at::TensorOptions().dtype(at::kFloat));
  int8_t* a_ptr = a_q.data_ptr<int8_t>();
  float* x_scale_ptr = x_scale.data_ptr<float>();
  const auto* x = input.data_ptr<c10::BFloat16>();
#if defined(__ARM_FEATURE_SVE)
  quantize_bf16_rows_sve(x, a_ptr, x_scale_ptr, M, K, Kp, nthreads);
#else
  openmp_parallel_for_rows(M, nthreads, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
      const auto* x_row = x + m * K;
      float max_abs = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        max_abs = std::max(max_abs, std::fabs(static_cast<float>(x_row[k])));
      }
      const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
      x_scale_ptr[m] = scale;
      int8_t* a_row = a_ptr + m * Kp;
      for (int64_t k = 0; k < K; ++k) {
        const long q = static_cast<long>(std::nearbyint(static_cast<float>(x_row[k]) / scale));
        a_row[k] = static_cast<int8_t>(std::max<long>(-127, std::min<long>(127, q)));
      }
    }
  });
#endif

#if defined(FUSED_CPP_HAS_I8GEMM) && defined(__ARM_FEATURE_SVE)
  {
    at::Tensor first_padded_scale = pad_f32_vector(first_weight_scale, first_Np);
    at::Tensor second_padded_scale = pad_f32_vector(second_weight_scale, second_Np);
    at::Tensor first_direct_output =
        first_N == first_Np ? first_output
                            : at::empty({M, first_Np}, first_output.options().dtype(first_output.scalar_type()));
    at::Tensor second_direct_output =
        second_N == second_Np ? second_output
                              : at::empty({M, second_Np}, second_output.options().dtype(second_output.scalar_type()));
    i8gemm_mt_dispatch_scaled_pair_b(
        a_ptr, first_packed_weight.data_ptr<int8_t>(),
        reinterpret_cast<uint16_t*>(first_direct_output.data_ptr<c10::BFloat16>()), static_cast<int>(first_Np),
        first_padded_scale.data_ptr<float>(), nullptr, second_packed_weight.data_ptr<int8_t>(),
        reinterpret_cast<uint16_t*>(second_direct_output.data_ptr<c10::BFloat16>()), static_cast<int>(second_Np),
        second_padded_scale.data_ptr<float>(), nullptr, static_cast<int>(M), static_cast<int>(Kp),
        static_cast<int>(nthreads), x_scale_ptr);
    if (first_N != first_Np) {
      copy_logical_columns<c10::BFloat16>(first_direct_output, first_output, M, first_N, first_Np, nthreads);
    }
    if (second_N != second_Np) {
      copy_logical_columns<c10::BFloat16>(second_direct_output, second_output, M, second_N, second_Np, nthreads);
    }
    return;
  }
#endif

  auto run = [&](const at::Tensor& packed, const at::Tensor& weight_scale, at::Tensor& output, int64_t N, int64_t Np) {
#if defined(FUSED_CPP_HAS_I8GEMM) && defined(__ARM_FEATURE_SVE)
    {
      at::Tensor padded_weight_scale = pad_f32_vector(weight_scale, Np);
      at::Tensor direct_output = N == Np ? output : at::empty({M, Np}, output.options().dtype(output.scalar_type()));
      i8gemm_mt_dispatch_scaled_b(
          a_ptr, packed.data_ptr<int8_t>(), reinterpret_cast<uint16_t*>(direct_output.data_ptr<c10::BFloat16>()),
          static_cast<int>(M), static_cast<int>(Kp), static_cast<int>(Np), static_cast<int>(nthreads), x_scale_ptr,
          padded_weight_scale.data_ptr<float>(), nullptr);
      if (N != Np) {
        copy_logical_columns<c10::BFloat16>(direct_output, output, M, N, Np, nthreads);
      }
      return;
    }
#endif
    at::Tensor acc = at::zeros({M, Np}, at::TensorOptions().dtype(at::kFloat));
    float* acc_ptr = acc.data_ptr<float>();
#if defined(FUSED_CPP_HAS_I8GEMM)
    i8gemm_mt_dispatch_f(a_ptr, packed.data_ptr<int8_t>(), acc_ptr, static_cast<int>(M), static_cast<int>(Kp),
                         static_cast<int>(Np), static_cast<int>(nthreads));
#else
    fallback_int8_accum_f32(a_ptr, packed.data_ptr<int8_t>(), acc_ptr, M, K, N, Kp, Np, nthreads);
#endif
#if defined(__ARM_FEATURE_SVE)
    write_scaled_output_bf16_sve(output.data_ptr<c10::BFloat16>(), acc_ptr, x_scale_ptr, weight_scale.data_ptr<float>(),
                                 nullptr, M, N, Np, nthreads);
#else
    write_scaled_output(output.data_ptr<c10::BFloat16>(), acc_ptr, x_scale_ptr, weight_scale.data_ptr<float>(), nullptr,
                        M, N, Np, nthreads);
#endif
  };
  run(first_packed_weight, first_weight_scale, first_output, first_N, first_Np);
  run(second_packed_weight, second_weight_scale, second_output, second_N, second_Np);
}
