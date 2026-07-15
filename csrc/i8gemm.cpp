#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>

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
                         int64_t M, int64_t N, int64_t Np) {
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
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

#if !defined(FUSED_CPP_HAS_I8GEMM)
void fallback_int8_accum_f32(const int8_t* a, const int8_t* b_kn, float* c, int64_t M, int64_t K, int64_t N, int64_t Kp,
                             int64_t Np) {
  at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
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
    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
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
    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
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
  }

  at::Tensor acc = at::zeros({M, Np}, at::TensorOptions().dtype(at::kFloat));
  float* acc_ptr = acc.data_ptr<float>();
  const auto* packed_ptr = packed_weight.data_ptr<int8_t>();

#if defined(FUSED_CPP_HAS_I8GEMM)
  i8gemm_mt_dispatch_f(a_ptr, packed_ptr, acc_ptr, static_cast<int>(M), static_cast<int>(Kp), static_cast<int>(Np),
                       static_cast<int>(nthreads));
#else
  (void)nthreads;
  fallback_int8_accum_f32(a_ptr, packed_ptr, acc_ptr, M, K, N, Kp, Np);
#endif

  const float* w_scale = weight_scale.data_ptr<float>();
  if (output.scalar_type() == at::kFloat) {
    write_scaled_output(output.data_ptr<float>(), acc_ptr, x_scale_ptr, w_scale, bias_ptr, M, N, Np);
  } else {
    write_scaled_output(output.data_ptr<c10::BFloat16>(), acc_ptr, x_scale_ptr, w_scale, bias_ptr, M, N, Np);
  }
}
