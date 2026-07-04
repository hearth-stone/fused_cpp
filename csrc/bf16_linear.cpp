#include <torch/extension.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#if defined(FUSED_CPP_HAS_BF16GEMM)
extern "C" {
#include "bf16gemm.h"
}
#endif

namespace {

int64_t RoundUp(int64_t value, int64_t quantum) {
  return ((value + quantum - 1) / quantum) * quantum;
}

void CheckCpuTensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
}

void CheckBf16Cpu2d(const at::Tensor& tensor, const char* name) {
  CheckCpuTensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 2,
              name, " must be 2-D, got ", tensor.dim(), "-D");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16,
              name, " must be torch.bfloat16, got ", tensor.scalar_type());
}

void CheckDimInt32(int64_t value, const char* name) {
  TORCH_CHECK(value > 0, name, " must be positive, got ", value);
  TORCH_CHECK(value <= std::numeric_limits<int>::max(),
              name, " exceeds int32 kernel limit: ", value);
}

template <bool OutputBf16>
at::Tensor Bf16LinearPrepackedImpl(const at::Tensor& input,
                                   const at::Tensor& packed_weight,
                                   int64_t K,
                                   int64_t N,
                                   int64_t Np,
                                   int64_t nthreads) {
#if defined(FUSED_CPP_HAS_BF16GEMM)
  const int64_t M = input.size(0);
  TORCH_CHECK(M <= std::numeric_limits<int>::max(),
              "bf16_linear_prepacked_to_dtype M exceeds int32 kernel limit: ",
              M);
  if (M == 0) {
    if constexpr (OutputBf16) {
      return at::empty({M, N}, input.options().dtype(at::kBFloat16));
    } else {
      return at::empty({M, N}, input.options().dtype(at::kFloat));
    }
  }

  at::Tensor input_contig = input.contiguous();

  if constexpr (OutputBf16) {
    at::Tensor output_bf16_padded =
        at::zeros({M, Np}, input.options().dtype(at::kBFloat16));
    bf16gemm_mt_dispatch_nld_b(
        reinterpret_cast<const bf16_t*>(input_contig.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const bf16_t*>(packed_weight.data_ptr<c10::BFloat16>()),
        reinterpret_cast<bf16_t*>(output_bf16_padded.data_ptr<c10::BFloat16>()),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(Np),
        static_cast<int>(nthreads));
    if (N == Np) {
      return output_bf16_padded;
    }
    return output_bf16_padded
        .index({at::indexing::Slice(), at::indexing::Slice(0, N)})
        .contiguous();
  } else {
    at::Tensor output_f32 =
        at::zeros({M, Np}, input.options().dtype(at::kFloat));
    bf16gemm_mt_dispatch(
        reinterpret_cast<const bf16_t*>(input_contig.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const bf16_t*>(packed_weight.data_ptr<c10::BFloat16>()),
        output_f32.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(Np),
        static_cast<int>(nthreads));
    if (N == Np) {
      return output_f32;
    }
    return output_f32.index({at::indexing::Slice(), at::indexing::Slice(0, N)})
        .contiguous();
  }
#else
  (void)input;
  (void)packed_weight;
  (void)K;
  (void)N;
  (void)Np;
  (void)nthreads;
  TORCH_CHECK(false,
              "bf16_linear_prepacked_to_dtype requires an AArch64 build with "
              "refs/i8gemm bf16gemm enabled");
  return at::Tensor();
#endif
}

template <bool OutputBf16>
void Bf16LinearPrepackedOutImpl(const at::Tensor& input,
                                const at::Tensor& packed_weight,
                                int64_t K,
                                int64_t Np,
                                int64_t nthreads,
                                const at::Tensor& output) {
#if defined(FUSED_CPP_HAS_BF16GEMM)
  const int64_t M = input.size(0);
  TORCH_CHECK(M <= std::numeric_limits<int>::max(),
              "bf16_linear_prepacked_to_dtype_out M exceeds int32 kernel limit: ",
              M);
  if (M == 0) {
    return;
  }

  at::Tensor input_contig = input.contiguous();
  output.zero_();

  if constexpr (OutputBf16) {
    bf16gemm_mt_dispatch_nld_b(
        reinterpret_cast<const bf16_t*>(input_contig.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const bf16_t*>(packed_weight.data_ptr<c10::BFloat16>()),
        reinterpret_cast<bf16_t*>(output.data_ptr<c10::BFloat16>()),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(Np),
        static_cast<int>(nthreads));
  } else {
    bf16gemm_mt_dispatch(
        reinterpret_cast<const bf16_t*>(input_contig.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const bf16_t*>(packed_weight.data_ptr<c10::BFloat16>()),
        output.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(Np),
        static_cast<int>(nthreads));
  }
#else
  (void)input;
  (void)packed_weight;
  (void)K;
  (void)Np;
  (void)nthreads;
  (void)output;
  TORCH_CHECK(false,
              "bf16_linear_prepacked_to_dtype_out requires an AArch64 build "
              "with refs/i8gemm bf16gemm enabled");
#endif
}

}  // namespace

std::tuple<at::Tensor, int64_t, int64_t, int64_t>
bf16_linear_prepare_weight(at::Tensor weight) {
  CheckBf16Cpu2d(weight, "bf16_linear_prepare_weight weight");
#if defined(FUSED_CPP_HAS_BF16GEMM)
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  CheckDimInt32(N, "bf16_linear_prepare_weight N");
  CheckDimInt32(K, "bf16_linear_prepare_weight K");

  TORCH_CHECK(K >= 8 && K % 8 == 0,
              "bf16_linear_prepare_weight requires K to be aligned to 8, got K=",
              K);
  const int64_t Np = std::max<int64_t>(8, RoundUp(N, 8));
  TORCH_CHECK(Np <= std::numeric_limits<int>::max(),
              "bf16_linear_prepare_weight padded N exceeds int32 kernel limit: ",
              Np);

  at::Tensor weight_contig = weight.contiguous();

  at::Tensor b_kn =
      at::zeros({K, Np}, weight.options().dtype(at::kBFloat16));
  const auto* w_ptr = weight_contig.data_ptr<c10::BFloat16>();
  auto* b_ptr = b_kn.data_ptr<c10::BFloat16>();
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      b_ptr[k * Np + n] = w_ptr[n * K + k];
    }
  }

  at::Tensor b_packed =
      at::empty({K * Np}, weight.options().dtype(at::kBFloat16));
  bf16_pack_B(
      reinterpret_cast<const bf16_t*>(b_kn.data_ptr<c10::BFloat16>()),
      reinterpret_cast<bf16_t*>(b_packed.data_ptr<c10::BFloat16>()),
      static_cast<int>(K),
      static_cast<int>(Np));
  return std::make_tuple(b_packed, K, N, Np);
#else
  TORCH_CHECK(false,
              "bf16_linear_prepare_weight requires an AArch64 build with "
              "refs/i8gemm bf16gemm enabled");
  return std::make_tuple(at::Tensor(), 0, 0, 0);
#endif
}

at::Tensor bf16_linear_prepacked_to_dtype(at::Tensor input,
                                          at::Tensor packed_weight,
                                          int64_t K,
                                          int64_t N,
                                          int64_t Np,
                                          bool output_bf16,
                                          int64_t nthreads) {
  CheckBf16Cpu2d(input, "bf16_linear_prepacked_to_dtype input");
  CheckCpuTensor(packed_weight, "bf16_linear_prepacked_to_dtype packed_weight");
  TORCH_CHECK(packed_weight.scalar_type() == at::kBFloat16,
              "bf16_linear_prepacked_to_dtype packed_weight must be torch.bfloat16, got ",
              packed_weight.scalar_type());
  TORCH_CHECK(packed_weight.is_contiguous(),
              "bf16_linear_prepacked_to_dtype packed_weight must be contiguous");
  CheckDimInt32(K, "bf16_linear_prepacked_to_dtype K");
  CheckDimInt32(N, "bf16_linear_prepacked_to_dtype N");
  CheckDimInt32(Np, "bf16_linear_prepacked_to_dtype Np");
  TORCH_CHECK(K >= 8 && K % 8 == 0,
              "bf16_linear_prepacked_to_dtype requires K to be aligned to 8, got K=",
              K);
  TORCH_CHECK(Np >= N && Np >= 8 && Np % 8 == 0,
              "bf16_linear_prepacked_to_dtype requires Np to be >= N and aligned to 8, got N=",
              N, " Np=", Np);
  TORCH_CHECK(input.size(1) == K,
              "bf16_linear_prepacked_to_dtype K mismatch: input K=",
              input.size(1), " packed K=", K);
  TORCH_CHECK(packed_weight.numel() == K * Np,
              "bf16_linear_prepacked_to_dtype packed_weight numel mismatch: expected ",
              K * Np, ", got ", packed_weight.numel());

  if (output_bf16) {
    return Bf16LinearPrepackedImpl<true>(
        input, packed_weight, K, N, Np, nthreads);
  }
  return Bf16LinearPrepackedImpl<false>(
      input, packed_weight, K, N, Np, nthreads);
}

void bf16_linear_prepacked_to_dtype_out(at::Tensor input,
                                        at::Tensor packed_weight,
                                        int64_t K,
                                        int64_t N,
                                        int64_t Np,
                                        bool output_bf16,
                                        int64_t nthreads,
                                        at::Tensor output) {
  CheckBf16Cpu2d(input, "bf16_linear_prepacked_to_dtype_out input");
  CheckCpuTensor(packed_weight, "bf16_linear_prepacked_to_dtype_out packed_weight");
  CheckCpuTensor(output, "bf16_linear_prepacked_to_dtype_out output");
  TORCH_CHECK(packed_weight.scalar_type() == at::kBFloat16,
              "bf16_linear_prepacked_to_dtype_out packed_weight must be torch.bfloat16, got ",
              packed_weight.scalar_type());
  TORCH_CHECK(packed_weight.is_contiguous(),
              "bf16_linear_prepacked_to_dtype_out packed_weight must be contiguous");
  CheckDimInt32(K, "bf16_linear_prepacked_to_dtype_out K");
  CheckDimInt32(N, "bf16_linear_prepacked_to_dtype_out N");
  CheckDimInt32(Np, "bf16_linear_prepacked_to_dtype_out Np");
  TORCH_CHECK(K >= 8 && K % 8 == 0,
              "bf16_linear_prepacked_to_dtype_out requires K to be aligned to 8, got K=",
              K);
  TORCH_CHECK(Np >= N && Np >= 8 && Np % 8 == 0,
              "bf16_linear_prepacked_to_dtype_out requires Np to be >= N and aligned to 8, got N=",
              N, " Np=", Np);
  TORCH_CHECK(input.size(1) == K,
              "bf16_linear_prepacked_to_dtype_out K mismatch: input K=",
              input.size(1), " packed K=", K);
  TORCH_CHECK(packed_weight.numel() == K * Np,
              "bf16_linear_prepacked_to_dtype_out packed_weight numel mismatch: expected ",
              K * Np, ", got ", packed_weight.numel());
  TORCH_CHECK(output.dim() == 2 && output.size(0) == input.size(0) &&
                  output.size(1) == Np,
              "bf16_linear_prepacked_to_dtype_out output must be [M, Np], got ",
              output.sizes(), " expected [", input.size(0), ", ", Np, "]");
  TORCH_CHECK(output.is_contiguous(),
              "bf16_linear_prepacked_to_dtype_out output must be contiguous");
  TORCH_CHECK(output.scalar_type() == (output_bf16 ? at::kBFloat16 : at::kFloat),
              "bf16_linear_prepacked_to_dtype_out output dtype mismatch: got ",
              output.scalar_type());

  if (output_bf16) {
    Bf16LinearPrepackedOutImpl<true>(
        input, packed_weight, K, Np, nthreads, output);
  } else {
    Bf16LinearPrepackedOutImpl<false>(
        input, packed_weight, K, Np, nthreads, output);
  }
}

at::Tensor bf16_linear_to_dtype(at::Tensor input,
                                at::Tensor weight,
                                bool output_bf16,
                                int64_t nthreads) {
  CheckBf16Cpu2d(input, "bf16_linear_to_dtype input");
  CheckBf16Cpu2d(weight, "bf16_linear_to_dtype weight");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "bf16_linear_to_dtype K mismatch: input K=", input.size(1),
              " weight K=", weight.size(1));

  auto prepared = bf16_linear_prepare_weight(weight);
  return bf16_linear_prepacked_to_dtype(
      input,
      std::get<0>(prepared),
      std::get<1>(prepared),
      std::get<2>(prepared),
      std::get<3>(prepared),
      output_bf16,
      nthreads);
}
