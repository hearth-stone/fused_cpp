#pragma once

// C ABI shim for the registered SDPA kernels.
//
// The kernels themselves consume contiguous logical tensors in:
//   q:   [B, N, L, E]
//   k:   [B, N, S, E]
//   v:   [B, N, S, Ev]
//   out: [B, N, L, Ev]  (always fp32)
//
// This header exposes raw-pointer entry points so llama.cpp can choose any
// registered SDPA version by string without constructing torch tensors. The
// generic strided entry accepts element strides for logical [B, N, T, D]
// tensors and materializes contiguous inputs when needed. The ggml_f32 helper
// matches ggml_flash_attn_ext's logical layout [head_dim, tokens, heads, batch]
// backed by llama.cpp BGE-style Qcur/Kcur/Vcur buffers.

#include <stdint.h>

#if defined(_WIN32)
#define FUSED_CPP_SDPA_API __declspec(dllexport)
#else
#define FUSED_CPP_SDPA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum fused_cpp_sdpa_dtype {
  FUSED_CPP_SDPA_DTYPE_F32 = 0,
  FUSED_CPP_SDPA_DTYPE_BF16 = 1,
};

typedef struct fused_cpp_sdpa_strides {
  int64_t b;
  int64_t n;
  int64_t t;
  int64_t d;
} fused_cpp_sdpa_strides;

FUSED_CPP_SDPA_API const char* fused_cpp_sdpa_last_error(void);

FUSED_CPP_SDPA_API int fused_cpp_sdpa_version_count(void);

FUSED_CPP_SDPA_API const char* fused_cpp_sdpa_version_name(int index);

// Generic raw-pointer entry.
//
// q/k/v are interpreted as dtype-selected buffers:
//   FUSED_CPP_SDPA_DTYPE_F32  -> float*
//   FUSED_CPP_SDPA_DTYPE_BF16 -> uint16_t* bit-compatible with bfloat16
//
// All strides are in elements, not bytes. q/k/v/out strides describe logical
// [B, N, T, D] indexing, where T is L for q/out and S for k/v.
//
// attn_mask may be null. If non-null, it must be contiguous fp32 [B, N, L, S].
// out is always fp32. scale <= 0 means use 1 / sqrt(E).
//
// Returns 0 on success and -1 on error. Use fused_cpp_sdpa_last_error() for
// the thread-local error string.
FUSED_CPP_SDPA_API int fused_cpp_sdpa_forward_strided(const char* version, int dtype, const void* q, const void* k,
                                                      const void* v, const float* attn_mask, float* out, int64_t B,
                                                      int64_t N, int64_t L, int64_t S, int64_t E, int64_t Ev,
                                                      fused_cpp_sdpa_strides q_strides,
                                                      fused_cpp_sdpa_strides k_strides,
                                                      fused_cpp_sdpa_strides v_strides,
                                                      fused_cpp_sdpa_strides out_strides, int is_causal, float scale);

// Convenience entry for llama.cpp / ggml flash-attn input layout, fp32 path.
//
// q/k/v/out are ggml dim0-fastest buffers, not C row-major q[E][T][N][B]
// arrays. The expected linear offsets are:
//   q(b,n,l,e)   = q   + b*(E*N*L)  + l*(E*N)  + n*E  + e
//   k(b,n,s,e)   = k   + b*(E*N*S)  + s*(E*N)  + n*E  + e
//   v(b,n,s,ev)  = v   + b*(Ev*N*S) + s*(Ev*N) + n*Ev + ev
//   out(b,n,l,ev)= out + b*(Ev*N*L) + l*(Ev*N) + n*Ev + ev
//
// For BGE-small single batch Qcur/Kcur/Vcur this corresponds to physical
// ggml tensors with ne = [head_dim, n_head, n_tokens] that build_attn views as
// [head_dim, tokens, heads, batch].
FUSED_CPP_SDPA_API int fused_cpp_sdpa_forward_ggml_f32(const char* version, const float* q, const float* k,
                                                       const float* v, float* out, int64_t B, int64_t N, int64_t L,
                                                       int64_t S, int64_t E, int64_t Ev, int is_causal, float scale);

#ifdef __cplusplus
}  // extern "C"
#endif
