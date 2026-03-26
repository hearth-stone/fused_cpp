#pragma once
#include <torch/extension.h>

// ── Tensor dtype/contiguity normalization helpers ────────────────────────────
// Use these at kernel entry points to avoid accessor type mismatches
// (e.g., vllm passes int32 block_table but accessor expects int64).

inline at::Tensor ensure_dtype(const at::Tensor& t, at::ScalarType dtype) {
    return t.scalar_type() == dtype ? t : t.to(dtype);
}

inline at::Tensor ensure_i64(const at::Tensor& t) {
    return ensure_dtype(t, at::kLong);
}

inline at::Tensor ensure_i32(const at::Tensor& t) {
    return ensure_dtype(t, at::kInt);
}

inline at::Tensor ensure_contiguous(const at::Tensor& t) {
    return t.is_contiguous() ? t : t.contiguous();
}
