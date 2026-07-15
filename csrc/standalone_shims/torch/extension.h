#pragma once

// Minimal torch/ATen compatibility shim for compiling the SDPA kernels directly
// into a non-PyTorch binary. Use it only with:
//   -DFUSED_CPP_SDPA_STANDALONE -Icsrc/standalone_shims
//
// It intentionally implements only the tiny subset used by the current SDPA
// sources: TORCH_CHECK/TORCH_WARN, at::BFloat16, at::TensorOptions, at::empty,
// and at::Tensor::data_ptr().

#if !defined(FUSED_CPP_SDPA_STANDALONE)
#error "standalone_shims/torch/extension.h requires FUSED_CPP_SDPA_STANDALONE"
#endif

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fused_cpp::standalone {

template <typename... Args>
std::string make_message(Args&&... args) {
  std::ostringstream os;
  (os << ... << std::forward<Args>(args));
  return os.str();
}

inline uint16_t fp32_to_bf16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float bf16_bits_to_fp32(uint16_t bits) {
  const uint32_t fp32_bits = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &fp32_bits, sizeof(value));
  return value;
}

}  // namespace fused_cpp::standalone

namespace at {

struct alignas(2) BFloat16 {
  uint16_t x;

  BFloat16() = default;
  BFloat16(float value) : x(::fused_cpp::standalone::fp32_to_bf16_bits(value)) {}

  operator float() const { return ::fused_cpp::standalone::bf16_bits_to_fp32(x); }
};

static_assert(sizeof(BFloat16) == sizeof(uint16_t), "standalone at::BFloat16 must be uint16-compatible");

enum ScalarType {
  kFloat,
  kBFloat16,
};

class TensorOptions {
 public:
  TensorOptions& dtype(ScalarType dtype) {
    dtype_ = dtype;
    return *this;
  }

  ScalarType dtype() const { return dtype_; }

 private:
  ScalarType dtype_ = kFloat;
};

class Tensor {
 public:
  Tensor() = default;

  Tensor(size_t nbytes, ScalarType dtype) : storage_(nbytes), dtype_(dtype) {}

  void* data_ptr() { return storage_.data(); }

  const void* data_ptr() const { return storage_.data(); }

  template <typename T>
  T* data_ptr() {
    return reinterpret_cast<T*>(storage_.data());
  }

  template <typename T>
  const T* data_ptr() const {
    return reinterpret_cast<const T*>(storage_.data());
  }

  ScalarType scalar_type() const { return dtype_; }

 private:
  std::vector<unsigned char> storage_;
  ScalarType dtype_ = kFloat;
};

inline size_t element_size(ScalarType dtype) {
  switch (dtype) {
    case kFloat:
      return sizeof(float);
    case kBFloat16:
      return sizeof(BFloat16);
  }
  throw std::runtime_error("standalone at::element_size: unknown dtype");
}

inline Tensor empty(std::initializer_list<int64_t> sizes, const TensorOptions& options) {
  size_t numel = 1;
  for (const int64_t dim : sizes) {
    if (dim < 0) {
      throw std::runtime_error("standalone at::empty: negative dimension");
    }
    numel *= static_cast<size_t>(dim);
  }
  return Tensor(numel * element_size(options.dtype()), options.dtype());
}

}  // namespace at

#define TORCH_CHECK(condition, ...)                                                 \
  do {                                                                              \
    if (!(condition)) {                                                             \
      throw std::runtime_error(::fused_cpp::standalone::make_message(__VA_ARGS__)); \
    }                                                                               \
  } while (0)

#define TORCH_WARN(...) \
  do {                  \
  } while (0)
