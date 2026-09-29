// ATen tensors whose storage comes from the shared page policy.
//
// Kept separate from page_policy.h so that header stays free of ATen: the policy
// is also used by plain std::vector allocators in translation units that do not
// want the tensor machinery.

#pragma once

#include <ATen/ATen.h>

#include <cstddef>

#include "page_policy.h"

namespace fused_cpp {

// Same contract as at::empty, but the storage is backed by whatever the page
// policy selects instead of the CPU caching allocator. Intended for the few
// buffers large enough for the page size to matter, above all the packed MoE
// weights, which the Python layer used to relocate to a hugetlbfs file with a
// second full-size copy after packing.
inline at::Tensor page_backed_empty(at::IntArrayRef sizes, const at::TensorOptions& options) {
  const at::ScalarType dtype = c10::typeMetaToScalarType(options.dtype());
  int64_t count = 1;
  for (const int64_t size : sizes) {
    TORCH_CHECK(size >= 0, "page_backed_empty got a negative extent: ", size);
    count *= size;
  }
  const std::size_t bytes = static_cast<std::size_t>(count) * c10::elementSize(dtype);
  if (bytes == 0) return at::empty(sizes, options);

  void* storage = page_alloc(bytes);
  TORCH_CHECK(storage != nullptr, "page_alloc failed for ", bytes, " bytes");
  return at::from_blob(storage, sizes, [bytes](void* pointer) { page_free(pointer, bytes); }, options.device(at::kCPU));
}

}  // namespace fused_cpp
