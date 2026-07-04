#pragma once

#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace fused_cpp::workspace {

class WorkspacePool;

class WorkspaceLease {
 public:
  explicit WorkspaceLease(WorkspacePool& pool);
  WorkspaceLease(const WorkspaceLease&) = delete;
  WorkspaceLease& operator=(const WorkspaceLease&) = delete;
  WorkspaceLease(WorkspaceLease&&) = delete;
  WorkspaceLease& operator=(WorkspaceLease&&) = delete;
  ~WorkspaceLease();

  void* alloc_bytes(std::size_t bytes, std::size_t alignment = 64);
  at::Tensor empty(at::IntArrayRef sizes, at::TensorOptions options,
                   std::size_t alignment = 64);

 private:
  WorkspacePool& pool_;
  std::unique_lock<std::mutex> lock_;
};

WorkspaceLease acquire();

bool enabled();
std::size_t capacity_bytes();
std::size_t high_water_bytes();
void reset_stats();

}  // namespace fused_cpp::workspace
