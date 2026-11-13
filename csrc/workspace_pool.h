/*
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
  at::Tensor empty(at::IntArrayRef sizes, at::TensorOptions options, std::size_t alignment = 64);

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
