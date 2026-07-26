// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace fused_cpp::moe::x86::avx512_bf16 {

enum class X86ExecutionIsa : uint8_t {
  kAvx512Bf16,
  kAmxBf16,
};

enum class AmxKernelPattern : uint8_t {
  kM1N2,
  kM2N2,
  kM1N4,
};

enum class AmxCacheStage : uint8_t {
  kW13,
  kW2,
};

enum class Avx512SmallMMultiNStage : uint8_t {
  kW13,
  kW2,
};

struct X86CpuIdentity {
  bool is_intel = false;
  int family = 0;
  int model = 0;
  int stepping = 0;
};

struct X86PolicyInput {
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t total_routes = 0;
  int64_t max_routes = 0;
  int64_t second_max_routes = 0;
  int64_t active_experts = 0;
  int64_t requested_threads = 1;
  bool allow_isa_selection = false;
  bool avx512_available = false;
  bool amx_available = false;
  X86ExecutionIsa requested_isa = X86ExecutionIsa::kAvx512Bf16;
};

struct X86PolicyDecision {
  X86ExecutionIsa isa = X86ExecutionIsa::kAvx512Bf16;
  int64_t execution_threads = 1;
  int64_t nsplit_target_rows = 64;
  bool route_skewed = false;
  const char* profile = "generic_v1";
};

X86CpuIdentity GetX86CpuIdentity();
X86PolicyDecision ResolveX86Policy(const X86PolicyInput& input);

bool UseAutomaticAvx512SmallMMultiN(Avx512SmallMMultiNStage stage, int rows, int reduction_size, int output_size,
                                    int cooperative_threads = 1);
AmxKernelPattern ResolveAutomaticAmxKernelPattern(int rows, int hidden_size, int intermediate_size);
int ResolveAutomaticAmxCacheBlocks(AmxCacheStage stage, int k_pad, int rows, int hidden_size, int intermediate_size,
                                   AmxKernelPattern pattern);

const char* X86ExecutionIsaName(X86ExecutionIsa isa);
const char* AmxKernelPatternName(AmxKernelPattern pattern);

}  // namespace fused_cpp::moe::x86::avx512_bf16
