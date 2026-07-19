// SPDX-License-Identifier: Apache-2.0
// AMX uses the shared x86 BF16 packing, executor, and Xbyak factory currently
// owned by avx512_bf16. This namespace remains the future ownership boundary
// if AMX grows ISA-specific source files beyond the shared JIT implementation.
namespace fused_cpp::moe::x86::amx_bf16 {}
