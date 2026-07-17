# Native MoE backend layout

The BF16 fused-expert operator is built as the independent Python extension
`fused_cpp._moe_C`. This keeps its runtime ISA dispatch separate from the
compile-time target used by the rest of `fused_cpp._C`.

## Ownership

```text
csrc/moe/
  module.cpp                 Python/native boundary
  common/                    architecture-neutral API, registry, and stubs
  arm/common/                shared ARM validation, scheduling, and execution
  arm/neon_bf16/             AArch64 NEON + BFMMLA kernels
  arm/sve_bf16/              AArch64 SVE + SVEBF16 kernels and helpers
  x86/avx2/                  reserved backend boundary; not implemented
  x86/avx512_bf16/           reserved backend boundary; not implemented
  x86/amx_bf16/              reserved backend boundary; not implemented
```

Common code may call an ISA implementation only through a backend descriptor
or an entry guarded by that backend's build macro. ISA directories must not
depend on Python bindings or planner code.

## ARM build and dispatch

Linux AArch64 builds one fat `_moe_C` extension:

- common and NEON objects use the `armv8.6-a+bf16+i8mm` baseline;
- SVE objects use `armv8.6-a+sve+bf16+i8mm` independently;
- Linux HWCAP/HWCAP2 selects SVE only when SVE, BF16, and SVEBF16 are present;
- `FUSED_CPP_MOE_SVE=0` removes SVE from runtime selection and forces NEON.

The implementation intentionally requires BF16/BFMMLA for the NEON backend.
There is no scalar or widening-only NEON fallback.

Backend IDs in packed metadata remain stable: `0` is `arm_neon_bf16` and `1`
is `arm_sve_bf16`. Packed weights are backend-specific. SVE packed weights are
also vector-length-specific, and execution validates the stored N tile against
the current runtime before entering a kernel.

## Adding an x86 backend

Implement the ISA-owned pack and compute entries below the corresponding x86
directory, then register an immutable descriptor in `common/backend.cpp` with
runtime CPUID/XCR0 checks. `auto` selection must never expose a backend until
its full prepare and execute contract is implemented. Do not reuse an ARM
backend ID or claim support from a placeholder translation unit.
