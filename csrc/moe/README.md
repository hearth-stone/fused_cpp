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

The SVE backend uses the pinned `3rdparty/xbyak_aarch64` submodule to generate
the production W13, W2 FP32, and W2 FP32 direct-route kernels. Initialize it
before building from a fresh checkout:

```bash
git submodule update --init --recursive 3rdparty/xbyak_aarch64
```

Generated kernels preserve the static assembly ABI and packed-A layouts while
specializing the final physical panel for every logical M from 1 through 12.
M1-M8 use the existing eight-row packed-A panel, M9-M12 use the twelve-row
panel, and only `2 * ceil(M / 2)` rows execute BFMMLA. Kernel generation is
prewarmed during SVE weight preparation, outside the forward-call timing path.
The M1-M8 generated K loop uses the static M2/M4/M8 two-bank load/compute state
machine. M9-M12 use the static M12 register-reuse schedule because the larger
accumulator set cannot coexist with two complete A/B banks.

`FUSED_CPP_MOE_SVE_IMPL` controls compute dispatch:

```text
auto  default; use Xbyak when the requested configuration is supported
jit   require Xbyak for the generated W13/FP32-W2 surfaces
asm   force the static assembly compatibility path
```

`FUSED_CPP_MOE_SVE_JIT_BULK_M=1` is an experimental JIT-only execution
variant. For M>=24 it moves the loop over complete M12 panels into generated
code; exact tails and packed formats are unchanged. The flag defaults to off.

The static assembly remains the fallback for builds without the initialized
submodule, split-K/Kc, identity W13, reciprocal-refinement or minimax SiLU, and
BF16 W2 route storage. Strict `jit` raises instead of falling back for a
generated surface such as Kc W13/FP32 W2; an explicitly selected static-only
surface such as BF16 route storage remains available. `FUSED_CPP_MOE_SVE=0`
still disables the entire SVE backend and selects the NEON implementation.

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
