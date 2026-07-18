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
  x86/avx512_bf16/           AVX-512 BF16 fused-SiLU executor and kernels
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
the production W13, W2 FP32, and W2 FP32 direct-route kernels, plus an
independent packed-A/packed-B BF16 GEMM operation with row-major FP32 output.
Initialize it before building from a fresh checkout:

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

The standalone `Operation::kGemmF32` surface has no fused epilogue and never
enables the experimental M1/M2 dual-N path. Its M1/M2/M4/M8/M12 instruction
schedules are compared directly, in the same process and with the same packed
A/B buffers, against `refs/i8gemm/lib/bf16gemm_sve.S`:

```bash
taskset -c 0 optimizations/fused_moe_sve/benchmarks/run_jit_vs_i8mm_pure_gemm.sh \
  --rows 1,2,4,8,12 --k 4096 --n 1024 --runs 201
```

`FUSED_CPP_MOE_SVE_IMPL` controls compute dispatch:

```text
auto  default; use Xbyak when the requested configuration is supported
jit   require Xbyak for the generated W13/FP32-W2 surfaces
asm   force the static assembly compatibility path
```

`FUSED_CPP_MOE_SVE_JIT_BULK_M=1` is an experimental JIT-only execution
variant. For M>=24 it moves the loop over complete M12 panels into generated
code; exact tails and packed formats are unchanged. The flag defaults to off.

`FUSED_CPP_MOE_SVE_W13_FIRST_PANEL_PREFETCH=1` enables an experimental
JIT-only W13 load variant for every exact M from 1 through 12. For each
thread-owned N range, only the first actual M panel issues one `PLDL1STRM` hint
2 KiB ahead of each B cache-line load; later M panels call the ordinary
generated kernel. M1-M8 preserve the two-bank K loop and disable hints on the
final N tile. M9-M12 disable hints for the final 2 KiB of that tile. This keeps
all hints inside the owned range.

`FUSED_CPP_MOE_SVE_FIRST_PANEL_PREFETCH=1` extends the same experiment to the
generated FP32 W2 and direct-route W2 kernels, using `PLDL2STRM` 1 KiB ahead.
It does not affect the static BF16-route W2 path. Full-NUMA tests show that W2
prefetch is harmful under contention, so both flags default to off and the
all-GEMM flag is for controlled experiments only. Either flag conflicts with
`FUSED_CPP_MOE_SVE_JIT_BULK_M=1`.

The static assembly remains the fallback for builds without the initialized
submodule, split-K/Kc, identity W13, reciprocal-refinement or minimax SiLU, and
BF16 W2 route storage. Strict `jit` raises instead of falling back for a
generated surface such as Kc W13/FP32 W2; an explicitly selected static-only
surface such as BF16 route storage remains available. `FUSED_CPP_MOE_SVE=0`
still disables the entire SVE backend and selects the NEON implementation.

Backend IDs in packed metadata remain stable: `0` is `arm_neon_bf16` and `1`
is `arm_sve_bf16`. Packed weights are backend-specific. SVE packed weights are
also vector-length-specific. The SVE vector length is fixed at build time by
`FUSED_CPP_SVE_VECTOR_BITS` (default `128`) and passed to the compiler through
`-msve-vector-bits`. The Xbyak kernels use that compile-time constant rather
than emitting `CNTB`. Importing `fused_cpp._moe_C` queries the importing
thread's actual VL with `PR_SVE_GET_VL` and fails immediately when it differs
from the build. Do not change the process or worker-thread VL after import.

## Experimental async short-expert pool

`fused_moe_bf16_tiled_async` has a default-off executor experiment for
transitioning cores from wide long-expert teams into narrow short-expert teams:

```text
FUSED_CPP_MOE_ASYNC_SHORT_POOL_THREADS=<narrow team width>
FUSED_CPP_MOE_ASYNC_SHORT_POOL_MAX_ROWS=<short-expert threshold; default 12>
```

The external plan still defines every active expert and the long-expert DAG.
Tasks at or below the row threshold are removed from their fixed intervals and
sorted into one global queue. The resident threads are partitioned into fixed
groups of the requested width. A group may claim whole short experts only
after every non-pooled task whose interval covers that group has completed.
Thus a 16-thread long team can release four 4-thread groups without creating
threads or changing affinity. Groups that are not covered by a long task enter
the pool immediately.

The experiment requires fused SVE execution, an aligned long-task plan, and a
pool width that divides the executor thread count. Dependencies attached to
pooled tasks are validated but replaced by interval-release eligibility; a
non-pooled task may not depend on a pooled task. The setting is intended for
executor and planner research and is not enabled by the production planner.

## x86 AVX-512 BF16 build and dispatch

Linux x86-64 builds the AVX-512 BF16 microkernel as a separate native object
with `-mavx512bf16`; the remaining `_moe_C` translation units retain the
baseline compiler target. Runtime CPUID and XCR0 checks prevent entry on CPUs
without AVX-512F/BW/VL, AVX-512 BF16, or enabled ZMM state. The backend is
available only for prepared fused-SiLU weights and can be disabled with
`FUSED_CPP_MOE_AVX512_BF16=0`.

Backend ID 101 is `x86_avx512_bf16`, with N tile 32. Its packed weights are not
compatible with ARM backends. The first implementation supports synchronous
execution with one or two worker threads, no bias, direct route output, and a
weighted AVX-512 merge. Full route groups use M12 panels, with a generic final
tail. Two-thread execution reuses OpenMP when available and has a standard
thread fallback. Scheduled/async/vLLM-staged x86 entrypoints remain
unimplemented. `FUSED_CPP_BUILD_MOE_ONLY=1` builds only `_moe_C`, which is
useful when validating this independent extension on an x86 checkout whose
main extension contains target-specific sources.

Design, commands, and measurements live in
`optimizations/fused_moe_avx512/`.

## Adding another x86 backend

Implement the ISA-owned pack and compute entries below the corresponding x86
directory, then register an immutable descriptor in `common/backend.cpp` with
runtime CPUID/XCR0 checks. `auto` selection must never expose a backend until
its full prepare and execute contract is implemented. Do not reuse an ARM
backend ID or claim support from a placeholder translation unit.
