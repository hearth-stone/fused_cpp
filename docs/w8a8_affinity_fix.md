# W8A8 MoE CPU affinity restoration

The W8A8 Plan V2 executor pins threads inside an OpenMP parallel region.
OpenMP thread 0 is the calling thread, so leaving its affinity pinned changed
the CPU set visible to subsequent operators and tests. On Arm-codex-internal,
one W8A8 invocation changed the pytest caller's affinity from 320 CPUs to CPU 0;
the three 8-thread shared MLP cases then skipped.

Each participating thread now saves its original affinity before binding and
restores it when leaving the parallel region, including reused OpenMP workers.
Calls without explicit CPU binding do not change affinity. Save, bind, and
restore errors are reported after the parallel region. Tensor contracts,
Plan V2 inputs, and numerical operations are unchanged.

The existing dynamic-quantization reference test also checks caller affinity
after both the first and repeated invocation, with 1, 2, and 4 total threads.
Its `finally` block restores the caller's affinity even if the assertion fails,
so a regression does not contaminate later tests.

## Validation (2026-09-29)

Linux AArch64, Arm-codex-internal, SVE VL256, Python 3.12.12,
`/home/zhangxu/final_test/fused_cpp` and its existing `test` virtual environment:

- Before the native fix, all three new affinity checks failed: caller CPU set
  became `{0}`.
- After rebuilding with `FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=16`, running
  `python -m pytest -q -rs tests/test_moe_w8a8.py tests/test_shared_mlp_bf16_tiled.py`
  gave **17 passed**, including all eight shared MLP cases.
- The full `python -m pytest -q -rs tests/` run gave **1582 passed, 1 skipped**
  in 49.67 seconds. Only the native-backend-unavailable case skipped because
  the native backend is available; no shared MLP cases skipped.

This is a correctness fix. No model evaluation or performance comparison is
claimed by these operator tests.
