# Adaptive M-by-K fused gather-pack

Date: 2026-08-12

## Decision

Enable adaptive M-by-K work sharing in the production SVE fused gather/pack-A
stage. The feature clears the predeclared gate: every sampled underfilled point
in M=`{1,9,12,13,24,37}` improves complete-call median time by more than 2%,
and the M=72/M=96 controls do not regress. This changes neither GEMM ownership
nor public behavior.

## Mechanism

The baseline assigns each complete physical M12 or M8 packed-A panel to one
worker. When fewer panels exist than team threads, workers with no panel wait at
the pre-W13 barrier. The candidate flattens `(M panel, K stripe)` into one work
domain only in that underfilled case.

- Every K stripe contains at least 32 BF16 elements.
- Every interior boundary is K8 aligned. This advances an M12 packed panel by
  192 bytes and an M8 panel by 128 bytes, so writers do not share cache lines.
- Adjacent stripes assigned to one worker are coalesced.
- If `K_pad < 64` or physical panel count is at least team width, execution
  remains M-only.
- The pre-W13 barrier remains: every W13 N owner reads all of packed A.

The production W13/W2 GEMMs remain N-split. Packed weights, packed-A layout,
SiLU arithmetic, output dtype/layout, backend ids, Plan V2, and Python APIs are
unchanged. The rollback boundary is the gather-pack work planner and its two
K-range writers in `csrc/moe/arm/common/fused_moe_bf16_tiled.cpp`.

## Platform and method

The baseline is clean Git `cbd215a`; the candidate is that source plus the
adaptive gather-pack change. Both were force-built with the same compiler and
flags:

```bash
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace --force
```

Machine and placement:

- `AmazonECS8Cores`, Arm Neoverse-V1, CPUs 0-7, one NUMA node;
- SVE256, 64 KiB private L1D/core, 1 MiB private L2/core, 32 MiB shared L3;
- GCC 12.4.0, Python 3.12.3, PyTorch 2.12.0;
- default page policy (THP), no HugeTLB override;
- `OMP_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`, and BLAS thread counts set to 1.

The operator shape is H=4096, F=512, top-k=1, polynomial-5 SiLU, the
`skip_weighted=True` direct-output path, and the JIT exact-M compute path. W2
accumulates in FP32 and converts its owned columns directly to the final BF16
output; no route-merge buffer is used. W13 is 8 MiB/expert and W2 is
4 MiB/expert. With SVE256, `n_tile=16`; a full 8T owner stripe is 1 MiB/thread
for W13 and 0.5 MiB/thread for W2. Thus this comparison keeps
`(threads, W13 window, W2 window) = (8, 1 MiB/thread, 0.5 MiB/thread)` fixed.

Each timed call executes eight experts sequentially, one 8-thread expert per
wave, and rotates between two disjoint eight-expert weight sets. Allocation and
weight packing are outside timing. Values below are 31-sample medians for the
complete eight-expert scheduled call, not isolated gather times or per-expert
latencies.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c 0-7 .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --hidden 4096 --intermediate 512 \
  --routes 1,9,12,13,24,37,72,96 --threads 8 \
  --experts 16 --measurement-experts 8 --experts-per-wave 1 \
  --warmup 5 --runs 31 --variants jit --switch-period 5
```

## Complete-call results

| M | Baseline median | Baseline p10-p90 | Candidate median | Candidate p10-p90 | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.861 ms | 0.845-0.879 ms | 0.841 ms | 0.814-0.868 ms | +2.32% |
| 9 | 1.185 ms | 1.163-1.234 ms | 1.089 ms | 1.064-1.125 ms | +8.82% |
| 12 | 1.339 ms | 1.322-1.358 ms | 1.190 ms | 1.170-1.207 ms | +12.55% |
| 13 | 1.969 ms | 1.956-1.997 ms | 1.781 ms | 1.755-1.826 ms | +10.60% |
| 24 | 2.331 ms | 2.316-2.376 ms | 2.200 ms | 2.176-2.300 ms | +5.98% |
| 37 | 3.716 ms | 3.685-3.841 ms | 3.607 ms | 3.581-3.686 ms | +3.01% |
| 72 | 5.864 ms | 5.792-6.001 ms | 5.795 ms | 5.771-5.882 ms | +1.20% |
| 96 | 7.994 ms | 7.539-8.951 ms | 7.873 ms | 7.630-8.885 ms | +1.54% |

The gain follows the intended mechanism: the sampled M=1..37 points cannot
provide eight physical M panels, so K work activates otherwise idle gather-pack
workers. At M=96 the eight M12 panels already fill the team and the production
selector remains on the baseline M-only traversal. A focused M=85
equality-boundary control was
7.084 ms M-only versus 7.135 ms when two-dimensional sharing was forced
(-0.7%), which is why `panel_count >= threads` stays M-only.

## Correctness and limitations

The target SVE build passed the complete pack-A suite, including bit-exact
M12/M8 layout checks, M=1/9/12/13/37/85/95 tails, TopK 1/2/6, K padding,
1T/multi-thread coverage, the 32-element K minimum, and invalid K alignment.
Normal, scheduled, async/Plan V2, dirty-scratch reuse, FP32/BF16 route output,
and the NEON fallback were also exercised by focused integration tests. The
final target runs reported 235 pack-A tests passed, 7 focused integration tests
passed, and 3 backend-dispatch tests passed. An additional exhaustive geometry
check validated 115,200 `(M, K_pad, team)` plans for complete, non-overlapping
panel/K coverage and the minimum K stripe. The two vLLM-staged comparator cases
require 12 available CPUs and were skipped on this eight-core host; that bridge
invokes the same helper one physical panel at a time and therefore remains
M-only.

The documented 192-core checkout was unavailable at its configured project
path, so this change does not claim cross-machine performance portability yet.
Its cross-machine residual risk is limited to the 32-element minimum-stripe
choice; layout and concurrency correctness are independent of that tuning.
