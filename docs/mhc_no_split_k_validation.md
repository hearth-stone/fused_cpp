# mHC continuous K accumulation

## Change and scope

The delivery baseline is `744855b` on
`feat/fused_cpp/dsv4_v028_aot_delivery`. Small-M SVE mHC projection previously
selected independent K partial sums when its M blocks could not occupy the
requested threads. Large M accumulated the same cache windows consecutively.
Different FP32 accumulation order changed mHC mixing coefficients and could
change BF16 activations after the strict residual/post/pre boundaries.

`deepseek_v4_mhc_sve_projection` now always distributes M blocks across workers
and accumulates each row consecutively along K. The independent partial-buffer
allocation and reduction are removed. The default 1 MiB cache budget remains:
DeepSeek V4 K=16384 still returns KC=8192 and processes two cache windows using
one accumulator. This differs from the earlier diagnostic that enlarged the
window to 2 MiB. Large-M execution retains its existing computation path.

This is a numerical bug fix with an explicitly requested default change. API,
packed weights, MR/KC reporting, BF16 boundaries, and unsupported-ISA behavior
are unchanged. Removing K parallelism can reduce small-M parallel utilization,
particularly for decode. Rollback is limited to the projection change and its
regression tests. No production environment variable is added.

## Regression coverage

`tests/test_deepseek_v4_mhc.py` embeds the same rows at a different panel position
in a 560-token batch and requires exact equality of projections and squared
sums. Cases cover M=1/6/7/8/379/553/554/555 with 8 and 80 threads, and the default
1 MiB window. The fused pre/RMSNorm check also requires exact equality of mixing
coefficients and BF16 output for M=1/379/555. Independent-reference, empty-input,
geometry, and post/head boundary coverage remains in the same test file.

On the unmodified VL256 baseline, the 19 new regression cases produce
**12 failed, 7 passed**. Failures reproduce the small-M K-splitting mechanism;
large-M cases pass. No tolerances are relaxed.

## Target validation

Validation uses Arm-codex-internal, the existing Python 3.12 / PyTorch CPU
virtual environment, and an isolated candidate under
`/home/zhangxu/codex/mhc-no-split-delivery-20261011`. The previously deployed
SiLU-fixed candidate remains the baseline. Build, correctness, real-input
replay and benchmark results are recorded under that directory's `results/`.

Build command (from the isolated candidate):

```bash
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=16 OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  numactl --physcpubind=0-15 --membind=0 \
  /home/zhangxu/final_test/test/bin/python setup.py build_ext --inplace
```

The VL256 build succeeds with GCC 13.2.0 and PyTorch `2.13.0+cpu`. With
LLVM OpenMP preloaded and CPU/NUMA-local placement on cores 0–79 / node 0:

```bash
PYTHONPATH=<candidate>/src LD_PRELOAD=/usr/local/lib/libomp.so:/usr/lib64/libjemalloc.so \
  LD_LIBRARY_PATH=/usr/local/lib OMP_NUM_THREADS=80 \
  numactl --physcpubind=0-79 --membind=0 \
  /home/zhangxu/final_test/test/bin/python -m pytest -q \
  <candidate>/tests/test_deepseek_v4_mhc.py
```

Result: **58 passed**, including all 19 new regressions, in 4.18 seconds.
The existing tolerances and independent-reference checks are unchanged.

A separate replay uses the saved 379-token embedding/attention tensors and
layer-zero weights from the real BF16 model. At the original 1 MiB window and
80 threads, projection, post_mix, res_mix and normalized input are exactly equal
between M=379 and the same target rows embedded at offset 2240 in M=3382.
Holding incoming coefficients fixed also makes post/pre BF16 output exactly
equal across these M shapes. Original unequal coefficients from the old traces
remain unequal inputs and are not used to claim end-to-end invariance.

Runtime testing covers VL256 on this host. VL128 runtime was not exercised.
Full-model batch invariance is a separate claim: the CPU output-head GEMM has
an independently observed M-dependent numerical effect.

## Performance evidence

Four processes run simultaneously with 80 CPUs/rank, CPU ranges 0–79,
80–159, 160–239 and 240–319, memory nodes 0/1/2/3, NI=-20, LLVM OpenMP and
jemalloc. Both builds retain the original 1 MiB window for this comparison.
Inputs are the recorded layer-zero target tensors, cyclically sampled for other
M sizes. Each operation has 5 warmups and 40 samples. Fresh-process rounds run
in baseline/fixed/fixed/baseline order; raw samples, P50/P90/P99, mean and standard
deviation are archived in `mhc-perf-{baseline,fixed}{1,2}-rank{0,1,2,3}.json`.

The complete post/pre operation at M=1 has per-rank medians across two rounds
of 0.204–0.218 ms before and 0.370–0.423 ms after (+70% to +104%). This small-M
parallelism loss is retained explicitly as the cost of the requested consistent
accumulation. Other shapes have substantial round-to-round variation, including
large M where the computation path is unchanged. These measurements do not
establish a stable whole-model slowdown or a speedup. The earlier 2 MiB
no-split diagnostic is a different candidate and its numbers must not be
reported as the cost of this patch.

## BF16 model batch-only integration

The rebuilt candidate was also used for real API requests on the same host:
TP4 x 80 CPUs, NI=-20, LLVM OpenMP, jemalloc, existing rank-local calibration
and prepack cache, 32 GiB KV/rank, no prefix caching or chunked prefill, and one
output token. The same 32 MMLU geography cases ran twice per 2K/4K/8K model and
scheduler budget. Scheduling was paused in keep mode until the pre-tokenized
batch was queued, then resumed; the target was first in the request list.

| Budget | Actual target batch requests | Actual target batch tokens |
| ---: | ---: | ---: |
| 2048 | 5 | 1805 |
| 4096 | 11 | 3975 |
| 8192 | 22 | 8001 |

For `mmlu-geography-test-014`, all captured stages from input through 43 layers
and final hidden state are bit-identical across these shapes and their repeats.
All 32 cases' four answer logits match exactly across settings and repeats;
all answers agree, with 29/32 correct in all six runs. Target logits are
A=38.0, B=38.25, C=32.5, D=34.0, output B (gold B).
This is tested batch-only behavior, not a universal full-model invariant.
Instrumentation adds copies/I/O, so its request times are not performance data.

Detailed requests, comparisons, launch environment and full target traces are
retained under
`/home/zhangxu/final_test/tp4_acceptance_20261010/bf16-batch-shape-fixed-v1`.
All owned model-service processes exited, port 8018 was closed, and the machine
marker was released. No existing deployment was overwritten.

## Complete delivery suite

After the focused and model checks, the complete delivery suite was run against
the rebuilt candidate on Arm-codex-internal:

```bash
PYTHONPATH=<candidate>/src:<candidate> \
  LD_PRELOAD=/usr/local/lib/libomp.so:/usr/lib64/libjemalloc.so \
  LD_LIBRARY_PATH=/usr/local/lib OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=FALSE OMP_DYNAMIC=FALSE \
  numactl --physcpubind=0-79 --membind=0 \
  /home/zhangxu/final_test/test/bin/python -m pytest -q -rs
```

Result: **1633 passed, 1 skipped in 48.46 seconds**, exit code 0. The skip is the
intentional backend-unavailable test in `test_deepseek_v4_inv_rope_woa.py`; this
host has the native backend. The complete log and exit status are retained in
`results/full-suite.log` and `results/full-suite.exit`. Native module loading was
verified to use the isolated rebuilt candidate, and the machine marker was
released after completion.
