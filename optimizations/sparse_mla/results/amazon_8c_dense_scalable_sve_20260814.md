# Sparse MLA Dense Scalable SVE 8x2VL (2026-08-14)

## Scope and decision rule

This record covers the production optimization that replaces the fixed 8x8
QK/PV compute in the fully shared, contiguous-KV, head-major dense subpath with
the same scalable SVE BFMMLA 8x2VL kernels used by the gathered sparse subpath.
The M dimension remains eight consecutive query heads of one token; 2VL is the
key-column dimension for QK and the value-column dimension for PV.

Primary change class: O (production optimization), validation L2+L3. Public
signatures, output/statistics definitions, online-softmax semantics, and BF16
numerical tolerance are unchanged. The internal packed K/V layout changes only
inside the SVE dense fast path. The fixed 8x8 implementation remains the
non-SVE fallback.

The predeclared adoption rule was:

- forced SVL128 and SVL256 must both pass the full Sparse MLA test file;
- the representative dense workload must not regress by more than 3% at either
  VL for one or eight threads; and
- the existing gathered sparse workload must show no stable regression.

The rollback boundary is the contiguous scalable K/V pack helpers, their dense
call-level buffers, and the dense QK/PV call sites in `csrc/sparse_mla.cpp`.

## Preconditions and implementation

The optimized subpath retains its existing BF16 MQA preconditions: eight-head
groups, `Dqk%4==0`, `Dv%8==0`, `topk%8==0`, no attention sink, and one common
contiguous KV interval for every query row. It is selected only when
`ceil(Sq/8) > max(1, requested_threads/2)`; shorter queries retain the guarded
fallback dispatch. Unsupported shapes and patterns also continue to their
existing fallbacks.

- Q keeps the existing four 2-row A panels per K=4 reduction block.
- Contiguous K is packed once per call into four scalable B vectors per K=4.
  The last SVL256 key tile may contain eight valid keys and eight zero columns.
- Contiguous V is packed once per call and output tile. A NEON 4x8 transpose
  emits the four BFMMLA B panels for each 128-bit SVE segment. The last
  SVL256 value tile may contain eight valid columns and is zero padded.
- QK invokes `qkt_8x2vl_bf16`; SVL128 computes 8x8 and SVL256 computes 8x16.
  Only valid columns of a final half tile enter online softmax.
- Online softmax is unchanged. Its BF16 probabilities are packed as 2-row A
  panels and passed to `pv_8x2vl_bf16`; final output lanes are predicated.
- The SVE L2 key chunk is rounded down to a runtime 2VL multiple. The non-SVE
  branch preserves its original `Sc_l2`, stack-resident 8x8 score tile, and
  fixed 8-column PV stepping.

The shared V transpose helper also avoids initializing temporary rows to zero
when a full eight-column segment is known to be valid. This affects the existing
gathered sparse V pack but does not change its packed representation.

## Method

- Host: `AmazonECS8Cores`, Neoverse V1, CPUs 0--7, Linux AArch64.
- Baseline source: clean commit `7a9f17bd308a`, where gathered sparse QK/PV was
  already scalable but the fully shared dense path still used fixed 8x8.
- Candidate build: release `_C`, GCC 13.3, `-O2
  -march=armv8.6-a+sve+bf16+i8mm`, profiling disabled, and no fixed
  `-msve-vector-bits` option.
- Dense workload: BF16 `q[2048,32,192]`, `kv[640,1,192]`, `topk=640`,
  `d_v=128`, 1,310,720 valid query/KV pairs, seed `20260814`.
- Source work: 26,843,545,600 FLOPs. Timing covers the public call, including
  validation, call-level packing, QK, online softmax, PV, output conversion,
  and OpenMP scheduling.
- Each point uses 5 warmups and 21 timed calls. Libraries are held to one
  thread. One-thread runs use CPU 0; eight-thread runs use CPUs 0--7.
- SVL128 is forced with `prctl(PR_SVE_SET_VL, 16)` before importing the
  extension; SVL256 uses 32 bytes. The launcher verifies the selected VL.

Build:

```bash
FUSED_CPP_RELEASE=1 FUSED_CPP_ENABLE_PROFILING=0 MAX_JOBS=8 \
  .venv/bin/python setup.py build_ext --inplace
```

Benchmark:

```bash
env PYTHONPATH=src OMP_NUM_THREADS=<threads> \
  FUSED_CPP_NUM_THREADS=<threads> OMP_DYNAMIC=FALSE \
  OMP_PROC_BIND=close OMP_PLACES=cores MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  SVE_VL_TEST_STUB_MOE=1 taskset -c <cores> \
  .venv/bin/python /tmp/run_with_sve_vl.py <16-or-32> \
  tests/bench_sparse_mla_scalable.py --pattern dense \
  --threads <threads> --warmup 5 --iters 21 --seed 20260814
```

## Dense results

| VL | Threads | Version | Median | Mean +/- std | P90 / P99 | Range | Source throughput | Latency change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| SVL128 | 1 | fixed 8x8 | 334.450 ms | 334.618 +/- 0.439 | 335.383 / 335.665 | 334.099--335.677 | 80.26 GFLOP/s | baseline |
| SVL128 | 1 | SVE 8x2VL | 250.042 ms | 250.094 +/- 0.301 | 250.126 / 251.114 | 249.698--251.332 | 107.36 GFLOP/s | -25.24%, 1.3376x |
| SVL128 | 8 | fixed 8x8 | 54.206 ms | 54.320 +/- 0.474 | 54.387 / 55.967 | 54.118--56.356 | 495.21 GFLOP/s | baseline |
| SVL128 | 8 | SVE 8x2VL | 36.384 ms | 36.413 +/- 0.117 | 36.531 / 36.690 | 36.200--36.701 | 737.79 GFLOP/s | -32.88%, 1.4898x |
| SVL256 | 1 | fixed 8x8 | 332.588 ms | 332.611 +/- 0.221 | 332.939 / 332.960 | 332.290--332.963 | 80.71 GFLOP/s | baseline |
| SVL256 | 1 | SVE 8x2VL | 213.853 ms | 213.868 +/- 0.072 | 213.964 / 214.022 | 213.769--214.032 | 125.52 GFLOP/s | -35.70%, 1.5552x |
| SVL256 | 8 | fixed 8x8 | 54.248 ms | 55.831 +/- 4.677 | 56.062 / 70.213 | 54.150--70.465 | 494.83 GFLOP/s | baseline |
| SVL256 | 8 | SVE 8x2VL | 32.610 ms | 32.658 +/- 0.120 | 32.771 / 33.004 | 32.506--33.060 | 823.18 GFLOP/s | -39.89%, 1.6635x |

The SVL256 baseline contains two long outliers, so the adoption comparison uses
the predeclared median. Its median is consistent with the other fixed-8x8
points, while the candidate distribution is narrow. The candidate clears the
3% rejection boundary in every configuration, including SVL128 where 2VL is
still eight columns: the gain there comes from the SVE BFMMLA compute and
scalable packed path rather than a wider tile.

## Gathered sparse regression check

The same final binary was also measured with the original DSV4-like sparse
workload: BF16 `q[2048,32,192]`, capacity 640 (`512+128`), `kv[2560,1,192]`,
`d_v=128`, and 777,792 valid pairs per head. The baseline is the selected
scalable sparse QK/PV result at commit `7a9f17bd308a`.

| VL | Threads | Baseline median | Candidate median | Latency change |
|---|---:|---:|---:|---:|
| SVL128 | 1 | 201.848 ms | 201.069 ms | -0.39% |
| SVL128 | 8 | 28.257 ms | 28.191 ms | -0.23% |
| SVL256 | 1 | 180.951 ms | 170.741 ms | -5.64% |
| SVL256 | 8 | 26.817 ms | 25.743 ms | -4.00% |

There is no measured gathered-path regression. The larger SVL256 change is
consistent with removing redundant zero initialization in full V-pack
segments, but this single-host result is recorded as supporting evidence rather
than a separate general performance claim.

## Correctness, fallback, and code generation

- Final release build succeeded and auto-detected a maximum SVL of 256 bits.
- Forced SVL128, CPUs 0--3, four OpenMP threads:
  `pytest -q tests/test_sparse_mla.py` -> `24 passed` in 31.10 seconds.
- Forced SVL256, CPUs 4--7, four OpenMP threads: the same file -> `24 passed`
  in 30.81 seconds. The file includes shared-dense output, statistics, `out=`,
  unsupported-shape fallback, `Dv=24`, and final-half-tile coverage.
- A separate AArch64 syntax compile of `csrc/sparse_mla.cpp` with
  `-march=armv8.6-a+bf16+i8mm` and no `+sve` succeeded, proving that the fixed
  non-SVE branch remains buildable. Non-SVE runtime performance was not
  remeasured.
- Final `qkt_8x2vl_bf16` and `pv_8x2vl_bf16` each contain 16 static `bfmmla`
  instructions per K=4 loop body. QK has no stack frame; PV has only the ABI
  save/restore of `d8`--`d15`. Neither kernel spills a Z register. Disassembly
  of the dense OpenMP workers contains direct calls to both symbols.

The output checksums vary slightly with VL because tile boundaries and BF16
rounding differ: dense is `-25521.105469` at SVL128 and `-25521.078125` at
SVL256 for one thread. The reference tests above validate the existing
numerical tolerance rather than requiring bitwise equality across VLs.

## Final candidate raw samples

```text
dense_svl128_1t=250.241,249.698,249.996,249.999,250.088,250.042,250.066,250.015,250.030,249.998,250.028,249.920,250.126,249.957,251.332,250.114,250.055,250.066,250.121,250.029,250.050
dense_svl128_8t=36.332,36.200,36.701,36.384,36.359,36.531,36.327,36.367,36.409,36.335,36.472,36.384,36.448,36.332,36.319,36.313,36.645,36.512,36.493,36.358,36.454
dense_svl256_1t=213.909,214.032,213.964,213.905,213.896,213.862,213.908,213.820,213.849,213.810,213.935,213.850,213.853,213.790,213.828,213.982,213.836,213.772,213.880,213.769,213.772
dense_svl256_8t=32.698,32.557,32.737,32.610,32.688,32.771,32.571,32.609,32.506,32.702,32.606,32.712,32.599,32.534,32.606,32.659,32.563,32.780,33.060,32.578,32.664
sparse_svl128_1t=200.939,201.186,200.889,201.272,200.960,201.206,200.817,201.148,200.850,201.553,200.937,201.517,201.069,201.159,200.753,201.286,200.906,201.151,200.856,201.177,200.733
sparse_svl128_8t=28.085,28.336,28.191,28.367,28.102,28.342,28.108,28.312,28.154,28.344,28.051,28.397,28.088,28.338,28.115,28.319,28.104,28.315,28.132,28.334,28.080
sparse_svl256_1t=170.644,171.547,170.741,171.342,170.741,171.181,170.653,171.213,170.727,171.078,170.470,170.993,170.412,170.991,170.537,170.932,170.509,170.928,170.413,170.837,170.687
sparse_svl256_8t=25.668,25.940,25.723,25.926,25.696,25.971,25.719,25.929,25.708,25.972,25.686,25.994,25.685,25.934,25.743,25.936,25.692,25.872,25.675,25.934,25.666
```

## Checkpoint decision

Adopt the scalable dense QK/PV 8x2VL path for supported SVE BF16 shared-dense
work. It passes both runtime VLs, materially beats fixed 8x8 in all deciding
measurements, and does not regress the gathered sparse path. Keep fixed 8x8 as
the non-SVE fallback. Cross-host performance remains unmeasured in this record;
claims are therefore scoped to the named Neoverse V1 host and workload.
