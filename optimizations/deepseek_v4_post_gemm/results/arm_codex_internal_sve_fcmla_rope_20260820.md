# SVE FCMLA RoPE with interleaved rotary cache

## Question

Can RoPE use SVE `FCMLA` with an offline interleaved `[cos,sin]` cache to beat
the production split-even/odd `MUL/FMLS/FMLA` implementation?

## Variants

- Baseline: interleaved input rows, `LD2` even/odd deinterleave, split
  `[all cos][all sin]` cache, `MUL/FMLS/FMLA`, and `ST2` or BF16 scatter.
- Layout control: interleaved cache, but otherwise the baseline BF16
  split-arithmetic implementation.
- FCMLA: interleaved input and cache, contiguous load/store, then
  `FCMLA #0` plus `FCMLA #90` in FP32. BF16 values are widened before FCMLA
  and rounded once on store.

The cache conversion is outside the timed region because the proposed model
contract would generate the interleaved layout once, rather than repacking it
for every operator invocation.

## Configuration

- Host: `Arm-codex-internal`, SVE256
- Affinity: CPU 0 for 1T; CPUs 0-79 for multithread tests
- RoPE dimension: 64
- Main-Q shape: 2048 tokens x 128 heads = 262,144 rows, head dimension 192
- KV-like shape: 2,048 rows, head dimension 192
- Timing: alternating baseline/candidate order, 51 medians unless noted
- Compiler: GCC, `-O2 -std=c++17 -march=armv8.6-a+sve`

GNU `objdump` confirms that the candidate loop contains exactly the intended
`fcmla ... #0` and `fcmla ... #90` pair and no helper call.

The Lab benchmark accepts `--unroll 1|2|4|8`. Each specialization uses a full
predicate for complete blocks and the original predicated loop for the tail.
Disassembly confirms 2, 4, or 8 FCMLA groups per main-loop branch and no vector
spill in the pure-RoPE loop body. GCC reuses the same architectural Z registers
between groups, so this is loop/control unrolling that relies on out-of-order
register renaming; it is not a hand-scheduled kernel with 2, 4, or 8
simultaneously live outputs. The QNorm prototype still spills one Z register
across `inverse_rms_bf16`; the unroll sweep does not remove that independent
issue.

## Correctness

For the 262,144-row Main-Q workload:

- FP32 maximum absolute error against the scalar reference: zero in the
  reported precision.
- Pure BF16 RoPE: 182 mismatches over 16,777,216 values (0.00108%), maximum
  absolute difference 0.015625. The baseline itself had 382 mismatches against
  the same scalar reference.
- BF16 QNorm+RoPE: 209 differences between baseline and FCMLA over 50,331,648
  values (0.000415%), maximum absolute difference 0.007812.

The differences are BF16 boundary-rounding effects from the changed fused
operation order, not a sign or cache-layout error.

## Performance

Positive gain means FCMLA is faster.

### Pure RoPE

| Workload | Threads | Dtype | Baseline | FCMLA | Gain |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 row | 1 | FP32 | 0.129 us | 0.131 us | -1.5% |
| 1 row | 1 | BF16 | 0.160 us | 0.145 us | +10.6% |
| 32 rows | 1 | FP32 | 0.488 us | 0.472 us | +3.4% |
| 32 rows | 1 | BF16 | 1.211 us | 0.890 us | +36.1% |
| 2,048 rows | 1 | FP32 | 65.354 us | 58.108 us | +12.5% |
| 2,048 rows | 1 | BF16 | 96.566 us | 82.142 us | +17.6% |
| 2,048 rows | 16 | FP32 | 3.381 us | 3.654 us | -7.5% |
| 2,048 rows | 16 | BF16 | 6.184 us | 5.065 us | +22.1% |
| 2,048 rows | 80 | FP32 | 4.650 us | 5.620 us | -17.3% |
| 2,048 rows | 80 | BF16 | 4.733 us | 5.593 us | -15.4% |
| 262,144 rows | 1 | FP32 | 3.761 ms | 4.081 ms | -7.9% |
| 262,144 rows | 1 | BF16 | 9.421 ms | 5.946 ms | +58.4% |
| 262,144 rows | 80 | FP32 | 49.0 us | 41.4 us | +18.5% |
| 262,144 rows | 80 | BF16 | variable | 73-75 us | +59% or more |

The interleaved-cache split-arithmetic control did not explain the BF16 gain.
At 2,048 rows/16T it took 6.848 us versus 5.053 us for FCMLA; at Main-Q/80T
it took at least 123 us versus about 74 us. The winning mechanism is therefore
the combined contiguous BF16 load/store and FCMLA kernel, not cache layout
alone.

### BF16 QNorm plus RoPE

The complete Lab stage includes RMS reduction, normalization of the 128 NoPE
dimensions, and the 64-dimensional RoPE. QNorm-only repeated runs remove the
preceding pure-RoPE benchmark's resource-state bias.

| Rows | Threads | Baseline | FCMLA | Paired gain |
| ---: | ---: | ---: | ---: | ---: |
| 262,144 | 1 | 27.950 ms | 22.751 ms | +22.9% |
| 262,144 | 80 | 359.20 us | 345.00 us | +4.18% |
| 262,144 | 80 | 357.47 us | 345.31 us | +2.94% |
| 262,144 | 80 | 345.67 us | 345.01 us | +0.70% |

The pure BF16 RoPE gain is large and repeatable, but RMSNorm and NoPE traffic
dilute it to a 0.7-4.2% complete-stage change at 80T. The baseline also showed
two performance modes, while FCMLA remained near 345 us.

## 80-thread KV-like regression diagnosis

With a fixed 2,048 rows, increasing threads reduces useful work per worker.
The paired BF16 curve is:

| Threads | Rows/core | Baseline us | FCMLA us | Paired gain |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | 6.086 | 4.996 | +21.8% |
| 32 | 64 | 4.430 | 3.952 | +12.3% |
| 40 | 51 | 4.296 | 4.165 | +3.2% |
| 48 | 43 | 4.260 | 4.472 | -4.7% |
| 64 | 32 | 4.366 | 4.862 | -10.2% |
| 80 | 26 | 4.949 | 5.228 | -5.1% |

To distinguish insufficient work per core from an FCMLA or cross-LLC shared
resource limit, a second sweep held 128 rows per core while increasing total
rows with thread count:

| Threads | Total rows | Baseline us | FCMLA us | Paired gain |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4,096 | 6.877 | 5.932 | +16.2% |
| 40 | 5,120 | 7.086 | 5.282 | +34.2% |
| 48 | 6,144 | 7.887 | 5.824 | +35.9% |
| 64 | 8,192 | 8.272 | 6.196 | +33.5% |
| 80 | 10,240 | 8.714 | 7.742 | +12.5% |

The positive 48T and 64T results cross the machine's 40-core LLC-domain
boundary, so that boundary and a global FCMLA execution ceiling are not the
primary cause. At SVE256, the split baseline handles eight complex pairs per
loop using two Z registers, while one FCMLA Z register holds four pairs. The
FCMLA path therefore executes twice as many short vector-loop iterations per
row. Its contiguous BF16 load/store savings dominate when each worker owns
enough rows, but cannot amortize loop startup, per-worker setup and OpenMP fixed
cost at only 26-43 rows per worker. The lower but still positive 80T
constant-work result indicates a secondary full-NUMA/topology overhead, not a
fundamental FCMLA scaling failure.

## FCMLA unroll sweep

The unroll sweep used the same SVE256 binary, data, affinity and alternating
measurement method as the preceding tests. All variants passed the scalar and
split-path checks. A non-multiple block case (`rows=7`, `rope_dim=48`) also
passed for unroll 1, 2, 4 and 8, covering the predicated remainder path.

### KV-like shape

Candidate times below compare FCMLA variants directly; lower is better.

| Rows | Threads | Stage | U1 | U2 | U4 | U8 | Best |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2,048 | 1 | FP32 RoPE | 43.388 us | 50.867 us | 47.247 us | 30.685 us | U8 |
| 2,048 | 1 | BF16 RoPE | 61.653 us | 66.767 us | 62.336 us | 49.816 us | U8 |
| 2,048 | 1 | BF16 QNorm+RoPE | 201.949 us | 214.944 us | 195.055 us | 174.418 us | U8 |
| 2,048 | 16 | FP32 RoPE | 3.074 us | 2.945 us | 2.915 us | 2.930 us | U4 |
| 2,048 | 16 | BF16 RoPE | 4.582 us | 4.463 us | 4.563 us | 4.404 us | U8 |
| 2,048 | 16 | BF16 QNorm+RoPE | 12.437 us | 12.555 us | 12.306 us | 12.086 us | U8 |
| 2,048 | 80 | BF16 RoPE | 4.297 us | 3.681 us | 4.447 us | 4.158 us | U2 |
| 2,048 | 80 | BF16 QNorm+RoPE | 5.072 us | 5.306 us | 5.534 us | 5.303 us | U1 |

Unroll 8 removes enough per-row control to improve the 1T BF16 candidate by
19.2 percent relative to unroll 1. At 16T the improvement is only 3.9 percent,
and at 80T fixed OpenMP and worker setup costs dominate the complete stage.

### Main-Q shape

| Rows | Threads | Stage | U1 | U2 | U4 | U8 | Best |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 262,144 | 1 | FP32 RoPE | 5.494 ms | 6.171 ms | 4.835 ms | 3.341 ms | U8 |
| 262,144 | 1 | BF16 RoPE | 5.488 ms | 5.562 ms | 5.562 ms | 5.569 ms | U1 |
| 262,144 | 1 | BF16 QNorm+RoPE | 22.349 ms | 22.408 ms | 22.285 ms | 22.373 ms | U4 |
| 262,144 | 80 | FP32 RoPE | 36.360 us | 36.992 us | 37.107 us | 37.545 us | U1 |
| 262,144 | 80 | BF16 RoPE | 65.393 us | 65.698 us | 66.336 us | 66.033 us | U1 |
| 262,144 | 80 | BF16 QNorm+RoPE | 273.612 us | 273.596 us | 279.151 us | 275.295 us | U1/U2 |

The production-relevant Main-Q BF16 result is flat within about one percent
for pure RoPE and favors U1/U2 for the complete 80T stage. Loop unrolling is
therefore useful for the single-core FP32 and small fixed-work cases, but it
does not close the BF16 Main-Q production gate by itself. Do not select a
global unroll factor from the single-core result.

An isolated `--qnorm-only` repeat measured U1/U2/U8 at 274.371, 275.334 and
277.906 us respectively, confirming that U1 is the best of these variants for
Main-Q/80T. The current U1 specialization is stronger than the initial FCMLA
prototype: it uses `PTRUE` for all eight complete vectors in a 64-dimensional
row and avoids generating `WHILELT` for each vector. This bulk-loop
restructuring reduced the candidate from the earlier approximately 345 us to
approximately 274 us, a 20.6 percent latency reduction. That change, rather
than 2/4/8 unrolling, is the important follow-up result.

## Decision

Do not migrate the global rotary-cache layout or default RoPE implementation
directly from this Lab experiment. The stronger bulk-loop result warrants the
narrow Main-Q production prototype, but the broad adoption gate still fails:
BF16 is not bit exact, FP32 and KV-like coverage is inconsistent, and no real
post-GEMM E2E result has yet validated the different cache contract.

Keep the Lab experiment for one narrow follow-up: integrate FCMLA only into the
production BF16 Main-Q QNorm+RoPE path with an explicitly prepared interleaved
cache, then measure the real four-row production kernel and post-GEMM E2E. KV,
FP32, inverse-RoPE and scalar fallbacks should retain their current layout and
implementation unless separately proven.
