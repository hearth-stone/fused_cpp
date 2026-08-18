# Amazon M5 Sparse MLA token-panel and B-block cache experiment

Date: 2026-08-17

## Decision

Retire the prototype and keep the production source unchanged. The complete
token-panel `{4,8,16}` by B-block `{64,128,256}` sweep did not approach the
predeclared 3% adoption gate on either the 2048 shared-prefix case or the 8192
shared-dense case. The best paired candidate improved 2048 by only 0.14%,
regressed 8192 by 0.30%, and regressed the later low-overlap sparse control by
0.19%.

## Prototype

Only the shared-prefix and fully shared-dense paths were changed. The QK/PV
microkernels remained the existing SVE `8x2VL` BFMMLA kernels. Adjacent tokens
formed the primary OpenMP work item; packed Q and online `(m,l,O)` state scaled
with the token panel, while score, BF16 probability, packed-P, and QK tile
scratch used one or two fixed B-block-sized slots per thread. Shared-prefix
causal work was split into a common rectangle ending at the largest full
B-block valid for every token in the panel, followed by a triangular fringe.

## Method

- Source baseline: commit `4e1df1e`; prototype was never committed.
- Host: `AmazonM5192Cores`, Neoverse V3, native SVL128, NUMA node 1,
  cores 96--191, 96 OpenMP threads.
- Affinity: `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `numactl
  --physcpubind=96-191 --membind=1`.
- Data: BF16, `h_q=32`, `d_qk=192`, `d_v=128`, seed 20260817.
- Timing: five warmups and 21 samples for the initial baseline; the nine-point
  screen used ten warmups and nine samples; selected paired comparisons used
  ten warmups and 21 samples. Values below are medians and checksums matched.
- Correctness: a direct native-versus-naive output/statistics test covering
  shared dense, ragged shared-prefix, and later indexed sparse passed for
  panel4/B64/single scratch and panel8/B256/double scratch.

The three performance cases were:

1. 2048 shared-prefix: `q[2048,32,192]`, `kv[2560,1,192]`, top-k capacity
   2560, 2,621,952 valid query-key pairs.
2. 8192 shared dense: `q[8192,32,192]`, top-k 640.
3. Later low-overlap sparse control: `q[2048,32,192]`, context start 10000,
   compressed capacity 512, window 128, `kv=15060`, top-k 640, 1,310,720
   valid pairs.

Timing command template (replace the case arguments with one of the three
lines below):

```bash
OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=96-191 --membind=1 \
  .venv/bin/python tests/bench_sparse_mla_scalable.py \
  --h-q 32 --d-qk 192 --d-v 128 --threads 96 --warmup 10 --iters 21 \
  --seed 20260817 <case arguments>

# 2048 shared-prefix
--pattern sparse --s-q 2048 --compressed-capacity 512 \
  --window-size 2048 --compress-ratio 4 --context-start 0
# 8192 shared dense, top-k = 512 + 128 = 640
--pattern dense --s-q 8192 --compressed-capacity 512 --window-size 128
# later low-overlap sparse
--pattern sparse --s-q 2048 --compressed-capacity 512 \
  --window-size 128 --compress-ratio 4 --context-start 10000
```

Each parameter point was built with compile-time values for token panel,
B-block, and scratch-buffer count, then the resulting extension `.so` was
saved separately. The baseline and candidate were swapped into the same Python
environment; compiler, link inputs, affinity, NUMA binding, tensors, and seed
were otherwise unchanged.

## Parameter sweep

Single-scratch results:

| Token panel | B block | 2048 shared-prefix | 8192 shared-dense |
|---:|---:|---:|---:|
| 4 | 64 | 11.650 ms | 9.214 ms |
| 4 | 128 | 11.635 ms | 9.222 ms |
| 4 | 256 | 11.623 ms | 9.237 ms |
| 8 | 64 | 11.635 ms | 9.231 ms |
| 8 | 128 | 11.621 ms | 9.255 ms |
| 8 | 256 | 11.623 ms | 9.226 ms |
| 16 | 64 | 11.650 ms | 9.243 ms |
| 16 | 128 | 11.640 ms | 10.833 ms |
| 16 | 256 | 11.646 ms | 9.327 ms |

The initial baseline was 11.658 ms for 2048 and 9.302 ms for 8192. These
screening differences are small enough to require a paired confirmation;
panel16/B128 also showed a clear 8192 regression. Changing panel8/B256 from
single scratch to double scratch produced 11.637/9.235 ms, so double buffering
did not help.

## Paired confirmation

Panel8/B256/single scratch was selected as a representative stable candidate:

| Case | Candidate | Baseline | Latency change |
|---|---:|---:|---:|
| 2048 shared-prefix | 11.619 ms, 4.622 TFLOP/s | 11.635 ms, 4.615 TFLOP/s | -0.14% |
| 8192 shared-dense | 9.250 ms, 11.607 TFLOP/s | 9.222 ms, 11.643 TFLOP/s | +0.30% |
| Later low-overlap sparse | 2.611 ms, 10.279 TFLOP/s | 2.606 ms, 10.300 TFLOP/s | +0.19% |

The sparse control stayed inside the 1% regression guardrail, but neither
shared path met the required 3% improvement.

## PMU evidence

`perf stat` was collected over 10 warmups plus 200 calls. The machine exposed
core PMU events `l2d_cache_refill`, `ll_cache_miss_rd`, `bus_access`, and
`mem_access`, but no uncore/DDR controller PMU. Therefore `bus_access * 64 B /
duration` below is only a cache-line-sized bus-traffic proxy, not measured DDR
read/write bandwidth. `kernel.perf_event_paranoid` was temporarily changed from
4 to 0 for collection and restored to 4 afterward.

| Case | Metric | Candidate | Baseline | Change |
|---|---|---:|---:|---:|
| 2048 | L2 data refills | 1,049,263,541 | 1,256,175,956 | -16.47% |
| 2048 | LLC read misses | 4,929,897 | 4,681,826 | +5.30% |
| 2048 | bus accesses | 3,008,254,960 | 3,391,021,031 | -11.29% |
| 2048 | duration | 3.2478 s | 3.2556 s | -0.24% |
| 2048 | 64-byte bus proxy | 59.29 GB/s | 66.67 GB/s | -11.07% |
| 8192 | L2 data refills | 795,651,017 | 796,042,185 | -0.05% |
| 8192 | LLC read misses | 22,991,036 | 22,681,300 | +1.37% |
| 8192 | bus accesses | 2,712,268,743 | 2,692,825,231 | +0.72% |
| 8192 | duration | 3.0613 s | 2.9991 s | +2.07% |
| 8192 | 64-byte bus proxy | 56.70 GB/s | 57.47 GB/s | -1.34% |

A second counter pass gave the same qualitative result: for 2048, candidate
L2 refills and bus accesses were 12.2% and 8.4% lower while LLC read misses
were 1.2% higher; for 8192, L2 refills and bus accesses differed by less than
0.1% and LLC misses by about 1%.

## Interpretation

The 2048 schedule does reduce private-cache and bus traffic, but the unchanged
wall time shows that this traffic is not the limiting resource at 96 threads.
The same QK, online-softmax, BF16-P conversion/packing, and PV work is still
performed by the same `8x2VL` kernels; the outer schedule only changes when B
is revisited. At 8192, a fully shared B range is already reused effectively by
the existing packed layout and cache hierarchy, so panelization removes almost
no refills and adds loop/state-management overhead. Increasing the panel to 16
can also enlarge the live A/C state enough to cause unstable regressions.

The next CPU-oriented FlashAttention experiment should change the amount or
shape of kernel work, not only outer-loop cache order: for example a fused
multi-query microkernel/epilogue that amortizes QK/softmax/PV state management,
or a schedule chosen from measured per-NUMA working-set pressure rather than a
fixed token panel.
