# Amazon C5 192-core SVE W2 direct BF16 route store

Date: 2026-07-16

## Scope

- Host: Amazon C5 192-core AArch64 system, NUMA0 CPUs 0-95 only.
- Build: project AArch64 SVE/BF16 extension at `-O2`, based on commit
  `eb086b2` plus the direct-BF16-route changes.
- Shape: tokens=2048, top-k=6, H=4096, F=512, eight uniformly routed experts.
- Execution: preplanned async path, eight concurrent expert teams, 12 threads
  per expert, split-W13 enabled.
- Method: five warmups and 31 measured calls per variant. The long-route result
  uses three independent benchmark processes and reports each process median.
- Affinity: `numactl --cpunodebind=0 --membind=0 taskset -c 0-95`.

Uniform routing gives 1536 routes per expert. The benchmark compares the full
2x2 store matrix: FP32/BF16 route elements and contiguous-down-plus-scatter or
direct route stores. Every variant uses the same W13 and W2 BFMMLA compute body.

## Implementation

The M12/M8/M4/M2/M1 W2 assembly epilogues accept the existing flat route-row
destination table. The BF16 variant converts FP32 accumulators with `BFCVT` and
stores each N owner's disjoint H stripe with `ST1H`. C++ dispatches this path in
the normal, scheduled, and async bridges and omits both the contiguous BF16
`down` allocation and route scatter. Route merge converts BF16 inputs to FP32
and retains FP32 top-k accumulation.

`FUSED_CPP_MOE_W2_BF16_ROUTE=1` enables BF16 route storage. Direct store follows
the default-on `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE` policy; setting the latter to
zero selects the BF16 contiguous-down plus scatter reference.

## Long-route result

| Process | FP32 scatter ms | FP32 direct ms | BF16 scatter ms | BF16 direct ms |
|---:|---:|---:|---:|---:|
| 1 | 9.333 | 8.082 | 8.241 | 7.733 |
| 2 | 9.314 | 8.100 | 8.229 | 7.724 |
| 3 | 9.285 | 8.087 | 8.248 | 7.726 |
| Median of medians | 9.314 | 8.087 | 8.241 | 7.726 |

BF16 direct has 4.68% higher inverse-time throughput and 4.47% lower latency
than the current FP32 direct path. Relative to BF16-with-scatter, direct BF16
reduces latency by 6.25%.

One traced call attributes the difference as follows. Trace instrumentation is
used only for stage attribution, not the headline E2E median.

| Variant | W13 critical ms | W2 direct critical ms | Merge total ms | E2E ms |
|---|---:|---:|---:|---:|
| FP32 direct | 4.875 | 2.002 | 0.570 | 8.019 |
| BF16 direct | 4.826 | 2.018 | 0.285 | 7.724 |

The W2 critical worker is effectively unchanged and is slightly slower in this
sample. Almost all measured gain comes from halving route-merge source traffic,
not from faster GEMM execution.

## Logical traffic

For this shape, one route-result tensor is 192 MiB in FP32 and 96 MiB in BF16.
Counting W2 output stores, scatter read/write when present, and route-merge
reads gives:

| Variant | Logical post-W2 payload |
|---|---:|
| FP32 scatter | 768 MiB |
| FP32 direct | 384 MiB |
| BF16 scatter | 384 MiB |
| BF16 direct | 192 MiB |

These are cache-level logical bytes, not a DRAM-traffic prediction.

## Route-length boundary

Shorter uniform tests kept the same eight teams and 96 total workers. Gains are
inverse-time throughput versus FP32 direct.

| Routes/expert | FP32 direct ms | BF16 direct ms | Gain |
|---:|---:|---:|---:|
| 12 | 0.439 | 0.440 | -0.22% |
| 48 | 0.630 | 0.627 | +0.55% |
| 192 | 1.403 | 1.385 | +1.31% |
| 1536 | 8.087 | 7.726 | +4.68% |

The direct BF16 conversion is neutral at the shortest routes. Its benefit grows
when route merge traffic becomes large enough to be visible in operator time.

## Correctness

The focused tests cover normal, scheduled, and async bridges; N-split and 2-D
dispatch; BF16 and FP32 route buffers; and route counts 1 through 23, exercising
M12 and every M8/M4/M2/M1 tail. Results on the target host:

```text
14 passed, 61 deselected
74 passed, 1 deselected  # complete non-slow fused-MoE file
```

BF16 direct is bitwise identical to BF16-with-scatter. Against FP32 direct on
the long-route benchmark, final operator-output maximum absolute error is
`1.1920928955078125e-07`, mean absolute error is `3.0599e-09`, and RMSE is
`6.7927e-09`. These are local operator metrics; real top-k=6 model-level
accuracy remains required before BF16 route storage can become the default.

## Conclusion

The implementation is retained as an opt-in SVE path. It removes another 192
MiB of logical post-W2 payload and gives a repeatable 4.7% long-route throughput
gain over FP32 direct storage, without changing W2 compute throughput. The
global default remains FP32 direct route storage until model-level validation is
available.
