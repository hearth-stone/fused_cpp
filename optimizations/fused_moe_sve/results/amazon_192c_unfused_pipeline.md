# Explicit unfused pipeline versus production fusion

Date: 2026-07-15

Host: `AmazonC5192Cores`, NUMA node 0, CPUs `0-95`. The benchmark used the
EP2 expert shape `H=4096`, `F=2048`, four concurrent experts, 24 threads per
expert, two W13 N ranges, two rotating copies, three warmups, and ten measured
iterations.

Both variants use the production SVE assembly. Plain W1/W3 call
`moe_sve_w2_packed_m12`, fused W13 calls
`moe_sve_w13_silu_poly5_packc_m12_rows_opt`, and both W2 stages call
`moe_sve_w2_packed_bf16_m12`. All four entrypoints execute the same
`M12_K4_BODY`. Buffers are preallocated and weights are distinct by expert and
rotating copy.

## Total wall time

| Routes per expert | Explicit unfused | Production fused | Fused speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 1020 | 8.831 ms / 23.254 effective TFLOP/s | 8.479 ms / 24.218 effective TFLOP/s | 1.0415x | 3.98% |
| 2040 | 16.970 ms / 24.202 effective TFLOP/s | 16.412 ms / 25.024 effective TFLOP/s | 1.0340x | 3.29% |

Wall time is measured from the first expert team entering its first stage to
the last team leaving W2. Each expert has an independent team barrier. Worker
pool wakeup and buffer allocation are outside the interval; required
dependencies between materialized stages remain inside it. The ten samples
balance both execution orders (`unfused -> fused` and `fused -> unfused`) across
the two rotating weight copies.

## Stage breakdown

The entries below are median maximum-team times in milliseconds.

| Routes | Extract | Pack A1 | W1 | SiLU | W3 | Multiply | Pack A2 | Unfused W2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1020 | 0.133 | 0.079 | 2.848 | 0.137 | 2.818 | 0.157 | 0.052 | 2.706 |
| 2040 | 0.310 | 0.139 | 5.432 | 0.288 | 5.349 | 0.346 | 0.061 | 5.161 |

| Routes | Fused gather-pack A1 | Fused W13 + SiLU/mul/packC | Fused W2 |
|---:|---:|---:|---:|
| 1020 | 0.124 | 5.625 | 2.736 |
| 2040 | 0.306 | 10.869 | 5.209 |

At 2040 routes, explicit W1 + SiLU + W3 + multiply + pack A2 totals about
11.48 ms by stage medians, versus 10.87 ms for fused W13. Direct gather-pack
also replaces about 0.45 ms of extract plus pack A1 with 0.31 ms. W2 is nearly
unchanged, which is the expected control when its input layout and assembly
entrypoint are identical.

## Correctness

For 2040 routes, the explicitly packed BF16 intermediate versus fused packC
had relative L2 error `1.05e-5`, maximum absolute error `6.10e-5`, and 65 bitwise
differences. Final W2 output had relative L2 error `4.96e-5`, maximum absolute
error `3.05e-5`, and 13,004 bitwise differences among 33,423,360 elements.

The nonzero difference is expected: the requested explicit sequence stores
`SiLU(gate)` before multiplying by `up`, while the fused epilogue evaluates
`gate * up / denominator` before BF16 conversion. GEMM accumulation order,
poly5 coefficients, exact `fdiv`, BF16 conversion, and W2 are otherwise held
constant.

This experiment ends at per-route BF16 W2 output. Route-weight accumulation
and scatter are intentionally excluded so they cannot dilute the kernel fusion
effect.

## Working-set scaling

The same `4 expert x 24 threads` configuration was swept to longer routes with
ten balanced-order samples per point. `Tensor MiB` is the size of one BF16
`[E,M,H]` or FP32 `[E,M,F]` materialized tensor across the four active experts.

| Routes | Tensor MiB | Explicit unfused | Production fused | Fused reduction | Unfused TFLOP/s | Fused TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1020 | 31.875 | 8.831 ms | 8.479 ms | 3.98% | 23.254 | 24.218 |
| 2040 | 63.750 | 16.970 ms | 16.412 ms | 3.29% | 24.202 | 25.024 |
| 3060 | 95.625 | 25.276 ms | 24.223 ms | 4.17% | 24.373 | 25.433 |
| 4080 | 127.500 | 33.504 ms | 32.031 ms | 4.40% | 24.517 | 25.644 |
| 6120 | 191.250 | 50.724 ms | 48.021 ms | 5.33% | 24.291 | 25.658 |
| 8160 | 255.000 | 67.785 ms | 63.968 ms | 5.63% | 24.236 | 25.682 |

The normalized input preparation cost provides the clearest stage-level
separation. From 2040 to 8160 routes, explicit `extract + packA1` rises from
`0.226` to `0.332 ms` per 1020 routes (`+47%`), while direct fused gather-pack
rises from `0.151` to `0.157 ms` (`+4%`). The unfused aggregate throughput
peaks near 4080 routes and then falls, whereas fused throughput remains flat to
slightly higher through 8160 routes.

This is controlled evidence that materialized stages scale worse as their
working set grows. It is consistent with a cache-capacity transition: one
materialized tensor reaches 191.25 MiB at 6120 routes versus 192 MiB aggregate
private L2 across the 96 cores, and the unfused path has multiple such live
tensors. The timing data alone does not establish the mechanism; the PMU
measurements below test that interpretation directly.

## PMU validation

After setting `kernel.perf_event_paranoid=1`, ordinary-user `perf 7.0.6`
successfully counted the child process and all worker threads. Each variant ran
in a separate process through `--variant unfused|fused`. The 2040-route runs
used five warmups and 600 measured iterations; the 8160-route runs used five
warmups and 200 measured iterations. The fixed shape, affinity, NUMA policy,
weights, and buffers were otherwise identical.

Six hardware events fit without multiplexing. The two event sets were:

```text
cycles:u,instructions:u,l1d_cache:u,l1d_cache_refill:u,l2d_cache:u,l2d_cache_refill:u
cycles:u,l3d_cache_refill:u,ll_cache_rd:u,ll_cache_miss_rd:u,mem_access:u,stall_backend_mem:u
```

The command template was:

```bash
perf stat -x, -e EVENT_SET -- \
  numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  optimizations/fused_moe_sve/benchmarks/bench_unfused_pipeline \
  --experts 4 --m M --h 4096 --f 2048 --threads-per-expert 24 \
  --w13-ranges 2 --copies 2 --cpu-start 0 --warmup 5 --iters ITERATIONS \
  --skip-check --variant VARIANT
```

`M=2040` used `ITERATIONS=600`; `M=8160` used `ITERATIONS=200`.
`VARIANT` was separately set to `unfused` and `fused` for each event set.

The values below are whole-process totals in billions. They include identical
buffer/weight initialization plus five selected-path warmups, so common setup
dilutes the within-shape fused reductions. Event sets were collected in
separate runs; only values from the same event set and shape are compared.

| Routes | Counter | Explicit unfused | Production fused | Reduction |
|---:|---|---:|---:|---:|
| 2040 | Instructions | 12,225.7 B | 12,171.3 B | 0.45% |
| 2040 | L1D refill | 43.037 B | 39.323 B | 8.63% |
| 2040 | L2D refill | 46.395 B | 42.856 B | 7.63% |
| 2040 | Last-level cache reads | 1.120 B | 0.533 B | 52.39% |
| 2040 | Last-level read misses | 0.940 B | 0.429 B | 54.34% |
| 2040 | Backend memory-stall cycles | 88.088 B | 49.712 B | 43.57% |
| 8160 | Instructions | 16,565.6 B | 16,492.0 B | 0.44% |
| 8160 | L1D refill | 57.321 B | 53.198 B | 7.19% |
| 8160 | L2D refill | 61.178 B | 55.192 B | 9.78% |
| 8160 | Last-level cache reads | 1.489 B | 0.664 B | 55.43% |
| 8160 | Last-level read misses | 1.296 B | 0.569 B | 56.06% |
| 8160 | Backend memory-stall cycles | 159.250 B | 67.023 B | 57.91% |

In the same long runs, fused median latency was lower by 3.98% at 2040 routes
and 5.72% at 8160 routes. Instructions changed by less than 0.5%, and the
architectural `mem_access` event changed by 0.63% at both shapes. In contrast,
absolute last-level read misses and memory-stall cycles fell sharply. The
memory-stall fraction of cycles fell from 2.83% to 1.65% at 2040 routes and
from 3.76% to 1.67% at 8160 routes.

An independent 8160-route repeat collected the four key events again. Its
last-level reads, last-level read misses, and backend memory-stall reductions
were 55.42%, 55.98%, and 57.23%, versus 55.43%, 56.06%, and 57.91% in the
first run. The direction and magnitude therefore exceed run-to-run PMU noise.

These counters confirm the direction of the cache-pressure mechanism: fusion
primarily avoids hierarchy refills and backend stalls rather than changing the
shared GEMM instruction body. The larger stall reduction at 8160 routes also
matches the route-length scaling and p99 results. `l3d_cache_refill` reports
zero on this platform, so `ll_cache_rd` and `ll_cache_miss_rd` are used as the
last-level proxies. They do not provide an exact DRAM-byte or bandwidth
measurement; uncore memory-controller counters would still be needed for that.

## Controlled tail latency

One hundred balanced-order samples were collected at each endpoint after five
warmups:

| Routes | Variant | Median | p99 | p99 reduction | CV |
|---:|---|---:|---:|---:|---:|
| 2040 | explicit unfused | 16.993 ms | 17.095 ms | - | 0.226% |
| 2040 | production fused | 16.383 ms | 16.428 ms | 3.90% | 0.127% |
| 8160 | explicit unfused | 67.454 ms | 67.749 ms | - | 0.223% |
| 8160 | production fused | 63.696 ms | 63.851 ms | 5.75% | 0.103% |

The controlled p99 gain grows with the working set, and fused execution has
about half the run-to-run coefficient of variation. These measurements show
operator-level tail stability on an otherwise idle NUMA node; they do not yet
establish serving p99 under concurrent requests, communication, or scatter.
