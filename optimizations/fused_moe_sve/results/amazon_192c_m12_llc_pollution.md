# M12 streaming-weight LLC pollution

Date: 2026-07-15

## Question

An M12 expert consumes every packed-B cache line once. It therefore has no
future B reuse to protect, but ordinary loads can still allocate in shared
cache and displace a different expert's reusable weights. This experiment
separates those two meanings of an LLC budget:

```text
residency demand: cache capacity whose retention benefits the same expert
stream pressure: one-pass traffic that can transiently allocate and evict
```

## Method

The experiment ran on CPU 48 of NUMA node 0 on `AmazonC5192Cores`. The node has
96 MiB shared L3 and each core has 2 MiB private L2. The polluter calls the
production `moe_sve_w13_silu_poly5_packc_m12_rows_opt` fused assembly kernel
for W13 and feeds its packed-C output directly to the production
`moe_sve_w2_packed_bf16_m12` assembly kernel. It uses the TP4 expert weight
dimensions:

```text
W13 range 0: K=4096, N=512, 4 MiB
W13 range 1: K=4096, N=512, 4 MiB
W2:           K=512, N=4096, 4 MiB
total per distinct expert:              12 MiB
```

The victim is a 32 or 64 MiB buffer containing one randomized dependent chain
through every cache line. A trial performs:

1. Scan a 192 MiB eviction buffer and probe the distinct victim once for the
   cold reference.
2. Traverse the victim ten more times to promote it, then record a hot probe.
3. Execute the requested number of distinct M12 polluter experts.
4. Probe the victim immediately after the polluter has finished.

Every trial has distinct victim and polluter addresses. Polluter execution is
sequential on the same core, so the post probe measures persistent cache
eviction after bandwidth contention has ended. Eleven trials were used for the
32 MiB sweep and nine for the 64 MiB cross-check.

## One-pass M12 stream

The paired loss is:

```text
max(0, (post - hot) / (cold - hot))
```

It uses the cold, hot, and post medians from the same process, avoiding
cross-process frequency differences. It is the fraction of the cold-versus-hot
latency gap exposed after the M12 stream, not an exact fraction of LLC lines.

| Victim | M12 experts | One-pass B | Hot probe | Post probe | Paired loss |
|---:|---:|---:|---:|---:|---:|
| 32 MiB | 0 | 0 MiB | 4.522 ms | 4.193 ms | 0.00% |
| 32 MiB | 1 | 12 MiB | 3.238 ms | 3.025 ms | 0.00% |
| 32 MiB | 2 | 24 MiB | 3.122 ms | 3.232 ms | 0.25% |
| 32 MiB | 3 | 36 MiB | 3.270 ms | 3.607 ms | 0.78% |
| 32 MiB | 4 | 48 MiB | 3.370 ms | 4.134 ms | 1.77% |
| 32 MiB | 6 | 72 MiB | 3.075 ms | 5.011 ms | 4.46% |
| 32 MiB | 8 | 96 MiB | 3.403 ms | 6.162 ms | 6.38% |
| 32 MiB | 12 | 144 MiB | 4.466 ms | 8.649 ms | 9.95% |
| 32 MiB | 16 | 192 MiB | 3.330 ms | 9.649 ms | 14.67% |
| 64 MiB | 0 | 0 MiB | 18.997 ms | 18.859 ms | 0.00% |
| 64 MiB | 1 | 12 MiB | 20.053 ms | 20.331 ms | 0.37% |
| 64 MiB | 2 | 24 MiB | 19.901 ms | 21.306 ms | 1.85% |
| 64 MiB | 4 | 48 MiB | 19.417 ms | 21.278 ms | 2.44% |
| 64 MiB | 8 | 96 MiB | 19.949 ms | 25.292 ms | 7.04% |
| 64 MiB | 12 | 144 MiB | 19.849 ms | 26.420 ms | 8.69% |

There is no byte-for-byte capacity displacement. One 12 MiB M12 expert has no
measurable effect on the robust 64 MiB resident victim. Even a 96 MiB one-pass
stream exposes only 7.0% of its cold-hot latency gap. The effect nevertheless
becomes monotonic and measurable as cumulative one-pass traffic grows, so the
stream is not a literal cache-bypass operation.

## Repeated-weight control

The 32 MiB victim was also tested with ten M12-equivalent panels over each
distinct polluter weight. This models M=120 packed-B reuse and promotes the
polluter lines repeatedly.

| Distinct weights | Unique B | B scans | Post probe | Raw cold-hot gap exposed |
|---:|---:|---:|---:|---:|
| 1 | 12 MiB | 10 | 7.123 ms | 8.54% |
| 2 | 24 MiB | 10 each | 11.497 ms | 18.97% |
| 3 | 36 MiB | 10 each | 15.118 ms | 27.34% |
| 4 | 48 MiB | 10 each | 17.933 ms | 33.56% |
| 6 | 72 MiB | 10 each | 28.110 ms | 57.26% |
| 8 | 96 MiB | 10 each | 37.704 ms | 80.10% |

This control includes both repeated promotion and ten times more packed-B
accesses, as a real ten-panel GEMM does. It is not a pure cache-capacity-only
comparison. It demonstrates why long-route weights have materially greater
residency value and replacement pressure than one-pass M12 weights.

## Scheduling interpretation

M12 packed B should be excluded from a `reusable_B_capacity` sum because no
later M12 panel can benefit from retaining it. It must not be excluded from the
whole contention model:

```text
M <= 12:
  reusable_B_capacity = 0
  cold_B_bytes        = full packed weight bytes
  stream/pollution    = nonzero, calibrated separately

M > 12:
  reusable_B_capacity = active W13/W2 stage bytes
  repeated_B_scans    = ceil(M / 12) - 1
```

The existing all-M12 concurrency sweep is consistent with this split. Active
M12 stage bytes can exceed the 96 MiB LLC without a capacity cliff, while
aggregate throughput converges to the approximately 360 GB/s cold-B read
ceiling. Thus short experts are free only in the reusable-capacity constraint;
they still consume the DRAM-bandwidth and packed-weight-turnover budgets.
