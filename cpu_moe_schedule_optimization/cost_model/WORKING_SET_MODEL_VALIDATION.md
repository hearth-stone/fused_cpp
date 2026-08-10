# Historical Exact-Range Owner-Cache Working-Set Model

> Retired by the v0.89 full-N migration. The measurements and commands below
> describe the deleted `R13=2,R2=1` geometry and must not be used as current
> calibration. The active `working_set_model.py` consumes one complete stage
> (`W13=4HF`, `W2=2HF`) and requires a new matching full-stage scan.

## Scope

This historical model covers the SVE exact `R13=2,R2=1` path with N-partitioned expert
teams. It is a shadow candidate filter and does not replace the active measured
contention model. The validated regime is long M12 bulk (`physical_panels >=
16`, currently routes >= 192).

## Formula

For BF16 `H` and rank-local `F`, each W13 range and W2 contain the same number
of packed weight bytes:

```text
S = 2 * H * F bytes
stages = [W13 range 0, W13 range 1, W2]
```

N-split assigns disjoint packed-B columns to each worker. The first reusable
capacity is therefore the aggregate owner-private cache, not LLC alone. For
`C` cores, private-cache size `L2`, `A` ways, and `r` ways reserved for packed-A,
stores, prefetch, and replacement headroom:

```text
C_owner = C * L2 * (A - r) / A
n_max   = ceil(C_owner / S) - 1
```

The strict inequality excludes the point that consumes the complete owner
budget. A GEMM-free scan fits resident bandwidth versus independent streams:

```text
B_owner(n) = B_max * (1 - exp(-n / n_sat))
n_min      = ceil(-n_sat * log(1 - u))
```

where `u` is the required bandwidth utilization, currently 95%. Candidate
working sets are `n in [n_min, n_max]`. Inside the band, compute granularity is
selected from streaming `T_iso`: keep shapes within 5% of the best isolated
makespan and choose the smallest `n`. This deliberately retains cache headroom
when two shapes have nearly equal compute throughput.

## Independent Scan Calibration

Host: `AmazonC5192Cores`, Neoverse-V3 NUMA0 CPUs `0-95`.

Hardware metrics:

| Metric | Value |
| --- | ---: |
| Cores | 96 |
| L2 per core | 2 MiB, 8-way |
| Shared L3 | 96 MiB |
| EP2 split stage `S` | 16 MiB |
| Reserved L2 ways | 2 |
| Owner-cache budget | 144 MiB |

The scan benchmark uses the same N-split ownership: each team member reads a
disjoint slice of one 16 MiB stream for 170 passes. It performs no GEMM or
activation instructions.

| Streams | Working set | Scan GB/s |
| ---: | ---: | ---: |
| 1 | 16 MiB | 6458.5 |
| 3 | 48 MiB | 8638.8 |
| 4 | 64 MiB | 8842.7 |
| 8 | 128 MiB | 9401.0 |
| 9 | 144 MiB | 9193.6 |
| 10 | 160 MiB | 9118.7 |
| 11 | 176 MiB | 8012.6 |
| 12 | 192 MiB | 4386.6 |
| 16 | 256 MiB | 1484.2 |
| 24 | 384 MiB | 846.9 |
| 32 | 512 MiB | 737.5 |

The resident points fit `B_max=8.991 TB/s`, `n_sat=0.85`; median absolute fit
error is 1.1%, maximum 7.4%. Thus `n_min=3`. The strict 144 MiB owner budget
gives `n_max=8`, so the predicted preferable band is **3--8 experts, or
48--128 MiB**.

Root PMU measurements used `l2d_cache_refill`, `ll_cache_rd`, and
`ll_cache_miss_rd`. Normalizing L2 refills by the cache lines scanned gives
approximately 0.2% at 8 streams, 1.7% at 10, 10.6% at 12, 39.4% at 16, 61.3%
at 24, and 71.9% at 32. LLC misses become material after the private-cache
refill cliff. This independently confirms that the first knee is private-L2
residency, not the nominal 96 MiB LLC capacity.

## Fused-Kernel Holdout

The held-out async fused-expert sweep uses 32 consecutive distinct EP2 weights,
all 96 cores, exact `R13=2,R2=1`, 3 warmups and 9 measured runs. Routes 1020 was not
in the contention grid used by the source profile.

| Routes | 48 MiB | 64 MiB | 128 MiB | 160 MiB | 192 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1020 aggregate TF/s | **23.723** | 23.645 | 23.004 | 19.465 | 14.410 |
| 2040 aggregate TF/s | 23.968 | **24.882** | 24.637 | 20.434 | 15.511 |

The model band contains every near-best point. The `T_iso` 5% headroom rule
selects 4 experts / **64 MiB** for both holdouts. Measured regret is 0.33% at
route 1020 and 0.00% at route 2040. The 128 MiB endpoint remains within 3.0%
and 1.0%, while the first point above the owner budget (160 MiB) loses about
18% on both routes.

On existing route 192/768/2040 profile points, recommended-shape regret is
0.00%, 0.79%, and 1.57%. Route 48 has 17.27% regret, proving that the cache
metric must not replace measured short-route overhead; those points remain
outside the model's applicability gate.

## Cross-Machine Holdout

The same split-only experiment was repeated on `AmazonECS8Cores`, Neoverse-V1:

| Metric | Value |
| --- | ---: |
| Cores | 8 |
| L2 per core | 1 MiB, 8-way |
| Shared L3 | 32 MiB |
| TP4 split stage `S` | 4 MiB |
| 6/8-way owner budget | 6 MiB |

The strict owner budget permits one stream, so the model degenerates to a
1-expert / 4 MiB band without fitting a multi-stream saturation curve. The
independent scan falls monotonically from 784 GB/s at 4 MiB to 584 GB/s at
8 MiB, 266 GB/s at 16 MiB, and 203 GB/s at 32 MiB. This is consistent with the
smaller aggregate private L2 even though the shared L3 is 32 MiB.

The current split fused kernel also selects one 8-thread expert:

| Routes | Predicted | Measured best | Regret | 16 MiB versus best |
| ---: | ---: | ---: | ---: | ---: |
| 1020 | 4 MiB | 4 MiB | 0.00% | -6.5% throughput |
| 2040 | 4 MiB | 4 MiB | 0.00% | -5.0% throughput |

The V1 result is a machine/shape holdout: it changes core count, private/shared
cache sizes, and split stage size. Together with V3 it supports using N-split
owner-private capacity as the first reusable-weight budget rather than a fixed
fraction of LLC.

## Reproduction

```bash
g++ -std=c++17 -O2 -pthread \
  cpu_moe_schedule_optimization/benchmarks/bench_weight_scan.cpp \
  -o /tmp/bench_weight_scan

taskset -c 0-95 /tmp/bench_weight_scan \
  --cpu-ids 0-95 \
  --groups 1,2,3,4,5,6,7,8,9,10,11,12,16,24,32 \
  --stream-mib 16 --passes 170 --warmup 2 --runs 9 \
  --output-csv /tmp/weight_scan.csv

.venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/working_set_model.py \
  cpu_moe_schedule_optimization/cost_model/profiles/\
contention_async_amazon_c5_192c_numa0_ep2_sve_F2048_splitw13_v2_20260714.json \
  --scan-csv cpu_moe_schedule_optimization/cost_model/profiles/\
weight_scan_amazon_c5_192c_numa0_ep2_split_20260715.csv \
  --private-cache-bytes-per-core 2097152 \
  --cache-ways 8 --reserved-ways 2 \
  --target-bandwidth-utilization 0.95 --iso-headroom 0.05
```

Raw scan, fused holdout, and serialized validation files for both machines are
stored under `cost_model/profiles/` with the `weight_scan_*_20260715` and
`working_set_*_20260715` names.
