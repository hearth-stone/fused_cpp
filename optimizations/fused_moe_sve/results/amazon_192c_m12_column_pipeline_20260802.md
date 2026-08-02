# AmazonC5192Cores M12 two-B column-pipeline experiment

Date: 2026-08-02

## Question

Can the 96-core Neoverse-V3 M12 compute efficiency be increased by keeping all
six packed-A row pairs resident and ping-ponging two packed-B column registers?

The experiment is a benchmark-only Xbyak probe. It does not change production
dispatch, packed layouts, stores, or fused epilogues. Both schedules execute the
same 10 loads and 24 BFMMLA instructions per K4 step.

## Method

- Host: `AmazonC5192Cores`, NUMA0 CPUs `0-95`.
- System: Linux `7.0.0-1006-aws`, GCC `15.2.0`, Python `3.12.13`, PyTorch
  `2.13.0+cpu`, SVE vector length 128 bits.
- Build: extension C++ at `-O2 -march=armv8.6-a+bf16+i8mm`; standalone
  correctness binary at `-O2 -march=armv8.6-a+sve+bf16+i8mm` and
  `-msve-vector-bits=128`.
- Workspace: uncommitted experiment based on `8e5e236`; baseline and candidate
  were generated and measured from the same binary and process environment.
- Allocation: 32 MiB HugeTLB through `/dev/hugepages-32M`.
- Geometry: cache-derived L1-hot `M=12`, `K=728`, `N=16`; A+B footprint is
  40,768 bytes per core.
- Baseline: production M12 load/compute schedule, probe mode 4.
- Candidate: six resident A pairs and B ping-pong in `z6/z7`, probe mode 11.
- Metric: useful `2*M*K*N` FLOPs divided by slowest-worker batch wall time.
- Correctness: a with-store version is compared bitwise with upstream i8mm.

Correctness passed at both `M12/K64/N64` and `M12/K4096/N1024` with
`max_abs_diff=0`.

## Scaling sweep

The first randomized sweep used five 131,072-kernel native batches per point:

| Cores | Baseline TFLOP/s | Column TFLOP/s | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.3478 | 0.3487 | +0.28% |
| 48 | 16.5107 | 16.3550 | -0.94% |
| 64 | 20.9300 | 21.0860 | +0.75% |
| 80 | 25.5132 | 25.6879 | +0.68% |
| 96 | 30.2159 | 30.1884 | -0.09% |

These roughly 0.1-second windows were too short to resolve a sub-percent
effect from slowest-worker tails. A second run therefore used 21 randomized
262,144-kernel batches at 1 and 96 cores:

| Variant | 1-core TFLOP/s | 96-core TFLOP/s | 96-core linear efficiency |
| --- | ---: | ---: | ---: |
| Baseline | 0.345633 | 30.420361 | 91.681% |
| Two-B column pipeline | 0.345638 | 30.617477 | 92.273% |

At 96 cores the candidate gains `0.648%` throughput and `0.592` percentage
points of linear efficiency. The single-core medians differ by only `0.002%`.
Across 21 samples the baseline range was `30.338--30.489 TFLOP/s`; the
candidate range was `30.257--30.687 TFLOP/s`, with one slow-worker outlier in
the candidate set.

## PMU attribution

Two long 96-core samples used the same useful FLOP count. Relative to the
baseline, the candidate's mean event counts changed as follows:

| V3 event | Candidate versus baseline |
| --- | ---: |
| `DISPATCH_STALL_IQ_LS` (`0x015e`) | -40.7% |
| `DISPATCH_STALL_IQ_VX` (`0x015f`) | +37.5% |
| `STALL_BACKEND_CPUBOUND` (`0x816a`) | -20.0% |
| `STALL_BACKEND_BUSY` (`0x816b`) | +3.3% |
| CPU cycles summed over active cores | -0.74% |
| Retired instructions | -0.13% |

The schedule slightly reduces average active-core cycles, consistent with the
long-window throughput gain, but it does not reduce vector queue fullness. The
higher `IQ_VX` count means the original hypothesis of relieving socket-wide
vector-dispatch backpressure is rejected; queue occupancy is shifted rather
than removed.

## Decision

Do not switch production M12 dispatch. The stable all-core gain is below the
existing 2% kernel-schedule adoption threshold used by the ILV comparison and
does not close the high-core issue-queue effect. Keep both no-store and
with-store modes as benchmark-only probes for future schedule comparisons.

## Commands

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_m12_column_pipeline_scaling.py \
  --widths 1,48,64,80,96 --cpu-ids 0-95 --batch-runs 262144 --repeats 5
```

The focused full-core confirmation used `--widths 1,96 --repeats 21`.
