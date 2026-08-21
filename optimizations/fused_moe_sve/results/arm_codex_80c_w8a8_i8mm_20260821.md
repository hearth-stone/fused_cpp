# Dynamic W8A8 i8mm MoE on Arm-codex 80C

## Scope

This Lab experiment measures a complete dynamic-W8A8 routed-MoE pipeline while
preserving the current production BF16 path. The first version used the SVE
i8mm no-pack hybrid kernel. The updated Lab default packs A once per stage and
uses the stable packed-A M12/M8 `SMMLA` kernels inside each existing expert
team. It does not add a pybind API, production dispatch branch, planner
dimension or environment variable.

## Pipeline

For every active expert:

1. gather each routed BF16 hidden row and dynamically quantize it to signed
   INT8 with one symmetric scale per row;
2. pack A cooperatively and run W13 as INT8 x INT8 to INT32 using M12 blocks,
   an M8 tail and per-thread N windows;
3. apply per-output-channel weight scale and per-row activation scale, clamped
   SwiGLU limit 10, and round the intermediate to BF16;
4. dynamically quantize the BF16 intermediate to signed INT8 per row;
5. pack the second A and run W2 with the same kernel policy, then dequantize
   into the FP32 route buffer;
6. use the existing ordered FP32 weighted top-k merge and BF16 final store
   semantics.

The benchmark consumes the same fixed-width lane membership and task order as
the materialized BF16 Plan V2. Early merge is disabled in both paths so merge
publication does not affect the comparison.

## Configuration

- Source base: Git `42e12d8` plus the Lab files named in the manifest.
- Host: `Arm-codex-internal`, NUMA3 CPUs `240-319`, local memory binding.
- Pages: ordinary pages.
- Build: SVE256, `-O2 -std=c++17`, Armv8.6-A BF16+i8mm.
- Shape: V4 Flash TP4, E256, T2048, TopK6, H4096, F512.
- Routing: captured `dsv4-real-2048-seq70`, 223 active experts, M1-M918.
- Plan: local quick calibration, strict `10 x 8T`, LPT; the BF16 comparator
  retains W13/W2 windows 4/32.
- Activation: clamped SwiGLU limit 10.
- Timing: seven warmups and 31 samples in each of three independent processes.

The updated W8A8 default is `(t,w13_window,w2_window,R13,R2)=(8,1,4,8,8)`.
With SVE256 this is 64 KiB of W13 and 32 KiB of W2 packed B per core. One team
therefore keeps 512/256 KiB active and advances through each stage in eight
windows. The former full owner stripes were 512/256 KiB per core.

## Correctness boundary

A deterministic E4/T8/TopK2/H64/F32 check filled both packed weight matrices
with one, used constant per-channel scales, and compared the full vector path
against a scalar reconstruction. It passed repeated execution with reported
maximum absolute error zero.

The packed update additionally checked M1-M4 narrow kernels, M13 M8 tails and
M25 M12-plus-tail execution at H4096/F512. Full owner stripes and one-tile
windows both reported maximum absolute error zero and matched hybrid checksums.
An M918/H4096/F512 random-weight run then compared every final BF16 element
between hybrid and packed M12 and reported `kernel_max_abs=0`.

The formal performance run deliberately initializes packed INT8 weights
directly because it measures the execution mechanism. Its checksum is stable,
but those weights are independent of the BF16 baseline weights. Therefore the
formal run does not measure quantization error or model quality.

## Results

### Production Plan V2 promotion

The explicit public `_moe_C` path was validated after the Lab decision using
the same captured routing, fixed 10x8T plan, `(1,4)` W13/W2 windows, CPUs
240-319, and NUMA3 memory binding. Fifteen measured invocations after five
warmups produced a 12.659 ms median, 12.532 ms minimum, and 12.965 ms maximum.
The current Lab executable measured 12.229 ms with two warmups and five runs;
the production bridge is therefore 3.52 percent slower in this short check and
remains 2.86x faster than the simultaneously remeasured 36.201 ms BF16 path.

The production path adds checkpoint per-channel W8 packing, a public prepared
weight type, strict homogeneous Plan V2 validation, per-calling-thread scratch
reuse, capability reporting, and explicit unsupported-mode errors. A negative
control which pinned workers to NUMA3 but first-touched the 1.5 GiB packed
weights outside a NUMA3 memory policy measured 139.98 ms. This is not an
operator result; it demonstrates that callers must establish the documented
NUMA memory policy before packing weights.

The two logical GEMMs contain 154.62 GFLOP per invocation. The following values
use the same captured routing, strict 10x8T schedule, placement and random W8
weights. Positive speedup means the packed-M12 candidate is faster.

| Configuration | Median ms | Relative to matching hybrid |
| --- | ---: | ---: |
| Hybrid, full owner stripes `(8,0,0)` | 16.300 | baseline |
| Hybrid, tuned windows `(8,2,4)` | 15.317 | baseline |
| Packed M12, full owner stripes `(8,0,0)` | 13.207 | 1.234x |
| Packed M12, tuned windows `(8,1,4)` | 12.108 | 1.265x vs tuned hybrid |
| Packed M12 tuned, independent repeat | 12.300 | 1.245x vs tuned hybrid |

At the same `(1,4)` windows, hybrid measured 15.403 ms and packed M12 measured
12.108 ms, a 1.272x speedup and 21.39 percent latency reduction. Window tuning
alone reduced hybrid by 6.03 percent and packed M12 by 8.32 percent. Relative
to the original full-stripe hybrid, the best packed result reduces latency by
25.72 percent. The BF16 comparator remained 36.29-36.44 ms in these runs.

Equivalent useful-GEMM throughput for the complete tuned W8A8 pipeline is
12.57-12.77 TOPS. Packed INT8 W13 plus W2 weights consume 1,610,612,736 bytes,
exactly half the BF16 matrix payload before scales; per-channel scales add about
5 MiB.

### Single-expert scaling

These results include both GEMMs, both dynamic quantizers, clamped SwiGLU,
route store and final merge for one expert with full owner stripes.

| Routes | Threads | Hybrid ms | Packed M12 ms | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 918 | 1 | 62.797 | 39.175 | 37.62% |
| 918 | 4 | 15.929 | 10.351 | 35.02% |
| 918 | 8 | 8.229 | 5.812 | 29.37% |
| 918 | 16 | 4.307 | 2.957 | 31.34% |
| 918 | 32 | 2.507 | 1.757 | 29.90% |
| 918 | 64 | 1.491 | 1.103 | 26.03% |
| 2048 | 1 | 138.896 | 89.969 | 35.23% |
| 2048 | 4 | 35.300 | 24.289 | 31.19% |
| 2048 | 8 | 18.173 | 12.404 | 31.75% |
| 2048 | 16 | 9.608 | 6.780 | 29.44% |
| 2048 | 32 | 5.358 | 3.833 | 28.47% |
| 2048 | 64 | 3.121 | 2.317 | 25.74% |

## Interpretation

The packed-A M12 update is material even after tuning the hybrid comparator.
It preserves the fixed Plan V2 teams and therefore avoids nested OpenMP: team
members cooperatively pack A, synchronize once, and retain their N ownership.
Short M uses the current narrow/hybrid kernels instead of paying M12 packing.

The 1 MiB-per-core rule does not hold for this workload. For one 4T expert,
512 KiB and 1 MiB W13 windows were within 0.1 percent at M918. With ten 8T
experts active, 64 KiB W13 and 32-64 KiB W2 were best. The relevant quantity is
the aggregate active B across concurrent teams: the selected windows expose
about 5 MiB W13 or 2.5-5 MiB W2 across all 80 cores, while larger windows add
cache pressure without useful kernel work.

A same-routing, same-LPT width check also rejected using 4T merely to obtain a
1 MiB W13 stripe. At full stripes, `20x4T` measured 14.067 ms versus 13.554 ms
for `10x8T`, a 3.79 percent regression. At identical 64/32 KiB per-core windows,
4T measured 12.754 ms versus 12.495 ms for 8T, a 2.07 percent regression. The
extra expert concurrency and reduced per-expert compute width outweighed any
benefit from the larger owner stripe. A 16T follow-up measured 14.809 ms with
the same small windows, 18.52 percent slower than 8T. Under route-count LPT,
8T balanced its ten lanes at 1228-1230 routes each; 4T was bounded by the
indivisible M918 expert and ranged from 593 to 918, while 16T balanced route
counts but assigned 42-47 mostly short experts to each of only five wide teams.

A 200-run PMU comparison on the same 10x8T LPT schedule separates cache level
from nominal capacity. Moving from full stripes to 64/32 KiB windows reduced
median time by 4.44 percent and cycles by 4.22 percent while increasing
instructions by 3.47 percent. L2D refills rose 11.60 percent and LLC reads rose
13.65 percent, but LLC read misses fell 17.77 percent and bus accesses fell 6.16
percent; L1D refills changed by only -2.66 percent. The smaller window therefore
does not simply eliminate L1 refills. It accepts additional packed-A/cache
traffic to turn expensive lower-level B misses into cache hits.

The window preference reverses for a single M4096 expert. At 8T, full W13/W2
owner stripes measured 24.161 ms, while `(1,4)` measured 24.818 ms, a 2.72
percent regression. At 16T, full stripes measured 12.916 ms and `(1,4)` measured
13.371 ms, a 3.52 percent regression. W13-only and W2-only sweeps also selected
their full stripes (8T: 512/256 KiB per core; 16T: 256/128 KiB). With a 16 MiB
W13 A matrix, a small B window traverses the entire packed A between consecutive
B tiles; a full stripe instead reuses each M12 A block across its owner B tiles
before advancing through M.

The current experiment is not directly adoptable. It still needs:

- checkpoint-derived per-channel W8 packing shared with the BF16 comparator;
- formal-shape operator error against the matching BF16 weights;
- stage timing to identify quantization and epilogue headroom;
- short-route and held-out routing distributions;
- model-level quality validation;
- a production runtime design that reuses Plan V2 without duplicating the Lab
  scheduler.

## Pure GEMM throughput

A separate GEMM benchmark reports two rates:

- `hybrid`: prequantized A8 and W8 through the fixed no-pack `SMMLA` kernel;
- `dispatch`: the current stable i8mm dispatcher, including its packed-A M12
  path, A packing, thread grid and INT32 store;
- `BF16 I/O`: BF16 input, per-row dynamic A8, the same INT8 GEMM,
  per-channel dequantization and BF16 output.

Both rates count the useful matrix work as `2*M*K*N`. They do not include
SwiGLU, the second GEMM, route scheduling or weighted merge.

| Shape | Threads | Fixed hybrid | Current dispatch | Dispatch gain | BF16 input/output |
| --- | ---: | ---: | ---: | ---: | ---: |
| W13 M918 K4096 N1024 | 1 | 202.939 GOPS | 334.962 GOPS | 65.06% | 195.002 GFLOP/s |
| W2 M918 K512 N4096 | 1 | 199.281 GOPS | 309.717 GOPS | 55.42% | 184.725 GFLOP/s |
| W13 M2048 K4096 N1024 | 1 | 198.377 GOPS | 338.378 GOPS | 70.57% | 190.879 GFLOP/s |
| W2 M2048 K512 N4096 | 1 | 200.041 GOPS | 305.873 GOPS | 52.91% | 185.232 GFLOP/s |
| W13 M918 K4096 N1024 | 8 | 1.631 TOPS | 2.575 TOPS | 57.84% | 1.558 TFLOP/s |
| W2 M918 K512 N4096 | 8 | 1.609 TOPS | 2.395 TOPS | 48.78% | 1.479 TFLOP/s |
| W13 M2048 K4096 N1024 | 8 | 1.639 TOPS | 2.617 TOPS | 59.66% | 1.565 TFLOP/s |
| W2 M2048 K512 N4096 | 8 | 1.614 TOPS | 2.343 TOPS | 45.12% | 1.476 TFLOP/s |
| W13 M918 K4096 N1024 | 80 | 9.917 TOPS | 19.987 TOPS | 101.54% | 8.879 TFLOP/s |
| W2 M918 K512 N4096 | 80 | 10.596 TOPS | 11.891 TOPS | 12.22% | 8.802 TFLOP/s |
| W13 M2048 K4096 N1024 | 80 | 10.457 TOPS | 22.435 TOPS | 114.55% | 9.739 TFLOP/s |
| W2 M2048 K512 N4096 | 80 | 11.646 TOPS | 14.507 TOPS | 24.57% | 9.877 TFLOP/s |

Every fixed-hybrid and current-dispatch pair was compared element by element
outside the timed region for all twelve shape/thread configurations; all INT32
outputs matched exactly. The single-core current dispatcher reaches 82.5-91.3
percent of the measured 370.82 GOPS register-only SMMLA peak. The fixed hybrid
remains the BF16-I/O comparator in this experiment.

An independent repeat of the full 7-warmup/31-sample process kept every dispatch
median within 1.06 percent except M2048 W2 at 80 threads, which moved from 14.507
to 15.010 TOPS (3.47 percent). No best-of-run value is substituted into the
table.

The largest 80-thread gains are on W13 because the fixed N-split has only 64
SVE256 N panels and therefore cannot occupy all 80 workers. The dispatcher uses
packed-A M12 blocks and an M-by-N thread grid, which both improves the kernel
body and exposes enough parallel work. W2 has 256 N panels, so the fixed hybrid
already occupies all workers and the dispatch gain is smaller.

One isolated M918 8T expert sustains about 1.99 equivalent TOPS for the complete
pipeline. Ten such experts would have a no-contention sum near 19.9 TOPS. The
captured routed workload sustains 12.57-12.77 TOPS because concurrent experts
share memory/cache service, route lengths vary, and the timing also includes
both quantizers, SwiGLU, route stores and merge.

## Command

The deterministic small-shape correctness command was:

```bash
FUSED_CPP_MOE_SVE_VECTOR_BITS=256 OMP_NUM_THREADS=4 \
numactl --physcpubind=240-243 --membind=3 \
  optimizations/fused_moe_sve/benchmarks/run_w8a8_i8mm.sh \
  --routes /tmp/w8a8_routes.bin --schedule /tmp/w8a8_schedule.csv \
  --tokens 8 --top-k 2 --experts 4 --hidden 64 --intermediate 32 \
  --threads 4 --team-width 2 --cpu-start 240 --warmup 1 --runs 2 --check
```

The formal performance command was:

```bash
PYTHONPATH=.:src FUSED_CPP_MOE_SVE=1 \
FUSED_CPP_MOE_W2_DIRECT_ROUTE=1 FUSED_CPP_MOE_W2_BF16_ROUTE=0 \
FUSED_CPP_MOE_PREPACK_THREADS=80 FUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
OMP_NUM_THREADS=80 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_w8a8_plan_v2.py \
  --profile tmp/arm_codex_numa3_80c_quick_20260821.json \
  --workload cpu_moe_schedule_optimization/planners/workloads/deepseek_v4_flash_2048_seq70.json \
  --tokens 2048 --top-k 6 --experts 256 --hidden 4096 --intermediate 512 \
  --threads 80 --team-width 8 --cpu-start 240 --tp-degree 4 \
  --w13-window 4 --w2-window 32 --warmup 7 --runs 31 --swiglu-limit 10 \
  --w8a8-gemm-kernel packed_m12 --w8a8-w13-window 1 --w8a8-w2-window 4
```
