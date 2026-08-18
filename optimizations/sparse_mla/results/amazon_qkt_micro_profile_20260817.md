# Sparse MLA sampled QKT micro-profile

Date: 2026-08-17

## Scope

The existing `qkt_total` profile includes `qkt_8x2vl_bf16`, copying its output
to score scratch, and the fused row-max reduction. This diagnostic adds a
separate measurement around only `qkt_8x2vl_bf16`. The timed function includes
BFMMLA, scale, VL-specific UZP, and contiguous score store. It excludes K/Q
pack, score copy/max, mask, softmax, packed-P generation, and PV.

## Measurement design

Timing every tile was rejected because M5 executes 392,480 QKT calls in this
workload and per-call clock reads increased `qkt_total` from about 36 to 53 ms.
The adopted deep-profile path is isolated in a cold noinline helper and samples
one of every 16 tile calls. Every sampled call has the same fixed reduction and
tile shape. An empty timer with the same sample count records clock-read cost:

```text
corrected_time = qkt_micro - qkt_micro_overhead
micro_flops = sampled_calls * 2 * 8 * key_tile * Dqk
micro_gflops = micro_flops / corrected_time
```

Enable it with both `FUSED_CPP_SDPA_PROFILE=1` and
`FUSED_CPP_SDPA_PROFILE_DEEP=1`. The ordinary `qkt_total` slot remains
available but includes deep-profile perturbation when this mode is enabled.

## Workload and commands

Both machines used BF16 `q[2048,32,192]`, `Dv=128`, topk capacity 640,
777,792 valid query-key pairs, seed 20260817, one thread, no benchmark warmup,
and one measured operator call per process. Five independent processes were
used.

M5 Neoverse V3/SVL128, NUMA1 core 96:

```text
numactl --physcpubind=96 --membind=1 env OMP_NUM_THREADS=1 \
  OMP_PROC_BIND=close OMP_PLACES=cores PYTHONPATH=src \
  FUSED_CPP_SDPA_PROFILE=1 FUSED_CPP_SDPA_PROFILE_DEEP=1 \
  .venv/bin/python /tmp/bench_sparse_mla_scalable.py \
  --threads 1 --warmup 0 --iters 1 --seed 20260817
```

Amazon ECS 8C Neoverse V1/SVL256, core 0, used the same environment and script
under `taskset -c 0`.

## Results

| Machine | Tile | Samples | Micro ms | Empty ms | Corrected ms | GFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| M5/SVL128 | 8x8x192 | 28,224 | 2.854 median | 0.611 median | 2.250 median of paired differences | 308.3 |
| Amazon 8C/SVL256 | 8x16x192 | 17,128 | 3.663 median | 0.538 median | 3.292 median of paired differences | 255.7 |

M5 paired corrected times were 2.250, 2.272, 2.243, 2.242, and 2.251 ms.
Amazon 8C times were 3.330, 3.123, 3.319, 3.102, and 3.292 ms. Timed output
checksums matched on every run.

## Default-path check

With deep profiling disabled, five warmups and 21 samples were compared against
the pre-instrumentation PV-contiguous binary in forward/reverse order:

| Machine | Instrumented session medians | Baseline session medians | Median-of-medians change |
|---|---:|---:|---:|
| M5/SVL128 | 93.552 / 93.812 ms | 93.225 / 93.714 ms | +0.23% |
| Amazon 8C/SVL256 | 122.899 / 122.811 ms | 123.370 / 123.268 ms | -0.38% |

At deployment thread widths, M5 96T was 1.601/1.600 ms versus baseline
1.602/1.600 ms (-0.03% by median-of-medians), and Amazon 8T was 20.431/20.591
ms versus 20.613/20.559 ms (-0.36%). There is no consistent default-path
regression. M5 direct dense/shared-prefix/later-sparse checks passed; Amazon
native SVL256 and forced SVL128 each reported `26 passed`.
