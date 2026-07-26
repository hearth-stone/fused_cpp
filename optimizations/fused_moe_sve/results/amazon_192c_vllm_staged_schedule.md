# vLLM-style staged MoE scheduling on Amazon C5 192 cores

Date: 2026-07-17

## Method

The benchmark used NUMA node 0 and CPUs `0-95` on `AmazonC5192Cores`. Both
variants used the same packed BF16 weights, SVE M12/M8/M4/M2/M1 assembly GEMM
bodies, fused W13 SiLU/packC epilogue, FP32 direct route store, and SVE U1
weighted merge. The fixed-team async control used two split-W13 ranges and had
ready-token merge disabled. The candidate used a global W13 N-range queue, a
full W13/W2 stage barrier, and a global W2 N-range queue.

All runs used `tokens=2048`, `H=4096`, `F=512`, `E=256`, `top_k=6`, 96 worker
threads, three warmups, and interleaved A/B timing order. Output was checked
bitwise before timing. Reported throughput counts W13 plus W2 GEMM FLOPs and
includes gather, fused epilogue, direct route store, and merge in wall time.

Representative command:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 256 --top-k 6 --threads 96 \
  --team-threads 16 --distribution hot-topk --route-dtype fp32 \
  --warmup 3 --runs 11 --stage-timing
```

## Six long experts

`hot-topk` routes every token to the same six experts, so each active expert
has `M=2048`. The fixed control is `6 x 16T`.

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-team async | 8.716 | 8.652 | 8.817 | 17.739 | baseline |
| vLLM staged | 10.027 | 9.836 | 10.510 | 15.421 | -13.07% |

The staged timing sample used 1 MiB as the available private-L2 budget and
selected W13/W2 task widths `64/256`, or 16 tasks per expert per stage. Its
wall-time decomposition was W13 `6.630 ms`, W2 `2.417 ms`, merge `0.697 ms`,
and E2E `10.050 ms`. Fixed teams share one gather-packed A per expert, whereas
the vLLM schedule rescans and packs A once per N-range; the global stage
barrier also prevents an expert from entering W2 early. Both costs are exposed
when six long experts already fill all 96 workers evenly.

## 256 short experts

`round-robin` gives every expert exactly `M=48`. Fixed team width was swept;
each point was measured in a separate process. Candidate medians varied from
`17.386` to `17.732 ms`; occasional roughly `32 ms` outliers affected P90 but
not the median.

| Threads/expert | Fixed median ms | Fixed TFLOP/s | Staged median ms | Staged vs fixed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.428 | 4.364 | 17.526 | +102.14% |
| 2 | 26.396 | 5.858 | 17.732 | +48.86% |
| 4 | 18.765 | 8.240 | 17.518 | +7.12% |
| 6 | 17.572 | 8.799 | 17.386 | +1.07% |
| 8 | **17.245** | **8.966** | 17.388 | **-0.82%** |

The dynamic staged queue is substantially better than an under-parallelized
fixed schedule, but it does not beat the best fixed width. Its practical value
is therefore as a schedule baseline and as evidence that N-range work stealing
can recover poor static choices. It is not a replacement for the existing
planner: with a suitable team width it is neutral for many short experts and
materially worse for balanced long experts.

## Production planner across paper workloads

Date: 2026-07-22

The follow-up comparison replaced the hand-written fixed-team control with the
actual `PlannedMoE` interval-DAG generated from
`contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260720.json`.
All workloads used 2048 tokens, TopK=6, 256 experts, H4096/F512, and exactly
12288 routes. The benchmark remained on NUMA0 CPUs `0-95` and used split-W13,
Xbyak exact-M kernels, BF16 direct route store, SVE U1 merge, and the production
ready-token merge heuristic. Each histogram was materialized into deterministic
TopK rows with no duplicate expert within a token and was checked exactly after
materialization.

Each case used five warmups and 21 interleaved-order samples. Both variants
shared input, packed weights, route IDs, route weights, and output buffers.
Their outputs were bitwise equal before timing. The medians below measure the
native operator only; production planning was performed before timing.

Representative command, repeated for every workload preset:

```bash
P=cpu_moe_schedule_optimization/cost_model/profiles/\
contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_\
splitw13_schema_v2_xbyak_exactm_20260720.json
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-tiered-hotspot --production-profile "$P" \
  --route-dtype bf16 --warmup 5 --runs 21
```

| Workload | Route histogram | Production shape | Production ms | vLLM staged ms | Production TFLOP/s | vLLM TFLOP/s | Production vs vLLM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | `256x48` | `6x16T` | 16.325 | 16.559 | 9.471 | 9.337 | +1.43% |
| Active set 8 | `8x1536` | `6x16T` | 11.015 | 13.574 | 14.037 | 11.391 | +23.23% |
| Active set 16 | `16x768` | `6x16T` | 8.946 | 11.085 | 17.284 | 13.949 | +23.91% |
| Active set 32 | `32x384` | `12x8T` | 8.663 | 10.966 | 17.848 | 14.100 | +26.58% |
| Active set 64 | `64x192` | `6x16T` | 10.064 | 10.854 | 15.364 | 14.246 | +7.85% |
| Active set 128 | `128x96` | `6x16T` | 12.299 | 12.831 | 12.572 | 12.050 | +4.33% |
| Tiered hotspot | `4x768 + 12x384 + 48x96` | `12x8T` | 8.743 | 11.328 | 17.684 | 13.649 | +29.56% |
| Long/short bimodal | `5x2040 + 174x12` | `6x16T` | 12.354 | **11.629** | 12.516 | **13.296** | **-5.86%** |

The production schedule is clearly better for concentrated, uniform-active-set,
and tiered workloads. Uniform `256x48` is effectively neutral at this noise
level. The long/short mixture is the one reversal: the global staged task queue
is 6.23% faster than production. The production plan assigns one long expert
plus 17-18 short experts to five lanes and 86 short experts to the sixth lane.
The global queue can redistribute those short N-range tasks after the long work
finishes, whereas the current production plan is a non-malleable static
partition.

### Why the bimodal workload reverses

A current-build six-long-expert control (`6xM2048`) used the same total 12288
routes and selected the same production `6x16T` shape:

| Workload | Production ms | vLLM staged ms | Production vs vLLM |
| --- | ---: | ---: | ---: |
| Six long experts | 8.281 | 9.477 | +14.44% |
| Five long + 174 M12 experts | 12.354 | 11.629 | -5.86% |
| Increment after replacing one long expert | +4.073 | +2.152 | -- |

Production therefore retains its packed-A sharing advantage on long experts.
The reversal occurs because the static shape also gives every M12 expert 16
threads. M12 contains only one row tile, so that width exposes per-expert team
and N-split overhead while the 174 experts remain bound to six fixed lane
queues. The vLLM task pool instead exposes every `(expert, N-range)` as
independent work and lets all 96 workers drain the short-expert tail.

An equal-width static ablation on the same bimodal input gave:

| Static assignment | Median ms |
| --- | ---: |
| Production LPT `6x16T` | **12.354** |
| Round-robin `6x16T` | 14.251 |
| Round-robin `12x8T` | 16.363 |
| Round-robin `24x4T` | 26.206 |
| vLLM staged | **11.629** |

LPT is materially better than round-robin, and reducing the global lane width
hurts the five long experts more than it helps the M12 experts. The missing
schedule is mixed-width: wide fixed teams for long experts plus narrow or
globally stealable lanes for short experts, with cores becoming reusable when a
long expert finishes. The current whole-call static shape cannot express that
transition.

Disabling production ready-token merge changed the bimodal production median
from 12.354 ms to 12.585 ms, while vLLM staged remained at 11.610 ms. Ready-token
merge therefore improves production by 1.87% in this case and does not explain
the reversal.

### Static 16T to four 4T lane transition

A benchmark-only follow-up used the existing async dependency bridge to express
mixed widths without changing the production planner. The five `M=2040`
experts occupy five 16-thread intervals. Completion of each long task releases
four disjoint 4-thread short-task chains on the same cores. The sixth
16-thread interval starts as four 4-thread chains immediately. There is no
work stealing and no native thread creation during the transition.

The profile reports isolated times of `6.121973 ms` for `M=2040/16T` and
`0.179066 ms` for `M=12/4T`. Static earliest-finish assignment put two short
experts on ten delayed lanes, one on the other ten delayed lanes, and 36 on
each of the four immediately available lanes. All 179 active experts appeared
exactly once in the DAG. Every first delayed short task depends on its 16T
parent, while subsequent tasks depend only on the preceding task in the same
4T lane.

The run used five warmups and 21 interleaved samples. Candidate outputs were
bitwise equal to the production async output before timing. The host was
Neoverse-V3 AArch64 with 192 CPUs; the run used NUMA0 CPUs `0-95`, Python
3.12.13, PyTorch 2.13.0+cpu, and GCC 15.2.0.

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-long-short-bimodal \
  --production-profile \
  cpu_moe_schedule_optimization/cost_model/profiles/\
contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_\
splitw13_schema_v2_xbyak_exactm_20260720.json \
  --static-16-to-4 --route-dtype bf16 --warmup 5 --runs 21
```

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Speedup vs production |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production `6x16T` | 12.390 | 12.339 | 12.546 | 12.479 | baseline |
| Static `16T -> 4x4T` | **11.017** | 10.938 | 11.143 | **14.035** | **+12.47%** |
| vLLM staged | 11.631 | 11.543 | 11.709 | 13.293 | +6.52% |

The static transition is 5.58% faster than the staged queue. It retains the
long expert's 16T packed-A sharing and gives every short expert the narrower
4T kernel without the staged implementation's global W13/W2 barrier or
per-N-range packed-A rebuild.

The same command with `--no-production-ready-token-merge` isolates the expert
DAG from ready-token merge overlap:

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Speedup vs production |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production `6x16T` | 12.560 | 12.464 | 12.690 | 12.310 | baseline |
| Static `16T -> 4x4T` | **11.089** | 11.020 | 11.226 | **13.944** | **+13.27%** |
| vLLM staged | 11.619 | 11.576 | 11.690 | 13.308 | +8.10% |

The transition therefore remains beneficial after merge overlap is removed;
the measured interval/scratch selection and team-barrier overhead do not erase
the M12 width and tail-balance gain. This experiment validates the combined
kernel-width plus static team-transition mechanism. It does not yet make that
transition a production planner action or separately identify each component's
isolated contribution.

### Dynamic short-expert pool after long-team completion

The next experiment replaced the static 4T chains with one native global queue
of whole short experts. The production plan still launches the five
`M=2040` experts as five 16T tasks. The 96 resident threads are also viewed as
24 fixed 4T groups. A group blocked by a long task joins the short queue as soon
as that long task completes; the four groups on the otherwise unused sixth
16T interval join immediately. No worker is created, rebound, or migrated.

The command above added `--dynamic-short-pool` while retaining
`--static-16-to-4`, so all four variants were interleaved in one process.
There were five warmups and 21 timed samples. Before timing, every candidate
was bitwise equal to production. Stage diagnostics reported all 174 short
experts exactly once: the 20 delayed groups claimed two or three tasks each,
and the four immediately available groups claimed 31 each.

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Speedup vs production |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production `6x16T` | 12.440 | 12.389 | 15.102 | 12.429 | baseline |
| Static `16T -> 4x4T` | 11.031 | 10.968 | 11.071 | 14.017 | +12.77% |
| Dynamic `16T -> 4T pool` | **10.697** | **10.650** | **10.872** | **14.455** | **+16.29%** |
| vLLM staged | 11.654 | 11.558 | 11.793 | 13.267 | +6.74% |

The dynamic executor was 3.12% faster than static assignment and 8.95% faster
than vLLM staged. It keeps the long expert's shared packed-A and has no global
W13/W2 stage barrier, while the queue removes dependence on isolated-time
predictions and preassigned short-lane lengths.

### Plan V2 native rerun

Date: 2026-07-26

The dynamic variant was migrated from the legacy
`FUSED_CPP_MOE_ASYNC_SHORT_POOL_*` switches to an explicit Plan V2
`tail_pool` bridge. The production control now enters the same native executor
through Plan V2 `strict`. On NUMA0 CPUs `0-95`, with split-W13, BF16 route
storage, five warmups, and 21 interleaved samples, the result was:

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Speedup vs strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plan V2 strict `6x16T` | 12.513 | 12.446 | 12.652 | 12.357 | baseline |
| Plan V2 `4T tail_pool` | **10.836** | **10.763** | **10.909** | **14.269** | **+15.47%** |
| vLLM staged | 11.733 | 11.356 | 11.851 | 13.178 | +6.64% |

Before timing, strict, tail-pool, and vLLM outputs were bitwise equal. The
runtime extension SHA256 was
`e652d9aad3025a4836d0406110bcbdf3fdc1356aaba2bf789595359a58a92559`,
which does not match the profile hash below. This rerun validates the native
Plan V2 action and direct ABI overhead, not cost-model calibration accuracy.

With `--no-production-ready-token-merge`, the result remained:

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Speedup vs production |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production `6x16T` | 12.540 | 12.511 | 14.917 | 12.330 | baseline |
| Static `16T -> 4x4T` | 11.136 | 11.059 | 11.210 | 13.885 | +12.61% |
| Dynamic `16T -> 4T pool` | **10.803** | **10.757** | **10.966** | **14.312** | **+16.07%** |
| vLLM staged | 11.604 | 11.529 | 11.699 | 13.325 | +8.07% |

The transition and queue therefore account for the result independently of
ready-token merge. This implementation is deliberately default-off and
benchmark-only. It validates an executor action but does not add that action to
the production planner decision space or its cost model.

Warm cached `PlannedMoE.plan_spec_for` latency was 0.092-2.120 ms depending on
active-expert count; cold shape search was 2.8-52.2 ms. These Python planning
costs are reported separately and are not included in the operator table. If
the measured warm planning latency is naively added to every call, production
remains faster for active-set 8/16/32/64 and tiered-hotspot, but loses uniform,
active-set 128, and bimodal. This sum is not a complete Python E2E number
because route-histogram construction and schedule-tensor materialization were
not timed symmetrically; it identifies bridge caching as a separate requirement.

The static-transition run used runtime `_moe_C` SHA256
`c5a8e295eb89fe0e244a7bd4bff68fa90ebbce7619528555ce699e71f1598584`;
the dynamic-pool run used
`623c43bd0808f6897bf9fb4cffd7b2af879273fe09211fae1fe45ce80f94b9ce`.
The calibration profile records
`6d4431670cfd2f6edd714f324f55ecc4ddd059caee1954b449e404c01cf122c2`.
Neither runtime identity matches the profile. The tables are therefore direct
operator comparisons using the schedule selected by the latest available
profile, not a validation of current cost-model accuracy. Refresh the profile
before using these numbers as production planner-regret evidence.
