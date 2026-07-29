# Planned global two-stage MoE on Amazon C5 192 cores

Date: 2026-07-29

## Question

This experiment separates two questions:

1. Does the production whole-expert pipeline have an inherent advantage over
   a global W13 barrier followed by global W2?
2. If global staging is retained, can independently planning W13 and W2 beat
   a carefully matched two-stage schedule?

## Implementation and method

The new benchmark-only native entrypoint executes one whole-expert Plan V2 for
W13, waits for every W13 task, and then executes an independent whole-expert
Plan V2 for W2. Each stage supports strict and tail-pool placement. W13 gathers
and packs A exactly once per expert; the global packed-C intermediate is kept
until W2. Both stages reuse the production SVE JIT exact-M kernels, split-W13
policy, per-task stage windows, direct BF16 route store, and weighted merge.

The four controlled variants are:

- `production_auto_stage_windows`: current whole-expert Plan V2 pipeline;
- `planned_staged_matched`: global barrier, with the same production plan used
  for W13 and W2;
- `planned_staged_independent`: global barrier, with W13 and W2 planned
  independently;
- `vllm_staged`: the older global `(expert, N-range)` queues, which repack A
  for every N range.

The host was `AmazonC5192Cores`; all runs were bound to NUMA0 CPUs `0-95`.
Every workload used 2048 tokens, H4096/F512, 256 local experts, TopK=6, BF16
route storage, 96 workers, three warmups, and 11 randomized-order samples.
All candidate outputs were bitwise equal before timing.

The primary structural comparison disabled ready-token merge for every
variant. This makes current versus matched differ in stage coupling rather
than merge overlap. A second full sweep retained the production default
ready-token merge, and five representative workloads were repeated with five
warmups and 21 samples.

Representative command:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-long-short-bimodal \
  --production-profile \
  cpu_moe_schedule_optimization/cost_model/profiles/\
contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_\
splitw13_schema_v2_xbyak_exactm_20260726.json \
  --route-dtype bf16 --no-production-ready-token-merge \
  --warmup 3 --runs 11 --stage-timing
```

The new extension hash differs from the 2026-07-26 calibration hash because
the native entrypoint changed. Kernel identity and packed-weight layout did
not change, but predicted absolute stage times are diagnostic rather than a
validated production calibration.

## Structural control: ready-token merge disabled

Positive gaps mean slower than the current whole-expert path.

| Workload | Current ms | Matched staged ms | Independent staged ms | vLLM staged ms | Matched gap | Independent gap | Independent vs matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Captured real routes | 14.825 | 15.601 | 15.742 | 15.641 | +5.23% | +6.19% | +0.91% |
| Active set 128 | 9.246 | 10.069 | 9.642 | 12.606 | +8.90% | +4.28% | -4.24% |
| Active set 16 | 9.354 | 9.789 | 9.814 | 10.594 | +4.64% | +4.91% | +0.26% |
| Active set 32 | 8.571 | 8.824 | 8.810 | 10.816 | +2.94% | +2.79% | -0.15% |
| Active set 64 | 8.981 | 9.167 | 9.263 | 10.853 | +2.08% | +3.15% | +1.05% |
| Active set 8 | 9.972 | 10.565 | 10.596 | 13.424 | +5.95% | +6.26% | +0.29% |
| Long/short bimodal | 10.600 | 11.210 | 10.842 | 11.678 | +5.75% | +2.29% | -3.28% |
| Tiered hotspot | 8.194 | 8.486 | 8.540 | 10.852 | +3.57% | +4.23% | +0.64% |
| Uniform | 16.579 | 18.531 | 17.138 | 17.025 | +11.77% | +3.37% | -7.52% |

Matched global staging is slower in all nine workloads, with median gap 5.23%
and range 2.08% to 11.77%. Independent staging is also slower in all nine,
with median gap 4.23% and range 2.29% to 6.26%. Disabling merge overlap
therefore does not remove the whole-expert advantage.

Independent planning makes a material difference exactly where the selected
stage plans differ:

| Workload | W13 plan | W2 plan | Gain over matched | Gap to current |
| --- | --- | --- | ---: | ---: |
| Active set 128 | `48x2T`, strict | `12x8T`, strict | +4.24% | +4.28% |
| Long/short bimodal | `6x16T`, pool 1T | `6x16T`, pool 2T | +3.28% | +2.29% |
| Uniform | `12x8T`, strict | `6x16T`, strict | +7.52% | +3.37% |

When both stages select the same shape and placement, independent versus
matched stays within approximately 1%. Stage-specific planning is therefore
useful, but its 3.28% to 7.52% recovery is not enough to pay for the global
barrier and lost expert-level pipeline/cache lifetime.

## Default runtime comparison

With production ready-token merge enabled, independent global staging remained
slower in all nine workloads: median gap 5.69%, range 2.87% to 10.92%. The
five-workload 21-sample confirmation was:

| Workload | Current ms | Independent staged ms | Gap |
| --- | ---: | ---: | ---: |
| Captured real routes | 14.736 | 15.760 | +6.95% |
| Active set 128 | 9.285 | 9.717 | +4.65% |
| Long/short bimodal | 10.409 | 11.138 | +7.00% |
| Tiered hotspot | 8.210 | 8.597 | +4.71% |
| Uniform | 16.602 | 17.203 | +3.62% |

A representative bimodal stage sample shows where independent planning helps:
matched staged used W13/W2/merge `7.431/3.082/0.281 ms`, while independent
used `7.045/3.128/0.294 ms`. Its E2E sample fell from `10.990` to
`10.648 ms`, but the current whole-expert median remained lower.

## Cost-model status

The first stage surrogate assigns expert setup plus two thirds of compute to
W13 and one third of compute to W2, then applies the existing stage geometry
and contention simulator. Across the nine no-ready-token measurements, its
independent-stage absolute-time MAPE was 12.8%, median absolute error 13.6%,
and maximum absolute error 34.1%. Cold Python stage search took 8.9 to
533.5 ms and is outside operator timing.

The model was good enough to identify the three useful stage-plan changes, but
its absolute accuracy is not sufficient for production. A production global
stage candidate would require directly calibrated W13/W2 isolated and
contention tables plus the native cold-search path.

## Decision

Keep `fused_moe_bf16_tiled_planned_staged` as an experimental control and do
not add global staging to the production planner. The experiment establishes
both intended conclusions:

- independent W13/W2 planning is better than a matched two-stage plan when
  their optimal shapes differ;
- the current whole-expert pipeline remains faster after controlling kernels,
  windows, A packing, direct store, merge implementation, and merge overlap.
