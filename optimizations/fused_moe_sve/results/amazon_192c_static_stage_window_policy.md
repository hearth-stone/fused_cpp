# Amazon 192-core static per-task stage-window policy

Date: 2026-07-26

Status: enabled by default for the exact measured profile identity

## Question

Test whether W13 and W2 packed-B windows should be selected by a deterministic
per-task rule after the planner has chosen expert placement and team width,
instead of adding both windows to the planner search space.

## Machine and method

- Host: `AmazonC5192Cores`, 192 Arm cores.
- Scope: NUMA0 CPUs `0-95` for the full workload matrix, plus independent
  NUMA1 CPUs `96-191` validation of the two highest-gain cases.
- Shape: TP4 `H=4096`, `F=512`, 256 local experts, 2048 tokens, TopK=6.
- Kernel: SVE BF16, split-W13, BF16 direct-route output.
- Baseline: current `production_auto` plan from the 2026-07-26 schema-v2
  profile.
- Candidate: the same plan with only `task_w13_window_bytes` and
  `task_w2_window_bytes` changed.
- Sampling: 5 warmups, 21 measured calls; table reports the median.

The benchmark rejects the candidate if any plan field other than the two stage
window arrays differs. Native correctness testing also forces small independent
W13/W2 windows and verifies bit-exact output against the legacy async bridge.

The runtime extension hash differs from the profile hash because the Plan V2
ABI gained the two optional arrays. Therefore these measurements validate the
post-plan runtime policy, not profile absolute-time accuracy or predicted
regret.

## Static policy

The policy key is `(route_count, actual_task_threads)`. Missing combinations
inherit the operator-wide profile policy.

| routes | threads | W13 MiB | W2 MiB |
| ---: | ---: | ---: | ---: |
| 49-95 | 8 | 1 | 1 |
| 96-143 | 1 | 0.125 | 0.125 |
| 96-143 | 2 | 0.125 | 0.25 |
| 96-143 | 4 | 0.25 | 0.5 |
| 96-143 | 8 | 1 | 0.5 |
| 144-287 | 1 | 0.125 | 0.125 |
| 144-287 | 2 | 0.25 | 0.25 |
| 144-287 | 4 | 0.5 | 0.5 |
| 144-287 | 8 | 1 | 0.5 |
| 288-575 | 1 | 1 | 0.5 |
| 288-575 | 2 | 1 | 0.25 |
| 288-575 | 4 | 2 | 0.5 |
| 288-575 | 8 | 4 | 1 |

The 1T/2T/4T entries come from the isolated and concurrent stage-window sweep.
The end-to-end workloads below selected 8T or 16T fixed teams plus a 1T
tail-pool, so this run directly validates the 8T entries and the inheritance
boundary.

## End-to-end results

Gain is `production_auto / static_stage_windows - 1`.

| workload | dominant routes / width | overridden tasks | W13:W2 MiB | auto ms | static ms | auto TFLOP/s | static TFLOP/s | gain |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 48 / 16T | 0 | inherit | 16.518 | 16.572 | 9.360 | 9.330 | -0.32% |
| active-set-32 | 384 / 8T | 32 | 4:1 | 8.697 | 8.555 | 17.778 | 18.074 | +1.66% |
| active-set-64 | 192 / 8T | 64 | 1:0.5 | 10.007 | 9.066 | 15.451 | 17.055 | +10.38% |
| active-set-128 | 96 / 8T | 128 | 1:0.5 | 11.932 | 9.312 | 12.958 | 16.604 | +28.14% |
| tiered-hotspot | mixed / 8T | 60 | 1:0.5, 4:1 | 8.771 | 8.179 | 17.629 | 18.904 | +7.23% |
| long-short-bimodal | 16T + 1T | 0 | inherit | 10.411 | 10.419 | 14.852 | 14.839 | -0.08% |
| DSV4 captured | mixed / 8T | 25 | 1:0.5, 4:1 | 14.698 | 14.730 | 10.520 | 10.497 | -0.22% |

An earlier 5-run active-set-64 smoke test measured +12.24%; the 21-run result
above, +10.38%, is used for conclusions.

Independent NUMA1 validation used the same 5-warmup/21-run protocol:

| workload | auto ms | static ms | static TFLOP/s | gain |
| --- | ---: | ---: | ---: | ---: |
| active-set-64 | 9.955 | 9.002 | 17.175 | +10.58% |
| active-set-128 | 11.932 | 9.199 | 16.809 | +29.71% |

## Default-on integration check

After enabling the exact-profile resolver, the benchmark omitted
`--static-stage-windows`. With a production profile present, it constructs an
explicitly disabled `production_auto` control and the default
`production_auto_stage_windows` plan:

| NUMA / workload | disabled control ms | default policy ms | default TFLOP/s | gain |
| --- | ---: | ---: | ---: | ---: |
| NUMA0 / active-set-128 | 12.013 | 9.199 | 16.807 | +30.58% |
| NUMA0 / uniform (no overrides) | 16.609 | 16.608 | 9.310 | +0.01% |
| NUMA1 / active-set-128 | 12.079 | 9.267 | 16.684 | +30.34% |

The default resolver therefore reproduces the measured policy on both rank CPU
sets and preserves the inherited path when no task matches a policy band.

## Conclusions

1. Per-task stage windows are useful when many 8T experts have 96-192 routes.
   The same task plan gains 10-28%, so the effect is not a scheduling-layout
   change.
2. The benefit falls to 1.66% by route 384, where a larger W13 window is again
   preferable.
3. Inheritance is effectively neutral: the two workloads with zero overrides
   differ by at most 0.32%.
4. A small number of overridden tasks is insufficient to move whole-call time.
   The captured distribution overrides 25 tasks but remains within noise.
5. The policy is default-on only for the exact dual-NUMA machine/profile
   identity and either validated 96-core rank CPU set. It remains a post-plan
   runtime rule, not a scored cost-model candidate, and must not be extrapolated
   to another machine, shape, kernel identity, or no-split profile. The
   controlled baseline passes `use_default_stage_window_policy=False`.

## Command

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE \
FUSED_CPP_MOE_PREPACK_THREADS=96 PYTHONPATH=src \
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-active-set-128 \
  --production-profile \
  cpu_moe_schedule_optimization/cost_model/profiles/contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260726.json \
  --route-dtype bf16 --warmup 5 --runs 21
```
