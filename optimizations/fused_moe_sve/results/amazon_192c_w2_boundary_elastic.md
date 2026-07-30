# Amazon 192C W2 Boundary Elastic Experiment

Date: 2026-07-29

## Scope

- Host: Amazon 192-core ARM, NUMA0 CPUs `0-95`
- Kernel: fused SVE, split-W13, exact-M JIT, direct BF16 route store
- Shape: TP4-style `H=4096`, `F=512`, `E=256`
- Baseline: the same strict Plan V2 task graph with ready-token merge disabled
- Elastic action: keep the selected team for W13, then try one planner-declared
  W2 expansion at the W13/W2 boundary
- Correctness: strict, finite-timeout elastic, and zero-timeout lock-free elastic
  outputs are bit-exact in the target-machine SVE test

The calibration profile predates this runtime binary, so the benchmark reports
an extension-hash warning. All numbers below are measured E2E times; no
profile-predicted time is used in the comparison.

## Results

| Workload | Cohort | Timeout | Strict median | Elastic median | E2E change | Natural | Preferred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Active set 8, `M=1536` | `2x8T -> 16T` | 0 | 9.981 ms | 10.240 ms | -2.53% | 0.00% | 0.00% |
| Tiered hotspot | `2x8T -> 16T` | 0 | 8.981 ms | 9.886 ms | -9.15% | 0.67% | 0.67% |
| Tiered hotspot | `2x8T -> 16T` | 5 us | 8.981 ms | 9.891 ms | -9.21% | 6.03% | 23.44% |
| Tiered hotspot | `2x8T -> 16T` | 20 us | 8.981 ms | 9.941 ms | -9.67% | 15.62% | 63.84% |
| DSV4 captured 2K/TopK6 | `2x8T -> 16T` | 0 | 14.809 ms | 19.690 ms | -24.79% | 6.53% | 6.53% |
| 256 active `M=1`, TopK1 | `4x2T -> 8T` | 0 | 9.081 ms | 11.840 ms | -23.30% | 0.21% | 0.21% |
| 256 active `M=1`, TopK1 | `4x2T -> 8T` | 5 us | 9.081 ms | 11.969 ms | -24.13% | 0.18% | 0.18% |
| 256 active `M=1`, TopK1 | `4x2T -> 8T` | 20 us | 9.081 ms | 11.978 ms | -24.19% | 0.25% | 0.46% |

`Natural` is `natural_opportunities / eligible_tasks`; `Preferred` also
includes cohorts formed after a positive wait. A controlled one-expert test
with the adjacent base team idle reaches one natural acquisition out of one
eligible task and remains bit-exact, proving that the expansion path itself is
reachable.

For the tiered workload, 5 us and 20 us produce 343 and 162 timeout fallbacks,
respectively. Their measured maximum W2-ready-to-assignment delays are
3.205 ms and 4.440 ms. This does not mean the runtime keeps waiting for a new
cohort after the deadline: a task that already lent its team to a
non-preemptible preferred-width W2 job must wait for that one finite job to
finish. After its deadline, the team cannot be borrowed again.

## Explicit Tail Migration

The active-set-8 strict plan has six 16-thread lanes. Task ids are lane-major,
so the two second-wave tail experts are task 1 on cores `0-15` and task 3 on
cores `16-31`. The explicit targets are:

```text
task 1: W13 0-15  -> W2 0-31
task 3: W13 16-31 -> W2 32-63
```

Task 3 first acquires its disjoint destination and then releases cores
`16-31`. The scheduler can use that released source interval to complete task
1's containing cohort in the same pass. The runtime never copies packed-C.

| Runs | Timeout | Strict median | Elastic median | E2E gain | Preferred |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 0 | 11.103 ms | 10.710 ms | +3.66% | 61/62 (98.39%) |
| 11 | 0 | 11.079 ms | 10.687 ms | +3.66% | 22/22 (100%) |
| 11 | 200 us | 11.079 ms | 10.713 ms | +3.42% | 22/22 (100%) |
| 11 | 500 us | 11.079 ms | 10.748 ms | +3.08% | 22/22 (100%) |

The SVE native test forces one containing expansion and one disjoint migration,
requires both preferred assignments, and compares the output bit-for-bit with
strict Plan V2. Bounded waiting is unnecessary for this planned target and is
slightly slower.

## Interpretation

1. The strict planner packs all selected teams densely. An aligned neighbor is
   therefore usually running W13 or W2 rather than idle, so free-thread
   availability elsewhere in the NUMA node does not imply a legal local
   cohort.
2. A positive timeout can raise the preferred-width rate, but it serializes
   complete W2 jobs inside the cohort. The extra W2 scaling is insufficient to
   repay that queueing in the measured cases.
3. Splitting every expert at the W13/W2 boundary has a task-count-dependent
   handoff cost. It is small for eight long experts but dominates the captured
   and `M=1` workloads.
4. The current action must remain explicit and outside production cost-model
   search. A future candidate needs a strict-hot-path handoff and an admission
   rule that charges boundary overhead and cohort queueing before enabling it.
5. Explicitly targeting the known tail tasks is materially different from
   enabling elastic handoff on every width-matching task. The paired migration
   removes the target-collision that prevented two 16-thread tails from both
   reaching 32 threads and improves this active-set-8 case by 3.66%.

## Reproduction

Use:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-tiered-hotspot \
  --production-profile <amazon-192c-schema-v2-profile.json> \
  --elastic-w2-transitions 8:16 \
  --elastic-timeout-us 0 \
  --route-dtype bf16 --warmup 2 --runs 7
```

Raw benchmark records:

- `w2_elastic_active8_final.json`
- `w2_elastic_tiered_final.json`
- `w2_elastic_tiered_wait_final.json`
- `w2_elastic_real_final.json`
- `w2_elastic_4x2to8_final.json`
- `w2_elastic_active8_migration_timeout_sweep.json`
- `w2_elastic_active8_migration_31runs.json`
