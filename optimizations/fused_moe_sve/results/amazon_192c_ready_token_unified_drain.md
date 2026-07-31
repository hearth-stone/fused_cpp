# Ready-token same-job drain

Date: 2026-07-30

## Question

Can the async fused-expert executor stop preserving a contiguous final merge
phase, let the resident workers drain all ready tokens, and claim multiple
tokens to provide prefetch lookahead without regressing the balanced case?

## Implementation

- Expert tasks remain higher priority than merge work.
- While expert compute remains, an idle worker claims one ready token.
- Once all expert tasks complete, the same resident worker invocation keeps
  running until every token is merged.
- Tail drain claims up to two queue slots. Before merging the current token, it
  prefetches the next token's TopK route rows, combine weights, and output.
- Queue order is release order; token IDs are not made contiguous.
- `FUSED_CPP_MOE_ASYNC_READY_TOKEN_DRAIN=0` restores the previous early-ready
  plus contiguous-final behavior.

## Machine and method

- Host: `AmazonC5192Cores`, Neoverse V3, NUMA0 CPU `0-95`
- Shape: `tokens=2048`, `top_k=6`, `H=4096`, `F=512`, 256 local experts
- Workload: `moe256-active-set-8`
- Plan: production `6x16T -> 4x24T` route-sliced tail
- Store/merge: BF16 direct-route store and SVE weighted merge
- Sampling: one packed-weight allocation, randomized variant order, 7 warmups,
  101 measured runs

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-active-set-8 \
  --production-profile cpu_moe_schedule_optimization/cost_model/profiles/contention_async_amazon_c5_192c_numa0_tp4_sve_F512_E256_splitw13_schema_v2_xbyak_exactm_20260727.json \
  --threads 96 --route-dtype bf16 \
  --ready-token-drain-batches 1,2,4,8 \
  --warmup 7 --runs 101
```

An earlier NUMA0 run was rejected because an unrelated `cpufb` process occupied
CPU95: production-auto moved from about 8.7 ms to 294 ms with a 164--552 ms
P10--P90 range. The table below was collected only after that process exited.

## Results

| Policy | Median (ms) | P10 (ms) | P90 (ms) | Gain vs legacy |
| --- | ---: | ---: | ---: | ---: |
| Legacy early + contiguous final | 8.691 | 8.630 | 8.963 | - |
| Same-job drain, batch 1 | 8.659 | 8.596 | 8.774 | 0.37% |
| Same-job drain, batch 2 | 8.637 | 8.571 | 8.706 | 0.63% |
| Same-job drain, batch 4 | 8.619 | 8.562 | 8.721 | 0.84% |
| Same-job drain, batch 8 | 8.648 | 8.583 | 8.783 | 0.51% |

Lightweight stage counters on the same workload reported:

| Policy | Merged in resident job | Left for final merge |
| --- | ---: | ---: |
| Legacy | 584 | 1464 |
| Same-job drain, batch 2 | 2048 | 0 |

The controlled 25%/75% two-group case improved from 10.266 ms to 10.181 ms
(0.83%). The balanced case activates the existing team-load gate and differed
by at most 0.15% across policies.

## Conclusion

The structural objective is met: no second merge worker dispatch is required,
and correctness remains bit-exact. The measurable gain is modest but repeatable
in the imbalanced tail, with a larger P90 improvement (2.95%) than median
improvement (0.63%).

Batch 1--8 spans only 0.040 ms. These measurements do not establish an
independent software-prefetch gain. Batch 2 remains the default because it
provides one-token lookahead while retaining finer tail load balance than the
larger batches. The planner and cost-model search space are unchanged.
