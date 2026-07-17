# Amazon C5 192-core async ready-token merge

## Setup

- Host: `AmazonC5192Cores`, Neoverse-V3
- Binding: NUMA0, CPUs `0-95`
- Shape: `tokens=2048`, `top_k=6`, `H=4096`, `F=512`, 12 experts
- Schedule: 12 concurrent teams, 8 threads per expert, split-W13
- W2/merge: FP32 direct-route store and SVE U1 weighted merge
- Statistic: 31 timed A/B pairs after five warmups; pair order is randomized
  from a fixed seed
- Baseline: `FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=0` forces the contiguous
  merge after expert compute
- Candidate: the ready-token path. The three-repeat table used explicit value
  `1`; the default-adoption check leaves the variable unset, which is equivalent
- Workspace base: `b0a1643` plus the uncommitted ready-token feature

The balanced distribution assigns 1024 routes to every expert. The controlled
heavy-tail distribution sends 25% of tokens to experts 0-5 and 75% to experts
6-11, giving route counts 512 and 1536 respectively. The two TopK groups are
disjoint, so the first 512 tokens can become ready while the longer group is
still computing.

## Accepted design

Each expert leader performs the following work after the team's final W2
barrier:

1. release-publish the expert task's completion state;
2. participate in one global completion-publication RMW;
3. scan its route tokens and acquire-read their TopK expert states;
4. claim a newly ready token with one CAS;
5. reserve one queue range and release-publish all claimed tokens in a batch.

The worker loop remains expert-first. A lane claims one ready token only when
its fixed interval has no eligible expert task. After the expert parallel
region joins, the original static contiguous merge skips tokens already merged
and processes every remaining range.

The feature applies only to the async SVE FP32 direct-route path. It also
requires

```text
max(ceil(M_i / 12) / threads_i) >= 1.25 * min(ceil(M_i / 12) / threads_i).
```

Balanced calls therefore retain the original execution path under the default.

## Rejected prototype

The first prototype decremented one shared atomic counter for every route. On
the balanced shape, 12,288 contended RMWs changed the median from 7.754 ms to
10.116 ms, a 23.35% regression. The accepted design replaces those writes with
mostly read-only TopK state checks, one CAS per ready token, and one queue-tail
RMW per publishing expert.

## Results

### Balanced distribution

The load gate disables early merge, so this measures default detection and
fallback overhead.

| Seed | Post barrier ms | Ready-token default ms | Gain |
|---:|---:|---:|---:|
| 20260716 | 7.750 | 7.761 | -0.14% |
| 20260717 | 7.846 | 7.838 | +0.10% |
| 20260718 | 7.774 | 7.796 | -0.29% |

The median per-run gain is -0.14%, below the previously measured roughly 1%
operator noise floor. A trace records zero early token merges, confirming that
the candidate used the old contiguous path.

### 25%/75% two-group distribution

| Seed | Post barrier ms | Ready-token ms | Gain |
|---:|---:|---:|---:|
| 20260716 | 10.242 | 10.219 | +0.23% |
| 20260717 | 10.237 | 10.203 | +0.33% |
| 20260718 | 10.219 | 10.189 | +0.29% |

One traced candidate call merged 605 of 2048 tokens before expert compute
finished. Its summed per-worker early-merge service time was 4.652 ms; this is
not wall time because those records run concurrently. The median gain across
repeats is +0.29%, also below the noise floor. The valid conclusion is that the
first version exposes the intended overlap without a measurable median
regression; these runs do not establish a material E2E speedup.

After the final review restored the async task-state atomics to their existing
`seq_cst` semantics, one additional 31-pair run measured -0.06% for balanced
routes and +0.48% for the two-group distribution. This confirms that the
reported no-regression result applies to the final source.

After default adoption, a direct unset-environment A/B measured 7.860 versus
7.870 ms (-0.13%) for balanced routes and 10.264 versus 10.233 ms (+0.31%) for
the two-group distribution. Thus the default dispatch has the same
no-regression behavior as explicit value `1`.

Periodic 16-21 ms host outliers occur in both variants. The benchmark retains
all raw samples and randomizes pair order so those events do not remain phase
locked to one flag. Median results are used for the implementation decision.

## Correctness

The focused test uses two independent TopK=2 groups: eight short tokens and 256
long tokens, four one-thread expert teams, and direct-route FP32 output. Five
concurrent candidate calls and one traced call bit-match the post-barrier
baseline. The trace must contain `stage=merge_ready_token`, proving that the
test exercises early merge rather than only the fallback.

```bash
PYTHONPATH=src taskset -c 0-3 .venv/bin/python -m pytest -q \
  tests/test_fused_moe_bf16_tiled.py \
  -k async_ready_token_merge_overlaps_imbalanced_experts
```

Focused result: `1 passed`.

## Reproduction

```bash
for distribution in balanced two-group; do
  PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
    .venv/bin/python \
    optimizations/fused_moe_sve/benchmarks/bench_async_ready_token_merge.py \
    --tokens 2048 --hidden 4096 --intermediate 512 \
    --experts 12 --top-k 6 --threads 96 --distribution "$distribution" \
    --warmup 5 --runs 31
done
```

The no-regression gate and explicit fallback support default adoption for the
async SVE FP32 direct-route path. Captured heavy-tail routing, multi-wave
schedules, and an expert/merge contention model remain required before planner
integration or changing the 1.25 load threshold.
