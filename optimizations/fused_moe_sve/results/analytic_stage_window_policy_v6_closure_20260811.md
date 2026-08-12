# Analytical Stage-Window Policy V6 Closure

## Decision

The analytical selector is closed for the declared 192-core TP4 transition
domain as a **window-selection model**. It is not closed as an exact absolute
latency model and is not adopted as the production planner policy.

The predeclared selection gate was maximum measured regret at or below 5% over
the full legal W13 x W2 Cartesian set. V6 reaches `1.10/3.45/4.40%`
median/P90/maximum regret, so all six holdout points pass.

The earlier policy-v2 `1.50/2.98/3.38%` result used a coordinate oracle and the
retired range ABI. It showed near-optimal local decisions, but it could not
close the current tile-window selector or establish that every candidate time
was explained.

## Scope

| Item | Value |
| :--- | :--- |
| Host | `AmazonC5192Cores`, 192 AArch64 cores |
| Placement | NUMA0 CPUs `0-95`, memory node 0 |
| Kernel | SVE JIT exact-M fused W13 and direct-route W2 |
| Shape | TP4, `H=4096`, `F=512`, BF16 packed weights |
| Holdout | M=`72,216,384`, T=`4,16` |
| Weights | 96 distinct experts, rotated by the runtime |
| Pages | 32 MiB HugeTLB at `/dev/hugepages-32M` |
| Sampling | 2 warmups + 15 shuffled rounds per Cartesian point |
| Calibration | `AmazonC5192Cores-numa0-sve-jit-hot-gemm-20260802` |
| Calibration SHA256 | `6289289248a4ff423c892abb5781b9fbd60002f9fd665a528ef3d8ddb1a9875d` |

Production Plan V2 defaults, the static stage-window band policy, task-width
search, packed layouts, and numerical behavior are unchanged. V6 is reachable
only through `AnalyticMoeCostModel.shadow_stage_window_policy()` and the Lab
benchmarks.

## Model

For stage `s`, packed N tile count `q_s`, team width `t`, and per-worker owner
window `w_s` in tiles:

```text
range_tiles = t * w_s
R_s         = ceil(q_s / range_tiles)
tile_bytes  = 2 * K_s * n_tile
```

The structural candidate set contains the full stripe and power-of-two windows
at or above one L1D, with any thread-starving tail removed. `M<=12` selects the
full stripe because one M panel has no repeated packed-B scan to protect.

The physical score contains exact GEMM work, A/B L2 refill, C writes, L2/LLC/
DRAM service, and stage-specific range restart. The rank-level capacity
surcharge counts only reusable packed B as resident state. Streaming A and C
still consume transfer service, but they do not consume the reusable-B LLC
budget a second time.

Candidates inside calibrated transfer uncertainty are treated as equivalent.
The tie-break targets

```text
max(L1D, 2 * tile_bytes, sqrt(L1D * effective_B_retention_L2))
```

which is 128 KiB on this host from `sqrt(64 KiB * 256 KiB)`. This is derived
from cache geometry and contains no route/thread latency lookup.

## Thin Calibration

The extra N-range cost must match the executed stage epilogue. A pure
full-no-store probe understates W13 because it omits SiLU, gate multiplication,
and BF16 packC.

Both probes use `M=12, K=264, N=64`, a 40,128-byte L1-hot working set, 256
warmups, 8192 timed calls, and seven randomized range-count orders.

| Probe | Intercept | Extra range per M panel | Maximum fit error |
| :--- | ---: | ---: | ---: |
| Pure full-no-store | 1192.09 ns | 6.70 ns | 0.180% |
| Fused W13 SiLU/packC | 2389.01 ns | 45.27 ns | 0.087% |

The calibration therefore uses:

```text
restart_s = exact_M_panels * (R_s - 1) * delta_s
delta_w13 = 45.269565 ns
delta_w2  =  6.695652 ns
```

W2 remains a pure-GEMM proxy. This is an explicit residual risk, not an inferred
W2 epilogue measurement.

## Falsified Alternatives

1. **One generic range restart constant is sufficient.** False. Fused W13 is
   6.76x the pure-GEMM increment under the same L1-hot geometry.
2. **All A, B, and C bytes form the rank LLC resident set.** False as a window
   budget. A and C are streaming demand; only repeated packed-B refill can be
   saved by retaining the owner window. They remain in bandwidth demand.
3. **The M=216 transition is caused by an inter-range barrier.** False. A
   synchronized 96-core M216/T4 fused-W13 probe still selects 8 tiles. The
   measured synchronization tax is 1.9-3.9% and does not move the optimum.

The independent fused-W13 service sweep also gives the expected mechanism:
small M prefers small windows, M216/T4 peaks at 8 tiles, and M384/T16 returns to
the full 8-tile stripe as A-rescan amortization dominates.

## Full Cartesian Holdout

| M | T | Analytical W13/W2 tiles | Runtime oracle tiles | Analytical ms | Oracle ms | Regret |
| ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| 72 | 4 | 2 / 16 | 1 / 8 | 5.236 | 5.015 | 4.40% |
| 72 | 16 | 2 / 16 | 1 / 8 | 6.206 | 6.100 | 1.74% |
| 216 | 4 | 8 / 16 | 2 / 16 | 13.335 | 13.008 | 2.51% |
| 216 | 16 | 8 / 16 | 8 / 16 | 13.419 | 13.419 | 0.00% |
| 384 | 4 | 8 / 16 | 8 / 16 | 23.275 | 23.275 | 0.00% |
| 384 | 16 | 8 / 16 | 8 / 32 | 21.876 | 21.774 | 0.47% |

Summary:

- median/linearly-interpolated-P90/maximum regret: `1.10/3.45/4.40%`;
- points within 2%: `4/6`;
- points within 5%: `6/6`;
- median candidate-rank Spearman correlation: `0.869`.

The full Cartesian oracle is stricter than the historical coordinate oracle.
The largest miss, M72/T4, is still inside the calibration's 5% uncertainty and
the declared decision gate.

## Remaining Boundary

Near-oracle selection does not imply exact candidate timing. Paired-round
W13/W2 additivity residual is 2.56-17.69% across the six groups, with the
maximum at M216/T4. M72/T16 has rank correlation 0.098 even though selected
regret is only 1.74%. V6 therefore captures the first-order decision boundary
but not every second-order stage interaction.

Before production adoption, the following remain required:

- machine-local packed-B retention and the same Cartesian gate on the 8-core
  ARM host;
- unseen routes and widths, especially outside T=4/16;
- mixed-distribution planner and E2E validation;
- either model or bound the W13/W2 pair interaction for absolute-time use.

## Reproduction

Thin service calibration:

```bash
numactl --physcpubind=0-95 --membind=0 \
  .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_analytic_services.py \
  --output /tmp/moe_analytic_services.json \
  --cpu-ids 0-95 \
  --range-warmup 256 --range-runs 8192 --range-repeats 7
```

Full Cartesian holdout:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=0-95 --membind=0 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_analytic_stage_window_tiles.py \
  --calibration \
  cpu_moe_schedule_optimization/cost_model/profiles/analytic_machine_amazon_c5_192c_numa0_sve_jit_hot_gemm_20260802.json \
  --output /tmp/stage_window_cartesian_v6_72_216_384_t4_16.json \
  --cpu-ids 0-95 --routes 72,216,384 --widths 4,16 \
  --measurement-experts 96 --warmup 2 --runs 15 \
  --full-cartesian --store-samples
```

Raw JSON was intentionally left out of the repository. The deciding files had
these SHA256 digests:

```text
8a6a79d27c00553f375c72b5cdc6a9a28bf790eb9beb3389d6935683beedfca2  stage_window_cartesian_v6_72_216_384_t4_16.json
33356cd3e3699b086ad9242a469c0d16c53b9941aa5bb1c541b41c01a475d177  stage_window_range_restart_20260811.json
dae8425db656963c3515213f6889572d7884facce4ddd9f94e89e26c108e5475  stage_window_service_fused_sync_m216_t4.json
```
