# Arm-codex 80C aggressor-M occupancy identification

Date: 2026-09-04.

## Decision

Do not add a structure. Same-LLC overlap is valid at every $M$, remote stays near
zero, and the leftover tax **falls** as aggressor $M$ grows. Utilization
(more peer bytes $\to$ more tax) is false. Occupancy independent of $M$ is
false. Duration occupancy (tax appears only when the peer is long) is false at
lane head: short transfer-bound peers tax more, not less.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed. Do not retrofit
$T_\mathrm{sat}$ or an additive residual onto this sweep.

## Question

After rank DRAM, rank LLC, and victim-asymmetric dilation are rejected, the
leftover hardware curve is a small, saturating, same-LLC-only tax on a 1-route
1T victim. Packed-B is almost independent of $M$. This probe asks whether that
leftover tracks peer occupancy or peer bytes.

## Locked protocol

Victim remains expert 0, $M=1$, 1T, logical 64 / CPU304 / LLC7. Counts are
`0,1,4`. Aggressor routes are `1,4,16,68`. Same-LLC uses logical `65-79`;
cross-LLC uses `0-14`. Unused background experts stay in the plan and are
dependency-delayed after the target-lane tail. $M$ and placement are randomized
together each round.

Predeclared signatures, all on head isolated-relative medians; this probe does
not add a structure:

| Signature | Rule |
| --- | --- |
| occupancy | same-LLC n1 and n4 change by $\le 0.05\,\mathrm{ms}$ from $M=1$ to $M=68$ |
| duration occupancy | n4 tax $\le 0.08\,\mathrm{ms}$ at $M=1$, $\ge 0.10\,\mathrm{ms}$ at $M=68$, and $M=16$ versus $M=68$ differ by $\le 0.05\,\mathrm{ms}$ |
| utilization | n4 tax grows by $\ge 0.15\,\mathrm{ms}$ both $M=1\to 68$ and $M=16\to 68$ |
| invalid_overlap | same-LLC n1/n4 `peer_overlap_experts` median $< 0.8\times$ count |

Command:

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_aggressor_m_occupancy.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --seed 20260909 \
  --output tmp/moe_aggressor_m_occupancy_fit_20260904.json
```

| Artifact | SHA256 |
| --- | --- |
| frozen v8 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_aggressor_m_occupancy_fit_20260904.json` | `f1daf1e5b4aa02cc9d0c61fcfbde72005eb4f678be2bb20e48d25e031a3217b0` |
| extension | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |

CPU mapping: logical 0 is CPU240 NUMA3 LLC6 (`240-279`); logical 64 is CPU304
NUMA3 LLC7 (`280-319`). Isolated head W13 is stable across $M$
(`0.373/0.368/0.368/0.368 ms`). n4 `peer_overlap_experts` is `4.00` at every
$M$. Automated `signature` is `inconclusive` against the three predeclared
bins.

## Head results

Isolated-relative medians, milliseconds. Session seed `20260909`, 31 paired
rounds.

| Aggressor $M$ | same n1 / n4 | cross n4 | same n4 W13 | n4 P10 / P90 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.198 / 0.660 | 0.037 | 0.809 | 0.637 / 0.701 |
| 4 | 0.204 / 0.639 | 0.050 | 0.785 | — |
| 16 | 0.136 / 0.402 | 0.023 | 0.635 | — |
| 68 | 0.106 / 0.290 | 0.004 | 0.567 | 0.274 / 0.317 |

$M=68$ n4 `+0.290 ms` matches the locked count-sweep n4 `+0.291 ms`. n4 at
$M=1$ is separated from $M=68$ in P10/P90. Remote n4 stays $\le 0.05\,\mathrm{ms}$.

## after_1 n4

| Aggressor $M$ | same n4 | cross n4 |
| ---: | ---: | ---: |
| 1 | $-0.014$ | 0.034 |
| 4 | 0.002 | 0.023 |
| 16 | 0.382 | 0.018 |
| 68 | 0.251 | $-0.022$ |

Short $M=1/4$ peers finish during the 1-route delay, so after_1 is not a
matched occupancy cell for those $M$. Head is the identification window.

## Why this is not utilization or $M$-independent occupancy

1. More aggressor routes do not increase the leftover tax. n4 falls
   $0.660\to 0.290\,\mathrm{ms}$ from $M=1$ to $M=68$.
2. Four same-LLC peers overlap the victim at every $M$ (`peer_overlap_experts=4`).
   Missing overlap does not explain the drop.
3. Isolated W13 is unchanged across $M$, so the matched control is not drifting
   with the larger `x` tensor.
4. Transfer-bound $M=1/4$ peers (same packed-B, DRAM-bound W13) produce the
   largest same-LLC tax. GEMM-bound $M=68$ peers produce the smaller leftover
   already seen on the count sweep.

The leftover is same-LLC contention among concurrent **transfer-bound** streams,
not inheritance of 68-route GEMM bytes and not a count-only occupancy fee.

## Next action

Do not add a default-off structure from this $M$ sweep. Session-2 is optional
confirmation; P10/P90 already separate $M=1$ from $M=68$. Do not read holdout.
Varying victim $M$ is not required to reject utilization: the small-expert
route sweep already showed the 1T tax collapsing as the victim itself grows.
The next probe splits 16 fill ports from 16 streams on the same LLC7 cores:
see [arm_codex_80c_fill_port_vs_stream_20260904.md](./arm_codex_80c_fill_port_vs_stream_20260904.md).
