# Arm-codex 80C rank LLC vs domain LLC scope ablation

Date: 2026-09-04.

## Decision

Do not add a new default-off structure. After rank DRAM is removed, rank LLC
is the entire leftover remote common mode: turning it off drives remote n15
from $+0.454\,\mathrm{ms}$ to $0$. The remaining local leftover is domain LLC,
and it still grows through n=15 instead of saturating near n≈4. Removing
domain LLC as well, then L2, leaves the 1-route 1T victim essentially
isolated ($+0.001\,\mathrm{ms}$). Hardware still has a $+0.312\,\mathrm{ms}$
same-LLC n15 tax. The missing mechanism is a victim-asymmetric, saturating,
same-LLC-only tax, not another shared-capacity scalar.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed.

## Locked split

The same fit-only artifacts as the rejected joint fit and the rank-DRAM
ablation. Session 2 is compared after scoring, never used to choose an arm.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json` | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json` | `76321dbba93e95089824104c487ebef7beb6dacb411909e87b0a75d44165cc02` |
| `tmp/moe_gather_absolute_pressure_fit_20260904.json` | `a9adb55381f99a1b2c2db934c692130d6ec28c07e26eadcda650070bb0b8abb1` |
| `tmp/moe_gather_absolute_pressure_repeat_20260904.json` | `b7d35dcda1f12e0c1afeccd72b45c2df42c9e84c895c5ae1b52ea9afcf06f625` |
| `tmp/moe_rank_llc_domain_scope_ablation_20260904.json` | `8f571995f877816a4b0b3c3ad5d92d3ee9690d1d757574cefd9f6ced2049f64a` |

Command:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/ablate_rank_llc_domain_scope.py \
  --base-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --phase-fit tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json \
  --pressure-fit tmp/moe_gather_absolute_pressure_fit_20260904.json \
  --pressure-validation tmp/moe_gather_absolute_pressure_repeat_20260904.json \
  --output-report tmp/moe_rank_llc_domain_scope_ablation_20260904.json
```

Rank DRAM stays disabled on every arm, using the previous first-call
`service_rate("dram_bytes")` patch. Rank LLC is the `llc_bytes` call that
passes every domain id; domain LLC is the one-key per-domain call. Rank LLC
is removed by returning infinite capacity when `len(llc_domain_threads) > 1`.
The first-call rule used for DRAM is forbidden here: domain LLC is invoked
before rank LLC.

All arms keep $\alpha_g=1$ and domain DRAM injection off. No parameter
search.

Predeclared pass for adding a structure, all required on one arm:

- remote n15 head absolute $\le 0.08\,\mathrm{ms}$
- n4 same-minus-cross $\ge 0.10\,\mathrm{ms}$
- same-LLC n15 minus n4 $\le 0.05\,\mathrm{ms}$

## Arm results

Isolated-relative head medians, milliseconds. Hardware from session 1.

| Arm | same n1 / n4 / n15 | cross n1 / n4 / n15 | n15$-$n4 same | n4 contrast | fit joint MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardware | 0.144 / 0.291 / 0.312 | 0.008 / 0.011 / 0.043 | 0.021 | 0.280 | — |
| no_rank_dram | 0.000 / 0.397 / 0.590 | 0.000 / 0.186 / 0.454 | 0.194 | 0.211 | 0.309 |
| domain_llc_only | 0.000 / 0.397 / 0.590 | 0.000 / 0.000 / 0.000 | 0.194 | 0.397 | 0.372 |
| no_llc | 0.000 / 0.000 / 0.001 | 0.000 / 0.000 / 0.000 | 0.001 | 0.000 | 0.376 |
| no_llc_no_l2 | 0.000 / 0.000 / 0.001 | 0.000 / 0.000 / 0.000 | 0.001 | 0.000 | 0.376 |

`rank_llc_fraction_of_leftover_remote_n15 = 1.0`. Same-LLC `domain_llc_only`
matches `no_rank_dram` exactly, as required: one active domain already uses
the domain curve for rank LLC. `no_llc` and `no_llc_no_l2` are identical on
this probe; L2 dilation $\approx 1.10$ does not move the 1-route span.

Local count shape after rank LLC is removed, same-LLC head:

| Count | hardware | domain LLC | no LLC |
| ---: | ---: | ---: | ---: |
| n1 | 0.144 | 0.000 | 0.000 |
| n2 | ~0.20 | 0.168 | 0.000 |
| n4 | 0.291 | 0.397 | 0.000 |
| n8 | ~0.30 | 0.447 | 0.001 |
| n15 | 0.312 | 0.590 | 0.001 |

`domain_llc_only` meets remote-near-zero and n4 contrast, but n15$-$n4 is
still $0.194\,\mathrm{ms}$. It is not one missing resource.

## Leftover resource

On `cross_llc_head_n15` after rank DRAM is removed, gather and the first W13
cold panel stay undilated. Later W13/W2 cold panels have phase dilation
$2.05$, rank `llc_bytes` dilation $2.44$, aggressor-domain LLC $2.84$, and
**victim-domain LLC $1.00$**. The victim inherits rank fabric sharing through
$\max(\mathrm{rank},\mathrm{local})$. Turning rank LLC off zeros that rank
term; victim phase dilation becomes $1.00$ even though the aggressor domain
detail remains $2.84$. That is the entire leftover remote common mode.

On `same_llc_head_n15`, victim-domain LLC dilation is $2.81$ and stays there
after rank LLC is removed. Turning all LLC off drops phase dilation to
$\approx 1.004$. L2 remains $\approx 1.10$ and does not account for the
$0.59\,\mathrm{ms}$ local leftover.

## Why this is not one missing resource

1. Rank LLC explains all leftover remote n15, not a fraction. That is
   directionally useful and passes the remote-near-zero gate.
2. The same arm keeps the growing local domain-LLC utilization
   ($0.168/0.397/0.447/0.590\,\mathrm{ms}$ at n2/4/8/15) and misses the
   hardware n≈4 plateau.
3. Removing domain LLC as well zeros the local leftover, so the $0.59\,\mathrm{ms}$
   was domain LLC of 68-route peers, not a second rank resource.
4. After rank DRAM, rank LLC, domain LLC, and L2 are gone, the model victim is
   isolated, while hardware still has a saturating same-LLC-only tax of
   $\approx 0.31\,\mathrm{ms}$ and a same-LLC n1 tax of $0.144\,\mathrm{ms}$.

The hardware curve is a small, saturating, same-LLC-only tax on a 1-route 1T
victim. The remaining model path is domain-local LLC utilization of overlapping
68-route GEMM streams.

## Next action

Completed and rejected: victim-asymmetric dilation is recorded in
[arm_codex_80c_victim_asymmetric_dilation_20260904.md](./arm_codex_80c_victim_asymmetric_dilation_20260904.md).
Own-demand zeros the inherited 68-route byte dilation and also zeros the
hardware local tax. Next is an occupancy identification independent of this
count sweep: vary aggressor $M$ at fixed count, or victim $M$ at fixed peers.
