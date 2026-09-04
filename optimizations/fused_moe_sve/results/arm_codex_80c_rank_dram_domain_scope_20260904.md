# Arm-codex 80C rank DRAM vs domain-only scope ablation

Date: 2026-09-04.

## Decision

Do not add a new default-off structure. Rank DRAM sharing is only about half
of the remote common-mode overprediction. The leftover is LLC dilation of
the 1-route 1T victim by overlapping 68-route streams. Domain-only injection
cannot make remote near zero and does not reproduce the hardware n≈4
plateau.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed.

## Locked split

The same fit-only artifacts as the rejected joint fit. Session 2 is
compared after scoring, never used to choose an arm.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json` | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json` | `76321dbba93e95089824104c487ebef7beb6dacb411909e87b0a75d44165cc02` |
| `tmp/moe_gather_absolute_pressure_fit_20260904.json` | `a9adb55381f99a1b2c2db934c692130d6ec28c07e26eadcda650070bb0b8abb1` |
| `tmp/moe_gather_absolute_pressure_repeat_20260904.json` | `b7d35dcda1f12e0c1afeccd72b45c2df42c9e84c895c5ae1b52ea9afcf06f625` |
| `tmp/moe_rank_dram_domain_scope_ablation_20260904.json` | `2403d7c1b98d928fa287c87bfaf3039ef19debfc8bb7277639260693a500bfa1` |

Command:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/ablate_rank_dram_domain_scope.py \
  --base-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --phase-fit tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json \
  --pressure-fit tmp/moe_gather_absolute_pressure_fit_20260904.json \
  --pressure-validation tmp/moe_gather_absolute_pressure_repeat_20260904.json \
  --output-report tmp/moe_rank_dram_domain_scope_ablation_20260904.json
```

Rank DRAM is removed by returning infinite capacity on the first
`service_rate("dram_bytes")` call in each placed-phase allocation. Later
calls keep the original `C_R(n_d)` used by domain injection. Inflating the
`dram_bytes` curve itself is forbidden: at $\beta=0.78$, $n=1$ would then
use $148\,\mathrm{GB/s}$ instead of $C_R(1)=34\,\mathrm{GB/s}$.

Predeclared pass for adding a structure, all required on one arm:

- remote n15 head absolute $\le 0.08\,\mathrm{ms}$
- n4 same-minus-cross $\ge 0.10\,\mathrm{ms}$
- same-LLC n15 minus n4 $\le 0.05\,\mathrm{ms}$

No parameter search. $\alpha_g=1$ except the rejected joint point
$(\beta,\alpha_g)=(0.78,0.25)$.

## Arm results

Isolated-relative head medians, milliseconds. Hardware from session 1.

| Arm | same n1 / n4 / n15 | cross n1 / n4 / n15 | n15$-$n4 same | n4 contrast | fit joint MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardware | 0.144 / 0.291 / 0.312 | 0.008 / 0.011 / 0.043 | 0.021 | 0.280 | — |
| rank_only | 0.082 / 0.729 / 0.897 | 0.082 / 0.729 / 0.893 | 0.168 | 0.000 | 0.712 |
| no_rank_dram | 0.000 / 0.397 / 0.590 | 0.000 / 0.186 / 0.454 | 0.194 | 0.211 | 0.309 |
| domain_only $\beta=1.00$ | 0.082 / 0.729 / 0.897 | 0.000 / 0.186 / 0.454 | 0.168 | 0.543 | 0.593 |
| domain_only $\beta=0.78$ | 0.082 / 0.729 / 1.057 | 0.000 / 0.186 / 0.454 | 0.328 | 0.543 | 0.649 |
| domain_only $\beta=0.50$ | 0.082 / 0.729 / 1.897 | 0.000 / 0.186 / 0.454 | 1.168 | 0.543 | 0.947 |
| domain_only $\beta=0.20$ | 0.164 / 1.028 / 5.706 | 0.000 / 0.186 / 0.216 | 4.678 | 0.842 | 3.210 |
| rejected joint | 0.082 / 0.729 / 1.057 | 0.082 / 0.729 / 0.893 | 0.328 | 0.000 | 0.689 |

`rank_dram_fraction_of_remote_n15 = 0.491`. Removing rank DRAM is
directionally useful — it is the best joint MAE on this probe — but remote
n15 remains $0.454\,\mathrm{ms}$ versus hardware $0.043\,\mathrm{ms}$.

## Leftover resource

On `cross_llc_head_n15` after rank DRAM is removed, target gather and the
first W13 cold panel stay undilated. Later W13/W2 cold panels have phase
dilation $2.05$, with `llc_bytes` dilation $2.44$ and `l2_bytes` $1.10`.
DRAM dilation is gone. Same-LLC n15 leftover LLC dilation is $2.81$, so LLC
is already partly domain-local, but remote LLC sharing is still far above
hardware.

Same-LLC n1 with rank DRAM enabled is almost entirely `dram_bytes` dilation
$1.195$ (LLC $1.03$ is hidden). Removing rank DRAM makes predicted n1 local
slowdown $0$, while hardware is $+0.144\,\mathrm{ms}$ local and $+0.008\,\mathrm{ms}$
remote. Rank DRAM had the right n1 magnitude and the wrong scope.

## Why this is not one missing resource

1. Rank DRAM explains about half of remote n15, not the remote-near-zero
   series.
2. Domain-only DRAM restores extra local contrast on top of leftover LLC,
   and at $\beta\le 0.78$ it keeps growing through n=15.
3. Tight $\beta=0.20$ explodes local n15 instead of saturating at n≈4.
4. The n=1 same-LLC-only tax is not the leftover LLC path.

The hardware curve is a small, saturating, same-LLC-only tax. The model
paths are growing rank/LLC utilization of 68-route GEMM streams.

## Next action

Completed and rejected: rank LLC versus domain LLC is recorded in
[arm_codex_80c_rank_llc_domain_scope_20260904.md](./arm_codex_80c_rank_llc_domain_scope_20260904.md).
Rank LLC is the entire leftover remote common mode; domain LLC is the
growing local leftover and does not saturate at n≈4. Victim-asymmetric
dilation is recorded in
[arm_codex_80c_victim_asymmetric_dilation_20260904.md](./arm_codex_80c_victim_asymmetric_dilation_20260904.md)
and is also rejected. Next is saturating same-LLC occupancy identification.
