# Arm-codex 80C victim-asymmetric dilation ablation

Date: 2026-09-04.

## Decision

Do not add a new default-off structure. The 1-route 1T victim is isolated
transfer-bound (`W13` DRAM $0.243\,\mathrm{ms}$ versus GEMM $0.189\,\mathrm{ms}$),
so it inherits the cohort DRAM/LLC dilation of overlapping 68-route streams.
Giving each task only its own offered rate over capacity removes that
inheritance: predicted local/remote n15 fall to $+0.001/+0\,\mathrm{ms}$.
Hardware still has a saturating same-LLC tax of $+0.312\,\mathrm{ms}$. Existing
service curves cannot identify that occupancy tax without a new parameter.

Keep frozen v8, keep holdout unread, and keep VND/LNS closed. Do not fit an
additive residual on this count sweep.

## Locked split

The same fit-only artifacts as the rejected joint fit and the DRAM/LLC
scope ablations. Session 2 is compared after scoring, never used to choose
an arm.

| Artifact | SHA256 |
| --- | --- |
| frozen v8 `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json` | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json` | `76321dbba93e95089824104c487ebef7beb6dacb411909e87b0a75d44165cc02` |
| `tmp/moe_gather_absolute_pressure_fit_20260904.json` | `a9adb55381f99a1b2c2db934c692130d6ec28c07e26eadcda650070bb0b8abb1` |
| `tmp/moe_gather_absolute_pressure_repeat_20260904.json` | `b7d35dcda1f12e0c1afeccd72b45c2df42c9e84c895c5ae1b52ea9afcf06f625` |
| `tmp/moe_victim_asymmetric_dilation_ablation_20260904.json` | `a95744288098b0cef535d2b6f509accf97a7c8fce5da361aba9e17f91002255c` |

Command:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/ablate_victim_asymmetric_dilation.py \
  --base-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --phase-fit tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json \
  --pressure-fit tmp/moe_gather_absolute_pressure_fit_20260904.json \
  --pressure-validation tmp/moe_gather_absolute_pressure_repeat_20260904.json \
  --output-report tmp/moe_victim_asymmetric_dilation_ablation_20260904.json
```

No parameter search. Domain DRAM injection stays off. Rank DRAM and rank LLC
stay enabled; this ablation changes **who inherits cohort dilation**, not
which capacity curve is used.

Predeclared pass for adding a structure, all required on one arm:

- remote n15 head absolute $\le 0.08\,\mathrm{ms}$
- n4 same-minus-cross $\ge 0.10\,\mathrm{ms}$
- same-LLC n15 minus n4 $\le 0.05\,\mathrm{ms}$

Capping peer count at four is forbidden as an addable arm: it would impose
the plateau by construction on the same sweep that defined the gate.

## Isolated bottleneck

| Expert | W13 cold GEMM | W13 cold DRAM | Isolated body |
| --- | ---: | ---: | --- |
| 1-route 1T victim | $0.189\,\mathrm{ms}$ | $0.243\,\mathrm{ms}$ | transfer-bound |
| 68-route 1T aggressor | $1.134\,\mathrm{ms}$ | $0.243\,\mathrm{ms}$ | GEMM-bound |

Cold packed-B traffic is almost the same ($8.39$ versus $8.50\,\mathrm{MiB}$
LLC). The victim is a short transfer-bound B stream; the aggressor is a
long GEMM stream with the same B footprint.

## Arm results

Isolated-relative head medians, milliseconds. Hardware from session 1.

| Arm | same n1 / n4 / n15 | cross n1 / n4 / n15 | n15$-$n4 same | n4 contrast | fit joint MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardware | 0.144 / 0.291 / 0.312 | 0.008 / 0.011 / 0.043 | 0.021 | 0.280 | — |
| symmetric | 0.082 / 0.729 / 0.897 | 0.082 / 0.729 / 0.893 | 0.168 | 0.000 | 0.712 |
| compute_bound_skip | 0.082 / 0.729 / 0.897 | 0.082 / 0.729 / 0.893 | 0.168 | 0.000 | 0.712 |
| same_llc_peers | 0.082 / 0.729 / 0.897 | 0.000 / 0.000 / 0.000 | 0.168 | 0.729 | 0.685 |
| own_demand | 0.000 / 0.000 / 0.001 | 0.000 / 0.000 / 0.000 | 0.001 | 0.000 | 0.376 |

`compute_bound_skip` matches symmetric because the victim is already
transfer-bound: skipping dilation on GEMM-bound phases does not protect it.
`same_llc_peers` zeros remote (rank DRAM/LLC no longer import cross-domain
68-route bytes) and leaves the growing local curve unchanged.
`own_demand` sets each task's DRAM/L2/LLC scale to
$\max(1, \mathrm{own\ offered}/\mathrm{capacity})$. One cold-B stream is
$\approx 24\,\mathrm{GB/s}$ versus rank DRAM $166\,\mathrm{GB/s}$ and domain
LLC $216\,\mathrm{GB/s}$, so own utilization is $<1$ and the victim becomes
isolated.

Event `resources.*.dilation` still records the cohort pressure; the rewritten
quantity is the victim **phase** dilation ($3.08 \to 1.00$ under
`own_demand`).

## Why this is not one missing resource

1. Symmetric sharing is the source of the $\approx 0.9\,\mathrm{ms}$ common
   mode: a transfer-bound 1-route victim inherits 15 concurrent 68-route
   DRAM/LLC offered rates.
2. Restricting inheritance to same-LLC peers fixes remote and keeps the
   wrong local growth through n=15.
3. Own-demand removes the overprediction and also removes the hardware
   local tax, so contrast disappears.
4. A count cap at n=4 would force the plateau on this same sweep and is
   not an identifiable occupancy law.

The hardware curve remains a small, saturating, same-LLC-only tax on a
1-route 1T victim. The model paths are either cohort byte-rate utilization
or isolated.

## Next action

Completed and rejected: aggressor-$M$ occupancy is recorded in
[arm_codex_80c_aggressor_m_occupancy_20260904.md](./arm_codex_80c_aggressor_m_occupancy_20260904.md).
The leftover tax falls from $+0.660\,\mathrm{ms}$ at $M=1$ to
$+0.290\,\mathrm{ms}$ at $M=68$. Do not add a structure.
