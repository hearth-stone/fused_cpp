# Arm-codex 80C absolute-pressure joint fit

Date: 2026-09-04.

## Decision

Reject the joint $(\beta,\alpha_g)$ candidate. Do not freeze it, do not replace
frozen v8, do not read holdout, and do not open VND/LNS.

The count sweep is a valid identification experiment. The two scalars in the
existing default-off domain-injection and gather-coupling structure cannot
jointly match isolated-relative slowdown and same-minus-cross contrast. The
unidentified quantity is the **scope and saturation of DRAM contention**, not
another numerical pair.

## Locked split

Fit only:

| Artifact | SHA256 |
| --- | --- |
| frozen v8 `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json` | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| `tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json` | `76321dbba93e95089824104c487ebef7beb6dacb411909e87b0a75d44165cc02` |
| `tmp/moe_gather_absolute_pressure_fit_20260904.json` | `a9adb55381f99a1b2c2db934c692130d6ec28c07e26eadcda650070bb0b8abb1` |

Same-family validation, never used to choose parameters:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_gather_absolute_pressure_repeat_20260904.json` | `b7d35dcda1f12e0c1afeccd72b45c2df42c9e84c895c5ae1b52ea9afcf06f625` |

Rejected search dump, not a freeze:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_absolute_pressure_candidate_20260904.json` | `016e2d35ac3caa2dc8687035b3eb32e81823ae9cb10f22ebd24a534005e0b82c` |
| `tmp/moe_absolute_pressure_fit_report_20260904.json` | `a1cb20a104d4a3173e2ad5ad51709790a71bd3bdde4bb6b4478bda27a574eb25` |

Holdout SHA256 values listed in `LOCKED_HOLDOUT_SHA256` were not read. The
report field `holdout_read` is `false`.

Command:

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_absolute_pressure_calibration.py \
  --base-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --phase-fit tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json \
  --pressure-fit tmp/moe_gather_absolute_pressure_fit_20260904.json \
  --pressure-validation tmp/moe_gather_absolute_pressure_repeat_20260904.json \
  --output-calibration tmp/moe_absolute_pressure_candidate_20260904.json \
  --output-report tmp/moe_absolute_pressure_fit_report_20260904.json
```

The fitter does not call `_fit_domain_and_gather_coupling` from
`fit_phase_reaccount_calibration.py`. Phase floor/scale is replayed, not
refit. Wide-team pressure, narrow-team correction, and whole-expert
fixed/route overheads are identity/zero. Loss is
`absolute_mae + contrast_mae`. Split modes enter the absolute family only.
`effective_traffic_multiplier` is effective coupling, not a measured byte
ratio. The coarse grid is $\beta\in[0.20,2.00]$ step 0.05 and
$\alpha_g\in[0.25,8.00]$ step 0.25; the fine grid is a neighborhood of the
coarse best. The upper bound 8 is intentional: expanding it toward 22.26 is
forbidden.

## Search result

| Quantity | Value |
| --- | ---: |
| coarse best | $\beta=0.80$, $\alpha_g=0.25$, joint MAE $0.694\,\mathrm{ms}$ |
| fine best | $\beta=0.78$, $\alpha_g=0.25$, joint MAE $0.689\,\mathrm{ms}$ |
| fit absolute / contrast / joint MAE | $0.549$ / $0.140$ / $0.689\,\mathrm{ms}$ |
| session-2 absolute / contrast / joint MAE | $0.549$ / $0.171$ / $0.719\,\mathrm{ms}$ |
| on search boundary | yes, $\alpha_g=0.25$ |
| near-optima (loss $\le 1.05 L^\star$) | 739 |
| $\beta$ span ratio in near set | 2.82 |
| $\alpha_g$ span ratio in near set | 11.0 |
| identifiable | false |
| rejected contrast-only $(0.787, 22.26)$ recovered | false |
| session-2 contrast bias | all positive, mean $0.171\,\mathrm{ms}$, systematic |
| session-2 absolute bias | mixed sign, not systematic |
| `replace_frozen_v8` | false |

Hardware facts used by the loss are in
[arm_codex_80c_absolute_pressure_20260904.md](./arm_codex_80c_absolute_pressure_20260904.md).
Same-LLC isolated-relative head slowdown saturates near four 68-route 1T
streams. Cross-LLC remains near zero. Split looks like local.

## Fit residuals

Isolated-relative median span, session 1, milliseconds. Residual is
measured minus predicted.

| Mode | Measured | Predicted | Residual |
| --- | ---: | ---: | ---: |
| same-LLC head n1/n2/n4/n8/n15 | 0.144 / 0.198 / 0.291 / 0.299 / 0.312 | 0.082 / 0.389 / 0.729 / 0.844 / 1.057 | +0.061 / -0.191 / -0.438 / -0.545 / -0.744 |
| cross-LLC head n1/n2/n4/n8/n15 | 0.008 / 0.008 / 0.011 / 0.033 / 0.043 | 0.082 / 0.389 / 0.729 / 0.840 / 0.893 | -0.075 / -0.381 / -0.718 / -0.807 / -0.850 |
| same-LLC after-1 n15 | 0.283 | 1.578 | -1.296 |
| cross-LLC after-1 n15 | 0.036 | 1.334 | -1.299 |
| same-minus-cross head n1/n4/n15 | 0.136 / 0.280 / 0.269 | 0.000 / 0.000 / 0.164 | +0.136 / +0.280 / +0.105 |

The model isolated head is $0.645\,\mathrm{ms}$ versus hardware
$0.556\,\mathrm{ms}$. That leftover phase-floor error is secondary; it is
not the source of the 0.9 ms concurrent overprediction.

Session 2 keeps the same predicted spans. Local absolute medians move by
about $0.02\,\mathrm{ms}$; remote-minus-isolated changes sign inside the
isolated session shift. Contrast residuals stay positive on every count and
phase, mean $0.171\,\mathrm{ms}$.

## Why $(\beta,\alpha_g)$ cannot close this

Offline scores of the same session-1 DAGs:

| Setting | same-LLC head n15 abs | cross-LLC head n15 abs | head contrast n1/n4/n15 |
| --- | ---: | ---: | ---: |
| hardware | +0.312 | +0.043 | 0.136 / 0.280 / 0.269 |
| joint $\beta=0.78$, $\alpha_g=0.25$ | +1.057 | +0.893 | 0 / 0 / 0.164 |
| domain off, $\alpha_g=0.25$ | +0.897 | +0.893 | 0 / 0 / 0.004 |
| domain off, $\alpha_g=1.00$ | +0.897 | +0.893 | 0 / 0 / 0.004 |
| $\beta=2.00$, $\alpha_g=0.25$ | +0.897 | +0.893 | 0 / 0 / 0.004 |
| $\beta=0.20$, $\alpha_g=0.25$ | +5.381 | +0.893 | 0.081 / 0.284 / 4.488 |

Gather coupling in $[0.25,1]$ does not move the n15 head prediction. A
loose domain cap is identical to domain-off. The common-mode
$+0.08/+0.73/+0.89\,\mathrm{ms}$ at $n=1/4/15$ is rank `dram_bytes` sharing
among overlapping 68-route W13/W2 streams. Domain injection can add local
excess only on top of that common mode. Tightening $\beta$ to recover
contrast then explodes the local absolute.

Hardware saturates near four local streams. The rank/domain utilization
keeps growing through $n=8$ and $n=15$. Split is local on hardware because
four LLC7 streams already sit on the plateau; the model has no such
plateau at the joint point.

## Unidentified physics

Do not add an empirical residual. The missing degree of freedom is not a
third scalar on the same offered-rate / equal-share cap:

1. Remote 68-route W13/W2 streams barely dilate a local 1-route 1T victim,
   but the event allocator applies the shared rank DRAM curve to every
   concurrent task.
2. Local victim slowdown saturates near four same-LLC streams rather than
   tracking utilization to $n=15$.
3. Because (1) already overstates remote and common-mode pressure, domain
   injection cannot match a $\approx 0.3\,\mathrm{ms}$ local plateau and
   $\approx 0$ remote at once.

## Next action

Completed and rejected: rank DRAM vs domain-only and rank LLC vs domain LLC
are recorded in
[arm_codex_80c_rank_dram_domain_scope_20260904.md](./arm_codex_80c_rank_dram_domain_scope_20260904.md)
and
[arm_codex_80c_rank_llc_domain_scope_20260904.md](./arm_codex_80c_rank_llc_domain_scope_20260904.md).
Do not add domain-only DRAM or a rank-LLC-off switch. Victim-asymmetric
dilation is recorded in
[arm_codex_80c_victim_asymmetric_dilation_20260904.md](./arm_codex_80c_victim_asymmetric_dilation_20260904.md)
and is also rejected. Next is saturating same-LLC occupancy identification.
