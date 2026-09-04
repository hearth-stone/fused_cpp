# Arm-codex 80C phase re-accounting result

Date: 2026-09-03.

## Decision

The isolated phase re-accounting is accepted as a useful analytical structure:
gather, W13, and W2 are calibrated independently, and small-stage service
floors replace whole-expert total residual hiding. The frozen candidate as a
complete contention model is rejected. It must not replace frozen v8 or enter
VND/LNS because its cross-LLC-fitted gather coupling catastrophically
overpredicts absolute background pressure and produces false dominance on real
traces.

## Locked fit and holdout split

Fit inputs only:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_isolated_phase_reaccount_all_widths_fit_20260903.json` | `76321dbba93e95089824104c487ebef7beb6dacb411909e87b0a75d44165cc02` |
| `tmp/moe_gather_injection_overlap_cross_llc_20260904.json` | `aea4736bb6a3b97d60dbaaee8aabf63b2e3837150259f9ebf7c8f5f180ebea1c` |

Fit-family validation, never used to select parameters:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_isolated_phase_reaccount_all_widths_repeat_20260903.json` | `418d53be563bac61cbe83d969c82ee483dc04d7f24740b009f28caaeb2762cc5` |
| `tmp/moe_gather_injection_overlap_cross_llc_repeat_20260904.json` | `f111549d33fe3cc221a9cd932c385cda6c9036d4e9cfedcd0b9b2d49485681dd` |

All old `M={1,2,5,6,12}` route-context artifacts and the three measured
real-trace shortlist artifacts were locked by SHA256 and read only after the
candidate calibration was written. The builder refuses any training input
whose digest is registered as holdout.

Generated ignored-workspace outputs:

| Artifact | SHA256 |
| --- | --- |
| candidate calibration | `9f8981b5a213dd9b6547dcb4b3e3f43bb9589b7b4435064574ef920bd7019377` |
| fit report | `9dbb798d3f889605a8432fafebcc483eecf069181d9b817b249bcfcfd432d866` |
| high-skew measured-state rescore | `045c8c7fc081d38948afacf64ccaa85b63a2683f373bba240f08458e09a0b4c8` |
| median measured-state rescore | `81d9f2e1196ff60b4224b2ec6b0f240bbe8abe9b53daffff9965ed24b7b97542` |
| uniformish measured-state rescore | `6996d208f23875276f7e20eedeff4f5a97d70d3ffe42a2e65e244d21065a07a8` |
| final holdout report | `a720c2d5caddcb149c33e8f5ea9cc3acb1e024c4787457213d147118172d57f8` |

## Isolated phase calibration

Two Arm NUMA3 sessions used disjoint fit routes
`{3,4,7,8,10,16,24,48,68,600,1800}` and all supported widths
`{1,2,4,8,16,32,40,80}`. Every point used five warmups, 31 randomized trace
rounds, four measured packed copies, and a dedicated disjoint 17-expert scrub
copy before each sample. The old holdout routes `{1,2,5,6,12}` are rejected by
the benchmark.

The first parser incorrectly summed per-thread trace records into core-ms for
multi-thread stages. Those artifacts were excluded. The final parser uses the
stage envelope from minimum worker start to maximum worker end.

The fitted phase form is:

```text
T_gather(M,T) = max(gather_min[T], gather_row[T] * ceil(M/T))
T_stage(M,T)  = max(stage_min[stage,T],
                    stage_scale[stage,T] * T_physical(stage,M,T))
```

Whole-expert fixed/per-route residuals and both old contention corrections are
reset to identity. Existing panel/range restart costs and physical service
curves remain.

Independent phase-session validation:

| Metric | MAPE | Maximum absolute relative error |
| --- | ---: | ---: |
| gather | 16.13% | 78.75% |
| W13 | 3.49% | 28.85% |
| W2 | 4.70% | 20.02% |
| total consistency only | 4.38% | 23.52% |

Gather's high relative maximum is on a tens-of-microseconds small-M/wide-team
point; it is not used to hide W13/W2 error.

On the old route holdout's isolated modes, which were not used for fitting:

| Metric | MAPE | Maximum absolute relative error |
| --- | ---: | ---: |
| gather | 23.75% | 57.15% |
| W13 | 1.46% | 2.65% |
| W2 | 9.18% | 14.78% |
| total | 3.69% | 6.16% |

This is the successful part of the experiment: isolated stage attribution is
substantially more faithful than the old pre-W13 whole-expert residual.

## Domain cap and gather/stream coupling

With old `wide_team_pressure` and `narrow_team_contention_correction` empty, the
cross-LLC after-1 W13-only contrast fits a domain-capacity scale of `0.787`.
The validation residual is `+0.0086 ms`.

Before a gather coupling, head contrast is predicted as `0.154 ms` versus
`0.244/0.248 ms` measured in fit/repeat. The residual is stable and exceeds
20 us and lies outside the fit paired interval, so the predeclared rule selects
a gather/stream coupling. Fitting only the cross-LLC head contrast produces an
effective gather traffic multiplier of `22.26`. It closes fit head residual to approximately zero and leaves repeat
head residual `+0.0040 ms`.

The multiplier is not a measured byte ratio. The holdout demonstrates that the
local-minus-remote contrast alone cannot identify absolute gather pressure: a
large common-mode error cancels in the contrast and drives the parameter to an
unphysical value.

## Pure holdout result

### Route/context absolute error

Across 25 points (`M={1,2,5,6,12}` times five contexts):

| Model | MAPE | Median absolute error | P90 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| frozen v8 | 21.72% | 17.33% | 46.65% | 57.84% |
| frozen phase-reaccount candidate | 273.97% | 301.80% | 642.94% | 689.42% |

The isolated rows remain accurate, but background rows are overpredicted by
multiple times. This alone rejects the candidate.

Among the 50 within-route mode pairs, hardware resolves 48 by paired P10/P90.
The candidate gets 46 directions correct and two wrong, so false dominance is
nonzero even though measured-best top-1 recall across the five route cases is
5/5.

### Three real traces

The current 64-per-operator sample did not reproduce every old measured state
hash. To avoid selection bias, the replay regenerated the complete legal
neighborhood for the exact frozen baseline and the union of the recorded
critical/random expert IDs, then scored only the old hardware shortlist hashes.
All 13/16/12 candidate hashes were recovered for high-skew/median/uniformish.

| Trace | Spearman | Measured best in model top-8 | Resolvable pairs | Correct | False dominance |
| --- | ---: | --- | ---: | ---: | ---: |
| high-skew | 0.345 | no | 43 | 33 | 10 |
| median | 0.075 | yes | 9 | 3 | 6 |
| uniformish | 0.368 | yes | 0 | 0 | 0 |

Top-16 retains the measured best on all traces, but that does not compensate
for the high-skew top-8 miss or sixteen false-dominance relations across the
two resolvable traces.

## Root cause and next probe

Phase time and domain locality are separately identifiable. Absolute gather
pressure is not identifiable from one local-minus-remote contrast because the
rank/common-mode response cancels. The next probe must vary gather aggressor
count and placement while measuring both:

1. target absolute slowdown versus isolated;
2. local-minus-remote contrast;
3. gather overlap core-ms and transition time;
4. at least one independent rank-wide background level.

Fit domain capacity and gather traffic jointly to absolute plus contrast rows,
not contrast alone. Keep the accepted phase floor/scale structure, but do not
persist the rejected `0.787/22.26` parameters or enable schema-v11 in planning.
