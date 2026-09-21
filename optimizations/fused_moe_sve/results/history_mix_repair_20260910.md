# Sequential pure-history and mixed-increment repair, 2026-09-10

## Scope and fitting order

Class M Lab change requested by the user: repair competition at histories2–4 first, then the remaining mixed increment. The no-background six-family model and original pure-pressure profile are frozen, as are pure history0/1/8 predictions. Production kernels, public schemas, build/defaults and active planner are unchanged. The new optional wrapper/profile is the rollback boundary.

Data are the previously collected prospective grid under `tmp/mixed_pressure_prospective_20260910/`. In this new fit, that grid is no longer entirely prospective evidence: session1 is calibration and session2 is retrospective same-condition holdout, both already inspected during diagnosis. No new native collection occurs in this turn.

Step1: for each stage and pure background at38 competitors, fit correction knots at h2/h4 from session1 measured time minus original prediction; h0/h1/h8 corrections are exactly zero. Eight new values total. Piecewise-linear history interpolation uses0/1/2/4/8. Under a mixture with small fraction f, combine pure corrections as `(1-f)*c_large(h)+f*c_small(h)`.

Step2: freeze those pure corrections. At histories0/1/2/4/8 and small counts8/19/30, set each mixed residual to session1 measured time minus the repaired pure-mixture prediction. Thirty residual nodes total, with zero mixed residual at pure endpoints0/38. Interpolate in fraction between0,8/38,19/38,30/38,1 and then in history. This replaces the symmetric4f(1-f)(a+b*h) correction, permitting asymmetric, nonmonotonic ratio/history behavior.

```text
P_repaired(h,f) = P_original(h,f) + (1-f)c_large(h) + f*c_small(h)
T_full(h,f) = P_repaired(h,f) + R(h,f)
R(h,0) = R(h,1) = 0
```

The repair applies only at total38, M1, history0–8. Zero-background calls return the existing model verbatim for its full supported row/history domain. Other pure counts pass through to the old model without repair; non38 mixed requests are rejected. This is not a continuous count-generalization model. Interior histories/fractions are interpolated but independently unvalidated.

This is deliberately a table-based repair:8 pure correction nodes and30 mixed residual nodes exactly match their38 calibration observations. Training accuracy is not evidence of generalization. The old independent mixed function is retained as the `prior` comparator; `pure_repaired` retains that old mixed function after step1, isolating each step's effect.

## Second-session results

Each table entry is MAE in us, followed by MAPE. All populations below use session2 only and overlap.

| Population | N | Before repair | Pure-history repair only | Both repairs |
| --- | ---: | --- | --- | --- |
| Pure h2/h4 | 8 | 32.249 / 6.734% | 4.216 / 0.662% | 4.216 / 0.662% |
| Mixed h2/h4 | 12 | 40.002 / 6.824% | 16.651 / 2.553% | 2.179 / 0.434% |
| All mixed | 30 | 21.805 / 3.742% | 12.464 / 2.033% | 3.731 / 0.630% |
| All competition | 50 | 19.070 / 3.468% | 8.980 / 1.472% | 3.740 / 0.630% |
| No background | 10 | 20.290 / 9.709% | 20.290 / 9.709% | 20.290 / 9.709% |

The pure intermediate-history error falls first, without using mixed data. The remaining mixed error falls further after fitting only the mixed residual. All50 competitive points have final MAE3.740us/MAPE0.630%; maximum error12.140us. Original no-background predictions remain unchanged, including their9.709% mismatch against this second-session harness.

Not every case improves. Against the pre-repair extension,30 competitive points improve,12 are unchanged and8 worsen. Against the pure-only repair, the mixed step improves21 points, leaves20 pure points unchanged and worsens9 mixed points. Largest final error is W2/h8/30 M2+8 M120: actual480.780us, prior480.330us, repaired468.640us. Error rises0.450→12.140us because the fitted table follows session1's468.640us observation. This is a meaningful repeated-session/overfitting limitation, not a universally superior prediction.

The formerly worst W13/h4/8+30 case changes from728.400us to792.550us against795.330us actual in session2, reducing error66.930→2.780us. The two-layer repair resolves this measured systematic error while retaining the regression above.

## Validation and limitations

Seven focused tests pass for zero-pressure preservation, original pure knots, pure/mixed layer separation, mixed arithmetic, frozen baseline identity and unsupported domain handling. Ruff and `git diff --check` pass. Real-profile checks confirm all original no-background fields and pure h0/h1/h8 times remain unchanged. Original profiles are hash-checked and not rewritten.

Results support a bounded calibrated reference at measured nodes. They do not validate unmeasured h3/h6, additional ratios, other total counts, other row families, background placements, real expert inputs or planner performance. Smoothing/generalization requires separately held-out geometry or new data; low second-session error alone is insufficient. Do not reuse this fit's training nodes as a future prospective test.

## Reproduction and retention

Inputs: frozen `tmp/kernel_history_state_20260910/fitted/model.json`, `tmp/conditional_pressure_refit_20260910/model.json`, `tmp/mixed_pressure_refit_20260910/model.json`, and both `tmp/mixed_pressure_prospective_20260910/session*.jsonl`. See [prospective measurement record](mixed_pressure_prospective_20260910.md) for exact native identity and collection protocol: CPU316/NUMA3,1T M1 W13/W2, H4096/F512, BF16 SVE256, N tile16, full owner stripes `(1,0,0,1,1)`, W13 B8MiB/W2 B4MiB, no explicit HugeTLB,5 warmup/31 measured randomized rounds per condition.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_history_mix_repair.py   --baseline tmp/kernel_history_state_20260910/fitted/model.json   --pressure tmp/conditional_pressure_refit_20260910/model.json   --mixed tmp/mixed_pressure_refit_20260910/model.json   --sessions tmp/mixed_pressure_prospective_20260910/session1.jsonl tmp/mixed_pressure_prospective_20260910/session2.jsonl   --output-dir tmp/history_mix_repair_20260910
.venv/bin/pytest -q tests/test_moe_history_mix_repair.py tests/test_moe_conditional_pressure.py
```

Use a fresh output directory on rerun. Source snapshots, generated profile/report, input hashes and before/after workspace status are retained under `tmp/history_mix_repair_20260910/`. No production regression/native rerun, commit or push is claimed.
