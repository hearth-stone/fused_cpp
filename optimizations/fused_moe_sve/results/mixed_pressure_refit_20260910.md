# Bounded mixed-competition increment, 2026-09-10

## Change and decision

Class M, optional Lab model. Keep the original no-background profile and the conditional pure-competition profile frozen. Add a separate mixed correction for the only measured mixture:19 continuous1T W13 M2 competitors plus19 M120 competitors, M1 W13/W2 foreground, history0–8. The wrapper returns the previous result verbatim for zero or pure competition and rejects other nonzero mixtures. No production/planner defaults, schemas, kernels or native build changed. Rollback consists of the new model/fitter/profile and this record.

For this19+19 mixture only:

```text
T_mixed,s(h) = T_linear_mix,s(h) + a_s + b_s * h
```

Fit two coefficients per stage by least squares on the three session1 mixed medians at histories0/1/8; four coefficients total, six training points. Do not refit the existing24 pure-pressure knots or any no-background parameter. Second-session mixed medians are excluded from fitting. Both sessions were inspected earlier, so this is retrospective repeated-condition validation, not prospective evidence or held-out mixture geometry.

Coefficients in microseconds (h is preceding full M12 panels):

| Stage | a | b per panel |
| --- | ---: | ---: |
| w13 | 45.073026 | 4.285658 |
| w2 | 27.516140 | 2.671842 |

This is an empirical mixed increment, not an identified physical LLC/memory mechanism. Linear history interpolation between measured histories is implemented but not independently validated. Other ratios/counts and other M values are rejected rather than silently generalized.

## Results

| Second-session population | N | Linear-mixture MAE (us) | Corrected MAE (us) | Linear MAPE | Corrected MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 6 | 47.344 | 1.847 | 7.307% | 0.301% |
| All competition | 30 | 12.281 | 3.182 | 2.137% | 0.736% |

All six second-session mixed points:

| Stage | History | Actual (us) | Linear mixture (us) | Corrected (us) |
| --- | ---: | ---: | ---: | ---: |
| w13 | 0 | 874.470 | 830.030 | 875.103 |
| w13 | 1 | 873.220 | 819.245 | 868.604 |
| w13 | 8 | 849.300 | 769.185 | 848.543 |
| w2 | 0 | 466.100 | 439.060 | 466.576 |
| w2 | 1 | 459.860 | 432.265 | 462.453 |
| w2 | 8 | 448.700 | 397.800 | 446.691 |

Mixed maximum absolute error drops from80.115 to4.616us. All-competition maximum remains14.820us, now a pure-background case whose prediction was deliberately not modified. All no-background and pure-competition predictions and fields remain exactly unchanged. The existing no-background discrepancy against the factorial data is still present.

These results support retaining a bounded19+19 correction, not treating the mixed response as solved generally. The next independent evidence would be newly collected data at an intermediate history and different mixing ratios/counts; this turn does not run a new native benchmark. The active planner remains unchanged.

## Reproduction and evidence

Inputs: `tmp/kernel_history_state_20260910/fitted/model.json`, `tmp/conditional_pressure_refit_20260910/model.json`, and both `tmp/kernel_history_pressure_20260910/session*.jsonl`. See the [factorial record](kernel_history_pressure_20260910.md) for source/binary identity and protocol: CPU316/NUMA3, SVE256 BF16 H4096/F512, full owner stripes `(1,0,0,1,1)`, W13 B8MiB/W2 B4MiB, ordinary allocations without explicit HugeTLB,5 warmup/31 measured rounds per condition. No raw samples are pooled across conditions.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_mixed_pressure.py   --baseline tmp/kernel_history_state_20260910/fitted/model.json   --pressure tmp/conditional_pressure_refit_20260910/model.json   --sessions tmp/kernel_history_pressure_20260910/session1.jsonl tmp/kernel_history_pressure_20260910/session2.jsonl   --output-dir tmp/mixed_pressure_refit_20260910
.venv/bin/pytest -q tests/test_moe_mixed_pressure.py tests/test_moe_conditional_pressure.py
```

Use a fresh output directory for reruns. Model/report/source snapshots and pre-edit status are retained under `tmp/mixed_pressure_refit_20260910/`. Generated profile records immutable baseline/pure-profile and input hashes. Seven focused tests pass, including exact unmixed preservation, mixed arithmetic, unsupported mixture rejection, and frozen-profile identity checks. Ruff and diff whitespace checks pass. No commit/push or production adoption is claimed.
