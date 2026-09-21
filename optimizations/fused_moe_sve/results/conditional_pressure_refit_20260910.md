# Frozen no-background baseline, conditional competition refit, 2026-09-10

## Decision and scope

The user requested prioritizing the existing no-background model and refitting competition. This class M Lab change adds a separate conditional competition wrapper and fitter; it does not edit the frozen six-family baseline profile or its implementation. Public contracts, production schema/kernel/build and active planner defaults are unchanged. Rollback is limited to the new optional model/fitter and generated residual profile. Validation is L1 invariant tests plus retrospective L3 model evidence; no production adoption is claimed.

The new pressure data only cover M1 W13/W2. Zero-background requests retain the existing rows1–12/history0–15 domain; nonzero competition is explicitly rejected outside M1, history0–8 or total competitors0–38. Interior interpolation is implemented but unmeasured histories/counts are not independently validated. Backgrounds remain the measured continuous1T W13 M2/M120 workloads; arbitrary expert shapes or multithread teams are outside scope.

## Formula and fitting split

Let B_s(r,h) be the original frozen no-background prediction. The new estimate is:

```text
T_s(1,h,n_small,n_large) = B_s(1,h) + Delta_s(h,n_small,n_large)
Delta_s(h,0,0) = 0 exactly
n = n_small + n_large
Delta_s(h,n_small,n_large)
  = (n_small/n) d_small,s(h,n) + (n_large/n) d_large,s(h,n), for n>0
```

Each pure-background residual d is piecewise linear in count through0/16/38 and then piecewise linear in history through0/1/8. The n=0 knot is exactly zero. The two measured count knots are `session1 measured pure-competition time minus frozen B(h)`. Residuals are effective timing corrections, not identified physical memory costs. They may absorb cross-protocol baseline mismatch, and cannot repair the zero-background baseline by construction.

Fit24 pure-background observations (2 stages × 3 histories × 2 types × 2 nonzero counts). This is a24-knot interpolation model and exactly reproduces these training nodes; training accuracy is not generalization evidence. Hold out the6 session1 mixed points and all30 session2 competitive points. All12 mixed points across both sessions remain excluded from fitting. These data were inspected in earlier work, so this is **retrospective** holdout, not a prospective experiment.

The independent comparator uses the same frozen baseline and only cold-history pressure residuals: `B(h) * [1 + Delta(0,p)/B(0)]`. It has8 fitted pure count knots versus24 in the conditional model. This isolates the value of history-dependent pressure within the frozen-baseline choice; it is not numerically identical to the previous experiment's comparator using freshly measured no-background values. Both models use the same linear type-mixture assumption.

The competitive return marks the old C/D/overlap component decomposition invalid because only the total residual was fitted; it must not be read as a new physical decomposition. No-background fields are delegated unchanged. A SHA256 check rejects substituting a different baseline profile.

## Retrospective validation

| Population | N | Independent MAE (us) | Conditional MAE (us) | Independent MAPE | Conditional MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All held-out competition | 36 | 63.573 | 18.023 | 12.674% | 2.991% |
| Session2 competition | 30 | 59.758 | 12.281 | 12.507% | 2.137% |
| Mixed backgrounds, both sessions | 12 | 83.594 | 47.038 | 13.707% | 7.283% |
| No background, both sessions | 12 | 18.647 | 18.647 | 8.279% | 8.279% |

The zero-background predictions are unchanged, including the existing8.279% mismatch against the later factorial harness. Do not hide this by refitting a no-background intercept. Held-out competitive maximum absolute error falls from211.356 to80.115us. Mixed competition remains systematically underestimated: all12 mixed errors are negative, with mean error−47.038us. Worst is session2 history8/W13/38 mixed: actual849.300us versus769.185us predicted, error−80.115us. Linear mixing of pure-competitor responses is insufficient; the model does not claim this remaining interaction is solved.

Retain as a Lab reference with the no-background baseline frozen. The refit improves the measured competitive holdout, but does not validate the other five row families, arbitrary history/count interpolation, dynamic pressure or planner-selected real plans. No new native measurements or production regression were run: the retained factorial datasets supply the evidence.

## Reproduction and validation

Baseline: `tmp/kernel_history_state_20260910/fitted/model.json`.
Baseline SHA256: `a820b080f39bb3766b440d7e7ba0d76a87c703d2ec931958ef0fd82fd55730a9`.

Input data: `tmp/kernel_history_pressure_20260910/session1.jsonl` and `session2.jsonl`; see [factorial protocol](kernel_history_pressure_20260910.md). CPU316, NUMA3, SVE256 BF16 H4096/F512, full owner stripes `(1,0,0,1,1)`, W13 B8MiB/W2 B4MiB, no explicit HugeTLB, 5 warmup/31 measured randomized rounds per condition. Use per-condition medians, not raw-call pooling.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_conditional_pressure.py   --baseline tmp/kernel_history_state_20260910/fitted/model.json   --sessions tmp/kernel_history_pressure_20260910/session1.jsonl tmp/kernel_history_pressure_20260910/session2.jsonl   --output-dir tmp/conditional_pressure_refit_20260910
.venv/bin/pytest -q tests/test_moe_conditional_pressure.py tests/test_moe_kernel_history.py
```

Use a new output directory when rerunning. Artifacts and source snapshots are retained under `tmp/conditional_pressure_refit_20260910/`; raw input paths and hashes are recorded in the generated profile. The10 tests pass, covering all exact-row zero-background preservation, residual/mixed arithmetic, zero-pressure limit, domain rejection, baseline identity and existing decomposition invariants. Ruff and `git diff --check` pass. No new dependencies, commits or remote writes to repositories.
