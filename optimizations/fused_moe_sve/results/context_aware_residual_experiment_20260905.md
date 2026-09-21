# Context-aware residual: development replay only (2026-09-05)

## Scope and predeclaration

Change class: Lab cost-model diagnostic.  This experiment neither changes the
frozen v8 calibration nor installs a planner scorer, pruning rule, acceptance
rule, or Production default.

Before the replay, the target was declared as frozen-v8 absolute-error and
within-parent ranking diagnostics.  The fixed ridge parameter is `alpha=10.0`;
the predeclared feature sets are `structure`, `temporal`, and `interaction`.
The split is leave-one-unique-parent-lineage-out (LOPO), with no shared
`state_hash` or `group_id` between fit and evaluation.  Session 1 supplies the
development labels.  Session 2 is a same-plan repeat diagnostic only, never an
unseen-plan generalization sample.  The intercept-only control below was added
after the first nine fits specifically to separate a common absolute offset
from a context-feature effect; it was not part of the initial predeclaration.

No hardware was run.  The only execution was a local offline replay:

```text
PYTHONPATH=.:src .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/prepare_context_residual_replay.py \
  --frontier tmp/moe_partial_order_vnd_20260904/high_skew_template_lns_frontier.json \
  --model-artifact tmp/moe_partial_order_vnd_20260904/high_skew_template_lns_model.json \
  --session1 tmp/moe_partial_order_vnd_20260904/high_skew_lns_session1.json \
  --session2 tmp/moe_partial_order_vnd_20260904/high_skew_lns_session2.json \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output tmp/context_aware_residual_20260905/development_replay_dataset.json
```

It completed in 9.5 seconds on the local development machine.  This is not a
hardware performance measurement.

## Artifact audit and dataset

The supplied frontier has 106 plans.  Every canonical payload rehashed to its
recorded `state_hash`, and both historical sessions contain exactly the same
106 hash keys.  All plan tasks have `w13_window_tiles == w2_window_tiles == 0`,
so this initial replay is explicitly limited to full-owner-stripe plans.

For 95 states with a stored model `event_ns`, the recomputed frozen-v8 baseline
matched exactly (maximum absolute drift: 0 ns).  The other 11 states are all
deliberately retained `candidate_worse_sentinel` rows and have no historical
`event_ns`; their newly recomputed values are labelled replay values rather
than historical measurements.

The model/frontier identity chain contains frozen calibration
`7928ba…aad3`, extension `dd554e…943f`, and frontier content hash
`ed4461…f7`.  The session files use a stale frontier pathname, but that digest
matches the supplied frontier content.  They do not themselves record the
calibration hash, model-source hash, shape key, protocol key, page policy,
NUMA/memory placement, build command, or original scorer source revision.  The
replay therefore records the current context-model source hash separately and
is a provenance-labelled development dataset, not a reconstruction of an
independently sealed experiment.

Unique parent groups are `4115ab…d175` (33), `514ced…b767` (35), and
`b43120…b70c` (37).  One state (`e7e669…d77c8`) belongs to two parents and is
excluded from all strict folds, leaving 105 rows.  Sampling is strongly
selected rather than i.i.d.: 92 incomparable-frontier candidates, 11
purpose-selected worse sentinels, and 3 anchors.  This restricts every result
below to a development diagnostic.

## Results

Each cell is `S1 corrected MAPE / S2 repeat corrected MAPE` in percent.  The
frozen-v8 baseline S1 MAPE is 9.321%, 9.134%, and 8.687% for holdout groups A,
B, and C respectively.

| Fixed diagnostic | Holdout A | Holdout B | Holdout C |
| --- | ---: | ---: | ---: |
| Intercept only | 1.789 / 1.890 | 1.959 / 2.009 | 1.546 / 1.387 |
| Structure | 1.546 / 1.641 | 1.620 / 1.653 | 1.282 / 1.103 |
| Temporal | 1.450 / 1.502 | 1.427 / 1.433 | 1.220 / 1.054 |
| Interaction | 1.469 / 1.547 | 1.464 / 1.484 | 1.275 / 1.120 |

The intercept-only control removes most of the absolute error.  Consequently,
the modest further reduction from context features is not evidence that a
context-aware model is ready for selection or adoption.

For S1, the following are `pairwise MAE / P90` in percentage points and
direction correctness among pairs whose hardware median margin exceeds 2%:

| Diagnostic | A | B | C |
| --- | --- | --- | --- |
| Intercept only | 2.704 / 3.497; 251/254 | 2.910 / 5.624; 246/258 | 3.196 / 3.992; 334/336 |
| Structure | 2.282 / 2.965; 245/254 | 2.375 / 4.681; 255/258 | 2.693 / 3.704; 336/336 |
| Temporal | 2.124 / 3.226; 249/254 | 2.069 / 4.739; 258/258 | 2.414 / 3.906; 324/336 |
| Interaction | 2.064 / 3.194; 249/254 | 2.049 / 5.086; 258/258 | 2.284 / 3.764; 321/336 |

Every group has more than 16 plans.  All four diagnostics obtained top-16
measured-best recall of 1/1 with 0.000% regret in every group.  This is weak
evidence because the same selected frontier supplies the labels.  Feature-range
extrapolation is also material: structure has 10/5/29 S1 out-of-range rows for
A/B/C; temporal has 11/11/31; interaction has 11/11/33.  Intercept-only has no
feature range concept.

## Decision and residual risk

The replay supports only a Lab follow-up: retain the adapter and the frozen-v8
feature extraction for future, separately sampled development data.  It does
not support changing v8, fitting to a sealed holdout, hardware-budget expansion,
automatic acceptance, dominance pruning, planner ranking adoption, or a
performance claim.  A meaningful next test needs a prospectively sampled,
identity-complete plan set with an untouched route/session holdout and recorded
machine/page-policy provenance.
