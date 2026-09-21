# Full-panel and history-conditioned tail cost, 2026-09-09

The bounded split improves isolated real-expert W13 estimates and synthetic tail timing. It does not explain the large real p11 M13 contention penalty. This is a Class M Lab experiment, with no production dispatch, calibration, planner or pruning change.

## Model and frozen protocol

For `M = 12q + r`, estimate `T(M,p) = q F(p) + R(r,q,p)`, with zero tail for `r=0`. `F` is the median of all full-panel timings in session1 M192 at each pressure endpoint. `R` is the session1 median final-panel time at preceding B scan counts 0,1,2,8,15, interpolated linearly in scan count. Count4 is excluded from fitting. Counts0/1/2 use the primary session; counts8/15 use the extension. Both costs interpolate linearly between 0 and38 independent same-LLC M12 background workers. This pressure is a task-count proxy, not a measured LLC service rate. No extra pressure multiplier is applied afterward.

Support is W13 K4096/N1024, one thread, M1–192 with remainder0/1/5/11. Unsupported remainders (including real M31 remainder7) fail explicitly. The query's scan count represents preceding complete panels in the same invocation; prewarmed external history, varying competitors, W2, other shapes and thread teams remain unsupported. The measured nonmonotonic M1 history is retained; there is no enforced cold-to-hot monotonicity.

All source datasets were previously inspected. The fit uses session1 only, is serialized before evaluating session2 or the real expert report, and is not subsequently tuned. This is retrospective validation, not a prospective holdout. Source hashes and the frozen parameter hash are in `tmp/full_tail_cost_20260909/report.json`; parameters are in `frozen.json`. Source collection protocol and native correctness checks are documented in [panel_transition_20260909.md](panel_transition_20260909.md) and [expert_layer_split_20260909.md](expert_layer_split_20260909.md).

The no-history ablation uses the **same full-panel cost** but always uses `R(r,0,p)` for the tail. It is not the previous production/v8 model.

## Synthetic validation

Second-session medians of31 measured rounds, five warmups, same native binary and independent random seeds. Full timing is the enclosing timer, not a sum of individual medians. Primary/extension repeated shapes are separate observations and not independent unique geometries.

| Endpoint / subset | Observations | No-history MAE (us) | History MAE (us) |
|---|---:|---:|---:|
| Tail itself, session2 tail shapes |48|18.036|3.710|
| Enclosing time, session2 tail shapes |48|14.295|11.614|
| Enclosing time, all session2 normal cells |66|14.782|12.832|
| Unfitted scan-count4, both sessions |12|14.099|16.494|

Tail MAPE is3.074%→0.576%, maximum absolute tail error61.19→13.65us. Overall time MAPE is0.372%→0.285%. Tail improvement does not ensure overall improvement because full-block bias and tail error can cancel. The count4 result is a retained counterexample. Six full-block shapes at the unfitted pressure16 have MAE28.23us / MAPE0.386%, max78.89us. Tail pressure interpolation at intermediate counts is **not validated** by those full-block cells.

The fitted full costs are1233.56us at0 peers and1254.99us at38 peers. Pooling all M192 positions omits the small position dependence of full blocks, which accumulates for larger M.

## Real expert transfer

No real-expert observations fit parameters. The previous estimator column is the isolated W13 estimate from the frozen core-pressure experiment. Matched CPU316 p11 expert-isolation data, second session:

| M | Actual isolated (ms) | Previous estimate (ms) | Split estimate (ms) | Previous / split absolute error (us) |
|---:|---:|---:|---:|---:|
|13|1.564|1.339|1.564|224.48 /0.12|
|17|1.807|1.702|1.841|105.26 /34.17|
|35|3.632|3.403|3.700|228.46 /68.53|
|65|6.625|6.239|6.777|385.38 /152.04|

All four improve in both sessions; the near-exact session2 M13 match should not be treated as expected precision (session1 error25.02us). Larger M retains accumulated full-block/protocol error. M31 is excluded rather than interpolating across unmeasured remainder kernels.

For joint execution, no fixed task count is inferred from the variable mixed-expert trace. Instead, compare actual p11 W13 with the model's range over0–38 synthetic competitors:

| M | Synthetic model range (ms) | Actual p11, session2 (ms) |
|---:|---:|---:|
|13|1.564–1.940|2.897|
|17|1.841–2.011|1.977|
|35|3.700–3.765|4.000|
|65|6.777–7.011|6.790|

These are sensitivity ranges, **not physical bounds or confidence intervals** for real competitors. M13 exceeds the upper modeled endpoint by0.957ms and M35 by0.235ms. M17/M65 falling inside does not validate their joint predictions. Background geometry, pressure time course and B lifecycle differ between protocols; this experiment cannot identify which causes the remaining residual, or assign that residual entirely to the tail.

Keep the full/tail split as an experimental representation. Do not integrate this table into the full-MoE predictor before validating intermediate tail pressure and real joint per-panel timing. No new hardware measurements or whole-forward improvement claim are made here.

## Reproduction and review

```sh
.venv/bin/pytest -q tests/test_moe_full_tail_cost.py tests/test_moe_panel_transition.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/full_tail_cost.py --output tmp/full_tail_cost_20260909
.venv/bin/ruff check optimizations/fused_moe_sve/benchmarks/full_tail_cost.py tests/test_moe_full_tail_cost.py
```

Use a fresh output directory when rerunning; existing frozen results are not overwritten. Ten focused tests passed, Ruff passed, and `git diff --check` passed. Review covered fitting/evaluation separation, unsupported-tail rejection, exact-block zero tail, counterfactual history accounting and source hashes. Rollback boundary is the new Lab script/tests/manifest record/documentation; existing model code is unchanged by this experiment.
