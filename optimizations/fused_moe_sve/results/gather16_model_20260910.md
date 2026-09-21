# Explicit16T gather stage on the no-team baseline, 2026-09-10

## Change and scope

The user requested adding gather to the model. The new optional Lab adapter `planner_gather16.CorePressureModel` extends the no-wide/narrow-residual baseline with an explicit `gather_pack_a` phase before each16T expert's W13. All original phases remain byte-for-field equal after the new prefix;1T and other widths are unchanged to avoid double counting their existing operator residual. No production/schema/native change or full planner rerun is made.

This class M change uses an empirical **measured-condition delay**. Its duration was observed during joint execution, so it is not multiplied by a second competition factor. The phase has no modeled additional shared-memory traffic. Consequently it does not yet model gather's bandwidth pressure on other tasks or its response to different competition. The event simulator does account for the delay's effect on phase overlap. Do not interpret this as a physically complete gather service model.

Supported added-stage domain:16T, H4096, M1–1945 (the measured route-count range). Other widths preserve the old model; outside-domain16T route counts are rejected. Original no-team calibration, GEMM costs and frozen M1 pressure profile remain unchanged.

## Formula and calibration

```text
worker_rows = ceil(M/16)
T_gather16(M) = 4.220us + 6.012680us * (worker_rows - 1)
```

Fit only the baseline-selected high-skew plan's session1:63 per-expert16T gather medians. The first-worker-row cost is the median among worker_rows=1; the additional-row slope is a nonnegative least-squares fit after fixing that first-row cost. This gives two coefficients. No lane-total or whole-plan error is used to fit them.

Hold out the other plan/session combinations:195 per-expert observations, including129 observations from session2. These are retrospective repeated workloads and overlap in experts/routes; they are not new prospective shapes. Held-out gather MAE5.924us/MAPE14.162%; session2 MAE6.074us/MAPE14.100%. Maximum held-out gather error109.283us. Absolute-error sum divided by actual-time sum is9.813%. The simple row model is not exact for every expert even though accumulated lane cost is close.

Predicted cumulative16T gather is about0.99–1.01ms per lane; actual mean per lane is about0.98–0.99ms in the retained traces. This replaces the missing named stage with an explicit cost, not an opaque GEMM multiplier.

## Fixed-plan results

All times are ms; actual measurements are the same two sessions used in the parent experiment, not new hardware runs.

| Fixed plan | Before gather | After gather | Actual S1 / S2 |
| --- | ---: | ---: | ---: |
| Baseline-selected |26.558|26.894|29.301 /29.347|
| M1-adapter-selected |26.284|26.952|28.654 /28.628|

Mean signed lane endpoint error, prediction minus actual, over both sessions:

| Plan | Width | Before gather (ms) | After gather (ms) |
| --- | ---: | ---: | ---: |
| anchor | 1 | -2.397 | -2.124 |
| anchor | 16 | -1.737 | -0.910 |
| repaired | 1 | -2.396 | -2.121 |
| repaired | 16 | -1.816 | -1.000 |

Adding gather reduces16T mean low bias by about0.82ms. It also changes predicted1T overlap, reducing1T low bias by about0.27ms even though no1T phase was modified. Whole-plan predicted time therefore does not simply increase by each lane's gather sum. Both plans are still predicted too fast and lane32 remains predicted critical after gather, unlike hardware's1T critical lanes.

Relative plan ranking is again incorrect: after gather the model prefers the baseline-selected plan by0.058ms, while hardware prefers the M1-adapter-selected plan by0.64–0.71ms. The earlier no-team ranking agreement was not robust proof of an accurate model. This negative result is retained; explicit gather is needed for accounting but does not solve the remaining real-plan small-M response error.

## Validation, retention and next boundary

Four focused tests pass across the gather/no-team suites: worker-row boundary, prefix position, unchanged underlying phases and other widths, no added duplicate DRAM demand, all team residual factors1, preserved calibration and explicit route-domain rejection. Ruff and diff checks pass. Current no-gather reconstruction matches the previous fixed-plan results. No new hardware, cold-search, untraced E2E, production parity or performance improvement is claimed.

The next work remains the real-plan1T/M1–4 stage response and old operator residual accounting, plus remaining16T gaps/W13 error. Gather traffic/competition response and other widths need independent evidence before broader adoption. No compensating width penalty was restored.

Artifacts and source snapshots: `tmp/gather16_model_20260910/`. Inputs: `tmp/planner_repaired_pressure_20260910/input.json`, its `high_skew/frontier.json`, `analysis1.json` and `analysis2.json`; original trace/bitwise identities remain in [the planner comparison](planner_repaired_pressure_20260910.md). Same CPU240–319/NUMA3, H4096/F512, BF16 SVE256,4×16T+16×1T, full stripes, early mergeoff, two31-call sessions and unchanged page policy. Source/calibration hashes are retained by this and the parent profile. Rollback is selecting the preserved no-gather adapter.

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_gather16.py   --source tmp/planner_repaired_pressure_20260910 --output-dir tmp/gather16_model_20260910
.venv/bin/pytest -q tests/test_moe_gather16_model.py tests/test_moe_no_team_residual.py
```

Use a fresh output directory on rerun. The new adapter accepts an explicit gather_profile path, defaulting to the retained artifact above. No commit/push or production default change.
