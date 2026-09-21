# Temporal-order differential diagnosis, 2026-09-10

## Finding

The high-skew ranking error has two connected layers: the model makes the16T lanes the predicted bottleneck, while hardware finishes on1T lanes; it also misses much of the late small-M slowdown on those1T lanes. The measured LPT regression is not explained by a slow last expert alone or a scheduler gap.

This is a read-only diagnosis of the two paired traces from [the common-selector model comparison](mean_model_compare_20260910.md). No fitting, model change, production change or new hardware run. Same H4096/F512/E256/T2048/top-k6,80-core ARM NUMA3, BF16 SVE256, N tile16, full stripes, early merge off. Reuse both31-round sessions and their previously verified raw identities/output correctness.

## The geometry really is the same

Both plans have4×16T +16×1T. Every expert retains its route count, CPU/team assignment and width. Ten lanes retain their own order; the other ten reverse their expert lists:16T lane starts16/48 and1T starts65/67/69/71/73/75/77/79. Logical lane start plus240 gives its physical first CPU. The change therefore alters both wide-team and narrow-team temporal order; the observations alone cannot attribute the entire regression to one group.

`anchor` is reverse-odd, selected by the old model; `new_selected` is LPT, selected by the new model. Paired LPT-minus-reverse-odd compute deltas are+1.57838/+1.57571ms. The new model predicts-0.06954ms; the old model predicts+0.09204ms. Old-model direction happens to agree, but neither predicts the magnitude.

## Layer 1: wrong bottleneck hides the order effect

The new model predicts expert230 on the16T lane beginning32 is last for both orders, at31.244/31.174ms. Actual LPT finishes on expert239 on the1T lane beginning69 in all62 measured calls. Reverse-odd finishes primarily on1T lanes68/70, occasionally69; one session1 call finishes on16T lane0.

Session2 lane-completion medians, ms:

| Lane start / width | Reverse-odd measured | LPT measured | Reverse-odd predicted | LPT predicted |
|---|---:|---:|---:|---:|
|32 /16T|27.505|27.823|31.244|31.174|
|68 /1T, own order unchanged|27.769|29.241|27.062|27.341|
|69 /1T, order reversed|27.721|29.375|27.102|27.289|
|76 /1T, own order unchanged|26.317|28.349|26.338|26.365|

Thus correcting a1T duration alone may not change the selected maximum at all: the overestimated16T path remains above it. A descriptive endpoint-substitution check confirms this masking. Replacing only modeled1T lane endpoints by measured endpoints still leaves the predicted order delta at-0.06954ms. Replacing only16T endpoints yields+0.387/+0.430ms, still far short of the measured+1.58ms. Even taking the maximum only over modeled1T lanes predicts just+0.23942ms. These are endpoint oracles, not physically resimulated plans or forecasts of a proposed fix.

The new model's total isolated lane costs are balanced near23.44–23.51ms and identical across orders. The placed16T W13/W2 totals are higher than measured, but these traces do not separate isolated-cost error from contention-scale error; there is no matched isolated16T intervention in this diagnosis.

## Layer 2: the missing time is mostly GEMM under a different late workload mix

Every1T lane is slower under LPT, including the eight1T lanes whose own expert sequence did not change. Session2 increases in lane-completion medians range roughly1.38–2.22ms across all16 narrow lanes. This is evidence of changed concurrent execution conditions rather than a purely within-lane ordering cost.

Additive per-call accounting on the1T lanes separates gather, W13, W2 and the remaining interval gaps. The table reports means of paired component differences so the components sum exactly; it is not a sum of independently taken medians and is not a decomposition of the global median makespan difference.

| Lane | Session | Gather delta | W13 delta | W2 delta | Gap delta | Lane-end delta |
|---|---:|---:|---:|---:|---:|---:|
|68, unchanged order|1|-0.164|+1.123|+0.609|-0.005|+1.563|
|68, unchanged order|2|-0.199|+0.971|+0.661|+0.013|+1.446|
|69, reversed order|1|+0.123|+1.384|+0.252|-0.001|+1.758|
|69, reversed order|2|+0.123|+1.319|+0.240|+0.014|+1.696|
|76, unchanged order|1|-0.081|+1.409|+0.648|+0.003|+1.978|
|76, unchanged order|2|-0.098|+1.319|+0.652|+0.010|+1.884|

Times are milliseconds. Gather often becomes faster; residual gaps are small. The missing differential is chiefly W13 and then W2. Model `operator` phases are inherited analytical costs and must not be equated to the measured gather envelope.

Small-M tasks around23–26ms show particularly large differences. Session2 W13 duration medians:

| Expert / M | Own lane order | Reverse-odd measured | LPT measured | Reverse-odd predicted | LPT predicted |
|---|---|---:|---:|---:|---:|
|13 /2|unchanged|0.942|1.531|0.767|0.780|
|169 /3|unchanged|0.736|1.554|0.767|0.780|
|31 /2|reversed|0.779|1.737|0.648|0.779|

Session1 repeats the pattern: expert13 is0.922→1.540ms, expert169 is0.735→1.566ms, expert31 is0.784→1.728ms. The final LPT expert239/M1 itself is faster, with session2 W13 decreasing1.232→0.586ms and W2 decreasing0.356→0.149ms. Its late completion reflects the accumulated preceding queue time, not a uniquely slow final-M1 execution.

## Why core count alone misses this window

Integrate per-worker GEMM intervals over each target W13 interval, excluding the target core and counting only its physical40-core LLC domain. These are time-averaged active workers, not independently measured memory-request rates. Session2:

| Target | Reverse-odd peer cores | LPT peer cores | Reverse-odd M≤4 peer cores | LPT M≤4 peer cores |
|---|---:|---:|---:|---:|
|expert13/M2|38.22|37.32|7.22|22.66|
|expert169/M3|38.78|37.24|6.91|24.88|
|expert31/M2|38.21|34.99|4.96|29.19|

The total active count decreases slightly while W13 slows substantially; the M≤4 portion increases. Session1 exhibits the same count/mix pattern. The frozen core-pressure response depends on peer GEMM cores and does not separately weight a core executing a large-M reused-B phase versus a small-M task. Small-M traffic intensity and synchronized phase transitions are plausible missing distinctions, but this trace does not identify LLC request throughput, DDR bandwidth, cache misses or frequency as the physical cause.

There is also a timing error in the modeled competitor mix. Under LPT, the16T lane48 reaches expert25/M164 at23.547ms and its first M1 task at25.359ms; the model places these transitions at27.685 and29.548ms. Lane16 reaches expert220/M215 at23.376ms and M1 at25.777ms; predictions are27.210 and29.666ms. The large-task phase lasts about4ms too long in the modeled timeline, so around the victim's24–26ms slowdown the model still places those wide teams in large-M execution. For expert31/M2, modeled overlap contains24 large-M peer cores while actual overlap has a median5.65. Therefore adding a workload-mix pressure term without first checking the wide-team timeline could still use the wrong competitor mix.

These overlap features are descriptive and endogenous to victim duration. They locate a repeatable missing distinction; they do not isolate the causal contributions of wide-team reversal, narrow-team reversal or individual memory resources. A controlled two-factor wide-only/narrow-only reversal would separate those effects if pursued next.

## Validation and reproduction

### Existing isolated1T comparison

The current frozen model also underestimates isolated small-M GEMM stages. Rechecked all17 source identities in `tmp/small_t_isolated_20260908/validated/report.json`; use its `model_ms` frozen-v8 column, not the separate unadopted affine candidate. Core-pressure preserves this isolated basis. Historical same-shape isolated results, microseconds:

| M / stage | Current isolated prediction | Measured S1 / S2 | Underprediction S1 / S2 | Underprediction percentage S1 / S2 |
|---|---:|---:|---:|---:|
|1 /W13|243.478|312.500 /320.240|69.022 /76.762|22.09% /23.97%|
|1 /W2|121.739|151.020 /153.130|29.281 /31.391|19.39% /20.50%|
|3 /W13|378.132|449.790 /408.320|71.658 /30.188|15.93% /7.39%|
|3 /W2|189.066|222.970 /206.190|33.904 /17.124|15.21% /8.30%|
|4 /W13|378.132|444.200 /410.860|66.068 /32.728|14.87% /7.97%|
|4 /W2|189.066|218.260 /204.840|29.194 /15.774|13.38% /7.70%|

These use the same H4096/F512, BF16 SVE256 N16 workspace protocol and two31-round sessions. M3/M4 target experts16/9 on physical CPU308 in the historical high-skew isolation grid, not current expert169 on CPU316 or expert127 on CPU309. M1 is historical expert231. There is no M2 isolated row in this verified dataset. Thus this is a same-shape base-error scale comparison, not an exact matched decomposition for current experts13/31/169. No new measurements were collected for this lookup.

The subsequent M<12 review covers all available1T real-expert points M1/3/4/7/8/10 in the same verified artifact. Current `CorePressureModel.predict_expert(M,1)` reproduces each archived physical-stage prediction exactly; the unadopted affine correction is not used. Additional points, microseconds:

| M / stage | Current prediction | Measured S1 / S2 | Signed error percentage S1 / S2 |
|---|---:|---:|---:|
|7 /W13|756.265|763.260 /749.670|-0.92% /+0.88%|
|7 /W2|378.132|390.900 /386.410|-3.27% /-2.14%|
|8 /W13|756.265|763.480 /749.850|-0.95% /+0.86%|
|8 /W2|378.132|390.030 /385.200|-3.05% /-1.83%|
|10 /W13|945.331|1040.470 /1044.990|-9.14% /-9.54%|
|10 /W2|472.666|530.700 /532.510|-10.94% /-11.24%|

Across those six real-expert M values and two sessions, W13 MAE is47.370us, MAPE9.542%, signed bias-45.202us and maximum absolute error99.659us; W2 MAE26.213us, MAPE9.745%, bias-26.213us and maximum absolute error59.844us. These are stage-only statistics, not whole-expert `T_iso` or gather accuracy. Equal weighting is per measured M/session, not workload frequency. Accuracy is nonmonotonic: M7/8 are close, whereas M1 and M10 have appreciable errors.

Separate cold-first synthetic one-panel probes in `tmp/panel_transition_20260909/report.json` and `tmp/w2_panel_isolated_20260909/report.json` cover M5/11. With zero peers, no prepass, panel instrumentation enabled and contiguous W2 layout, W13 M5 measures633.52/632.37us versus prediction567.199us (10.47%/10.31% low); W13 M11 measures1230.97/1233.02us versus1134.397us (7.85%/8.00% low). W2 M5 measures319.71/320.56us versus283.599us (11.29%/11.53% low); W2 M11 measures641.74/642.92us versus567.199us (11.62%/11.78% low). Do not pool these with real-expert measurements: synthetic W2 M1 itself differs at183.26/186.12us compared with real-expert151.02/153.13us. The reviewed datasets still lack isolated M2/6/9 points, and real-expert M5/11 coverage is missing. This review does not claim complete M1–11 validation.

For context, current LPT expert169/M3 W13 has predicted0.780ms versus measured1.554ms (about0.774ms low), while these historical isolated M3 errors are0.030–0.072ms. The base error is real but much smaller in the available observations. Do not assign an exact percentage of the current residual to contention by subtracting different experts/sessions. Exact current-target isolation would be required for that causal layer split. The separate real M13/M17 isolation in `tmp/early_merge_off_20260909/report.json` likewise has base error: session2 W13 underprediction0.249/0.110ms and W2 underprediction0.131/0.038ms, respectively.

Artifacts: `tmp/order_diagnosis_20260910/{analyze.py,pressure.py,report.json,pressure.json,models.json,layouts.json,parsed1.json,parsed2.json}`. Source measurement remains `tmp/mean_model_compare_20260910/high_skew/`; both gzip traces were decompressed locally and their raw SHA256 rechecked. No remote execution was needed. The existing parser verifies task/worker completeness and call/pair order, then the diagnosis verifies source compute medians, CPU/team/expert membership and per-call1T component conservation.

From the repository root, after decompressing the source `session1.trace.gz`/`session2.trace.gz` into the diagnosis directory:

```bash
.venv/bin/python tmp/order_diagnosis_20260910/analyze.py
.venv/bin/python tmp/order_diagnosis_20260910/pressure.py
```

Both scripts completed. All124 measured call envelopes reproduce the source medians. Both sessions have the same geometry and repeated stage/mix pattern. Existing benchmark output/merge checks remain valid; this diagnosis does not assert a new optimization gain. Endpoint substitution is explicitly descriptive; no oracle values enter the active planner. Current experimental baseline and production code are unchanged.
