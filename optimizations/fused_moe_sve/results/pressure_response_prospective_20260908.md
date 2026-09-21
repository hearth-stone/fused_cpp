# Prospective frozen-response validation on new pressure levels

## Result

**The frozen linear response generalizes well to four newly measured reader
counts at the calibrated M/widths, but does not strictly beat the isolated
baseline in every group.** Eleven of12 shape/session gates pass. No refitting,
threshold reselection, or victim/supply reanchoring was performed.

Primary evaluation includes only new levels3/6/10/14. Each M aggregate contains
24 medians: three widths ×four new levels ×two independent sessions.

| M | Predictor | MAPE | Maximum absolute percentage time error |
|---:|---|---:|---:|
| 1 | Frozen isolated constant | 9.709% | 25.733% |
| 1 | Frozen selected response | **.749%** | **2.347%** |
| 12 | Frozen isolated constant | .424% | 1.303% |
| 12 | Frozen selected response | **.194%** | **.445%** |

This is time-prediction accuracy, not measured execution speedup. Error is
relative to total kernel time, not the incremental slowdown. All primary
evaluation pressures are at or below their first-session fitted maxima.

## Per-group results and failure

The prespecified gate is, separately for each shape/session on the four new
levels: MAPE≤2%, maximum≤5%, and MAPE strictly below the isolated constant.
The0 and16 reader controls are excluded from this gate and its averages.

| M | T | Session | Isolated MAPE | Response MAPE | Response max | Gate |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | 1 | 1 | 18.507% | 1.537% | 2.347% | pass |
| 1 | 1 | 2 | 19.125% | 1.108% | 1.927% | pass |
| 1 | 2 | 1 | 5.492% | .727% | 1.701% | pass |
| 1 | 2 | 2 | 5.781% | .483% | .794% | pass |
| 1 | 4 | 1 | 4.650% | .368% | .846% | pass |
| 1 | 4 | 2 | 4.702% | .270% | .509% | pass |
| 12 | 1 | 1 | .267% | .322% | .441% | **fail: baseline comparison** |
| 12 | 1 | 2 | .436% | .169% | .293% | pass |
| 12 | 2 | 1 | .371% | .250% | .445% | pass |
| 12 | 2 | 2 | .403% | .110% | .208% | pass |
| 12 | 4 | 1 | .469% | .168% | .389% | pass |
| 12 | 4 | 2 | .600% | .149% | .274% | pass |

The worst primary prediction is session1 M1/1T,3 readers: actual324.86us,
predicted332.48us (+2.347%). The isolated prediction304.89us is −6.147%.
M12/1T's failed group misses only the strict baseline superiority clause;
the absolute accuracy limits pass. This does not authorize retroactively
changing the gate or selecting the isolated predictor using these test results.

## Separately retained controls

Zero-reader real-kernel times are not used to reanchor the model. The largest
zero-reader response error is1.194% (M1/2T,session2). All other zero-reader
errors are smaller; M12 errors are≤.160%. Thus there is modest baseline drift,
but no large fresh-anchor correction is hidden in the reported scores.

The maximum16-reader response error is1.812% (M1/1T,session2). The2T16-reader
supply pressure is13.539%/13.668% versus the fitted maximum13.133%; those four
M/width/session control points are flagged as above-fit-range in the output.
They are not mixed into the unseen-level primary evaluation. Do not interpret
them as broad extrapolation validation.

## Protocol, identity, and scope

This is newly collected data after model freezing, not a second evaluation of
the original two sessions. It tests **new reader counts at known M/width and
the same synthetic background implementation**, not unseen M, changed access
patterns, mixed experts, another machine, or a plan-visible pressure estimator.

- Arm-codex-internal, NUMA3 memory and CPUs240–319; controller240,
  victims304+, background288–303 in the same LLC domain. Preflight load
  average.01/.00/.08; no substantial load observed.
- Same existing native binary as calibration, no rebuild or native changes:
  SHA256`aec4ec8050ecb32910ec92fc40b64ea049edba5e886891a13c3e39df45c7e8b6`.
- Frozen model SHA256 before and after:
  `b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`.
- W13 H4096/F512,K4096/N1024,BF16,SVE256,Ntile16,full owner stripes;
  M1/M12×1/2/4T. B8MiB total,8/4/2MiB per owner. Production JIT degree5
  SiLU, unchanged constant1/64 inputs and numerical reference0x3f3b.
- New levels0/3/6/10/14/16:36 real cells,18 independent B-only controls,
  18 empty controls per round. Each session:5 discarded warmups and31 recorded
  rounds, randomized order; seeds29808/39808. Total4,464 recorded cells,
  including2,232 numerically checked real calls.
- One process/persistent allocations per session; independent allocations
  between sessions; persistent touched output,256MiB scrub each cell,4-copy B
  rotation; background starts5ms before victim release. Scrub is not proof of
  exclusively DRAM-sourced loads. Per-reader32MiB footprint and cache effects
  remain part of the intervention.
- Identical core PMU and48 DDRC events to calibration; same timer/gate semantics.
  Each worker resets/reads PMU independently per cell; complete grid, numerical
  scope, worker/counter coverage and running ratios≥.99 are checked. DDRC
  counters are not fitted model inputs; D is separately measured B-only time.
- Existing ordinary allocations/compiler/page regime retained; no system page
  policy changes, HugeTLB setup or production dependency changes. No fresh
  page-residency audit or instrumentation-free full-session repeat.

Prediction remains T_hat=T0*(1+a*max(0,D/D0−1)), with a,T0,D0 from the original
first-session fit. The old selected linear form is used everywhere, including
the failing M12 group. All per-cell errors are retained, not filtered for sign,
pressure ordering, or fit-range convenience.

## Reproduction and retained evidence

Remote artifacts:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/pressure_response_prospective_20260908/`.
Contains exact runner/helper snapshots, frozen_fit.json, correctness.jsonl,
pmu_smoke.jsonl and both formal sessions. Binary and build provenance remain
in the prior`tmp/memory_pressure_curve_20260908/`directory. Original snapshots
were not overwritten. Local raw copies and evaluation.json use the new suffix.

```sh
# Remote, reuse the verified binary. Second session: session2.jsonl, seed39808.
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/pressure_response_prospective_20260908/bench_phase_supply.py \
  --binary tmp/memory_pressure_curve_20260908/phase_supply_native \
  --output tmp/pressure_response_prospective_20260908/session1.jsonl \
  --pressure-curve --pressure-levels 0 3 6 10 14 16 --seed 29808
# Local, never call the fit CLI on these validation sessions.
.venv/bin/python optimizations/fused_moe_sve/benchmarks/replay_pressure_response.py \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --sessions tmp/pressure_response_prospective_20260908/session1.jsonl tmp/pressure_response_prospective_20260908/session2.jsonl \
  --output tmp/pressure_response_prospective_20260908/evaluation.json
.venv/bin/python -m pytest -q tests/test_moe_pressure_response.py tests/test_moe_pressure_curve.py tests/test_moe_phase_supply.py tests/test_moe_kernel_response.py
```

Class E/M Lab validation only. Changed Python runner/grid metadata to accept
explicit valid pressure levels; old defaults and input compatibility remain.
Raw loader follows the declared grid; the original curve summary rejects a
different grid rather than silently applying its original decision logic.
The evaluation helper accepts declared levels, while the original fitting
routine still rejects incomplete/different training grids. New replay entrypoint
rejects grid/binary/round-count mismatches and checks the frozen file is unchanged.

39 focused tests passed, including new-grid evaluation, invalid levels, immutable
anchors and identity mismatch checks; both72-cell no-PMU/PMU smokes passed before
formal timing. The old fitted profile was reproduced exactly without overwriting
the original file. No new native compile or production/full-operator tests were
needed or run. No production model, calibration, pruning, API or default changed.

Decision: retain as a conditional-response validation reference. The M1 result
is encouraging on new data; M12 needs a restrained claim because its incremental
cost is small and one group does not outperform the constant baseline. Further
work should validate changed supply contexts or real mixed kernels, not claim
the plan→pressure mapping has been solved. No commit requested or created.
