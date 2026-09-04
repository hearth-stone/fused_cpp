# CPU MoE cost-model re-accounting handoff

Date: 2026-09-03

This document is the handoff for continuing the CPU MoE analytical cost-model
re-accounting work. It records the current repository state, evidence, rejected
fits, immutable fit/holdout split, and the next implementation sequence. Do not
infer missing details from branch names: the relevant work is currently
uncommitted and the Arm machine was updated by direct file sync.

Update: the six-step re-accounting pass described below has now been executed.
The isolated phase structure passed, but the complete candidate failed holdout
and was rejected. The authoritative final result is
`arm_codex_80c_phase_reaccount_20260903.md`; the remaining next action is the
absolute-pressure/aggressor-count probe, not the former Step A.

## 1. User objective

Complete one clean re-accounting pass:

1. Calibrate `gather_pack_a`, W13, and W2 from isolated phase traces instead of
   fitting whole-expert residual time.
2. Fit the per-LLC-domain DRAM injection cap only from the cross-LLC probe.
3. Use the same-domain head/after difference to decide whether a separate
   gather/stream coupling term is still identifiable.
4. Set the old `wide_team_pressure` and
   `narrow_team_contention_correction` to identity before adding only the
   residual physics not explained by the new model.
5. Keep the old route sweep and all three real traces as pure holdout; they must
   never influence fitting or parameter choice.
6. Evaluate absolute error, pairwise direction, top-K recall, and false pruning.

The work is offline/Lab. Do not change Plan V2, native ABI, kernels, production
quick planning, or default dispatch. Do not connect the result to VND/LNS until
all holdout gates pass.

## 2. Repository and workflow constraints

- Repository: `/Users/zhangxu/Codes/vllm-aarch64/fused_cpp`.
- Read the root `AGENTS.md` before editing.
- `.codegraph/` exists: use `codegraph explore` before structural navigation.
- Required skills for code/model changes:
  `../skills/impact-analysis/SKILL.md`, `../skills/test-selector/SKILL.md`, and
  `../skills/code-review-gate/SKILL.md`.
- Model changes are class M; update
  `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md` in the same change.
- Experimental benchmark changes are class E; keep the manifest current.
- Read `docs/agent_benchmark_hygiene.md` and
  `docs/agent_optimization_governance.md` before new measurements.
- Preserve the dirty worktree. Do not revert or reformat unrelated changes.
- Do not commit, push, rebase, or mutate a remote repository unless explicitly
  requested.

## 3. Current implementation state

### 3.1 Analytical model

File: `cpu_moe_schedule_optimization/cost_model/analytic_model.py`.

Current uncommitted identity:

```text
ANALYTIC_MODEL_SCHEMA_VERSION = 10
ANALYTIC_MODEL_NAME = phase_ecm_llc_domain_dram_injection_v8
```

Two optional, default-off calibration structures exist:

1. `GatherPressureCalibration`

   ```text
   T_gather(M,T) = fixed_ns + row_ns * ceil(M/T)
   Q_gather = effective_traffic_multiplier
              * M * H * (input_element_bytes + packed_element_bytes)
   ```

   When enabled, `gather_pack_a` becomes an explicit DRAM-requesting phase.
   The current prototype redistributes the old non-gather whole-expert residual
   proportionally into W13/W2 so the previous isolated total remains close.
   This redistribution is transitional and should be removed/replaced by the
   requested phase-separated calibration.

2. `DramDomainInjectionCalibration`

   ```text
   C_domain(n) = min(
       C_rank_curve(n),
       capacity_scale * C_rank_saturated / number_of_LLC_domains,
   )
   D_domain = max(1, offered_domain / C_domain)
   D_task,dram = max(D_rank, max(D_domain touched by task))
   ```

   The placed event explanation records per-domain injection active threads,
   offered rate, capacity, utilization, and dilation. Enabling this calibration
   requires calibrated LLC domains. Missing fields retain the old rank-only
   path.

The structure has unit tests, but no new machine calibration is frozen or
tracked. The frozen v8 calibration remains authoritative.

### 3.2 New/updated Lab probes

- `optimizations/fused_moe_sve/benchmarks/bench_small_expert_context.py`
  supports `--target-routes` and strict disjoint packed-weight scrub.
- `optimizations/fused_moe_sve/benchmarks/bench_gather_injection_overlap.py`
  compares a fixed one-route target against fifteen 68-route 1T aggressors in
  the same or opposite LLC domain, at lane head and after one one-route delay.
- Tests:
  `tests/test_moe_small_expert_context.py` and
  `tests/test_moe_gather_injection_overlap.py`.

### 3.3 Documentation already updated

- `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`, sections 9.43 and
  9.44, changelog through v1.53.
- `cpu_moe_schedule_optimization/TODO.md`.
- `optimizations/fused_moe_sve/manifest.yaml`.
- `optimizations/fused_moe_sve/results/arm_codex_80c_small_expert_context_20260903.md`.
- `optimizations/fused_moe_sve/results/arm_codex_80c_dram_domain_injection_20260903.md`.

## 4. Machine and calibration identity

- Host alias: `Arm-codex-internal`.
- Remote repository: `/home/zhangxu/codex/fused_cpp`.
- Python: `/home/zhangxu/codex/fused_cpp/.venv/bin/python`.
- NUMA rank: node 3, CPUs `240-319`, bind with
  `taskset -c 240-319 numactl --membind=3`.
- LLC6: CPUs `240-279`.
- LLC7: CPUs `280-319`.
- Target in the locality probe: logical core 64, physical CPU304, LLC7.
- Frozen calibration remote path:
  `bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`.
- Frozen calibration local copy:
  `/tmp/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`.
- Calibration SHA256:
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Shape: BF16 fused-SiLU, `H=4096`, `F=512`, backend N tile 16,
  W13/W2 full owner-stripe windows.

Important topology trap: logical cores 48-62 map to physical CPUs288-302 and
are still in LLC7 with CPU304. The first locality run used that placement and
must not be treated as cross-LLC evidence. True remote placement is logical
cores 0-14, physical CPUs240-254, in LLC6.

## 5. Locked data partition

The next agent must encode this split in the analysis script and reject path
overlap by SHA256. Do not rely on naming conventions alone.

### 5.1 Fit corpus

Allowed for fitting:

1. A **new isolated phase-trace corpus**, not yet collected. It must be a new
   artifact and must report gather/W13/W2 separately. Do not use any old route
   sweep value, including its `isolated_head` field, for fitting.
2. Cross-LLC fit session:

   `tmp/moe_gather_injection_overlap_cross_llc_20260904.json`

   SHA256:
   `aea4736bb6a3b97d60dbaaee8aabf63b2e3837150259f9ebf7c8f5f180ebea1c`.

Recommended internal validation, not parameter tuning:

`tmp/moe_gather_injection_overlap_cross_llc_repeat_20260904.json`

SHA256:
`f111549d33fe3cc221a9cd932c385cda6c9036d4e9cfedcd0b9b2d49485681dd`.

### 5.2 Explicitly excluded artifact

`tmp/moe_gather_injection_overlap_20260904.json`, SHA256
`7a5f048926621f8775b15565aa3118d8d0c9189eda2f56d2e4e8d423aa7697c1`,
is the topology-mistake run where both placements were inside LLC7. It may be
used only as a same-LLC-distance diagnostic, never as cross-LLC fit evidence.

### 5.3 Pure holdout: route/context sweep

Do not tune any parameter after viewing errors on these artifacts:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_small_expert_context_scrub_20260903.json` | `61c0e929ad7a575831f972bdbb2c690df243e028062cdbb10044e66d15867e34` |
| `tmp/moe_small_expert_context_scrub_repeat_20260903.json` | `90f34a8498a2a3ad1abfdae69b21ff4b9ab97f08cef1192dcc62c23bf8c07d4f` |
| `tmp/moe_small_expert_context_scrub_m2_20260903.json` | `be6b83475360c330ba4e053c90f19fc162465228eef835661b692fac569ca1a1` |
| `tmp/moe_small_expert_context_scrub_m5_20260903.json` | `277d3896e5d04bafedbf931edba38c0ff29ea6a6b9ad542ad3ff5d78d261df74` |
| `tmp/moe_small_expert_context_scrub_m6_20260903.json` | `28cf59e56d6c281df3245fcc941e19885496ec6e7df9bb4b7f3634d59e1b4376` |
| `tmp/moe_small_expert_context_scrub_m12_20260903.json` | `81c195d34902e0d9002a59f85d0c8bccfc884a6395bacb22031aa69a5c01b0dd` |

### 5.4 Pure holdout: three real traces

Primary measured shortlist artifacts:

| Trace | Artifact | SHA256 |
| --- | --- | --- |
| high-skew | `tmp/pairwise_high_skew_holdout_20260902.json` | `b23ece71ffecbbbc839747ac4be679b34c02c7998a3737764aeee63964ddb67f` |
| median | `tmp/pairwise_median_holdout_20260902.json` | `fe103f49f816d930b12c021e2ca5d8966bbf0283670c61f245f3464f5d3f92d3` |
| uniformish | `tmp/pairwise_uniformish_holdout_20260902.json` | `bba234ae99508af9fc9f7026b43150e8c6e09f50026049e53f4392a4d908a8d8` |

Existing aggregate pairwise report:

`tmp/moe_pairwise_ordering_holdout_v1.json`, SHA256
`c90f73e727d26a3eeb7f350a84e38f811aef4f69db74c3224b2ada7b23330e15`.

These hardware measurements must not select phase scales, domain capacity, a
gather/stream term, residual buckets, or uncertainty margins. They are final
evaluation only.

## 6. Evidence already established

### 6.1 LLC scrub and cold-weight source

The strict protocol retains four rotating measured packed copies and runs a
dedicated disjoint packed-weight scrub before every sample. Address-filtered Arm
SPE reduced sampled target-W13 L3-hit incidence from about 6.9% before explicit
scrub to 1.7% after scrub. SPE used a 30-cycle minimum-latency filter, so this is
a source gate, not an unbiased byte fraction.

### 6.2 Route-dependent victim sensitivity

Hardware target spans and background deltas:

| Target M | Isolated | 1T-only delta | 16T-only delta | Full delta | After-68 minus full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.633 ms | +0.272 ms | +0.558 ms | +0.811 ms | -0.523 ms |
| 2 | 0.629 ms | +0.302 ms | +0.552 ms | +0.836 ms | -0.553 ms |
| 5 | 0.966 ms | +0.113 ms | +0.392 ms | +0.621 ms | -0.530 ms |
| 6 | 0.941 ms | +0.128 ms | +0.414 ms | +0.655 ms | -0.540 ms |
| 12 | 1.909 ms | +0.022 ms | +0.167 ms | +0.283 ms | -0.202 ms |

The excess is primarily in W13. M1/M2 form one tail bucket, M5/M6 a second,
and 1T-peer sensitivity is nearly gone at M12.

### 6.3 Phase-timing root cause

Representative full-head trace:

- Target W13: approximately `0.558-1.711 ms`.
- First 68-route 1T peer enters W13 near `1.037 ms`.
- First 16T peer enters W13 near `1.438 ms`.
- Before peer W13 begins, peer `gather_pack_a` is already active.

Frozen v8 places width-specific whole-expert residual entirely in a zero-traffic
operator phase before W13. For M1 it predicts approximately:

```text
operator residual  0.158 ms
W13                0.243 ms
W2                 0.122 ms
total               0.524 ms
```

Hardware isolated M1 is approximately:

```text
gather              0.008 ms
W13                 0.415 ms
W2                  0.207 ms
total               0.633 ms
```

The old isolated total is partly right for the wrong phase decomposition, so
its concurrent overlap is wrong.

### 6.4 Cross-LLC locality result

Two independent 31-round sessions:

| Context | Fit median / P10 | Repeat median / P10 |
| --- | ---: | ---: |
| local minus cross-LLC, head | +0.244 / +0.221 ms | +0.248 / +0.230 ms |
| local minus cross-LLC, after-1; 15/15 peers in W13 | +0.225 / +0.212 ms | +0.233 / +0.210 ms |

The locality effect remains when gather overlap is removed. This selects a
per-LLC-domain injection structure as the first missing resource. A separate
gather/stream coupling may remain as a smaller residual.

### 6.5 Rejected stacked fits

1. Gather-only prototype using trace-derived `fixed_ns=5 us`,
   `row_ns=2.9 us`, and an in-sample effective-traffic multiplier near 3:

   - old 25-point absolute-relative-error mean: 21.7%;
   - same-corpus gather prototype: 12.6%;
   - independent M1 repeat five-context MAPE: 17.34%;
   - M1 1T-only and after-68 remained overpredicted by 38.03% and 26.74%.

   This parameter is not frozen because the 12.6% result is in-sample.

2. Domain-cap prototype with existing gather and frozen-v8 residuals:

   - rank-only predicted local-minus-remote contrast: about 0.004/0.004 ms;
   - `capacity_scale=0.76` predicted 0.145/0.235 ms versus hardware
     0.244/0.225 ms;
   - contrast mean absolute error fell from about 0.230 ms to 0.055 ms;
   - old 25-point route/context MAPE worsened from 21.7% to about 28.5%.

   Removing old wide/narrow corrections without phase re-accounting was worse.
   This proves double counting; it does not reject the domain resource.

## 7. Required next implementation

### Step A: collect a new isolated phase fit corpus

Do not fit from the old route-sweep artifacts. Add a dedicated isolated phase
benchmark or a mode in the current benchmark that emits one row per `(M,T)`:

- `gather_pack_a` median/P10/P90;
- W13 median/P10/P90;
- W2 median/P10/P90;
- total span only as a consistency check;
- exact physical CPUs and LLC domain;
- W13/W2 window tiles and ranges;
- scrub identity, packed-copy count, hashes, warmups, runs, and seed.

Recommended disjoint fit routes are values such as
`M={3,4,7,8,10,16,24,48,68}` for 1T and representative medium/long routes for
8T/16T. This leaves old `M={1,2,5,6,12}` as shape holdout while still sampling
the neighboring M12 tail buckets. If the model requires the exact held-out
route to identify a parameter, the parameter is not sufficiently physical.

Use at least two independent sessions. One may fit; the second validates the
phase calibration without retuning.

### Step B: replace whole-expert residual fitting

Create a reproducible builder/analyzer under
`optimizations/fused_moe_sve/benchmarks/`; do not hand-edit calibration JSON.

The candidate calibration should:

- preserve physical service curves, topology, kernel geometry, stage-window
  policy, call setup, and measured panel/range restart costs;
- set width-specific whole-expert `expert_fixed_ns` and `route_ns` to zero
  unless a separately traced non-gather/non-W13/non-W2 phase justifies them;
- fit gather duration only to gather observations;
- fit W13 scale only to W13 observations;
- fit W2 scale only to W2 observations;
- avoid an arbitrary per-route latency lookup table if a physical tail-bucket
  or execution-geometry feature explains the variation;
- output residual tables by `(M,T,stage)` and record training SHA256 values.

The current scalar `w13_scale`/`w2_scale` may be insufficient because M1/M2
have much larger stage residual than M12/68. Prefer a compact calibration keyed
by existing physical geometry, such as effective M12 rows/tail class and team
width, rather than raw route identity.

### Step C: zero old contention residuals before fitting domain injection

For the candidate calibration:

```text
wide_team_pressure = identity / empty
narrow_team_contention_correction = identity / empty
```

Do this before fitting `DramDomainInjectionCalibration`. Do not stack the new
resource on frozen-v8 residuals and then compensate with another correction.

### Step D: fit domain injection only on cross-LLC contrasts

Fit `capacity_scale` using the first valid cross-LLC artifact. The objective
should use paired local-minus-remote deltas, not absolute target spans, so
isolated phase error cannot leak into the domain parameter.

Evaluate the selected value unchanged on the independent cross-LLC repeat.
Report head and after-1 separately. Do not average them before inspecting the
residual.

### Step E: decide whether gather/stream coupling is needed

After phase re-accounting and domain-cap fitting, compare:

```text
head locality residual      # gather -> W13 transition
after-1 locality residual   # 15/15 peers already in W13
```

Add a gather/stream coupling term only if one phase class retains a stable,
same-sign residual across both sessions that exceeds the paired measurement
resolution. The term must be fit without using route or real-trace holdouts.
Otherwise keep gather and W13 on the same domain-injection resource with
different offered rates.

### Step F: immutable holdout replay

Freeze the candidate calibration before reading evaluation metrics.

Route/context report must include:

- absolute span MAPE, median absolute relative error, P90 absolute relative
  error, and maximum error;
- stage MAPE for gather, W13, and W2 separately;
- per-M and per-context residuals;
- pairwise direction accuracy among the five modes at each M;
- explicit comparison with frozen v8 and the rejected stacked prototype.

Real-trace report must include, for all three traces:

- candidate event score recomputed from the canonical executable state;
- hardware paired gain already stored in the artifact;
- Spearman as a diagnostic, not an adoption gate by itself;
- top-K recall for at least K=8,16,32;
- measured-best retention per trace;
- false pruning/dominance count for every hardware-resolvable pair;
- decision coverage and number of incomparable pairs;
- selected-plan regret and whether the 2% action margin would replace the
  incumbent.

Use the existing `pairwise_plan_ordering.py` and
`analyze_plan_pairwise_ordering.py` metric definitions where possible. If the
three raw artifacts do not contain sufficient canonical state to rescore, use
the exact recorded route/seed/configuration to regenerate states **score-only**;
do not rerun or reuse hardware outcomes for fitting.

## 8. Suggested acceptance decision

Predeclare the final decision before holdout replay:

- No fit/holdout artifact SHA overlap.
- Phase calibration improves isolated stage errors without hiding them in total
  residual.
- Domain locality direction is correct in both cross-LLC sessions and phase
  windows.
- Route/context holdout MAPE is lower than frozen v8's 21.7%, with no new large
  systematic M/context bucket.
- Zero false dominance/pruning on hardware-resolvable real-trace pairs.
- Measured best retained at the declared top-K on all three traces.
- No selected-plan regression above the existing 2% action margin.

If only the absolute MAPE improves but pairwise/top-K fails, keep the model as a
diagnostic and do not enter VND/LNS. If pairwise safety passes but coverage is
zero, report it as safe but vacuous, as with the previous partial-order result.

## 9. Reproduction commands already used

Cross-LLC fit session pattern:

```bash
ssh Arm-codex-internal '
  cd /home/zhangxu/codex/fused_cpp &&
  PYTHONPATH=.:src \
  taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_gather_injection_overlap.py \
    --analytic-calibration \
      bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
    --warmup 5 --runs 31 --weight-copies 4 \
    --seed 20260905 \
    --trace-dir /tmp/moe_gather_injection_overlap_cross_llc_20260904 \
    --output /tmp/moe_gather_injection_overlap_cross_llc_20260904.json'
```

Independent repeat used seed `20260906` and the corresponding
`cross_llc_repeat` output path.

Current focused validation:

```bash
.venv/bin/python -m pytest -q \
  tests/test_moe_analytic_model.py \
  tests/test_moe_gather_injection_overlap.py \
  tests/test_moe_small_expert_context.py \
  tests/test_moe_executable_plan_neighborhood.py \
  tests/test_moe_pairwise_plan_ordering.py \
  tests/test_moe_pairwise_ordering_report.py
```

Last local result: `153 passed`. Last Arm result for the analytical model and
new overlap probe: `112 passed`. Ruff, `py_compile`, YAML parsing, and
`git diff --check` also passed at that point.

## 10. Current worktree warning

The worktree was dirty before this handoff. Relevant status includes:

```text
 M cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md
 M cpu_moe_schedule_optimization/TODO.md
 M cpu_moe_schedule_optimization/cost_model/analytic_model.py
 M cpu_moe_schedule_optimization/planners/executable_plan_neighborhood.py
 M optimizations/fused_moe_sve/benchmarks/bench_executable_neighborhood_audit.py
 M optimizations/fused_moe_sve/benchmarks/bench_narrow_lane_merge_transition.py
 M optimizations/fused_moe_sve/manifest.yaml
 M tests/test_moe_analytic_model.py
 M tests/test_moe_executable_plan_neighborhood.py
 M tests/test_moe_narrow_lane_merge_transition.py
?? cpu_moe_schedule_optimization/planners/pairwise_plan_ordering.py
?? optimizations/fused_moe_sve/benchmarks/analyze_plan_pairwise_ordering.py
?? optimizations/fused_moe_sve/benchmarks/bench_gather_injection_overlap.py
?? optimizations/fused_moe_sve/benchmarks/bench_small_expert_context.py
?? optimizations/fused_moe_sve/results/arm_codex_80c_dram_domain_injection_20260903.md
?? optimizations/fused_moe_sve/results/arm_codex_80c_pairwise_ordering_mvp_20260902.md
?? optimizations/fused_moe_sve/results/arm_codex_80c_small_expert_context_20260903.md
?? tests/test_moe_gather_injection_overlap.py
?? tests/test_moe_pairwise_ordering_report.py
?? tests/test_moe_pairwise_plan_ordering.py
?? tests/test_moe_small_expert_context.py
```

Treat all of these as user work. Inspect targeted diffs and preserve unrelated
changes. The remote Arm worktree also contains direct-synced uncommitted files;
it is not a clean-commit paper environment.

## 11. Immediate next action

The interrupted continuation had only completed artifact discovery; no
re-accounting builder or new isolated phase corpus was created. The immediate
next action is therefore Step A: implement and test a dedicated isolated
phase-trace fit benchmark with an explicit artifact role (`fit`) and disjoint
route set, then collect two Arm sessions. Do not start by tuning the existing
domain `capacity_scale` again.
