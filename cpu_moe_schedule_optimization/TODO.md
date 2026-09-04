# MoE scheduling implementation checklist

Status refresh: 2026-09-02. The latest formal Arm 80-core planner run is bound
to `6b6b4d1` and summarized by the report commit `1b7e353`, but most
supplemental executor/planner measurements below are still bound to the prior
extension SHA256
`5ed0b9c440acbf151cfbc22d95b00b6eee13fcd977ea2a16050018ee8cce73a9`.
They close experimental questions, not the final artifact-freeze gate. Rebuild
and rerun every headline result from one clean paper commit before submission.
The durable project summary for these provisional measurements is
[`amazon_192c_paper_closure_20260831.md`](../optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md).

The policy-aware TP2/EP2 scheduling update is implemented as one change set.
Cross-profile interpolation deliberately remains disabled. A captured routing
summary now passes exact-profile validation, but interpolation still requires
out-of-profile measurements across F, parallel degree, and machine topology.

The current ARM SVE compute baseline is now closed on the non-ILV exact-M
M8/M12 schedules. The upstream ILV comparison did not change production
dispatch, so it does not by itself require a cost-table refresh. The active
priority order is:

Overall conclusion: the ARM main path is now substantially complete. SVE JIT
exact-M kernels, the double-buffered K-loop, direct FP32 route stores, weighted
merge, the dynamic short-expert pool, the native cold planner, and stage-aware
cost modeling have all landed. Legacy split/range controls were replaced by the
single `(threads, window_tiles)` stage geometry: a task still covers full N, but
may do so as multiple serial owner windows. The next phase should not accumulate
isolated micro-optimizations. It should close the remaining high-value variables
first, then refresh calibration and validate the system coherently. In order:

1. Replace the template paper body with a claim-aligned draft and update the
   paper-readiness/model documents to the current evidence.
2. Freeze one source revision and regenerate every headline result through the
   declarative paper runner. The 192C raw artifacts are preserved under ignored
   workspace storage, and the Arm 80C route/calibration assets plus high-skew
   suite are now runner-managed with expected SHA256 checks.
3. Close or explicitly narrow the cost-model claim. Placement-aware LLC and
   single-team/full-cohort wide-team pressure now pass the current Arm 80-core
   three-trace gate; temporal top-candidate ranking, multi-layer holdout, and
   cross-machine calibration remain open.
4. Replace whole-problem CP-SAT as the primary optimizer with an executable,
   event-model-guided VND/LNS track. Retain CP-SAT only as a reduced exact
   oracle or bounded neighborhood repair until it demonstrates independent
   value over the full incumbent.
5. Add the current upstream Arm CPU MoE baseline, a full multi-layer vLLM run,
   a second Arm machine, and measured TP/EP communication before making system
   or portability claims.
6. Keep W8A16/W8A8, BF16 route storage, shared-expert extensions, W2-to-merge
   fusion, router fusion, communication pipelining, and large-K specialization
   outside the primary paper claim until the BF16 Arm path is closed.

## Next planner track: event-guided VND, LNS, then ALNS

Do not start by adding adaptive operator weights. First establish that the
executable-plan neighborhood is locally useful, repairable, and sufficiently
cheap to evaluate. The primary objective is the complete heavy event-model
makespan; every intermediate state must lower directly to a legal strict Plan
V2. The current analytical full, one-step, fixed-width greedy, and measured
controls remain immutable incumbents. CP-SAT is not the whole-problem primary
optimizer in this track.

### Step 0: canonical executable search state

- [x] Represent the rank as a gap-free partition of contiguous fixed-width
  lanes, each carrying an ordered expert sequence, and represent LLC domains as
  a second contiguous partition. Every active expert appears exactly once.
  Existing full-plan lanes that cross a domain boundary are preserved and
  report both intersected domain ids; later topology-changing moves must not
  create new cross-domain lanes.
- [x] Implement deterministic conversions among the current full-plan result,
  the search state, and Plan V2. Preserve width, physical core interval, lane
  order, dependencies, windows, early-merge setting, and output semantics.
- [x] Add a canonical state hash and reject duplicate states. Round-trip tests
  must reproduce the original Plan V2 and event score; Arm correctness must be
  bitwise identical before any performance search. The implementation and
  focused tests are in `planners/executable_plan_state.py` and
  `tests/test_moe_executable_plan_state.py`; Arm-codex NUMA3 additionally
  passed the SVE Plan V2 strict/tail-pool versus legacy bitwise execution test.

### Step 1: falsifiable neighborhood audit

- [x] On the current uniformish, median, and high-skew traces, enumerate or
  sample order-only legal neighbors around `full_selected`: same-lane adjacent
  swap/insertion, same-width cross-lane relocation/swap, and same-width
  cross-LLC relocation.
- [x] Record neighbor count, valid/duplicate ratio, event-score delta, best
  gain, evaluation throughput, and hardware result for the event top candidates.
  Compare critical-event-guided expert selection with uniform random selection.
- [x] Continue to VND only if useful locality is observed: at least one stable
  improving neighbor, critical guidance materially enriches the top candidates,
  and event improvements are not erased systematically by hardware execution.
  If local scores are effectively random or all useful plans require global
  reconstruction, reject LNS and move to template-level global search.

**Decision: the Step-1 gate failed; do not enter Step 2 with the current event
model.** The committed Arm 80C run evaluated 561--601 neighbors per trace and
measured four event-top candidates from each of the critical and random arms
(24 candidates total). No candidate had positive paired P10 speedup. Event to
hardware Spearman was `0.143/-0.156/-0.690` on uniformish/median/high-skew, and
high-skew event-top candidates regressed by up to `14.50%`. Critical improving
fractions were `22.6%/20.4%/48.8%`, versus random `24.6%/43.3%/47.4%`, so the
guidance did not consistently enrich candidates. Keep Step 2 gated off until
temporal-order/event ranking is corrected, or move directly to template-level
global search. See
`optimizations/fused_moe_sve/results/arm_codex_80c_executable_neighborhood_audit_20260902.md`.

### Step 2: deterministic variable-neighborhood descent baseline

Status: gated off by Step 1; do not implement against the current event model.

Temporal-ranking remediation in progress before this gate may be reconsidered:

- [x] Reproduce the high-skew critical-path crossing with runtime phase traces
  and separate early-merge time from scheduled expert compute.
- [x] Add an independent three-state 1T victim probe and fit width-specific
  expert fixed/per-route overhead without using planner-neighbor measurements.
- [x] Freeze the new Arm calibration and repeat the uniformish, median, and
  high-skew neighborhood audit. Point ranking improved only on high-skew and
  exposed new sub-1%-predicted counterexamples, so VND remains gated off.
- [x] Add a serial-lane uncertainty guard and a 2% minimum actionable gain;
  candidates below the resolution margin may be measured but cannot replace
  the incumbent automatically.
- [x] Repeat the committed three-trace suite with the uncertainty-aware decision
  output and require zero selected-plan regression above 2%. Keep point-ranking
  accuracy and safe planner selection as separate reported results. Commit
  `d51cb0e` retained baseline on all traces; measured shortlist regret was
  `0.242%/0.578%/0.516%`, with no stable local improvement. Point ranking below
  1% remains unresolved, so Step 2 stays closed.
- [x] Add an anchor-relative pairwise validation report and a three-way partial
  order without changing the analytical mean model. A same-v8 cross-session
  replay retained the measured best at top-8 on all three traces and made zero
  false-dominance decisions, but all 41 evaluated pairs were incomparable and
  decision coverage on the three hardware-resolvable pairs was zero. This is a
  safe diagnostic, not a useful VND pruning relation. A targeted two-lane swap
  probe did not justify a generic temporal correction; see
  `results/arm_codex_80c_pairwise_ordering_mvp_20260902.md`.
- [x] Serialize measured-shortlist affected lanes, exact before/after task and
  route sequences, isolated loads, and compressed placed-event context. On the
  high-skew trace, three cross-lane swaps with `+4.02--5.70 ms` affected-tail
  exposure regressed `12.61--15.17%`; the no-new-tail swap was unresolved at
  `+0.40%`. No placed critical lane/expert switched. An excluded synthetic
  `68-route tail <-> 1-route head` probe reproduced the slowdown but left only
  `+1.95/-0.17` point background/isolated residual, so no new physical
  parameter or calibration revision is justified.
- [x] Connect the anchor-relative comparator to the offline shortlist and a
  reusable best-improvement search loop. Only confidently worse candidates are
  dominance-pruned; incomparable candidates stay eligible and budget deferral
  is reported separately. The runner requires matching model/extension identity
  plus zero-false-pruning and requested top-K recall gates. Known-counterexample
  replay has zero false pruning; the mixed historical corpus fails top-8 but
  passes top-16, so the guarded offline default is top-16. The same-v8 41-pair
  replay remains 41/41 incomparable, therefore this closes plumbing and safety,
  not search effectiveness or the multi-start VND measurement below.
- [x] Decompose one fixed 1-route 1T expert across full-head, after-68,
  16T-only, 1T-only, and isolated contexts without changing total work. Two
  Arm 80C sessions found a stable `+0.823/+0.881 ms` full-head penalty: roughly
  two thirds from 16T background, one third from 1T peers, and negligible
  interaction. Delaying the same expert behind 68 routes removed
  `0.557/0.545 ms`, almost entirely from W13. Frozen v8 underpredicts full-head
  by about 58% and treats 1T-only as isolated. The concrete context is now
  reproduced independently, but no parameter is added until a predeclared
  route sweep establishes its shape and held-out explanatory value; see
  `results/arm_codex_80c_small_expert_context_20260903.md`.
- [x] Harden the small-expert probe to strict-DRAM measurement semantics while
  retaining measured-copy rotation. Every sample now runs an untraced,
  synchronous full-workload scrub on a fifth disjoint 660 MiB packed copy,
  followed by the measured mode on one of four rotating copies. Two scrubbed
  sessions preserve the decomposition (`+0.811/+0.819 ms` full,
  `+0.558/+0.559 ms` 16T, `+0.272/+0.270 ms` 1T), while address-filtered SPE
  reduces sampled target-W13 L3-hit incidence from about `6.9%` to `1.7%`.
  Scrub is excluded from target timing and trace output.
- [x] Run the predeclared scrubbed `M={1,2,5,6,12}` context sweep and prototype
  explicit gather pressure. Full-head excess falls from `+0.811/+0.836 ms` at
  M1/M2 to `+0.283 ms` at M12; 1T-only excess falls to `+0.022 ms`. An opt-in,
  default-off gather phase plus residual time redistribution lowers in-sample
  25-point mean absolute relative error from `21.7%` to `12.6%`, but one global
  traffic multiplier overpredicts M1 1T-only/after-68 as `1.254/1.171 ms`
  versus `0.907/0.921 ms`. Do not freeze this calibration or open VND/LNS.
- [x] Add an independent placement/overlap probe separating same-LLC and
  cross-LLC 1T background in gather-transition and pure-W13 windows. Two
  31-round sessions reproduce local-minus-remote penalties of `0.244/0.248 ms`
  at head and `0.225/0.233 ms` after one route, with every paired P10 above
  `0.21 ms`. Select per-LLC-domain memory injection as the primary resource.
  The opt-in cap reduces locality-contrast mean absolute error from about
  `0.230 ms` to `0.055 ms`, but stacking it on frozen-v8 corrections worsens
  the old route-context holdout MAPE to about `28.5%`; do not freeze a value.
- [x] Re-account gather/W13/W2 from a new disjoint isolated phase corpus across
  all supported widths, reset whole-total and old wide/narrow residuals, fit
  domain injection only from cross-LLC after-1, and add gather coupling only
  after a stable head residual. Isolated holdout W13/W2/total MAPE improves to
  `1.46/9.18/3.69%`, but the contrast-only gather multiplier is unidentifiable:
  full route-context MAPE is `273.97%`, high-skew top-8 misses measured best,
  and real traces contain 16 false-dominance pairs. Reject the full candidate;
  keep frozen v8 and VND/LNS closed.
- [x] Add an absolute-pressure identification probe that sweeps local and remote
  gather aggressor count. Jointly fit isolated-relative slowdown and locality
  contrast; do not fit another parameter from a single difference. Reuse the
  accepted phase floor/scale form, and retain the old route/real traces as
  untouched holdout. Two Arm sessions saturate local slowdown near four
  68-route streams and keep remote near zero. The session-1 joint fit lands
  on $\beta=0.78$, $\alpha_g=0.25$ at the multiplier bound, with fit/validation
  joint MAE `0.689/0.719 ms` and 739 near-optima. Rank `dram_bytes` sharing,
  not gather coupling, supplies the ~0.9 ms common-mode overprediction.
  Reject the candidate; keep frozen v8, holdout unread, and VND/LNS closed.
- [x] Change DRAM contention scope rather than retune $(\beta,\alpha_g)$.
  First ablate rank-wide `dram_bytes` versus domain-only injection on the
  locked count-sweep DAGs, then add a default-off structure only if one
  missing resource explains the local n≈4 plateau and near-zero remote.
  Do not read holdout or add an empirical residual. Rank DRAM accounts for
  about 49% of remote n15 (`+0.893` to `+0.454 ms`); leftover LLC dilation
  is `~2.44`. No arm saturates at n≈4 or makes remote near zero. Do not add
  a structure.
- [x] Ablate rank LLC versus domain LLC on the same locked DAGs. Rank LLC
  accounts for 100% of leftover remote n15 (`+0.454` to `0 ms`); same-LLC
  is unchanged, as required. Domain LLC still grows through n=15
  (`0.168/0.397/0.447/0.590 ms`). After rank DRAM, rank LLC, domain LLC,
  and L2 are removed, the 1-route 1T victim is isolated (`+0.001 ms`)
  while hardware same-LLC n15 remains `+0.312 ms`. Do not add a structure.
- [x] Propose victim-asymmetric dilation on the same locked DAGs: a 1-route
  1T victim must not inherit 68-route peer GEMM LLC/DRAM dilation, and any
  remaining same-LLC tax must saturate near four streams. Add a default-off
  structure only if one arm hits remote-near-zero, n4 contrast, and the
  n≈4 plateau together. Do not read holdout or add an empirical residual.
  `compute_bound_skip` matches symmetric because the 1-route 1T W13 victim
  is transfer-bound. `same_llc_peers` zeros remote and leaves the growing
  local curve. `own_demand` zeros both (n15 `+0.001/0 ms`). No arm hits all
  three gates. Do not add a structure. The leftover is a saturating
  same-LLC occupancy tax that cannot be identified on this count sweep.
- [x] Identify the leftover saturating same-LLC occupancy tax with a probe
  independent of this count sweep: vary aggressor $M$ at fixed count. Do not
  fit $T_\mathrm{sat}$ or $n_\mathrm{knee}$ on the existing n=1/2/4/8/15
  curve, do not add an empirical residual, and do not read holdout. Probe is
  `bench_aggressor_m_occupancy.py`: victim $M=1$, counts 0/1/4, aggressor
  $M\in\{1,4,16,68\}$, same/cross LLC, head and after_1. Session-1 overlap is
  valid and remote n4 is near zero. same-LLC n4 head is
  $+0.660/+0.639/+0.402/+0.290\,\mathrm{ms}$ at $M=1/4/16/68$: the tax falls
  as aggressor $M$ grows. Predeclared occupancy / duration occupancy /
  utilization all miss. Do not add a structure. The leftover is same-LLC
  contention among concurrent transfer-bound streams, not 68-route byte
  utilization.
- [x] Split 16 fill ports from 16 streams on LLC7 logical `48-63`. Arm
  session-1: 1T / 1×16T / 16×1T same-LLC head $+0.018/+0.014/+0.160\,\mathrm{ms}$.
  `fill_ports` is false; predeclared `one_stream` misses the $0.20\,\mathrm{ms}$
  gap. Do not add a structure. Do not read holdout.
- [x] Count streams, not threads: on cores `48-63` the 16-thread ladder is
  $+0.003/+0.035/+0.075/+0.155\,\mathrm{ms}$ at $1/2/4/16$ streams, and
  4×4T matches 4×1T ($+0.075$ vs $+0.070$). On `43-63`, `8+8+4+1` matches
  four mix-start 1T ($+0.110$ vs $+0.103$) not 21×1T ($+0.301$). Signature
  `stream_count`. Do not add a structure. Do not read holdout.
- [x] Put one layer's W13+W2 into one contiguous DRAM allocation and compare
  it with the current two packed tensors. Arm session-1: 16×1T leftover
  split/unified $+0.153/+0.157\,\mathrm{ms}$ (diff $0.004$). Signature
  `layout_neutral`. This cannot collapse 16 concurrent expert streams into
  one. Do not add a structure. Do not read holdout.
- [x] Put the same contiguous W13+W2 block on mmap+MADV_NOHUGEPAGE versus
  mmap+MADV_HUGEPAGE and verify AnonHugePages. Arm session-1: 4 KiB huge
  pages $0$, THP coverage $100\%$; 16×1T leftover small/THP
  $+0.167/+0.162\,\mathrm{ms}$ (diff $0.005$). Signature `page_neutral`.
  Do not add a structure. Do not read holdout.
- [x] Write the leftover-identification handoff:
  `optimizations/fused_moe_sve/results/cost_model_leftover_identification_handoff_20260904.md`.
  Do not add a structure from this chain.
- [x] Attribute the stream-count leftover with simultaneous core PMU, all LLC7
  L3C slices, and all NUMA3 DDRC counters. Use a PMU-only active-task plan so
  dependency-delayed post-victim work is excluded, and gate counters around
  each measured cell after the disjoint full-weight scrub. Two joint 31-run Arm
  sessions show that DDR read-command latency and victim LLC-miss/backend-stall
  response saturate near four distinct packed-B fills while aggregate DRAM
  traffic continues from about 10 to 108 MiB/call. Do not add a formula from
  five cells; 4x1T remains noisy and needs an independent count/team sweep.
  See `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_pmu_20260904.md`.
- [x] Sweep 0/1/2/4/8/16 distinct packed-B blocks at fixed 16 peer threads,
  independently repeat 4x1T, and compare anchored DDRC-queue and victim-LLC
  single-feature models with leave-one-count-out. Queue/LLC LOCO MAE is
  `0.0453/0.0364 ms`, but LLC pressure becomes negative at count 1 and has a
  larger max error. Queue transfers better to the two 4x1T layout checks
  (`-0.0274/-0.0010 ms` error versus LLC `+0.0558/+0.0248 ms`) but still
  underpredicts counts 8/16. Prefer queue only for follow-up; freeze neither
  model. See
  `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_count_loco_20260904.md`.
- [x] Remove process-per-cell baseline drift: keep all modes and packed copies
  in one long-lived process, randomize isolated/candidates per round, and use
  direct `perf_event_open` reset/enable/disable/read for each cell. Main and
  independent 4x1T sessions have 31 paired rounds and 60 counters/cell, all at
  running ratio 1.0. Paired queue/LLC LOCO MAE is `0.0515/0.0778 ms`; affine
  sensitivity is `0.0355/0.0432 ms`. Queue wins and tracks the 4-to-8-stream
  transition, but its linear form overpredicts 1--4 and underpredicts 8--16.
  Accept the protocol; freeze neither feature and add no knee/residual. See
  `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_paired_pmu_20260904.md`.
- [x] Hold packed-B count and expert starts fixed while changing request-arrival
  shape: compare start-aligned 4x4T/4x2T/4x1T and 8x2T/8x1T in two independent
  same-process paired PMU sessions. At count 4, narrower teams lower queue
  latency but victim-span differences remain inside zero-crossing intervals.
  At count 8, 8x1T versus 8x2T lowers queue by `16.90/14.06 cycles` and victim
  span by `0.1133/0.0775 ms`; both P90s remain negative in both sessions.
  Reject victim-asymmetric narrow harm. Request pressure depends on distinct B
  count and team-width/issuer shape, but two counts do not identify a formula.
  See
  `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_request_shape_20260904.md`.
- [x] Run the locked `count={4,6,8} x width={1T,2T}` proxy grid with count 6
  as untouched holdout in two sessions. Fit count 4/8 only and test distinct-B,
  requester threads, their product, and a measured-W13-overlap oracle through
  `proxy -> DDR queue -> victim slowdown`. No proxy passes all queue, slowdown,
  direction, and 20% parameter-stability gates. Proxy-to-queue drift is
  `25.8--32.3%`; queue-to-slowdown drift is only `7.5%`. Set
  `stop_absolute_model_expansion=true`; keep frozen v8 and move safety work to
  partial order, top-K recall, and false-pruning replay. See
  `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_proxy_grid_20260904.md`.

- [ ] Starting independently from full, one-step, greedy, and fixed-width
  controls, run best-improvement descent over same-lane insertion, same-width
  cross-lane relocation, pair swap, and cross-domain relocation.
- [ ] Keep every accepted state executable and preserve the best incumbent at
  all times. Report per-operator proposals, accepted improvements, event calls,
  wall time, and best-so-far curve.
- [ ] Use the known high-skew `cp_sat_06` result (`33.148 ms` versus the
  `34.484 ms` full incumbent) as a diagnostic target: determine whether local
  executable moves can recover the opportunity, not merely whether one seed
  improves one model score.

### Step 3: topology-preserving width neighborhoods

- [x] Add domain-local lane split/merge moves such as `16 -> 8+8`,
  `8 -> 4+4`, and their inverses. Reassign only experts on affected lanes and
  maintain the exact 40-core domain partition throughout the move. Existing
  cross-domain incumbent lanes are preserved but cannot be width-move endpoints.
- [x] Add adjacent-width expert migration through existing lanes before
  permitting arbitrary per-expert core intervals. Every candidate must remain
  directly executable without fluid-to-contiguous lowering. Changed-width
  tasks refresh deterministic W13/W2 windows before Plan V2 lowering.
- [x] Measure order-only, width-only, and combined ablations. Retain width
  moves only when they improve held-out plan regret rather than just expanding
  the search space. The prefinal Arm 80C three-trace run found two stable
  high-skew lane merges (`+1.436%/+1.248%` paired median,
  `+0.262%/+0.340%` P10) but no stable uniformish/median candidate. Keep the
  operators in Lab, but do not enter width-VND: the model ranked a regressing
  split above both merges and no robust prediction crossed 2%. Rerun the
  committed `paper_experiments/suites/arm_width_neighborhood_audit.json` after
  calibrating the `1T+1T -> 2T` merge/concurrency transition. See
  `results/arm_codex_80c_width_neighborhood_audit_prefinal_20260903.md`.

- [x] Independently calibrate the high-skew `1T+1T -> 2T` transition without
  planner-neighbor traces. The three-case Arm 80C probe separates a 2T
  fixed/per-route overhead from discrete 1T/2T full-cohort resource
  corrections and reduces all nine declared synthetic validation errors below
  3%. The three captured route traces remain the commit-bound holdout; see
  `results/arm_codex_80c_narrow_lane_merge_calibration_20260903.md`.
- [x] From model commit `1fcbc7b`, rerun `arm_width_neighborhood_audit`.
  All three guarded decisions retained baseline; combined shortlist regret was
  `0.386%/0.637%/0.983%`. The old 1T-to-2T merge gains did not reproduce, but
  the model-ranked high-skew `16T -> 8T+8T` split was stable at `+1.037%`
  paired median, `+0.334%` P10, and 30/31 wins. Retain width operators in Lab,
  but keep width-VND disabled because predicted gain was only `0.278%` and
  median/high-skew point Spearman remained near zero. See
  `results/arm_codex_80c_width_neighborhood_narrow_calibrated_20260903.md`.
  An independent same-commit high-skew repeat kept the split at `+1.271%`
  median but changed its P10 to `-0.632%`; another merge changed from
  `+0.864%/-0.813%` median/P10 to `+1.437%/+0.433%`. No candidate has positive
  P10 in both formal sessions, confirming that sub-2% width gains are not yet
  cross-session stable.

### Step 4: make event-guided search affordable

- [x] Precompute immutable expert/width phase descriptions and memoize plans by
  canonical hash. Profile event evaluation before changing the search budget.
  The current Lab evaluator caches exact scores and lane phases; on the frozen-v8
  three-trace repeat, exact scoring sustained `6.34--9.90` plans/s and the cheap
  screen sustained `186--349` plans/s.
- [ ] Add a two-level evaluator: generate hundreds of legal neighbors with a
  cheap affected-lane/domain delta bound, then run the complete event model on
  only the best diverse subset. Verify that screening does not discard the
  measured best on the audit corpus. The first evaluator projects `1.57--5.06x`
  wall-time speedup and `42--85%` fewer exact calls, but failed this gate: every
  tested `8/16/32` per-operator budget discarded the uniformish measured-best
  state. Its retained-baseline regret was only `0.603%` and the candidate was
  not stable, but the strict recall requirement remains unmet.
- [ ] If evaluation remains dominant, incrementally replay only affected event
  intervals or move the deterministic simulator to native code. Report plans
  evaluated per second and quality versus equal wall-clock budget.

### Step 5: critical-window large-neighborhood search

- [ ] Derive expert criticality from the event log: final critical lane,
  maximum LLC/DRAM or wide-team dilation interval, cross-domain finish
  imbalance, and idle-core tail.
- [ ] Implement destroy sizes 4/8/16 over those critical windows. Keep all
  unaffected lanes fixed and repair the removed experts with bounded beam
  search over legal lane, domain, adjacent width, and a small set of insertion
  positions; start with beam widths 16/32/64.
- [ ] Maintain a diverse elite pool rather than one trajectory. Diversity must
  cover width histograms, LLC-domain assignments, critical-expert placement,
  and temporal order, not only distinct serialized start vectors.
- [ ] Compare LNS with VND under identical event-call and wall-clock budgets.
  Adopt LNS only if larger destroy/repair neighborhoods escape reproducible VND
  local optima on held-out traces.

### Step 6: decide whether adaptation is warranted

- [ ] Measure which destroy/repair operators win on each trace class. Add ALNS
  reward-weight updates only if operator effectiveness is complementary across
  workloads; otherwise retain the simpler fixed-mixture LNS.
- [ ] If adopted, reward separately for a new global best, current-state
  improvement, elite-pool admission, duplicate, and invalid repair. Freeze the
  update rule before the final holdout and provide operator/size ablations.
- [ ] Add bounded diversification through multiple starts, tabu state hashes,
  or occasional worse-state acceptance. The returned plan must always be the
  best incumbent, independent of the exploratory trajectory.

### Step 7: establish the near-optimality claim

- [ ] Build exact reduced instances from real routes with 8/16/32 experts and
  compare VND/LNS/ALNS against exhaustive search, branch-and-bound, or reduced
  CP-SAT. Report model-objective regret to the true reduced optimum.
- [ ] Strengthen and report large-instance core-work, critical-chain,
  per-domain, LLC, and DRAM lower bounds. Label calibrated relaxations
  separately from hardware certificates; do not infer a global guarantee from
  a loose resource bound.
- [ ] Report 1/10/30/60/300-second anytime curves and multiple independent
  starts. Require the final event-guided planner to preserve full/one-step/
  greedy incumbents, achieve measured shortlist regret no larger than 5%, and
  have no trace regression above 2%.
- [ ] Run the final comparison across complete 43-layer captured requests and a
  second Arm machine when available. Report `T_plan`, `T_execute`, and
  `T_plan + T_execute` separately.

### Step 8: connect the offline reference to runtime planning

- [ ] Compare the lightweight quick planner and fixed 8T fallback against the
  frozen event-guided offline reference on every evaluated layer. Report quick
  regret and planning latency rather than claiming it inherits offline quality.
- [ ] Only after the strict search is closed, add dynamic tail-pool recourse as
  a separate runtime stage and measure its incremental gain and variance. Do
  not include tail-pool behavior in the strict near-optimality certificate.
- [ ] Keep CP-SAT available only for reduced exact validation or repair of a
  bounded critical neighborhood; promote it again only if it improves the
  executable event incumbent under equal wall-clock budget.

The legacy weight-split removal is complete across the production stack:

- [x] Remove global and per-task W13/W2 range counts, byte windows, split flags,
  their environment controls, old Plan V2 tensors, and native fallback branches.
- [x] Make W13/W2 cover one full-N stage; represent any serial subdivision only
  as per-worker tile windows derived from `(threads, window_tiles)`.
- [x] Remove legacy split/range stage-window search and cache identity. Restore
  only the exact per-worker tile-window traversal, which still covers full N
  and is selected deterministically after team width rather than searched as a
  free planner variable.
- [x] Migrate calibration generators, benchmark defaults, timeline/schema docs,
  empirical/analytic models, and the native cold planner to full-N geometry.
- [x] Close the 192-core TP4 analytical tile-window selection subproblem on the
  declared transition domain. Policy v6 uses independent service/cache
  calibration and reaches `1.10/3.45/4.40%` median/P90/max regret on a full
  W13 x W2 Cartesian holdout for M=`72,216,384`, T=`4,16` (6/6 below 5%).
  This is a shadow-policy gate, not a production-default switch.
- [ ] Repeat the current tile-window holdout on the 8-core ARM machine after
  replacing its transferred packed-B retention prior, then extend 192-core
  coverage to unseen routes, widths, and mixed planner workloads. Do not reuse
  the retired range-policy `1.50/2.98/3.38%` result as current evidence.

1. Replace the AmazonECS8Cores transferred packed-B retention prior with a
   machine-local multi-team retention/refill probe and add topology-aware LLC
   service. Its cache topology and L1-hot/L2/LLC/DRAM service curves are now
   measured locally; only this retention term remains non-local.
2. Repeat unseen routes, widths outside `4T/16T`, mixed distributions, and
   complete planner validation. The current 192-core full-Cartesian result
   closes only its declared transition domain.
3. Validate the implemented companion-profile rank-lifetime switch on held-out
   EP workloads, and isolate the remaining uniform-96 full-call residual.
4. Add measured gather/pack, route merge, communication, and distributed TP/EP
   terms after the compute model passes its gates.

Paper-facing contribution scope, reusable evidence, prohibited claims, and the
required main-table matrix are maintained in
[`../docs/moe_paper_readiness.md`](../docs/moe_paper_readiness.md). The
checklist below remains the implementation ledger; paper closure uses the
additional P0 gates here.

## 0. Paper-critical closure (refreshed 2026-08-31)

The provisional AmazonC5192Cores NUMA1 closure run currently establishes:

- a five-process local packed-B repeated-scan fit with an effective knee at
  `0.662 x 2 MiB`, miss floor `17.37%`, nominal-L2 miss `68.03%`, and
  two-L2 miss `79.80%`;
- analytical isolated holdout MAPE `7.96%`, contention MAPE/P90
  `10.85/16.08%`, and maximum measured shape regret `9.21%`;
- a high-skew captured trace where measured `8T reverse-even` is `17.282 ms`
  versus `22.857 ms` for the full-planner selection, a paired `24.43%`
  latency reduction despite the model ranking 16T ahead of 8T;
- complete captured-trace two-stage grids where the best independently staged
  W13/W2 execution remains `1.93--5.87%` slower than the best whole-expert
  plan; and
- a repaired canonical explicit same-GEMM comparator: fused execution is
  `17.87%` slower at `24 x 4T, M=48`, neutral within `0.79%` at
  `12 x 8T, M=192`, and `19.11%` lower latency at `6 x 16T, M=2040`.

These measurements require the artifact-integration and clean-rerun items
below before becoming final paper tables.

- [ ] Freeze one paper commit and regenerate every headline profile/result from
  that source and extension. The current provisional results mix the prior
  extension snapshot with the comparator repair in `266da2c`.
- [x] Rerun the controlled nine-case catalog with the default FP32 direct-route
  path on AmazonC5192Cores NUMA1 using 31 samples and the current empirical
  profile.
- [ ] Add the nine-case catalog to a committed >=31-sample paper suite and
  repeat it on a second Arm machine. If BF16 route storage appears in any main
  table, first close its real model-level quality gate.
- [x] Repair and measure the canonical explicit same-GEMM unfused control with
  the production FEXPA-plus-quadratic SiLU approximation. H64/F64 plus
  H4096/F512 M=`24,192,2040` correctness gates pass; the paper comparison must
  use the standard `SiLU(gate) * up` artifact, not the rejected
  exact-association control that adds an extra gate read.
- [ ] Compare the fused executor with the current upstream Arm CPU MoE path.
  Record identical shapes, routing, affinity, page policy, warmups/runs,
  absolute latency, and numerical checks.
- [x] Replace reconstructed-only evidence for the first route-distribution
  study with three complete 2048-token TopK=6 captures (uniformish, median,
  and high-skew). The converted `.pt` expert/layer tensors are elementwise
  identical to their source `.npz` files and contain no duplicate expert per
  token.
- [ ] Run complete multi-layer traces across multiple requests, including at
  least one full prefill sweep and one decode-oriented corpus. The current
  three selected layers do not close this gate.
- [ ] Decide the cost-model paper claim. Either refresh the empirical
  phase-aware model as the primary model, or make the analytical backend pass
  isolated MAPE <=10%, contention P90 <=15%, and maximum measured regret <=5%
  on the declared two-machine domain. The current local-retention result passes
  isolated MAPE (`7.96%`) but still fails contention P90 (`16.08%`) and maximum
  regret (`9.21%`); otherwise narrow the claim and domain explicitly.
- [x] Close the analytical-full wide-team misranking on the current Arm-codex
  three-trace domain with a one-step width uncertainty gate. On NUMA3 80C, 5
  warmups, 31 randomized paired rounds, and four rotating weight copies, the
  committed `b219627` runner improves high-skew/median/uniformish by
  `15.27/17.95/5.69%` paired medians and leaves `0/0.26/0%` measured-set
  regret. It does not change quick/request-path, explicit-shape, empirical, or
  native planners.
- [ ] Close the production quick-planner quality regression. Active-set 8/16
  still select over-wide homogeneous teams on the existing evidence; do not
  claim quick dominates fixed-width execution until a separate gate passes.
- [x] Define the current full planner as an offline planner/autotuner, not a
  proven oracle or request-path algorithm. Keep empirical full-search latency
  separate from the older 142-strict + 327-dynamic analytical search result.
- [ ] Write the exact quick/full candidate spaces, objectives, pruning, and
  complexity in the paper; report empirical and analytical full-search costs
  separately.
- [x] Generate provisional compact quick/full tables for the nine-case catalog
  and three captured traces, including cold/warm planning time, execution time,
  selected action, and raw 31-sample arrays.
- [ ] Finish moving the provisional real-route, retention, two-stage, and
  unfused cases into `paper_experiments/`. The 192C artifacts are preserved in
  `tmp/moe_paper_archive/amazon_192c_20260831`; ignored route/calibration assets
  now have runner-enforced tree hashes, and `arm_high_skew_closure.json` covers
  the three Arm-codex 80C planner traces. Amazon 192C machine configuration and
  a clean frozen >=31-sample rerun remain blocked while that host is unavailable.
- [ ] Include `T_plan + T_execute` in every final planner comparison. The
  captured-trace artifact already reports it, but the complete paper matrix
  must use the public runtime path rather than only internal assignment cost.
- [ ] Replace the remaining template content in the companion paper's
  `main.tex`: write the abstract and keywords, remove `Ease of Use` and the
  IEEE example sections, and add complete cost
  model, planner, execution, evaluation, related-work, and limitations
  sections.
- [ ] Align the paper thesis with the demonstrated scope: Linux AArch64 SVE
  BF16 is primary; x86, W8A16/W8A8, general plan-then-adapt superiority, and a
  near-optimal full planner are not current headline claims.
- [ ] Replace both method figure placeholders, add the kernel/cost/planner
  tables, create the missing bibliography database, and synchronize the
  Chinese translation after the English method is stable.
- [ ] Run model-level quality validation for the production FEXPA SiLU and
  DeepSeek clamp-10 path. Keep BF16 route storage out of headline tables unless
  its separate real TopK=6 model-level gate also passes.
- [ ] Add a real multi-layer vLLM measurement and measured TP/EP communication
  before making layer- or model-level distributed performance claims.
- [ ] Either exclude 30k-token execution from the paper domain or repair the
  current FP32 direct-route int32-offset limit (TopK=6 reaches it at about
  21,845 tokens) and rerun the default path without forcing fallback.

## 1. Schema-v2 profiling

- [x] Synchronize two NUMA-local ranks and aggregate pairwise wall maxima.
- [x] Use M12-aligned contention routes plus M1/M2/M4/M8 isolated tail points.
- [x] Stream distinct expert weights; use all local experts for full-call
  contention and eight experts for isolated streaming cost.
- [x] Use the runtime planner's LPT lane assignment for mixed-width shapes.

## 2. Policy-aware exact-shape cost model

- [x] Key `T_iso` and contention data by the complete calibration domain:
  sharded F, kernel identity, NUMA topology, and concurrent-rank count; retain
  require `full_n_team_stripes` and reject old split measurements.
- [x] Use exact measured shape data by default instead of collapsing all shapes
  with the same active-expert count into one derate.
- [x] Preserve once-per-call cost with authoritative full-call anchors rather
  than multiplying normalized group cost by the number of waves.
- [x] Make route lookup M12-aware, including the M1/M2/M4/M8 tail kernels.
- [x] Reject incompatible, ambiguous, or grid-mismatched profiles.

## 3. Shape planner and full-stage geometry

- [x] Search `core_shape` only; derive each stage's owner stripe from the task's
  actual width before scoring and lowering.
- [x] Use packed working-set bytes to prune candidates, while retaining measured
  latency as the objective.
- [x] Remove legacy W13/W2 split/range geometry from Plan V2; retain only the
  exact per-task window-tile parameter, with zero meaning the full owner stripe.
- [x] Add explicit physical CPU sets, a calibration-aware cache key, full
  bucketed routing signatures, and confidence-aware tie breaking.

## 4. TP/EP layer evaluator

- [x] Select compute profiles from TP/EP degree, sharded F, local expert count,
  and machine topology instead of hard-coded TP4/EP4 files.
- [x] Evaluate global compute wall time as the maximum of rank-local plans.
- [x] Use rank-local routing histograms for EP load imbalance.
- [x] Generalize all-reduce/all-to-all communication formulas beyond the fixed
  P=4 two-pair topology.

## 5. Stage-aware working-set model

- [x] Add full W13 and W2 phases to the event simulation if exact-profile tables
  do not generalize adequately across F or unseen shapes.
- [x] Derive contention from the sum of active phase working sets rather than
  only the number of active expert tasks.
- [x] Add exhaustive dual-rank uniform/hotspot regret validation and an input for
  real routing histograms.

## 5.1 Analytical backend migration (2026-07-26)

- [x] Separate logical GEMM work, exact SVE kernel demand, and machine response.
- [x] Replace route/thread latency lookup with cache-capacity formulas and
  route-independent matrix/L1/L2/LLC/DRAM service curves.
- [x] Split each full W13/W2 stage into zero-demand setup, first-panel cold-B, and
  remaining-panel steady-B phases; preserve physical demand exactly.
- [x] Derive contention from per-resource offered load and calibrated service
  capacity using only threads that actually request that resource.
- [x] Expose event-level working set, spill fraction, offered rate, capacity,
  utilization, and dilation without a pairwise slowdown matrix.
- [x] Let the analytical backend generate homogeneous and two-width shapes while
  preserving empirical profile behavior.
- [x] Add an empirical-holdout validator for isolated error, contention error,
  and measured planner regret.
- [x] Produce an independent thin calibration on one NUMA rank of
  AmazonC5192Cores without inferring service ceilings from the route/thread
  table. The 2026-08-01 result reduced maximum shape regret from 32.31% to
  8.17%, but true isolated MAPE/contention P90 remained 10.22%/47.79% and failed
  the production gates.
- [x] Replace the register-only BFMMLA compute ceiling with an M12
  full-no-store GEMM service whose L1/L2 geometries are derived from Linux sysfs
  cache capacities. On AmazonC5192Cores the L1 probe is M12/K728/N16
  (40,768 B); the 4096-run 2026-08-02 refresh measured 0.340/30.377 TFLOP/s at
  1/96T, while holdout MAPE/contention P90/max regret were
  9.18%/49.57%/8.17%. Only the isolated gate now passes.
- [x] Replace the transferred AmazonC5192Cores packed-B retention prior with a
  machine-local single-core repeated-scan probe. Five independent M12/M120
  process pairs locate the stable miss floor at `17.37%` around 1.25 MiB, the
  fitted knee at `0.662 x 2 MiB`, nominal-L2 miss at `68.03%`, and two-L2 miss
  at `79.80%`. Rebuilding the current thin model with these anchors yields
  isolated holdout MAPE `7.96%`, contention MAPE/P90 `10.85/16.08%`, and
  maximum shape regret `9.21%`; the local prior improves absolute contention
  but does not close ranking.
- [ ] Measure independent exact-M M1-M11 core-efficiency ratios relative to the
  M12 L1-hot service. Admit them only if they improve unseen tail routes without
  becoming another route/thread latency table. The current complete-call
  exact-M speedup A/B does not by itself provide these independent service
  ratios.
- [x] Measure the corresponding machine-local cache topology and
  L1-hot/L2/LLC/DRAM service curves on AmazonECS8Cores. The runtime uses
  `backend_n_tile=16`; the profile marks its current 1/8-L2 packed-B retention
  term as a transferred prior rather than claiming a local measurement.
- [x] Retire the old range-count stage-window generator after the full-N ABI
  migration. Its 3.38% coordinate-oracle holdout remains historical evidence
  only; the current tile-window runtime is covered by shadow policy v6 and the
  2026-08-11 full-Cartesian holdout.
- [x] Separate analytical stage-window structure from thin calibration: exact
  tile geometry, reusable-B-only shared LLC pressure, and cache-derived
  tie-breaking are formulas; service curves, retention anchors, uncertainty,
  and W13/W2 range-restart constants are machine calibration.
- [ ] Model or bound the remaining W13/W2 second-order pair interaction before
  using the selector for absolute-time prediction. Extreme unselected pairs
  have up to 17.69% paired-round residual even though selected regret is below
  5% throughout the declared 192-core domain.
- [ ] Calibrate multi-team packed-B retention/refill and below-NUMA LLC topology
  from independent probes; do not add a task-pair slowdown table. The new local
  single-core retention curve closes only the task-local reuse input, not the
  wide-team/cache-data pressure exposed by the high-skew planner miss.
- [ ] Validate unseen routes, widths, mixed distributions, and full-stage
  owner-stripe behavior against the acceptance gates in
  `cost_model/ANALYTIC_MODEL.md`.
- [ ] Switch the production default only after both machines pass; retain the
  empirical backend as an explicit fallback and regression oracle.
- [ ] Add dedicated physical demand for gather/pack, route scatter/merge,
  communication, and cross-rank lifetime changes before claiming full E2E time.

## Deferred external validation

- [ ] Cross-profile interpolation stays disabled. A target-model routing dump
  now exists and passes the exact-profile regret gate, but interpolation still
  needs out-of-profile measurements across F, parallel degree, and machine
  topology before it can be enabled.

## 6. Absolute-time accuracy follow-up (2026-07-13)

The exact-profile planner currently has selected-plan error from -2.93% to
+4.40% and maximum measured regret below 2% on TP2/EP2 uniform, hotspot, and
captured-trace cases. Thirty-sample medians have an approximately 1% practical
noise floor. Stage traces show that the existing 2/3 W13 + 1/3 W2 split is
accurate: W13 accounts for 67.3-67.8% of W13+W2 time.

### P0: close the current calibration update

- [x] Commit the 2026-07-13 TP2/EP2 `R13=2/R13=1` profiles (with historical
  filenames retained), the expanded contention route grid, and the validator
  noise/stage-trace modes as separate reviewable changes.
- [x] Update `POLICY_MODEL_VALIDATION.md` to the 2026-07-13 profiles and real
  routing validation. Remove the stale statement that no target routing dump
  is available.
- [x] Remove the untracked legacy F512 schema-v1 profiles from the active
  workspace; retain old calibration results only in git history.

### P0: locate the remaining absolute-time error

- [x] Diagnose the EP2 hotspot. Symmetric dual-rank component replays match the
  profile within 0.4%, while full peer-idle replays are 1.4-7.0% faster. The
  mixed rank measures 63.0 ms against another mixed rank, 59.7 ms against an
  idle rank, and 60.6 ms against the real 32x96 cold rank. The +4.4% hotspot
  error comes from applying the `concurrent_ranks=2` derate for the long rank's
  whole DAG after the short rank has completed, not from `T_iso`, intra-rank
  expert contention, or the 2/3 W13 + 1/3 W2 stage split.
- [x] Generate matching single-rank companion profiles and jointly simulate all
  rank DAGs. Advance tasks with dual-rank rates while both ranks are active,
  then switch the remaining rank to the single-rank rate at the rank-completion
  event. Do not estimate EP wall time as the maximum of independently predicted
  all-dual-rank makespans. Current exact-M EP2 single/dual pairs on two 32-core
  ranks reduce the tiered-hotspot estimate from 55.605 ms to 54.770 ms
  (-0.835 ms, -1.50%); incompatible or missing companions retain the old
  conservative upper bound.
- [ ] Treat the uniform 96-route full-call residual separately. Route 96 is an
  exact isolated point but not a contention-route anchor, so the current full
  call estimate interpolates between 48 and 192. Add targeted full-call anchors
  only if the residual remains after rank-lifetime contention is modeled.
- [ ] Add held-out heterogeneous full-call cases for the planner-relevant
  shapes `(32)`, `(16,16)`, `(16,8,8)`, and `(8,8,8,8)`. Cover mild skew,
  bimodal hot/cold, one-dominant, hotspot, and captured routing; do not fit and
  validate on the same distributions.
- [ ] If TP heterogeneous residuals remain after fixing `T_iso`, add a
  low-overhead phase timeline and model W13/W13, W13/W2, and W2/W2 overlap from
  the active route/team distribution instead of `max(routes)` alone. Keep the
  measured 2/3 + 1/3 stage split unless new data disproves it.

Acceptance gates:

- selected-plan absolute error <= 3%;
- top-three candidate median absolute error <= 3% and P90 <= 5%;
- measured planner regret <= 2%;
- all comparisons use enough repeats to resolve the approximately 1% noise
  floor.

### P1: complete the end-to-end latency model

- [ ] Keep expert compute and route-combine costs separate. For TP2 top-k=6,
  add the measured approximately 1.1 ms weighted merge plus route-dependent
  gather/scatter; retain communication as a separate topology-bound term.
- [ ] Validate TP2 versus EP2 with an actual distributed/runtime execution.
  The current evaluator combines measured compute profiles with analytical
  communication and is not an end-to-end distributed measurement.
- [ ] Generate exact schema-v2 TP4/EP4 profiles before drawing TP4-versus-EP4
  conclusions; do not extrapolate the TP2/EP2 tables across parallel degree.

### P2: scheduling and kernel follow-ups

- [x] Compare upstream M8/M12 ILV schedules with the matching non-ILV
  pure-GEMM kernels on SVE256 Neoverse-V1 and SVE128 Neoverse-V3.
  M8 ILV regressed V3 W13 by 5.40% warm and 2.66% rotating-cold. M12 ILV
  delivered only 0.27-1.09% on the stable V3 cases and mixed results on V1,
  so neither schedule is adopted. Keep non-ILV as the fused JIT baseline.
  Full five-run W13/W2 results are in
  `../optimizations/fused_moe_sve/results/amazon_8c_192c_upstream_m8_m12_ilv.md`.
- [x] Test an M12 two-B-register column pipeline against the L1-hot full loop
  on all 96 Neoverse-V3 cores. The 21-repeat median rose from 30.420 to 30.617
  TFLOP/s (+0.648%), and linear efficiency rose by 0.592 percentage points,
  but `DISPATCH_STALL_IQ_VX` increased about 37.5%. This is below the 2%
  adoption threshold, so retain the existing M12 production schedule. The
  rejected probe source was removed from the active tree and remains available
  at Git commit `4825bf9`. Full results are in
  `../optimizations/fused_moe_sve/results/amazon_192c_m12_column_pipeline_20260802.md`.
- [x] Close W13-only first-panel software prefetch as a production candidate.
  It hides cold-B latency under low or moderate concurrency, but crosses over
  near 18-20 concurrent streams on AmazonC5192Cores and can regress reusable-B
  long routes. The environment flag, production dispatch, generated code and
  benchmark variant were retired on 2026-08-10; preserve the result document
  and Git history only. Reopen only with a new hardware result or a
  cache-retaining prefetch policy.
- [x] Add an offline no-contention CP-SAT oracle for the unrestricted supplied
  width set. It selects one fixed thread width per expert, removes dominated
  modes, returns incumbent and best-bound regret intervals, and compares the
  current strict plan with the same quantized `T_iso`. Keep OR-Tools optional
  and outside the production planner.
- [ ] Add conservative per-mode hardware lower-bound durations and aggregate
  matrix/LLC/DRAM capacity bounds before interpreting the CP-SAT result as a
  physical performance certificate rather than a `T_iso` model oracle.
- [x] Add a benchmark-only large/small route core partition for DSV4 TP4. Keep
  the two classes on disjoint NUMA-local core regions, use independent LPT lane
  chains, and preserve the production Plan V2 kernel/window/merge path. On the
  192-core host's NUMA0, `48C x 8T` for `M>48` plus `48C x 1T` for `M<=48`
  reduced the 51-run median from 13.367 to 11.625 ms (+14.99% throughput), with
  0.214 ms measured class-completion skew.
- [ ] Replace isolated-LPT scoring for the large/small partition with
  phase-local compute/L2/LLC/DRAM offered demand and cross-class service
  capacity. The current global simulator predicts 14.932 ms versus 11.593 ms
  measured (+28.81%), and underestimates the small-region dilation by 2.93x.
  Do not add this candidate to the production planner or plan cache until
  absolute error is <=3%, selected regret is <=2%, and no held-out workload
  regresses by more than 2% on two ARM machines.
- [x] Validate the fixed-lane temporal-order seeds on the 192-core host before
  interpreting their modeled gain as runtime gain. Use interleaved LPT/temporal
  calls for every catalog workload, report strict-only E2E and stage times, and
  require no held-out regression above 2%. The current TP4/F512 profile predicts
  +10.01% for captured DSV4 and +8.52% for tiered hotspot, but the cross-class
  absolute-time gate above is still open. The first 2026-08-11 NUMA0 E2E A/B
  found strict DSV4 `+8.34%`, tiered `-0.35%`, and strict long/short bimodal
  `-5.90%`; default tail-pool DSV4/bimodal were `+10.65%/+0.11%`. The candidate
  therefore fails the 2% gate. Phase traces localize bimodal's loss to a
  2040-token fixed-owner ready-merge burst (`0.35-1.94 ms` tail), not GEMM:
  disabling ready-token early merge gives `+5.88%/+6.07%` in two independent
  51-pair runs. A strict-only conservative gate now uses token count, route counts,
  and predicted expert finish waves after the cached compute plan to prove a ready
  burst lower bound; it disables early merge when at most one default owner batch
  (`2T` tokens) can remain outside that burst. The 2026-08-11 strict-only
  7-warmup/51-pair NUMA0 catalog rerun closes this host gate: bimodal improves
  `+6.28%`, DSV4 improves `+8.80%`, and tiered hotspot is `-0.56%`; all six
  unchanged-order controls remain within `0.89%` by median. The gate changes
  only bimodal (`early_merge: null -> false`) and creates no held-out median
  regression above 2%. On 2026-08-16 this routing-aware gate was superseded by
  the user-requested global fixed-on policy after m5 TP2 DSV4 resolved auto to
  on for all 43 captured layers and removing the gate cut forced-miss planner
  overhead by about 94%. The earlier TP4/F512 bimodal evidence remains a known
  cross-workload regression risk; fixed-on is not a model-derived claim and
  cross-machine promotion remains governed by the separate validation rule
  above.
- [ ] Measure lane-tail weighted idle loss on a larger routing corpus. Current
  TP cases have a perfect-rebalance upper bound of only 0.7-2.4%; prototype
  same-width-lane work stealing only if representative cases repeatedly exceed
  5%.
- [ ] **P0: Reduce end-of-run tail idle without weakening the selected head
  plan.** The 2026-08-12 AmazonC5192Cores NUMA0 DSV4 trace (`0-95`, 2048
  tokens, top-k 6, TP4/F512, 223 active experts) records `71.32 core-ms` of
  tail idle, equal to 6.62% of available core time and 73.42% of all visible
  bubble area. Start with bounded same-width suffix ownership transfer and
  residual-M repartition for the last lanes; do not introduce width changes,
  waiting, or cross-NUMA migration. Close only after paired E2E runs show a
  repeatable gain, P90 does not regress by more than 2%, correctness is
  unchanged, and a fresh trace confirms that tail area actually falls.
- [x] Close the first four P0 candidates on the same DSV4 case. Full
  `1/2/4/8T` cohort regrouping was `-0.849%`; opportunistic `2/4T` regrouping
  was `-0.613%` and created no widened task; changing the existing 1T LPT queue
  to ascending M was neutral (`-0.069%` for LPT); the model-selected E219
  suffix DAG was `-0.077%`; and bounded runtime residual-M was `-0.358%`.
  Remove those implementations rather than preserving default-off branches.
  A manually selected static E218/core40 residual-M split repeated
  `+0.48%--+1.20%` (five-run mean about `+0.93%`), but did not clear the 2% gate and
  other terminal targets were neutral or negative. Keep only its Lab
  comparator and evidence; it does not close P0. See
  `optimizations/fused_moe_sve/results/amazon_192c_dsv4_tail_candidate_closure_20260812.md`.
- [ ] **P1: Attribute and reduce the W13-to-W2 same-task barrier wait.** The
  same trace records `12.08 core-ms` (1.12% of available core time; 12.43% of
  visible bubbles). Each worker has the same 128-column W13 stripe (16 N8
  tiles and about 1 MiB of packed-B), while the slow local worker rotates
  across tasks, so first separate runtime service variation, topology, and
  kernel completion skew with repeated traces and counters. Evaluate a
  per-range ready handoff or static worker remap only after that attribution;
  do not remove the packed-C publication barrier without proving producer and
  W2 consumer ordering. Accept it as residual overhead if a safe candidate
  cannot produce a repeatable E2E improvement.
- [ ] **P2: Reduce deterministic Gather-to-W13 residual-panel imbalance.** The
  same trace records `3.87 core-ms` (0.36% of available core time; 3.98% of
  visible bubbles). For 8-thread experts, `split_evenly(panel_count, 8)` gives
  low local worker IDs the remainder panels; local worker 0 is last for 20 of
  28 tasks and averages 1.27x normalized gather time. Prototype K-splitting
  only the `panel_count % threads` remainder panels, using the existing K8
  partition with a minimum K chunk of 32 BF16 values, while preserving full-M
  ownership for the quotient panels. Keep global K-split disabled: the
  optimistic critical-path headroom in this trace is only about 0.08 ms, so
  promote this only if a broader routing corpus shows material E2E gain.
- [x] Add an experimental non-preemptive W13-to-W2 elastic boundary. A zero
  timeout uses only immediately idle same-NUMA workers; explicit local
  `2x8T->16T` and `4x2T->8T` cohorts may wait for a bounded interval and always
  retain the original-team fallback. Export natural-opportunity, preferred,
  timeout, and wait counters without changing production strict/tail-pool.
- [x] Measure elastic natural-opportunity rate and E2E gain on the 192-core
  host's NUMA0 paper workloads. Natural opportunity is `0-6.53%` for measured
  `8T->16T` workloads and `0.21%` for the `2T->8T` short-task workload; every
  measured elastic variant is slower than strict. Keep it outside production
  cost-model ranking. See
  `optimizations/fused_moe_sve/results/amazon_192c_w2_boundary_elastic.md`.
- [x] Retire W2 boundary elasticity from the active runtime, Plan V2 schema,
  planner bridge, and benchmark CLI. Generic cases regressed 2.5%--24.8%; the
  planner-visible `+3.66%` tail case is covered by bounded tail repartition.
  Preserve the implementation at Git `0b58091` and the result report above.
- [x] Allow the experimental elastic bridge to assign a disjoint same-NUMA W2
  target cohort. The runtime acquires the destination before releasing the
  source, so paired tail tasks can realize mappings such as
  `0-15 -> 0-31` and `16-31 -> 32-63` without copying packed-C.
- [x] Measure the explicit migration path on the 192-core host's NUMA0
  active-set-8 tail case. The 31-run zero-timeout median improves from
  `11.103 ms` to `10.710 ms` (`+3.66%`) with `61/62` natural preferred
  assignments. A separate 11-run sweep shows no benefit from 200/500 us
  waiting. Keep the action planner-selected rather than globally enabled.
- [ ] Run model-level accuracy evaluation for the optional Taylor-poly4 SiLU
  path. Kernel/output tests are complete, but only elementwise and operator
  output error have been measured so far.

## 7. End-to-end fusion and pipeline follow-up (2026-07-16)

These are research candidates, not active planner decisions. The current fused
boundary already covers gather-pack, W13, SiLU-and-multiply, packed-C production,
and packed-A W2 consumption. New work should therefore target the W2 output
path, task-tail overlap, and model-runtime boundaries before attempting deeper
W13 epilogue changes.

For the representative `tokens=2048`, `top_k=6`, `H=4096` case, the logical
route result contains 192 MiB in FP32 or 96 MiB in BF16. The current FP32
`W2 -> down -> route_out -> merge` path transfers about 768 MiB of logical
payload before counting the merge accumulator. These are cache-level logical
bytes, not predicted DRAM traffic.

### P0: remove post-W2 materialization and merge overhead

- [ ] Add a dedicated SVE weighted-route merge kernel. Iterate over H vectors
  outside the top-k loop so the FP32 accumulator remains in registers across all
  routes, then convert directly to BF16. Specialize `top_k=6`, retain a generic
  fallback, preserve per-element route order, and add 2-D token/H partitioning
  for short-token calls. The same epilogue should optionally fold routed scaling,
  shared-output addition, and residual addition without changing FP32
  accumulation semantics.
  - 2026-07-16 experiment: the SVE route merge tested U1/U2/U4 variants.
    `top_k=2/4/6/8` uses compile-time adjacent binary
    trees; other values use a one-token-row ordered runtime-loop fallback.
    On the 192-core host NUMA0, isolated BF16-route gains were 11-24% across
    top-k 2-8, while FP32 route regressed for top-k 7-8. Preplanned async E2E
    gains remained at or below 1.04%. U1 is now the SVE default to eliminate the
    per-worker FP32 allocation and accumulator traffic. On 2026-08-10 the
    sequential/U2/U4 source and runtime selector were retired after the
    production decision; the last comparison implementation is in `7fc10fc`.
    An ordered fixed-top-k policy, short-token 2-D partitioning, model-level
    precision validation, and the additional folded epilogues above remain open.
- [x] Prototype a precision-neutral W2 FP32 direct-route-store epilogue. Pass the
  M12 row destination table to assembly and write each N owner's disjoint H
  columns directly to `route_out`, eliminating the per-team `down` buffer and
  scatter copy. Measure whether irregular row stores reduce W2 throughput enough
  to offset the removed traffic before changing dispatch.
  - 2026-07-16 experiment: `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=1` wires M12 and
    M8/M4/M2/M1 SVE epilogues through sync, scheduled, and async bridges. On
    NUMA0 of the 192-core host, eight 12-thread experts improved by 15.82% at
    1536 routes/expert and 16.44% at 2040 routes/expert. The traced W2 critical
    worker regressed only 0.7-1.2%, while removing 1.33-1.95 ms of scatter and
    halving logical post-W2 traffic. At 12 routes/expert the median regressed
    0.95%, M=48 was neutral, and M=192 improved by only 2.18%. A later
    1001-pair M=48 run showed that the sub-10 us difference is below E2E noise,
    so direct route is now default-on; `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=0`
    retains the contiguous-down plus owner-scatter fallback.
  - [x] Measure uniform and complete captured distributions across short,
    medium, and long routes on the current AmazonC5192Cores binary, and generate
    the matching default FP32 direct-route iso/contention profile. The current
    bounded result is `-3.12/-0.69/+0.01/+4.36/+3.98%` for synthetic
    M=`12,48,192,1536,2040`; captured high-skew gains are `+2.68/+3.68%` at
    2048/4096 tokens, while several uniform/median points remain around the
    noise floor. Do not reuse the historical `15--16%` as a general claim.
  - [ ] Repeat the direct-route matrix on a second Arm machine through the
    frozen paper runner and retain route-buffer first-touch controls before
    promoting a cross-machine claim.
- [x] Add a direct BF16 route-store variant after the FP32 prototype. The SVE
  M12/M8/M4/M2/M1 W2 epilogues convert each result vector to BF16 and store it
  directly at the route-row destination; final top-k accumulation remains FP32.
  Sync, scheduled, and async bridges support all four FP32/BF16 and
  scatter/direct combinations.
  - 2026-07-16 experiment: on NUMA0 of the 192-core host, eight concurrent
    12-thread experts with 1536 routes/expert measured 7.726 ms for BF16 direct
    versus 8.087 ms for FP32 direct (median of three independent 31-run process
    medians), a 4.68% throughput gain and 4.47% latency reduction. BF16 direct
    was 6.25% lower latency than BF16-with-scatter and reduces modeled logical
    post-W2 payload from 384 MiB to 192 MiB. M=12 was neutral, M=48 gained 0.55%,
    and M=192 gained 1.31% in throughput. BF16 direct is bitwise identical to
    BF16-with-scatter; versus FP32 direct, local operator output maximum absolute
    error was `1.19e-7` on the long-route test.
  - [ ] Validate real `top_k=6` model-level accuracy before enabling BF16 route
    storage by default. `FUSED_CPP_MOE_W2_BF16_ROUTE=1` remains opt-in; when it
    is enabled, direct route store is selected by the existing default-on
    `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE` policy.
- [x] Make the Python `out=` argument a true native output buffer instead of
  allocating a native result and copying it afterward.
  - 2026-07-16 implementation: normal, scheduled, and async native entrypoints
    accept an optional contiguous CPU BF16 output and bind all final stores to
    that caller-owned storage. Python no longer performs a trailing `copy_`.
    Native validation rejects gradients, internal overlap, and overlap with
    input, packed weights, or routing tensors; successful writes increment the
    caller tensor's version counter. This removes the temporary final output
    but deliberately leaves the operator-owned TopK `route_out` workspace
    unchanged; collective overlap still requires a separate chunk-ready API.

P0 validation must report standalone W2, scatter, weighted merge, local operator
E2E, logical traffic, and model-level numerical error. Use uniform, hotspot, and
captured routing; routes must cover M12 bulk and M1/M2/M4/M8 tails. Performance
claims must include both long-route throughput and short-route latency.

### P1: overlap independent end-to-end stages

- [x] Prototype per-token route-completion accounting so idle lanes can merge
  tokens whose complete top-k set is ready while a long expert tail is still
  running. Keep expert work preferred over merge work.
  - 2026-07-16 implementation: the default async SVE direct-route path uses
    expert completion states,
    one publication RMW per expert, one CAS per ready token, and batched queue
    publication. Idle async workers merge one ready token at a time; the
    existing contiguous merge handles all leftovers after expert completion.
    A `ceil(M/12)/threads` max/min ratio below 1.25 keeps the old path. On the
    192-core host NUMA0, balanced and 25%/75% two-group distributions changed
    three-run medians by less than 0.4%; one trace of the latter merged 605/2048
    tokens early. The rejected per-route atomic counter prototype regressed
    23.35%. Set `FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=0` to retain the
    post-expert merge fallback.
  - [ ] Add captured and multi-wave heavy-tail distributions, model the
    expert/merge contention term, and revisit the 1.25 load threshold if the
    hidden merge time does not exceed queue/state overhead by the noise floor.
- [ ] Represent the shared expert as an independently schedulable job, run it
  concurrently with routed experts when resource contention permits, and fold
  shared-output/routed-scaling/residual addition into the final merge epilogue.
- [ ] Pipeline merged token/H chunks into async all-reduce or reduce-scatter and
  write directly into the communication buffer. Validate real TP/EP execution;
  do not infer this gain from the analytical communication model alone.
- [ ] Replace the duplicated Python `bincount` and C++ route scan with one
  contiguous CSR route-metadata build (`counts`, `offsets`, flat routes, token
  indices). Let a native interval planner consume the same metadata and avoid
  Python schedule-tensor materialization, with special attention to decode and
  many-layer control overhead.
- [x] Retire the former operator-wide and per-task range-policy experiments.
  Their implementation and active calibration files now live only in Git
  history; production uses full-N stages and width-derived owner stripes.
- [ ] Replace general async task-list polling with a lane-chain executor for the
  current disjoint interval plans, while retaining the general DAG path for plans
  with genuinely overlapping intervals. Measure dispatch and barrier overhead on
  route 1-48 before adopting it.

### P2: model-boundary experiments

- [ ] Prototype router-GEMM epilogue plus online top-k and route-histogram
  generation for the actual DeepSeek routing semantics. Prioritize decode and
  short-route latency; the prefill logits tensor is too small to assume a useful
  gain without measurement.
- [ ] Low priority: after direct FP32 route store is validated, optionally run a
  bounded experiment that multiplies each W2 row by its route weight in the W2
  epilogue so the final merge becomes an ordered sum. This does not remove the
  `route_out` write/read traffic and changes FP32 rounding relative to the
  current merge FMA, so reject it if W2 regresses by more than 1% or local E2E
  improves by less than 2%. Do not combine weighting with BF16 store until the
  additional rounding point has a separate model-level accuracy result.
- [ ] Evaluate a coarse ready-chunk communication pipeline that continues shared
  expert work and remaining route compute while earlier merged chunks are in
  flight. Include rank synchronization and collective launch overhead.

### Explicitly deprioritized fusion boundaries

- Do not prioritize full W2-to-merge fusion. The expert-major W2 schedule
  finishes one token's top-k routes in different teams and at different times;
  preserving ordered FP32 accumulation would require contended atomics, a
  rendezvous buffer, or a token-major schedule that sacrifices expert-weight
  locality. Reconsider only if a real multi-layer, cold-weight workload shows
  that weighted merge plus `route_out` traffic contributes at least 10-15% of
  local MoE E2E time after ready-token overlap. Prefer reusable `route_out`
  workspace, the existing ordered merge, and folding post-merge operations
  before changing this execution boundary. (`top_k=1, skip_weighted=true`
  already bypasses merge.)
- Do not pursue full W13-to-W2 fusion by default. Avoiding the BF16 `[M,F]`
  intermediate either recomputes W13 for W2 H tiles or repeatedly spills FP32 W2
  partial outputs; require a concrete loop ordering and traffic proof before an
  experiment.
- Do not move gather into every W13 N owner by default. It duplicates route
  address work and input reads by the team width; only a single-thread or very
  short-route microbenchmark can justify a specialized path.
- Do not prioritize further SiLU approximation work as an E2E optimization. Its
  measured contribution is only 2.3-3.7% of W13+W2 time. Reciprocal refinement
  and minimax3 were retired on 2026-08-10 after closed experiments; use the
  retained poly4/5/6 precision contract for any future model-level study.

The old AUTO selector, bucketized wave-DP, and wave-based Phase 4 items in
`FINDINGS.md` remain historical only because wave scheduling is deprecated.
They are not active TODOs for the async interval-DAG planner.
