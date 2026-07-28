# MoE scheduling implementation checklist

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
merge, the dynamic short-expert pool, the native cold planner, and stage-window
cost modeling have all landed. The next phase should not accumulate isolated
micro-optimizations. It should close the remaining high-value variables first,
then refresh the calibration data and cost model once and validate them as a
coherent system.

1. Produce independent thin analytical calibrations on AmazonECS8Cores and one
   NUMA rank of AmazonC5192Cores.
2. Validate unseen routes, widths, mixed distributions, split policy, and
   per-stage weight windows against the analytical acceptance gates.
3. Fix cross-rank lifetime switching for the remaining EP absolute-time error.
4. Add measured gather/pack, route merge, communication, and distributed TP/EP
   terms after the compute model passes its gates.

## 1. Schema-v2 profiling

- [x] Synchronize two NUMA-local ranks and aggregate pairwise wall maxima.
- [x] Use M12-aligned contention routes plus M1/M2/M4/M8 isolated tail points.
- [x] Stream distinct expert weights; use all local experts for full-call
  contention and eight experts for isolated streaming cost.
- [x] Use the runtime planner's LPT lane assignment for mixed-width shapes.

## 2. Policy-aware exact-shape cost model

- [x] Key `T_iso` and contention data by the complete profile policy: sharded F,
  split-W13 mode, kernel identity, NUMA topology, and concurrent-rank count.
- [x] Use exact measured shape data by default instead of collapsing all shapes
  with the same active-expert count into one derate.
- [x] Preserve once-per-call cost with authoritative full-call anchors rather
  than multiplying normalized group cost by the number of waves.
- [x] Make route lookup M12-aware, including the M1/M2/M4/M8 tail kernels.
- [x] Reject incompatible, ambiguous, or grid-mismatched profiles.

## 3. Joint policy and schedule planner

- [x] Search `(w13_split, core_shape)` rather than core shape alone.
- [x] Use packed working-set bytes to prune candidates, while retaining measured
  latency as the objective.
- [x] Return split policy as an explicit plan field and replace the
  process-global environment-variable control with an operator argument.
- [x] Add explicit physical CPU sets, a policy-aware cache key, full bucketed
  routing signatures, and confidence-aware tie breaking.

## 4. TP/EP layer evaluator

- [x] Select compute profiles from TP/EP degree, sharded F, local expert count,
  and machine topology instead of hard-coded TP4/EP4 files.
- [x] Evaluate global compute wall time as the maximum of rank-local plans.
- [x] Use rank-local routing histograms for EP load imbalance.
- [x] Generalize all-reduce/all-to-all communication formulas beyond the fixed
  P=4 two-pair topology.

## 5. Stage-aware working-set model

- [x] Add W13 chunk and W2 phases to the event simulation if exact-policy tables
  do not generalize adequately across F or unseen shapes.
- [x] Derive contention from the sum of active phase working sets rather than
  only the number of active expert tasks.
- [x] Add exhaustive dual-rank uniform/hotspot regret validation and an input for
  real routing histograms.

## 5.1 Analytical backend migration (2026-07-26)

- [x] Separate logical GEMM work, exact SVE kernel demand, and machine response.
- [x] Replace route/thread latency lookup with cache-capacity formulas and
  route-independent matrix/L1/L2/LLC/DRAM service curves.
- [x] Derive contention from per-resource aggregate requested rate inside the
  W13/W2 range event simulator.
- [x] Let the analytical backend generate homogeneous and two-width shapes while
  preserving empirical profile behavior.
- [x] Add an empirical-holdout validator for isolated error, contention error,
  and measured planner regret.
- [ ] Produce independent thin calibrations on AmazonECS8Cores and one NUMA rank
  of AmazonC5192Cores. Do not infer service ceilings from the route/thread table.
- [ ] Validate unseen routes, widths, mixed distributions, split/no-split, and
  byte-window policies against the acceptance gates in
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

- [x] Commit the 2026-07-13 TP2/EP2 split/no-split profiles, the expanded
  contention route grid, and the validator noise/stage-trace modes as separate
  reviewable changes.
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
- [x] Close W13-only first-panel software prefetch as a production candidate.
  It hides cold-B latency under low or moderate concurrency, but crosses over
  near 18-20 concurrent streams on AmazonC5192Cores and can regress reusable-B
  long routes. Keep the explicit environment flag and benchmark as regression
  evidence, but leave production dispatch off and do not add a planner gate.
  Reopen only with a new hardware result or a cache-retaining prefetch policy.
- [x] Add an offline no-contention CP-SAT oracle for the unrestricted supplied
  width set. It selects one fixed thread width per expert, removes dominated
  modes, returns incumbent and best-bound regret intervals, and compares the
  current strict plan with the same quantized `T_iso`. Keep OR-Tools optional
  and outside the production planner.
- [ ] Add conservative per-mode hardware lower-bound durations and aggregate
  matrix/LLC/DRAM capacity bounds before interpreting the CP-SAT result as a
  physical performance certificate rather than a `T_iso` model oracle.
- [ ] Measure lane-tail weighted idle loss on a larger routing corpus. Current
  TP cases have a perfect-rebalance upper bound of only 0.7-2.4%; prototype
  same-width-lane work stealing only if representative cases repeatedly exceed
  5%.
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
  - 2026-07-16 experiment: `FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL` provides
    U1/U2/U4 SVE variants. `top_k=2/4/6/8` uses compile-time adjacent binary
    trees; other values use a one-token-row ordered runtime-loop fallback. The
    legacy `FUSED_CPP_MOE_SVE_ROUTE_MERGE_TREE_UNROLL` name remains an alias.
    On the 192-core host NUMA0, isolated BF16-route gains were 11-24% across
    top-k 2-8, while FP32 route regressed for top-k 7-8. Preplanned async E2E
    gains remained at or below 1.04%. U1 is now the SVE default to eliminate the
    per-worker FP32 allocation and accumulator traffic; explicit value `0`
    retains the sequential compatibility path, while U2/U4 remain experimental.
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
  - [ ] Measure uniform, hotspot, and captured distributions across short and
    medium routes, and regenerate iso/contention profiles under the default
    direct-route path. Existing profiles predate this default and represent the
    contiguous-down plus scatter implementation. Separate route-buffer
    first-touch page faults from steady kernel time when evaluating tail latency.
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
- [x] Promote the operator-wide packed-B window into planner policy identity and
  schema-v2. Jointly search measured `(weight_window_bytes, core_shape)` variants,
  model the actual W13/W2 range counts, and forward the selected option to the
  async kernel. Historical profiles map to window 0.
- [x] Extend Plan V2 with independent per-task W13 and W2 window overrides and
  add a named deterministic post-plan policy hook. Unsupported route/width
  combinations inherit the selected operator-wide policy, so this does not
  enlarge the planner search space.
- [ ] Regenerate isolated/contention calibration for independently selected
  W13/W2 windows before admitting those combinations to cost-model scoring or
  expanding the default beyond the exact dual-NUMA AmazonC5192Cores
  TP4/F512 split profile. The measured profile-bound rule is a deterministic
  runtime exception, not a scored candidate.
- [ ] Replace general async task-list polling with a lane-chain executor for the
  current disjoint interval plans, while retaining the general DAG path for plans
  with genuinely overlapping intervals. Measure dispatch and barrier overhead on
  route 1-48 before adopting it.

### P2: model-boundary experiments

- [ ] Prototype router-GEMM epilogue plus online top-k and route-histogram
  generation for the actual DeepSeek routing semantics. Prioritize decode and
  short-route latency; the prefill logits tensor is too small to assume a useful
  gain without measurement.
- [ ] After direct FP32 route store is validated, optionally multiply each W2 row
  by its route weight in the W2 epilogue so the final merge becomes an ordered
  sum. Do not combine weighting with BF16 store until the additional rounding
  point has a separate model-level accuracy result.
- [ ] Evaluate a coarse ready-chunk communication pipeline that continues shared
  expert work and remaining route compute while earlier merged chunks are in
  flight. Include rank synchronization and collective launch overhead.

### Explicitly deprioritized fusion boundaries

- Do not pursue full W13-to-W2 fusion by default. Avoiding the BF16 `[M,F]`
  intermediate either recomputes W13 for W2 H tiles or repeatedly spills FP32 W2
  partial outputs; require a concrete loop ordering and traffic proof before an
  experiment.
- Do not move gather into every W13 N owner by default. It duplicates route
  address work and input reads by the team width; only a single-thread or very
  short-route microbenchmark can justify a specialized path.
- Do not prioritize further SiLU approximation work as an E2E optimization. Its
  measured contribution is only 2.3-3.7% of W13+W2 time; retain polynomial and
  reciprocal variants as accuracy/performance experiments, not the main fusion
  program.

The old AUTO selector, bucketized wave-DP, and wave-based Phase 4 items in
`FINDINGS.md` remain historical only because wave scheduling is deprecated.
They are not active TODOs for the async interval-DAG planner.
