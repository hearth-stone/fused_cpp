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
merge, the dynamic short-expert pool, the native cold planner, and stage-aware
cost modeling have all landed. Legacy split/range controls were replaced by the
single `(threads, window_tiles)` stage geometry: a task still covers full N, but
may do so as multiple serial owner windows. The next phase should not accumulate
isolated micro-optimizations. It should close the remaining high-value variables
first, then refresh calibration and validate the system coherently.

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
- [ ] Measure independent exact-M M1-M11 core-efficiency ratios relative to the
  M12 L1-hot service. Admit them only if they improve unseen tail routes without
  becoming another route/thread latency table.
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
  from independent probes; do not add a task-pair slowdown table.
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
  measured contribution is only 2.3-3.7% of W13+W2 time. Reciprocal refinement
  and minimax3 were retired on 2026-08-10 after closed experiments; use the
  retained poly4/5/6 precision contract for any future model-level study.

The old AUTO selector, bucketized wave-DP, and wave-based Phase 4 items in
`FINDINGS.md` remain historical only because wave scheduling is deprecated.
They are not active TODOs for the async interval-DAG planner.
