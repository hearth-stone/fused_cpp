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

## Open code work (refreshed 2026-09-22)

Consolidated from `CURRENT.json`, the sections below, the decision records and
the working tree. It indexes open work and adds the items no section recorded
yet; the detailed sections stay authoritative for their own history.
Paper-facing items live in
[`../docs/moe_paper_readiness.md`](../docs/moe_paper_readiness.md) ("Paper
TODO"). Live run status stays in `optimizations/fused_moe_sve/CURRENT.json`.

Both measurement machines have been unreachable since 2026-09-21 (`Arm-codex`
rejects the SSH key, C9g times out), so everything that needs hardware is
parked behind access rather than behind a decision. Items marked **needs a
machine** cannot start until then.

Closed since the 2026-09-20 review:

- [x] Window table under the thread-major order. The 2T rows were re-measured
  in the new order rather than carried over from V3, and E11 validated the
  table on 18 unused layers: V4 was 0.27% faster (16/18), inside the frozen
  0.3% tie band, and the same batch put windows at 1.69% against no windows
  (v1.117). V4 is registered in place of V3 on the user's decision - lineage
  consistency, not performance (v1.118).
- [x] Native (C++) hot-wide planner. `csrc/moe_planner/hot_wide_planner.{h,cpp}`
  with `NativeHotWidePlanner` bound through `quick_bindings.cpp`: 0.703 ms ->
  0.205 ms per layer (3.4x) with 45/45 plans identical to the Python planner
  (`ec85f54`, v1.123). This also removes the cost argument that blocked a short
  event-scored refinement pass; see the fast-planner gap below.
- [x] 2T lanes, as a model and product question. The mechanism is the live
  weight footprint of an LLC domain, not the lane width: at matched footprint
  the widths agree to 0.008-0.083, at matched width with the footprint free
  they spread by 0.40 (P9, v1.121). The value is bounded at 0.82% by measured
  per-expert core time, and six hand-built 2T variants lost on 18/18 layers,
  with the "keep small experts off 2T" rule measuring 5.10% *slower* (E13,
  v1.120). 2T stays out of the search space and gets no model correction. What
  is still open is the code audit, below.
- [x] Window tables resolving on shape alone - see the one-click section below
  (`ca843f3`, v1.125).

P0, code that blocks a paper table:

- [ ] **Needs a machine.** Upstream baseline timing on Arm (R2 of "Related-work
  comparison plan (adopted 2026-09-21)" below, which fixes its scope and
  attribution). `optimizations/moe_upstream_baselines/` is correctness-only and
  was run on x86, where the `fused_cpp` backend is skipped (28 passed, 7
  skipped). Next: build it on Arm-codex, run the correctness suite with
  `fused_cpp` included, then add a timing harness that follows
  `docs/agent_benchmark_hygiene.md` (per-backend thread placement,
  upstream-recommended flags, jemalloc, identical routes). vLLM `neon` is the
  only upstream with a real Arm BF16 path; llama.cpp BF16 `vec_dot` is scalar
  and SGLang runs a torch loop, so state what each row means. The clamped
  SwiGLU (`swiglu_limit=10`) case is not exercised yet.
- [ ] **Needs a machine.** Four-rank concurrent validation (see "Deferred
  external validation"). It bounds every current claim, which is one TP rank
  with three idle nodes. E10 already shows cross-node DRAM traffic is visible
  (+0.8--13.1% from streaming load), so expect a level shift; the question is
  whether gains and ranking survive. C9g is a cheaper partial answer: two nodes,
  and its per-domain footprint is already 2-4x Arm-codex's.
- [ ] **Needs a machine.** Final planner matrix P4 (43 layers x 3 requests)
  through the public runtime path with `T_plan + T_execute`, after P1 fixes the
  planner definition. The definition has moved since P1 was written: the
  deployed fast path is now `hot_wide` (`enable_moe_planner_fast()`), not the
  old homogeneous quick, so P1 and P3 need rewording before P4 runs. `T_plan`
  is now the native planner's 0.205 ms, not Python's 6.6 ms.
- [ ] **Needs a machine.** Independent repeat of E2/E3/E4. Each is one session
  set; the talk outline lists this as a gap. One repeat on fresh layers is
  enough to put a spread on the 12.9% / 9.6% headline numbers. Fold it into P4
  rather than running it separately if P4 happens first.
- [ ] **Needs a machine.** Cumulative K1 ablation on one frozen binary (static
  kernel, exact-M, adaptive gather-pack, fused W13, W2 direct route, windows,
  merge). Today every row comes from a different date, machine, and build,
  several glibc-era.
- [ ] **Needs a machine.** Fused W13 cache blocking in production.
  `tmp/fusion_stage_20260920` measured the fused path 6--8% slower than the
  explicit pipeline at M=24--192 on 20 busy 4T lanes without blocking and up to
  15.6% faster with `--w13-ranges 8`, in the standalone comparator only,
  blocking on the fused side only, one session for the extension run, no PMU.
  Open: check how this relates to the production W13 window (`w13 = 1 tile` in
  most bands), repeat with a second session and L2 refill counters, and only
  then decide whether a kernel or policy change follows.
- [ ] Calibration cost as one number: probe points and wall time for P1--P8 plus
  the window table, against the 418-point / 18 min 25 s empirical profile. Two
  measured anchors exist now: the C9g service probe at 21 s and its training
  profile at 60--106 s per shape.
- [ ] Artifact freeze prerequisites: move the remaining provisional cases into
  `paper_experiments/` (section 0) and add suites for the 2026-09-20..22
  experiments (M2/M4, E2--E7, E9--E14, P9) so the frozen commit can regenerate
  them.

P1, product and model follow-ups (user decisions pending in `CURRENT.json`):

- [ ] Finish the 2T width audit. `IntervalPlanner.widths` still falls back from
  the caller to the model's `supported_widths` to `_default_widths` (every power
  of two), while `reliable_widths` is consulted only for tail-pool candidates
  and the `model_lns` defaults, so production plans can still contain 2T lanes
  that the model mispredicts. E13 closed the question of whether 2T is worth
  predicting; it did not change this fallback chain. Audit which widths each
  production entry point actually emits, then extend the filter to every
  candidate source or state why not. Note that the answer is machine-specific:
  on C9g 2T is the *fastest* width at every M up to 192 (v1.123), so the filter
  must be a property of the calibration, not a constant.
- [ ] The ~1.09 level bias. Decomposed 2026-09-21 over 546 measured plans:
  15.3% of the error variance is the per-layer level and 84.7% is between plans
  of one layer, so the bias is not one constant and a near-constant correction
  cannot fix ranking - which is exactly how v12, v13 and v14 failed (v1.124).
  Still unexplained mechanically; report it as a level plus a ranking residual
  rather than as one number.
- [ ] **Needs a machine.** E15: the candidate ranking residual
  `0.61*window_credit - 2.94*loading_share`, frozen in
  `tmp/v15_candidate_20260921/analysis.md`. It survives leave-one-workload-out
  and leave-one-set-out on the existing corpus, which v14 never faced, but by
  the E14 precedent it must be shown on layers nobody has run. Gates weighted to
  ranking and selection, the reverse of E14's emphasis.
- [ ] **Needs a machine.** Load-context window choice (M4 defect C1: the static
  choice loses up to 2.7%, the gain of one window ranges 2--21% with load).
  Blocked on the stop rule unless it changes a plan choice by >= 2% in both
  sessions.
- [ ] Small-M slowdown under load (S1 fail, under-predicted by up to 0.4) and
  the `p16_r4` preference on 8T-type layers: recorded as limitations unless P4
  shows they cost a gate. C9g's calibration puts a number on the isolated half
  of this: the `M < threads` region is over-predicted by up to 42% because the
  physics charges every core while most hold no row, and a non-negative overhead
  cannot subtract it.
- [ ] Fast planner gap to the reference (3.8%, event-level phase overlap). No
  cheap rule closed it, but the precondition named in the 2026-09-20 review is
  now met: the native planner is 3.4x faster, so a short event-scored refinement
  pass is affordable. Decide whether to spend that budget on refinement or keep
  it as headroom for `T_plan`.
- [ ] The FP32 direct-route int32 offset limit (about 21,845 tokens at TopK=6):
  repair it or exclude 30k-token runs from the paper domain (section 0).
- [ ] User decisions still open: whether to power a proper V4-versus-V3 adoption
  test for a 0.3% effect; whether to calibrate 32T at all (the full-load grid
  cannot represent it and it needs a filler-load design); whether to commit or
  drop the unadopted v14 footprint code in
  `cost_model/probe_event_model.py` (+95 lines, measured but rejected in E14).

Correctness and second-machine follow-ups (2026-09-21..22):

- [ ] **Needs a machine.** W8A8 on a 128-bit SVE machine. Diagnosed statically
  as a vector-length assumption, not the race the first C9g pass reported:
  `i8gemm_k_hybrid` / `narrow` / `narrow2` store a fixed 16 int32 columns per
  row while their N loop steps `svcntb()/2`, so below 256 bits they write 8
  columns past each tile, into the neighbouring thread's stripe and past the
  accumulator's end. `refs/i8gemm/lib/i8gemm_msplit_k.S` shares the macro, and
  both W8A8 benchmarks call the hybrid kernel directly. The one-condition fix
  (`svcntb() != 32` forces the packed NEON path) is written down in
  `tmp/w8a8_vector_length_20260921/analysis.md` but deliberately not applied: it
  cannot be compiled or tested without a machine. Three falsifiable predictions
  are pre-registered there, the cheapest being that a `k > 1024, rows > 16`
  shape at width 2 must already be correct on C9g today.
- [ ] **Needs a machine.** The analytic model's missing per-panel term. The
  kernel re-streams an expert's weights once per M12 panel, so about 210 of
  every 495 us does not scale with the panel's rows; the parameter-free identity
  `cost(16) = cost(4) + cost(24) - cost(12)` lands within 0.5% where the
  calibrated model is off by 9.4% (`tmp/panel_boundary_20260921/analysis.md`).
  `phase_model._formula_iso` and `tiso_roofline.panel_histogram` already carry
  the structure - the latter has no consumer outside its own test - and
  `analytic_model` does not. Profile routes 13, 17, 20 and 36 to separate the
  two, and routes 9, 10, 13, 14, 17, 18 to separate the tail table's `routes %
  12` from the m8 dispatch's `routes % 8`.
- [ ] **Needs a machine.** Port the event-model probe curves to C9g; until then
  that machine has no contention side and cannot run the fast planner. Tracked
  with the rest of the portability work in the next section.

Repository hygiene (no measurements needed):

- [ ] Triage the uncommitted working tree: 270 untracked paths (was about 390),
  mostly the 2026-09-05..11 joint-model era (`results/*_202609*.md`,
  `benchmarks/analyze_*.py`, `tests/test_moe_*.py`), plus
  `moe_upstream_baselines/`, `scripts/experiment_remote_job.py` and its tests,
  `docs/moe_talk/`, and a 3099-line `manifest.yaml` addition. Per
  `docs/git_workflow.md`: commit what is still evidence or live tooling, archive
  the retired joint-model experiments, and keep the
  `WideTeamPressureCalibration.occupancy_scale` change with its tests as its own
  commit. Also still uncommitted and unrelated to this workstream: the
  shared-packed-A / vLLM staged-queue work in
  `csrc/moe/arm/common/fused_moe_bf16_tiled.cpp` and the files that move with
  it. 39 commits on local `main` are unpushed against `origin/main`, and the
  thread-major commit on `Arm-codex` was never pushed either.
- [ ] Mark superseded items. "Cost model track after the jemalloc switch" still
  lists model revisions that v10/v11 replaced (width residual, DRAM demand
  rule, window-aware event model); "0. Paper-critical closure" still describes
  the quick regression against fixed 8T that the `hot_wide` adoption changed.
  Close or annotate them so the open count reflects real work.
- [ ] Switch remaining consumers (LNS tool defaults, machine configs, frozen
  anchor plans chosen under v8) to the current model only by separate decision;
  until then list which consumer uses which calibration in one place.

## One-click calibration and deployment portability (2026-09-21)

What a new ARM machine or a new parallel shape needs today, and what is missing
before "calibrate once, then just call it" holds. The machine layer is already
shape-independent and automatic: `calibrate_moe_planner_quick` probes cache
geometry and service rates with synthetic micro-geometries that never read the
expert's H/F, detects `backend_n_tile` and the LLC domains, and derives a
machine id. `enable_moe_planner_quick` takes the shape as arguments, so TP4 to
TP2 needs no new machine probe. Four gaps remain.

- [x] Window tables resolved on shape alone. `AMAZON_C5_192C_TP4_F512_V5`
  carried no `machine_ids`, so every machine with H=4096, F=512 and n_tile 8
  inherited it - C9g's TP4 shape exactly, whose own grid selects different
  windows. Fixed 2026-09-21: the table names the two NUMA0 calibrations it was
  measured against, and `_validate_registry` refuses a registered table without
  `machine_ids`. Unregistered machines now keep the full stripe.
- [ ] `overheads` are zero after a quick calibration.
  `build_analytic_calibration.build_calibration` defaults
  `expert_fixed_ns`/`route_ns` to 0 and only `--training-profile` fills them;
  `calibrate_moe_planner_quick` does not pass one. Measured, the term is
  44.3 us + 264.6 ns/route on C9g TP4, 80.2 us + 127.0 ns on TP2 and 210 us +
  3373 ns on Arm-codex, so small-M is systematically under-predicted. The
  training profile costs 60-106 s per shape (`profile_contention_async.py` on
  C9g), so folding it into `enable_moe_planner_quick` behind the H/F it already
  takes would close the gap at a bounded cost. Shape-dependent: it must rerun
  when the parallel strategy changes.
- [ ] Event-model probe curves are per machine and outside the one-click path.
  `enable_moe_planner_fast` needs a probe-calibrated event model file that only
  a full probe run produces, so C9g can use the quick runtime but not the fast
  planner or anything on the contention side.
- [ ] Profile-based models carry no machine identity. `ContentionCostModel`
  exposes `policy` but no `calibration`, so `machine_id` is always `None` and,
  after the fix above, that path never resolves a window table. Before the fix
  it resolved one by shape regardless of machine, which is the same defect. If
  windows are wanted there, the profile payload needs a machine id; until then
  the empirical path is full-stripe by construction.
- [ ] The C5 machine's own quick-calibration id is unknown, so a quick
  calibration there no longer resolves its table. Record the id the next time
  that machine is used and add it, or re-measure the table under a current
  calibration.

Making the first three true would give: one machine probe per machine, one
short shape probe per parallel strategy, one registered window table per
(machine, shape), and plan search at runtime keyed on the bucketed route
histogram. Only the last of these is per-request.

## Related-work comparison plan (adopted 2026-09-21)

Purpose: give the paper an attributable comparison against the CPU MoE systems
it positions against, instead of one wall-clock number that mixes microkernel
and scheduling differences. Positioning, the verified code evidence for
"different microkernels, same scheduling structure", and the 2x2 table are in
[`../docs/moe_paper_readiness.md`](../docs/moe_paper_readiness.md)
("Related-work comparison design"). This section holds the experiments,
acceptance and cost. Supersedes the scope of the "Upstream baseline timing on
Arm" item above, which stays open as the R2 work.

Measured cells: **A** vLLM Arm op (upstream kernel, flat queue), **B**
`staged_queue` control (our kernel, flat queue), **C** production quick/full
(our kernel, planned schedule). `A -> B` is contribution 1, `B -> C` is
contribution 3, `A -> C` is context only.

Arm-codex NUMA3 is one shared measurement resource and the window-table chain
(E11) is in flight in another session. R1-R3 queue behind it.

### R1 Scheduling ablation (B versus C) - the only experiment that licenses a scheduling claim

- [ ] P0 code. Derive `staged_queue` from `fused_moe_bf16_tiled_vllm_staged`
  (`csrc/moe/arm/common/fused_moe_bf16_tiled.cpp:4491`). Two deltas only:
  (a) replace the per-N-task A rescan with one gather plus one pack per expert,
  matching KTransformers (`kt-kernel/operators/amx/moe_base.hpp:310` and `:327`)
  rather than vLLM; (b) make the column-block size a benchmark-settable
  parameter instead of deriving it from the L2 budget. Everything else -
  packing, fused W13 SiLU epilogue, direct-route store, SVE merge, numerical
  contract - stays identical to production. Experimental entry point only: no
  change to default dispatch, Plan V2, or any public contract.
- [ ] Workload axis, ordered by route heterogeneity, all at 2048 tokens,
  `H=4096`, `F=512`, `E=256`, TopK=6: `moe256-uniform`,
  `moe256-active-set-{32,64}`, `moe256-tiered-hotspot`,
  `moe256-long-short-bimodal`, `dsv4-real-2048-seq70` (223 active, max 918,
  mean 55.1, std 119.6; head of six experts at 500-918 routes, a mass at 27-28,
  71 single-route experts). The figure is relative `B -> C` gain against this
  axis, not a single number.
- [ ] Arms: `staged_queue` with its block size swept ({128, 256, 512,
  L2-derived}, report its own best), fixed-width homogeneous LPT for every
  calibrated legal width, quick, full. Report `T_plan + T_execute`, not
  execution time alone.
- Expectation before running, from the retired 192C series
  (`optimizations/fused_moe_sve/results/amazon_192c_vllm_staged_schedule.md`,
  geometry retired, shape reference only): the flat queue lost 13.07% on six
  long experts and tied fixed-8T (-0.82%) on 256 uniform `M=48` experts. The
  paper claim is therefore conditional - when per-expert assignment matters -
  not a universal win.

### R2 External baseline (A) - closes the P0 upstream-baseline gate

- [ ] `optimizations/moe_upstream_baselines/` already selects isa `neon` and
  `-march=armv8.2-a+bf16+dotprod+fp16 -DARM_BF16_SUPPORT` on aarch64
  (`vllm_cpu_moe/__init__.py:21,33`); it has never been built or run on
  Arm-codex. Run the correctness suite there first, with `fused_cpp` included,
  against the frozen acceptance (relative L2 <= 1e-2 and
  `assert_close(atol=7e-2, rtol=7e-2)`); the existing error table is x86.
- [ ] Then time it on the R1 shapes and routes. vLLM uses `#pragma omp parallel
  for`, so pin with `OMP_NUM_THREADS=80 OMP_PROC_BIND=close` under the same
  node-3 placement, and record that its thread model differs from our executor.
- [ ] Shared expert follows the contract already fixed in the lab README
  (vLLM applies it outside the op with the scaling folded into router weights).
- llama.cpp may be added as a third kernel point but stays out of the
  performance table: its Arm BF16 `vec_dot` is scalar and is a correctness
  oracle only.

### R3 Mechanism (B versus C)

- [ ] Process-level PMU on two or three representative workloads: concurrent
  weight footprint over time plus LLC/DDRC counters, via `--profile-variant`
  and `benchmarks/linux_perf_event.py`. Purpose is to tie `B -> C` to the P9
  result that the contention state variable is the footprint, not the lane
  width. Do not claim a single hardware mechanism; that test already failed.

### Protocol and acceptance

- jemalloc preloaded with purging disabled; refuse to run when it is not mapped.
- node3 `numactl --physcpubind=240-319 --membind=3`, exclusive. During a
  measurement, nodes 0-2 may run compute-bound work only; no memory-streaming
  work anywhere (E10: +13.1% from three streaming nodes).
- One process per data point; 3 warmups, 11 runs, A/B/B/A interleaving, output
  checked bitwise before timing.
- Record commit and uncommitted diff, `.so` sha256, `LD_PRELOAD`/`MALLOC_CONF`,
  team width and backend N tile, W13/W2 stage bytes, affinity.
- Resolution gate: measured noise is 0.38% (E6 repeat) with a 0.261% median
  session difference inside one build. Any cell under 1% is reported as a
  measured tie, never as a win.

### Phases and cost

| Phase | Work | Machine | Estimate |
| --- | --- | --- | --- |
| P0 | `staged_queue` implementation and correctness tests | no | half a day |
| P1 | R1 grid | yes | 30-60 min |
| P2 | R2 correctness then timing | yes | 1-2 h including first build |
| P3 | R3 PMU | yes | 30 min |
| P4 | Tables, figures, related-work text | no | one day |

Machine time is about 2-3.5 h, inside the 2026-09-15 standing authorization.

### Risks

- "Your kernel was co-designed with your scheduler." Mitigation: the control
  gets the same packing, fusion and one-shot gather, its block size is swept,
  and R3 reports the mechanism rather than only the number.
- A tie on uniform workloads is half the thesis, not a failure; the
  heterogeneity axis exists to express that.
- Every number remains one TP rank on one 80-core node with three idle nodes
  until the four-rank validation closes. The abstract says "MoE operator time of
  one TP rank" until then.

## Cost model and planner plan (adopted 2026-09-19)

Purpose: turn the open-ended mechanism diagnosis into a bounded path to a paper
claim. The acceptance target is decision quality (does the model pick a good
plan), not per-term physical fidelity. Evidence so far: the large measured wins
are coarse structural choices (team width 5--18%, hot expert pinned on a wide
lane or windows 2--10%, tail pool about 30% on bimodal); fine order/transfer
moves are 2--4% or inside noise under the resident-workspace baseline.
Paper status: [`../docs/moe_paper_readiness.md`](../docs/moe_paper_readiness.md).

Agreed positioning (2026-09-19): the cost model serves two decisions, kernel
tile windows and planner time prediction, and is accepted on selection regret
and ranking; MAPE/P90 are diagnostics. The planner's product is quick (fast plan,
small `T_plan`); full/LNS is the reference for "how far is quick from a carefully
searched plan", not a deployed algorithm or an oracle.

User decisions (2026-09-19):

- [x] One fused model in the paper. The analytical event model supplies the
  skeleton (phases, event simulation, LLC-domain placement; widths 1--80T, whole
  layer plans, the planner's consumer). The joint run model
  (`tmp/joint_cost_model_20260911`) supplies mechanism: declared and conserved
  per-stage read/write request budgets calibrated against DDRC bytes, and a
  contention response split into globally shared (DRAM) and same-LLC-domain
  (LLC, L2 refill) parts. Width-indexed coefficients (B_t/S_t, c1/c2) survive
  only as a bounded residual whose size the paper reports. Only structure and
  holdout-verified facts are imported from the joint model; it did not pass its
  own acceptance and is not imported wholesale. Its queue response, block-history
  demand, and W13-to-W2 migration are out of v10 (future work).
- [x] Bounded revision: exactly one revision round (v10) below, then freeze,
  whatever the gate result.

Track M (cost model), in order:

- [x] M1a Lane-level spans of the `tmp/single_expert_width_20260919` traces
  versus v9: no neighbor-of-wide-team term exists. A 4T M=384 task takes
  15.5-16.3 ms next to 12-20 lanes with or without a 32T team (load slowdown about
  1.05-1.10, flat), while v9 predicts a cliff (1.015 at 12 lanes, 1.43 at 20).
  The 32T-loaded whole-call gap = 4T T_iso 7-9% low + that flat slowdown +
  pre-compute/merge time outside the makespan (4-6% of the call). Missing data:
  slowdown at 1-10 lanes and one- versus two-domain placement.
  [Record](../tmp/single_expert_width_20260919/decision.md).
- [x] M1b Joint-model inventory:
  [`inventory.md`](../tmp/joint_model_inventory_20260919/inventory.md).
  Importable (holdout-passed or two-session repeat): conserved per-call request
  budgets with realized rate = budget x speed; per-stage read budgets by width
  and M; only measured DRAM plateau 292-294 GB/s read (170 GB/s refuted as a
  capacity, the model's 380 GB/s unverified); slowdown convex in total DRAM load,
  multiplicative on T0 and zero when neighbors' weights are cache-resident;
  8T GEMM/gather contention globally shared, a same-domain share for narrow
  foregrounds (local about 2x cross); narrow solo stage costs transfer within
  0.6-1.6%. Not importable: all fitted capacity numbers, the rational response
  form, M/pressure interpolation, history models. Gap: every high-load probe ran
  at 141-276 GB/s background read, real loaded plans measure 44-71 GB/s, and the
  report has no 32T data.
- [x] M2 frozen 2026-09-19:
  [`tmp/m2_validation_20260919/decision.md`](../tmp/m2_validation_20260919/decision.md)
  (group W: 18 unseen-layer workloads x 17 frozen bridges; group S: same task in
  different plans, with disjoint fit and validation cells; group N: V3 windows at
  unseen M). Measured once in M4; group N may run earlier. Original item:
  Freeze the validation set and gates before fitting. Whole plans on at
  least 6 layers x 3 captured requests, families: homogeneous 4/8/16T, hot
  expert pinned on 16/32T plus narrow lanes, mixed width, windowed (V3). Two
  sessions, jemalloc, 31 paired samples. Gates: per-family measured/predicted
  median within 0.92--1.08; selection regret@1 <= 5% on every layer; report
  regret@2/@3 and Spearman. Fit data and validation layers are disjoint.
  Window selection is its own group: per (t, M) band, regret of the selected
  (w13, w2) window against the measured best window, gate <= 5%.
  Plan-dependent slowdown is its own group: the same task (width, M) placed in
  different plans; gate on the predicted versus measured slowdown difference,
  with the four-step ablation (no contention / width coefficients only /
  mechanism / mechanism plus residual) reported on it.
- [x] M2b decided by an offline check (2026-09-19,
  [record](../tmp/window_model_check_20260919/decision.md)): option B for v10. On
  the 36 full-load 80C cells the v9 `score_stage_window` pick has measured regret
  median 4.96%, P90 17%, max 24% and hits the best window in 2/36 cells (always
  FULL: 8.0/39/67%; V3 in-sample 0.14/2.1/2.7%). The objective is binary in the
  owner window and has no W2 sensitivity; 80C lacks a machine-local packed-B
  retention calibration. Task-level window gains under full load are 17-40% (2T),
  4-27% (4T), 3-16% (8T), 2-8% (16T), so v10 must apply r(t, M) in the event model
  used by full/LNS, not only in quick cost. Load-context dependence of the window
  choice is tested by M2 group C (Amendment 1); a retention probe that would enable
  option A is optional and separate. Original item:
  Decide the relation between the window table and the event model,
  which does not see windows today (`dag_makespan` ignores them; the 80C table
  is measured per (t, M)). Option A: the event model explains/predicts the
  window choice on 80C as policy v6 did on 192C. Option B: the table is a
  declared calibration artifact (model structure plus N measured points) and
  the paper reports its calibration cost. Supersedes the open "make the event
  model window-aware" item below.
- [x] M2 groups N and C and the S fit cells measured 2026-09-19
  ([record](../tmp/m2_validation_20260919/decision.md), After collection).
  N passes after a reported scoring fix (32/32 unseen-M cells <= 5% regret, max
  3.65%; window gain 18-46% at 2T, 4.5-36% at 4T, 3-25% at 8T). C1 fails: the
  static window choice loses up to 2.7% (5% on a 0.6 ms 16T task) in small-M and
  large-M backgrounds, and the gain of the same window ranges from 2% (isolated)
  to 21% (full same-M load), so r(t, M) needs a load context before the choice
  does. S fit: slowdown of a 4T task is ~0 up to 5 concurrent lanes, then 1.05 /
  1.08 / 1.12 at 9 / 14 / 19 lanes (M 384; 1.03 / 1.04 / 1.06 at M 1024), 1.24 next to
  19 lanes of chained M-24 experts; the added time looks per-expert additive.
  Small-M targets slow down 1.2-1.6x under full load.
- [x] M3 done as a probe-calibrated v10 (MATHEMATICAL_MODEL v1.107; user decision
  2026-09-19: no regression on plan timings). Four measured service curves
  D_LL/D_LS/D_SL/D_SS(t, n) over loading/steady phases, composition rule frozen
  before measurement, isolated time (1 + 0.065) x v9 phases + O(t), t_over 0.78 ms
  measured; no B_t/S_t, no spill rule, no residual. Contention sits almost entirely
  between weight-loading phases (D_LL up to 1.6-2.0, D_SS <= 1.03); the single
  hardware-curve test failed (width-indexed table). A first regression attempt
  (ladder of fitted forms) is kept only as an ablation baseline.
  [Probes](../tmp/v10_probes_20260919/decision.md),
  [model](../tmp/v10_model_20260919/v10.py). Superseded plan text:
  Revision v10 (M class, MATHEMATICAL_MODEL update), fused model:
  (a) memory demand: per-stage read/write request budgets, conserved, calibrated
  against DDRC bytes, replacing spill 1.0 for concurrent lanes; (b) contention
  response split into globally shared (DRAM) and same-LLC-domain parts, carrying
  the non-additive high-total-load slowdown; (c) intra-team overhead (today
  B_t) as a fixed per-expert cost (fork/join, sync, gather) amortized by
  per-expert work; (d) slowdown from neighboring teams (today S_t - B_t) comes
  from (b), with the unexplained remainder kept as a reported bounded residual;
  (e) no neighbor-of-wide-team term (M1a found none); whole-call predictions add
  pre-compute and route-merge time. Inputs limited to the M1b inventory. Refit on A1 layer data plus the single-expert cells only.
- [x] M4 measured once, 2026-09-20
  ([record](../tmp/m2_validation_20260919/decision.md)): W1 PASS (measured /
  predicted 1.05-1.08 in every family); W2 FAIL on one workload by 0.003 points
  (regret@1 median 0, max 5.003%, 17/18 <= 5%; regret@2 max 3.4%; v9 event max
  10.4%, frozen quick cost max 15.1%); S2 PASS (slowdown MAE 0.030 vs null 0.062,
  v9 0.043), S1 FAIL (small-M target under-predicted by up to 0.4). Narrowed
  claim adopted: two-plan shortlist within 3.4% plus the measured-consensus tuner;
  contention layer reported as a measured service table valid for large-M tasks.
  No second revision round. Open defects: 5-8% level bias, p16_r4 preference on
  8T-type layers, small-M slowdown, static window choice (C1). Original item:
  Run M2 once. Pass: v10 is the 80C paper model; switch consumers by
  separate decision. Fail: keep v10 frozen and narrow the claim to "the model
  prunes to a k-plan shortlist and the measured-consensus tuner
  (`bench_selection_pair.py tune`) picks", reporting regret@k and tuning cost.
  No second revision round before the non-model paper gates have evidence.
- [ ] M5 Deferred until M4: service re-probe under jemalloc, A2 1T split
  identifiability, window-aware event model, second machine calibration.
  Do each only if M4 shows it limits a gate.

Track P (planner):

- [x] Reference objective (user decision 2026-09-19): the offline reference is
  searched and reported on the event objective (`dag_makespan_placed`), the same
  quantity the cost model predicts; the robust objective (max with 1.15 x heaviest
  lane T_iso) is not the yardstick. Needs a model-objective LNS loop and the
  compound split-and-reassign move from P2.
- [ ] P1 Fix the paper's planner definition: request-path quick over a small
  structured family (homogeneous widths, hot-pinned shapes, tail pool; windows
  from the table), and the event-guided full + fixed-mixture LNS as the offline
  reference used to measure quick regret. No ALNS; CP-SAT only for reduced
  exact instances.
- [x] P2 first round done:
  [`record`](../tmp/planner_reduced_optimum_20260919/decision.md). Exact optima
  for E=6 (exhaustive) and E=8 (branch and bound, conditional on "no task runs
  faster in a plan than alone", 0 violations in 1.15 M audited states); E>=16 not
  tractable. Reference (best of full/one_step/VND/LNS) gap to optimum on the event
  objective: median 0, max 7.26% (frozen gate inconclusive: above 5%, below 10%);
  on the robust objective max 0.6%. Quick: median 0.001%, max 8.9%. Full 80C
  layers: reference / contention-free lower bound 1.26 (upper bound on the gap,
  not an estimate); quick / reference 1.089-1.094. Follow-ups: fix the yardstick
  objective (event vs robust: a robust-accepting descent lands up to 12% above
  the event optimum); add a compound split-and-reassign move (the >3% gaps are
  mixed shapes single moves cannot reach); add a model-objective LNS loop to the
  repository (the existing LNS driver is hardware-diagnostic only); reach the
  contention regime with a contention-aware bound or scaled-machine instances
  (reduced optima sit within 0-2.9% of the packing bound). Original item:
  No hardware needed, can start now: reduced exact instances (8/16/32
  experts from real routes) with exhaustive or reduced CP-SAT optimum under the
  model objective; core-work and critical-expert lower bounds on full
  instances; 1/10/30/60/300 s anytime curves (Step 7).
- [ ] P3 After M4: close the quick regressions with the frozen model. Gates:
  quick no worse than fixed 8T by more than 2% on any catalog case or captured
  layer (currently -11--14% on dense active-set cases); quick ranks the pinned
  and windowed candidates of the M2 set with regret <= 5%, or those shapes stay
  out of its candidate space and the paper says so.
- [ ] P4 Final planner matrix from the frozen commit: 43 layers x 3 requests,
  every calibrated fixed width / quick / full / LNS reference / measured
  shortlist oracle, with `T_plan`, `T_execute`, and their sum. Report the quick
  gap in two layers: model-objective gap to full/LNS (search quality, anchored
  by the P2 reduced-instance optima) and measured gap to the best plan of the
  measured candidate set (search plus model). Also report `T_plan` against one
  layer's execution time and the token count below which a fixed width wins.

Integration and revision round (user decision 2026-09-20: wire v10 into the code, fix the
model or the search where they fall short, reach a near-optimal search, extract a fast
planner from it). Records: [`v10 integration/E1`](../tmp/v10_integration_20260920/decision.md),
[`P6 probes/v11`](../tmp/v11_probes_20260920/decision.md),
[`E2`](../tmp/v11_validation_20260920/decision.md),
[`fast planner/E3`](../tmp/fast_planner_20260920/decision.md).

- [x] v10 as a planner cost model (`cost_model/probe_event_model.py`, asset
  `probe_event_v10_20260919.json`, opt-in; native planner export disabled; numerically equal
  to the Lab module on all 306 M4 plans, 8.6 ms per 220-task evaluation) and a model-objective
  LNS (`planners/model_lns.py`).
- [x] E1 (18 M2 workloads, measured): v10-driven search exploits model error. Searched plans
  measured / predicted 1.325 (with 2T lanes) and 1.175 (without), anchors 1.09-1.11; the
  2T-free search still ran 6.5% faster than the best frozen M2 candidate. Traced per-task
  diagnosis (E1d): under load the model under-predicts narrow-lane mid-M tasks (2T M 25-384
  1.19-1.41, 4T M 13-192 1.12-1.26, 8T M<=24 1.30; large-M 8-32T 0.97-0.99); isolated times
  are not the cause; intra-lane dispatch gaps are 5-40 us.
- [x] v11 (MATHEMATICAL_MODEL v1.109, asset `probe_event_v11_20260920.json`): measured
  isolated correction c(t, M), M-indexed steady curves, and loading equivalence w(M) of a
  mid-M background steady phase (0.79 at M 24 to 0.11 at M 384), all from probes P6
  (`tmp/v11_probes_20260920`), composition frozen before measurement, no fit on plan timings.
- [x] Search: multi-start descent plus rebalance / recreate / reorder-all moves. On the 36
  reduced instances of P2 the search now reaches the exact optimum on every one (max 0.004%;
  single-start was up to 2.0% off).
- [x] E2 (18 fresh layers, measured once): the v11 search restricted to lanes >= 4T is the
  fastest of eight plans on 17/18 workloads, 13.2% faster than production quick (median,
  -14.9% to -5.5%), 10.2% faster than the best anchor construction, 3.7% faster than the v10
  search. v11 selection over the 2T-free plans: regret@1 median 0%, max 7.9%, @2 max 1.8%
  (v10: 3.4% / 14.8% / 5.5%). 2T lanes stay out of the search space (measured / predicted
  1.237, sign agreement 0/18) and are reported as a limitation.
- [x] E3 (18 more fresh layers, measured once,
  [record](../tmp/fast_planner_20260920/decision.md)): the reference search (v11 LNS, 90 s)
  is 12.87% faster than production quick (median, -15.0% to -7.6%) and the measured best plan
  on 17/18 workloads; 10 s of search is within 0.61% of it. The fast planner
  `planners/hot_wide_planner.py` (wide lanes for hot experts + 4T bulk, per-width load scale,
  hot-first-then-ascending order; 30 ms warm in Python) is 9.63% faster than production quick
  on 18/18 workloads but 3.82% behind the reference (F2 not met). Measured / predicted is
  1.09-1.10 for all three, so the gap is search depth, not model error: template scoring with
  the event model closes 0.6 points (272 ms), order patterns 0.8 points (98 ms), load
  rebalancing nothing; the rest is event-level phase overlap between lanes.
- [x] Production adoption (user decision 2026-09-20,
  [record](../tmp/v13_adoption_20260920/decision.md)): the V3 window table is registered for its
  machine (`machine_ids`), `PlannedMoE` has a `hot_wide` search mode, `MoePlannerRuntime` takes
  an event calibration, and `enable_moe_planner_fast()` installs the fast planner. Verified on
  the machine: the runtime reproduces the measured `hwt` bridges on 18/18 workloads and plans a
  layer in 6.6 ms against the quick runtime's 10.4 ms. Measured against the pre-adoption default
  (quick, full stripe; E4, 18 layers, 18/18): fast planner -12.75%, reference search -14.90%,
  window registration alone -1.94%.
- [x] The 2T defect chased (user decision 2026-09-20,
  [probes](../tmp/v12_probes_20260920/decision.md),
  [validation](../tmp/v13_validation_20260920/decision.md)): P7 measured that contention grows
  as the background lanes narrow (4T target at ~76 other cores: 2.01 with 2T lanes, 1.91 with
  4T, 1.70 with 8T, 1.40 with 16T) and P8 that the composition is right for 4T and wider
  backgrounds (1.03-1.10, including a real-layer route mix) but under-predicts 2T backgrounds by
  18-49%. v13 (2T lanes count as loading in every phase plus the width factors) reproduces the
  probe cells (0.978) but fails E5 on fresh layers (W1/W2/W3 all fail, regret 11.95% against
  v11's 4.45%): v11 stays the reference model and 2T stays out of the search space.
- [x] E6, is the searched plan a measured local optimum
  ([record](../tmp/search_reliability_20260920/decision.md)): on six workloads, nine model
  neighbours each (0-2% and 2-6% above the incumbent) plus 600 s of further search, measured in
  a verified idle window. The best neighbour is at most 0.17% faster (inside the 0.38% session
  spread), the neighbours the model puts 2-6% behind measure 1.1-3.7% slower on all six, the
  Spearman inside the neighbourhood is 0.62 in median (0.09-0.85: no ranking power in the 0-2%
  band), and 600 s more search gains 0.69% in median. The reference is a measured local optimum
  of the model's neighbourhood; no global claim. One earlier session set was discarded after an
  external load spike on the shared machine (sar: load 191, about 38 foreign cores).
- [x] E7, does the search space miss a better plan
  ([record](../tmp/search_reliability_20260920/decision.md)): on the same six workloads, LLC
  placement variants (invisible to the model by construction) measure 0.04-0.30% faster than the
  searched plan, tail-pool plans 1.2-3.6% slower on five of six (the planner's tail-pool score
  says the opposite), and the one workload with an equal-score different-width plan measures it
  slower. Gate passed: nothing outside the search space beat the searched plan by more than 1%.
  Route slicing was not exercised (the candidate list came back empty) and stays open.
- [x] Route slicing and the tail-pool score (user decision 2026-09-20,
  [record](../tmp/search_reliability_20260920/decision.md), MATHEMATICAL_MODEL v1.115).
  Slicing: the repository's bounded-tail family cannot run here (96-core widths, two-terminal-task
  precondition); implemented instead as `slice`/`unslice` moves in the search, and shown
  worthless on these workloads by a bound - the hottest expert takes 0.38 of the work bound in
  median (0.44 at most) over all 54 measured layers, so no expert constrains the makespan.
  Tail pool: traced runs locate the error outside the tail-pool simulation. The automatic
  candidates pool onto 1T, a width the probes never measured (curves clamp at 2T), and those
  pooled tasks measure 3.25-4.25x their predicted isolated time. The model now publishes
  `calibrated_widths` / `reliable_widths` and the planner keeps automatic pool widths inside
  them, which flips the r008_l21 preference back to the strict plan, as measured.
- [x] E9, does the stage window order matter
  ([record](../optimizations/fused_moe_sve/results/window_order_thread_major_20260920.md),
  MATHEMATICAL_MODEL v1.116): thread-major against window-major, two builds of one checkout,
  6 real layers x 11 plans = 66 points, A B B A. Median -0.027%, sign 37/66, p10/p90
  -0.245%/+0.291% against a 0.261% median session spread inside a build; per-layer Spearman
  0.82-0.99, the fastest plan never moved, and all 66 plan outputs hash equal. Thread-major
  adopted: each worker owns a contiguous N stripe and windows cut inside it.
- [x] E10, what concurrent work on the other NUMA nodes costs a measurement
  ([record](../optimizations/fused_moe_sve/results/numa_interference_20260920.md)):
  +0.8--13.1% depending on how many nodes stream DRAM. Written into
  `docs/agent_benchmark_hygiene.md` as a protocol rule, and the reason Arm-codex runs are
  serialised across sessions.
- [x] E11, does the window table survive the order change
  ([record](../optimizations/fused_moe_sve/results/window_table_thread_major_20260921.md),
  v1.117/v1.118): the 4/8/16T grids and the 2T rows were re-measured thread-major, composed
  as V4 by the unchanged 2026-09-19 rule. V4 and V3 differ in 16 cells, but the grid's own
  repeat moves choices by as much, so they are measured ties; on 18 fresh layers V4 was
  0.27% faster (16/18), short of the frozen 0.3% adoption gate. The same batch measured
  windows at 1.69% against no windows on whole plans. V4 registered on the user's decision
  for lineage consistency - every row, including 2T, now comes from the current kernel order.
- [x] E12/P9, what the 2T error actually is
  ([record](../optimizations/fused_moe_sve/results/narrow_lane_cache_mechanism_20260921.md),
  v1.119/v1.121): not the window term (E12's three frozen readings all negative) and not the
  lane width. The state variable is rho, the live weight bytes an LLC domain holds over its
  capacity: at matched rho the widths agree to 0.008-0.083, at matched width with the
  footprint free they spread by 0.40. The window PMU evidence (DRAM reads -27..-75%, L2
  refills -0..-9%) puts it at the shared LLC, not private L2. The model's curves are indexed
  by (target width, other cores) and cannot express one core count at different slices.
- [x] E13, is 2T worth anything
  ([record](../optimizations/fused_moe_sve/results/second_machine_c9g_20260921.md) for the
  cross-machine part, v1.120): closed on Arm-codex. Measured per-expert core time bounds the
  gain at 0.82%, and six hand-built 2T variants lost on 18/18 layers by 15-20%; the rule the
  mechanism suggested - keep routes <= 16 experts off 2T lanes - measured 5.10% slower,
  because the imbalance costs more than the LLC overflow. **This result does not transfer**:
  on C9g 2T is the fastest width at every M up to 192 (v1.123).
- [x] E14, does the footprint correction beat v11
  (`tmp/footprint_validation_20260921`, v1.122): v14 indexes the correction by rho and fades
  it in with domain occupancy. Descriptively it improved 588 already-measured plans
  (1.090 -> 1.061), and on 9 fresh layers it passed the level gate (+0.021 against a 0.020
  threshold) but failed ranking (Spearman 0.94 -> 0.89) and selection (regret@1 median/max
  0.00%/0.00% against 1.89%/4.16%), with `lns_v14` measuring 1.89% slower and winning 0/9.
  Not adopted; v11 remains the reference. The mechanism stands, the composition rule does not.
- [x] Native (C++) hot-wide planner (`ec85f54`, v1.123): 0.703 ms -> 0.205 ms per layer,
  45/45 plans identical.
- [x] Second machine (C9g: Neoverse-V3, 2 x 96 cores, 96 MiB L3 per node, SVE-128 so
  n_tile 8)
  ([record](../optimizations/fused_moe_sve/results/second_machine_c9g_20260921.md), v1.123):
  the footprint mechanism holds across machines and shapes - two shapes whose per-expert
  weights differ by 2x land on one rho curve - while the level and the best lane width do
  not. Its analytic calibration reaches 2.0%/4.5% MAPE in the planner's own region.
- [x] Error decomposition and the window table's machine scope
  ([record](../optimizations/fused_moe_sve/results/offline_findings_20260921.md),
  v1.124/v1.125): 84.7% of the plan error variance is between plans of one layer and only
  15.3% is the per-layer level, which is why v12-v14 all moved the level and left the
  ranking alone; a candidate ranking residual is frozen for E15. Separately, a window table
  no longer resolves on a machine it was not measured on.
- [ ] Open: E15 (the ranking residual on fresh layers); the analytic model's missing
  per-panel term; W8A8 below 256-bit SVE; the model's unexplained level bias, now split into
  a per-layer level and a within-layer ranking residual; the paper's planner matrix (P1/P4).
  The 2T mechanism and value questions are closed above, and the native planner is done.

Stop rules: no new physical term unless it changes a measured plan choice by
>= 2% in both sessions; no new order/transfer search layers; glibc-era results
keep their label and are not refit.

## Cost model track after the jemalloc switch (2026-09-19)

Benchmarks and deployment on Arm-codex now preload jemalloc with purging disabled
(`docs/agent_benchmark_hygiene.md`, Allocator State). Plan V2 calls without a
resident route workspace carried 15-35% page-fault cost under glibc; the joint
model inputs used `FixedRouteWorkspace` or native probes and are clean
([correction](../tmp/dram_write_20260919/decision.md)). Live status stays in
`optimizations/fused_moe_sve/CURRENT.json`.

- [x] Rerun the four window experiments under jemalloc; replace the opt-in
  candidate table with `ARM_CODEX_NUMA3_80C_TP4_F512_N16_V3` (not registered).
  [Record](../tmp/jemalloc_rerun_20260919/decision.md), MATHEMATICAL_MODEL v1.105.
- [x] Recalibrate the contaminated v8 layers (wide-team pressure, 1T/2T
  by_width, narrow correction) as v9: gate-layer holdout measured/predicted median
  0.898 (v8) -> 0.984 (v9); published under
  `bench_assets/moe_paper/arm_codex_numa3_80c_jemalloc/` with machine config
  `arm_codex_internal_jemalloc.json`. Addresses the contaminated part of
  "Revalidate absolute v8 errors under resident-output methodology" above.
  [Record](../tmp/jemalloc_recal_20260919/decision.md), MATHEMATICAL_MODEL v1.106.
- [ ] Service rates under jemalloc: a single short probe failed the 3% rule
  (DRAM 80T -8%, LLC +-10%, L1 40T +24% vs glibc; glibc DRAM 80T also 14% below
  the August probe). Decide on a repeated full re-probe before trusting v2 services.
- [ ] Switch existing consumers (LNS tool defaults, machine configs, frozen
  anchor/frontier plans chosen with v8) to v9 only by separate decision.
- [x] Unseen whole-plan error (analytic, full stripe, descriptive): on the 48
  unique jemalloc window_validation plans v9 has meas/pred 0.97-0.99, Spearman
  0.67-1.00, regret 0% (layer 12) and 3.6-8.6% (layer 29); frozen quick scores
  regret 0.5-17%. Errors concentrate on pinned wide lane + narrow lanes (meas/pred
  0.79-0.94). [Record](../tmp/whole_plan_eval_20260919/decision.md).
- [x] Single large-M expert per width (4/8/16/32T, M 256-2048, isolated and under
  4T background): 32T over-predicted 24-28% (width dilation B32/S32 1.45/1.52 vs a
  width-independent single-task residual of 1.06-1.11); 4T loaded over-predicted
  17-23% via the DRAM term (spill 1.0, offered 2.4x capacity) while measured load
  slowdown is 1.06-1.12x. [Record](../tmp/single_expert_width_20260919/decision.md).
- [ ] Model revision (MATHEMATICAL_MODEL): width residual tied to per-expert work
  instead of per-task B_t/S_t; refit on A1 layer data plus the single-expert cells.
- [x] PMU check of the DRAM term: loaded measured/modeled DRAM reads 0.15-0.17
  (v9 spill 1.0 at capacity vs measured about 5x compulsory, 44-71 GB/s); isolated
  teams re-read weights at 21-27 GB/s (model assumes LLC-resident, not binding).
  [Record](../tmp/narrow_dram_pmu_20260919/decision.md).
- [ ] Model revision: DRAM demand/LLC spill for concurrent lanes consistent with
  the PMU bytes; first identify the term behind the 32T-target loaded whole-call
  under-prediction (1.06-1.22), which the DRAM over-estimate currently masks.
  (M1a: no masked term; see the plan section.)
  Validate on the pinned whole-plan set.
- [ ] Joint model whole-plan error needs in-domain data: only 5/54 of these points
  fit its domain (M <= 1718, widths <= 16 inside one LLC domain, no 32T).
- [ ] Rank pinned hot-expert plans: the isolated-LPT quick cost does not
  (window_validation G1/G1w/G2 0/3/1 of 6 under jemalloc), although pinned or
  windowed plans beat the baseline by 2-10%. Required before any pinned-shape
  candidate or window default.
- [ ] (Tracked as M2b in the plan above.) Make the event model window-aware: `dag_makespan` and full search ignore
  windows; only quick cost uses r(t, M). Choose between a per-task time scale and
  resource terms (PMU: W13 1-tile windows cut DRAM reads 27-75%, glibc-era).
- [ ] Joint model acceptance items remaining: block-history demand and
  independent-session stability; same-condition contention response and dynamic
  mixed phases; real W13 to W2 migration; unseen whole-plan error and
  equal-budget planner search.
- [ ] A2 1T fit identifiability: the fixed/route split moves across runs
  (fixed 100-210 us; jemalloc route_ns 3.4 vs glibc 11.5 us) while totals agree;
  under jemalloc the non-fit peer modes are overpredicted by 4.5-5%. Constrain the
  fit or validate totals only before relying on the split.
- [ ] Glibc-era window diagnostics (m2040, m_mid, m24, peff, window_pmu,
  shared_combined) keep their glibc label; rerun one only when a decision needs it.
- [ ] Low priority, needs approval: resident route_out inside the runtime, which
  removes the allocator dependence.

## Next planner track: event-guided VND, LNS, then ALNS

- [x] Execute the four-step workspace baseline plan in order: explicit profile
  and identity separation; common-session anchor/elite confirmation; frozen-v8
  model reassessment without fitting; independent median smoothing confirmation.
  Current experimental anchors: median2ab43572, high-skew189d70 (+61ea6f elite),
  uniformish retained. Weak-start VND remains valid. Smoothing beats old anchor
  but not current elite, so not adopted. Old residuals cannot authorize pruning.
  [Registry/evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_workspace_baseline_confirmation_20260907.md).
  [Mandatory baseline for new offline runs](../optimizations/fused_moe_sve/results/workspace_experiment_baseline.md).

- [x] Workspace model/search audit: reuse52 unique existing plan measurements and
  replay46 representative old VND/LNS plans in8 sessions. Greedy insertion now
  gains3.350/3.377%; median elite2ab43572 still passes despite old candidate_worse
  label; high-skew LNS elites retain gains. Uniformish only repairs weaker parent.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_workspace_model_search_audit_20260907.md).
- [ ] Revalidate absolute v8 errors under resident-output methodology with separate
  calibration/evaluation data. Preserve2ab43572 as an extended-domain pruning
  counterexample. Do not fit old output first-touch residuals as bandwidth terms.
- [x] Correct geometry reporting: frozen calibration and actual packed object
  use backend_n_tile16, not the8 stated in earlier workspace prose. Runtime unchanged.

- [x] Replay8 historical order/pressure/GEMM/interleave frontiers with workspace,
  16 sessions. Only median block retains the old robust positive classification;
  neither historical actionable winner retains its >2% gate. No winner overwrite.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_workspace_order_replay_20260907.md).
- [x] Confirm median smooth: same complete plan passes2 GEMM-frontier sessions
  but not both pressure-frontier sessions. Do not cherry-pick the passing subset.
- [ ] Audit remaining workspace-regime model errors, starting with high-skew
  union p02 predicted+12.041% versus measured-3.282/-2.430%; no new physical term yet.

- [x] Replay both complete seven-plan transfer frontiers with workspace, two
  untraced sessions each. No >2% stable winner; median bad relocation becomes
  +1.11% but misses P10; high-skew cross-domain relocation becomes neutral,
  same-domain swap/relocation remain around-6.6%/-13%. Retain anchors.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_workspace_transfer_replay_20260907.md).
- [x] If continuing: replay older order frontiers under the workspace baseline
  before treating old allocation-mode gains/residuals as valid in the new regime.

- [x] Fixed max_tokens workspace Lab version: preallocate/touch once, exclusive
  lease, overflow rejection, NaN overwrite checks. Two sessions improve anchor
  steady state3.078/3.382%, swap8.310/8.278%, relocation8.224/8.719%; initial
  allocation/touch8.049/8.176ms. No production-default change.
  [Evidence/usage](../optimizations/fused_moe_sve/results/arm_codex_80c_fixed_route_workspace_20260907.md).

- [ ] Low priority: grow route-output workspace capacity safely at an idle boundary.
  First version uses explicit max_tokens, rejects overflow and does not shrink/grow.

- [x] Output pretouch control: two independent sessions, all outputs equal. Head
  W2 collapses2.804/2.876→0.539/0.542ms; outside-touch minor faults1/1. Reject
  per-call full clearing:9.1–9.3ms pretouch worsens inclusive latency17–22%.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_route_output_pretouch_20260907.md).
- [x] Next candidate, if requested: explicit route-output workspace lifecycle with
  amortized first touch and concurrent-call isolation. Measure initialization and
  steady state separately; do not reuse old first-touch penalties as contention calibration.

- [x] Trace actual transfer task timelines in two sessions per trace. Median's
  recipient consumes slack and is last only2/62 rounds; high-skew swap/relocation
  recipient suffix is last62/62. Donor acceleration only grows compute slack.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_transfer_task_trace_20260907.md).
- [ ] If deeper attribution is needed, decompose existing M164 head/donor phase
  records; do not infer bandwidth or true idle time from task envelopes alone.

- [x] Implement bounded same-width swap/relocation, split same/cross LLC, preserving full task metadata.
- [x] Measure frozen median/high-skew transfer frontiers, two sessions each, six candidates plus anchor.
  No candidate passes both-session >2% gate. Median same-LLC swap gains0.750/0.576%;
  high-skew model-best candidates regress5.8–13.2%. Retain both original anchors.
  [Protocol/status](../optimizations/fused_moe_sve/results/arm_codex_80c_same_width_transfer_20260907.md).
- [ ] Replay existing transfer evidence against isolated lane-load/critical-lane changes;
  assess whether a load-aware shortlist guard can identify large regressions without
  losing known measured gains. Do not launch another hardware layer or refit yet.

- [x] Add explicit interleave templates (`--proposal-set interleave`) and broader
  order transforms (`extended`) to the bounded offline freeze entrypoint. Keep
  original `legacy` default, per-family representatives and seven-plan hardware cap.
  [Usage](../optimizations/fused_moe_sve/results/interleave_order_templates_usage_20260907.md).
- [x] Compare interleave/extended from latest measured anchors with frozen shared
  unions (median8/high-skew9 plans), two sessions each. No actionable winner;
  median block relocation yields 1.716/2.342% but misses S1 margin. Preserve it
  as near-elite, retain anchors, and stop this order-only budget.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_order_strategy_compare_20260907.md).
- [ ] New-domain calibration/gates remain necessary before automatic VND/LNS
  acceptance/pruning or production default changes. No new task-migration
  experiment has been launched.

Bounded offline selection workflow (2026-09-06):

- [x] Freeze score-best and historical fallback; deduplicate complete PlanV2 bridges.
- [x] Add `bench_selection_pair.py tune`: at most two unique plans, two independent
  measured sessions, then persist the complete hardware-consensus winner.
- [x] Replay median/high-skew evidence into winner artifacts; uniformish deduplicates
  without a hardware run. Reject mismatched/incomplete evidence and label session
  disagreement inconclusive. Small consensus gains remain non-actionable.
- [ ] Broaden evaluation before any production selector/default adoption.

- [x] Complete one bounded order-only layer around median/high-skew hardware
  winners: 72 model candidates, six measured candidates plus anchor per trace,
  two sessions. Median retains anchor; high-skew diversity rotation
  `34c08621...` passes at 3.887/2.243% paired speedup (model predicted -4.478%).
  Full bridge/evidence: [bounded order report](../optimizations/fused_moe_sve/results/arm_codex_80c_bounded_order_extension_20260906.md).
  No next layer or model refit has been launched.

- [x] Pressure-balanced order proposal screen: global/domain smoothness,
  bursty control and alternating routes, two traces and two sessions each.
  No candidate passes the unchanged >2%/positive-P10 gate in both sessions.
  High-skew alternating gains 1.037/1.960% but remains sub-threshold. Do not
  adopt isolated smoothness as a ranking objective or fit a new physical term.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_pressure_balanced_orders_20260906.md).

- [x] GEMM-average density direct comparison: high-skew smooth beats uneven by
  3.695/3.973% and the prior rotation anchor by 2.355/3.464%, passing both-session
  gates. Median has no benefit. Retain as a candidate-generating heuristic, not
  a universal selector or physical-model fit; no next layer started.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_gemm_density_comparison_20260907.md).

- [x] Fixed high-skew smooth/uneven pair under 0/4/16 remote-reader pressure:
  user restored PMU access; pilot and two formal sessions completed. Smooth
  gain shrinks from 5.259/4.780% to 3.239/3.285% as extra pressure increases.
  No support for increasing gain over this range; no certified absolute-low
  endpoint. All reader processes cleaned up; no additional OS settings changed.
  [Protocol/blocker](../optimizations/fused_moe_sve/results/arm_codex_80c_density_pressure_sweep_20260907.md).

- [x] Synthetic M experiment (E60, 10x8T, early merge off): five uniform references
  and four clustered/staggered grids, two sessions. (4,32), (16,128), (64,512)
  pass at ~3.4%, ~25.2%, ~17.4%; (1,8) has no stable gain. Do not infer monotonic
  gain from average bandwidth alone or generalize to real traces without tests.
  [Evidence](../optimizations/fused_moe_sve/results/arm_codex_80c_synthetic_m_density_20260907.md).

Usage and evidence: [selection pair report](../optimizations/fused_moe_sve/results/arm_codex_80c_selection_pair_validation_20260906.md).

Current Cursor handoff (2026-09-06):
[`template_lns_cursor_todo.md`](../optimizations/fused_moe_sve/results/template_lns_cursor_todo.md).
Tasks 2A, 1A, 3, and 6A are closed. Lab presamplers
`structural_coverage_then_random_v1` and `structural_coverage_closure_width_v1`
were rejected. The full-parent critical cross-domain d4 pool has 452 unique
members (digest `a10eeba6...`); 11/452 were measured historically. Diagnostic
budget is 456 unique plans and exceeds the 80 LNS-candidate slots. Task 6B is
unauthorized; Task 5 stays closed. Keep N=25 shuffle-truncate and injected
elites. Historical records retain their scope.

Independent context-residual track (opened 2026-09-05):
[`context_aware_residual_experiment_20260905.md`](../optimizations/fused_moe_sve/results/context_aware_residual_experiment_20260905.md).
The Lab candidate retains frozen v8 as its event-time baseline and adds a
23-feature regularized residual. It does not change VND/LNS scoring, pruning,
calibration, or production. Experimental runs use a terra/medium subagent.

- [x] Implement plan-visible structure/timeline/interaction features, frozen
  identity validation, label-free time prediction, and focused negative tests.
- [x] Run fixed-alpha parent-disjoint development replay on the published
  high-skew LNS frontier; exclude cross-parent duplicate states. Compare a
  constant-offset control before attributing absolute-error gains to context.
- [ ] Validate on a prospectively sampled plan set with untouched evaluation
  groups and session provenance. Existing selected-frontier results do not
  establish unbiased neighbor recall or justify adoption.
- [ ] If that test supports a residual model, calibrate its own uncertainty
  on separate data before considering any search integration. Do not reuse
  the v8 pairwise radius. The 452-plan hardware budget is still not authorized.

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
- [x] Post-hoc planner-visible geometry on the same locked grid, without
  measured overlap: \(n_B\sqrt{t}\), frozen-v8 isolated peer W13 core-ms,
  isolated operator core-ms, and \(a n_B+b n_{\text{threads}}\). No proxy
  passes both sessions and 20% drift. Session-2 operator and two-coefficient
  fits can pass count-6 alone, but 8x2T queue is 48.17 vs 29.14 cycles on
  identical live teams (ratio 1.653). Keep
  `stop_absolute_model_expansion=true`. See
  `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_plan_geometry_proxies_20260905.md`.
- [x] Repeat the locked proxy grid three more times (seeds 20260926/27/28) on
  Arm NUMA3 without changing v8. 8x2T paired queue medians were 12.58 / 32.19 /
  27.89 cycles versus historical 48.17 / 29.14. The node was not exclusive
  (`bench_meformer_`, `tokio-rt-worker`). Do not freeze a queue term. See
  `optimizations/fused_moe_sve/results/arm_codex_80c_stream_pressure_queue_repeat_20260906.md`.

- [x] Starting independently from full, one-step, greedy, and fixed-width
  controls, run best-improvement descent over same-lane insertion, same-width
  cross-lane relocation, pair swap, and cross-domain relocation. The Arm 80C
  three-trace model-only replay evaluated 7,326 unique neighbors from 12 named
  starts; all starts stopped after iteration zero because no lower gain bound
  cleared the 2% margin.
- [x] Keep every accepted state executable and preserve the best incumbent at
  all times. Report per-operator proposals, accepted improvements, event calls,
  wall time, and best-so-far curve. The replay made zero accepted moves, used
  7,350 event calls and 1,067.58 s, and records all requested fields in the raw
  artifacts. Operator-only calibration resolution avoids candidate placed-event
  replay that cannot hit any calibrated context key.
- [x] Use the known high-skew `cp_sat_06` result (`33.148 ms` versus the
  `34.484 ms` full incumbent) as a diagnostic target: determine whether local
  executable moves can recover the opportunity, not merely whether one seed
  improves one model score. No high-skew start has an acceptable first move, so
  monotone partial-order VND cannot recover it. Its saved width histogram also
  differs from the full start in at least 53 expert-width assignments; the old
  artifact lacks the exact task sequence needed for a canonical connectivity
  proof. See
  `results/arm_codex_80c_partial_order_vnd_model_replay_20260904.md`.
- [x] Freeze the high-skew four-start top-16 incomparable frontier with complete
  canonical states and PlanV2 bridges, add two worse sentinels per start, and
  measure all 76 deduplicated plans in two independent 31-round sessions. Two
  incomparable candidates are stable above 2% in both sessions: a fixed-width
  adjacent swap at `+4.690/+3.988%` and a greedy `16T -> 8T+8T` split at
  `+3.819/+3.376%`. No worse sentinel is a cross-session false prune. Adopt a
  hardware-assisted beam next; defer template-level LNS until the measured beam
  stops improving. See
  `results/arm_codex_80c_partial_order_hardware_frontier_20260904.md`.
- [x] Run the measured beam through depth 3 while expanding the absolute global
  incumbent, not only candidates that improve a weaker parent. Corrected depth
  2 found state `514ced0d...` fastest in both sessions at `31.808/31.721 ms`,
  with `+2.146/+3.228%` median over original full, despite only
  `+0.745/+1.055%` over its immediate parent. Depth 3 found no stable >2%
  relative candidate in either session and no repeated absolute winner. Stop
  local beam and hand off to template-level LNS. See
  `results/arm_codex_80c_hardware_assisted_beam_20260904.md`.

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
  imbalance, and idle-core tail. The first template-LNS gate reuses the existing
  tail-weighted event-dilation score plus an equal-budget random control; add the
  other declared signals only if the next cross-trace replay shows a recall gap.
- [x] Implement lane-atomic target destroy sizes 4/8/16 over critical windows.
  Keep all unaffected lanes fixed and jointly repair legal width templates,
  domain placement, assignment, and four bounded temporal-order policies with
  placement beam widths 16/32/64. Record the full lane closure because a target
  destroy size can move more experts than its nominal 4/8/16 target.
- [x] Maintain a diverse elite pool rather than one trajectory. Diversity must
  cover width histograms, LLC-domain assignments, critical-expert placement,
  and temporal order, not only distinct serialized start vectors. The high-skew
  gate used three measured incumbents, two proposal restarts each, six
  scope/size operators, and retained 92 unique top-16 states across 9 shapes.
- [ ] Compare LNS with VND under identical event-call and wall-clock budgets.
  Adopt LNS only if larger destroy/repair neighborhoods escape reproducible VND
  local optima on held-out traces. The high-skew equal-call/equal-hardware gate
  passed: LNS used 3,512 event calls and 106 plans versus local beam 3,535/106,
  found 40 cross-session stable unique candidates with zero false pruning, and
  improved the preserved `514ced0d...` median by `3.961/3.528%`. LNS model wall
  time was 1,193.47 s versus 507.06 s. Repeat on median and uniformish before
  marking this cross-trace item complete. See
  `results/arm_codex_80c_template_lns_20260904.md`.
- [x] Repeat the frozen LNS operator mixture on median and uniformish without
  importing the high-skew winner as a start. All three traces found a consensus
  winner more than 2% faster than the strongest preserved anchor in both
  sessions. Median also exposed two stable false-pruning sentinels, including
  its absolute best; uniformish exposed one strict-P10 false acceptance. Keep
  the LNS neighborhood, but disable partial-order acceptance and dominance
  pruning for LNS. See
  `results/arm_codex_80c_template_lns_suite_20260904.md`.
- [x] Build a relation-agnostic diverse hardware shortlist that reserves
  operator, target destroy, actual closure, width histogram, LLC-domain
  assignment, and model-score quantile coverage without using the partial-order
  relation. The frozen measured-suite design replay at K=16 retains the absolute
  measured best and consensus winner on high-skew L1/L2, median, and uniformish
  with zero selected-best regret. Freeze selector v1. See
  `results/arm_codex_80c_lns_diverse_shortlist_replay_20260904.md`.
- [x] After freezing selector v1, generate one independent median proposal
  frontier with seed `20261010`, measure the nested top-32 audit in two Arm
  NUMA3 sessions, and require top-16 to contain the top-32 absolute best and
  consensus winner with zero selected-best regret. Do not change the selector
  after opening that frontier. Nested recall passed, including K=8; the
  1-restart neighborhood missed the two-session 2% strongest-full gate
  (`+2.579/+1.916%`). Adopt selector v1 for offline LNS shortlists only. See
  `results/arm_codex_80c_lns_diverse_independent_median_20260905.md`.
- [x] Split the 797 s median search wall and apply one equivalent optimization
  of the confirmed enumeration hotspot. Closure/template time is negligible;
  `_beam_assign_tasks` was the isolated-profiler hotspot (full 7.045→2.245 s,
  one-step 67.573→17.221 s). Frozen seed `20261010` ranking (hashes, sampling,
  scores, quantiles, ordered top-16/32) is unchanged. Production-path wall
  797.01→687.55 s; remaining cost is exact event 341.43 s and diagnostic
  shortlist 278.77 s. See
  `results/arm_codex_80c_lns_search_breakdown_beam_equiv_20260905.md`.
- [x] Keep selector v1 / K=16 / the current operator mixture frozen. Compare
  1-restart versus multiple restarts under the same candidate-evaluation and
  hardware-plan budget, including the four reconstructed controls and the known
  median elite `2ab43572...`. Report cross-session stable gain, seed success
  rate, regret versus that elite, and total search/measure cost. Do not treat
  extra budget as a restart win. Top-32 versus all generated candidates remains
  an open sample, not a closed recall proof. Equal cap 2,400 / 85 plans: both
  arms select `0418b884...` and beat full by >2% in both sessions; 2-restart
  also beats the elite in both sessions and overlaps K=16 in only 15/64.
  Adopt 2 restarts with `N=25` for offline median LNS. See
  `results/arm_codex_80c_lns_restart_budget_20260905.md`.
- [x] Keep selector v1 / K=16 / 2-restart `N=25` frozen and run a second
  independent median proposal seed. Seed `20261011` used the same 2,400 exact
  cap and 64+16 LNS hardware slots, generated 2,362 unique candidates in
  550.90 s, and never sampled `0418b884...`. K=16 overlap with seed `20261010`
  was 3/64. Selected-best vs full was `−0.018% / +0.371%`, so the two-session
  2% neighborhood gate fails. The injected previous winner still beat full in
  both sessions. Keep the allocation; do not replace the seed-`20261010`
  proposal. See
  `results/arm_codex_80c_lns_second_proposal_seed_20260905.md`.

### Step 6: decide whether adaptation is warranted

- [x] Measure which destroy/repair operators win on each trace class. Add ALNS
  reward-weight updates only if operator effectiveness is complementary across
  workloads; otherwise retain the simpler fixed-mixture LNS. Cross-domain
  repair dominates all first layers, while the only high-skew depth-2 stable
  move is domain-local d8. This is depth-dependent complementarity, but do not
  add ALNS weights until the independent median holdout of the
  relation-agnostic diverse shortlist passes; comparator safety remains closed,
  and design-replay recall is not that holdout.
- [ ] If adopted, reward separately for a new global best, current-state
  improvement, elite-pool admission, duplicate, and invalid repair. Freeze the
  update rule before the final holdout and provide operator/size ablations.
- [x] Add bounded diversification through multiple starts, tabu state hashes,
  or occasional worse-state acceptance. The returned plan must always be the
  best incumbent, independent of the exploratory trajectory. The fixed LNS
  uses canonical-deduplicated controls/elites and two proposal restarts per
  parent; model decisions never replace the incumbent in diagnostic mode, and
  the returned state is selected only from cross-session hardware consensus.

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
- [ ] (Gate definition superseded 2026-09-19: selection regret is primary,
  MAPE/P90 are diagnostics; see the plan section above.)
  Decide the cost-model paper claim. Either refresh the empirical
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
- [x] Wire the existing frozen-v8 $B_t/S_t$ occupancy scale into homogeneous
  quick width ranking (LPT packing still isolated $T_{iso}$). Arm 80C 5/31/4
  paired rerun: active-set-8/16 no longer choose 40T and are tied with fixed
  8T (`-0.03%/+0.30%`). Denser cases over-narrow to 20x4T and lose
  11--14% (active64/128/tiered). Bimodal still wins (`+30.26%`). Quick still
  does not dominate fixed 8T. See
  `optimizations/fused_moe_sve/results/arm_codex_80c_quick_vs_fixed8_wide_team_20260906.md`.
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

- [ ] Four-rank concurrent validation (added 2026-09-20). Every current
  Arm-codex 80C result is one TP rank on NUMA3 (`--physcpubind=240-319
  --membind=3`, `concurrent_ranks=1`) with the other three NUMA nodes idle,
  while the deployment is TP4 on 320 cores with all ranks executing the same
  route histogram at once. Run the same plan on all four NUMA nodes
  concurrently (one process per node, barrier-aligned start, per-rank call
  time and max-over-ranks recorded) for the held-out layers already used in
  E2/E4 and report: (a) per-rank slowdown against the single-rank
  measurement, (b) whether the fast-planner and reference-search gains over
  the production quick path survive, (c) whether plan ranking is preserved
  (regret@1 under concurrency). All-reduce stays out of scope; state that
  explicitly. Until this is measured, claims are limited to "MoE operator time
  of one TP rank".

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
