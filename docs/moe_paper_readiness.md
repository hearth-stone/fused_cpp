# CPU MoE Paper Readiness And Evidence Map

Status date: 2026-08-27.

This document is the paper-facing index for the CPU MoE work in this
repository. It does not replace the mathematical, schema, implementation, or
experiment sources of truth. Its purpose is to keep the proposed paper claims,
current implementation, admissible evidence, and remaining gates aligned.

## Proposed Paper Thesis

The coherent system claim is:

> A contention-aware runtime for Mixture-of-Experts inference on many-core Arm
> CPUs that couples a moldable fused-expert execution substrate, a calibrated
> phase/resource cost model, and an executable interval-DAG planner.

The three contributions should be stated narrowly:

1. **Moldable fused execution.** An Arm SVE BF16 expert pipeline that preserves
   shared packed-A reuse while fusing gather-pack, W13, SwiGLU and packed-C
   production, W2 direct-route stores, and deterministic FP32 weighted
   reduction. Team
   width and per-stage owner windows remain explicit scheduling controls.
2. **Phase/resource cost model.** A model that lowers exact kernel demand into
   setup, cold packed-B, and steady packed-B phases, then maps those phases
   through calibrated compute, cache, and DRAM services in an event simulator.
3. **Executable planner.** A route-histogram planner that emits Plan V2
   core-interval DAGs, with a bounded online homogeneous search and a broader
   offline mixed-width/contention-aware search.

Do not describe contribution 1 merely as "fusing SiLU with GEMM": current CPU
MoE systems already implement that boundary. The paper-specific mechanism is
the combination of Arm SVE exact-M execution, shared packed-A ownership,
direct-route output, tile-window control, and a planner-executable task ABI.
The detailed implementation, evidence, negative-result, and AI-drafting dossier
for this contribution is
[`moe_fused_expert_contribution.md`](moe_fused_expert_contribution.md).

## Recommended Paper Structure

1. **Introduction:** CPU MoE route imbalance, weight/cache pressure, and why a
   fixed expert width or a kernel-only optimization is insufficient.
2. **Background and motivation:** routed expert shapes, current CPU execution,
   measured width/route non-monotonicity, and the long/short failure case.
3. **System overview:** one diagram connecting routing metadata, fused expert
   tasks, the model, quick/full planning, Plan V2, and the native executor.
4. **Moldable fused execution:** packed layouts, exact-M mapping, gather-pack
   sharing, W13/packed-C, W2 direct-route output, merge, and tile windows.
5. **Phase/resource cost model:** exact demand, service calibration, cold and
   steady phases, contention events, assumptions, and uncertainty.
6. **Executable planner:** mathematical problem, candidate space, quick/full
   algorithms, tail actions, lowering, complexity, and caching.
7. **Implementation:** Python/native boundary, calibration lifecycle, CPU
   placement, compatibility fallback, and reproducibility controls.
8. **Evaluation:** kernel, model, planner, full runtime, ablations, negative
   results, and generalization.
9. **Related work and limitations.**

The main paper should describe one current algorithm and one frozen runtime.
Historical wave planners, retired range ABIs, and rejected probes belong in an
artifact appendix rather than the method narrative.

## Authority And Versioning

Use the following order when two documents disagree:

1. current source and tests;
2. `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md` for planner and model
   semantics;
3. `cpu_moe_schedule_optimization/planners/plan_schema.md` and
   `cost_model/profile_schema.md` for serialized contracts;
4. `optimizations/fused_moe_sve/manifest.yaml` for feature lifecycle;
5. dated result reports for measurements on their recorded binary and profile;
6. historical design, weekly, and deprecated-wave documents.

Every paper table must record the source commit, extension hash, calibration
identity, machine, CPU set, NUMA and page policy, model shape, route workload,
warmups, samples, statistic, baseline, and candidate. A dated report whose
binary or calibration identity differs from the evaluated runtime is mechanism
evidence only; it is not current planner-regret evidence.

## Intended Primary Scope

The initial paper scope is deliberately smaller than the repository:

- Linux AArch64;
- Arm SVE BF16 fused-SiLU experts;
- standalone or tensor-parallel rank-local execution;
- fixed packed weights and non-preemptive expert tasks;
- prefill-oriented route histograms up to the calibrated route limit;
- FP32 direct-route storage and FP32 weighted accumulation for headline
  numerical claims;
- one NUMA-local planner domain per rank.

Keep the following outside the primary claim unless separately closed:

- W8A16 and W8A8 experimental weight formats;
- BF16 route storage before model-level quality validation;
- x86, NEON-only, and cross-ISA performance portability;
- expert-parallel dispatch and measured distributed collectives;
- router fusion, communication overlap, and cross-layer global scheduling;
- retired wave planners, weight-range ABIs, W2 resize, and timed-release
  experiments;
- synthetic shared-expert scheduling as a main result. It may be an extension
  after validation with the real model path.

## Current System Snapshot

### Fused execution

The current SVE path includes:

- adaptive M-by-K gather-pack for underfilled expert teams;
- exact logical M=1--12 JIT kernels with the production packed-A ABI;
- fused W13 SwiGLU and BF16 packed-C production;
- W2 FP32 direct-route storage by disjoint N owners;
- SVE weighted route reduction with FP32 accumulation;
- optional ready-token merge in the async executor;
- Plan V2 strict, tail-pool, and route-sliced strict execution;
- deterministic per-task W13/W2 owner windows expressed in whole N tiles.

The default numerical paper path should retain FP32 route storage. BF16 route
storage is implemented but remains an experimental quality tradeoff.

### Cost models

There are two distinct backends and the paper must not conflate them:

- **Empirical schema-v2 model:** exact calibration-domain profiles,
  full-workload anchors, and a phase-aware contention simulator. This remains
  the production/default model for the full empirical planner and the holdout
  oracle for analytical validation.
- **Analytical model:** exact kernel demand plus thin machine service
  calibration. It is explicitly installable for the production quick runtime,
  but its absolute contention and shape-selection gates have not all passed.

The analytical stage-window selector is a narrower successful subproblem. It
is not evidence that the complete analytical makespan model has converged.

### Planners

The current planner modes are:

- **quick:** homogeneous team shapes, isolated-cost LPT lane assignment,
  strict Plan V2, native C++ candidate selection when available;
- **fixed quick:** one caller-selected homogeneous width, used as a supported
  deployment fallback and A/B control;
- **shared quick:** a bounded mixed-width family with one synthetic all-token
  shared expert;
- **full:** mixed-width strict shapes, temporal ordering, derived tail-pool and
  bounded-tail candidates, scored with the phase-DAG model.

`MoePlannerRuntime` uses quick search, disables route-plan caching by default,
and may precompute the dense `T_iso[M,T]` grid with
`initialize_planner(max_routes)`. The environment control
`FUSED_CPP_MOE_PLANNER_FIXED_THREADS` selects fixed quick when a stable width is
preferred over the multi-width search. The full analytical search is an
offline performance-oracle path, not a request-path algorithm.

## Evidence That Can Be Reused

### Kernel mechanism evidence

| Mechanism | Recorded result | Paper use | Source |
| --- | --- | --- | --- |
| Explicit fusion | 3.29--5.63% lower latency than a same-GEMM explicit pipeline as route working set grows; LLC misses and backend memory stalls also fall | Controlled fusion ablation, not an upstream system comparison | `optimizations/fused_moe_sve/results/amazon_192c_unfused_pipeline.md` |
| FP32 W2 direct route | -0.95% at M=12, +2.18% at M=192, +15.82% at M=1536, +16.44% at M=2040 | Route-output ablation with the default numerical path | `optimizations/fused_moe_sve/results/amazon_192c_w2_direct_route.md` |
| Exact-M JIT | Largest gains occur at formerly overcomputed M=5/6 and M=9/10; full-M12 controls are neutral | Tail-kernel ablation on two Arm machines | `optimizations/fused_moe_sve/results/amazon_8c_192c_xbyak_exact_m.md` |
| Adaptive gather-pack | 2.3--12.6% complete-call gain for underfilled M=1--37 at 8T, with M=72/96 controls not regressing on the measured host | Underfilled-team ablation; needs a second-machine E2E repeat for a general claim | `optimizations/fused_moe_sve/results/amazon_8c_adaptive_mk_gather_pack_20260812.md` |
| Tile windows | 12.4% on uniform and 10.6% on captured DSV4 in the recorded planner A/B | Scheduling-control evidence; rerun on the frozen paper binary | `optimizations/fused_moe_sve/results/amazon_192c_stage_window_tiles_20260810.md` |

The explicit-fusion result is historical mechanism evidence only. A 2026-08-27
rerun attempt showed that the standalone explicit path still evaluates poly5
while the current production fused W13 uses FEXPA+poly2; F=512 and F=2048 both
failed the benchmark's predeclared relative-L2 gate. Do not use a new timing
from that comparator until its activation and rounding contract matches the
current fused path.

The July 2026 nine-workload vLLM-style table is useful for motivating load
imbalance and tail-pool execution, but it is bound to retired profiles and
older geometry. It must be rerun before becoming a headline table.

### Cost-model evidence

| Model/subproblem | Current result | Gate status |
| --- | --- | --- |
| Analytical isolated time on AmazonC5192Cores | 9.18% MAPE on 108 true holdout points | Passes the 10% isolated gate |
| Analytical contention | 49.57% P90 absolute error | Fails the 15% contention gate |
| Analytical shape selection | 8.17% maximum measured regret | Fails the 5% regret gate |
| Analytical tile-window selector v6 | 1.10/3.45/4.40% median/P90/max regret on six declared transition points | Passes only the declared window-selection subproblem |
| Historical empirical phase model | Approximately 2% median error on an 8-core heterogeneous holdout | Promising mechanism evidence; stale kernel/profile, must be refreshed |

The paper may currently claim that the analytical model explains and selects
tile windows in a narrow domain. It may not claim general contention-accurate
or machine-portable scheduling.

### Planner evidence

The latest analytical quick/full measurements are preliminary because they are
recorded only in the mathematical-model changelog:

- on one 80-core captured DSV4 workload, analytical full selected a 34.418 ms
  plan versus quick at 37.141 ms, while the best measured top-six candidate was
  33.907 ms; selected shortlist regret was 1.51%;
- the complete full search evaluated 142 strict and 327 dynamic candidates and
  took about 42 seconds cold;
- the public quick path after dense `T_iso` initialization measured
  4.584/4.612 ms median/P90 for 1000 calls on that setup;
- fixed-8T planning measured 2.698/2.708 ms through the public path;
- quick beat fixed 8T by 22.13% on long/short bimodal but regressed by 15.54%
  and 11.31% on active-set 8 and 16 because it selected an over-wide 40T team.

These results establish an online-quality gap and an offline-search-cost gap;
they do not yet support a robust online-planner superiority claim.

## Claims Allowed Now

The current evidence supports these bounded statements:

- the SVE executor implements a fused, moldable whole-expert substrate with
  explicit task width and tile-window controls;
- direct W2 route stores remove materialization/scatter traffic and benefit
  long routes while remaining neutral around the short-route noise floor;
- exact-M specialization removes selected tail overcompute without changing the
  packed ABI;
- event-based contention modeling is materially more appropriate than applying
  one slowdown for an entire heterogeneous call;
- full mixed-width search can find better plans than homogeneous quick search
  on the measured DSV4 case;
- Plan V2 can execute strict, whole-expert tail-pool, and bounded route-sliced
  schedules without resizing a running task.

## Claims Not Yet Allowed

Do not claim any of the following without new evidence:

- best or state-of-the-art CPU MoE performance;
- a general fused-kernel speedup over current upstream vLLM, oneDNN, or another
  declared production baseline;
- analytical contention prediction within the declared acceptance gate;
- quick planning that consistently beats fixed-width greedy execution;
- online full-search feasibility;
- end-to-end TP/EP improvement, because current distributed communication is
  analytical rather than a measured runtime;
- model-quality preservation for BF16 route storage, approximate SiLU, W8A16,
  or W8A8;
- cross-machine portability from a transferred packed-B retention prior;
- whole-model or serving throughput improvement from operator-only timings.

## Related-Work Boundary

The paper must distinguish its claims from at least these existing directions:

- current vLLM already exposes CPU/Arm MoE expert implementations and fused
  activation boundaries; the kernel novelty must be the SVE execution/dataflow
  and planner-executable moldability, not generic MoE or SwiGLU fusion
  ([vLLM kernel matrix](https://github.com/vllm-project/vllm/blob/main/docs/design/moe_kernel_features.md),
  [CPU MoE source](https://github.com/vllm-project/vllm/blob/main/csrc/cpu/sgl-kernels/moe.cpp));
- FasterMoE combines a performance model with fine-grained scheduling and
  dynamic expert handling for distributed GPU training
  ([PPoPP 2022](https://doi.org/10.1145/3503221.3508418));
- Tutel adapts parallelism and pipelining to dynamic MoE workloads across GPUs
  ([arXiv:2206.03382](https://arxiv.org/abs/2206.03382));
- Lina reallocates distributed inference resources based on expert popularity
  to reduce all-to-all imbalance
  ([arXiv:2210.17223](https://arxiv.org/abs/2210.17223)).

The intended differentiation is exact rank-local execution on many-core Arm
CPUs: moldable non-preemptive expert jobs, cache/DRAM contention, executable
core-interval DAGs, and planner/kernel co-design without changing router
semantics or replicating experts.

## Required Paper Evaluation

### Workloads

Retain the deterministic nine-case catalog for controlled coverage, but add:

- complete, non-reconstructed TopK traces across many layers;
- at least one full prefill sweep and one decode-oriented sweep;
- multiple token counts, TopK values, and active-expert counts;
- at least two model shapes, including the target DeepSeek configuration;
- standalone and TP rank-local execution; EP only after measured integration.

The current `dsv4-real-2048-seq70` artifact retains exact top-16 counts and a
moment-matched synthetic tail. Label it as a reconstructed routing summary,
not a full real trace.

### Baselines

At minimum compare against:

- the current upstream Arm CPU MoE implementation;
- the explicit same-GEMM unfused pipeline;
- fixed-width homogeneous LPT for every calibrated legal width;
- the vLLM-style staged queue as a scheduling mechanism control;
- quick, fixed quick, and full planner modes;
- a measured candidate-set oracle, with CP-SAT used only inside its declared
  surrogate domain.

Add oneDNN or another library baseline only when the operator shape and
numerical contract are genuinely comparable.

### Main tables

1. **Kernel:** absolute latency and effective throughput by machine, model
   shape, route workload, and baseline.
2. **Cost model:** isolated and contention MAPE/P90/max, rank correlation,
   selected-plan regret, coverage, and calibration cost on held-out data.
3. **Planner:** planning latency, operator latency, total
   `T_plan + T_execute`, selected shape/action, and measured regret.
4. **Model/runtime:** full-layer or full-model latency and throughput with
   numerical/model-quality validation.

### Ablations

Keep the paper ablation set small and cumulative:

1. static bucketed kernel;
2. exact-M JIT;
3. adaptive gather-pack;
4. fused W13/packed-C;
5. W2 direct-route store;
6. tile-window policy;
7. weighted/ready-token merge;
8. fixed width versus quick versus full planner.

Retired experiments belong in an appendix or artifact ledger, not the main
design narrative.

## Paper-Critical Gates

### P0: freeze and reproduce

- choose one paper commit and rebuild every headline result from it;
- generate matching empirical profiles for every evaluated machine and
  geometry;
- store compact raw table data and a command manifest for every main figure;
- rerun all headline measurements with FP32 route storage, or close the BF16
  route model-quality gate before using BF16 results.

### P0: cost model

- replace the 8-core transferred packed-B retention prior with local probes;
- validate unseen routes, widths, mixed shapes, and multi-LLC placements on at
  least two Arm machines;
- pass isolated MAPE <=10%, contention P90 <=15%, and maximum selected regret
  <=5%, or narrow the claim and domain explicitly;
- report calibration wall time and number of measured points versus the
  empirical table baseline.

### P0: planner

- remove or gate the active-set 8/16 wide-team regressions;
- define the quick/full relationship and the exact candidate space in paper
  pseudocode;
- report measured regret over the full workload matrix;
- either reduce full cold search substantially or define it explicitly as an
  offline oracle/autotuner;
- include planner overhead in all end-to-end comparisons.

### P1: system completeness

- validate a real multi-layer vLLM path;
- measure TP and EP communication rather than adding only analytical terms;
- add complete routing traces and disclose the trace collection policy;
- produce a reproducible artifact with profiles, workload manifests, table
  builders, and exact commands.

## Source Map

- Mathematical semantics:
  `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`
- Analytical model:
  `cpu_moe_schedule_optimization/cost_model/ANALYTIC_MODEL.md`
- Plan V2:
  `cpu_moe_schedule_optimization/planners/plan_schema.md`
- Planner implementation:
  `cpu_moe_schedule_optimization/planners/interval_planner.py`
- Production runtime:
  `src/fused_cpp/moe/planner_runtime.py`
- Native executor:
  `csrc/moe/arm/common/fused_moe_bf16_tiled.cpp`
- SVE experiment lifecycle:
  `optimizations/fused_moe_sve/manifest.yaml`
- Integration guide:
  `docs/vllm_bf16_tiled_moe_integration.md`
- Active implementation checklist:
  `cpu_moe_schedule_optimization/TODO.md`
