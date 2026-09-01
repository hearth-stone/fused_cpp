# CPU MoE Paper Readiness And Evidence Map

Status date: 2026-09-01.

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
  bounded-tail candidates, scored with the phase-DAG model for offline search
  and autotuning.

`MoePlannerRuntime` uses quick search, disables route-plan caching by default,
and may precompute the dense `T_iso[M,T]` grid with
`initialize_planner(max_routes)`. The environment control
`FUSED_CPP_MOE_PLANNER_FIXED_THREADS` selects fixed quick when a stable width is
preferred over the multi-width search. The full analytical search is an
offline reference/autotuning path, not an oracle and not a request-path
algorithm.

## Evidence That Can Be Reused

### Kernel mechanism evidence

| Mechanism | Recorded result | Paper use | Source |
| --- | --- | --- | --- |
| Explicit fusion | Current canonical comparator: fused is 17.87% slower at M=48, neutral within 0.79% at M=192, and 19.11% lower latency at M=2040 | Fusion is a large-route dataflow win, not a universal speedup or upstream comparison | `optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md` |
| FP32 W2 direct route | Current uniform controls: -3.12/-0.69/+0.01/+4.36/+3.98% at M=12/48/192/1536/2040; captured high-skew gains are +2.68/+3.68% at 2048/4096 tokens | Bounded route-output ablation; do not reuse the historical 15--16% as current performance | `optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md` |
| Exact-M JIT | Largest gains occur at formerly overcomputed M=5/6 and M=9/10; full-M12 controls are neutral | Tail-kernel ablation on two Arm machines | `optimizations/fused_moe_sve/results/amazon_8c_192c_xbyak_exact_m.md` |
| Adaptive gather-pack | 2.3--12.6% complete-call gain for underfilled M=1--37 at 8T, with M=72/96 controls not regressing on the measured host | Underfilled-team ablation; needs a second-machine E2E repeat for a general claim | `optimizations/fused_moe_sve/results/amazon_8c_adaptive_mk_gather_pack_20260812.md` |
| Tile windows | Current captured traces: +7.23/+4.85% on uniformish 2048/4096; other cases range from -0.16% to +1.12%, and 30,720-token cases are neutral | Scheduling-control evidence in a bounded route domain; rerun on the frozen paper binary | `optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md` |

The canonical explicit comparator now uses the same production
FEXPA-plus-quadratic SiLU evaluator. Its remaining numerical difference comes
from the two independent W1/W3 GEMMs versus interleaved W13 accumulation and a
different, mathematically equivalent FP32 rounding boundary. The production
sizes pass the declared relative-L2 gate. The result is still a standalone
same-GEMM mechanism control: it excludes route construction and merge and does
not replace a current upstream baseline.

The July 2026 nine-workload vLLM-style table is useful for motivating load
imbalance and tail-pool execution, but it is bound to retired profiles and
older geometry. It must be rerun before becoming a headline table.

### Cost-model evidence

| Model/subproblem | Current result | Gate status |
| --- | --- | --- |
| Analytical isolated time on AmazonC5192Cores | 7.96% holdout MAPE with machine-local retention | Passes the 10% isolated gate |
| Analytical contention | 10.85% MAPE and 16.08% P90 absolute error | Close, but still fails the 15% contention gate |
| Analytical shape selection | 9.21% maximum measured regret | Fails the 5% regret gate |
| Machine-local packed-B retention | Knee at 0.662 x private-L2 capacity; improves contention P90 from 16.97% to 16.08% but leaves regret unchanged | Retention is justified, but is not the missing ranking variable by itself |
| Analytical tile-window selector | Approximately 0.83/3.05/4.18% median/P90/max regret after the boundary repeat | Passes only the declared six-point window-selection subproblem |
| Historical empirical phase model | Approximately 2% median error on an 8-core heterogeneous holdout | Promising mechanism evidence; stale kernel/profile, must be refreshed |

The paper may currently claim that the analytical model explains and selects
tile windows in a narrow domain. It may not claim general contention-accurate
or machine-portable scheduling.

### Planner evidence

The latest 96-core matrix uses 31 samples per case. Full cold/warm planning is
0.55--7.06/0.11--3.19 ms on the measured synthetic and captured workloads, so
full is practical as an offline/autotuning search in this bounded candidate
space. It is not an oracle:

- on long/short bimodal, full selected a 4T tail pool and reduced strict
  execution from 17.25 ms to 12.61 ms (+36.8%);
- on captured uniformish and median traces, full found measured-best plans at
  16.89 ms and 18.97 ms;
- on the captured high-skew trace, full selected 16T LPT at 22.857 ms while 8T
  reverse-even measured 17.282 ms, a 24.43% paired reduction;
- measured/predicted rank Spearman on the high-skew manual candidates is only
  0.692, diagnosing a ranking rather than candidate-coverage failure.

These results show that broader planning can expose useful schedules, while
also making cost-model ranking the principal blocker to a near-optimal claim.

A 2026-09-01 Arm-codex NUMA3 80C follow-up closes one bounded part of that
problem. Analytical full now uses a one-step width uncertainty gate when a
narrower calibrated width overlaps the expected winner. On complete
high-skew/median/uniformish traces, the gate improves legacy full by
19.63/22.56/5.43% in paired medians and leaves 0/0.07/0.20% measured-set
regret. The run used 5 warmups, 31 randomized paired rounds, and four rotating
weight copies. It remains development-snapshot evidence pending a committed
runner repeat and does not close the 192-core or general temporal-order gate.

## Claims Allowed Now

The current evidence supports these bounded statements:

- the SVE executor implements a fused, moldable whole-expert substrate with
  explicit task width and tile-window controls;
- direct W2 route stores remove materialization/scatter traffic and provide a
  modest benefit on sufficiently long or skewed routes while short/moderate
  cases remain around the noise floor;
- exact-M specialization removes selected tail overcompute without changing the
  packed ABI;
- event-based contention modeling is materially more appropriate than applying
  one slowdown for an entire heterogeneous call;
- full search can find better plans than homogeneous quick search on some
  measured workloads, but its current ranking can also miss the best fixed
  width and temporal order badly on high skew;
- Plan V2 can execute strict, whole-expert tail-pool, and bounded route-sliced
  schedules without resizing a running task.

## Claims Not Yet Allowed

Do not claim any of the following without new evidence:

- best or state-of-the-art CPU MoE performance;
- a general fused-kernel speedup over current upstream vLLM, oneDNN, or another
  declared production baseline;
- analytical contention prediction within the declared acceptance gate;
- a universal fused-versus-explicit speedup or a current 15--16% direct-route
  improvement;
- quick planning that consistently beats fixed-width greedy execution;
- full planning as an oracle or generally near-optimal planner;
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

The deterministic `dsv4-real-2048-seq70` catalog artifact retains exact top-16
counts and a moment-matched synthetic tail. Label it as reconstructed. The
2026-08-31 supplemental matrix adds three complete single-layer TopK traces
(uniformish, median, and high-skew), but complete many-layer and multi-request
coverage is still missing.

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

- extend the completed single-core machine-local packed-B retention probe to
  active multi-team LLC/refill behavior;
- validate unseen routes, widths, mixed shapes, and multi-LLC placements on at
  least two Arm machines;
- pass isolated MAPE <=10%, contention P90 <=15%, and maximum selected regret
  <=5%, or narrow the claim and domain explicitly;
- report calibration wall time and number of measured points versus the
  empirical table baseline.

### P0: planner

- remove or gate the production quick active-set 8/16 wide-team regressions;
- repeat the analytical-full one-step width gate on a committed snapshot and a
  second Arm machine; its current three-trace 80C result is closed only inside
  that declared domain;
- define the quick/full relationship and the exact candidate space in paper
  pseudocode;
- report measured regret over the full workload matrix;
- either reduce full cold search substantially or define it explicitly as an
  offline reference/autotuner;
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
- Current supplemental measurements:
  `optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md`
- Arm-codex analytical-full gate:
  `optimizations/fused_moe_sve/results/arm_codex_80c_high_skew_planner_gate_20260901.md`
