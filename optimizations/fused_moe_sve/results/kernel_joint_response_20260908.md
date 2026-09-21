# Exact-M W13 baseline and conditional supply response

## Decision

**Keep exact-M isolated calibration as the next bounded direction; do not adopt
a fixed compute/memory fraction or deploy this response model.** Frozen v8 and
Production are unchanged. No W2 expansion was run.

On Arm-codex-internal, H4096/F512, W13 K4096/N1024, M1/4/8/12 ×1/2/4T:

- Session1 scrubbed-B isolated medians predict the same cells in session2 with
  **0.437% MAPE, 1.396% maximum error**, versus v8 stage-only **7.499% MAPE**.
  This is repeatability of measured shapes, not interpolation or a compute/memory decomposition.
- A conditional B-supply response gives **2.008% MAPE** on 96 held-out
  kernel/condition/session medians. A constant preloaded baseline gives 4.873%,
  but a stronger **cache-state-matched isolated lookup gives 2.252%**.
  The latter is an additional diagnostic, not a replacement for the preregistered gate.
- Only **3/12 kernel/width combinations** pass every preregistered gate.
  M1/1T still has a 17.074% worst error. Low average error does not authorize export.

## Protocol and scope

Class E hardware measurement plus Lab-only class M conditional analysis. No
production schema, kernel implementation, default dispatch, calibration, or
pruning changes. Rollback is confined to the new joint harness mode, driver,
analyzer, and documentation; legacy phase/panel/A-source protocols remain.

Each session uses one process, persistent touched A/B/output allocations,
four B copies, 256MiB scrub before every cell, and randomized cell order.
Victims are CPU304–307 according to width; background readers CPU288–295;
controller CPU240; allocation policy NUMA3; allowed CPUs240–319.
All use the existing production JIT service, SVE256, BF16, Ntile16.
W13 inputs are constant 1/64 and full packed output is checked against BF16
SiLU(1), 0x3f3b. This is not a randomized numerical-coverage test.

Per round: 72 real cells (4 M ×3 widths ×2 B policies ×3 peer counts),
18 independent M1 B-only controls, 9 empty controls. Two sessions, seeds
9408/19408, each five discarded warmup rounds and 31 recorded rounds:
6,138 recorded cells total, including 4,464 numerically checked real cells.
Separate no-PMU and PMU 99-cell correctness smokes passed before formal sessions.
The legacy20-cell W13/W2 protocol also passed a one-round no-PMU compatibility smoke after analysis.

Core PMU: cycles, instructions, L2 refill, LLC read miss, backend stall.
Each worker independently reset/enables/disables/reads its counter group per cell;
all recorded running/enabled ratios pass ≥0.99. No DDRC/L3C counters in this grid.
Reported duration is earliest victim start to latest victim end, not the
controller gate. Empty median envelopes are at most 0.36us across both sessions;
session1 controller gates are approximately20–28us and are not charged to kernels.
The formal sessions include PMU instrumentation; no full timing-only repeat was run.

The B-preloaded condition performs two stripe reads before peer startup, outside
timing. **It is not a pure-compute or guaranteed L2-hit condition.** The per-worker
B footprints are8/4/2MiB, all exceeding the measured private L2 capacity1.25MiB.
Preload can mostly affect LLC residency; the subsequent5ms peer startup can
change it again. Likewise scrub is a reproducible intervention, not proof every
load comes from DRAM. These conditions are diagnostic and do not redefine the
production cold-weight workload policy.

## Frozen fit and gates

Let H be session1 peers0 B-preloaded actual W13 time, and D be the matched
**separately measured B-only kernel duration**, not bytes/bandwidth or DDR latency.
Fit only the two peers0 states:

```text
beta_raw = (T_scrubbed - H) / (D_scrubbed - D_preloaded)
beta = clip(beta_raw, 0, 1)             # diagnostic prediction only
T_hat(D) = H + beta * max(D - D_preloaded, 0)
```

Predictions for peers4/8 in both sessions use the frozen session1 H/beta and
the held-out state's measured supply D. Session2 beta is used only to assess
repeat stability. This is a **conditional oracle-supply test**, not a plan-visible
pressure predictor; it cannot yet score an unmeasured plan.

Preregistered gates: supply separation≥10%, raw beta within[0,1] in both sessions,
absolute beta drift≤0.15, held-out MAPE≤5%, maximum error≤10%, and no worse than
constant H. Clipping does not rescue invalid raw beta. Gates are unchanged.

| M | T | beta S1 / S2 | Response MAPE | Max error | Matched lookup MAPE | Gate |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | 1 | .649 / .739 | 6.32% | 17.07% | 8.68% | fail: error |
| 1 | 2 | 1.480 / 1.621 | 6.19% | 12.30% | 3.53% | fail: beta/error |
| 1 | 4 | .505 / .718 | 1.52% | 3.05% | 2.23% | fail: separation/drift |
| 4 | 1 | .169 / .202 | 3.08% | 9.06% | 3.56% | pass |
| 4 | 2 | 1.035 / 1.151 | 2.87% | 5.13% | 3.79% | fail: beta |
| 4 | 4 | .276 / .376 | 1.67% | 3.72% | 2.06% | fail: separation |
| 8 | 1 | −.019 / −.082 | .42% | 1.14% | .49% | fail: beta |
| 8 | 2 | .116 / .141 | .63% | 1.38% | .61% | pass |
| 8 | 4 | .765 / .873 | .62% | 1.19% | 1.34% | fail: separation |
| 12 | 1 | .031 / −.035 | .26% | .47% | .29% | fail: beta |
| 12 | 2 | .002 / .054 | .29% | .45% | .29% | pass |
| 12 | 4 | .109 / .083 | .24% | .42% | .16% | fail: separation |

Each row averages eight held-out medians equally; no per-round independence or
bootstrap confidence claim is implied. Full cell mean/std/P90/P99 and counter
medians are in the JSON report. Negative beta near zero can be noise or a
non-monotonic intervention response; it is not negative physical memory cost.
4T separation is only8.24%/8.49%, so its attractive errors do not identify a
reliable response coefficient.

## What this explains, and what it does not

1. **Exact-M baseline matters.** Scrubbed isolated M1/2T is221.10/221.32us;
   M4/2T241.28/241.91us. Frozen v8 has respective second-session errors
   −13.87%/−21.21%, whereas a same-shape first-session lookup is within0.3%.
   This experiment uses a native kernel envelope, not the prior full-workload
   stage wrapper, so absolute values must not be substituted across protocols.
2. **Exposure differs by M, but it is not a constant percentage.** Session1
   scrubbed M1/1T changes300.49→400.98us with eight peers; M8/1T changes
   757.61→760.02us. Their LLC-miss counts both increase substantially
   (6,975→69,830 and6,777→64,793). This is consistent with more hidden memory
   delay for M8, but aggregate PMU does not uniquely separate A/B or overlap.
3. **B-only response is not interchangeable with full-kernel response.** For
   the M1/1T session1 eight-peer case, the model predicts332.52us versus400.98us.
   B-only increases248.12→297.49us, a different response from the real kernel.
   In session2, real time is339.91us and B-only259.29us; LLC misses also fall
   to24,405/26,680. Thus the same reader count did not recreate identical supply
   conditions across allocations/sessions. The measured supply proxy tracks
   some variation but does not fully explain its time impact. No specific DDR
   queue cause is established by these counters.
4. **The physical decomposition is not identified.** Checking
   `T=F+C+D-alpha*min(C,D)` with v8's M12-derived compute proxy leaves3/12
   cases algebraically unidentifiable;6/12 other cases require out-of-bounds
   alpha or negative F. Only3/12 satisfy physical bounds, and even those use a
   proxy C, not independently measured pure computation for each kernel.
   Values such as beta>1 indicate proxy/form mismatch, not >100% memory time.

## Next bounded step

For the user's current **small-T isolated** objective, prioritize an exact-M /
kernel-family baseline with independent unseen-M validation, especially kernel
boundaries and tails. Do not claim these four measured M points establish
interpolation accuracy. Keep memory sensitivity as a separately labelled
diagnostic; do not replace wide/narrow terms or enlarge W2 probes on this result.
If returning to physical decomposition, first obtain matched kernel-family
compute/supply interventions and identify overlap; merely fitting more fractions
to total time does not solve the ambiguity.

## Reproduction and identity

Source commit at launch: `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`, dirty Lab tree.
GCC13.2.0, Linux5.10.0-247.0.0.146.oe2203sp4.aarch64, THP policy`always`, ordinary
aligned allocations (no explicit HugeTLB; actual page residency not sampled here).
Binary SHA256:`e6d9b49e07127032050f179d42a89d45e6e2378a1aa18697eddb9474e983db8c`.
Native source SHA256:`1340496d49617324b0b505e3da94924f520b4eaeee986c1ac7f3ea5748f20ce3`.
Production JIT source SHA256:`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
The standalone binary links that JIT source, not the Python extension.
Frozen v8 JSON SHA256:`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
The reused v8 helper uses degree4; the native W13 path uses its existing degree5
SiLU approximation. v8 is an unchanged diagnostic reference, not a new matched
epilogue calibration.

Raw inputs and generated output stay outside source control:
`tmp/kernel_joint_response_20260908/{session1,session2,correctness,pmu_smoke}.jsonl`,
`report.json`, `build_inputs.tar.gz`. The remote directory has the same suffix
under`/home/zhangxu/codex/fused_cpp/` and retains the binary.
Session1 SHA256:`b5d7e5c162093ba00bc44388ce4de3bf15cf1145d43a4e97b2a89bfc99154007`;
session2:`9d056ea81170ce9b1914348ad2ef0ea13db9ac33b0d599ca6cbe819dbdc86192`.

```sh
# Remote build, from repository root; driver and dependencies copied to tmp first.
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm \
  -I3rdparty/xbyak_aarch64 -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/kernel_joint_response_20260908/phase_supply_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/kernel_joint_response_20260908/phase_supply_native
# Session2 uses seed19408 and a new session2 output; existing output is never overwritten.
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/kernel_joint_response_20260908/bench_kernel_response.py \
  --binary tmp/kernel_joint_response_20260908/phase_supply_native \
  --output tmp/kernel_joint_response_20260908/session1.jsonl --seed 9408
# Local analysis
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_kernel_response.py \
  --sessions tmp/kernel_joint_response_20260908/session1.jsonl tmp/kernel_joint_response_20260908/session2.jsonl \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --output tmp/kernel_joint_response_20260908/report.json
.venv/bin/python -m pytest -q tests/test_moe_kernel_response.py tests/test_moe_phase_supply.py
```

Validation assessment: share with caveats. The conditional arithmetic and
complete-cell checks pass;20 focused tests passed, including frozen-fit leakage
protection. The measured supply input,
cache intervention, fixed input values, limited shape set, two-session scope,
and instrumentation prevent a production accuracy claim. No planner or full
operator tests were run because no production behavior changed.
