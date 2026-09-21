# Six parameterized kernel-history functions, 2026-09-10

## Representation

The user requested the six row-pair kernel families combined with cold/transition/warm states, and asked whether this should be18 or6 functions. The implementation uses **six callable family objects sharing one function form**, with exact rows, stage, observed history and memory-service scale as arguments. Three named states describe the history; they do not require18 separate code paths or18 constant times.

```text
F_family(exact_rows, stage, preceding_full_panels, memory_scale)
family = ceil(exact_rows / 2), exact_rows in1..12
```

W13 and W2 have separate parameter banks. Exact rows remain visible because odd-row handling differs in the JIT, even when the core row-pair count is shared. Odd rows1/3/5/7/9/11 calibrate their pair; even2/4/6/8/10 are validation points. Full12 receives its own history curve within family6 because it appears repeatedly in the full-M composition, while11 remains the tail member.

The history argument counts preceding full M12 panels that accessed the same stage's B copy. Operational labels are `cold_reference` at0, `transition_reference` between0 and8, and `warm_reference` from8 onward. They are not measured cache-level residency. The continuous curve may be nonmonotonic, including an initial slowdown after the first full block. Warm-state timing can reflect execution/A/output history as well as B reuse.

Three alternatives are evaluated:

- `cold_only`: reuse the zero-history time for every block.
- `three_anchor`: interpolate0/1/8 reference histories and hold the8 value thereafter; the bounded approximation corresponding to three state anchors.
- `history`: interpolate the0/1/2/8/15 measured histories, with history4 excluded from calibration.

## Any-M decomposition and evidence limits

For positive integer `M = 12*q + r`, compose q full12 blocks at successive history values and, if r is nonzero, one r-row tail. For example:

```text
M31 = F6(rows12, history0) + F6(rows12, history1) + F4(rows7, history2)
```

The `decompose` function represents any positive integer M. The calibrated history domain stops at15 preceding blocks, giving M≤192 for cold-start composition. Larger histories require `allow_extrapolation=True`; the terminal curve value is then held constant and the result is marked extrapolated. Long repeated full blocks are represented compactly rather than allocating one result per block.

History belongs to a particular expert/B copy and stage. It must not carry across unrelated weights or from W13 B to W2 B. There is no automatic eviction model: callers must reset/update history when the assumed B state changes. Constant pressure is used by the convenience whole-M composer; dynamic pressure can be supplied to individual block calls by a future caller, but is not validated here.

The prior partial-overlap model supplies normalized C/D/rho ratios. New measured history baselines scale both C and D, preserving the baseline identity; increasing `memory_scale` then acts on D. This is an effective history correction rather than an assertion that every warming effect is a memory-only change. Only no-background histories are measured in this experiment. Every pressure>1 query is explicitly flagged `pressure_history_interaction_validated=False`; warm-state pressure and the full12 pressure response are not validated by this profile.

## New measurements

New optional `kernel_history_native.cpp` calls the unchanged production JIT. Original pressure harness/source/binary snapshots remain untouched. The new grid is exact rows1–12 ×W13/W2 ×preceding panels0/1/2/4/8/15:144 cells. A144-cell smoke precedes two independent5-warm/31-measured sessions, seeds601001/601002, smoke601000.

Each cell runs the complete prefix and final exact-row block with one B copy, advancing A, packed W13 C and W2 route rows as a full stage would. Every prefix panel, the final block and the enclosing prefix-plus-tail interval are timed separately. Output checks cover all logical W13 values and the guard after its physical packed extent; W2 checks every logical output and every untouched element in the192MiB route-output array. All block timings must be positive and their sum must not exceed the enclosing interval.

Foreground CPU316, NUMA3; allowed CPUs240–319 at launch, no background workers. H4096/F512, BF16 SVE256, N tile16. W13 uses degree5 fused SiLU with constant1/64 inputs and verifies BF16 SiLU(1); W2 varies packed A by row and verifies exact FP32 results. Four B copies rotate by round;256MiB scrub and a fixed5ms wait precede every cell. Full W13 B8MiB, W2 B4MiB, full owner stripes `(1,0,0,1,1)`. Ordinary vector allocations, no explicit HugeTLB or sampled page-residency claim.

The same standalone protocol supplies its own cold and repeated-history baselines; earlier real-expert absolute times are not pooled into this fit. These are synthetic stage calls, not full MoE runtime/gather/merge measurements.

## Calibration and validation protocol

Calibrate only session1 odd rows plus full12, histories0/1/2/8/15:70 final-block medians. Keep history4, even rows2/4/6/8/10 and all session2 points for validation. This split was specified before the formal new sessions were inspected. It repeats known kernel types/protocols on the same machine; it is not a cross-machine or real-input holdout.

Cold-only uses14 calibration medians; three-anchor uses42; history uses70. Compare all alternatives on the same held-out population. Evaluate both final-block error and full-M composed time, because good tail estimates alone do not establish accurate totals. Prefix timing samples are not additional calibration observations. The complete actual prefix-plus-tail interval is the full-M reference.

## Measured results and decision

Both sessions completed: 10,512 numerically checked cells including the smoke and warmups, with 8,928 measured cells. There are 288 condition medians, of which70 calibrate the history model and218 are held out. All native/build/driver stderr files are empty. The largest block CV across these condition groups is5.91%.

Errors below use the same218 held-out condition medians; MAE is in microseconds. Full-M actual time encloses the complete prefix and tail, rather than only the last block.

| Model | Block MAE | Block MAPE | Full-M MAE | Full-M MAPE | Full-M maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold-only |13.634|5.203%|58.821|1.276%|450.410|
| Three anchors (0/1/8) |6.332|2.704%|12.386|0.434%|65.871|
| Continuous history (0/1/2/8/15) |2.593|0.808%|12.810|0.344%|59.765|

Continuous history improves block error substantially over this specific three-anchor approximation. Full-M MAPE and maximum error also improve, but full-M MAE increases by0.424us. W2 full-M MAE rises from14.811 to16.030us; W13 falls from9.960 to9.590us. Summed component errors can cancel differently, so better block estimates do not imply improvement in every aggregate metric. This does not compare against every possible18-function formulation.

For continuous history, all144 second-session points have block MAPE0.754% and full-M MAPE0.352%. The48 history4 points have block MAPE2.082% and full-M MAPE0.291%; the nonlinear transition between histories2 and8 remains the weakest interpolation region. The120 held-out even-row points have block MAPE0.865% and full-M MAPE0.387%, supporting the paired-row approximation under this protocol.

The M1 history is visibly nonmonotonic (session1 final-block medians, us):

| Stage | History0 | History1 | History2 | History4 (held out) | History8 | History15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W13 |306.45|374.80|325.33|286.27|282.73|282.63|
| W2 |186.90|227.49|149.78|136.12|133.21|131.28|

The second session repeats the first-prefix bump. These measurements justify retaining history explicitly rather than imposing a monotonic cold-to-warm curve. They do not isolate physical B residency from other execution history.

The largest continuous-history full-M error is session2 W2, final rows1 after8 full panels (M97): actual5082.180us versus predicted5141.945us (+59.765us). Its tail alone is131.760us versus133.210us; most of the discrepancy comes from the accumulated full-panel estimates.

Retain the six parameterized callable objects as a Lab reference. This experiment validates no-background block composition and bounded history interpolation. Warm-state competition, eviction, changing pressure within a stage, multi-thread teams and planner selection are independent unvalidated extensions. The previously selected experimental planner baseline remains unchanged.

## Validation completed

- `.venv/bin/pytest -q tests/test_moe_kernel_history.py`:6 passed, covering family membership, exact-row composition, nonmonotonic history, explicit bounded extrapolation and pressure caveats.
- The144-cell native smoke and both144×36 formal sessions passed all logical-output, guard, identity, CPU and timing-envelope checks on CPU316/NUMA3.
- Ruff checks on the three new Python implementation files and the focused test pass; `clang-format --dry-run --Werror` on the native harness and `git diff --check` pass.
- No full production regression or planner benchmark was run for this Lab-only change; no production adoption or end-to-end improvement is claimed.

## API and reproduction

```python
from pathlib import Path
from optimizations.fused_moe_sve.benchmarks.kernel_state_model import SixKernelStateModel

model = SixKernelStateModel(Path("tmp/kernel_history_state_20260910/fitted/model.json"))
block = model.functions[3](7, "w13", history=2)
whole = model.predict_m(31, "w13")
```

```bash
.venv/bin/pytest -q tests/test_moe_kernel_history.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/kernel_history_state_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/kernel_history_state_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_kernel_state_model.py \
  --sessions tmp/kernel_history_state_20260910/session1.jsonl tmp/kernel_history_state_20260910/session2.jsonl \
  --overlap tmp/small_m_overlap_20260910/validated/model.json \
  --output-dir tmp/kernel_history_state_20260910/fitted
```

Use fresh outputs for reruns. Data/scripts/native binary are retained in `tmp/kernel_history_state_20260910/` locally or on `Arm-codex-internal` under `/home/zhangxu/codex/fused_cpp/` as appropriate. Build flags are C++17/O3/pthread, armv8.2-a+bf16+sve, SVE256, unchanged production JIT source. Native binary SHA256 `dfd5688c22bb6a5785d6e46738d26217209248abbbc73b2c8eb6c4d3d1913341`; `build_identity.txt` records source/JIT identities. No production kernel, planner default, calibration schema or active model baseline is changed; no commit was made.

## Follow-up: address and gap controls

The [address/gap diagnostic](kernel_history_controls_20260910.md) fixes M1 tail addresses and scans0–5000us inter-panel busy-wait gaps. The post-prefix slowdown persists at fixed addresses and through5ms in both W13/W2 sessions, weakening simple address-offset or short-lived backlog explanations. Its magnitude varies materially between sessions, so the history curve remains a protocol-specific empirical reference rather than an identified B-temperature law or a stable universal transition penalty.

## Follow-up: history and competition are not universally separable

The [history/pressure factorial](kernel_history_pressure_20260910.md) measures M1 W13/W2 at histories0/1/8 under six background conditions, in two sessions. History1 independently slows the target, but that incremental slowdown largely vanishes under competition; independent-product MAPE is10.06%, with all20 pointwise interaction intervals below1. History8 produces pressure-dependent reuse benefits (product MAPE13.92%). Thus the unvalidated history/pressure multiplication in this prototype must not be promoted as a validated general rule. A conditional history factor or history-dependent competition response remains future work; the active planner baseline is unchanged.
