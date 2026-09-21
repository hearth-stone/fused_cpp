# Localizing the M1 small/large-background time gap

## Finding

The approximately31us historical gap is **not a fixed extra cost**. It is a
repeatable but variable penalty that appears when A/B memory streams coexist
with matrix execution. The experiments narrow its source to that interaction;
they do not identify a unique load queue, issue port, vector register resource,
prefetch mechanism, or DDR latency parameter.

Key evidence:

- With A/B loads and four matrix instructions/K4, large backgrounds are slower.
- Removing the dependence on loaded A/B values does **not** remove the gap:
  matrix operands may be preloaded registers holding the same BF16 values.
- Removing A/B loads while retaining register matrix execution **does** remove
  the gap: approximately192.6–192.8us under either background.
- Keeping A/B loads but replacing the four matrix instructions with four NOPs
  or integer additions does **not** reproduce the positive gap.
- Stronger accumulator serialization makes the victim slower overall but
  reduces its background-sensitive increment.

The old model remains frozen. Do not add a31us constant, an A/B/C bandwidth
coefficient, or a hardware-port correction from these results.

## Two bounded experiments

Stage1 varies compute count/dependencies at fixed M1/1T load geometry. Both
sessions contain36 cells/round,5 warmup+31 recorded rounds, seeds89808/99808.
The first sweep showed that resident matrix operands retain the gap. A focused
stage2 was then chosen to distinguish compute-only slowdown from coexistence:
24 cells/round, two fresh31-round sessions,5 warmups, seeds109808/119808.
This adaptive diagnostic sequence is not a model holdout or a claim of blind
hypothesis selection. No parameters were fitted in either stage.

All comparisons use none,16×M1,16×M120 backgrounds; M120 is ten M12 panels.
The background cores, per-thread allocations, copy rotation, victim shape and
measurement protocol remain fixed within each stage. Every background is an
actual production W13 JIT with synthetic constant inputs, not a full MoE trace.

## Stage1: count and dependency sweep

Values are large-background minus small-background **paired medians**, in us.

| Victim/probe | Session1 gap | Session2 gap |
|---|---:|---:|
| Real W13 | +19.89 | +36.26 |
| A+B loads,0 matrix ops | −18.70 | −11.22 |
| A+B +1 loaded matrix op/K4 | +16.55 | +16.22 |
| A+B +2 loaded matrix ops/K4 | +13.64 | +27.09 |
| A+B +4 loaded matrix ops/K4 (production full-no-store) | +22.62 | +36.66 |
| A+B +8 loaded matrix ops/K4 | +26.34 | +29.05 |
| A+B +4 register-resident matrix ops/K4 | +25.71 | +35.68 |
| A+B +4 matrix ops in one accumulator chain | +9.57 | +15.86 |
| A+B +4 NOPs/K4 | −18.61 | −10.82 |

All positive matrix gaps above have bootstrap95% lower bounds above zero;
the AB/NOP gaps have upper bounds below zero. The gap is not strictly monotonic
in operation count, and this sweep does not establish a physical threshold.
The1/2-op variants consume fewer loaded B columns even though all loads remain;
they are diagnostic interventions, not production-equivalent kernels.

Loaded4 minus resident4 background-gap differences have intervals spanning zero
in both sessions:2.60us [−9.94,5.01] and1.93us [−2.46,10.19]. Thus an immediate
RAW dependency on freshly loaded matrix operands is not necessary for the gap.

Serial4 takes approximately492/502us (small/large,S1), versus422/444us for
loaded4. It is not a proposed optimization; its smaller gap shows that the
exposed penalty depends on execution overlap/critical dependency structure.

## Stage2: confirmation with no A/B traffic and integer control

All values below are paired large−small differences. Brackets are bootstrap95%
intervals across31 round pairs, not simultaneous or population bounds.

| Probe | Session1 delta(us) | Session2 delta(us) |
|---|---:|---:|
| Real W13 | +37.41 [34.82,43.99] | +37.15 [29.12,42.10] |
| AB load-only | −14.11 [−19.68,−12.52] | −12.45 [−15.72,−9.31] |
| AB +loaded4 matrix | **+38.53 [30.48,43.12]** | **+37.62 [35.80,39.98]** |
| AB +resident4 matrix | **+38.96 [27.24,43.15]** | **+33.52 [24.43,39.11]** |
| AB +4 NOPs | −12.92 [−16.91,−9.75] | −12.81 [−15.05,−10.23] |
| No A/B +resident4 matrix | **−.06 [−.38,.08]** | **−.10 [−.60,.43]** |
| AB +4 integer additions | −11.65 [−15.56,−7.54] | −14.13 [−19.26,−11.41] |

Absolute medians in the confirmation:

| Probe | S1 small /large(us) | S2 small /large(us) |
|---|---:|---:|
| AB load-only | 401.98 /386.70 | 397.84 /385.44 |
| AB +loaded4 | 415.67 /455.54 | 420.81 /458.43 |
| AB +resident4 | 419.03 /457.33 | 424.38 /457.13 |
| No A/B +resident4 | 192.79 /192.68 | 192.58 /192.58 |
| AB +integer4 | 398.78 /387.38 | 399.52 /385.27 |

The no-A/B variant retains loop/address arithmetic, parameter reads and stack
save/restore; it is not an idealized arithmetic-peak benchmark. Its null A/B
startup check proves it does not dereference either input stream. Resident
operands contain BF16 1/64, matching the loaded data, not zeros. Integer4 adds
to one scratch integer dependency chain; it controls active integer instructions,
not every possible scheduling or execution-port characteristic of BFMMLA.

These controls argue against pure matrix service slowing under the large
background, and against raw retired instruction count alone explaining the
inversion. They support an interaction specific to the tested matrix/stream
coexistence, without proving a specific microarchitectural bottleneck.

## Counter evidence and causal limits

The victim's CPU304 thread_siblings_list is`304` (no SMT sibling). Existing and
new full/AB measurements have cycles/time ratios near2.9GHz; no large overall
frequency change is observed that would account for the7–9% gap. This does not
exclude every power-management effect on individual execution resources.

In confirmation loaded4, instruction count is853,095 under every background.
Comparing separate small/large counter medians:

| Session | Extra cycles | Extra backend-stall cycles | LLC misses small→large | L2 refills small→large |
|---:|---:|---:|---:|---:|
| 1 | 115,998 | 116,197 | 51,509→98,527 | 130,325→130,324 |
| 2 | 103,902 | 105,309 | 54,999→99,474 | 130,249→130,236 |

At approximately2.9GHz this is the same scale as the observed extra tens of us.
It is **supporting evidence of additional backend waiting**, not an exact stall
decomposition: counter medians are separate, backend events can cover multiple
causes, and a generic backend-stall count is not a memory-latency measurement.

Domain27 DDR read throughput during loaded4 is approximately74GB/s under small
backgrounds and45GB/s under large backgrounds. The aggregate queue ratio is also
lower in the large-background case, while victim LLC misses are much higher.
More LLC misses at similar L2 refill counts are consistent with a changed
lower-cache service path, but counters do not identify A/B addresses or prove
how many cycles each miss costs. DDRC events include background traffic and span
a longer controller gate; they are not victim-specific queue latency.

A plausible interpretation is that changed cache/memory supply interacts with
the matrix execution stream's issue/retirement resources and ability to overlap
memory service. The current controls cannot distinguish queue/window pressure,
vector resource arbitration, prefetch timing or other overlapping mechanisms.
Do not claim that BFMMLA itself simply became slower: the no-A/B matrix control
does not show that slowdown. Do not claim direct loaded-operand waiting is the
sole cause: resident operands retain the gap.

## Implementation and validation

Only Lab files changed. `m1_supply_probe.h` extends the narrow M1 skeleton with
0–8 operations and loaded/resident/serial/NOP/integer choices; no default
production path or production JIT source is modified. Both B-only and standard
loaded4 machine code must match their production counterparts byte-for-byte.
These startup assertions passed in both binaries. Probe output allocations must
remain poisoned; real victim and every active background output are verified.

All load-containing variants keep the same addresses and load counts. M1
logical A8KiB, paired/padded unique A request16KiB, A line coverage64KiB,
total repeated A requests1MiB; B requests8MiB. No-A/B is explicitly the exception.
The analyzer checks instruction deltas against65536 K4 steps per call, allowing
only64 wrapper instructions. All checks passed. Resident initialization adds
three fixed instructions; removing A/B removes five loads per step.

H4096/F512,K4096/N1024,BF16,SVE256,Ntile16,full owner stripe1T. CPU240 controller,
304 victim,288–303 backgrounds; NUMA3 allocation. Persistent touched outputs,
256MiB scrub per cell,4-copy victim and per-background B rotation; background
startup5ms. Existing ordinary allocations/page regime unchanged; actual page
residency not audited. Constant data1/64; no random-input correctness claim.

Stage1 records2,232 cells,stage2 records1,488; each has186 real victim numerical
checks across its two sessions, plus background checks during relevant probe
and empty cells. Core PMU cycles/instructions/L2 refill/LLC read miss/backend
stall;48 DDRC events with per-device enabled-time denominators. Raw mean/std/
P90/P99 and paired intervals are retained. No full timing-only repeat, sanitizer,
full MoE trace, W2, or production optimization benchmark was run.

Stage1 binary SHA256:
`75e5c4ed849a2f55253aeb860b2b87c2902f190181defadb74bc8b290be881e3`.
Stage2 binary SHA256:
`1d4d3c6810eebd67f720ed76f29367e2e99ec95dd76bd23ae447db87ede9637d`.
Production JIT source unchanged:
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
Frozen model SHA256 unchanged:
`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`.
GCC13.2.0,C++17/O3,existing SVE256 build flags. Different stages are analyzed
separately, not pooled across their binaries or grid compositions.

## Reproduction and retained evidence

Remote root`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/`, directories
`tmp/m1_compute_issue_20260908/` and`tmp/m1_issue_controls_20260908/`.
Both retain exact native/header/driver/model snapshots, build_inputs.tar.gz,
binaries, no-PMU/PMU smokes, sessions and legacy W13/W2 smoke. Local raw copies,
build snapshots and report.json use the same relative directories.

Build with the GCC command in`ab_supply_contrast_20260908.md`, substituting the
appropriate tmp suffix and retaining the adjacent`m1_supply_probe.h`.

```sh
# Stage1 remote session1. Session2 uses seed99808 and a new output path.
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/m1_compute_issue_20260908/bench_phase_supply.py \
  --binary tmp/m1_compute_issue_20260908/phase_supply_native \
  --output tmp/m1_compute_issue_20260908/session1.jsonl --issue-sweep --seed 89808
# Stage2: suffix m1_issue_controls_20260908, --issue-controls,
# session1 seed109808; session2 seed119808.
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_m1_compute_issue.py \
  --sessions tmp/m1_compute_issue_20260908/session1.jsonl tmp/m1_compute_issue_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/m1_compute_issue_20260908/report.json
# Repeat analysis with the stage2 suffix; no fitting occurs.
.venv/bin/python -m pytest -q tests/test_moe_phase_supply.py tests/test_moe_ab_supply_contrast.py tests/test_moe_pressure_curve.py tests/test_moe_pressure_response.py tests/test_moe_kernel_response.py
```

47 focused tests pass. Both stages passed full-grid no-PMU and PMU smokes before
formal runs; their old20-cell W13/W2 protocols passed afterward. Retain as a
bounded mechanism-localization reference. Production, frozen response, v8,
planner/pruning and dependencies remain unchanged; no commit or deployment.
