# A-only and A+B supply contrast for the M1 background counterexample

## Decision

**Do not add an A-supply coefficient to the frozen model from this evidence.**
The real approximately31us small/large-background gap repeats, but A-only is
almost unchanged and A+B load-only changes in the **opposite direction**.
The gap persists when computation is retained and the epilogue/output store is
removed. This narrows the problem to load/compute joint execution behavior;
it does not identify a pure compute penalty, a specific queue mechanism, or
independently additive A/B costs. The old model remains byte-for-byte frozen.

## Main paired contrasts

Victim geometry is M1/1T W13,H4096/F512,K4096/N1024. Compare16 M120 background
kernels against16 M1 background kernels, using the same allocations within
each session. All values below are medians of31 round-matched differences in
microseconds; brackets are bootstrap95% intervals, not simultaneous bounds.

| Probe | Session1 large−small | Session2 large−small |
|---|---:|---:|
| Real W13 | **+34.77 [31.57,39.31]** | **+29.29 [23.22,35.12]** |
| B-only | +3.18 [1.14,7.00] | −.08 [−3.15,4.29] |
| A-only | −.33 [−.51,.04] | −.20 [−.45,−.01] |
| A+B load-only | **−15.21 [−17.52,−9.30]** | **−15.74 [−18.36,−10.66]** |
| Full compute, no epilogue/store | **+38.90 [26.76,44.94]** | **+33.57 [22.65,38.48]** |

The prespecified positive-discrimination criterion for A/AB is not met. AB
clearly discriminates the contexts, but its sign is incompatible with simply
charging an extra positive A-supply delay for the slower real workload. A-only's
change is sub-microsecond and has no stable positive direction. A contribution
inside the full kernel is **not excluded**: removing compute or the other load
stream changes request timing, dependencies and cache behavior.

## Absolute medians

These are separate cell medians, so their subtraction need not equal the paired
median differences above.

| Probe | S1 none /small /large (us) | S2 none /small /large (us) |
|---|---:|---:|
| Real W13 | 301.58 /424.44 /459.44 | 303.67 /428.96 /457.43 |
| B-only | 257.34 /335.77 /339.54 | 257.05 /338.72 /338.53 |
| A-only | 27.16 /27.53 /27.29 | 27.42 /27.66 /27.35 |
| A+B load-only | 297.14 /403.82 /387.43 | 300.31 /402.90 /389.67 |
| Full-no-store | 299.63 /420.88 /459.39 | 301.53 /428.30 /459.19 |

For illustration, the session1 difference between full-no-store and AB medians
is approximately17us under small backgrounds, versus72us under large ones.
Those are differences between two distinct instruction streams, **not measured
pure compute time**. The additional matrix instructions change overlap and
the memory request issue process as well as doing arithmetic.

The extra contrast `(AB_large−AB_small)−(B_large−B_small)` is −18.77us in S1
and−12.88us in S2, with both intervals below zero. This is a descriptive
difference-in-differences between probes, not a decomposition of real runtime.

## Probe construction and validation

Production JIT has B-only and full-no-store modes but no A-only/pure AB mode.
The new **Lab-only**`m1_supply_probe.h`implements only the narrow M1 SVE256
load/control skeleton. It does not copy a full operator or modify production
JIT modes, enums, cache keys, public APIs, or model formulas.

The Lab generator preserves the production small-M double-buffered state
machine, N-tile traversal, packed physical8-row A stride, and B-before-A load
ordering. It removes matrix instructions and all epilogue/stores; A-only also
removes B loads, while retaining address/control arithmetic. It is not a
continuous-A bandwidth loop and not a pure-compute isolation experiment.

Before any cell runs:

- Its B-only variant must be **byte-for-byte identical** to the existing
  production M1 B-only machine code; startup aborts otherwise. This passed.
- A-only is called with a null B pointer, B-only with a null A pointer, and
  AB with a null output pointer. These startup checks passed.
- Each timed load/no-store/empty cell must leave the poisoned victim output
  allocation unchanged. Real W13 and every active background's final complete
  output are checked against the existing BF16 SiLU(1) reference0x3f3b.

Dynamic instruction counts support the intended interventions, identically
across all three backgrounds and both sessions (medians):

| Probe | Retired instructions |
|---|---:|
| A-only | 328,804 |
| B-only | 525,413 |
| A+B | 590,950 |
| Full-no-store | 853,094 |
| Real W13 | 858,523 |

AB−B adds65,537 instructions, consistent with65,536 A loads plus one wrapper
instruction. Full-no-store−AB adds exactly262,144 instructions, matching four
BFMMLAs per K4 iteration ×1024 iterations ×64 N tiles. These checks establish
the load skeleton/intervention, not equivalence of timing after removing work.

## A/B accounting

For this M1 geometry:

- Logical A payload:8KiB (4096 BF16 values).
- Unique A bytes requested by paired-row loads, including padding:16KiB.
- Cache-line coverage at64-byte lines:64KiB.
- N traversal repeats that A request sequence64 times:65,536 `ld1rqh`
  instructions,1MiB total requested A bytes. Each instruction requests16 bytes
  and broadcasts into a32-byte SVE register; register replication is not a
  second16-byte memory request.
- B:262,144 vector load instructions,8MiB requested bytes, same packed B
  addresses/order as the production probe.

Payload, instruction-request bytes, cache-line coverage and measured refill
are different quantities. Core counters here are aggregate events, not
address-resolved A/B traffic. The tiny A-only LLC-miss counts do not establish
that A is unaffected or always cache-resident inside the complete kernel.

## Interpretation and next boundary

The no-store result shows that output allocation/touch or the W13 epilogue is
not necessary for this background gap to appear. It does not prove their
contribution is exactly zero. The opposite AB-load-only sign shows that pure
load service times cannot simply be added to account for the real gap.

A remaining hypothesis is that inserting the real matrix instruction and
dependency pattern changes memory-level parallelism, prefetch effectiveness,
cache residency or exposed stalls differently under the two backgrounds.
This experiment does not choose among those mechanisms. A bounded next
intervention could hold load addresses/counts fixed while controlling compute
dependency/spacing; do not fit A/B coefficients or assume a DDR queue cause
from the current result. No further probe or new physical parameter was added.

## Protocol, identity, and reproduction

Two independent processes/sessions, seeds69808/79808,5 warmup+31 recorded
randomized rounds.18 cells per round:five real/probe variants and one empty
control under each of none/16M1/16M120.1,116 recorded cells across both sessions,
including186 numerically checked real victim cells; background outputs are also
checked during probe/empty cells when backgrounds are active.

Arm-codex-internal, NUMA3 memory, allowed CPUs240–319; controller240,victim304,
background288–303, each1T. Backgrounds are actual production W13 JIT kernels,
with synthetic1/64 data, private A/output and four private8MiB B copies per
thread. M120 executes ten M12 panels. Background startup5ms, no panel barriers;
four-copy victim rotation,256MiB scrub per cell,persistent touched outputs.
Ordinary aligned allocations; no page-policy changes, no explicit HugeTLB,
no fresh page-residency audit. Fixed values do not replace broad numerical tests.

Core PMU: cycles,instructions,L2 refill,LLC read miss,backend stall.48 DDRC
events retain their own enabled-time denominators and include victim+background.
The empty **kernel** envelope is approximately.23–.27us in S1, whereas the
controller/uncore gate is approximately739–769us. DDRC averages therefore do
not localize a27us A-only window; they must not be interpreted as A-specific
latency or divided by the shorter victim time. Counter running ratios and
complete grid/numerical scope are checked; full cell statistics are retained.

Old frozen model SHA256 unchanged:
`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`.
Measured binary SHA256:
`c33fc959e0bcfa3c40f2524b8a6e44f2aacc90b04b72f5a5652ce18d375cd2a3`.
Production JIT source unchanged:
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
GCC13.2.0,C++17/O3,SVE256/BF16,Ntile16. New binary with matched load skeleton;
real control repeats the old gap, not a claim of complete binary equivalence.

Remote snapshot/raw directory:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/ab_supply_contrast_20260908/`.
Local raw copies, build_inputs.tar.gz and report.json use the same relative
directory. Actual native/header/driver/JIT inputs and frozen model are archived.

```sh
# Remote GCC command is identical to real_kernel_background_20260908.md,
# with tmp suffix ab_supply_contrast_20260908; copy m1_supply_probe.h alongside cpp.
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/ab_supply_contrast_20260908/bench_phase_supply.py \
  --binary tmp/ab_supply_contrast_20260908/phase_supply_native \
  --output tmp/ab_supply_contrast_20260908/session1.jsonl --ab-supply --seed 69808
# Session2: session2.jsonl, seed79808; both default to5 warmup and31 rounds.
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_ab_supply_contrast.py \
  --sessions tmp/ab_supply_contrast_20260908/session1.jsonl tmp/ab_supply_contrast_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/ab_supply_contrast_20260908/report.json
.venv/bin/python -m pytest -q tests/test_moe_ab_supply_contrast.py tests/test_moe_phase_supply.py tests/test_moe_pressure_curve.py tests/test_moe_pressure_response.py tests/test_moe_kernel_response.py
```

Class E Lab diagnostic.45 focused tests pass;18-cell no-PMU and PMU smokes pass
before formal timing;old20-cell W13/W2 smoke passes afterward. No sanitizer,
full timing-only repeat, full operator or W2 experiment was run. No fitted
coefficient, production calibration, default build, pruning or commit changed.
Retain the probe and negative result as a bounded load/compute interaction
reference, not a new A-supply correction.
