# W13 memory-pressure response curve

## Result

The background load successfully increases observed DDR traffic and B-only
service time. **M1 slows substantially; M12 has a small but measurable slowdown,
without a replicated >2% crossing in the tested range.** This supports different
exposure to supply degradation, not a measured compute/memory fraction or a
proven physical compute-to-memory-bound threshold.

At16 readers, domain27 DDR read throughput is approximately72–76 GB/s during
real kernels. For1T, independently measured B-only time grows approximately
25.6% in both sessions. M12's small response therefore does not mean the
background intervention failed. No claim is made that this saturates the machine.

| Width | M1 slowdown S1 / S2 | M12 slowdown S1 / S2 | M12 time, zero →16 readers, S1 |
|---:|---:|---:|---:|
| 1T | 34.16% / 32.86% | 1.05% / 1.31% | 1205.70 →1218.73 us |
| 2T | 12.43% / 13.37% | 1.34% / 1.53% | 605.63 →613.88 us |
| 4T | 9.71% / 10.02% | 1.74% / 1.47% | 305.90 →310.69 us |

Percentages are medians of31 round-matched ratios, not ratios of displayed
medians. At16 readers M12's bootstrap lower bounds are positive in all six
width/session combinations: there is a small repeatable cost, not zero effect.
The preregistered >2% operational criterion is separate from statistical evidence
of any nonzero effect. It is not a physical bottleneck classifier.

## Detailed 1T curve

| Readers | M1 slowdown S1 / S2 | M12 slowdown S1 / S2 | DDR GB/s during M12 S1 / S2 | B-only slowdown S1 / S2 |
|---:|---:|---:|---:|---:|
| 0 | 0 / 0% | 0 / 0% | 2.88 / 2.86 | 0 / 0% |
| 1 | 11.79 / 10.16% | .09 / .19% | 21.16 / 21.28 | 9.69 / 8.09% |
| 2 | 5.73 / 5.45% | .07 / .13% | 20.15 / 19.96 | 5.40 / 4.98% |
| 4 | 4.77 / 5.15% | .10 / .23% | 20.61 / 20.67 | 4.28 / 3.99% |
| 8 | 14.72 / 13.35% | .36 / .50% | 38.52 / 38.58 | 10.53 / 11.05% |
| 12 | 24.11 / 24.06% | .43 / .86% | 56.16 / 55.98 | 16.71 / 18.13% |
| 16 | 34.16 / 32.86% | 1.05 / 1.31% | 72.54 / 72.34 | 25.66 / 25.58% |

For M12/1T, the domain27 occupancy/command ratio changes23.02→50.28 in
session1 and23.24→50.67 in session2. This is an aggregate queue feature in
counter units, **not measured victim request latency in nanoseconds**.

Reader count is not proportional to achieved bandwidth, nor monotonic in kernel
slowdown: one reader hurts M1 more than two/four in both sessions. These low-count
conditions have different active footprints and request concurrency. The data
do not isolate the cause; do not explain the reversal solely with average DDR
bandwidth or force a monotone fit through the prescribed count grid.

The static curve uses independent B-only slowdown on x and real-kernel slowdown
on y, with three width facets and separate session markers/dashes. Lines only
guide the eye. Curve and all84 cells/session are retained under
`tmp/memory_pressure_curve_20260908/curve.png` and`report.json`.

## Experimental contract

- Class E Lab diagnostic, no cost-model fitting or production adoption. Reuses
  the existing native phase harness and production JIT internal service.
- Arm-codex-internal, H4096/F512, W13 K4096/N1024, BF16,SVE256,Ntile16,
  full owner stripes (window0),1/2/4T. Total B8MiB; owner stripes8/4/2MiB.
  Exact-M kernels M1/M12, existing degree5 W13 SiLU, constant inputs1/64.
- ControllerCPU240, victims304+, background readers288–303, all NUMA3 memory.
  Victim and background cores are disjoint and in LLC domainCPUs280–319.
  No other load was observed at preflight (load average0.00/0.01/0.00).
- Read-only background: each active reader traverses its own32MiB allocation,
  one uint64 per64-byte line,64KiB accounting chunks. Total active footprint
  changes with reader count; this is a synthetic streaming/cache-pressure load,
  **not a pure bandwidth-only intervention or a faithful mixed-M workload**.
- Background starts after preparation and runs5ms before victim release.
  Persistent touched output, same allocations inside each session,4-copy B
  rotation,256MiB scrub before every cell, no B preload in this experiment.
  Scrub does not establish that every request must come from DRAM.
- Grid:2 real M ×3 widths ×7 reader levels=42 real cells,21 B-only controls,
  21 empty controls. Each session runs5 warmup+31 recorded randomized rounds;
  seeds9808/19808. Two independent processes/allocations;5,208 recorded cells,
  including2,604 numerically checked real calls.
- Real output checked against BF16 SiLU(1)=0x3f3b; B-only/empty records explicitly
  carry numerical_checked=false. Constant inputs do not replace broad numerical
  tests; no production arithmetic was changed.
- Per-worker core PMU: cycles,instructions,L2 refill,LLC read miss,backend stall.
  DDRC:48 events across domains25/27 (flux_rd,read_cmd,read_cmd_occupancy).
  Counter running/enabled ratios≥0.99 required. No L3C in this grid.
- Kernel elapsed time is earliest victim start to latest victim end. DDRC
  counters cover the longer controller gate; rates use each device's own enabled
  interval, never the kernel time denominator. DDR traffic includes victim and
  background; software background_line_bytes is not physical DDR traffic.
- Existing GCC13.2.0/Linux5.10/SVE256 environment; ordinary aligned allocations,
  prior THP policy always; no explicit HugeTLB and no per-cell page-residency audit.

## Decision and limitations

Prespecified decision: an operational slowdown level requires a paired median
>2% with bootstrap95% lower bound>0 in both sessions. First observed levels are
M1:1/8/4 readers at1/2/4T; M12:none. These are observed grid levels, not precise
thresholds or simultaneous multiple-comparison confidence guarantees.

Do not fit a knee where the data have not traversed it. In particular, M12 might
hide most incremental supply cost, but this experiment has not independently
identified its compute service time, in-kernel memory latency, or overlap.
Likewise B-only25.6% degradation need not equal M12's internal memory degradation.

The next independent question, if pursued, is whether a controlled stronger
intervention can increase measured supply degradation beyond this range while
keeping footprint/topology fixed. More reader count alone changes context and
is not evidence of a bandwidth threshold. No expansion or model deployment was
performed in this task. Keep this probe as a bounded curve/PMU regression reference.

## Reproduction and validation

Source baseline HEAD`c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus owned Lab
changes in an already dirty worktree. Native binary SHA256:
`aec4ec8050ecb32910ec92fc40b64ea049edba5e886891a13c3e39df45c7e8b6`.
Production JIT source SHA256:
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
Frozen v8 is not loaded or refitted. Old and new protocols/data are not pooled.

Remote raw/binary/build snapshot directory:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/memory_pressure_curve_20260908/`.
Local raw inputs, generated report and build_inputs.tar.gz use the same relative
directory. Original JSONL outputs are opened exclusively and never overwritten.
Build command is the prior kernel-joint report's GCC command with the tmp suffix
changed to`memory_pressure_curve_20260908`.

```sh
# Remote, after building in the new snapshot directory.
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/memory_pressure_curve_20260908/bench_phase_supply.py \
  --binary tmp/memory_pressure_curve_20260908/phase_supply_native \
  --output tmp/memory_pressure_curve_20260908/session1.jsonl --pressure-curve --seed 9808
# Second independent invocation: session2.jsonl, seed19808.
# Smokes use --warmup 0 --rounds 1, once with --no-pmu and once with PMU.
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_pressure_curve.py \
  --sessions tmp/memory_pressure_curve_20260908/session1.jsonl tmp/memory_pressure_curve_20260908/session2.jsonl \
  --output tmp/memory_pressure_curve_20260908/report.json
# Optional static plot uses an isolated dependency environment, not project dependencies.
uv run --no-project --with matplotlib python optimizations/fused_moe_sve/benchmarks/analyze_pressure_curve.py \
  --sessions tmp/memory_pressure_curve_20260908/session1.jsonl tmp/memory_pressure_curve_20260908/session2.jsonl \
  --output tmp/memory_pressure_curve_20260908/report.json --plot tmp/memory_pressure_curve_20260908/curve.png
.venv/bin/python -m pytest -q tests/test_moe_pressure_curve.py tests/test_moe_phase_supply.py tests/test_moe_kernel_response.py
```

23 focused tests pass. No-PMU and PMU84-cell smokes passed before formal timing;
the old20-cell W13/W2 no-PMU protocol passed after the change. Production/full
operator tests and a full timing-only repeat were not run; no corresponding
correctness, end-to-end speedup, or instrumentation-free accuracy claim is made.
Rollback is limited to the Lab curve mode, analyzer, tests and documentation.
