# SPE control, exact-IP and profiling-overhead audit

Status: implementation and two-session audit complete. Control isolation and
reported capture-integrity checks passed; neither tested sampling period is a
low-perturbation measurement of these kernels.
No production kernel, planner, calibration or model parameter changes.

## Scope and protocol

Class E Lab measurement audit. This follows the historical
[SPE capability attempt](spe_capability_20260908.md), whose perf enable ACK added
5.57 ms (isolated) versus 207.69 ms (38/12 background) after preparation.
The old single-cell timings are not a reference performance dataset.

The new `bench_spe_audit.py` enables SPE and waits for ACK **before** sending
`PREP_GO`. The native command executes the existing preparation (persistent
workspace, poison, both-domain 256 MiB scrub, background startup and 5 ms lead-in)
and then immediately calls `Run()`. No Python/perf handshake remains between
preparation and execution. `prepared_to_gate_ns` includes native worker dispatch
and readiness; it is not the same endpoint as the old Python PREP-to-GO metric.
It is not claimed to be zero or free from scheduling effects.

SPE records include preparation. `SPE_INFO` reports each actual function pointer,
exact generator size and machine bytes after byte-for-byte verification.
Filtering requires AUX TID/CPU, EL0, aligned PC, and `start <= PC < start+size`.
It does not use the size of an anonymous executable mapping. B-only is 264 bytes;
full-no-store is 340 bytes. Neither probe writes the output, and neither is a
GEMM output-equivalence test. Existing native load/compute skeleton, no-store,
and real-background numerical checks remain enabled.

The installed perf warns that `-t` overrides `-C`. The effective method is
TID-attached sampling of the worker pinned to CPU 304, checked against raw AUX
identity. Do not claim that the `-C` option independently enforces the filter.

## Experiment design

- Arm-codex-internal, root `/home/zhangxu/codex/fused_cpp`.
- NUMA3, allowed CPUs 240–319, controller 240, foreground 304.
- M1/1T, K4096, N1024, SVE256/BF16, B-only and full-no-store.
- Background: isolated or 50 real M12/1T tasks (38 same LLC, 12 other LLC).
- Same process and weight allocations within each session, 4-copy rotation.
- Every round randomizes all condition/probe/profiling combinations; matched
  comparisons use the same round and copy. Five warmup rounds are excluded;
  31 measured paired rounds per combination, independently repeated twice.
- Profiling modes: no perf (0), attached but disabled (-1), period 1024,
  period 4096. Jitter and user-mode load filter enabled; 16 MiB AUX buffer.
- No fitting or reuse of profiled timings as cost-model calibration.

The purpose is to quantify interference, not to require a low-overhead result.
Clean recorded AUX does not establish unbiased hardware sampling. Unknown
latency fields stay unnamed and raw; TIME_CONV is not used for attribution.

## Source, build and artifacts

Lab implementation: `bench_spe_audit.py`, `analyze_spe_audit.py`, and additive
`SPE_INFO`/`PREP_GO` commands in `phase_supply_native.cpp`. Old PREP/GO remains.
Rollback boundary is these Lab additions; no supported ABI/default dispatch changes.

Remote artifact directory: `tmp/spe_control_audit_20260909/`, retaining smoke,
session metadata, incremental cell JSONL, per-cell perf data and logs. The parser
retains decoded dumps and a successful-decoder marker; partial decode artifacts
must not silently be reused. Exact schedule/completion checks precede analysis.

Binary SHA256: `489e13e96c5b244252a401d28b6820bb11847f37892ce6a77a555e0caf0b954b`.
Local HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus uncommitted Lab
changes; this is not a clean-commit production build. Existing unrelated dirty
work was preserved. No commit was requested or made.
Unchanged production JIT SHA256:
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
Compiler GCC 13.2.0; C++17/O3/pthread, `-march=armv8.2-a+bf16+sve`,
`-msve-vector-bits=256`, existing Xbyak build definitions and sources.
Kernel/perf versions are retained in perf headers. THP policy `always`; session1
process smaps_rollup observed 3,141,632 kB AnonHugePages and zero explicit HugeTLB.
This is process-level backing evidence, not a per-allocation page guarantee.

```sh
# From remote project root, after syncing the named Lab files/dependencies.
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm \
  -I3rdparty/xbyak_aarch64 -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/spe_control_audit_20260909/phase_supply_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/spe_control_audit_20260909/phase_supply_native

numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/spe_control_audit_20260909/bench_spe_audit.py \
  --binary tmp/spe_control_audit_20260909/phase_supply_native \
  --output-dir tmp/spe_control_audit_20260909/session1 --seed 19508
# Independent session2 uses its own output directory and seed 19509.

numactl --physcpubind=240 --membind=3 .venv/bin/python \
  tmp/spe_control_audit_20260909/analyze_spe_audit.py \
  tmp/spe_control_audit_20260909/session1 \
  tmp/spe_control_audit_20260909/session2 \
  --output tmp/spe_control_audit_20260909/audit_summary.json
```

## Validation

Local: 70 tests passed using
`.venv/bin/pytest -q tests/test_moe_spe_cell.py tests/test_moe_spe_audit.py tests/test_moe_phase_supply.py tests/test_moe_dual_m12.py`.
Targeted Ruff and `git diff --check` passed. Native build and all 16 smoke cells
completed. A real smoke dump additionally passed the AUX byte-range ledger.
Unit tests cover ACK termination, recorder exit policy, schedule/rotation,
metadata validation, exact range boundaries, EL/TID filtering, loss flags,
missing latency and incomplete byte ranges.

Smoke is not a performance estimate: one round, no warmup. It suggested substantial
profiling overhead and motivated the formal repetitions below. Do not interpret
its single-sample bootstrap endpoints as confidence bounds.

Target integration also passed old PREP/GO W13 and W2 numerical checks (M1/1T,
isolated, base stage protocol), AB `SPE_INFO` byte identity (276 bytes), and
rejection of unsupported `SPE_INFO 0`. An initial manual W2 check mistakenly
used the W13-only `dual` protocol and was rejected by its existing shape guard;
rerunning with the supported base protocol passed. No guard was relaxed.

## Results

Both sessions completed 576 cells: total 1,152, including 160 warmup and 992
measured cells. Of these, 496 measured cells used active SPE. All native scope,
background and correctness checks passed. Binary identity matched across sessions.
The full summary is retained remotely and locally under
`tmp/spe_control_audit_20260909/audit_summary.json`.

### Control isolation

Across all session/condition/probe/mode groups, median native preparation-to-gate
delay was 142.05–516.06 us; the largest individual measured delay was 1,122.23 us
(an unprofiled isolated cell). The old 207.69 ms post-preparation perf handshake
is structurally absent: perf readiness/enable ACK occurs before PREP starts.
This does not claim perf ACK itself became faster. Background and isolated
native dispatch delays still differ, and old/new timer endpoints differ.

### Exact-IP and reported capture integrity

| Check | Session 1 | Session 2 |
| --- | ---: | ---: |
| Active measured files decoded | 248 | 248 |
| Decoded records before exact-IP filter | 2,846,498 | 2,840,228 |
| Kernel records retained | 56,266 | 56,846 |
| Out-of-range/non-kernel records excluded | 2,790,232 | 2,783,382 |
| Files failing reported capture checks | 0 | 0 |
| Decoder stderr nonempty | 0 | 0 |

The combined filter retains 113,112 kernel records and excludes 98.01% of the
decoded records. There were no observed wrong AUX identities, reported lost/error
records, nonzero/unparsed AUX flags, incomplete/orphan records, missing kernel
TOT/ISSUE fields, unparsed latency fields, packet-offset gaps or declared AUX
length mismatches. All raw bytes remain available for rechecking the parser.

This is **not proof of zero hardware sampling bias/collision**. SPE does not
record every load. Each file still reports unhandled TIME_CONV, which this
single-cell exact-IP method does not use. Raw latency units and unnamed counter
semantics remain unverified; they are not relabeled DDR queue latency.

### Profiling overhead

Times are median kernel microseconds. Percentages are median of matched-round
relative changes versus **no perf**, not ratios of the displayed medians and
not cost-model prediction errors. Each entry has 31 paired rounds per session.

| Probe / background | No perf us S1 / S2 | Period 1024 us S1 / S2 | Overhead % S1 / S2 | Period 4096 us S1 / S2 | Overhead % S1 / S2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-only / isolated | 260.91 / 260.09 | 379.06 / 383.43 | 45.60 / 47.64 | 303.49 / 304.86 | 16.99 / 17.49 |
| Full-no-store / isolated | 300.04 / 301.45 | 698.60 / 694.84 | 132.18 / 129.79 | 478.92 / 489.28 | 58.27 / 62.15 |
| B-only / 38+12 | 1074.48 / 1071.67 | 1259.37 / 1256.08 | 17.32 / 17.34 | 1146.09 / 1154.90 | 6.90 / 7.29 |
| Full-no-store / 38+12 | 1301.48 / 1290.57 | 1911.51 / 1911.57 | 48.87 / 47.06 | 1602.00 / 1609.59 | 22.74 / 24.82 |

Paired-median 95% IID bootstrap intervals (2,000 resamples; independent sessions
provide a separate repeatability check):

| Probe / background | 1024 S1 | 1024 S2 | 4096 S1 | 4096 S2 |
| --- | --- | --- | --- | --- |
| B-only / isolated | [44.95, 46.86] | [46.16, 48.48] | [16.00, 17.82] | [16.22, 18.97] |
| Full-no-store / isolated | [127.59, 134.83] | [121.97, 132.81] | [52.30, 63.96] | [57.00, 63.29] |
| B-only / 38+12 | [16.12, 18.11] | [15.94, 18.58] | [6.34, 7.84] | [5.97, 9.08] |
| Full-no-store / 38+12 | [45.94, 50.40] | [43.83, 49.83] | [21.62, 25.26] | [22.60, 27.45] |

Attached-but-disabled paired median changes ranged from -0.34% to +1.65% across
the eight session/case groups; its absolute medians and intervals are in the JSON.
The much larger active-SPE changes cannot be explained by attachment alone.
The measured overhead includes the entire active profiling intervention, including
sampling preparation and collection; it does not isolate a specific hardware
sampling-port, interrupt or AUX-memory mechanism.

### Period sensitivity

Per-cell median retained counts ranged from 136–190 (B-only, period1024) and
570–622 (full-no-store, period1024), versus 39–50 and 136–148 at period4096.
These are counts, not a measured probability of sampling every executed load.

B-only isolated raw TOT medians changed from 149.5 to 69 in session1 and from
164 to 68 in session2 when moving 1024 to4096. Full-no-store raw TOT medians
stayed at7 despite large timing changes. Thus neither raw sample median nor
record count can be treated as an additive explanation of unprofiled wall time.
The summary preserves per-PC offsets and per-record raw counters for follow-up;
no physical interpretation or refit is made here.

## Decision and bounded next step

- Retain the native control isolation and exact-IP/capture-integrity audit.
- Reject periods1024 and4096 as low-perturbation calibration sources for this
  M1/1T domain. The overhead is large, replicated and context-dependent.
- Do not subtract one universal overhead percentage or fit the frozen model to
  these instrumented times/latencies.
- If continuing SPE, first test much sparser sampling against the same paired
  no-perf/disabled controls, then evaluate sample sufficiency and distribution
  stability. Narrower native sampling windows may reduce preparation collection,
  but cannot be assumed to remove in-kernel sampling overhead.
- No new physical parameter or production adoption. Other widths, M values,
  placements and real-trace workloads have not been validated by this audit.
