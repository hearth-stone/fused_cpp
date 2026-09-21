# Exact-M phase memory-supply contrast — 2026-09-07

## Decision

The controlled contrast is complete. The same streaming background changes
victim memory-event counts for both small and large M, but the elapsed-time
penalty is strongly M/width dependent. This is consistent with a memory supply
change whose exposure depends on reuse, prefetch, startup amortization and
compute/transfer overlap. It does **not** prove an equal victim memory-service
latency increase in both shapes, or identify compute overlap as the sole cause.

Do not fit a new cost term from this experiment. First investigate the existing
cold/steady transfer and overlap treatment; do not apply a uniform contention
multiplier to the entire GEMM. No calibration, planner, pruning, production
kernel, extension or default was changed.

## Measured time

Each row compares isolated with the same eight-reader background, at fixed M,
width and stage. Times are medians in ms. Percentages are medians of within-round
paired slowdown, so they need not equal the ratio of the two time medians.

| M | Width | Stage | Session 1 isolated → background | Paired slowdown | Session 2 isolated → background | Paired slowdown |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | W13 | 0.3037 → 0.3466 | +13.91% | 0.3042 → 0.3463 | +13.65% |
| 1 | 1 | W2 | 0.1527 → 0.1778 | +14.92% | 0.1541 → 0.1810 | +17.38% |
| 1 | 16 | W13 | 0.0764 → 0.0777 | +2.96% | 0.0754 → 0.0784 | +3.39% |
| 1 | 16 | W2 | 0.0359 → 0.0366 | +1.39% | 0.0361 → 0.0372 | +1.26% |
| 1341 | 1 | W13 | 134.6541 → 134.8624 | +0.16% | 134.6604 → 134.9078 | +0.19% |
| 1341 | 1 | W2 | 64.4738 → 64.5449 | +0.16% | 64.4943 → 64.5468 | +0.13% |
| 1341 | 16 | W13 | 8.4512 → 8.4378 | −0.14% | 8.4591 → 8.4549 | −0.07% |
| 1341 | 16 | W2 | 4.1045 → 4.0910 | −0.09% | 4.0957 → 4.0846 | −0.20% |

Small-M/1T slowdown is repeatable: bootstrap median 95% intervals are
12.64–14.88% and 12.33–15.09% for W13; 12.62–16.73% and 13.02–19.18% for W2.
M1/16T W2 is less conclusive (session 2 interval −0.46–6.93%). Treat sub-percent
large-M differences as near-flat, not an optimization opportunity. Full
distributions, mean/std, P90/P99 and paired intervals are in the summary JSON.

## What the counters add

Counts below sum the victim workers and then take the per-cell median. They
are event counts, not measured bytes or victim latency.

| Victim | LLC read-miss, isolated → background, S1 / S2 | Backend-stall change, S1 / S2 |
|---|---|---|
| M1/1T W13 | 1,613 → 20,267 / 1,704 → 17,240 | +22.7% / +21.6% |
| M1/1T W2 | 1,386 → 13,588 / 1,425 → 15,398 | +26.6% / +28.3% |
| M1341/1T W13 | 187,809 → 280,946 / 189,550 → 277,738 | +0.18% / +0.22% |
| M1341/1T W2 | 65,982 → 84,585 / 65,498 → 86,113 | +0.12% / +0.10% |
| M1341/16T W13 | 253,587 → 413,857 / 253,857 → 415,491 | −0.41% / −0.11% |
| M1341/16T W2 | 118,588 → 130,483 / 117,383 → 128,261 | −0.26% / −0.54% |

Retired instruction counts are identical between background conditions in
every shape/stage. For M1/1T, L2 refill counts are also nearly unchanged while
LL read-miss and backend stalls increase. This points to a change below L2 or
in prefetch/cache service, rather than additional useful computation.

For large M, more LL read-miss events do not translate into a comparable
increase in backend stalls or elapsed time. Large M does much more computation
and reuses B across M12 panels; compare event density as well as absolute counts.
For example, background W13/16T has about 1.5 LL read-miss events per thousand
instructions at M1341, versus about 146–150 at M1. This is **not** a cache-hit
ratio: the event denominators and prefetch accounting are different.

DDRC occupancy/read-command ratios rise for large-M W13/16T from about 28 to
37 in SCCL25 and 27–28 to 35 in SCCL27. W2/16T rises from about 32 to 38 and
30–31 to 36. Background read-flux estimates across both domains are roughly
66–81 GB/s for the long/1T cells, using the existing probe's 32-byte flux unit.
These establish changed aggregate conditions, not victim-specific queue latency
and not saturation of the machine. Occupancy ratios are not converted to ns.

## Measurement limitations that constrain the conclusion

1. **Uncore gates are wide.** Opening/closing 36 device groups takes about
   1.09–1.17 ms in empty controls. Gate/kernel ratios are roughly 4.3–8.3 for
   M1/1T, 15–32 for M1/16T, 1.13–1.29 for M1341/16T and 1.01–1.02 for
   M1341/1T. Small-M uncore data describe the condition around the phase, not
   its exact traffic. Each device's own enabled interval is used for flux
   rates; empty totals are not blindly subtracted.
2. Core counters are worker-local, bracket the kernel directly and exclude
   kernel/hypervisor execution. `stall_backend` is generic: this machine does
   not expose a memory-specific stall event. LLC read-miss does not by itself
   measure per-request latency or establish the source of every weight read.
   Scrub + rotation do not prove that all in-phase accesses bypass caches.
3. This is a **synthetic packed phase-only harness**, not real-trace execution.
   It reuses production exact-M JIT generation, but has its own persistent
   output allocation and synthetic inputs; gather/merge are outside scope.
   W2 input is independent, not the W13 result. The eight-reader background is
   one fixed pressure point, not the real workload or a peak-bandwidth test.
4. The roughly 14–17% small-M penalty does not reproduce the much larger
   full-workload workspace penalty. That residual remains open. This experiment
   cannot decide whether every remaining small-M cost is bandwidth-related.
5. No time-resolved separation of the initial cold panel and subsequent M12
   panels was collected. Additional misses with unchanged total time support
   hidden/amortized exposure, but do not quantify hidden memory-service time.

HiSilicon uncore devices have independent counters, not process attribution;
see the [Linux PMU documentation](https://kernel.org/doc/html/v5.18/admin-guide/perf/hisi-pmu.html).
The [Arm LL-cache event guide](https://learn.arm.com/learning-paths/servers-and-cloud-computing/triggering-pmu-events/llcache/)
provides event context; no unverified implementation-specific prefetch rule is
assumed here.

## Protocol and provenance

- Class E, Lab-only native diagnostic; L1 protocol tests, target build/smoke and
  two-session performance evidence. Rollback is limited to the new harness,
  Python driver/analyzer/test and manifest/report entry.
- Host `Arm-codex-internal`, remote root `/home/zhangxu/codex/fused_cpp`;
  Linux `5.10.0-247.0.0.146.oe2203sp4.aarch64`, GCC 13.2.0, C++17 `-O3`.
- NUMA3 memory; controller CPU240; victim CPU304 or304–319; background
  CPU288–295. Victim/background cores share LLC domain SCCL25. L3C and DDRC
  collected across SCCL25 and SCCL27 because NUMA3 memory traffic reaches both.
- BF16 inputs/weights fixed to 1/64. W13 K4096,N1024 (8 MiB B); W2 K512,N4096
  (4 MiB B). SVE256 N tile16, full owner stripes, one window/stage. Width1
  owns64/256 tiles, width16 owns4/16 tiles for W13/W2 respectively; R13=R2=1.
  W13 output is packed BF16; W2 output is FP32 at route positions `7*r`.
- Max1344 padded input rows; persistent 12288×4096 FP32 output (~192 MiB),
  allocated/touched before measurements. Ordinary aligned allocations, no
  explicit HugeTLB; system THP policy `always`, PMD size2 MiB. Actual per-buffer
  THP coverage was not measured. This is not the production workspace object.
- Four B copies,256 MiB scrub by all16 victim workers before every cell;
  background8×32 MiB touched-line streams with5 ms startup before GO.
- Same process and allocations for all cells in one session; fresh process in
  session2. Seeds9107/19107;5 warmup+31 recorded rounds; randomized20-cell order
  (16 real cells+4 empty controls), copy index round modulo4.
- Core events: cycles, instructions, L2 refill, LL read-miss, backend stall.
 40 L3C+48 DDRC counters in36 groups. Count reset and enabled/running deltas are
  handled separately. Every core and uncore running ratio was1.0.
- Each session completed720 cells:576 real numerical checks and144 empty
  controls. Across both sessions:1152 real checks,288 controls;992 real cells
  and248 controls remain after warmup exclusion. An additional16-cell
  correctness smoke passed before PMU collection.
- Initial smoke failed due to harness stripe-address handling; corrected before
  measurements. `correctness.jsonl` retains the failed run remotely;
  `correctness_v2.jsonl` is the successful run. No production kernel fix.
- Constant-input logical outputs and poisoned-output coverage pass (W2 exact
  binary0.125, W13 BF16(SiLU(1))=0x3f3b). This is a harness coverage/reference
  check, not a new random-input production-kernel equivalence claim.
- Repository base `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing dirty
  user work and the new Lab files. Actual source snapshot retained remotely in
  `tmp/phase_memory_supply_20260907/build_sources.tar.gz`.
- Binary SHA256 `4e00f896653785e36d20bf398b4a864a520f3f1bc9a90b1d66dea6a060303ffc`.
  JIT source SHA256 `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
  Frozen extension remains `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`;
  the standalone executable does not load that extension.

## Reproduction and retained evidence

Raw `session1.jsonl`, `session2.jsonl`, `pmu_smoke.jsonl`, `correctness_v2.jsonl`
are retained locally and remotely under `tmp/phase_memory_supply_20260907/`.
The local `summary.json` holds all eight contrasts for each session.

Remote build from repository root:

```sh
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm -I3rdparty/xbyak_aarch64 \
  -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/phase_memory_supply_20260907/phase_supply_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/phase_memory_supply_20260907/phase_supply_native
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/phase_memory_supply_20260907/bench_phase_supply.py \
  --binary tmp/phase_memory_supply_20260907/phase_supply_native \
  --output tmp/phase_memory_supply_20260907/session1.jsonl --seed 9107
```

Use a fresh output path on replay (exclusive creation prevents overwrite).
Session2 uses seed19107. Correctness-only smoke adds `--no-pmu --warmup 0
--rounds 1`; PMU smoke omits `--no-pmu`.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_phase_supply.py \
  tmp/phase_memory_supply_20260907/session1.jsonl \
  tmp/phase_memory_supply_20260907/session2.jsonl \
  --output tmp/phase_memory_supply_20260907/summary.json
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_phase_supply.py
```

Focused tests:6 passed; Ruff check/format and `git diff --check` passed.
No production suite rerun because no production code changed. No commit made.

Next bounded discriminating experiment, if pursued: separate first-panel versus
steady-panel victim counters under the same background, with a smaller/faster
uncore gate or separately qualified supply window. Keep new workspace ordering
counterexamples as holdout; do not equate generic stalls or aggregate queue
occupancy with a calibrated victim memory-service time.
