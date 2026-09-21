# W13 per-panel sequence and tail transition

## Result

Completed two primary66-cell sessions and two30-cell late-tail sessions with
one unchanged native binary. Complete M12 panel times are almost flat from the
first scan; short-tail behavior is different. Isolated M1 tails decrease from
~324–333us at position2 to~280–282us at positions9/16. With38 M12 backgrounds,
M1 tails remain~636–663us late in the sequence; no universal fully-hot endpoint
or common transition length is established. M5 and M11 behave differently.

This supports modeling tail/context separately rather than assuming every
panel after the first receives identical ideal hot-B service. It does not prove
cache residency, physical warming duration, or explain the entire real p11 M13
penalty. No parameter fit, production model, kernel/default change or commit.

## Scope and protocol

Class E Lab-only native probe, runner/analyzer/tests/report/manifest. Production
JIT reused without modification. Arm-codex-internal root
`/home/zhangxu/codex/fused_cpp`, NUMA3, allowedCPU240–319/membind3. Native
foreground fixedCPU316. Backgrounds:0/16/38 independent real M12/1T W13 tasks
on CPU280–319 excluding316, first requested count (at38 CPU319 remains unused).
Backgrounds rotate four independent8MiB B copies after complete calls; no team
sharing or width change. Python command round trips are outside preparation and
execution; no post-preparation perf/Python handshake.

Foreground W13,K4096,N1024,BF16/SVE256,1T,N16 tile, full8MiB B owner stripe,
window_tiles0,R13=1. Each M12/tail kernel scans the SAME B stripe; consecutive
panels use distinct A rows unless explicitly marked A-reuse. W2 is unmeasured.
Constant BF16 inputs1/64 use existing expected fused W13 output0x3f3b. All live
rows/columns are checked, including tail physical layouts; output is poisoned
before every cell. Background output, active calls and exact CPU pinning are
checked; inactive backgrounds must havezero calls. This is a synthetic kernel
harness, not the real-weight full-MoE experiment.

Persistent allocations/four foreground copies. Before each cell the foreground
reads256MiB scrub, then backgrounds start with5ms native lead-in (same wait even
when isolated). No claim that every first load is guaranteed DRAM-resident.
No per-panel PMU enable/read or team barrier. Steady-clock timestamps enclose
each kernel call and the whole loop; arrays are preallocated. Background stop,
verification and serialization happen AFTER the enclosing timer.

B-preconditioning uses8 M12 calls before the target with separate A/C buffers.
`warm=8` scans target B; `warm=-8` executes identical kernels on separate B.
This controls much of the background-age and compute-prepass intervention;
prepass median durations differ by only about-0.12% to+0.15% across matched
conditions/sessions. Neither name guarantees a specific cache level. The
baseline `warm=0` performs no prepass. It must not be pooled with the prepass
control as an identical background-age regime.

Primary grid66:
- full M12 sequences M12/24/36/48/96/192 with0/16/38 backgrounds;
- M1/5/11 and M13/17/23/M25/29/35 with0/38 backgrounds;
- target/disjoint B prepasses for M1/5/11/13/17/192 with0/38 backgrounds;
- matched uninstrumented full-loop controls M13/M192 with0/38 backgrounds;
- M192 same-A-address controls with0/38 backgrounds.

Primary sessions592001/592002:5 warmups+31 measured randomized paired rounds,
same copy per round. Complete66-cell numerical smoke precedes them.
After reviewing early-tail results, a bounded descriptive extension tests tails
M1/5/11 at positions2/3/5/9/16 (M=12*(position-1)+tail),0/38 backgrounds.
It retains positions2/3 as bridges; no model/threshold fitting or unseen-holdout
claim. Extension sessions592011/592012 use the same binary and31/5 protocol.
Original runner snapshot/results remain intact; extension runner lives separately.

Across four formal sessions:6,912 cells, including960 warmup and5,952 measured.
All native and reader checks pass; no active run remains.

## Complete M12 blocks

M192, session2 median panel times inus:

| Panel position | Isolated | 16 backgrounds | 38 backgrounds |
| --- | ---: | ---: | ---: |
| 1 | 1231.39 | 1238.49 | 1265.62 |
| 2 | 1232.74 | 1235.21 | 1261.31 |
| 3 | 1235.13 | 1235.47 | 1257.59 |
| 4 | 1231.55 | 1235.37 | 1254.00 |
| 8 | 1233.84 | 1235.37 | 1251.29 |
| 16 | 1234.83 | 1236.21 | 1253.34 |

Isolated blocks stay near1.23ms;38-background early decline is~1%, not a large
multi-panel time collapse. S1 repeats the pattern. First M12 time is consistent
across total M12/24/36/48/96/192: S2 isolated1231.07–1233.78us and38-background
1265.55–1266.58us. This supports treating the first complete block similarly
across these measured shapes, but not naming its source definitively cold DDR.

Time flatness does NOT establish absence of cache warming: M12 compute/load
balance can hide changes in memory service. No per-panel cache counters were
collected in this low-perturbation timeline.

## Tail position curves

Late-tail extension, session2 median tail time inus; each row uses the SAME tail
kernel at every position. A position9 M1 tail corresponds to total M97.

| Tail rows / background | Position2 | Position3 | Position5 | Position9 | Position16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 / isolated | 323.93 | 320.60 | 315.21 | 281.81 | 280.33 |
| M1 /38 | 681.52 | 664.21 | 665.61 | 662.69 | 636.47 |
| M5 / isolated | 607.73 | 606.62 | 604.24 | 604.56 | 608.65 |
| M5 /38 | 766.84 | 744.00 | 743.88 | 733.59 | 718.47 |
| M11 / isolated | 1233.34 | 1234.26 | 1234.16 | 1234.88 | 1235.14 |
| M11 /38 | 1260.35 | 1259.53 | 1261.20 | 1258.59 | 1253.73 |

S1 M1 tails are332.83/322.58/315.57/282.41/280.62us isolated, and
679.29/664.36/663.87/650.58/637.57us with38 backgrounds. Thus isolated late
M1 timing repeats closely, but pressured position9 varies across sessions and
still decreases at16. Do not declare a unique transition boundary from these
sparse sampled positions. Position also changes elapsed background time.

The primary-grid standalone cold-ish M1 control is304.52us isolated inS2,
LESS than its early post-M12 tail326.01us. Therefore a simple monotone blend
from a standalone cold M1 time to a hot time is not a complete explanation.
Address/history/kernel-transition effects remain possible.

M5 tails are already approximately stable after the first full panel when
isolated. M11 tails resemble full M12 time. Tail row count/kernel family changes
how much memory-state variation is visible; one transition term cannot be
assumed transferable to every remainder.

## Target-B versus disjoint-B prepass controls

Primary session2 medians, us. Both alternatives first execute8 M12 prepasses
with separate A/C; only the B address is target versus disjoint.

| Target | Background | Disjoint-B prepass | Target-B prepass |
| --- | ---: | ---: | ---: |
| Standalone M1 | 0 | 307.97 | 282.72 |
| Standalone M1 | 38 | 706.62 | 649.44 |
| M13 tail M1 | 0 | 362.79 | 283.75 |
| M13 tail M1 | 38 | 674.62 | 646.94 |
| Standalone M5 | 0 | 595.34 | 602.91 |
| Standalone M5 | 38 | 773.51 | 727.46 |

For standalone M1, matched-round median changes are-23.14us isolated
(95% IID paired bootstrap interval[-25.74,-21.70]) and-54.16us with38
([-61.16,-43.68]); not ratios of the displayed medians. M5 isolated instead
slightly worsens: paired+7.11us. All contrasts, including contrary results,
remain in JSON. Preconditioning does not benefit all kernels uniformly.

Target-B-preconditioned M1 time~283us agrees with isolated late-tail~280–282us,
supporting a B-history-sensitive endpoint. This is evidence of an operational
preconditioning effect, not direct proof of complete B cache residency. A/B
address layout, replacement and prefetch behavior are not separately measured.

## Perturbation and remaining interpretation boundaries

Matched instrumented versus uninstrumented total-time changes across M13/M192,
background0/38, both sessions range-0.226% to+0.354%. Paired median absolute
changes range-3.63 to+13.05us. Every per-panel sum is bounded by its enclosing
whole-loop timer. No timestamp overhead is subtracted as a cache correction.
All numeric checks pass; four focused local tests cover grids, tails, controls,
paired arithmetic and ledger validation. Ruff/diff checks pass.

A-reuse M192 total changes from19754.28→19741.82us isolated and
20090.26→20012.55us with38 backgrounds inS2. It is a full-M12 control only;
it does not identify A's contribution to short-tail position effects.

These data do not reproduce the entire real-M13 p11 penalty. Synthetic M13
with38 M12 backgrounds has a first block~1.27ms and tail~0.69ms, while the real
context experiment measured W13 near2.9ms. Different input/packing/runtime and
background histories prevent numerical substitution. The measured isolated
M1-tail transition itself is only tens of microseconds, not an explanation for
a~1ms missing joint increment. Preserve the real-context failure separately.

## Artifacts and reproduction

New files: `panel_transition_native.cpp`, `bench_panel_transition.py`,
`analyze_panel_transition.py`, `tests/test_moe_panel_transition.py`. Rollback is
limited to these Lab files/report/manifest; no supported API or global build
change. Existing user work preserved. Foreground/background JIT unchanged SHA256
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
Native binary SHA256
`28bb9b703a7e95be781a34c4a6bc6546d1238c5e6db360df44c86a314dd16de5`.
GCC13.2.0,C++17/O3/pthread,armv8.2-a+bf16+sve,SVE256. Ordinary allocations under
the existing page policy, no verified HugeTLB/residency claim. Source basis is
the current dirty workspace; no clean-commit build claim.

Local/remote root:`tmp/panel_transition_20260909/`. Primary and tail_extension
subdirectories retain raw JSONL, stderr, reports and separate runner snapshots
remotely; original binary/source retained remotely. No old artifacts overwritten.
`panel_timing.png`/`.svg` and `plot.py` provide the reviewed static figure.
Matplotlib3.11.1 was loaded from an existing local cache; no project dependency
was installed/changed. The figure uses session2 medians with P10–P90 spread,
zero-based axes and explicit panel/background labels. It does not fit curves.

Remote project-root build:

```sh
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm \
  -I3rdparty/xbyak_aarch64 -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/panel_transition_20260909/panel_transition_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/panel_transition_20260909/panel_transition_native
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/panel_transition_20260909/bench_panel_transition.py \
  --binary tmp/panel_transition_20260909/panel_transition_native \
  --output tmp/panel_transition_20260909/session1.jsonl --seed 592001
```

S2 uses592002 and session2 output. Smoke used592000, `--rounds 1 --warmup 0`. Extension uses the preserved
`tail_extension/bench_panel_transition.py`, same binary, `--tail-extension`,
seeds592011/592012, and new tail_extension/session1/2 outputs.

Local analysis, exclusive output paths (choose fresh names for reruns):

```sh
.venv/bin/pytest -q tests/test_moe_panel_transition.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_panel_transition.py \
  --sessions tmp/panel_transition_20260909/session1.jsonl tmp/panel_transition_20260909/session2.jsonl \
  --output tmp/panel_transition_20260909/report.json
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_panel_transition.py \
  --sessions tmp/panel_transition_20260909/tail_extension/session1.jsonl tmp/panel_transition_20260909/tail_extension/session2.jsonl \
  --output tmp/panel_transition_20260909/tail_extension/report.json
```

The requested collection is complete. Keep first/full-panel and remainder-kernel
behavior separate; do not install a universal cold→transition→hot formula from
these timings alone.
