# Three findings that needed no machine (2026-09-21)

Both machines went unreachable after the C9g port (`Arm-codex` rejects our SSH key, C9g times
out). This is what the open questions gave up to a static read of the code and an offline
analysis of data already on the host. **Nothing here was built, run or remeasured**, and none of
it changes production, the model, the tables or the planner. Lab records:
`tmp/w8a8_vector_length_20260921/`, `tmp/panel_boundary_20260921/`, `tmp/v15_candidate_20260921/`.

---

## 1. C9g's W8A8 failure is a vector-length assumption, not a race

`second_machine_c9g_20260921.md` recorded the W8A8 plan-v2 path as non-deterministic at every
team width above one, clean at width one, clean under ASAN, and concluded "the primary defect is
a race", pointing at `Workspace::barrier`. Both halves of that are wrong.

`TeamBarrier::wait` (`csrc/moe/arm/i8mm_w8a8/kernels.cpp:100`) is a correct generation barrier,
and every thread of a lane executes the same number of waits. What the team width actually
selects is a **microkernel family**: `packed` at `kernels.cpp:329` is exempted only when
`width >= 2`, so width 1 runs the NEON `i8gemm_k_nld*` kernels and width 2 or 4 runs
`i8gemm_k_hybrid` / `narrow` / `narrow2`.

Those SVE kernels step N by `svcntb()/2` columns but their store macro `HSTORE_PAIR`
(`refs/i8gemm/lib/i8gemm_hybrid.S:74`) writes a hard-coded 64 bytes per row - four 16-byte
`st1w`s at 16-byte strides with a 16-byte `ext` between them. At a 256-bit vector length that is
exactly the 16-column tile. **At 128 bits the N step is 8 columns, the `ext` is the identity, and
the kernel writes 8 columns of duplicates past the tile it was asked to compute.** C9g is a
128-bit machine; `Arm-codex` is 256-bit, the only width these kernels are correct at.

Everything observed follows: the leak crosses into the neighbouring thread's N stripe, so results
depend on write order; it corrupts 8 of the W13 stage's `2F` columns, which feed 8 of W2's `F`
accumulation terms, so all 64 output columns move by ~3e-4; four stripes (width 4) are worse than
two; and the last thread's last tile writes 32 bytes past the end of `w13_acc`, which is the
`free(): chunks in smallbin corrupted`. ASAN saw none of it because the overflowing store is in
hand-written assembly, which it does not instrument - **a clean ASAN run is not evidence against
an assembly out-of-bounds write**, and treating it as such is what sent the first diagnosis to
the barrier.

`refs/i8gemm/lib/i8gemm_msplit_k.S` carries the identical macro, and both W8A8 benchmarks call
the hybrid kernel directly, so they would report wrong numbers, not just wrong timings, on any
non-256-bit machine.

The minimal fix looked like one condition - take the packed path whenever `svcntb() != 32` -
and was **not applied**, because it could not be compiled or tested while both machines were
unreachable. Three falsifiable predictions were pre-registered in the lab record, the cheapest
being that a `k > 1024, rows > 16` shape at width 2 on C9g must already be correct today.

### Measured 2026-09-22: the vector length is confirmed, this mechanism is not the whole defect

Both machines came back and the predictions were run, one process per case because heap
corruption poisons every later observation in the same process
(`tmp/w8a8_vector_length_20260921/probe.py`):

| H=F | tokens | width | rows | family | `Arm-codex` 256-bit | C9g 128-bit |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 48 | 2 | 24 | hybrid | repeatable | `free(): chunks in smallbin corrupted` |
| 64 | 48 | 4 | 24 | hybrid | repeatable | `free(): chunks in smallbin corrupted` |
| 256 | 96 | 2 | 48 | hybrid | repeatable | `free(): chunks in smallbin corrupted` |
| 2048 | 96 | 2 | 48 | **packed** | repeatable | `corrupted size vs. prev_size` |
| 2048 | 96 | 4 | 48 | **packed** | repeatable | `corrupted size vs. prev_size` |
| 2048 | 192 | 4 | 96 | **packed** | repeatable | `free(): chunks in smallbin corrupted` |

**Six of six correct at 256 bits, six of six corrupt at 128 bits.** That the defect follows the
vector length rather than the thread count is now measured across two microkernel families and
two machines, which is stronger evidence than the static read gave.

But the last three shapes take the **packed NEON path** and never execute `HSTORE_PAIR`, and they
corrupt the heap all the same. So `i8gemm_k_hybrid`'s fixed 16-column store is a real
vector-length bug and it explains the hybrid rows, but it is **not the whole defect**: at least
one more vector-length assumption sits in the packed path (`i8gemm_k_nld*`,
`i8_pack_A_neon_m8_asm`, `pack_m12`/`prepare_packed_a`, or the packed-B geometry, whose
documented layout in `refs/i8gemm/lib/i8gemm.h` is an 8-column N block while the wrapper uses
`svcntb()/2`).

**The proposed fix is therefore wrong and must not be applied**: forcing the packed path below
256 bits moves the work onto a path that is also broken there.

### Located: it is one mechanism, and my file was wrong

`setup.py:688` picks `i8gemm_backend = "sve" if target_has_sve else "neon"`, so an SVE build
compiles `i8gemm_sve.S` and **never compiles `i8gemm_k.S`**. `i8gemm_k.S` is the NEON fallback,
and it is the file I read when I called the packed family "VL-generic NEON". The packed
kernels `i8gemm_k_nld_m12` and `i8gemm_k_nld{,1,2,4}` are defined in `i8gemm_sve.S`, which
carries the same construction as `i8gemm_hybrid.S`: `ptrue p2.s, vl4` (:811), an N step of
`cntb; lsr #1` = VL/2 columns (:816), and a `DEINT_QUADS` macro (:530) that writes four 16-byte
`st1w`s with `ext ..., #16` between them - 64 bytes, 16 int32 columns, per row. Its own comments
say "8x2VL block" and "the 8x16 tile", which agree only at 256 bits.

**Every SVE i8gemm microkernel hard-codes a 16-column deinterleave store while deriving its N
step from the runtime vector length.** Below 256 bits each store lays down 8 valid columns and 8
duplicates past the tile. `i8gemm_msplit_k.S` shares the macro, and `setup.py` hands these
kernels to every SVE machine, not only 256-bit ones.

This predicts that width is irrelevant, since width only chooses between two equally affected
families - and width 1 fails too, which supersedes the 2026-09-21 note that a single-thread lane
was clean (that run shared a process; a corrupted chunk is only detected when it is freed):

| H=F | tokens | width | C9g 128-bit | `Arm-codex` 256-bit |
| --- | --- | --- | --- | --- |
| 64 | 48 | 1 | `free(): chunks in smallbin corrupted` | repeatable, max abs err 2.9e-06 |
| 2048 | 96 | 1 | `corrupted size vs. prev_size` | repeatable, max abs err 1.0e-03 |
| 2048 | 96 | 2 | `corrupted size vs. prev_size` | repeatable, max abs err 1.0e-03 |

With the six-shape run above: **10 of 10 correct at 256 bits, 10 of 10 corrupt at 128 bits**,
over widths 1, 2 and 4 and both families.

### Confirmed on the kernel alone, with no MoE code involved

`i8gemm_vl_unit.cpp` links only `i8gemm_sve.S` and `i8gemm_hybrid.S` - no torch, no threads, no
MoE wrapper, no windows. It asks each kernel for exactly `N = svcntb()/2` columns, which is the
tile width the kernel's own N loop computes, and gives C a `ldc` of 64 so an overrun shows up as
nonzero columns instead of corrupting the heap:

```
SVE vector length 128 bits, n_tile = VL/2 = 8 columns
asking each kernel for N = 8 columns, with ldc = 64 so an overrun is visible

i8gemm_k_hybrid : highest nonzero column = 15, expected at most 7 -> WROTE PAST ITS N
i8gemm_k_nld_m12: highest nonzero column = 15, expected at most 7 -> WROTE PAST ITS N
```

Both families write columns 0-15 when asked for 8: exactly the fixed 64-byte, 16-column store,
eight columns past the tile. **The defect is in the i8gemm library, not in how the MoE path calls
it.** The kernels are internally inconsistent at any vector length other than 256 bits: no caller
contract can repair a loop that advances N by 8 while each iteration writes 16.

The library intends to support this machine. `refs/i8gemm/results/reports/HANDOFF.md:28` states
that V3 is `VL=128bit` so `n_tile = VL_bytes/2 = 8`, and warns against carrying over the
"V3 VL=256 / n_tile=16" assumption - which is precisely the assumption still frozen into the
store macros.

Three repairs are open, from narrowest to widest: refuse W8A8 unless `svcntb() == 32` (one
condition, validatable on both machines today, and it narrows the execution contract in
`docs/public_contracts.md`, which currently names only "Linux AArch64 SVE+i8mm"); build the
VL-independent NEON `i8gemm_k.S` when the vector length is not 256, which restores function but
changes the global build for a machine class; or make the store macros VL-generic inside the
vendored library. None is applied. Evidence and probes: `tmp/w8a8_vector_length_20260921/`.

---

## 2. The C9g panel residual is a missing per-panel term

The C9g calibration left two residual structures. Reading the residual matrix rather than the
summary shows the second one was described wrongly: the sign change is not a local artefact
"just past a boundary" but a **transition at routes 12 -> 16 that holds at every width** and
persists out to routes 192, and it is present at 1T and 2T, so it is not the "M < threads"
region either.

Normalising the measurements says why. Cost per `m12_effective_rows` row is a constant 41 us
**only where routes is a multiple of 12** (41.75, 41.04, 40.71, 40.80, 41.03 at routes
12/24/48/96/192) and jumps to 49.22 at routes 16. The cost is additive in M12 panels, and a
panel is not twelve rows: a fit gives 495 us per full panel against a marginal 23.8 us per row,
so about 210 us of every panel does not scale with its rows - the kernel re-streams the expert's
weights once per M panel (`fused_moe_bf16_tiled.cpp:1243` walks all of N and K inside the
12-row loop). Twelve rows amortise 210 us to `210/12 + 23.8 = 41 us/row`, exactly the rate the
calibration fits on large M. A four-row tail panel pays the same 210 us for a third of the rows.

The parameter-free consequence `cost(16) = cost(4) + cost(24) - cost(12)` lands within 0.5% at
1T and 0.2% at 2T, where the calibrated model is off by -9.4% and -2.8%; TP2 is the same with
larger amplitude. Where the identity and the model disagree they bracket the measurement, and
that band (4-16 threads) is where the per-thread B stripe straddles the 2 MiB L2.

The structure already exists twice in the repository and is missing in the one place that
matters: `phase_model.py:484` `_formula_iso` **is** this identity, and
`tiso_roofline.py:143` charges `b_read_bytes` per panel - but that module has no consumer
outside its own test. `analytic_model.py`, the path calibrated per machine and used by the
planner, carries a `panel_count` that feeds only the frontend and L1 terms and treats every
panel after the first as steady, i.e. as if its weights were resident.

`m12_effective_rows` is **not** the defect; its tail table reproduces the kernel's dispatch
exactly. Its one real mismatch is the modulus - the model applies the tail table to
`routes % 12` while the m8 dispatch applies it to `routes % 8` - which diverges at routes 9, 10,
13, 14, 17, 18, none of which are profile points.

---

## 3. Why v12-v14 could not fix ranking, and a candidate that survives a strict holdout

Decomposing `log(measured/predicted)` over 546 measured plans in 75 layers:

| component | sd | share of variance | can it reorder plans? |
| --- | --- | --- | --- |
| per-layer level | 1.5% | 15.3% | no |
| within-layer, plan to plan | 3.5% | **84.7%** | yes |

Five sixths of the spread is between plans of the same layer. A correction that is near-constant
within a layer - which a per-domain footprint term largely is - can only move the level. That is
precisely what E14 measured (1.090 -> 1.061 level, ranking not better), and it means **a
candidate should be scored on its within-layer variation before it is built**, which v12, v13
and v14 all skipped.

Centring inside each layer, the ranking-relevant residual correlates with `window_credit`
(r = +0.672), `share_2t` (+0.556), `loading_share` (-0.527) and `utilization` (+0.487); the width
features carry nothing. The sign says the model **over-credits windows differentially**: within
a layer, the plan given more window credit is the plan that runs slower than predicted. Both
leading features are computed by the model itself with no measurement.

`log ratio = 0.61*window_credit - 2.94*loading_share` explains 65% of the within-layer variance.
In-sample fit is what v12-v14 were also good at, so it decides nothing; out of sample:

- **leave-one-workload-out** (75 folds): Spearman median 0.817 -> 0.917, mean regret@1
  1.65% -> 0.27%, better in 35 layers and worse in 8;
- **leave-one-set-out**, no layer of the tested set anywhere in training: E5 0.642 -> 0.875 and
  5.72% -> 0.07% regret; E12 0.943 -> 1.000; E6b 0.705 -> 0.755; E9 and E3 unchanged; E11 the
  only slight loss (0.17% -> 0.21%), on plans that differ by 0.44%, inside the noise.

The `window_credit` coefficient is stable over every fold of both schemes (+0.41..+0.74).
Almost all of the gain is E5 because E5 is almost all of the headroom - E3, E6b and E9 already
select at 0.00-0.03% regret - so a plain E3+E5 -> rest split is a wash, and that is a statement
about the corpus, not about the term.

This is a bounded residual on quantities the model already computes, not a mechanism, and it is
**not adopted**: E14 is the precedent that a correction which improves an already-measured
corpus must still be frozen and shown on layers nobody has run. Unlike v14 it passes
leave-one-set-out, which v14 was never subjected to - a better prior, not a result. An E15
design with gates weighted to ranking and selection is drafted in the lab record.

---

---

## 4. The i8gemm vector-length defect is fixed, and verified on both vector lengths

The repair is in the store macros themselves, chosen at assembly time. `setup.py` already
forwards the extension's `define_macros` to native sources (`setup.py:186`) and `.S` files run
through cpp, so `FUSED_CPP_MOE_SVE_VECTOR_BITS` - which the build detects and
`csrc/moe/common/backend.cpp:281` verifies against the running thread - is available to the
assembler with no build change.

Each of `DEINT_QUADS` (`i8gemm_sve.S`) and `HSTORE_PAIR` (`i8gemm_hybrid.S`, both halves) now
emits **two 128-bit quads per 128 bits of vector length** instead of four unconditionally, which
is exactly the `n_tile = svcntb()/2` columns its N loop steps:

```c
#ifndef FUSED_CPP_MOE_SVE_VECTOR_BITS
#define FUSED_CPP_MOE_SVE_VECTOR_BITS 256
#endif
#if FUSED_CPP_MOE_SVE_VECTOR_BITS != 128 && FUSED_CPP_MOE_SVE_VECTOR_BITS != 256
#error "i8gemm SVE kernels support a 128- or 256-bit vector length; the 4-quad epilogue
        and its scale/bias register budget have not been generalised further"
#endif
#define I8GEMM_SVE_QUAD_PAIRS (FUSED_CPP_MOE_SVE_VECTOR_BITS / 128)
```

and the second quad pair, its `ext` rotations and its `fadd` bias terms sit behind
`#if I8GEMM_SVE_QUAD_PAIRS >= 2`. 43 changed lines across the two files.

512 bits is refused at assembly time rather than generalised: the `bias` mode holds four quads
of bias in z8-z11 and `LOAD_SCALED_TILE_AUX` loads four scale quads, so eight would need the
register budget reworked, and there is no machine here to validate it on. A build error is the
right answer for the direction that otherwise leaves half of every tile unwritten.

### Verification

| check | result |
| --- | --- |
| kernel-only unit check at 128 bits | `i8gemm_k_hybrid` and `i8gemm_k_nld_m12` now write columns 0-7 when asked for 8 (was 0-15) |
| **256-bit output unchanged** | disassembly of both files at `-DFUSED_CPP_MOE_SVE_VECTOR_BITS=256` is **identical** to the original, so this is a no-op on `Arm-codex` |
| 512-bit build | fails with the `#error`, as intended |
| `test_moe_w8a8.py` on C9g | **7 passed** (previously corrupted the heap) |
| four shapes against the dynamic-quantisation reference on C9g | repeatable, max abs err 2.861e-06 at H=64 and 1.007e-03 at H=2048 - **the same values `Arm-codex` produces** |
| MoE suites on C9g | 298 passed |
| kernel suite on C9g | 80 passed, 1 skipped |

The two machines now agree numerically on the same shapes, which is a stronger statement than
"no longer crashes".

### Not fixed, and deliberately so

`DEINT_SCALED_QUADS` and `LOAD_SCALED_TILE_AUX` in `i8gemm_sve.S` carry the same defect - the
latter loads four scale and four bias quads for what is only `n_tile` columns, so below 256 bits
it reads past those arrays - and `i8gemm_msplit_k.S` shares the store macro. None of them is
reached by the W8A8 MoE path (`i8gemm_k_scaled_*` and `i8gemm_msplit_dispatch` have no caller
here), so there is no way to validate a change to them from this repository. They stay broken and
recorded rather than changed blind.

### Delivery is unresolved

`refs/` is in `.gitignore:15`, so **nothing under it can be committed**; the patch currently
exists only in the working trees of this host and C9g, and as
`tmp/w8a8_vector_length_20260921/i8gemm_vl.patch`. The content is recorded above so the knowledge
survives regardless. Three ways to version it, for the user to choose: a `.gitignore` exception
for the two files, a patch applied by the build, or our own copy of the two `.S` files under
`csrc/moe/arm/i8mm_w8a8/` that `setup.py` prefers.

## What is now waiting on a machine

| item | first action when access returns |
| --- | --- |
| W8A8 | run the three predictions, then apply the `svcntb() != 32` gate and rerun `test_moe_w8a8` at widths 1/2/4 |
| W8A8 on `Arm-codex` | confirm the 64/64/48 width-2 case passes there (the vector-length account requires it) |
| panel term | profile routes 13, 17, 20, 36 in isolation; they separate the panel identity from the current per-row rate by more than 10% at 1T |
| tail modulus | routes 9, 10, 13, 14, 17, 18 separate `routes % 12` from `routes % 8` |
| E15 | six fresh layers, gates frozen on ranking and selection |
| C9g contention | port the event-model probe curves; unchanged by anything above |
