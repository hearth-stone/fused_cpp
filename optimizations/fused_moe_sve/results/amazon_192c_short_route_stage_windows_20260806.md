# Amazon 192-core short-route packed-B stage windows

Date: 2026-08-06

Status: measured and validated end to end; the production policy band is **not**
landed yet, so the default policy still leaves `M < 49` on the inherited window.

## Question

The calibrated default policy `amazon_c5_192c_tp4_f512_v1` has no route band
below 49 routes. Every expert with `M < 49` therefore inherits the operator-wide
legacy split-W13 geometry: two 4 MiB W13 ranges and one 4 MiB W2 range. Two
questions follow.

1. Does a short-route expert lose useful packed-B bandwidth in that regime, and
   if so by how much and why?
2. Does closing the gap survive end to end on the production planner, without
   regressing workloads that have no short-route experts?

## Machine and method

- Host: `AmazonC5192Cores`, 192 Arm Neoverse-V3 cores, two 96-core NUMA nodes,
  2 MiB private L2 per core, 96 MiB LLC per node.
- Scope: NUMA0 CPUs `0-95`, NUMA-local memory, explicit 32 MiB HugeTLB backing.
- Shape: TP4 `H=4096`, `F=512`, so one expert holds 12 MiB of packed BF16
  weights (8 MiB W13 plus 4 MiB W2) and each stage window is 4 MiB.
- Kernel: SVE BF16 JIT exact-M, split-W13, one packed-B window target applied to
  both W13 and W2.
- Weights: every timed task uses a distinct expert, so packed B is always
  streamed rather than re-read from a previous call.

Isolated probe:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_heterogeneous_overlap.py \
  --output /tmp/balanced_m28_T4.json \
  --hidden-size 4096 --ffn-hidden-size 512 --num-experts 224 \
  --big-experts 28 --big-routes 317 \
  --small-experts 192 --small-routes 28 --small-threads 4 \
  --small-lane-sweep 24 --small-window-sweep 0,2,1,0.5,0.25 \
  --big-core-splits 56 --cpu-ids 0-95 --warmup 3 --runs 11
```

End-to-end A/B:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_short_route_stage_windows.py \
  --preset dsv4-real-2048-seq70 --warmup 5 --runs 31 \
  --output .../short_route_windows_20260806/dsv4-real-2048-seq70.json
```

The isolated probe uses 192 experts because 192 divides every measured lane
count (96, 48, 24, 12). An earlier 195-expert sweep is retained in the data
directory but must not be used for width comparisons: the leftover three tasks
put a lane-count-dependent tail on the critical path, which biases wide teams by
up to 4 percentage points.

`useful GB/s` counts the 12 MiB of compulsory packed weight per expert exactly
once, so it measures delivered useful bytes rather than DRAM traffic. The
measured peak on this node is `367.4 GB/s`.

## Result 1: the governing variable is the per-thread window

`M=28`, three M12 panels (`12+12+4`), 192 experts, all 96 cores busy, useful
GB/s. Rows align the four team widths at equal per-thread window
`window_bytes / threads`:

| per-thread window | 96x1T | 48x2T | 24x4T | 12x8T |
| ---: | ---: | ---: | ---: | ---: |
| 4 MiB | 123.8 | | | |
| 2 MiB | 133.0 | 128.4 | | |
| 1 MiB | 203.7 | 203.6 | 209.7 | |
| 0.5 MiB | 236.1 | 241.7 | 249.4 | 240.5 |
| 0.25 MiB | **285.0** | 297.7 | **300.8** | 290.8 |
| 0.125 MiB | | **298.9** | 300.8 | **291.7** |
| 0.0625 MiB | | | 297.7 | 287.7 |
| 0.03125 MiB | | | | 280.5 |

At every per-thread window the four widths agree within `3.0%-5.6%`. The
per-thread window spans `123.8-300.8 GB/s`, a factor of `2.43`. Therefore the
per-thread packed-B window is the governing variable and team width is a
second-order effect. Adding threads and shrinking the window are interchangeable
ways to reach a given per-thread window; neither dominates the other.

Team width does set the attainable ceiling through memory-level parallelism.
Best per width: `1T 285.0`, `2T 298.9`, `4T 300.8`, `8T 291.7`. The `4T`
optimum is `5.5%` above `1T` and `3.1%` above `8T`, so the interior optimum is
real but small.

## Result 2: the loss is bounded by the M12 panel count

Define the effective re-read factor `p_eff = 367.4 / measured`. Taking the best
width at each per-thread window:

| per-thread window | best useful GB/s | `p_eff` |
| ---: | ---: | ---: |
| 4 MiB | 123.8 | 2.97 |
| 2 MiB | 133.0 | 2.76 |
| 1 MiB | 209.7 | 1.75 |
| 0.5 MiB | 249.4 | 1.47 |
| 0.25 MiB | 300.8 | 1.22 |
| 0.125 MiB | 300.8 | 1.22 |
| 0.0625 MiB | 297.7 | 1.23 |

`p_eff` is bounded above by the M12 panel count `ceil(M/12) = 3` and reaches it
at `4 MiB` per thread. It decreases monotonically as the per-thread window
shrinks and plateaus at about `1.22` from `0.25 MiB` per thread. The residual
`1.22` is not explained by packed-B re-reads; it covers per-range dispatch, A
rescans and the `M4` third panel.

The mechanism is that the kernel loops M panels outside and N tiles inside, so
each additional M panel walks the whole packed-B window again. When the
per-thread slice of that window does not stay in the 2 MiB private L2, and the
aggregate concurrent window exceeds the 96 MiB LLC, the walk returns to DRAM.

`M=12` is the control. It has exactly one panel, so there is no packed-B reuse
to protect and the window should not matter:

| window | 96x1T | 24x4T |
| ---: | ---: | ---: |
| legacy 4 MiB | 364.7 | 362.2 |
| 2 MiB | **367.4** | 362.5 |
| 1 MiB | 366.2 | 363.1 |
| 0.5 MiB | 365.4 | 359.6 |
| 0.25 MiB | 365.3 | 347.7 |

The spread at `1T` is `0.7%`, and only over-shrinking at `4T`
(`0.0625 MiB` per thread) costs `4.0%`. This confirms that the effect measured
above is packed-B panel reuse and not a generic window-size artifact.

## Result 2b: the retained level is private, not the shared LLC

With every core busy the aggregate active packed-B window is `C * omega`, so a
sweep that only varies the window target cannot separate a private-L2 effect
from a shared-LLC capacity effect. Varying the active core count at fixed team
width separates them. Team width is fixed at `4T`, lane counts are `3/6/12/24`
so that 192 experts divide exactly, and `p_eff` is measured as
`BW(M=12) / BW(M=28)` at the same point, which normalises out whatever DRAM or
MLP ceiling applies at that core count. The `M=12` control varies by at most
`1%` across windows at `C=96`, so the normaliser is sound.

| `omega` | `C=12` | `C=24` | `C=48` | `C=96` |
| ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 2.35 (agg 12 MiB) | 2.11 (24) | 1.81 (48) | 1.73 (96) |
| 0.5 MiB | 2.64 (6) | 2.19 (12) | 1.68 (24) | 1.46 (48) |
| 0.25 MiB | 2.14 (3) | 1.76 (6) | 1.25 (12) | 1.21 (24) |
| 0.125 MiB | 2.12 (1.5) | 1.79 (3) | 1.20 (6) | 1.20 (12) |

A shared-LLC capacity mechanism predicts that `p_eff` depends on the aggregate
window. At a fixed aggregate of 12 MiB the four measured combinations give
`2.35, 2.19, 1.25, 1.20`, a factor of two, and they track `omega` monotonically.
Conversely, at fixed `omega` the aggregate varies eightfold while `p_eff` moves
by only `1.36x-1.77x`, and it *improves* as aggregate pressure grows. The
shared-LLC capacity hypothesis is therefore rejected with the wrong sign, and
the per-thread window remains the governing variable.

The two effects are separable and multiplicative. At `C=96`, shrinking `omega`
from 1 MiB to 0.125 MiB gives `1.73 -> 1.20`, a factor `1.44`. At
`omega = 1 MiB`, raising `C` from 12 to 96 gives `2.35 -> 1.73`, a factor
`1.36`. The predicted corner `2.35 / (1.36 * 1.44) = 1.20` matches the measured
`(C=96, omega=0.125)` value exactly.

Two consequences follow. First, the deterministic `g(M, t)` policy form needs no
active-core term, and a policy calibrated at 96 cores errs conservatively when
fewer cores are active, because `p_eff` only improves with `C`. Second, the
effective retained capacity is about `0.25 MiB` per thread, roughly one eighth
of the nominal 2 MiB private L2, which is consistent with packed A, the
intermediate and the C output streaming through the same L2 and evicting B.

Why `p_eff` improves with `C` at fixed `omega` is not identified here. Two
candidates remain: hardware prefetchers throttled by memory pressure at high
`C` inject less speculative traffic through L2, or the ratio is inflated at low
`C` because `M=28` carries three panel loops per expert while `M=12` carries one
and the per-expert time is shorter at low `C` (`0.437 ms` versus `1.436 ms`).
Separating these requires PMU counters or an independent per-expert fixed-cost
calibration.

## Result 3: end-to-end A/B on the production planner

Four variants, one process, interleaved order, 5 warmups and 31 measured calls,
all bit-exact against `legacy` before timing:

- `legacy`: production auto plan with the stage-window policy disabled.
- `policy_v1`: production auto plan with the calibrated default policy.
- `manual_ext`: the `legacy` plan with short-route windows applied post hoc.
  The task graph, widths and placement are byte-identical to `legacy`.
- `policy_ext`: production auto plan with an extended policy, so the cost model
  re-scores every candidate with the short-route windows lowered.

The extended band is `13 <= M <= 48` with
`1T: 0.25 MiB, 2T: 0.5 MiB, 4T: 1 MiB, 8T: 1 MiB` applied to both W13 and W2.
`M <= 12` is deliberately excluded.

| preset | in-band experts / routes | legacy | `policy_v1` | `policy_ext` | ext vs `policy_v1` |
| --- | --- | ---: | ---: | ---: | ---: |
| `dsv4-real-2048-seq70` | 120 / 3314 (27%) | 15.207 ms | 15.155 ms | **14.116 ms** | **+7.36%** |
| `moe256-uniform` | 256 / 12288 (100%) | 17.010 ms | 17.000 ms | **13.688 ms** | **+24.20%** |
| `moe256-long-short-bimodal` | 0 / 0 | 11.428 ms | 11.405 ms | 11.415 ms | -0.09% |
| `moe256-active-set-128` | 0 / 0 | 12.135 ms | 9.646 ms | 9.630 ms | +0.17% |
| `moe256-tiered-hotspot` | 0 / 0 | 9.194 ms | 8.379 ms | 8.351 ms | +0.34% |

Reversing the interleaved variant order reproduced the two in-band cases:
`dsv4` `manual_ext +7.98%` and `policy_ext +7.83%`; `uniform` `+25.13%` and
`+25.17%`. Forward and reverse medians differ by at most `0.9` percentage
points. P90 spikes follow the run position rather than the variant, so only
medians are reported.

Two observations matter for risk:

- `manual_ext` and `policy_ext` agree within `0.2%` on every preset. All of the
  gain comes from the window values; the cost model's re-search did not change
  any shape, execution mode or placement. Every preset kept the shape
  `policy_v1` had already selected.
- `moe256-uniform` is `M=48` for all 256 experts, exactly one route below the
  existing band's `min_routes=49`, so `policy_v1` overrides zero tasks and
  gains `0.06%`. This is a coverage gap, not a calibration limit.

## Interpretation

The default policy already encodes small windows where it has bands: it uses
`128 KiB` at `1T` for `96-287` routes. The short-route gap is a missing band,
not a missing mechanism. Extending it is additive: three presets with no
in-band expert moved by at most `0.34%`, which is inside run-to-run noise.

Two earlier conclusions in this repository need to be read with this result.

- `results/amazon_192c_weight_windows.md` reports a best per-thread window of
  about `1 MiB`. That calibration used `M=2040`, where A is `16.7 MB` per range
  and range multiplication is expensive. It does not transfer to short routes,
  whose optimum here is `0.125-0.25 MiB` per thread.
- The four-state count in `MATHEMATICAL_MODEL.md` section 9.23 assumes packed B
  is hot on every revisit, that is `p_eff = 1`. That holds only when the
  per-thread window is small enough. The `M24` underprediction of `16.53%`
  recorded there is consistent with `p_eff = 2` at the window used.

## Landed

The band is now the production default. On 2026-08-07 the policy input unit
changed from per-range bytes to the per-thread window, which is the invariant
these sweeps measured. The calibrated table collapsed from 26 per-range numbers
to 8 per-thread windows plus 6 cells that sit one factor-of-two step away, and
the short-route band landed as `amazon_c5_192c_tp4_f512_v2`. Two independent
confirmations came out of that work:

- PMU. `l2d_cache_refill` times 64 measures bytes crossing into L2 and does
  include hardware prefetches, which the `M=12` control verifies: crossing bytes
  over compulsory packed-B is `1.02`. At `M=28`, `24x4T`, after subtracting the
  A and C traffic that does not grow with the range count, the counter gives
  `p_eff = 1.83 / 1.15 / 1.16` against `1.75 / 1.22 / 1.23` from wall time. So
  `p_eff` is literally the number of times packed B crosses the L2 boundary.
- The large-`M` regime is dominated by A, not B. At `M=2040`, `24x4T`, shrinking
  the per-thread window from `1 MiB` to `0.0625 MiB` raises L2-boundary traffic
  from `6.13 GB` to `50.37 GB`, and even at `1 MiB` that traffic is already
  `21.8x` the compulsory packed-B. Per-range A is `2*M*K`, which is `229 KiB`
  for W13 at `M=28` and stays resident, versus `16.7 MB` at `M=2040`, which does
  not. That asymmetry, not the B reuse rate, is why the optimal window rises
  with `M`.

Re-measured default against default in one session on the same extension:

| preset | V1 default | V2 default | speedup | legacy noise |
| --- | ---: | ---: | ---: | ---: |
| dsv4-real-2048-seq70 | 15.156 ms | 14.087 ms | **+7.59%** | -0.21% |
| moe256-uniform | 16.983 | 13.664 | **+24.29%** | +0.38% |
| moe256-long-short-bimodal | 11.453 | 11.421 | +0.28% | -0.72% |
| moe256-active-set-128 | 9.612 | 9.639 | -0.28% | -0.50% |
| moe256-tiered-hotspot | 8.348 | 8.321 | +0.33% | +0.10% |

The `legacy` variant does not use the policy at all, so its own session-to-session
swing bounds the noise floor, and it is larger than the movement on the three
presets with no in-band expert. `--reverse-order` gives `7.38%` and `23.67%`.

## Result 4: the W13 and W2 windows are not equally important

`profile_heterogeneous_overlap.py --small-w2-window-sweep` crosses the two
per-thread windows. Per-task stage windows only exist in Plan V2, so that mode
routes through `fused_moe_bf16_tiled_async_plan`; the operator-wide path is
unchanged. Fixed `24x4T`, all 96 cores busy, 192 homogeneous experts:

| M | best (w13, w2) MiB | best GB/s | w2 spread at best w13 | production band | gap |
| ---: | :--- | ---: | ---: | :--- | ---: |
| 13 | (0.125, 0.25) | 336.1 | 0.67% | (0.25, 0.25) | -1.36% |
| 28 | (0.25, 0.25) | 302.4 | 4.81% | (0.25, 0.25) | **exact** |
| 48 | (0.125, 0.25) | 270.5 | 0.69% | (0.25, 0.25) | -2.95% |
| 120 | (0.125, 0.125) | 174.2 | 9.53% | (0.125, 0.125) | **exact** |
| 320 | (0.5, 0.125) | 62.9 | 1.52% | (0.5, 0.125) | **exact** |

Three conclusions.

The three production band values are reproduced exactly. `M=28` sits in the new
`13-48` band, `M=120` in `96-143`, `M=320` in `288-575`, and the two-dimensional
optimum lands on the calibrated pair in all three. Those pairs were originally
found by searching per-range bytes cell by cell, so this independently validates
both the table and the per-thread parameterization.

W13 is the strong axis and W2 is the weak one. Moving `w13` one step off its peak
costs 3% to 30%. Holding `w13` at its peak, the spread across three or four `w2`
values is usually under 1.5%; the larger spreads at `M=28` and `M=120` come
entirely from the cliff at `w2 = 0.5 MiB`, not from slope near the optimum. The
earlier caveat that sharing one budget makes the E2E gains a lower bound is
therefore withdrawn: it does not materially understate them.

The `H/F = 8` shared-A argument holds only at large M. At `M=320` the two optima
differ by 4x in the predicted direction. At `M=120` they are equal, and at
`M <= 48` W2 prefers the *larger* window. There both stages' shared-A is only
13-393 KiB and fits private L2, so the re-scan term is negligible for both; what
sets the short-route `w2` optimum is not identified, but the effect is under
1.5%.

The `13-48` band keeps `w13 = 0.25 MiB`. The isolated optimum is `0.125 MiB` at
`M=13` and `M=48`, worth 1.36% and 2.95%, but `M=28` peaks at the current value
and a band must pick one. End to end, `0.125 MiB` gives 7.38% against legacy on
`dsv4-real-2048-seq70` versus 7.25% for the current value, inside noise; on
`moe256-uniform` it improves 13.693 to 13.513 ms, but only because the cost model
flips the shape from `8T` to `1T`. In the same run the `manual` variant, which
keeps the legacy shape and changes only windows, regresses from 13.686 to
13.803 ms. The gain is below the 2% adoption threshold and that shape has no
isolated calibration at this window, so it is not landed.

## Result 5: the legacy geometry already divides by the team width

Holding the per-thread window fixed and sweeping the team width shows the
invariance measured at 1 to 8 threads does not extend further:

| t | M=28 at 0.25 MiB/thread | M=120 at 0.125 MiB/thread |
| ---: | ---: | ---: |
| 1 | 288.3 GB/s (-4.2%) | 153.6 GB/s (-11.4%) |
| 2 | 296.2 (-1.6%) | 162.8 (-6.1%) |
| 4 | **300.9** | **173.4** |
| 8 | 295.0 (-2.0%) | 166.6 (-3.9%) |
| 16 | 261.9 (**-13.0%**) | 139.2 (**-19.7%**) |
| 32 | 219.7 (**-27.0%**) | 105.7 (**-39.0%**) |

Beyond 8 threads the width itself costs more than any window can recover, most
likely through intra-team synchronisation and the collapse in concurrent experts
to 6 or 3. So a per-thread window is not transferable to 16 and 32 threads.

That turns out not to matter, because the operator-wide legacy geometry is itself
a per-thread window that shrinks with the team:

```
split-W13 gives one 4 MiB W13 range and one 4 MiB W2 range, so
omega = 4 MiB / t:  t=1 -> 4 MiB,  t=2 -> 2,  t=4 -> 1,
                    t=8 -> 0.5,    t=16 -> 0.25,  t=32 -> 0.125
```

Wide teams were therefore never in the bad regime. Measured against legacy at
those widths, the best window is worth only `+2.4%` (M=28, 16T), `+0.8%`
(M=28, 32T), `+4.2%` (M=120, 16T) and `+8.5%` (M=120, 32T, where legacy at
0.125 MiB is now slightly *too small*). Widths 16 and 32 are left inherited.

The real hole was the narrow end of the `49-95` band, which V1 calibrated at 8
threads only. At `M=72`, legacy against the best window:

| t | legacy | best window | best omega | gain |
| ---: | ---: | ---: | :--- | ---: |
| 1 | 63.2 GB/s | 214.1 | 0.125 MiB | **3.39x** |
| 2 | 79.1 | 232.4 | 0.125 | **2.94x** |
| 4 | 147.6 | 243.9 | 0.0625 | **1.65x** |
| 8 | 170.7 | 233.7 | 0.0625 | 1.37x (V1 uses 0.125, worth 1.35x) |

`amazon_c5_192c_tp4_f512_v3` fills the three narrow cells at the measured optima
and leaves the 8-thread cell exactly as V1 calibrated it. No preset in the
catalog plans a 49-95 route expert at 1, 2 or 4 threads, so V2 and V3 build
element-identical plans on all five presets; the wall-clock differences of at
most 1.26% are noise by construction. The value is latent: those cells now have
a calibrated window instead of a 32x-too-large inherited one, and the cost model
no longer scores them with the full-workload anchor.

## Result 6: split/no-split cannot leave the candidate set

Since `w13_split` is only a boolean alias for `g_w13 in {4, 8} MiB`, the natural
next step would be to drop it as a planner search dimension once the per-thread
bands cover enough ground. They never will, and two of the three reasons are
deliberate:

| why a task inherits | can a band cover it |
| :--- | :--- |
| `M <= 12` | **should not**. With one M panel there is no packed-B reuse to protect; the measured window effect is ±0.7% |
| `M > 575` | could, with new calibration. At `M=2040` the optimum is about 1 MiB per thread, which is what legacy already gives at 4T |
| `t > 8` | **should not**. Result 5 measured that `4 MiB / t` is already near optimal at 16 and 32 threads, and the per-thread window stops transferring there |

V3 covers 52% of the `(routes, threads)` grid. The share of production tasks still
on the operator-wide geometry is 0% for `moe256-uniform` and
`moe256-active-set-128`, 6% for `moe256-tiered-hotspot`, 35% for
`dsv4-real-2048-seq70`, and **100%** for `moe256-long-short-bimodal`, whose routes
are only `{12, 2040}` and widths only `{1, 16}` so every cell falls into one of the
three rows above. For that workload split/no-split is the *only* window control.

The operator-wide geometry is therefore not a retirable compatibility layer but
the live window source for every inherited task, and the `policy_variants()`
requirement of a complete legacy pair is what keeps both live geometries
calibrated. The candidate set for this identity is already the minimal two, split
and no-split, with no measured window variants, so there is no dimension to
collapse.

## Result 7: the optimum steps where the shared-A scan outgrows L2

The two competing terms in Result 4 predict a threshold. Each thread scans all of
A inside one range, since the N axis is partitioned across the team, so the
per-thread scan is `2*M*K`. While that fits private L2 it survives across ranges
and range multiplication is nearly free; once it does not, the cost is
proportional to the range count. For W13 with `K = H = 4096` that crossing is at
`M = 256`; for W2 with `K = F = 512` it is at `M = 2048`, a ratio of exactly
`H / F = 8`.

Fixed `24x4T`, 192 homogeneous experts, `w2` held at 0.125 MiB per thread, useful
packed-B bandwidth in GB/s:

| M | A_w13 / L2 | 1/16 | 1/8 | 1/4 | 1/2 | 1 | best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 0.47 | 169.8 | **174.2** | 167.1 | 143.7 | 132.0 | 1/8 |
| 160 | 0.62 | 124.9 | **128.7** | 127.6 | 112.1 | 102.9 | 1/8 |
| 176 | 0.69 | 114.0 | **117.8** | 116.4 | 106.5 | 95.9 | 1/8 |
| 192 | 0.75 | 105.5 | **108.2** | 107.9 | 102.7 | 91.3 | 1/8 |
| 200 | 0.78 | 99.8 | 103.1 | **103.3** | 99.2 | 87.6 | 1/4 |
| 224 | 0.88 | 86.7 | 88.2 | 88.9 | **89.6** | 81.4 | 1/2 |
| 240 | 0.94 | 78.5 | 76.0 | 79.4 | **84.1** | 78.2 | 1/2 |
| 256 | **1.00** | 68.2 | 63.5 | 71.5 | **78.3** | 74.6 | 1/2 |
| 300 | 1.17 | 49.4 | 47.7 | 61.0 | **67.5** | 66.1 | 1/2 |
| 320 | 1.25 | 44.6 | 44.0 | 56.8 | **63.0** | 61.8 | 1/2 |

The threshold holds. The optimum climbs 1/8 -> 1/4 -> 1/2 as `A_w13 / L2` goes
0.62 -> 0.78 -> 1.00 and saturates exactly where A fills L2. It is a two-step ramp
rather than a jump, starting around `A / L2 = 0.62`, consistent with A not having
exclusive use of L2: packed B, the intermediate and C stream through it too. The
1/4 "plateau" is a single point at `M=200` and only 0.2% better than 1/8, so it
does not earn a band of its own.

The step is width-independent, which is what the mechanism predicts because
`2*M*K` carries no team-width term. At `M=256` all four widths peak at 1/2 MiB per
thread:

| t | 1/8 | 1/4 | 1/2 | 1 | cost of the old band's 1/8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.6 | 55.2 | **67.1** | 65.0 | **-61%** |
| 2 | 62.8 | 68.2 | **72.8** | 71.1 | -16% |
| 4 | 63.5 | 71.5 | **78.3** | 74.6 | -23% |
| 8 | 63.1 | 73.5 | **80.1** | 74.1 | -27% |

That exposes a defect in the V1 table: the `144-287` band prescribed 1/8 MiB
across its whole range, but everything from `M=224` up wants 1/2 MiB.
Interpolating between `M=200`, where 1/8 leads by 3.9%, and `M=224`, where 1/2
leads by 1.6%, puts the crossover at `M ~ 217`.
`amazon_c5_192c_tp4_f512_v4` splits the band into `144-215`, which keeps V1's own
values including its 8-thread override, and `216-287` at `(1/2, 1/8)` with no
overrides. The seam sits at 216, an integral number of M12 panels, and costs at
most 0.2% there. `288-575` is untouched.

Almost nothing in the catalog lands in the new band: three of
`dsv4-real-2048-seq70`'s 223 tasks, about 2% of its routes, and zero elsewhere. So
all five presets move within +-0.78% end to end, against a `legacy` variant that
swings -1.28% to +0.34% in the same comparison, and every chosen shape is
unchanged. The gain is latent, like the 49-95 fill.

## Not established

- The band ties the W13 and W2 targets to one per-thread window. Result 4 shows
  W2 is the weak axis, so this does not materially cost anything, but the
  mechanism that sets the short-route W2 optimum is not identified.
- The mechanism behind the residual `g(C)` term in result 2b, that is why
  `p_eff` improves as more cores stay busy at fixed per-thread window.
- Only widths `1, 2, 4, 8` are covered. Result 5 shows that is deliberate rather
  than a gap: the inherited legacy window is `4 MiB / t`, which already lands near
  the measured optimum at 16 and 32 threads, and at those widths the per-thread
  window is no longer transferable. The residual headroom there is 0.8-8.5%,
  measured but not landed, and no catalog preset plans a banded expert that wide.
- The `49-95` band's narrow cells are calibrated but unexercised: no catalog
  preset plans a 49-95 route expert below 8 threads, so their 1.65x-3.39x
  isolated gain has no end-to-end confirmation.
- The LLC-to-DRAM segment is still unmeasured. `ll_cache_miss_rd` counts only
  demand misses, `13 MB` where `2.4 GB` actually moved, and
  `l3d_cache_refill` reads zero on Neoverse-V3.
- One host, one profile identity. The policy is profile-bound by design, so the
  band must not be extrapolated to other machines, `F` values or parallel
  degrees without repeating the measurement.
- The runtime extension is the 2026-08-03 build of the current ARM source. It is
  newer than the profile used for planning, so these numbers validate the
  runtime policy, not the profile's absolute-time accuracy.

## Data

- `results/data/short_route_windows_20260806/balanced_m{12,28}_T{1,2,4,8}.json`:
  balanced 192-expert window sweeps.
- `results/data/short_route_windows_20260806/sep_m{12,28}_t4.json`: the
  active-core separation experiment at fixed `4T` team width.
- `results/data/short_route_windows_20260806/{preset}.json` and
  `{preset}_reversed.json`: end-to-end A/B against the candidate band, per-run
  samples included.
- `results/data/stage_window_omega_20260807/{preset}.json`: the same A/B after
  the band landed, with the simplified `legacy` / `policy` / `manual` variants.
- `results/data/stage_window_omega_20260807/w2_2d_m{13,28,48,120,320}.json`: the
  two-dimensional W13/W2 per-thread window calibration.
- `results/data/stage_window_omega_20260807/w13_eighth_{preset}.json`: the
  end-to-end test of the rejected `w13 = 0.125 MiB` band.
- `results/data/stage_window_omega_20260807/omega_inv_m{28,120}_t{1..32}.json`:
  per-thread window invariance across six team widths.
- `results/data/stage_window_omega_20260807/wide_m{28,120}_t{16,32}.json`: legacy
  against explicit windows at 16 and 32 threads.
- `results/data/stage_window_omega_20260807/band4995_m72_t{1,2,4,8}.json`: the
  49-95 band's narrow-width calibration.
- `results/data/stage_window_omega_20260807/v3_{preset}.json`: the V2-to-V3 A/B.
- `results/data/stage_window_omega_20260807/thresh_m{120..320}.json` and
  `thresh_m256_t{1,2,8}.json`: the shared-A threshold sweep and its width check.
- `results/data/stage_window_omega_20260807/v4_{preset}.json`: the V3-to-V4 A/B.
- `results/data/heterogeneous_overlap_20260806/`: the earlier 195-expert sweeps
  and the heterogeneous co-scheduling probes that led to this measurement.
