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

