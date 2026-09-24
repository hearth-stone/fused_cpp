# Probe-calibrated event model (v10/v11) on C9g (2026-09-24)

User decision (2026-09-24): calibrate the event model that the fast (hot_wide) planner depends on
on C9g by the Arm-codex procedure, instead of running fast with Arm-codex terms. Design and
deviation: `tmp/c9g_event_model_20260924/design.md`. C9g node 0 (CPUs 0-95, `--membind=0`), TP4
expert shape, SVE-128 (n_tile 8), jemalloc never-purge. Phase skeleton: the guarded C9g research
calibration.

## Probes

The Arm-codex probes P1-P4 (four dilation curves), P5 (whole-call overhead) and P6 (v11 mid-M
terms) were ported with only the core grid changed. Background core counts are
{0, 8, 16, 32, 48, 64, 80} plus the largest 4T background next to the target (92 for 2T and 4T,
88 for 8T, 80 for 16T, 64 for 32T). There were two sessions, 246 P1-P4 cells and 187 P6 cells
each.

One deviation from the frozen design, before any build. Session 1 at the Arm-codex background
sizing (1.2x the modeled target) left 12 P1-P4 and 28 P6 cells uncovered: the target outlived
its background, because contention here is far stronger than the model the sizing uses. The
background was raised to 3.0x, session 1 was discarded, and both sessions were rerun. Only two
cells stay uncovered (dropped by rule).

Also changed on 2026-09-24 by user decision: C9g chains keep only a fixed cooldown after our own
previous process, with no load-average wait. The machine is exclusive.

## The calibration (`build_c9g.py`, `bench_assets/moe_paper/amazon_c9g_96c_tp4/probe_event_v11_c9g_20260924.json`)

| term | C9g | Arm-codex |
| --- | --- | --- |
| eps, O(t) (P2 n=0 single-expert cells, same grid fit) | 0.08, 1 us x t (fit MAE 1.2%) | 0.065, 10 + 2.5 us x t |
| t_over (P5, median of 16 plan-session values) | 0.45 ms | 0.78 ms |
| D_LL, 4T target: n = 32 / 48 / 64 / max | 2.56 / 3.76 / 4.72 / 5.86 (n = 92) | 1.12 / 1.32 / 1.64 / 1.90 (n = 76) |
| D_LS, max over widths | 1.48 (2T) - 2.25 (16T) | <= 1.35 |
| D_SL, 4T at max n | 1.28 | 1.13 |
| D_SS, 2T / 4T / 8T at max n | 1.92 / 1.51 / 1.16 | <= 1.03 |
| loading weight w(24 / 48 / 96 / 192 / 384) | 0.79 / 0.71 / 0.50 / 0.26 / 0.27 | 0.79 / 0.56 / 0.35 / 0.18 / 0.11 |
| isolated correction c(t, M 24-384) | 2T 1.03-1.10, 4T 1.00-1.08, 8T 0.99-1.11, 16T 0.92-0.95 | 2T 0.93-0.95, others ~1 |

Contention on C9g is far stronger than on Arm-codex, and it is not confined to weight loading.
Loading against loading reaches 5-6x at full occupancy, and even steady phases dilate by up to
1.9x on narrow lanes. 17 points differ by more than 3% between sessions (session mean used).
The mechanism test fails as on Arm-codex: core-time excess per expert does not collapse across
widths. The curves are a width-indexed service table.

Not re-measured: g0 = 0.02 (window isolated gain) is carried over from Arm-codex. The base
calibration prices only its calibrated widths (1/2/4/8/16/32/48/96). The probes cover 2-32T,
so 48T and 96T are extrapolated.

## Check on data the calibration never saw

Plans: the 27 workloads x homogeneous widths measured in `tmp/c9g_width_error_20260923` (18 real
TP4 layers plus 9 catalog presets, two sessions). Script: `check_width_table.py`.

Measured / event-predicted call time, median per width: 1T 1.28, 2T 1.11, 4T 1.10, 8T 1.05,
16T 1.06, 32T 0.83, 48T 0.66, 96T 0.61. The 4-16T level matches Arm-codex's 1.07-1.10. Wide
teams are overpriced beyond the probed range.

Regret of each scorer's width choice against the fastest measured width in the set:

| scorer | widths 4/8/16/32 (fast): median / mean / max / >1% | calibrated divisors 1-96: median / mean / max / >1% |
| --- | --- | --- |
| quick score | 0.34% / 2.01% / 24.2% / 12 of 27 | 1.31% / 2.88% / 23.6% / 14 |
| analytic DAG | 1.73% / 2.18% / 7.8% / 17 | 1.93% / 2.29% / 7.8% / 18 |
| **event v11 (C9g)** | **0.00% / 1.09% / 14.2% / 7** | **0.00% / 2.10% / 23.6% / 9** |

The C9g event model is the best width selector of the three. Its misses:

- Six workloads miss by 1.5-3.0%, mostly 16T preferred where 8T is faster.
- The long/short bimodal batch picks 8T where 16T is 14.2% faster: the long experts on 8T run
  slower than predicted.
- Over all widths, active-set-32 picks 2T at 23.6% worse: 2T is under-predicted, the same
  residual Arm-codex's v11 kept. The asset marks 1T/2T unreliable, and fast never uses them.

Next: derive the fast planner's templates and lane scales on C9g under this model, then validate
fast against production quick on fresh layers.
