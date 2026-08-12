# Amazon 192-Core DSV4 16T Schedule Check

Date: 2026-08-09

Status: historical result. The fixed `(1,1)` / `(2,1)` operator-wide range
controls used by this experiment were retired in model changelog v0.88. The
current runtime uses per-task tile windows, so these absolute times are not an
active calibration; the width and scheduling conclusions remain provenance.

## Setup

- Host: `AmazonC5192Cores`, NUMA0 CPUs `0-95`, NUMA-local allocation.
- Workload: `dsv4-real-2048-seq70`, 2048 tokens, TopK=6, 256 experts,
  223 active experts and 12,288 routes.
- Shape: H=4096, F=512, TP4 BF16 SiLU.
- Pages: explicit 32 MiB HugeTLB.
- Runtime: current SVE JIT extension, early merge on, equal-width strict tail
  stealing on for strict plans.
- Sampling: 5 warmups and 20 untraced timed calls, seed `20260739`.

The reconstructed routing distribution has 75 experts at M<=12, 120 at
M=13-48, one at M=153, and 27 at M>=192. This is not a homogeneous long-route
workload.

## Results

| Plan | Stage ranges | Median ms | Versus old early+tail | Versus current best |
| --- | --- | ---: | ---: | ---: |
| Old saved `12x8T`, early+tail | fixed legacy `(2,1)` | 14.751 | baseline | +10.10% |
| Old saved `12x8T`, tail only | fixed legacy `(2,1)` | 14.626 | +0.85% | +9.17% |
| Current `12x8T` | fixed `(2,1)` | 14.311 | +3.08% | +6.81% |
| Current `6x16T` | fixed `(2,1)` | 14.454 | +2.05% | +7.88% |
| Current `6x16T` + M<=12 2T pool | task policy, effectively `(2,1)` | 14.442 | +2.14% | +7.80% |
| Current `12x8T` | task stage-window policy | **13.398** | **+10.10%** | baseline |

Percentages are speedups, `old/new - 1`. The direct same-version,
same-range comparison is the important width result: `6x16T` is 1.00% slower
than `12x8T`. Moving the 75 M<=12 experts into a 2T dynamic pool changes the
16T result by only 0.08%.

For completeness, fixed `(1,1)` measured 14.728 ms at `12x8T` and 14.579 ms
at `6x16T`, a 1.02% 16T gain. The sign therefore depends on the range policy,
and the magnitude remains about 1%; it is not a robust reason to replace the
DSV4 plan.

## Timeline Evidence

The representative fixed `(2,1)` traces show why 16T does not win:

| Shape | Internal idle core-ms | Tail idle core-ms |
| --- | ---: | ---: |
| `12x8T` | 79.70 | 14.35 |
| `6x16T` | 130.30 | 7.47 |

Six 16T lanes reduce final tail waiting, but serialize more of the 223-expert
task chain inside each lane. Internal idle core-time rises by 50.61 core-ms,
which is much larger than the 6.88 core-ms tail reduction. The 16T plus 2T
short pool still records 116.70 internal idle core-ms, so pooling only M<=12
does not close the gap.

Standalone HTML timelines:

- previous early+tail:
  `tmp/moe_timeline/dsv4-real-2048-seq70/amazon_192c_dsv4_real_early_tail_timeline.html`
- fixed `6x16T`:
  `tmp/moe_timeline/dsv4-real-2048-seq70/amazon_192c_dsv4_real_fixed_6x16t_range2_timeline.html`
- `6x16T` plus 2T short pool:
  `tmp/moe_timeline/dsv4-real-2048-seq70/amazon_192c_dsv4_real_6x16t_short2t_pool_timeline.html`

## Decision

Do not replace the DSV4 `12x8T` plan with a blanket 16T policy. The homogeneous
cost-table result does not transfer to this 223-expert mixed distribution.
Keep 16T as a candidate for homogeneous M>=24 waves and low active-set cases;
for DSV4, the current 8T plan with per-task stage ranges is 7.88% faster than
the tested 16T plan.
