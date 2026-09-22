# Does the isolated cost carry a per-panel term? Measured on route counts chosen to discriminate

## Status

Two questions were asked. The main one - is the isolated cost additive in M12 panels rather
than in rows - comes back **inconclusive under its own frozen rule**, and the model is not
changed. The secondary one is settled: the kernel's tail table is keyed on `routes % 8`, and
the model's use of `routes % 12` is wrong by 11% at routes 9 and 10.

## Design

`tmp/c9g_panel_20260922/design.md`, frozen before collection. Isolated profile
(`profile_contention_async.py`) with the route buckets extended to
`1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,24,36,48,96,192,2040`, widths 1-96, TP4 (F=512) and
TP2 (F=1024), 144 experts, node 0, jemalloc never-purge, `scripts/wait_idle.sh` before each
pass. Two passes per shape. Amazon C9g, the instance rebuilt on 2026-09-22, calibrated from
the median of three service probes.

The panel account says `cost(q) = a + P*floor(q/12) + tail(q%12)`, from which four
parameter-free identities follow. The frozen rule: confirmed if the identity holds within 3%
at all four route counts in both passes and the current row-rate model is wrong by more than
5% on at least two; rejected if the identity is wrong by more than 5% anywhere; otherwise
inconclusive and the model is not changed.

## The identity, at one thread

Repeat error between the two passes is 0.01-1.36%, so the machine is not the limit.

| shape | routes | identity | pass a | pass b | row-rate model |
| --- | --- | --- | --- | --- | --- |
| TP4 | 13 | `cost(1)+cost(24)-cost(12)` | **+2.0%** | **+3.6%** | -30.1% |
| TP4 | 17 | `cost(5)+cost(24)-cost(12)` | -0.9% | -0.5% | -2.4% |
| TP4 | 20 | `cost(8)+cost(24)-cost(12)` | -0.7% | -0.4% | -8.2% |
| TP4 | 36 | `2*cost(24)-cost(12)` | -0.3% | -0.2% | +0.3% |
| TP2 | 13 | | **+4.9%** | **+4.9%** | -33.7% |
| TP2 | 17 | | -1.0% | -1.6% | -7.2% |
| TP2 | 20 | | -1.2% | -1.1% | -12.0% |
| TP2 | 36 | | +0.9% | -0.2% | -3.4% |

Worst identity error over both passes: 3.6% (TP4) and 4.9% (TP2). Both land in the
3-5% band, so by the frozen rule this is **inconclusive on both shapes**, and the per-panel
term does not become a model change on this evidence.

What the numbers also show, and what the rule deliberately does not let count:

- The failure is entirely at **routes 13**. At 17, 20 and 36 the identity is within 1.6% on
  every pass of both shapes, which is near the repeat error.
- At routes 13 it overpredicts, with the same sign in all four passes - a reproducible
  structural miss, not scatter.
- The current row-rate model is wrong by 30% and 34% at routes 13, and by 8% and 12% at
  routes 20. So the identity being imperfect does not rescue the row rate; both are wrong at
  13, in opposite directions and by very different amounts.

Routes 13 is one full panel plus a one-row tail, and the identity prices that tail with
`cost(1)`, measured with no panel before it. Why that should be the one arrangement the
identity misses, while the five-row and eight-row tails fit, is not established here.

## The tail table's modulus, at one thread

The model applies its tail capacity to `routes % 12`; the kernel's m8 dispatch
(`packc_w13_tail_dispatch`: tail 1->m1, 2->m2, 3->m4, 4->m4, 5/6/7->m8) applies the same
table to `routes % 8`. Pricing both effective-row counts the same way and comparing to
measurement:

| routes | TP4 `%12` | TP4 `%8` | TP2 `%12` | TP2 `%8` |
| --- | --- | --- | --- | --- |
| 9 | **+11.4%** | -5.2% | **+11.3%** | -5.7% |
| 10 | **+10.7%** | -5.2% | **+10.8%** | -5.7% |
| 13 | +2.8% | +2.2% | +4.9% | +5.2% |
| 14 | -1.0% | +2.2% | +4.4% | +5.4% |
| 17 | +5.6% | **-0.7%** | +4.3% | **-1.3%** |
| 18 | +5.4% | **-0.9%** | +4.5% | **-0.9%** |

`% 8` wins decisively at routes 9, 10, 17 and 18 on both shapes, and the two are
indistinguishable at 13 and 14. The static reading of the dispatch is confirmed: **the model's
modulus is wrong**, and it costs 11% at routes 9 and 10, where `% 12` rounds a nine-row task
up to a full twelve-row panel that the kernel never runs.

Neither table is right at routes 9 and 10 - `% 8` is 5% low on both shapes - so fixing the
modulus alone moves the error from +11% to -5%, not to zero.

### Correction (2026-09-22): the modulus comparison used a table the model does not use

The section above compares the kernel's `routes % 8` dispatch against the model applying its
tail table to `routes % 12`. That is the `static_bucketed` tail policy. **These calibrations
use `xbyak_exact_m`**, where `_exact_m` is true and `m12_effective_rows(q) == q` for every q -
the model rounds nothing at all. Scoring the row count the model actually charges against the
two tables:

| routes | measured (TP4) | exact q, the model's | `% 8` | `% 12` |
| --- | --- | --- | --- | --- |
| 9 | 454.8 us | **-5.2%** | -5.2% | +11.4% |
| 10 | 457.7 us | **-5.2%** | -5.2% | +10.7% |
| 13 | 765.0 us | +2.8% | +2.2% | +2.8% |
| 14 | 764.9 us | **-1.0%** | +2.2% | -1.0% |
| 17 | 842.9 us | **-0.7%** | -0.7% | +5.6% |
| 18 | 844.9 us | **-0.9%** | -0.9% | +5.4% |
| 20 | 895.5 us | **-0.6%** | -0.6% | -0.6% |

Mean absolute error: exact 2.3%, `% 8` 2.4%, `% 12` 5.4% on TP4; 3.4%, 3.6%, 5.9% on TP2.

So **the active path is already the best of the three and there is no modulus defect to fix**.
The `% 12` claim stands only against the `static_bucketed` policy, which no calibration in use
here selects. The earlier statement that "the model's modulus is wrong and it costs 11% at
routes 9 and 10" is withdrawn.

What survives is the residual the tables were competing to explain: **routes 9 and 10 are
under-predicted by 5.2% and 5.7% by every candidate**, exact included. That is the real open
item, and no row-count table addresses it, since exact and `% 8` charge the same rows there.

## What changes

Nothing yet. The identity result is inconclusive by the rule that was frozen for it, and the
modulus result is a planner-visible cost-model change that needs its own decision: routes 9
and 10 sit below the planner's own working region (width <= 16, M >= 24), so the 11% error has
no measured consequence for plan choice yet.

## Open

- Routes 13 is the one arrangement both accounts miss. A profile at routes 25, 26, 37 and 38 -
  one and two panels plus a one-row and two-row tail - would say whether the miss follows the
  tail size or the panel count.
- Neither table explains routes 9 and 10. The 5% that `% 8` leaves is the next residual.
- Everything here is at one thread. The 4-16 thread region, where the offline analysis found
  the identity and the model bracketing the measurement, is untouched by this run.
