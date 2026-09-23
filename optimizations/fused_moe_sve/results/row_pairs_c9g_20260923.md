# Rows are charged in pairs: one structure behind three residuals

## Status

A dense sweep of every route count from 1 to 30 shows the isolated cost is a staircase of
period two. Replacing the model's effective row count `q` with `2*ceil(q/2)` takes the mean
absolute error at one thread from 2.47% to **0.12%** on TP4 and from 2.25% to **0.10%** on
TP2, with the worst point going from 17% to 0.6%. This is a candidate, measured on the data
that produced it, and it gets a pre-registered validation before anything changes.

It also dissolves three residuals this session had been chasing separately, and it shows the
per-panel account was right all along - what was wrong was how the identity priced a one-row
tail.

## Why a dense sweep

Three rounds of hypothesis-specific route counts each refuted their own frozen reading:
routes 13/17/20/36 refuted the per-panel identity as stated, routes 25/26/37/38 refuted both
the tail-shaped and panel-shaped accounts of its miss, and the `% 12` versus `% 8` comparison
turned out to be against a tail policy no calibration selects. Rather than guess a fourth,
this measured **every** count from 1 to 30 and read the structure off the data.

Isolated profile, routes `1..30` plus `36, 48, 96, 192, 2040`, widths 1-96, TP4 and TP2, two
passes per shape, `wait_idle` before each, node 0, jemalloc never-purge. Repeat error between
passes 0.01-0.99%.

## The staircase

TP4 at one thread, cost and the increment per added route:

| routes | us | increment | | routes | us | increment |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 297.7 | +23.1 | | 12 | 504.1 | +1.6 |
| 4 | 298.7 | **+1.0** | | **13** | 762.3 | **+258.2** |
| 5 | 352.2 | +53.5 | | 14 | 762.6 | **+0.3** |
| 6 | 354.0 | **+1.8** | | 15 | 788.0 | +25.4 |
| 7 | 405.2 | +51.2 | | 16 | 788.9 | **+0.9** |
| 8 | 406.5 | **+1.3** | | 17 | 841.6 | +52.7 |
| 9 | 453.2 | +46.7 | | 18 | 842.8 | **+1.2** |
| 10 | 455.9 | **+2.7** | | **25** | 1249.5 | **+263.1** |

Going from an odd count to the next even is nearly free - 0.3 to 2.7 us - while going from
even to odd costs about 50 us. **The kernel charges row pairs**, which is the 2x2 shape of the
`bfmmla`/`smmla` result: an odd row count fills a pair and wastes half of it.

Separately, 12 -> 13 and 24 -> 25 jump by 258 and 263 us on TP4, and by 544 and 528 on TP2 -
about twice, matching twice the weights. That is the new M12 panel re-streaming the expert's
weights, which is exactly the per-panel term.

## Scoring the candidate

| | model's `exact` (`q`) | `2*ceil(q/2)` |
| --- | --- | --- |
| TP4, mean absolute error over routes 3-30 | 2.47% | **0.12%** |
| TP4, worst | 16.92% (routes 13) | 0.60% |
| TP2, mean | 2.25% | **0.10%** |
| TP2, worst | 17.06% (routes 13) | 0.63% |

Point by point on TP4: routes 13 goes from -16.9% to +0.0%, routes 25 from -10.4% to +0.2%,
routes 9 from -4.8% to +0.6%.

## Three residuals, one cause

| what was recorded | what it was |
| --- | --- |
| routes 9 and 10 under-predicted by 5.2-5.7% by every candidate row count | 9 is odd and is charged as 10 |
| the panel identity's fixed offset at routes 13 - about 19 us on TP4, 85 us on TP2 | 13 is odd and is charged as 14 |
| the offset hits one-row tails on TP4 but both one- and two-row tails on TP2 | a one-row tail makes an odd count and a two-row tail an even one; TP4's two-row tails were already correct |

**The per-panel term itself was right.** The identity missed at routes 13 not because the
panel account was wrong but because it prices a one-row tail with `cost(1)`, and `cost(1)` is
not a row - routes 2 costs 24 us *less* than routes 1 on TP4, because m1 and m2 are different
kernels. Pairing explains the miss; the 258 us jumps confirm the panel.

## Why three earlier rounds missed it

Each round picked a handful of route counts to discriminate a hypothesis, and the sets
happened to mix odd and even: 13 and 17 odd, 20 and 36 even. The even points fitted, which
read as "the identity is basically right", and the odd points' deviation was attributed to
panel structure. A period-two staircase is invisible to a sample that does not control parity.

## What does not follow yet

- `2*ceil(q/2)` is fitted on the sweep that produced it. E14's lesson is that descriptive fit
  on already-measured data does not survive fresh points, so this needs a pre-registered
  validation at counts this sweep does not cover.
- One thread only. The pairing should be width-independent, since it is a property of the
  microkernel's result shape, but that is an assumption here.
- Routes 1 and 2 are anomalous in their own right: routes 2 is cheaper than routes 1 on both
  shapes (-24.4 us TP4, -9.2 us TP2). Distinct kernels, not pairing.
- Routes 14 -> 15 increments by 25.4 us against the ~50 us of other even-to-odd steps, and
  26 -> 27 by 25.0. Unexplained, and it sits just after a panel boundary.
