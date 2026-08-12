# Amazon 192-Core Range-Native Cost-Profile Refresh

Date: 2026-08-09

Status: historical result. The `(1,1)` / `(2,1)` operator-wide range identity
was removed from the planner and active profile catalog in model changelog
v0.88. These tables document that transition and must not be loaded as current
production calibration; current plans carry per-task tile windows instead.

## Scope

This is the first full TP4/F512 cost-table refresh after removing the public
split/no-split interface. It measures the two exact production stage-range
identities instead:

- `(W13 ranges, W2 ranges) = (1, 1)`
- `(W13 ranges, W2 ranges) = (2, 1)`

The profiles were measured on `AmazonC5192Cores` with the current SVE JIT
extension.

- CPUs: rank 0 on `0-95`, rank 1 on `96-191`
- Memory: NUMA nodes 0 and 1, respectively
- Shape: H=4096, F=512, BF16 SiLU, TP4
- Experts: 256 global, local, and measured experts
- Weight behavior: timed calls walk distinct consecutive expert weights
- Aggregation: synchronized ranks, pairwise maximum
- Huge pages: `/dev/hugepages-32M`
- Sampling: 5 warmups and 20 measured calls per point
- Isolated routes: `1..12,24,48,96,192,384,768,1536,2040`
- Threads: `1,2,4,8,16,32,48,64,96`
- Contention routes: `1..12,24,48,192,768,2040`
- Contention shapes: 14 shapes from `1x96T` through `96x1T`

Both profiles contain 180 isolated points and 238 contention points. Their
recorded implementation hashes are:

- source: `6fe10e44a6acec0d62c7e80549b393706327f4c74c36f50eca3c504832d1d7b3`
- extension: `f6fd824ea7934dd6a5523cce753e1d308e145217f180624470b46a6319815175`

The aggregate profile SHA256 values are:

- `(1,1)`: `2cb309aa0350fe77fc2cc8a40470e829a05519dee2334bb3c0b945e815d2f307`
- `(2,1)`: `6c3593ef16fad4394b6786a66a9c31cc2ad9e7c6bb9745fc92a333a2a6691f43`

## Isolated Results

The table reports the measured best width for one expert. Short-route minima
at very wide teams are latency minima and must not be read as throughput
recommendations; the contention table is authoritative for many experts.

| Routes | `(1,1)` best | ms | TFLOP/s | `(2,1)` best | ms | TFLOP/s | Faster |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32T | 0.0618 | 0.204 | 64T | 0.0672 | 0.187 | `(1,1)` 8.7% |
| 12 | 16T | 0.0864 | 1.747 | 16T | 0.0889 | 1.698 | `(1,1)` 2.9% |
| 24 | 64T | 0.1143 | 2.642 | 64T | 0.1171 | 2.578 | `(1,1)` 2.5% |
| 48 | 64T | 0.1459 | 4.140 | 64T | 0.1505 | 4.014 | `(1,1)` 3.1% |
| 192 | 64T | 0.3637 | 6.643 | 64T | 0.3991 | 6.053 | `(1,1)` 9.8% |
| 384 | 64T | 0.5960 | 8.107 | 64T | 0.7362 | 6.563 | `(1,1)` 23.5% |
| 768 | 64T | 1.0543 | 9.166 | 96T | 1.4489 | 6.670 | `(1,1)` 37.4% |
| 1536 | 96T | 2.0266 | 9.537 | 96T | 2.9429 | 6.567 | `(1,1)` 45.2% |
| 2040 | 96T | 2.6622 | 9.642 | 96T | 3.9351 | 6.523 | `(1,1)` 47.8% |

At M=2040, `(2,1)` is 3.4% faster at 4T, but becomes 29.2% slower at 32T
and 47.8% slower at 96T. The extra W13 range therefore helps only in a
limited working-set regime; it is not a general replacement for `(1,1)`.

## Contention Results

Each row executes all 256 experts. Aggregate TFLOP/s includes both NUMA ranks'
expert work represented by the synchronized full-call time.

| Routes | `(1,1)` best | ms | TFLOP/s | `(2,1)` best | ms | TFLOP/s | Global best |
| ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 1 | `48x2T` | 8.755 | 0.736 | `48x2T` | 8.766 | 0.735 | `(1,1)`, +0.1% |
| 12 | `48x2T` | 8.952 | 8.636 | `48x2T` | 8.965 | 8.624 | `(1,1)`, +0.2% |
| 24 | `6x16T` | 12.503 | 12.367 | `6x16T` | 11.409 | 13.552 | `(2,1)`, +9.6% |
| 48 | `6x16T` | 15.552 | 19.884 | `6x16T` | 14.355 | 21.541 | `(2,1)`, +8.3% |
| 192 | `6x16T` | 33.269 | 37.180 | `6x16T` | 34.629 | 35.720 | `(1,1)`, +4.1% |
| 768 | `6x16T` | 119.331 | 41.463 | `24x4T` | 125.701 | 39.362 | `(1,1)`, +5.3% |
| 2040 | `6x16T` | 320.269 | 41.036 | `24x4T` | 326.530 | 40.249 | `(1,1)`, +2.0% |

The policy crossover is real: `(2,1)` wins at M=24/48, while `(1,1)` wins
from M=192 onward. For M<=12 the range choice is below 0.5% at the measured
contention optimum.

## Measurement Quality

| Policy | Full-call interval median | P90 | Rank asymmetry median | P90 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(1,1)` | 0.81% | 1.54% | 0.36% | 3.78% | 11.06% |
| `(2,1)` | 0.81% | 1.35% | 0.44% | 4.13% | 9.02% |

The low interval and median rank asymmetry make the tables suitable for model
work. The remaining rank outliers are concentrated in short-route, wide-team
points and should not be used as standalone topology conclusions.

## Model Check

The existing 2026-08-02 thin analytical calibration was evaluated against the
new profiles without fitting to them.

| Policy | Isolated MAPE | Isolated P90 | Contention MAPE | Contention P90 | Mean ranking regret | Max regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1,1)` | 11.64% | 27.21% | 11.00% | 19.23% | 1.68% | 10.17% |
| `(2,1)` | 11.26% | 21.86% | 11.47% | 19.40% | 2.05% | 10.79% |

The ranking average remains useful, but both policies fail the 5% maximum
regret gate. The failures are concentrated around the M=24/48 policy and team
width crossover. The analytical calibration must be refreshed before these
profiles become production inputs.

## Old Versus New

Against the 2026-07-27 profiles on the identical grid:

| Policy | Isolated median delta | Isolated P90 absolute | Contention median delta | Contention P90 absolute |
| --- | ---: | ---: | ---: | ---: |
| `(1,1)` | -5.87% | 29.64% | -13.68% | 32.85% |
| `(2,1)` | -3.21% | 27.23% | -12.44% | 34.56% |

This drift is materially larger than measurement noise. The old profiles
cannot represent the range-native runtime's absolute time.

## Decision

The 192-core range-native measurement is complete and schema/catalog loading
passes. The generated profiles remain validation artifacts under `tmp/` for
now. They do not replace the active profiles until the analytical calibration
is refreshed and planner/E2E validation passes. The 8-core ARM range-native
refresh remains outstanding.
