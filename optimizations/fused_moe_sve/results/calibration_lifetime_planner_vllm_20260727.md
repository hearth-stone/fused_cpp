# SVE calibration, rank lifetime, and planner/vLLM comparison

Date: 2026-07-27

## Scope

This refresh used the current SVE JIT exact-M extension and distinct expert
weights. All profile points used 5 warmups and 20 measured runs.

- AmazonECS8Cores: CPU 0-7, standalone H4096/F512/E8 split/no-split pair.
- AmazonC5192Cores single rank: CPU 0-95, TP4 H4096/F512/E256
  split/no-split pair.
- AmazonC5192Cores dual rank: CPU 0-95 and 96-191, synchronized TP4
  H4096/F512/E256 split/no-split pair, aggregated by per-run rank maximum.
- AmazonC5192Cores EP2 lifetime: 32 cores per rank, CPU 0-31 and 96-127,
  H4096/F2048/E32 split/no-split single/dual pairs.
- Isolated routes: 1-12, 24, 48, 96, 192, 384, 768, 1536, 2040.
- Contention routes: 1-12, 24, 48, 192, 768, 2040.

The 192-core single/dual profiles have identical source and extension hashes,
180 isolated points, 14 shapes, and 238 contention points. The cost-model
companion-grid check passes. These are empirical schema-v2 calibrations; the
independent service-curve calibration required by the analytical backend
remains a separate TODO.

## Calibration movement

Relative to the active 2026-07-26 dual-rank profiles:

| Policy | Isolated median delta | Isolated P90 absolute delta | Full-call median delta | Full-call P90 absolute delta |
| --- | ---: | ---: | ---: | ---: |
| no-split | +1.32% | 8.63% | +0.31% | 7.77% |
| split | +0.64% | 10.41% | +0.19% | 7.81% |

The central tendency is stable. The active profiles are replaced atomically
because the current source/extension hashes differ from the old pair.

Representative best dual-rank full-call points across both policies:

| Routes/expert | Policy and shape | Wall time |
| ---: | --- | ---: |
| 1 | split `48x2T` | 9.261 ms |
| 12 | split `96x1T` | 9.593 ms |
| 24 | split `12x8T` | 14.480 ms |
| 48 | split `12x8T` | 18.701 ms |
| 192 | split `12x8T` | 39.220 ms |
| 768 | no-split `12x8T` | 130.862 ms |
| 2040 | no-split `12x8T` | 332.696 ms |

On AmazonECS8Cores, the refreshed best points are split `8x1T` at routes
1/12, no-split `8x1T` at route 24, split `8x1T` at route 48, and split `1x8T`
at routes 192/768/2040. Their full-call times are 0.796, 1.144, 2.139, 4.048,
14.066, 54.690, and 146.744 ms respectively.

## Rank lifetime

The evaluator now advances a rank with the concurrent-rank model until the
second-to-last rank completes, then maps the current phase's remaining fraction
to the matching single-rank model. Completed phases, dependencies, and call
setup are not replayed. Missing or hash/grid-incompatible companions retain the
old conservative all-ranks-active estimate.

As a profile-pair check, a 256-expert route-48 rank finishes at 15.852 ms while
a route-192 rank requires 38.843 ms under the dual-rank phase model. Switching
the latter at 15.852 ms to the measured single-rank model predicts 36.905 ms,
removing 1.939 ms or 4.99% from the previously over-derated tail.

The actual EP2 evaluator was then tested with the tiered hotspot histogram.
The old all-dual-rank estimate is 55.605 ms, with rank plans `(32,)` and
`(16,16)`. Loading the matching EP2 single-rank companions changes the compute
estimate to 54.770 ms, a reduction of 0.835 ms or 1.50%. This confirms that the
production `evaluate_ep()` exact-policy lookup activates the lifetime switch;
it is not relying on the TP4 profile across modes.

## Planner versus vLLM

The benchmark used NUMA0 CPU 0-95, split-W13, FP32 direct-route output,
ready-token merge, 5 warmups, and 20 interleaved measured runs. Every variant
was bit-exact against the production strict reference.

| Workload | Production plan | Production auto | vLLM staged | Production advantage |
| --- | --- | ---: | ---: | ---: |
| uniform | `12x8T`, strict | 17.559 ms | 18.146 ms | 3.34% |
| active-set-8 | `12x8T`, strict | 10.344 ms | 14.547 ms | 40.63% |
| active-set-16 | `(32,16,16,16,16)`, strict | 9.484 ms | 11.812 ms | 24.55% |
| active-set-32 | `12x8T`, strict | 9.237 ms | 11.913 ms | 28.96% |
| active-set-64 | `12x8T`, strict | 10.458 ms | 11.804 ms | 12.87% |
| active-set-128 | `12x8T`, strict | 12.582 ms | 13.445 ms | 6.85% |
| tiered-hotspot | `12x8T`, strict | 9.264 ms | 12.103 ms | 30.65% |
| long-short-bimodal | `6x16T`, 1T tail pool | 11.364 ms | 12.776 ms | 12.43% |
| captured seq70 | `12x8T`, strict | 15.437 ms | 16.683 ms | 8.07% |

Production wins all nine workloads. The median advantage is 12.87%, with a
3.34%-40.63% range. Cold planning is 0.622-4.462 ms and is not included in
the warm operator medians. Per-run samples and complete planner metadata are
stored under `results/data/planner_vllm_20260727/`.
