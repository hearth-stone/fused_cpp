# mHC OpenMP boundary experiment

## Decision

The measured default gate rejected both changes, but the user explicitly chose
the unified lower-boundary design. Team reuse and sqrsum/projection `nowait`
are therefore enabled inside the explicit SVE candidates. Public Torch
dispatch remains unchanged; the regressions below are accepted known risks.

## Team reuse

The candidate reused one team while preserving separate static work-sharing
loops for control rows, Sinkhorn blocks and pre-apply rows. Its projection also
executed residual sqrsum and GEMM ownership in one team with `nowait` between
the independent loops. The existing staged native entrypoints and candidate
were measured in the same process with alternating AB/BA order.

| Threads | Staged, ms | Team reuse, ms | Change |
| ---: | ---: | ---: | ---: |
| 1 | 40.670 | 40.710 | +0.1% |
| 8 | 6.334 | 6.417 | +1.3% |
| 16 | 3.368 | 3.387 | +0.5% |
| 32 | 1.916 | 1.954 | +2.0% |
| 64 | 0.990 | 0.991 | +0.1% |
| 80 | 0.908 | 0.842 | -7.3% |

Only the 80-thread point improved. A fixed thread threshold would be specific
to this machine and shape, while the lower-width regressions fail the adoption
gate. All three Pre outputs were bit-identical.

## Final-head nowait

The final-head sqrsum loop used `nowait`, allowing a thread to enter its M12
projection ownership before all sqrsum rows completed. Separate-build medians
were compared with the immediately preceding baseline:

| Threads | Barrier, ms | Nowait, ms | Change |
| ---: | ---: | ---: | ---: |
| 1 | 55.076 | 58.169 | +5.6% |
| 8 | 8.404 | 8.297 | -1.3% |
| 80 | 1.917 | 1.948 | +1.6% |

The result is not a stable improvement. Overlapping two residual readers may
also increase cache/refill contention; this is an accepted tradeoff under the
explicit user decision.

## Validation

- Machine: `Arm-codex-internal`, NUMA3.
- Shape: `T=2048`, `C=4`, `H=4096`.
- Placement: CPUs beginning at 240, memory bound to NUMA3.
- Runtime: Torch bundled libgomp, `OMP_DYNAMIC=FALSE`, close core binding.
- Statistic: five warmups, 21 median samples.
- Correctness: 39 tests passed on the experimental SVE256 build.

## Enabled-build verification

After removing the comparator and making the lower-boundary path unconditional,
SVE128 and SVE256 each passed all 39 tests. The SVE256 enabled build measured:

| Threads | Pre, ms | Post-Pre, ms | Final Head, ms |
| ---: | ---: | ---: | ---: |
| 1 | 41.393 | 58.411 | 56.299 |
| 8 | 6.812 | 9.955 | 8.360 |
| 80 | 0.844 | 2.310 | 1.983 |

The remote build was restored to SVE256 after validation.
