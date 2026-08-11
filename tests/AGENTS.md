# Test And Benchmark Guide

These rules apply under `tests/` in addition to the repository guide and
`../../rules/python-test.md`.

## Test Roles

- `test_*.py` files are correctness and regression tests. They must be
  deterministic and must not pass or fail based on latency.
- `bench_*.py` and profiling scripts measure performance. They must validate
  outputs before reporting candidate performance when practical.
- Keep heavyweight hardware sweeps out of default pytest collection.

## Isolation

- Use deterministic seeds and explicit dtypes, shapes, tolerances, thread counts,
  affinity, and backend selection.
- Use pytest fixtures or `monkeypatch` for environment changes and restore all
  process-global state. Do not let test order select a backend or kernel.
- Mark or skip unsupported ISA/hardware cases with a concrete reason. A skip on
  one machine does not prove the path correct.
- Do not weaken an existing tolerance to accept a candidate without documenting
  the numerical mechanism and updating the public precision contract when
  applicable.

## Coverage Expectations

- Cover typical shapes, tails/non-aligned shapes, empty/invalid inputs, fallback
  selection, and repeated invocation for changed behavior.
- Concurrency tests need bounded termination and must cover one worker plus the
  affected multi-worker topology.
- API/schema tests should verify both accepted payloads and rejected stale or
  ambiguous payloads.
- Performance tools must report command, machine, affinity, NUMA/page policy,
  warmups, samples, statistic, baseline, candidate, and relative delta.

## Validation Discipline

- Use the L0-L3 levels in `docs/change_policy.md`; a higher level supplements
  rather than replaces lower-level correctness checks.
- Run the narrowest direct test first, then broader integration tests only when
  the impact crosses modules or processes.
- Never replace a correctness assertion with a benchmark observation.
- Keep generated results and raw traces outside `tests/`; concise durable
  conclusions belong under the relevant operator's `optimizations/.../results/`.
