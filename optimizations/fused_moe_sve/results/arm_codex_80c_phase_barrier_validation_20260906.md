# DDR phase contention: controlled barrier-density validation

Status: two synthetic sessions completed; small positive high-density effect,
not an explanation of multi-percent MoE residuals and not a calibration input.

## Question and claim boundary

Does repeatedly synchronizing streams with equal address phase cause a measurable
penalty that is reduced by distributing their starting phases? This is an
independent Lab diagnostic, not a change to packed weights, MoE kernels, frozen
v8, planner selection, or production defaults. A positive synthetic result would
justify a separate MoE-context validation, not a new calibration coefficient.

The predecessor is
`arm_codex_80c_order_width_selection_and_ddr_interleave_20260906.md`.
Its Python thread-pool experiment did not enforce simultaneous worker starts,
did not scrub between cells, and changed the address subsets. Its reported
2.38% is exploratory, not a baseline for this deciding test. The inferred
4 KiB / 64 KiB channel mapping remains a hypothesis: virtual address residues
are recorded as address geometry, not asserted physical channel identities.

## Controlled contrast

Use native pinned workers and a fixed total read volume. Keep worker count,
instruction path, allocation, address set, barrier count, and traversal length
identical across three arms:

- `aligned`: identical zero relative offset for every worker.
- `common_shift`: every worker receives the same nonzero offset. This is still
  aligned, but checks effects of nonzero offset and cyclic traversal.
- `dispersed`: workers receive different 4 KiB offsets spanning a presumed
  64 KiB period. Each reads its entire original range exactly once by cyclic
  traversal; it must not gain by omitting addresses or reading fewer bytes.

Scan chunk counts 1, 8, 32, 128, 512, with the same barriers in all arms at
each count. The intended main configuration is 16 workers, 32 MiB per worker,
four copies (2 GiB total target data), and a disjoint LLC scrub before every
cell, outside the counter and timing scope. All chunks are at least 64 KiB.
Use the same copy for arms of a paired round and rotate copies across rounds.
Randomize arm order and chunk-count order with a saved seed. Retain warmups
separately from 31 measured paired rounds. Repeat in two independent processes.

Target: Arm-codex-internal, NUMA3 memory, CPUs 288–303 unless the recorded
preflight finds them unavailable. Fail on affinity errors. Do not modify THP,
HugeTLB pools, other workloads, or machine-wide settings. Record actual memory
backing where observable; THP policy alone is not proof of huge-page backing.

## Correctness and measurement checks

Before a deciding run:

- Verify the traversal covers the same byte/line set once in every arm, including
  wraparound and all chunk counts, and validate checksums on nonconstant data.
- Test argument rejection and affinity failure; avoid swallowed worker errors
  or barrier deadlock. Preserve production build isolation.
- Measure a no-load synchronization control and worker release skew so a
  barrier-dominated regime can be identified instead of interpreted as memory
  pressure. Do not mechanically subtract control time from loaded time.
- Collect all available DDRC instances on both SCCLs 25 and 27; save device
  names, event encodings, raw per-cell values and enabled/running times. Do not
  silently turn partial PMU coverage into a full-channel claim.
- Record DRAM read volume per arm. Equal requested bytes do not guarantee equal
  DRAM traffic: cache/prefetch differences remain a confound.
- Check for overlapping memory-intensive work across the entire NUMA3 memory
  fabric, not just the selected cores. Never stop another user's job.

Preflight found an existing job on CPUs 240–247, which shares node3 memory
resources. Its activity must be resolved before declaring a clean hardware
session; low host load or disjoint worker affinity is insufficient.

## Decision rule

Report absolute time, paired gain distribution and uncertainty for both
`dispersed` versus `aligned` and `dispersed` versus `common_shift`, separately
for every chunk count and session. Report all counts, including regressions;
do not choose a winning count and call it the whole result. Save individual
rounds, not only medians.

Evidence supporting repeated phase contention requires reproducible benefit
against both controls, increasing benefit over at least a non-barrier-dominated
range as synchronization density increases, and no material traffic mismatch
or overlapping-workload explanation. Strict monotonicity is not required at
the barrier-dominated endpoint. Confidence intervals and cross-session
repeatability matter more than a single positive median.

No reproducible benefit in valid cells argues against this mechanism for this
probe configuration. It does not prove all MoE phase effects impossible.
Traffic mismatch, excessive release skew, incomplete counters or interfering
workload makes the mechanism attribution inconclusive, not positive or negative.
Queue-counter evidence, if available and validated, is corroboration; aggregate
channel balance alone neither proves nor disproves transient queue concentration.

## Execution record

The user subsequently lifted the terra-medium-only restriction and requested
direct implementation and execution by the main agent. The subagent was stopped.
Exact build/run commands, source identity, raw artifact locations,
correctness outcomes and measured results will be appended after review and
execution. The completed measurements are recorded below.

### Pilot corrections (before deciding sessions)

The apparent cpufb workload was an old waiting shell; a fresh process inspection
found no running cpufb compute process and no active high-CPU user job. Nothing
was stopped. Full-session cleanliness is still bounded by preflight observation.

The first native pilot used pthread barriers and showed 7–16 us initial release
skew, with a 6.29 ms no-load control at 512 chunks. This is not adequate evidence
against a short-lived phase effect. It is retained as `pilot.jsonl`; the deciding
candidate uses an identical acquire/release atomic barrier in all arms, with a
fresh smoke/pilot before measurement. This is a protocol correction for release
jitter, not a selected performance result. Native times include per-chunk clock
instrumentation consistently across arms; barrier wait includes load imbalance
and cannot be subtracted as pure synchronization overhead.

The probe reads one byte per 64-byte line (512 MiB line footprint per cell), not
all payload bytes or MoE SIMD instructions. Cyclic traversal covers exactly the
same cache-line address set in each arm. Four allocation copies are used; the
next, distinct 512 MiB copy scrubs before every cell. The memory-policy observation
and `/proc` mapping snapshots are included in JSONL metadata. Offsets are fixed
within each process; independent allocations/sessions do not exhaust physical
address or channel-map uncertainty. PMU intervals include IPC/dispatch whereas
native elapsed time excludes it; no-load counters expose that additional traffic.

## Completed result

Each cell below uses 31 paired rounds after five warmups, independently in each
process. Positive gain means dispersed is faster; gains are medians of paired
ratios, not ratios of the separately reported medians.

| Chunks | S1 aligned / dispersed ms | S2 aligned / dispersed ms | S1 gain vs aligned / common shift | S2 gain vs aligned / common shift |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.34827 / 3.34846 | 3.35123 / 3.35065 | −0.022% / −0.080% | −0.052% / −0.086% |
| 8 | 3.35879 / 3.36026 | 3.36489 / 3.36107 | −0.053% / −0.039% | +0.097% / +0.065% |
| 32 | 3.38834 / 3.37972 | 3.39354 / 3.38017 | +0.250% / +0.160% | +0.266% / +0.292% |
| 128 | 3.43767 / 3.42481 | 3.44461 / 3.42603 | +0.332% / +0.402% | +0.433% / +0.411% |
| 512 | 3.60660 / 3.59434 | 3.60725 / 3.58586 | +0.395% / +0.408% | +0.337% / +0.425% |

Descriptive 10,000-resample paired-median bootstrap 95% intervals at 32/128/512
are above zero for both controls in both sessions. They are not adjusted for
multiple comparisons or serial correlation, and do not constitute a causal test
by themselves. At 512, aligned-reference intervals are [0.334%, 0.461%] in S1
and [0.235%, 0.539%] in S2. There is a rise from low to moderate barrier density,
not a strictly monotonic rise through every count in both sessions.

Measurement checks and limitations:

- All measured per-worker checksums match the deterministic full-line reference,
  for all three arms, five counts and four copies; PMU running ratios are at
  least 99%. Both SCCLs and all 16 discovered DDRC PMU devices were collected.
- Paired DRAM byte ratios are within about 0.11% of unity at the median. Total
  observed reads are about 481 MiB for a 512 MiB footprint: do not claim all
  lines missed LLC or that cache effects have been eliminated completely.
- Initial release-skew medians are about 0.4 us, P90 up to 1.39 us. Per-chunk
  release skew was not saved, so repeated ideal phase alignment is not proven.
- No-load time is about 0.32–0.37 ms at 512 chunks versus loaded 3.59–3.61 ms;
  synchronization is material but no longer the dominant elapsed cost. Atomic
  barrier traffic and clock instrumentation are part of this synthetic workload.
- The sum of DDRC occupancy divided by sum of commands at 512 decreases from
  49.46 to 48.74 in S1, and 49.38 to 48.27 in S2 (aligned to dispersed). This
  supports a queue-related interpretation only conditionally on the PMU event
  semantics and instance mapping; it is not independently validated latency.
- S2 has a common slow regime: low-count P90 times reach about 3.86 ms versus
  roughly 3.35 ms medians. Pairing reduces this confound but does not identify
  its cause. No within-session workload monitor was collected; pre/post process
  checks do not certify continuous system idleness. Do not suppress these rounds.
- The probe is a 16-worker line stream with cyclic wraparound and fixed offsets,
  not packed W13/W2 execution. Instruction mix, prefetch, physical mapping,
  core count, and MoE barrier behavior remain external-validity gaps.

Decision: the result is consistent with a small phase-sensitive effect when
streams are frequently resynchronized. It does **not** establish that phase
competition explains the earlier large residual, nor justify a physical model
parameter. A follow-up, if requested, should reproduce the contrast in a
representative MoE phase while preserving packed format and equal bytes, with
better physical-phase/release diagnostics. Do not enlarge the absolute model
using this synthetic effect alone.

## Reproduction and validation record

Main checkout HEAD at execution: `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`.
Only the new Lab wrapper/native source plus unchanged `linux_perf_event.py` were
copied to an isolated remote directory; existing uncommitted production/planner
changes were not synced or used by this standalone test. Compiler: GCC 13.2.0,
`-O3 -std=c++17 -pthread -Wall -Wextra`. No extension build or calibration used.

Remote directory:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/phase_barrier_20260906_root/`.
Local ignored data: `tmp/phase_barrier_20260906_root/`.
Raw deciding artifacts: `session1.jsonl`, `session2.jsonl`; descriptive analysis:
`summary.json`. JSONL metadata retains source/binary hashes, event config,
NUMA maps and memory backing snapshots. The pilot observed about 2,088,960 KiB
AnonHugePages against 2,097,516 KiB anonymous memory, not merely THP policy.

Native binary SHA256:
`ced8b5c700a27c0b1a28b0ba1abba4b28d89d26d29ec91608f4d18c97dde990e`.
Raw session SHA256 values:

- S1: `180600f370a87c9cccde38daecce03c02bae732b591c507f2a2285222df4acd0`
- S2: `0ff24a1ae3995883e9ef0a58236dff92392859376a8158210b4bdcd45e02ced5`

From the remote directory (use fresh output names on reruns):

```bash
g++ -O3 -std=c++17 -pthread -Wall -Wextra ddr_phase_barrier_native.cpp -o phase_native_spin
timeout 180 numactl --membind=3 /home/zhangxu/codex/fused_cpp/.venv/bin/python \
  bench_ddr_phase_barrier.py --native-bin ./phase_native_spin \
  --seed 20260906 --output session1.jsonl
timeout 180 numactl --membind=3 /home/zhangxu/codex/fused_cpp/.venv/bin/python \
  bench_ddr_phase_barrier.py --native-bin ./phase_native_spin \
  --seed 20260907 --output session2.jsonl
```

Local commands:

```bash
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_moe_ddr_phase_barrier.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_ddr_phase_barrier.py \
  tmp/phase_barrier_20260906_root/session1.jsonl \
  tmp/phase_barrier_20260906_root/session2.jsonl \
  --output tmp/phase_barrier_20260906_root/summary.json
```

Final local tests: 15 passed (including paired-analysis and duplicate-cell checks).
Ruff and `git diff --check` passed; manifest YAML parsed successfully.
Native two-worker no-PMU smoke passed; both
full spin-barrier sessions also enforce per-cell checksum and PMU checks.
These are standalone Lab correctness/performance checks, not MoE integration or
production adoption tests. No commit or production/default/model edit was made
for this phase experiment.
