# Agent Benchmark Hygiene

Read this document before collecting, comparing, or reporting performance.

## Required Reporting

Record enough information to reproduce the result:

- repository commit and relevant uncommitted changes;
- machine, CPU affinity, NUMA and memory placement;
- compiler/build mode and relevant environment variables;
- operator, shape, dtype, layout, implementation, and backend;
- warmup count, sample count, synchronization, and reported statistic;
- baseline and candidate absolute values plus relative change;
- failed shapes, regressions, skipped checks, and measurement noise.

Do not mix debug and release builds, different input data, different page
policies, or different synchronization methods in one comparison.

## Evidence Retention

Exploratory scratch output may be temporary. Before a result supports adoption
or an agent handoff, preserve its required raw artifacts in an existing durable
ignored results directory or external artifact store; an OS temporary directory
must not be their only copy. Do not add large raw data to source commits.

Keep one compact run record in the existing report/manifest: exact artifact
locations (including remote host/root when needed), source/build and calibration
identity, commands, relevant protocol, decision, and missing evidence. Reuse
identity fields already required by the experiment; do not introduce new
cryptographic hashes everywhere unless requested or required by a specification.

Preserve negative results used by later comparisons too. Before retiring source,
record its recoverable implementation revision and verify that the evidence
needed to reproduce the conclusion is retained. Do not overwrite an original
artifact to repair its report. Missing old artifacts limit the claims they
support; they do not automatically block unrelated work or require rerunning
every historical experiment.

## Fused MoE Geometry

Production fused MoE benchmark and analysis work must report:

- team width and backend N tile;
- full W13 and W2 stage bytes;
- per-thread owner window for each stage;
- `(t, w13_window_tiles, w2_window_tiles, R13, R2)`.

A stage is determined by `(threads, window_tiles)`, where `window_tiles` counts
whole packed-B N tiles per thread. A team consumes
`threads * window_tiles` tiles per window and covers the stage in:

```text
ceil(total_tiles / (threads * window_tiles))
```

windows; the last may be short. `window_tiles = 0` selects the full owner
stripe and the single-window `full_n_team_stripes` geometry. Use that name only
when both stages use full stripes.

Report runtime windows as tile counts, not bytes. Byte budgets may be converted
only by `FullStageGeometry.window_tiles_from_bytes`; byte values cannot identify
a unique execution geometry. Retired byte-window and split-W13 controls must not
be reintroduced. External N-range loops must be labelled experimental and must
not emit production calibration profiles.

## Cache State Of Inputs And Weights

Repeated samples must not inherit cache state from earlier samples. Unless an
experiment names a different condition, operator and whole-plan benchmarks
measure producer-hot activations with cold weights:

- **Cold B:** rotate packed-weight copies so the bytes touched between two uses
  of one copy are at least four times the total LLC of the measured CPU set.
- **Producer-hot A:** before every timed call, an untimed producer step rewrites
  the per-call inputs (hidden states, top-k ids and weights, and any other
  operator input) on the measured CPU set from rotating source copies. Source
  copies follow the same four-times-LLC rule, so the producer reads cold data
  and leaves A resident the way a preceding operator would.
- Keep producer threads from competing with the timed call, for example with
  passive OpenMP waiting, and exclude the producer from the timed interval.

Cold A, A left over from earlier calls, and scrub-based eviction are not
defaults. Cold A may be measured only as a named condition, such as for
kernel-state calibration or a sensitivity check. A scrub must cover every LLC
domain of the measured CPU set, not only the scrubbing thread's domain.

Report the A and B states, copy counts, total copy bytes relative to the LLC
and the producer step. Results collected under another state keep their
original label and are not mixed with producer-hot results in one comparison.

## Allocator State For Fused MoE Runs

Fused MoE runs on Arm-codex preload jemalloc with purging disabled, matching the
intended deployment:

```bash
LD_PRELOAD=/usr/lib64/libjemalloc.so.2 \
MALLOC_CONF=oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1
```

The library path is per machine: `/usr/lib64/libjemalloc.so.2` on `Arm-codex`,
`/usr/lib/aarch64-linux-gnu/libjemalloc.so.2` on Amazon C9g. Resolve it on the
machine rather than copying a path between them.

The Plan V2 runtime allocates its FP32 route output per call. Under glibc
defaults with THP `always`, or jemalloc with default decay, that memory is
returned between calls and every call page-faults and zeroes it again (15-35%
of full-load time; `tmp/dram_write_20260919/decision.md`). Record `LD_PRELOAD`,
`MALLOC_CONF` and the allocator libraries mapped by the process, and refuse to
run when jemalloc is not mapped. Results collected under the glibc default keep
that label and are not mixed with jemalloc results in one comparison.

## Single-Thread Benchmarks

Bind each process to one dedicated core and constrain dependent libraries:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c <core> <command>
```

On `Arm-codex-internal` / `Arm-codex`, independent benchmark cores are `0`,
`80`, `160`, and `240`. Use one process per core and report the selected core.

On `AmazonECS8Cores`, use one process per core from `0` through `7`. Example:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskset -c 0 .venv/bin/python tests/bench_microkernel_qkt.py'
```

## Machine Sharing And Other NUMA Nodes

`Arm-codex` has four 80-core NUMA nodes (node0 `0-79`, node1 `80-159`, node2
`160-239`, node3 `240-319`) over two sockets, so nodes 2 and 3 share a socket.
Measurements run on node3 with `--physcpubind=240-319 --membind=3`; other work
often runs at the same time. What that costs was measured on 2026-09-20 with the
unchanged plan benchmark on node3 (22 points, two sessions per condition,
`tmp/numa_interference_20260920/decision.md`), background placed only on the
other nodes:

| background on other nodes | node3 plan time |
| --- | --- |
| BF16 matmul, one node at 3.9 TFLOP/s | +0.01% (node2), +0.24% (node1) |
| memory streaming, one node at 266 GB/s | +0.83% (node2), +1.84% (node1) |
| memory streaming, three nodes | +13.1% |

Rules that follow:

- Compute-bound work - builds, tests, planner searches, analysis - may run on
  nodes 0-2 during a measurement on node3.
- Memory-streaming work must not overlap a measurement, wherever it is placed:
  large copies, dataset generation, archive extraction, another MoE benchmark.
  Sequence it before or after, or accept and report the contamination.
- The idle guard must watch the 5-minute load average as well as the 1-minute
  one, and a chain waits out a fixed cooldown before its first measured session.
  A 1-minute average is too fast a filter to separate a session from the work
  immediately before it: on C9g a session admitted at 1-minute load 0.39, with
  the 5-minute average still at 3.10, came out 0.54% off a clean baseline whose
  own session-to-session spread is 0.1%, and it was the run that had just
  finished 27 s earlier - our own - that it had not been separated from. The
  guard means "no foreign work of unknown kind" and "not immediately after our
  own work"; it never means "other nodes are harmless". `scripts/wait_idle.sh`
  implements it:

  ```sh
  source scripts/wait_idle.sh
  wait_idle            # 1-min <= 2 and 5-min <= 4, held for 60 s, giving up after 30 min
  ```

  It returns nonzero on timeout so a chain records the failure rather than
  measuring through it.
- A measurement whose cores are shared with a foreign job is not recoverable by
  any of this. The E6 r022 sessions measured 79-100 ms against 27-28 ms (+190%),
  far beyond anything foreign nodes cause, and were discarded.

Node-3 core frequency stays at its 2900 MHz maximum under the heaviest of these
backgrounds, so the effect is memory-path contention beyond the node, not power
or frequency; the reason node1 (other socket) costs more than node2 (same
socket) is not identified.

Repeated on Amazon C9g on 2026-09-22 - two nodes of 96 cores, one 96 MiB LLC
instance per node, one socket - with the full-load grid on node 0 and the same
backgrounds on node 1
(`results/numa_interference_c9g_20260922.md`):

| background on node 1 | node-0 grid time |
| --- | --- |
| BF16 matmul, 96 threads | -0.06% and +0.15%, 4 of 12 cells slower |
| memory streaming, 96 threads | +1.06% and +0.90%, 12 of 12 cells slower |

The rule above holds on both machines and the per-node streaming cost is the
same to within the measurement, although C9g shares one LLC slice among 96 cores
where `Arm-codex` shares one among 40, and carries 2-4x the per-domain weight
footprint. Cross-node interference therefore lives on the path between nodes,
not in the last-level cache, and is a different mechanism from the footprint
effect that governs window value.

Session-to-session reproducibility, measured on C9g from four identical idle
sessions run back to back: per-cell spread median 0.49% and at most 1.14%, with
session medians inside 0.1%. Treat a difference below that as unresolved
regardless of how many runs a single session averages - within one session the
same cell reproduces to about 0.1%, which is not the same quantity.

## Sanity Checks

Before comparing GFLOP/s with peak, read
`docs/agent_performance_references.md`. If measured throughput exceeds the
relevant peak, first verify thread count, OpenMP/runtime behavior, timing scope,
and the FLOP formula.

Performance conclusions require measured data. A successful build, profiler
estimate, analytical prediction, or one favorable sample is not a speedup.

## Calibrating From Service Probes

Never build a calibration from a single service-probe run. Run the probe at
least three times and pass every run to
`cost_model/build_analytic_calibration.py`, which combines them point by point
with the median; a single path still works and warns.

```sh
for i in 1 2 3; do
  .venv/bin/python cost_model/profile_analytic_services.py --output probe_r$i.json ...
done
.venv/bin/python cost_model/build_analytic_calibration.py probe_r1.json probe_r2.json probe_r3.json ...
```

This is not caution about noise in general. Some service points are not
reproducible while their neighbours are, so the usual "repeat it and look at the
spread" does not protect a chain that reads one run. Measured on Amazon C9g
(2026-09-22), three consecutive runs on an idle machine:

| resource | spread across three runs |
| --- | --- |
| `gemm_core_flops` at 4T | 47.4% |
| `gemm_core_flops` at 96T | 45.6% |
| `matrix_flops` at 1T | 46.3% |
| `l2_bytes`, every width | 0.4-1.9% |
| `dram_bytes`, `llc_bytes` | under 2% |

One of those three runs produced a degenerate calibration: every fitted overhead
zero, training MAPE 20.3% against 10.2-10.7% for the other two, and the
planner's own region at 7.8% against 2.3%. Nothing in the chain's output says
which run you got - the degenerate calibration builds cleanly and reports
plausible service rates.

Two consequences for reading older evidence:

- A calibration asset built before this rule is one sample. It may be a good
  one - C9g's 2026-09-21 asset reproduces the median-of-three fit to within the
  run-to-run scatter - but that is luck, not procedure. Re-derive before
  relying on one quantitatively.
- Comparing raw service rates between two runs or two machines is the wrong
  acceptance test. C9g's `l2_bytes` is reproducibly 13-21% higher on a rebuilt
  instance, and that difference does not reach the fitted model at all: the
  fitted expert-fixed cost, per-route cost and stage scale all land inside
  their own sample-to-sample scatter. Compare the fitted parameters and the
  residuals in the planner's region, not the probe.

## Remote Jobs And Event Continuation

For new long remote runs use `scripts/run_experiment_notify.py launch-remote`.
The remote helper detaches the job with its own timeout and exclusive job-state
path. The local monitor checks that same job id using ordinary program-level
status requests; no model calls are needed for these requests. A lost SSH reply
never resubmits the experiment. Remote terminal results, transport availability,
and queue delivery are separate records.

Example (replace the unique paths and command with the frozen experiment spec):

```sh
.venv/bin/python scripts/run_experiment_notify.py launch-remote \
  --state-dir tmp/RUN/monitor \
  --current-state optimizations/fused_moe_sve/CURRENT.json \
  --thread THREAD --label LABEL --result-location REMOTE_RESULTS \
  --host Arm-codex-internal \
  --remote-state /home/zhangxu/codex/fused_cpp/tmp/RUN/job \
  --remote-cwd /home/zhangxu/codex/fused_cpp \
  --remote-python /home/zhangxu/codex/fused_cpp/.venv/bin/python \
  --timeout-seconds 4800 -- bash path/to/frozen_run.sh
```

The monitor uses a 30-second program-level interval by default and a 120-second
continuous-unavailability recovery window. On terminal status it queues one
result event. If remote state cannot be recovered within that window, it queues
one `remote_state_unknown` event; this is not experiment failure. An unknown
submission is queried, never replayed. Native Linux PID start-time/boot identity
is checked for running workers. Process absence without a terminal result stays
unknown, not success.

```sh
# Read the single task entry and its referenced run status; no remote call.
.venv/bin/python scripts/run_experiment_notify.py status optimizations/fused_moe_sve/CURRENT.json
# Recover monitoring only. Never rerun the experiment command.
.venv/bin/python scripts/run_experiment_notify.py resume tmp/RUN/monitor
```

Resume uses an OS lock to avoid concurrent monitors in one state directory. After
an unknown terminal observation it creates a new recovery-monitor record pointing
to the same remote job. Prior observations remain immutable. A confirmed terminal
job is not restarted or re-notified by resume. Ambiguous queue delivery is not
automatically retried. Queue acceptance proves submission, not model execution.

`CURRENT.json` owns current task phase, active run pointer, latest decision and
next action. Launch/monitor update phase and run pointer atomically; decision and
acceptance fields are updated at analysis/handoff. An active or unresolved run
cannot be replaced by another launch through the same current entry. Immutable
config/design files describe the original run, never its current progress.
The single-entry model intentionally serializes experiments for this task;
parallel jobs require an explicit resource/scheduling design, not extra hidden
current files.

When phase is `waiting_external`, finish independent useful work and yield.
Do not run repeated model-driven `ps`, sleep or kqueue turns, or send repeated
"still waiting" updates. Resume analysis on a result/fault event or user request.
Repository code cannot pause the client's separate Goal auto-continuation.
If that trigger remains enabled, disclose the limitation once; do not claim
zero waiting-time model calls, and do not mark research complete/blocked merely
to suppress continuations. Client-side scheduling must provide the event-only
behavior; no unsupported Goal API is called by this script.

The old `launch --transport ssh` interface remains for compatibility with old
run records. It tracks SSH process termination, not durable remote job lifetime,
and `resume` will not automatically migrate such jobs. Recover those by checking
the original remote process and terminal artifacts; never blindly restart.

The agreed cost-model scope/machines retain the user's <=4h authorization.
Timeouts, retries and the whole experiment budget must be included. Local host
shutdown still stops monitoring; `resume` is explicit recovery, not a durable
cross-host service. The remote task can survive the SSH client disappearing but
not remote host shutdown. Do not put secrets in argv or stored run configs.

## Experiment Decisions And Validation Cost

Before collection, state the competing explanations, which observation changes
the next action, and when this line of investigation stops. Choose repetitions
and pilot scope for the uncertainty being resolved; 31 rounds is a protocol
choice, not a universal default. Once frozen, do not lower gates after seeing
results. Batch implementation and its focused validation into a reviewable unit;
additional tests or artifacts need a concrete failure mode or decision they help
resolve. Source/native identity and raw integrity matter; do not duplicate hashes
and progress narratives in every document.

### Notification permissions and receiver receipts

A detached child inherits its launch sandbox. A supervisor launched inside the
workspace sandbox cannot necessarily write Codex's state database: the real
paused-Goal probe failed with `attempt to write a readonly database`. Detaching
alone does not grant notification permissions. Launch the supervisor through the
approved host execution path when using the real `codex queue`; do not change
Codex database permissions, copy its database, or silently bypass approval.
Validate the actual queue and receiver on that execution path before relying on
unattended continuation. Fake-queue tests do not establish delivery capability.

Each notification carries a stable event ID derived from its run path, target
thread, and immutable completion record. The receiving turn must run:

```sh
.venv/bin/python scripts/run_experiment_notify.py receive STATE --event-id ID --thread THREAD
# Only `claimed` starts handling. After handling and recording the conclusion:
.venv/bin/python scripts/run_experiment_notify.py acknowledge STATE --event-id ID --thread THREAD
```

`received.json` records exclusive receipt; `processed.json` records completed
handling, not experimental acceptance. `status` exposes both. Duplicates return
`already_received` or `already_processed` and must not repeat work. Delivery and
receipt use separate OS locks. A crash after claiming requires explicit recovery
of the original analysis; no automatic lease expiry or replay of side effects.
These receipts provide deduplication, not a transaction over arbitrary analysis
and external actions. Preserve the original failed delivery evidence; do not
blindly retry an ambiguous send. A new probe may verify a repaired execution path
without replaying the experiment or pretending the earlier delivery succeeded.

### Reusing notification evidence before a long run

`launch-remote` and `resume` check notification health before starting a monitor
or submitting work. A fresh local short probe records the launching thread,
host, Codex executable hash, Codex home, runner hash and health-checker hash in
its immutable config. After the event is actually received and acknowledged,
register it in the single current entry:

```sh
.venv/bin/python scripts/run_experiment_notify.py notification-health \
  --current-state optimizations/fused_moe_sve/CURRENT.json --probe-state tmp/PROBE
```

`CURRENT.json.notification_health` links the evidence, its context and the latest
preflight result. An old probe without a launch fingerprint cannot be retrofitted
as current evidence. The gate verifies successful completion, accepted delivery,
matching receive/processed receipts and unchanged context. It also creates and
removes a temporary file in the Codex state directory under the actual launching
permissions; no Codex database is modified by this permission check. Run the
launcher through the approved host path. A sandbox permission failure records a
failed preflight without discarding the previously verified host evidence.

Configuration or notification-code changes require a fresh acknowledged short
probe. Unchanged evidence is reused; the check itself sends no messages and
starts no model calls. Actual delivery failure marks notification health failed.
This is a launch-time guard, not a guarantee against a later client outage,
database-specific failure, or filesystem permission change. Goal scheduling
remains a separate client control; this check does not pause or resume Goals.

## Compact Result Review And Decision Handoff

Use `docs/templates/experiment_decision.md` for one decision question. Freeze
competing explanations, discriminating observations, quality gates and stopping
conditions before collection. Label retrospective triage separately; do not
promote an inspection threshold to an acceptance gate after seeing results.

For the existing `receiver_small_middle_demand` v1 records:

```sh
.venv/bin/python scripts/summarize_demand_run.py build \
  --root tmp/joint_cost_model_20260911/receiver_small_middle_demand \
  --spec tmp/p2_demand_review_20260916/spec.json --out tmp/NEW_REVIEW
.venv/bin/python scripts/summarize_demand_run.py show \
  --bundle tmp/NEW_REVIEW --id A0049
```

The adapter verifies raw/prediction hashes, expected grids, duplicate cells,
round coverage and recorded quality gates; exit code 2 means a failed quality
check, not an acceptable model. Malformed/missing inputs fail rather than being
silently skipped. It reuses existing PMU validation and is not a replacement for
the raw collector's full protocol/PMU-running-fraction validator.

Read `review.md` first. `summary.json` contains grouped errors and all checks;
`anomalies.json` indexes exceptions by source path and JSON pointer. `show`
expands one source record after rechecking its hash. Statistics use only eligible
comparisons and retain signed errors, negative observations and unavailable
relative errors without clipping. The current adapter measures request MiB/call,
not runtime prediction, joint interference or isolated tail cost. Other dataset
schemas need explicit adapters; do not infer their units or validation role.

After review, record the decision and exceptions once, then put only the decision,
next action and evidence links in CURRENT.json. The manifest owns mechanism
lifecycle, the summary owns computed statistics, and the decision record owns
interpretation. Avoid rereading raw data or repeating validated analyses unless
an anomaly, changed input or new claim justifies it. Do not silently auto-adopt a
model or launch a follow-up merely because collection or summarization succeeded.
