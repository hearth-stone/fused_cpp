# Fixed smooth/uneven plan pair under external memory pressure

Status: completed. Smoother wins at all three levels, but its advantage narrows
as external reader pressure rises in both sessions. Lab-only. Initial permission
failure and recovery are retained below as history.

## Result and decision

The pressure ladder reproduced in both formal sessions: reader-only flux was
about 0.011, 27.7 and 66.2 GB/s for 0/4/16 readers. Loaded DDRC flux and
occupancy/command increased with reader count for both plans. Thus the
intervention increased observable pressure, not just a nominal label.

| Readers | S1 smooth / uneven ms | S2 smooth / uneven ms | Paired gain median S1 / S2 | P10 S1 / S2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 30.27499 / 31.84474 | 30.40919 / 31.86208 | **5.259% / 4.780%** | 4.794% / 4.355% |
| 4 | 31.13149 / 32.63247 | 31.24238 / 32.62048 | **4.846% / 4.461%** | 4.262% / 3.995% |
| 16 | 33.03767 / 34.11655 | 33.07415 / 34.13986 | **3.239% / 3.285%** | 2.914% / 2.846% |

Wins are 31/31 at each of the six level/session cells. The trend is decreasing,
not increasing. It is not just a larger denominator: differences between separate
plan medians also narrow from 1.45–1.57 ms at zero readers to 1.07–1.08 ms at
16 readers. Those differences are not paired confidence intervals.

Loaded occupancy/command diagnostic (smooth / uneven):

| Readers | Session 1 | Session 2 |
| ---: | ---: | ---: |
| 0 | 107.19 / 128.67 | 110.26 / 129.24 |
| 4 | 117.57 / 137.99 | 120.32 / 138.27 |
| 16 | 139.76 / 155.02 | 141.17 / 156.12 |

Loaded read flux was about 154–155/147–148 GB/s (smooth/uneven) without readers
and 193–194/187–188 GB/s at 16 readers. These are aggregate rates, not victim-only
traffic or hardware-peak comparisons. Lower occupancy/command for the smoother
plan is consistent with less waiting, but does not verify its predicted temporal
pressure curve or independently identify DDR versus remote-interconnect effects.

Conclusion: this does **not** support a growing smoothness benefit over the
tested extra-pressure range. It does not disprove a low-to-overload threshold
elsewhere: zero readers still means the full 80-core MoE workload, not a certified
absolute-low-pressure endpoint. This is one fixed high-skew pair under remote
reader interference. Do not fit a pressure threshold or launch another sweep
from this result alone.

All four copies passed bitwise checks at every level in both sessions. All 48
event counters met the >=99% running-ratio check. Each of the 16 large reader
arrays was verified on N3 in both sessions. Post-run process inspection confirmed
all experiment reader/spawn processes exited. No further OS security setting
was changed by the agent. Four local tests and Ruff/diff checks passed.

Raw session SHA256:

- S1: `2e29b7dc3c829151ffc802213c054f59d2b33c5d7e946f90890818efc9e9921b`
- S2: `f927fe12641f152ced7b2723860e7a152b2ab0ab44c49275aacc6aba4bcef9ff`

Both sessions match the frozen frontier, runner and extension identities.
Local/remote raw records and `summary.json` are retained in the directory below.
Analyze with `analyze_density_pressure_sweep.py --sessions <session1.json>
<session2.json> --output <summary.json>` using fresh output paths.

## Initial execution blocker (resolved)

The target confirms node2 CPUs160–239 and node3 CPUs240–319. The pilot started
the 16 reader processes and reached PMU initialization, but
`perf_event_open(hisi_sccl25_ddrc0_0.flux_rd)` failed with `PermissionError: EACCES`.
Read-only diagnosis found `kernel.perf_event_paranoid=2` and `sudo -n -l` requires
a password. No sysctl, capability, permissions or machine-wide security setting
was changed. The `finally` cleanup stopped all reader processes; follow-up `ps`
showed no surviving experiment reader/spawn processes.

The pilot failed before the background-only calibration or MoE allocation/timing,
so `pilot.json` was not created. Do not treat this as a performance result or run
the formal comparisons without the predeclared pressure validation. Required
external action: an administrator must authorize this benchmark's uncore PMU
reads, e.g. execute the scoped benchmark with appropriate privileges. Passwords
must not be supplied in chat or embedded in scripts.

Local focused checks: three tests passed (invalid reader levels, counter units,
pilot-vs-formal distinction); Ruff and diff checks passed. Native MoE checks and
the actual pressure ladder remain unrun due to this blocker.

### Recovery and pilot

After the user changed the machine configuration, read-only inspection reported
`perf_event_paranoid=-1`. The agent did not change or restore this setting.
Retrying the same pilot succeeded, including all foreground-copy and background
checksum checks and PMU running-ratio checks. All 16 reader arrays had 65,536
anonymous pages on N3 (256 MiB each), with `bind:3`; no misplaced large array
was found in the recorded NUMA maps.

Background-only flux for the two 400 ms pilot samples:

| Active readers | Sample 1 GB/s | Sample 2 GB/s |
| ---: | ---: | ---: |
| 0 | 0.0120 | 0.0098 |
| 4 | 27.948 | 27.609 |
| 16 | 66.236 | 66.209 |

The predeclared traffic-ladder prerequisite passes. In the pilot the loaded
occupancy/command ratios also rose from 0 to 16 readers for both plans, but
single pilot timings are not used for the formal performance conclusion.
Two full sessions are run with the unchanged reader counts, plans and protocol.

## Scope and identifiability

Use the previously frozen high-skew GEMM-density pair unchanged (request008,
layer38). The foreground uses all 80 node3 cores, so local extra reader threads
would introduce CPU oversubscription. Instead, launch 16 persistent reader
processes on node2 CPUs160–175, each with a disjoint 256 MiB array allocated
under inherited node3 memory binding. Activate 0/4/16 readers, retaining all
allocations across levels. No user process is stopped and no OS tuning is changed.

These are nominal **no extra / medium extra / high extra** pressure levels,
not certified absolute low/medium/high DRAM utilization. A no-background MoE
workload may already saturate memory. Remote readers also contend on interconnect
and potentially other shared paths; this is not a pure local-DDR-cap intervention.
Do not interpret a result as proving the entire low-to-saturation threshold curve.

Measure all 16 node3 DDRC PMUs (SCCL25+27), read flux, command count and occupancy.
Two background-only 400 ms samples per level verify the load ladder. Save every
reader's NUMA map and fail affinity/worker startup errors. Proceed from pilot
only if background-only node3 read traffic distinguishes the levels; otherwise
stop this hardware-dependent contrast rather than rename identical levels.

## Protocol and criteria

High-skew only, two fixed executable plans at three reader counts, two independent
foreground process sessions. Randomize levels within each round and plan order
within each level. Pair by level and round, same packed-weight copy. Fixed tensor
seed, four copies, disjoint 216 MiB scrub before each measured call, five warmups
and 31 effective rounds per session. Set reader count, await acknowledgement and
settle 50 ms outside timing before each level. Reader checksum runs continuously.
Check both MoE outputs bitwise on every copy at all three levels before timing.

Foreground: NUMA3 CPUs240–319, E256/H4096/F512, BF16 SVE N tile8, full stripes,
4x16T+16x1T, W13/W2 window tiles0 and R13=R2=1. Each full expert's stage bytes
are W13 8 MiB/W2 4 MiB; per-thread footprints divide by the existing team width.
No widths, tasks, routes, ordering, calibration or extension changes between levels.

Primary gain is paired `100*(uneven/smooth-1)` at each pressure level. Report
absolute times, P10/median/P90 and wins separately per session. A larger high-load
gain than zero-extra-load gain in both sessions supports pressure sensitivity;
consistent growth through all three levels is stronger evidence. If the gain is
flat, decreases, or differs by session, do not claim "only high pressure helps".
Do not retune levels after seeing deciding results. No model parameter fitting.

PMU counts include background and foreground and cannot isolate victim latency.
Occupancy/command is a diagnostic ratio, not validated physical latency. Counter
reset/enable occurs after scrub and outside foreground timing; enabled intervals
include small sequential ioctl boundaries beyond the kernel timing window.
All counters must have running/enabled ratio >=99%. Actual backing and all-DRAM
access are not established merely by THP policy or scrub.

## Artifacts and commands

Runner: `optimizations/fused_moe_sve/benchmarks/bench_density_pressure_sweep.py`.
Use the existing `tmp/gemm_density_20260907/high_skew_frozen.json` unchanged.
New local/remote relative output directory: `tmp/density_pressure_sweep_20260907/`.
Remote root: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp`.

```bash
env OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:src \
  timeout 300 numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/density_pressure_sweep_20260907/bench_density_pressure_sweep.py \
  --frontier tmp/gemm_density_20260907/high_skew_frozen.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request008_case009_zh2048-010.pt \
  --seed 20260907 --output tmp/density_pressure_sweep_20260907/session1.json
```

First run `--pilot` with output `pilot.json` (one warmup, one effective round,
diagnostic only). After readiness review, run two formal sessions using seeds
20260907/20260908 and session1/session2 filenames. Existing files are rejected.
Each session preserves reader maps, event configs, source/extension/frontier
hashes, background-only counters and every paired measurement. No native build.
