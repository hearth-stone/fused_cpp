# SVE fused MoE experiments

This directory contains SVE fused-MoE features and standalone experiments. The
weighted route-merge U1 kernel and async ready-token merge are enabled by
default for their supported SVE paths; neither changes the fused-MoE API.

The paper-facing scope, admissible evidence, and open gates are maintained in
[`../../docs/moe_paper_readiness.md`](../../docs/moe_paper_readiness.md). The
manifest in this directory remains the source of truth for whether an
individual feature is enabled, experimental, diagnostic, or retired. A dated
result report is valid only for its recorded source, binary, profile, machine,
and execution geometry.

Cross-machine paper experiments can be declared and run through
[`paper_experiments/`](paper_experiments/). Machine JSON files keep SSH,
affinity, NUMA, and build differences outside benchmark implementations, while
each local result bundle records the committed source revision, extension hash,
rendered command, raw logs, and structured output. The smoke and pilot suites
validate orchestration and mechanism coverage; neither is a headline paper
matrix without the larger sample and workload gates in the paper-readiness
document.

Production W13 and W2 each cover one complete packed-N stage. Legacy boolean
split, range-count, and byte-window controls are gone. Their replacement is one
explicit per-task owner window in Plan V2: `(threads, window_tiles)` uniquely
defines a stage, each team window covers `threads * window_tiles` N tiles, and
successive windows cover the complete N domain. A zero Plan V2 value selects
the full owner stripe `ceil((N/v)/threads)`. Window experiments must construct
an explicit Plan V2 rather than changing production dispatch through process
environment. Sections explicitly labelled historical retain the earlier
split/range implementations only as provenance.

The production `FUSED_CPP_MOE_FUSED_2D_SPLIT` adapter was also removed on
2026-08-10. It always lowered to `tm=1, tn=threads`, and its SVE wrappers only
accepted `row_begin=0`, so it was an N-split compatibility layer rather than a
mixed M-by-N implementation. The actual mixed-MN experiment remains isolated
in `bench_mn_split.cpp`; restore Git `8e9fcbd` only when reproducing the retired
production adapter.

## Paper-facing current status

The initial paper should use the BF16 SVE path with FP32 direct-route storage
and FP32 weighted accumulation. W8A16, W8A8, BF16 route storage, experimental
SiLU approximations, and synthetic shared-expert scheduling have separate
quality or generalization gates and should not be mixed into the primary BF16
claim.

The strongest kernel evidence currently consists of the explicit-unfused
control, exact-M tail specialization, adaptive gather-pack, FP32 W2 direct
route store, and tile-window ablations. The July 2026 nine-workload
vLLM-style comparison is mechanism evidence tied to retired profiles and older
geometry; rerun it on the frozen paper binary before using it as a headline
table.

The current Amazon 192-core closure measurements are indexed in
[`results/amazon_192c_paper_closure_20260831.md`](results/amazon_192c_paper_closure_20260831.md).
They narrow the claims materially: canonical fusion is beneficial at large M
but regresses the short control, current direct-route gains are about 3--4% in
the clear long/high-skew cases, tile-window gains are concentrated in the
uniformish 2048/4096 domain, and the best two-stage schedules remain
1.93%--5.87% slower than the best whole-expert schedules on three captured
traces. Most of that matrix predates comparator commit `266da2c`; it is
provisional until the declarative runner reproduces it from one frozen build.

The deployment planner is `MoePlannerRuntime` quick search: homogeneous
isolated-LPT shapes and strict Plan V2, with an optional fixed-width fallback.
The broader mixed-width/contention-aware full planner is an offline
reference/autotuning mode, not an oracle: the current high-skew case contains a
24.43% gap between its selection and an existing 8T/order candidate. Kernel
result reports must state which planner mode and fixed-width control were used,
and must include planner latency when claiming end-to-end improvement.

## Adaptive M-by-K gather-pack

The production fused-SVE gather/pack-A stage uses physical M12 panels plus at
most one M8 tail panel. Complete panels remain the unit of work whenever they
can occupy the expert team. For an underfilled team, the stage flattens
`(M panel, K stripe)` into a single work sequence and divides that sequence
across the existing team. This removes Gather-to-W13 idle time without changing
the GEMM's N-split ownership.

The minimum K stripe is 32 BF16 elements. Boundaries are aligned to K8, which
maps to 192 bytes in an M12 packed panel and 128 bytes in an M8 packed panel;
there is therefore no cross-worker output-cache-line sharing. A worker's
adjacent stripes in the same panel are coalesced. The adaptive path is disabled
when `K_pad < 64` or the M-panel count is at least the team width. All workers
still meet at the existing pre-W13 barrier because every W13 N owner reads the
complete shared packed A.

On the eight-core Neoverse-V1 host, the complete eight-expert scheduled call
improved by 2.3-12.6% at every sampled M in `{1,9,12,13,24,37}` with
8T/expert; M=72 and M=96 controls did not regress. Correctness, commands,
cache geometry, sample spread, and the adoption decision are recorded in
[`results/amazon_8c_adaptive_mk_gather_pack_20260812.md`](results/amazon_8c_adaptive_mk_gather_pack_20260812.md).

## Xbyak exact-M compute kernels

The default one-chunk SVE path generates W13 fused-SiLU/packC, W2 FP32, and W2
FP32 direct-route kernels with the pinned `xbyak_aarch64` submodule. The GEMM
body retains the production static-assembly K granularity and accumulator
mapping. For M<=8 it also reproduces the static M2/M4/M8 K-loop state machine:
one A/B register bank computes while the alternate bank already holds the next
K4 panel, with the same current/next tail branches. M=9..12 retains the static
M12 single-bank schedule because its 20-24 accumulators leave no registers for
a second complete A/B bank. The generated tail specializes every logical
M=1..12: its compute height is
`2 * ceil(M / 2)`, while its packed-A height remains eight rows for M<=8 and
twelve rows otherwise. This removes the static M4/M8/M12 bucket overcompute
without changing packed weights, intermediate layout, or public APIs.

Use `FUSED_CPP_MOE_SVE_IMPL=asm` for the static reference and
`FUSED_CPP_MOE_SVE_IMPL=jit` for strict validation of the generated W13 and
FP32-W2 surfaces. The default `auto` selects generated code where supported and
retains assembly for the non-default epilogues documented in
`csrc/moe/README.md`.

The A/B benchmark keeps both implementations in one process but measures
steady blocks. After each implementation switch it executes one unmeasured
transition call, preventing code-switch I-cache replacement from being charged
to the timed sample:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --routes 1,2,3,4,5,6,7,8,9,10,11,12 --threads 1,2,4,8 \
  --warmup 8 --runs 40 --switch-period 4

```

With H=4096, F=512, eight distinct experts, and `R13=2,R2=1`, the 192-core host's
NUMA0 gained 13.6% for M5/M6 and about 10% for M9/M10 at 1T. M1-4, M7/8, and
M11/12 are within 0.5% of static assembly at 1T. Reproducing the double-buffer
state machine removed the earlier M7/8 regressions of 2-4% at 1T-4T. The
8-core host retains roughly 4-12% on the exact tails, while equal-compute-height
controls stay within about 1.2%. Full commands and tables are in
[`results/amazon_8c_192c_xbyak_exact_m.md`](results/amazon_8c_192c_xbyak_exact_m.md).

The upstream M8/M12 ILV schedules were compared directly with the matching
non-ILV assembly and pure-GEMM JIT on both SVE256 Neoverse-V1 and SVE128
Neoverse-V3. M8 ILV regresses V3 by 5.40% warm and 2.66% rotating-cold for
W13. M12 ILV gains only 0.27-1.09% on the stable V3 cases and is mixed on V1.
Neither schedule is adopted; the current non-ILV fused JIT remains the
default. Method, five-run ranges, and W13/W2 tables are in
[`results/amazon_8c_192c_upstream_m8_m12_ilv.md`](results/amazon_8c_192c_upstream_m8_m12_ilv.md).
The rejected `--include-ilv` benchmark adapter was removed from the active tree
on 2026-08-10; restore Git `fe0ca5a` to reproduce that exact comparison. The
upstream assembly file remains an external reference, not an active fused-MoE
variant.

An M12-only JIT probe that keeps all six packed-A pairs live and ping-pongs two
packed-B column registers was also tested on all 96 V3 cores. A 21-repeat long
window improved median L1-hot throughput by only 0.648% and linear efficiency
by 0.592 percentage points; the vector issue-queue-full event increased by
about 37.5%. The schedule was rejected and removed from the active tree; it
remains reproducible from Git commit `4825bf9`, with measurements in
[`results/amazon_192c_m12_column_pipeline_20260802.md`](results/amazon_192c_m12_column_pipeline_20260802.md).

The bulk-M experiment is bitwise correct but performance-neutral across the
192-core host NUMA0 grid. Its runtime flag, JIT state machine, cache dimension,
tests, and benchmark selector were removed from the active tree on 2026-08-10;
the implementation remains available at Git commit `3faf244`. Corrected
paired-window results are in
[`results/amazon_192c_xbyak_bulk_m.md`](results/amazon_192c_xbyak_bulk_m.md).

The M1/M2 dual-N experiment shared packed-A loads across two adjacent N tiles,
but its low-concurrency gain reversed at larger expert waves: about +2.3% at
8 active experts became -5.4% to -5.7% at 24 and as low as -5.0% at 96. The
environment flag and generated dual-N state machine were removed on 2026-08-10;
the source remains at Git commit `f671f7e` and the synchronization evidence is
in [`results/amazon_192c_xbyak_m2_dual_n.md`](results/amazon_192c_xbyak_m2_dual_n.md).

The timed-task-release lowering of the offline cold-phase CP-SAT schedule was
also removed from Plan V2 and the native executor. It reduced average active
cores from 93.51 to 57.18 and regressed throughput by 28.82%; release timing
itself was not the missing factor. Offline start times remain oracle analysis
data, while the executable bridge and benchmark remain at Git commit `94e977d`.
The evidence is in
[`results/amazon_192c_cold_phase_oracle_runtime_20260731.md`](results/amazon_192c_cold_phase_oracle_runtime_20260731.md).

The removed first-panel-prefetch experiment gave the W13-only candidate median
gains of about 1.93% at 1T and 2.10% at 4T over M1-M12 in five-process tests.
Prefetching W2 added no stable isolated benefit and regressed by about 7.36%
relative to W13-only at 24 concurrent 4T experts. Reusable W13 B panels also
regressed, reaching about -4.35% e2e at M48/4T. The implementation, environment
flags, tests, and benchmark variants have been removed from the active tree;
the measurements remain in this report and the implementation remains in Git
history:
[`results/amazon_192c_w13_first_panel_prefetch.md`](results/amazon_192c_w13_first_panel_prefetch.md).

The one-off small-M instruction decomposition is complete. Its BA-only,
BFMMLA-only, A-only, control-only, store, and fixed-A JIT modes and dedicated
perf harnesses were removed from the active tree on 2026-08-10 and remain at
Git commit `9aa9248`; the evidence remains in
[`results/amazon_192c_small_m_gemm_bandwidth.md`](results/amazon_192c_small_m_gemm_bandwidth.md).
Machine calibration retains only the stable B-only, M12 full-no-store, and M12
matrix-only services, with their historical numeric mode ids unchanged.

The NUMA0 single-core cold-weight calibration uses 64 rotating H4096/F512
experts and a standalone SVE `LD1H` reader. The production-like 12 MiB chunk
read ceiling is 40.04 GB/s; the M2 fused W13+W2 stages reach 36.50 GB/s
(91.2%), while M12 reaches 23.99 GB/s as its physical BFMMLA issue efficiency
rises to 71.3%. Exact definitions, M1-M12 tables, and reproduction commands are
in
[`results/amazon_192c_single_core_m1_m12_efficiency.md`](results/amazon_192c_single_core_m1_m12_efficiency.md).

The retired pure-W13 M1/M2 benchmark removed the fused W13 epilogue by running
the same exact-M packed-A/B JIT K-loop with the plain FP32 GEMM store. With the
production `R13=2,R2=1` geometry, cold-weight M1/M2 sustain about 36.1-36.6 GB/s,
or 90-91% of the calibrated read ceiling. This is essentially the same as the
complete M2 W13+W2 stage, locating the remaining bandwidth gap in the GEMM
load/compute loop rather than SiLU or pack-C. The source remains at Git commit
`9aa9248`; results and the historical reproduction command are in
[`results/amazon_192c_pure_w13_gemm_m1_m2.md`](results/amazon_192c_pure_w13_gemm_m1_m2.md).

## SVE weighted route merge

`csrc/moe/arm/sve_bf16/route_merge.cpp` provides the SVE weighted merge for the
token-major `[tokens, top_k, hidden]` route buffer. Fixed `top_k=2/4/6/8`
dispatches to compile-time templates. Template recursion first forms adjacent
weighted pairs and then a power-of-two-prefix binary tree; for example,
top-k=6 is `((0+1)+(2+3))+(4+5)`. Other positive values use a runtime slot
loop that retains one token row's accumulator vector in an SVE register and
preserves slot order. The existing `top_k=1, skip_weighted=true` path
still writes the W2 result directly during scatter and bypasses merge entirely.

The SVE backend uses the former U1 implementation exclusively. It keeps the
accumulator in SVE registers and writes BF16 directly without allocating a
thread-local FP32 row. The sequential accumulator and U2/U4 variants were
retired after showing no consistent full-operator advantage; their source is
available in Git history at `7fc10fc`. Fixed templates change FP32 association
relative to the historical sequential baseline; the dynamic fallback does not.
The standalone benchmark checks the production kernel against scalar
references for FP32 and BF16 route buffers:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  optimizations/fused_moe_sve/benchmarks/bench_route_merge_tree \
  --tokens 2048 --top-k 6 --hidden 4096 --threads 96 \
  --source both --warmup 5 --runs 31 --check

# Rotate a source/output ring larger than cache to measure single-pass traffic.
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  optimizations/fused_moe_sve/benchmarks/bench_route_merge_tree \
  --tokens 2048 --top-k 6 --hidden 4096 --threads 96 --copies 8 \
  --source both --warmup 8 --runs 31 --check
```

The 192-core-host NUMA0 measurements are recorded in
[`results/amazon_192c_route_merge_tree.md`](results/amazon_192c_route_merge_tree.md).

## W2 direct route store

The default SVE W2 epilogue consumes each expert's flat route-row table
and stores its FP32 result directly into the token-major `route_out` tensor.
M12/M8/M4/M2/M1 kernels retain the production BFMMLA body; only the store
epilogue replaces the fixed contiguous row stride with a per-row destination.
N-split workers still own disjoint `n_tile`-aligned H ranges, so no two workers
write the same output element. The existing FP32 weighted route merge is
unchanged.

An opt-in BF16 route-buffer variant uses matching assembly epilogues to convert
and store each W2 result directly as BF16. It does not allocate either the
contiguous `down` tensor or a separate FP32 route tensor. The weighted merge
loads BF16 routes and still accumulates all top-k contributions in FP32:

```bash
FUSED_CPP_MOE_W2_BF16_ROUTE=1 <fused MoE command>
```

Disable the path and retain the contiguous `down` plus scatter fallback with:

```bash
FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=0 \
  <fused MoE command>
```

Dispatch requires the SVE fused-SiLU packed-A path, an unbiased W2,
`N_pad == H`, and route offsets representable by the SVE signed 32-bit scatter
offsets for the selected FP32 or BF16 element size. Unsupported cases retain
the existing contiguous W2 store plus scatter. FP32 direct route is default-on
because it preserves bitwise FP32 output, removes the per-team `down`
allocation, improves long-route E2E by roughly 14-16% on the 192-core host's
first NUMA node, and is statistically neutral for short routes.

BF16 route storage remains opt-in pending model-level accuracy validation. On
the same host, the representative eight-expert, 12-thread/expert, M=1536 test
improved from 8.087 ms for FP32 direct to 7.726 ms for BF16 direct (median of
three process medians), or 4.68% higher throughput. The gain was neutral at
M=12 and reached 1.31% at M=192. The four-way benchmark checks FP32 scatter,
FP32 direct, BF16 scatter, and BF16 direct in one process.

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_w2_direct_route.py \
  --path async --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --threads 96 --warmup 5 --runs 31
```

The implementation, traffic accounting, and measurements are recorded in
[`results/amazon_192c_w2_direct_route.md`](results/amazon_192c_w2_direct_route.md)
and
[`results/amazon_192c_w2_direct_bf16_route.md`](results/amazon_192c_w2_direct_bf16_route.md).

## Async ready-token route merge

The async bridge enables an executor that overlaps route merge with an
imbalanced expert tail by default. Retain the post-expert merge explicitly
with:

```bash
FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=0 <async fused MoE command>
```

It is restricted to the SVE direct-route path and supports either FP32 or BF16
route storage. After every expert team's final W2 barrier, the leader
release-publishes task completion, checks the TopK
expert state for its route tokens, claims each newly ready token once, and
publishes claimed tokens to the worker queue in one batch. The async worker
loop always selects eligible expert work first; only an otherwise idle lane
claims one ready token. When expert compute ends, the existing static,
contiguous merge skips completed tokens and handles every remaining range.

The executor estimates each team's work as `ceil(M / 12) / threads`. If the
maximum is less than 1.25 times the minimum, it retains the post-expert
contiguous merge because a balanced schedule has no useful idle interval. The
default does not change planner decisions or cost tables. Explicit value `1`
is equivalent to leaving the variable unset.

Run balanced and controlled heavy-tail A/B tests with:

```bash
for distribution in balanced two-group; do
  numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
    .venv/bin/python \
    optimizations/fused_moe_sve/benchmarks/bench_async_ready_token_merge.py \
    --tokens 2048 --hidden 4096 --intermediate 512 \
    --experts 12 --top-k 6 --threads 96 --distribution "$distribution" \
    --warmup 5 --runs 31
done
```

The design, rejected per-route atomic prototype, and initial NUMA0 measurements
are recorded in
[`results/amazon_192c_async_ready_token_merge.md`](results/amazon_192c_async_ready_token_merge.md).

## vLLM-style staged scheduling baseline

`fused_moe_bf16_tiled_vllm_staged` is an explicit experimental entrypoint for
isolating scheduling from GEMM implementation. It uses the production SVE
packed weights, M12/M8/M4/M2/M1 assembly kernels, fused SiLU/packC epilogue,
direct route store, and weighted merge. Only the execution order changes:

1. every contiguous `(expert, W13 N-range)` is claimed from one global atomic
   task queue;
2. all workers finish W13 before a global stage barrier;
3. every `(expert, W2 N-range)` is claimed from a second global queue;
4. route merge runs after the complete W2 stage.

Each W13 N-range rescans the expert input and gather-packs one M12 panel at a
time, matching the referenced vLLM CPU implementation rather than sharing one
packed A inside a fixed expert team. W13 and W2 task widths use the same policy
as that implementation: take the smaller of the private-L2 capacity limit and
the `num_threads / top_k` parallelism limit, then align to the native N tile.
The available-L2 budget is half of the detected private L2. The entrypoint
requires fused-SiLU SVE weights and does not alter default dispatch.

Compare it with a fixed-team async schedule while holding kernels and route
epilogues constant:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 256 --top-k 6 --threads 96 \
  --distribution hot-topk --team-threads 16 \
  --warmup 3 --runs 11 --stage-timing
```

The same benchmark accepts the paper workload catalog and a schema-v2 profile
to compare the vLLM-style queue against the actual production planner:

```bash
P=cpu_moe_schedule_optimization/cost_model/profiles/\
contention_async_amazon_c5_192c_dual_numa_tp4_sve_F512_E256_\
fulln_schema_v2_xbyak_exactm_20260727.json
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-tiered-hotspot --production-profile "$P" \
  --route-dtype bf16 --warmup 5 --runs 21
```

For the long/short bimodal preset, `--static-16-to-4` adds a benchmark-only
external DAG between those controls. Each `M=2040` expert keeps one 16-thread
team; when it completes, that exact interval becomes four independent
4-thread lanes for `M=12` experts. A free 16-thread interval starts as four
4-thread lanes immediately. Short experts are assigned statically by the
profile's isolated-time estimate, so this tests kernel width and team
transition overhead without adding work stealing or changing production
planner decisions:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-long-short-bimodal --production-profile "$P" \
  --static-16-to-4 --route-dtype bf16 --warmup 5 --runs 21
```

On AmazonC5192Cores NUMA0, the static transition reduced the median from
`12.390 ms` for production `6 x 16T` to `11.017 ms` and was also faster than
the `11.631 ms` staged queue. With ready-token merge disabled, the corresponding
medians were `12.560 ms` and `11.089 ms`, confirming that the gain comes from
the expert DAG rather than merge overlap. This remains an experimental
benchmark variant; the production planner does not emit this external static
team-fission DAG.

The native Plan V2 executor always supports a whole-expert `tail_pool`, but
`PlannedMoE` now decides whether to use it. The default planner compares strict
execution with route thresholds `1/2/4/8/12` and aligned pool widths `1/2/4T`;
the selected threshold determines which experts enter the pool, and the
selected width determines how many resident workers claim each pooled expert.
Fixed head tasks retain their planner-selected teams. `dynamic_tail_pool=False`
requests the strict control, while `tail_pool_threads=T` forces a particular
pool width for experiments.

The benchmark emits strict and planner-selected variants by default. Adding
`--dynamic-short-pool` also emits the historical forced-4T comparator. All
variants use the native Plan V2 bridge rather than the legacy
`FUSED_CPP_MOE_ASYNC_SHORT_POOL_*` environment switches:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-long-short-bimodal --production-profile "$P" \
  --dynamic-short-pool --route-dtype bf16 \
  --warmup 5 --runs 21
```

On AmazonC5192Cores NUMA0, the planner selected a `6x16T` fixed head plus a
`1T`, `M<=12` tail pool for the long/short bimodal workload. Median operator
time fell from `12.529 ms` strict to `10.408 ms` (`+20.39%`,
`14.856 TFLOP/s`), versus `10.801 ms` for forced 4T and `11.728 ms` for vLLM
staged. Across uniform, active-set 8/16/32/64/128, tiered-hotspot, and the
captured DSV4 routing workload, the planner retained strict execution; paired
strict/auto medians differed by at most 0.38%, which is timing noise because
the generated native plan is identical. Those experiments measured the Python
cold planner at `60.581 ms` for bimodal and `272.151 ms` for captured routing;
cache hits were `0.944 ms` and `2.396 ms`. The current C++ cold planner keeps
the same plan/ranking and, with the default 8 candidate workers, measures
`1.421 ms` and `3.488 ms` on the same host. Caching remains useful, but the
first search no longer carries the old Python latency.

On the 192-core host's first NUMA node, six balanced `M=2048` experts were
13.1% slower with the staged queue than with fixed `6 x 16T` teams. For 256
uniform `M=48` experts, staged scheduling was within 1% of the best swept fixed
width (`8T/expert`) while greatly outperforming deliberately under-threaded
fixed schedules. The commands, stage timing, and full sweep are recorded in
[`results/amazon_192c_vllm_staged_schedule.md`](results/amazon_192c_vllm_staged_schedule.md).

## Explicit unfused pipeline reference

`bench_unfused_pipeline` measures the full fusion boundary with the production
M12 SVE assembly body held constant. Its explicit baseline executes:

1. route extraction to a BF16 row-major tensor;
2. M12 pack-A;
3. W1 GEMM to FP32;
4. standalone poly5 + exact-division SiLU to FP32;
5. W3 GEMM to FP32;
6. standalone multiply and BF16 materialization;
7. a second M12 pack-A;
8. W2 GEMM with BF16 output.

The control directly gather-packs input, calls the production fused
`moe_sve_w13_silu_poly5_packc_m12_rows_opt` entrypoint, then calls the same W2
BF16 entrypoint as the baseline. W1, W3, fused W13, and W2 all use the same
`M12_K4_BODY`; only the dataflow and epilogues differ. Buffers are preallocated,
weights are distinct per expert and per rotating copy, and no allocator work is
inside the measured region. The experiment covers expert compute through W2
output and deliberately excludes route-weight accumulation/scatter.
Warmups consume the first copies and timed iterations continue from that
offset, so `copies >= warmup + iters` guarantees no copy reuse.

The default shape models EP2 (`H=4096`, `F=2048`) with four concurrent experts,
24 threads per expert, and the current `R13=2,R2=1` policy:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  optimizations/fused_moe_sve/benchmarks/bench_unfused_pipeline \
  --experts 4 --m 2040 --h 4096 --f 2048 \
  --threads-per-expert 24 --w13-ranges 2 --copies 2
```

The default `--variant both` alternates execution order for direct timing
comparisons. Use `--variant unfused` or `--variant fused` to isolate one path
when collecting external profiler or PMU counters.

## Single-core weight-window experiment

`bench_single_core_weight_window` calls the production
`moe_sve_w2_packed_bf16_m12` entrypoint directly on one pinned core. It keeps
M and K fixed while increasing N, so one packed-B weight is exactly
`K * N * sizeof(bf16)` bytes. Every warmup and measured invocation uses a
different packed-B address and value. The used weights are followed by a
configurable cold-tail allocation that is first-touched after them, preventing
initialization from leaving the measured weights resident. Immediately before
the profiler boundary, the benchmark reads one element from every cold-tail
cache line; streaming stores therefore cannot bypass the intended eviction.

The companion runner waits until all allocation and first-touch work is done,
attaches `perf` to the stopped process, and then resumes only the GEMM region:

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_single_core_weight_window.py \
  --m 120 --k 4096 --cpu 48 --numa-node 0 \
  --warmup 2 --runs 7 --cold-tail-mib 192 \
  --output-json /tmp/single_core_weight_window.json
```

Use `--extra-events` to append machine-specific PMU or software events without
changing the default cache-counter set, for example:

```bash
--extra-events l1d_tlb,l1d_tlb_refill,l2d_tlb_refill,page-faults
```

Inter-invocation packed-B reuse is prohibited by construction. Reuse of the
same B by the ten M12 panels inside one `M=120` GEMM is intentional: that is
the behavior used to expose the private-L2 and shared-LLC residency windows.

## M12 streaming-B load experiment

`bench_m12_streaming_b` isolates B-load policy without changing production
dispatch. Its standalone assembly copies the production M12 BFMMLA body and
BF16 store epilogue exactly, then provides normal `LD1H`, `LDNT1H`, and
`PLDL1STRM`/`PLDL2STRM`/`PLDL3STRM` variants. The production symbol is used
only as the bit-exact correctness reference. Every timed call consumes a
distinct cold 4 MiB packed-B matrix, and a 4 KiB guard after every copy
prevents the final software prefetch from touching the next invocation's
weight. `--workers N` runs strictly synchronized one-core-per-B waves and
reports aggregate wall time rather than independent worker medians.

Run the complete candidate sweep with:

```bash
python3 optimizations/fused_moe_sve/benchmarks/run_m12_streaming_b.py \
  --shape all --cpu 48 --numa-node 0 \
  --warmup 5 --runs 31 --cold-tail-mib 192
```

Use `--variants` for an isolated paired comparison, for example:

```bash
python3 optimizations/fused_moe_sve/benchmarks/run_m12_streaming_b.py \
  --shape w13 --variants baseline_ld1h,pldl1strm_2048_x1 \
  --cpu 48 --numa-node 0 --warmup 5 --runs 51
```

On the 192-core host's first NUMA node, the W13-range shape
`M=12,K=4096,N=512` improved by a median 9.99% across seven paired processes
with one `PLDL1STRM` hint 2 KiB ahead. The W2 shape
`M=12,K=512,N=4096` improved by only 0.73% with one `PLDL2STRM` hint 1 KiB
ahead. Pulling W2's B stream into L1 was strongly
negative; this is consistent with displacing its otherwise L1-resident 12 KiB
packed-A panel. `LDNT1H` had no stable benefit. The result supports a
W13-like large-K specialization, not a generic M12 load replacement; no
production path is enabled by this experiment.

A 2026-07-27 follow-up tested one-hint `PLDL3STRM` distances from 512 to
4096 bytes. W13 changed by less than 0.3% at 1, 8, and 24 workers and was
slightly negative at 48 workers. W2 gained about 1-2% at low/moderate
contention but regressed by 4.9% at 96 workers; a 4 KiB lead also regressed at
24 workers. L3 streaming prefetch therefore remains benchmark-only and is not
the cache-safe high-concurrency replacement for the retired L1 policy.

Commands and complete measurements are recorded in
[`results/amazon_192c_m12_streaming_b.md`](results/amazon_192c_m12_streaming_b.md).

## M12 assembly K-block experiment

The same standalone assembly also provides a dense set of fixed Kc variants
that change the GEMM loop order from `Ntile -> full K` to
`Kchunk -> Ntile -> K4`. Each N tile's 24 FP32 accumulator vectors are stored
in an accumulator-native scratch layout between K chunks; the final chunk alone
runs the production BF16 store epilogue. Storing and reloading FP32 partials is
bit exact, so all variants match the production M12 result bit for bit.

Two packed-B addressing modes isolate the layout dependency. `kblock_*` keeps
the then-production Ntile-major layout and therefore starts one short, strided cold
B stream per `(Kchunk,Ntile)`. `kblock_packed_*` uses a benchmark-only
Kchunk-major/Ntile-minor pack, making each complete K chunk contiguous while a
small packed-A K slice is reused across all N tiles. Timed weights are already
in the selected layout; repacking is not included in GEMM time.

For an isolated fixed Kc, use one variant per process:

```bash
python3 optimizations/fused_moe_sve/benchmarks/run_m12_streaming_b.py \
  --shape w13 --variants kblock_packed_800 \
  --cpu 48 --numa-node 0 --warmup 5 --runs 51 --repeat 7 \
  --cold-tail-mib 192 --unique-scratch --prewarm-a
```

On the 192-core host's first NUMA node, the dense `Kc=704..896` sweep for
W13 `M=12,K=4096,N=512` selected `Kc=800`: isolated throughput increased
from about 283 to 315 GFLOP/s, or 11.5%. Its active packed-A plus one-N-tile B
window is 31.25 KiB. This matches the measured first-order rule
`Kc ~= L1D_bytes / (4 * (M + n_tile))`, which targets half of the 64 KiB L1D
and predicts Kc near 819. A nearby aligned value must still avoid a tiny final
K chunk; Kc=816 leaves a 16-deep tail and loses to Kc=800.

That isolated cold-panel gain does not carry through to a long route. With
`M=2040`, one timed call executes 170 consecutive M12 panels against the same
4 MiB B range. Baseline and Kc=800 measured 25.5168 and 25.3171 ms respectively,
only a 0.79% throughput gain. After the first panel, cached B makes the baseline
fast enough that K-block partial-C traffic cancels most of the A-residency gain.

PMU comparison held L2 refills at roughly 65.9K lines/call, matching the
compulsory cold 4 MiB B stream, while Kc=800 cut L1 refills from 11.3K to
5.35K and raised IPC from 4.10 to 4.63. Keeping the production B layout was
negative, demonstrating that loop interchange and weight packing must be
changed together. Run each Kc in a separate process for absolute timing:
interleaving variants measurably changes cache/prefetch state. The result is
the measurement basis for the production integration described below.

Implementation details and the complete measurements are recorded in
[`results/amazon_192c_m12_kblock.md`](results/amazon_192c_m12_kblock.md).

## Small-M assembly K-block experiment

`bench_msmall_kblock` extends the standalone Kchunk-major experiment to the
production M8/M4/M2/M1 compute bodies. Kc is dynamic, while each height keeps
its production BFMMLA pipeline and BF16 store. M1 follows production by using
the M2 body with its second row predicated away. Multiple K-tail shapes match
the corresponding production kernels bit for bit.

The M12 footprint formula needs a physical-layout correction. Every small-M
packed-A K4 block has a 64-byte stride. M4 and M2/M1 load only part of that
line, but the complete line occupies L1, so all four heights have:

```text
W_tile(Kc) = (16 A bytes/K + 16 B bytes/K) * Kc
Kc_half_L1 = (64 KiB / 2) / 32 = 1024
```

The measured result qualifies the original rule. M8 is capacity-sensitive:
Kc=896-960 uses a 28-30 KiB A+B window and improves by about 11.7%, while
Kc=1024 reaches 32 KiB and falls to about 8.0%. M4 has a broad Kc=384-768
plateau at roughly +7.3%. M2/M1 are cold-B-latency dominated and remain near
+8.4% over Kc=16-704; capacity no longer identifies one optimum.

PMU counts keep L2 refill fixed near 65.8K lines, matching the compulsory 4 MiB
B stream. M8 gains by reducing L1 refills from 16.0K to 4.3K. M2/M1 retain
near-B-sized L1 refill counts and much higher memory-stall exposure, so K
splitting mainly changes request scheduling and uses otherwise idle issue
slots. The half-L1 formula is therefore a physical capacity ceiling and an
approximate optimum only for sufficiently compute-heavy kernels.

```bash
for m in 8 4 2 1; do
  python3 optimizations/fused_moe_sve/benchmarks/run_msmall_kblock.py \
    --m "$m" --variants baseline,kblock \
    --k-blocks 384,512,640,704,768,832,896,960,1024 \
    --warmup 5 --runs 51 --repeat 5 --cpu 48 --numa-node 0 \
    --cold-tail-mib 192 --unique-scratch --prewarm-a
done
```

Implementation details, dense sweeps, cache-color controls, and PMU results are
recorded in
[`results/amazon_192c_msmall_kblock.md`](results/amazon_192c_msmall_kblock.md).
The benchmark entries remain standalone. Production uses one full K chunk;
the historical runtime integration is retained in Git `44dbcd9`.

## Historical production Kc path

This section records a retired production experiment. The active SVE backend
packs B as `Ntile -> K4` and always computes one full K chunk. The
`FUSED_CPP_MOE_SVE_KC` and `FUSED_CPP_MOE_SVE_KC_L1_PERMILLE` controls, their
Kchunk-major packed-B layout, and production correctness branches were removed
on 2026-08-10. The standalone K-block probes and reports remain reproducible.

One Kc is used by every Mr because a prepared weight has one physical layout.
The retained production contract is one full K chunk:

```text
Kc = K
```

Historically, the two removed environment controls selected an 8-aligned Kc or
a calibrated fraction of L1D. The 49% selector mapped to Kc=800 on the SVE128
192-core host and Kc=568 on the SVE256 8-core host.

With an explicit calibrated Kc, the isolated cold-B result is positive for
every Mr on the 192-core host
(roughly +6% to +13%). On the 8-core host, Kc=568 is the maximin compromise:
M12/M4/M2/M1 improve by about 2.23%/2.35%/0.95%/0.58%, while M8 regresses by
about 1.48%. Production `R13=2,R2=1` E2E improves by about 2.0% at route=12 and
3.5% at route=2040 for two 4-thread experts on the 8-core host.

The isolated result must not be extrapolated to a saturated expert wave. On
NUMA0 of the 192-core host, 24 concurrent 4-thread experts changed from
0.599 to 0.662 ms at route=12 and from 30.508 to 30.895 ms at route=2040.
Sweeping Kc through 1024/1536/2048 reduced but did not reverse this loss. Cold
B is then memory-bandwidth dominated, so A-side L1 residency has little value
while partial-C instructions remain. Cost tables must therefore be regenerated
with the production path. On 2026-07-19 the default was restored to one K
chunk. The production selector was retired because its applicability depended
on concurrent memory pressure, while the standalone probes already preserve
the useful cache-residency result without expanding the runtime compatibility
surface.

Calibration, correctness, production commands, and repeated timings are in
[`results/amazon_192c_8c_production_kc.md`](results/amazon_192c_8c_production_kc.md).

## M12 LLC-pollution experiment

`bench_m12_llc_pollution` separates packed-B residency from simultaneous DRAM
bandwidth contention. On one pinned core it promotes a configurable victim
buffer with a randomized dependent cache-line traversal, executes distinct
M12 polluter experts, and then immediately probes the victim again. Each
polluter uses the production fused M12 W13 SiLU/multiply/packC kernel followed
by the production M12 W2 kernel, with two 4 MiB W13 ranges and one 4 MiB W2
matrix. Different trials use distinct victim and polluter addresses.

`--polluter-panels 1` models a route-12 expert that reads each B line once.
Larger values repeatedly use the same 12 MiB weight and model the cache
promotion performed by a longer route. The victim is a latency-sensitive
residency probe rather than a GEMM, so instruction throughput cannot hide LLC
eviction.

```bash
numactl --cpunodebind=0 --membind=0 \
  optimizations/fused_moe_sve/benchmarks/bench_m12_llc_pollution \
  --victim-mib 64 --polluter-experts 8 --polluter-panels 1 \
  --trials 9 --evict-mib 192 --cpu 48
```

## Threaded full-stage working-set experiment

`run_thread_weight_working_set.py` drives the production-fused path of
`bench_unfused_pipeline` with several thread mappings:

- `nsplit`: one active expert uses T threads, each owning an N stripe;
- `expert-fixed` / `expert-total`: T one-thread experts execute concurrently,
  with either fixed per-expert weight size or approximately fixed aggregate
  active weight size;
- `team-fixed-route`: a fixed total thread budget is divided among several
  multi-thread expert teams while route count per expert remains fixed;
- `team-fixed-work`: the same team shapes are used while total route count and
  total GEMM FLOPs remain fixed.

All modes execute one full W13 and W2 stage. The instantaneous packed-B working
set is `active_experts * max(full_W13_bytes, full_W2_bytes)`. The runner also
records the largest width-derived per-thread N stripe, full wall time,
aggregate TFLOP/s, and W13/W2 stage times. Each invocation uses a distinct
weight copy.

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/full_stage_thread_working_set.json \
  --threads 1,2,4,8,16,32,64,96 --routes 192,2040 \
  --experiments nsplit,expert-fixed,expert-total \
  --nsplit-stage-mib 4,16,64 --expert-stage-mib 0.5,2 \
  --total-stage-mib 32,64,96 --warmup 2 --runs 7
```

To compare one highly threaded expert with several multi-thread experts under
the same 96-thread budget:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/full_stage_expert_team.json \
  --experiments team-fixed-route,team-fixed-work \
  --team-experts 1,2,3,4,6,8,12 --team-total-threads 96 \
  --team-route 2040 --team-total-routes 2304 \
  --team-stage-mib 16 --warmup 2 --runs 7
```

## Fixed-active-B route-fragmentation experiment

`bench_fragmented_route_pipeline` tests whether route fragmentation matters
when total FLOPs, worker count, and the maximum simultaneously active packed-B
stage are fixed. Twenty-four four-thread teams each execute 2040 total routes.
Replacing one long expert creates `Q` independent experts of `2040/Q` routes,
so expert-task and unique-weight counts grow while at most 24 experts remain
active. The default dynamic schedule starts long routes first and lets any free
four-thread team claim the next task. `--schedule slot` keeps replacement tasks
on their original team as a no-rebalancing control. Split factors must produce
M12-aligned routes; the default factors `2,5,10` produce routes `1020,408,204`.

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_fragmented_route_pipeline.py \
  --output /tmp/full_stage_fixed_active_b_fragmentation.json \
  --teams 24 --base-routes 2040 --hidden 4096 --intermediate 512 \
  --threads-per-team 4 --schedule dynamic --split-factors 1,2,5,10 \
  --replaced-teams 6,12,24 --warmup 2 --runs 7
```

The benchmark uses distinct packed weights for every logical expert task and a
different complete data copy for every warmup and timed invocation. Its E2E
region covers gather-pack, fused W13, and BF16 W2, including per-task team
barriers, but excludes planner, scatter, and route-weight accumulation.

To attach perf only after allocation, weight initialization, and worker-pool
creation, use the profiler runner. It stops the initialized process, attaches
to all pinned worker TIDs, and excludes the main thread:

```bash
.venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/profile_fragmented_route_pipeline.py \
  --output /tmp/fixed_active_b_fragmentation_perf.json \
  --teams 24 --base-routes 2040 --hidden 4096 --intermediate 512 \
  --threads-per-team 4 --schedule dynamic --split-factors 1,2,5,10 \
  --cpu-start 0 --numa-node 0 --warmup 2 --runs 7 --repeats 3
```

## Elastic N-split experiment

The experiment invokes the production `moe_sve_*_m12` assembly symbols with
three scheduling variants:

- `static`: one fixed contiguous N range per worker over the full M dimension.
- `epoch_fixed`: the same mapping, with M divided into fixed 12-row-aligned
  epochs. This isolates kernel fragmentation.
- `phase_claim_fixed` / `elastic_phase`: workers claim one N lane spanning every
  consecutive M epoch with the same lane count. This preserves each core's
  packed-B stripe affinity and has no barrier until the lane count changes.
- `strict_epoch_claim`: a stress reference that synchronizes every epoch.

An N lane always spans an entire M epoch. The experiment deliberately does not
schedule individual SVE N tiles because that would rescan packed A once per tile
instead of once per participating lane.

The benchmark reports three separate costs:

- `fragmentation_over_static_pct`: splitting M into epochs without a barrier.
- `phase_claim_over_epoch_fixed_pct`: atomic lane claiming with barriers only
  at real lane-count changes.
- `strict_claim_over_epoch_fixed_pct`: the cost of synchronizing every epoch.
- `elastic_phase_over_piecewise_phase_pct`: changing the lane count after
  accounting for the fixed-low and fixed-high phase-claim costs.

Use `--epoch-rows 1020` with `M=2040` for one mid-GEMM resize point, or
`--epoch-rows 204` for a ten-epoch synchronization stress test. Input and
packed-weight buffers rotate across copies during timing.

Build and check on an SVE BF16 AArch64 machine:

```bash
make -C optimizations/fused_moe_sve/benchmarks
make -C optimizations/fused_moe_sve/benchmarks check
```

Run the default 96-core NUMA-local sweep:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  python3 optimizations/fused_moe_sve/benchmarks/run_sweep.py
```

The measured 192-core-host NUMA0 results are recorded in
[`results/amazon_192c_numa0.md`](results/amazon_192c_numa0.md).

The controlled full-pipeline fusion comparison is recorded in
[`results/amazon_192c_unfused_pipeline.md`](results/amazon_192c_unfused_pipeline.md).

The unique-weight single-core cache-window measurements are recorded in
[`results/amazon_192c_single_core_weight_window.md`](results/amazon_192c_single_core_weight_window.md).

## W2 boundary elasticity (retired)

The experimental Plan V2 W13-to-W2 elastic cohort path has been removed from
the active source tree. Generic workloads regressed by 2.5%--24.8% because
boundary handoff and cohort formation outweighed the scarce natural regrouping
opportunities. One explicitly placed active-set tail improved by 3.66%, but the
production bounded tail repartition captures that planner-visible case without
carrying a second runtime scheduler and ABI. The implementation remains in Git
history at `0b58091`; measurements and commands remain in
`results/amazon_192c_w2_boundary_elastic.md`.

## Historical packed-B stage ranges (retired)

This section documents the removed range/window implementation and its old
measurements. None of the APIs, policies, scripts, or defaults described below
is active production behavior; current behavior is the full-stage contract at
the top of this file.

The production fused SVE expert accepts positive `w13_ranges` and `w2_ranges`
on the normal, scheduled, and task-DAG async entrypoints. Plan V2 carries the
required positive `task_w13_ranges` and `task_w2_ranges` arrays. The values are
the exact number of sequential packed-B N ranges; there is no boolean W13
selector, byte-window runtime argument, inheritance sentinel, or geometry
environment variable. Direct calls default to `R13=1,R2=1`; a production
planner supplies the exact pair selected by its profile and per-task policy.

For a GEMM with packed dimensions `(K, N)` and SVE BF16 N tile `v`, one packed-B
tile contains `2*K*v` bytes. A byte/MiB calibration target `S` is lowered before
the runtime call as

```text
max_tiles = max(1, floor(S / (2*K*v)))
ranges = ceil((N/v) / max_tiles)
```

The runtime then balances whole N tiles across that exact range count. A target
smaller than one packed-B tile therefore lowers to one tile per range. For TP4
`H=4096, F=512`, a 4 MiB calibration target corresponds to W13/W2 counts
`2/1`; 2 MiB gives `4/2`, and 1 MiB gives `8/4`.

Each range still uses the existing N-split team and the same assembly kernel.
There is no barrier between adjacent ranges. W2 owner-scatter mirrors every
range's potentially discontiguous column ownership, so the existing
W2-to-scatter barrier remains elided. Consequently the range count controls
loop ordering and the nominal active-range bound, not strict synchronized cache
residency. A smaller range also caps useful team width at its N-tile count; the
planner must account for this before selecting very small ranges.

The planner chooses a global pair only when an exact schema-v2 profile contains
the same W13/W2 range identity. Historical boolean/byte fields in old JSON are
ignored and cannot reconstruct or score another identity. Compare byte-target
calibration points, lowered to exact ranges by the benchmark, with:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_weight_windows.py \
  --experts 24 --routes 2040 --hidden 4096 --intermediate 512 \
  --threads-per-expert 4 --window-mib 0,4,2,1 --warmup 3 --runs 9
```

The 192-core NUMA0 measurements are recorded in
[`results/amazon_192c_weight_windows.md`](results/amazon_192c_weight_windows.md).
For the tested TP4 shape, the best windows were 4/2/1 MiB for 4/2/1 threads per
expert respectively. This is consistent with about 1 MiB of packed B per
active worker, but the private-L2 and aggregate-cache effects remain
confounded; treat it as a measured selection rule, not a universal constant.
That sweep used `M=2040`; the short-route optimum is 4-8x smaller, so the
1 MiB-per-worker rule must not be extrapolated below about 49 routes. See the
short-route section below.

The async benchmark also accepts independent experimental pairs such as
`--window-pairs-mib 4:4,4:2`. The route sweep in
[`results/amazon_192c_stage_weight_windows.md`](results/amazon_192c_stage_weight_windows.md)
shows that the two stages can prefer different windows at medium routes, while
long routes still prefer about 1 MiB per worker for both stages.

`PlannedMoE` can apply a named deterministic `TaskStageWindowPolicy` after it
has selected the task DAG and widths. Analytical byte targets are quantized at
that boundary, and the resulting positive exact counts populate the Plan V2
arrays. The policy is not a free search dimension, but the cost model scores
each candidate using the ranges that will actually execute.

The first policy is default-on only for the exact dual-NUMA
AmazonC5192Cores TP4 `H=4096,F=512,E=256`, 96-core/rank, SVE JIT exact-M
`R13=2,R2=1` profile identity. `PlannedMoE` resolves it independently for each
candidate profile, so nonmatching profiles retain their own exact baseline
pair. Pass `use_default_stage_window_policy=False` to build a
controlled baseline. The benchmark below compares that disabled baseline with
the default policy; `--no-static-stage-windows` suppresses the comparison:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --preset moe256-active-set-128 --production-profile <schema-v2-profile> \
  --route-dtype bf16 --warmup 5 --runs 21
```

On the unchanged production-auto plan, static stage windows improved the
active-set-128/64/32 cases by 28.14%/10.38%/1.66% and tiered-hotspot by 7.23%.
Uniform and long/short cases were not overridden and differed by -0.32%/-0.08%.
The captured DSV4 distribution changed by -0.22%. NUMA1 independently measured
+10.58%/+29.71% for active-set-64/128, so the policy is enabled for both exact
rank CPU sets but is not generalized to other machines or shapes. Full results
and applicability limits are recorded in
[`results/amazon_192c_static_stage_window_policy.md`](results/amazon_192c_static_stage_window_policy.md).

That policy originally had no route band below 49 routes, so every expert with
`M < 49` retained the operator-wide baseline geometry. Because the kernel loops M
panels outside and N tiles inside, each additional M12 panel walks the whole
packed-B window again, so useful packed-B bandwidth is governed by the per-thread
window `window_bytes / threads`. At `M=28` the effective re-read factor falls
from 2.97 to 1.22 as that per-thread window shrinks from 4 MiB to 0.25 MiB,
matching the `ceil(M/12)=3` upper bound at 4 MiB; four team widths agree within
3.0-5.6% at equal per-thread window, so adding threads and shrinking the window
are interchangeable and width is second order. The `M=12` control varies by 0.7%
because a single M panel has no packed-B reuse. PMU confirms the mechanism
directly: `l2d_cache_refill * 64` over compulsory packed-B is 1.02 for the `M=12`
control and 1.87 -> 1.19 for `M=28` as the window shrinks, matching the
wall-clock factor within 5%. Sweep it with:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_heterogeneous_overlap.py \
  --output /tmp/short_route_windows.json --cpu-ids 0-95 \
  --hidden-size 4096 --ffn-hidden-size 512 --num-experts 224 \
  --big-experts 28 --big-routes 317 --big-core-splits 56 \
  --small-experts 192 --small-routes 28 --small-threads 4 \
  --small-lane-sweep 24 --small-window-sweep 0,2,1,0.5,0.25
```

Since the per-thread window is the invariant, the policy takes it as its input
unit and lowers it to per-range bytes once, where the team width is known. That
collapsed the calibrated table from 26 per-range numbers to 8 per-thread windows
plus 6 cells one factor-of-two step away. The old boolean selector merely named
two endpoints of this same geometry: an 8 MiB W13 matrix with `R13=2` has
`g=4 MiB`, while `R13=1` has `g=8 MiB`. They are now ordinary exact identities.

The `13 <= M <= 48` band then landed as the production default, at 0.25 MiB per
thread for widths 1/2/4 and 0.125 MiB at width 8. Default against default it
gives +7.59% on `dsv4-real-2048-seq70` and +24.29% on `moe256-uniform`; three
presets with no in-band expert move within ±0.33%, which is under the noise floor
set by the policy-free `legacy` variant. `moe256-uniform` is `M=48`, exactly one
route below the old first band, which is why the gap was that large.

The current default `amazon_c5_192c_tp4_f512_v4` then filled the `49-95` band at
1, 2 and 4 threads, which V1 had calibrated at 8 threads only, and split the
`144-287` band at 216 routes. Both matter because the baseline `R13=2` geometry is
itself a per-thread window of `4 MiB / t`, so it is 32x to 8x too large at narrow
widths and worth 3.39x, 2.94x and 1.65x of isolated bandwidth at `M=72`. The same
identity explains why widths 16 and 32 retain the baseline: `4 MiB / 16` and
`4 MiB / 32` already land near the measured optimum, leaving only 0.8-8.5% there,
and per-thread windows stop being transferable above 8 threads anyway, where the
width itself costs 13-39%.

The `144-287` split follows a threshold the model predicts. Each thread scans all
of A inside one range, so the per-thread scan is `2*M*K`, and the optimum saturates
exactly where that fills private L2: `M=256` for W13, `M=2048` for W2, a ratio of
`H / F = 8`. The old band used 0.125 MiB per thread across its whole range while
everything from `M=224` up wants 0.5 MiB, which at `M=256` cost 61% at one thread
and 23% at four. That also explains the shape of `omega*(M)` overall: it is flat
and noisy below the threshold, where A stays resident and the objective is nearly
flat, then climbs once A no longer fits.

One calibration is still open, and one gap is deliberate. W13 and W2 share one
per-thread window; a two-dimensional sweep found W13 is the strong axis, where one
step off the peak costs 3-30%, while W2 varies under 1.5% at the peak W13, so
sharing does not materially cost anything. That sweep also reproduced the
calibrated pair exactly at M=28, M=120 and M=320, independently validating both
the table and the per-thread parameterization. Widths above 8 remain uncovered by
choice as described above. Note that widening a band's coverage set flips
`can_use_full_workload_anchor` in the cost model, which can move the chosen shape
even where no window value changes, so every extension needs its own A/B. Sweep
the two axes with `--small-window-sweep` crossed with `--small-w2-window-sweep`,
and reproduce the A/B with:

```bash
FUSED_CPP_MOE_HUGETLBFS_PATH=/dev/hugepages-32M \
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_short_route_stage_windows.py \
  --preset dsv4-real-2048-seq70 --warmup 5 --runs 31
```

Full tables, the `M=12` control and the not-established list are in
[`results/amazon_192c_short_route_stage_windows_20260806.md`](results/amazon_192c_short_route_stage_windows_20260806.md).
The default-on integration rerun, with no enable flag, measured +30.58%/+30.34%
for active-set-128 on NUMA0/NUMA1 and +0.01% for the non-overridden uniform
control.

The exact-range thread/weight mapping measurements are recorded in
[`results/amazon_192c_thread_weight_working_set.md`](results/amazon_192c_thread_weight_working_set.md).

The fixed-active-B route-fragmentation measurements are recorded in
[`results/amazon_192c_fixed_active_b_fragmentation.md`](results/amazon_192c_fixed_active_b_fragmentation.md).

The M12 streaming-weight LLC-pollution measurements are recorded in
[`results/amazon_192c_m12_llc_pollution.md`](results/amazon_192c_m12_llc_pollution.md).

The four-core M12 refill-contention experiment is recorded in
[`results/amazon_192c_short_refill_contention.md`](results/amazon_192c_short_refill_contention.md).
It separates a chip-wide high-cache-data-activity effect from an additional
topology-local L2-refill service penalty; neither is described adequately by
aggregate NUMA DRAM bandwidth alone.
