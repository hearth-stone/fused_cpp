# SVE fused MoE experiments

This directory contains SVE fused-MoE features and standalone experiments. The
weighted route-merge U1 kernel and async ready-token merge are enabled by
default for their supported SVE paths; neither changes the fused-MoE API.

## Xbyak exact-M compute kernels

The default one-chunk SVE path generates W13 fused-SiLU/packC, W2 FP32, and W2
FP32 direct-route kernels with the pinned `xbyak_aarch64` submodule. The GEMM
body retains the production static-assembly K8 granularity and accumulator
mapping, but the first generator does not yet reproduce the static M8/M4/M2
double-buffered load/compute software pipeline. The generated tail specializes
every logical M=1..12: its compute height is
`2 * ceil(M / 2)`, while its packed-A height remains eight rows for M<=8 and
twelve rows otherwise. This removes the static M4/M8/M12 bucket overcompute
without changing packed weights, intermediate layout, or public APIs.

Use `FUSED_CPP_MOE_SVE_IMPL=asm` for the static reference and
`FUSED_CPP_MOE_SVE_IMPL=jit` for strict validation of the generated W13 and
FP32-W2 surfaces. The default `auto` selects generated code where supported and
retains assembly for Kc and the non-default epilogues documented in
`csrc/moe/README.md`.

The A/B benchmark keeps both implementations in one process but measures
steady blocks. After each implementation switch it executes one unmeasured
transition call, preventing code-switch I-cache replacement from being charged
to the timed sample:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_xbyak_exact_m.py \
  --routes 5,6,9,10,12,192,2040 --threads 1,4,8,16,32,64,96 \
  --warmup 3 --runs 11 --switch-period 5
```

With H=4096, F=512, eight distinct experts, and split-W13, the 192-core host's
NUMA0 gained 11.75% for M5/M6 and about 10% for M9/M10 at 1T. M12 was within
0.54% of static assembly. Across M192/M2040 and 1-96T, every measured point was
within 0.86%, with no long-route throughput regression. The 8-core host gained
roughly 5-12% on those exact tails; M12 and long routes remained within about
1.3%. Full commands and tables are in
[`results/amazon_8c_192c_xbyak_exact_m.md`](results/amazon_8c_192c_xbyak_exact_m.md).

## SVE weighted route merge

`csrc/moe/arm/sve_bf16/route_merge.cpp` provides the SVE weighted merge for the
token-major `[tokens, top_k, hidden]` route buffer. Fixed `top_k=2/4/6/8`
dispatches to compile-time templates. Template recursion first forms adjacent
weighted pairs and then a power-of-two-prefix binary tree; for example,
top-k=6 is `((0+1)+(2+3))+(4+5)`. Other positive values use a runtime slot
loop that retains one token row's accumulator vectors in SVE registers and
preserves slot order. U1, U2, and U4 process one, two, or four independent
hidden-axis vectors per loop. The existing `top_k=1, skip_weighted=true` path
still writes the W2 result directly during scatter and bypasses merge entirely.

The SVE backend defaults to U1, which keeps the accumulator in SVE registers
and writes BF16 directly without allocating a thread-local FP32 row. Select a
different unroll, or restore the sequential accumulator explicitly, with:

```bash
FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL=0  # sequential compatibility path
FUSED_CPP_MOE_SVE_ROUTE_MERGE_UNROLL=2  # or 1 / 4
```

`FUSED_CPP_MOE_SVE_ROUTE_MERGE_TREE_UNROLL` remains a compatibility alias when
the new variable is unset. Fixed templates change FP32 association relative to
the sequential baseline; the dynamic fallback does not. The standalone
benchmark checks both policies against exact scalar references for FP32 and
BF16 route buffers. The E2E benchmark uses a preplanned async schedule by
default, avoiding planner time in the measured operator:

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

numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_route_merge_e2e.py \
  --path async --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 8 --top-k 6 --threads 96 --warmup 5 --runs 31
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
24 threads per expert, and the current two-range split-W13 policy:

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
`PLDL1STRM`/`PLDL2STRM` variants. The production symbol is used only as the
bit-exact correctness reference. Every timed call consumes a distinct cold
4 MiB packed-B matrix, and a 4 KiB guard after every copy prevents the final
software prefetch from touching the next invocation's weight.

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
The benchmark entries remain standalone; production uses separate generic-Kc
symbols for M12/M8/M4/M2, with M1 predicated through M2.

## Production Kc path

The production SVE backend now packs B as `Kchunk -> Ntile -> K4` and dispatches
W13 and W2 through `moe_sve_kc_kernel_m12/m8/m4/m2`. This applies to normal,
2D, scheduled, async, and vLLM-staged execution, including direct FP32/BF16
route stores. The final K chunk alone runs SiLU or the selected W2 epilogue;
earlier chunks preserve FP32 accumulators in thread-private scratch. NEON and
the legacy SVE symbols remain compiled, but only NEON is the runtime fallback;
the legacy SVE symbols are standalone controls and are not fed production Kc
packed weights.

One Kc is used by every Mr because a prepared weight has one physical layout.
The production default is one full K chunk:

```text
Kc = K
```

`FUSED_CPP_MOE_SVE_KC` explicitly pins an 8-aligned Kc. Setting
`FUSED_CPP_MOE_SVE_KC_L1_PERMILLE` explicitly enables the calibrated L1
selector, where `bytes_per_K = 2 * (12 + n_tile)` and the environment value is
the permille of detected L1D available to the M12 A+B window. L1D is read with
`_SC_LEVEL1_DCACHE_SIZE`, with a 64 KiB fallback. The historical 49% selector
maps to Kc=800 on the SVE128 192-core host and Kc=568 on the SVE256 8-core
host. Both controls are read once at process start; they must be set before
packing weights and must remain unchanged while those weights are used.

With an explicit calibrated Kc, the isolated cold-B result is positive for
every Mr on the 192-core host
(roughly +6% to +13%). On the 8-core host, Kc=568 is the maximin compromise:
M12/M4/M2/M1 improve by about 2.23%/2.35%/0.95%/0.58%, while M8 regresses by
about 1.48%. Production split-W13 E2E improves by about 2.0% at route=12 and
3.5% at route=2040 for two 4-thread experts on the 8-core host.

The isolated result must not be extrapolated to a saturated expert wave. On
NUMA0 of the 192-core host, 24 concurrent 4-thread experts changed from
0.599 to 0.662 ms at route=12 and from 30.508 to 30.895 ms at route=2040.
Sweeping Kc through 1024/1536/2048 reduced but did not reverse this loss. Cold
B is then memory-bandwidth dominated, so A-side L1 residency has little value
while partial-C instructions remain. Cost tables must therefore be regenerated
with the production path. On 2026-07-19 the default was restored to one K
chunk; split-K remains an explicit process-level experiment for calibrated
low-pressure deployments. A future selector would need concurrent memory
pressure rather than isolated Kc timing alone.

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

## Threaded split-W13 working-set experiment

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

All modes use two W13 ranges. The instantaneous packed-B working set is
`active_experts * max(W13_chunk_bytes, W2_bytes)`. The runner also records the
largest per-thread N stripe, full wall time, aggregate TFLOP/s, and W13/W2 stage
times. Each invocation uses a distinct weight copy.

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/run_thread_weight_working_set.py \
  --output /tmp/split_w13_thread_working_set.json \
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
  --output /tmp/split_w13_expert_team.json \
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
  --output /tmp/split_w13_fixed_active_b_fragmentation.json \
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

## Configurable packed-B windows

The production fused SVE expert accepts `weight_window_bytes` on the normal,
scheduled, and async entrypoints. `None` reads
`FUSED_CPP_MOE_WEIGHT_WINDOW_BYTES`, zero keeps the existing policy (two W13
ranges and one W2 range when split-W13 is enabled), and a positive value limits
the nominal packed-B bytes in every sequential W13 and W2 N range.

For a GEMM with packed dimensions `(K, N)` and SVE BF16 N tile `v`, one packed-B
tile contains `2*K*v` bytes. The implementation computes

```text
max_tiles = max(1, floor(weight_window_bytes / (2*K*v)))
ranges = ceil((N/v) / max_tiles)
```

and balances whole N tiles across those ranges. Therefore a target smaller than
one packed-B tile rounds up to one tile. For TP4 `H=4096, F=512`, the current
4 MiB stage corresponds to W13/W2 range counts `2/1`; 2 MiB gives `4/2`, and
1 MiB gives `8/4`.

Each range still uses the existing N-split team and the same assembly kernel.
There is no barrier between adjacent ranges. W2 owner-scatter mirrors every
window's potentially discontiguous column ownership, so the existing
W2-to-scatter barrier remains elided. Consequently the byte target is a loop
ordering and nominal active-range bound, not a strict synchronized cache
residency limit. A smaller range also caps useful team width at its N-tile
count; the planner must account for this before selecting very small windows.

The current planner does not choose this experimental variant automatically.
Old split/no-split profiles describe only the 4 MiB-equivalent policy and must
not be reused to score 1/2 MiB windows. Compare explicit sizes with:

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

The split-W13 thread/weight mapping measurements are recorded in
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
