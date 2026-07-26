# x86 AVX-512/AMX BF16 fused expert optimization TODO

This list records the remaining x86 AVX-512 and AMX work in dependency order.
Every item must be measured against the current automatic backend/pattern/cache
policy with identical packed weights and inputs. Correctness is required before
timing; performance claims must include pinned 1/2/4/8-core results on
`AmazonC8i8Cores` when they change thread scheduling, and pinned one-/two-core
results on `AmazonC8i2Cores` for isolated microkernel changes.

## P1: remove work around the matrix instructions

- [x] **W13 SiLU microkernel:** keep polynomial constants resident in ZMM
  registers, interleave two output rows so independent vector chains overlap,
  and evaluate `VRCP14PS` as an accuracy-gated alternative to `VDIVPS`.
  Preserve explicit `baseline`, `resident`, `pipelined`, and `rcp14` modes for
  same-process A/B testing. Validate degrees 4/5/6, odd M tails, every AMX
  pattern, BF16 equality/tolerance, standalone W13 throughput, and full-expert
  latency. **2026-07-20:** `resident` is now automatic and `baseline` remains
  selectable. It is bit-exact, shrinks the representative M16/degree-5 JIT
  body by 28.6%, improves the K=256 microkernel by 8.4%, and is neutral to
  +1.7% on measured H=4096 full experts. `pipelined` was exact but did not
  reliably beat resident; `rcp14` was slightly approximate and neither is the
  default. See
  [`results/amazon_c8i_2core_amx_silu_20260720.md`](results/amazon_c8i_2core_amx_silu_20260720.md).
- [x] **W2 store/merge epilogue:** remove repeated per-row address generation
  in `m1n4`; evaluate an expert-contiguous FP32 route buffer that accepts
  `TILESTORED` directly; retain direct weighted BF16 output for top-k=1. Measure
  store bandwidth and the later route-merge cost separately. **2026-07-20:**
  added cache-key-isolated `baseline`, `combined`, and `tile_store` modes.
  `combined` performs one route-address calculation for each M1N4/N64 row;
  `tile_store` writes TMM accumulators directly to expert-contiguous FP32 rows
  and merges through a precomputed flat-route-to-row map. Both are bit-exact,
  including M/N/K tails and one/two threads; top-k=1 direct BF16 keeps the ZMM
  conversion path. At K=512, rotated standalone `tile_store` improved W2 by
  1.0-14.5%; mapped merge was 0-11.8% slower and end-to-end results ranged
  from +1.7% to -4.5%.
  The stable automatic policy therefore remains `baseline`; both alternatives
  are retained for future dimension-aware dispatch. A K=32 store-dominated
  sweep explains the instability: `combined` raised effective output bandwidth
  by up to 17.8%, but wide-stride direct `TILESTORED` increased latency by
  12.8-22.8%. See
  [`results/amazon_c8i_2core_amx_w2_epilogue_20260720.md`](results/amazon_c8i_2core_amx_w2_epilogue_20260720.md).
- [x] **Weighted top-1 direct-output epilogue:** for top-k=1 route weights that
  are not known to equal one, multiply each W2 row by its route weight in ZMM,
  convert directly to BF16, and write caller output without allocating or
  merging an FP32 route workspace. Implement and measure this independently for
  AVX-512 (accumulators already in ZMM) and AMX (after the required
  `TILESTORED` scratch step). Preserve the existing exact-one
  `skip_weighted=True` fast path. **2026-07-26:** AVX-512 intrinsic/JIT and all
  three AMX patterns now have cache-key-isolated weighted-direct epilogues.
  Direct and workspace output are BF16 bit-exact; top-k>1 retains the merge
  path. Automatic mode is deliberately limited to route workspaces of at least
  256 KiB, while `FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT=1/0` preserves
  forced A/B modes. At H=4096/F=512/M=2048, AMX hot-route latency improved
  29.44→18.36 ms at 1T and 9.56→2.76 ms at 8T; balanced E8 improved
  37.57→23.43 ms at 1T and 7.32→3.39 ms at 8T. AVX-512 improved
  246.68→234.76 ms at 1T and 37.13→30.90 ms at 8T. See
  [`results/amazon_c8i_8core_p1_workspace_20260726.md`](results/amazon_c8i_8core_p1_workspace_20260726.md).
- [x] **Scratch/workspace lifecycle:** allocate persistent, uninitialized
  per-worker scratch, zero only K-tail bytes that W2 can observe, and avoid a
  second scratch buffer when a selected pattern cannot consume it.
  **2026-07-26:** top-k=1 calls with one active expert and K32-aligned
  H now read the original contiguous input directly, eliminating the input
  scratch allocation and M-by-H gather copy. On one C8i core this improved the
  output-reused direct-BF16 M512/M1024/M2048 cases by 12.4%/15.4%/17.2%. See
  [`results/amazon_c8i_8core_single_expert_1t_20260726.md`](results/amazon_c8i_8core_single_expert_1t_20260726.md).
  The W13-to-W2 intermediate now uses a concurrency-safe, grow-only pool backed
  by uninitialized BF16 tensors. Automatic mode enables it for AMX working sets
  of at least 256 KiB, while explicit `1`/`0` overrides retain A/B
  reproducibility; only the F16-to-K32 row tail is cleared. Rotated
  H=4096/F=512/M=2048 measurements improved 1/2/4/8-thread latency by
  4.9%/1.0%/4.8%/3.9%. Gathered input now uses the same concurrency-safe,
  grow-only lifecycle. Automatic mode selects it for AMX aggregate input
  scratch of at least 256 KiB and clears only the K32 row tail; AVX-512 keeps
  transient input by default. At M=2048/E2, AMX improved by
  4.9%/10.1%/12.8%/27.4% at 1/2/4/8T, while forced AVX-512 was only
  0.1%/3.7% faster at 1/8T. AMX JIT scratch sizing remains pattern-specific
  (0/2/4 KiB), so unselected tile-store buffers are not allocated, and the
  weighted top-1 path above removes the last avoidable FP32 route workspace.
  See
  [`results/amazon_c8i_8core_persistent_intermediate_20260726.md`](results/amazon_c8i_8core_persistent_intermediate_20260726.md)
  and
  [`results/amazon_c8i_8core_p1_workspace_20260726.md`](results/amazon_c8i_8core_p1_workspace_20260726.md).
- [x] **Cooperative N-split scheduling:** support 1--256 requested workers,
  retain the global expert queue for balanced/high-concurrency routes, form
  per-expert teams when active experts underfill the machine, and use sorted
  waves when one route is at least 64 rows and twice the next-largest route.
  Team workers split gather, W13 F16 blocks, and W2 N32 blocks with barriers
  between dependent phases; final token reduction order is unchanged.
  **2026-07-22:** exact 1-vs-8-thread output passed on AVX-512 and all AMX
  patterns, including H/F tails and a skewed-wave case. On an 8-core C8i,
  AMX H4096/F512/M2048 improved 26.45→5.76 ms for one hot expert and
  39.22→11.89 ms for `[1536,256,256]` routes. Balanced E2/E8 retain the
  non-wave path. See
  [`results/amazon_c8i_8core_nsplit_20260722.md`](results/amazon_c8i_8core_nsplit_20260722.md).

## P2: improve AMX tile and cache pipelines

- [x] **Persistent tile state and macro-M loop:** the explicit
  `FUSED_CPP_MOE_AMX_TILE_STATE=macro_m` path keeps one tile configuration
  across the full M16/M32 units in each cache window and leaves tails on the
  original `per_call` path. C8i correctness passed for all AMX patterns at
  1/2/4/8 threads, including direct-BF16 and tile-store output. In the rotated
  H4096/F512/M2048 sweep, `macro_m` was 0.954x/0.985x/1.005x/0.997x as fast as
  `per_call` at 1/2/4/8 threads; a reversed-order repeat reproduced the 1T
  regression and left 8T tied. Therefore `auto` remains `per_call` and
  `macro_m` remains an explicit experimental variant. See
  [`results/amazon_c8i_8core_amx_macro_m_tile_state_20260726.md`](results/amazon_c8i_8core_amx_macro_m_tile_state_20260726.md).
- [x] **N64/K32 B-side streaming layout:** the explicit
  `x86_amx_bf16_n64` backend groups two adjacent N32 blocks into one N64
  superblock and stores each K32 left/right pair contiguously. All three AMX
  patterns and N-split tails passed correctness, but C8i8 single-thread
  end-to-end changes were within about +/-1% for the automatic pattern while
  8-thread latency regressed 3.4%-9.3% at M64-2048. `m1n2` regressed by up to
  10.5%. Therefore N32 remains the automatic/default packed layout and N64 is
  retained only as an explicit experiment. See
  [`results/amazon_c8i_8core_amx_n64_layout_20260726.md`](results/amazon_c8i_8core_amx_n64_layout_20260726.md).
- [x] **B-side load hints:** after rejecting N64 as a production layout, test
  only the retained N32 B stream with cache-key-isolated `TILELOADD`,
  `TILELOADDT1`, `PREFETCHT0`, and `PREFETCHT1` variants. **2026-07-26:**
  C8i8 H4096/F512 showed a stable `TILELOADDT1` crossover at M=128:
  median latency improved 5.3%-9.7% for M128-2048 across 1/2/4/8 threads,
  while M32-64 was neutral or slower. M512 counters showed 5.3% fewer L1D
  pending-miss cycles, 80.1% fewer L1D replacements, and 7.9% fewer top-down
  memory-bound slots with unchanged instruction count. Full next-panel
  software prefetch instead added 13.7% instructions and regressed M512 by
  14.7%-22.9%. Automatic N32 policy now uses `TILELOADD` below 128 per-expert
  routes and `TILELOADDT1` at or above 128; explicit modes remain validation
  overrides. See
  [`results/amazon_c8i_8core_amx_b_load_hints_20260726.md`](results/amazon_c8i_8core_amx_b_load_hints_20260726.md).
- [x] **True K-load software pipeline:** the explicit m1n2-only
  `FUSED_CPP_MOE_AMX_K_LOAD_PIPELINE=pipelined` path ping-pongs two complete
  A/B operand banks, issuing the alternate bank's tile loads before consuming
  the current bank. **2026-07-26:** C8i8 H4096/F512 five-process repeats gave
  +0.75%/+0.66%/+0.77% median paired speedup at M64/512/2048. M2048 counters
  showed cycles -1.01%, L1D pending-miss cycles -0.90%, and top-down
  memory-bound slots -1.92% despite instructions +0.48%, confirming real
  load/compute overlap. The gain is limited to `m1n2`; automatic `m2n2`/`m1n4`
  remained 14%-17% faster end to end on the representative shapes, and exact
  H4096/F512 has no m1n2 N tail. Therefore `auto` remains `baseline` and the
  pipelined path remains an explicit experiment. See
  [`results/amazon_c8i_8core_amx_k_load_pipeline_20260726.md`](results/amazon_c8i_8core_amx_k_load_pipeline_20260726.md).
- [x] **Dimension-aware policy calibration:** backend ID 104
  `x86_bf16_auto` now keeps one AMX-compatible K32/N32 weight copy and selects
  AVX-512/AMX, effective thread count, AMX pattern, cache window, and skew-wave
  target from M/H/F, the route histogram, and a versioned CPU profile.
  **2026-07-26:** C8i8 calibration selected AVX-512 for
  `M*H*F <= 8192`, one worker for aggregate `sum(M)*H*F <= 131072`,
  H/F-specific `m2n2`/`m1n4` thresholds, and a
  `clamp(ceil(64*4096*512/(H*F)),16,256)` wave target. Tiny
  H64/F16/M1 with 8 requested workers fell from 12.5/19.0 us for forced
  AVX-512/AMX to 5.9 us auto; production H4096/F512/M64 stayed within 0.1%
  of forced AMX at 1T. Explicit ISA/pattern/cache/profile environment
  overrides remain available for validation. See
  [`results/amazon_c8i_8core_dimension_aware_policy_20260726.md`](results/amazon_c8i_8core_dimension_aware_policy_20260726.md).

## P3: improve AVX-512 register and cache schedules

- [ ] **Small-M multi-N kernels:** use the spare ZMM capacity at small exact M
  to compute multiple adjacent N blocks per A traversal. Start with W13
  multi-F16 and W2 N64/N128 candidates for M=1--4, retain current exact-M
  kernels as the baseline, and select wider kernels only where their additional
  accumulators improve end-to-end latency.
- [ ] **Bulk-M generated loop:** move adjacent full M12 panels inside one JIT
  body/cache window so they share the call frame, invariant setup, and address
  generation. Preserve the current exact-M1--11 tail specializations and
  compare separately at small K and the production H/F dimensions, where call
  overhead may have very different importance.
- [ ] **K-loop scheduling and prefetch:** test deeper K unrolling, independent
  accumulation chains where register pressure permits, and explicit packed-B
  prefetch distance. Use instruction/cache counters to distinguish useful
  load/compute overlap from extra front-end work.
- [ ] **Shape-selective AVX-512 cache blocking:** keep the current unblocked
  default while calibrating W2 N-window blocking by M/H/F and CPU model. Prior
  C8i measurements found only 2.5%-3.3% W2 gains and neutral or harmful W13
  blocking, so no global AVX-512 cache-window default should be inferred from
  the AMX policy.

## P4: broader design experiments

- [ ] **1M x 6N kernel:** quantify whether six accumulators plus one A and one B
  tile outperform `m1n4` for large N despite losing K double buffering.
- [ ] **Two-dimensional cache blocking:** jointly block M and N when the A,
  packed-B, and output working sets exceed private L2; evaluate AVX-512 and AMX
  independently against their existing N-window policies, and reject schedules
  that reduce intermediate traffic by repeatedly streaming the full expert
  weights.
- [ ] **W13-to-W2 L1 fusion feasibility:** revisit only after the items above.
  For AMX, eight TMM registers, W13 gate/up accumulation, vector SiLU, and
  partial W2 accumulation compete for tile state and force spills or repeated
  W2 reads. For AVX-512, model the cost of keeping or spilling partial W2
  accumulators across F chunks. Require an ISA-specific traffic model showing a
  net win before implementation.

## Evidence checklist

- Record every attempted kernel variant, including regressions, in
  `csrc/SDPA_VERSIONS.md` and this optimization's result reports.
- Keep the old path selectable until the replacement passes exact/tolerance,
  tail, pattern, routing, and 1/2/4/8-thread tests.
- Use rotated same-process measurements when variants can share one binary;
  report median, best, run count, affinity, CPU frequency/thermal caveats, and
  whether packing/JIT warm-up is excluded.
- Update `manifest.yaml`, this TODO, and `README.md` when a variant changes
  status or becomes the automatic default.
