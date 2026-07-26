# x86 AMX BF16 fused expert optimization TODO

This list records the remaining AMX work in dependency order. Every item must
be measured against the current automatic pattern/cache policy with identical
packed weights and inputs. Correctness is required before timing; performance
claims must include pinned 1/2/4/8-core results on `AmazonC8i8Cores` when they
change thread scheduling, and pinned one-/two-core results on
`AmazonC8i2Cores` for isolated microkernel changes.

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
- [ ] **Scratch/workspace lifecycle:** allocate persistent, uninitialized
  per-worker scratch, zero only K-tail bytes that W2 can observe, and avoid a
  second scratch buffer when a selected pattern cannot consume it.
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

## P2: improve tile and cache pipelines

- [ ] **Persistent tile state and macro-M loop:** avoid repeated tile-config and
  call-frame setup across adjacent M panels while preserving exact M tails.
  An explicit `FUSED_CPP_MOE_AMX_TILE_STATE=macro_m` path now keeps one tile
  configuration across the full M16/M32 units in each cache window and leaves
  tails on the original `per_call` path. Keep this item open, and keep `auto`
  on `per_call`, until the C8i correctness and rotated 1/2/4/8-thread sweep is
  complete.
- [ ] **B-side streaming layout:** test `TILELOADDT1`/prefetch and an N64,
  K-major packed-B superblock that feeds two adjacent N32 tiles with less
  address arithmetic and better L2 locality.
- [ ] **True K-load software pipeline:** schedule the next A/B tile loads far
  enough ahead of `TDPBF16PS` to cover load latency; verify with counters that
  it improves load/compute overlap rather than only increasing instruction
  count.
- [ ] **Dimension-aware policy calibration:** make pattern, cache-window, and
  thread decisions depend on M/H/F, route skew, and CPU model. Environment
  variables remain validation overrides, not normal runtime requirements.

## P3: broader design experiments

- [ ] **1M x 6N kernel:** quantify whether six accumulators plus one A and one B
  tile outperform `m1n4` for large N despite losing K double buffering.
- [ ] **Two-dimensional cache blocking:** jointly block M and N when the A,
  packed-B, and output working sets exceed private L2; compare against the
  existing N-window-only policy.
- [ ] **W13-to-W2 L1 fusion feasibility:** revisit only after the items above.
  With eight TMM registers, W13 gate/up accumulation, vector SiLU, and partial
  W2 accumulation compete for tile state and force spills or repeated W2 reads;
  require a traffic model showing a net win before implementation.

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
