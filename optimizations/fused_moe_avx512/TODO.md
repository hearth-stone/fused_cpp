# x86 AMX BF16 fused expert optimization TODO

This list records the remaining AMX work in dependency order. Every item must
be measured against the current automatic pattern/cache policy with identical
packed weights and inputs. Correctness is required before timing; performance
claims must include pinned one- and two-core results on `AmazonC8i2Cores`.

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
- [ ] **W2 store/merge epilogue:** remove repeated per-row address generation
  in `m1n4`; evaluate an expert-contiguous FP32 route buffer that accepts
  `TILESTORED` directly; retain direct weighted BF16 output for top-k=1. Measure
  store bandwidth and the later route-merge cost separately.
- [ ] **Scratch/workspace lifecycle:** allocate persistent, uninitialized
  per-worker scratch, zero only K-tail bytes that W2 can observe, and avoid a
  second scratch buffer when a selected pattern cannot consume it.
- [ ] **Two-core scheduling:** keep a persistent worker team and split a hot
  expert across cores when route skew leaves one worker idle. Compare balanced,
  skewed, and single-hot-expert routing without changing numerical order in the
  final token reduction.

## P2: improve tile and cache pipelines

- [ ] **Persistent tile state and macro-M loop:** avoid repeated tile-config and
  call-frame setup across adjacent M panels while preserving exact M tails.
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
  tail, pattern, routing, and one-/two-thread tests.
- Use rotated same-process measurements when variants can share one binary;
  report median, best, run count, affinity, CPU frequency/thermal caveats, and
  whether packing/JIT warm-up is excluded.
- Update `manifest.yaml`, this TODO, and `README.md` when a variant changes
  status or becomes the automatic default.
