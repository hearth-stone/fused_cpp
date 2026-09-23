# Related work on C9g: flat queue, KTransformers' design and the planner

## Status

The related-work comparison (`cpu_moe_schedule_optimization/TODO.md`, "Related-work comparison
plan") ran on Amazon C9g NUMA0 (96 cores, SVE-128) for the TP4 expert shape (H=4096, F=512,
E=256, TopK=6), because Arm-codex performance runs are parked. Every arm uses this
repository's GEMM microkernel; only the schedule, and for K the dataflow, differ.

- **B -> C (scheduling) is conditional.** The planned schedule beats the best flat staged
  queue on uniform and moderately skewed batches (up to 13.0%), ties on the real DSV4 layer and
  on decode-like batches, and loses on two workloads: 32 equal experts (22.6%) and five
  2040-route experts among 174 tiny ones (10.5%).
- **KTransformers' design with our GEMM loses to ours** by 12.6-31.7% at its best block size
  and by 41-96% at its own N_BLOCK = 256, except on 32 equal experts, where quick itself picks a
  bad width. Dataflow and scheduling each account for part of it: on the same flat queue,
  fusing SiLU into W13 and storing routes directly is worth 7.8-47.6%.
- **No static shape reproduces the flat queue on the long/short workload.** An M split makes
  it worse (the long experts' weights are re-read per slice); what the flat queue has is a
  dynamic tail. On 32 equal experts an M split that balances the lanes beats the flat queue
  by 5.1%.
- **The production quick planner picked 1T and 2T where it should not.** Fixed: calibrations
  can name `unreliable_widths`, and the one-click calibration excludes {1, 2} by default.
- **vLLM's Arm op (A) matches this repository's kernel under a flat schedule** (within 2% on
  four of six workloads) and beats the production quick plan on the real DSV4 layer by 3.2%.
  No general kernel speedup over vLLM on Arm can be claimed.

## Setup

Arms, one process per (workload, session), shuffled per round, two sessions:

- **B, flat staged queue:** `fused_moe_bf16_tiled_vllm_staged` with A packed once per expert,
  column block swept over {L2-derived, 128, 256, 512} and reported at its best; plus vLLM's
  literal per-task A repack for reference.
- **C, planned:** production quick (`PlannedMoE`, guarded calibration = the median C9g TP4
  calibration plus `unreliable_widths` {1, 2}), quick on the unmodified calibration
  (`quick_unguarded`), full search, and fixed homogeneous widths 1-96. The registered C9g TP4
  window table applies. Reported: plan time + execution time.
- **K, KTransformers' dataflow and scheduling with this repository's GEMM** (new,
  `fused_moe_bf16_tiled_kt_staged`, below), block {128, 256, 512}.

Protocol: jemalloc never-purge, producer-hot A, two rotating cold weight copies, 3 warmups +
11 runs, fused arms bitwise equal before timing, K and hand-built split plans within the BF16
contract (atol = rtol = 7e-2), `wait_idle` before each process.

Workloads: the six 2048-token catalog presets and three decode-like batches (16, 64 and 256
tokens with uniform random top-6 routing).

## R1: flat queue against the planner (ms, mean of two sessions)

| workload | quick (w) | quick unguarded (w) | best fixed | best flat queue | flat / quick | full, execute only |
| --- | --- | --- | --- | --- | --- | --- |
| moe256-uniform | 13.65 (4T) | 14.40 (2T) | 13.41 (4T) | 15.42 (bL2) | +13.0% | 13.50 |
| moe256-active-set-32 | 11.50 (4T) | 11.46 (2T) | 9.23 (16T) | 8.91 (bL2) | -22.6% | 8.76 |
| moe256-active-set-64 | 8.68 (4T) | 10.81 (2T) | 8.60 (4T) | 9.43 (bL2) | +8.7% | 8.94 |
| moe256-tiered-hotspot | 8.94 (8T) | 9.20 (8T) | 8.84 (8T) | 9.40 (bL2) | +5.1% | 8.77 |
| moe256-long-short-bimodal | 12.25 (16T) | 12.65 (16T) | 12.04 (16T) | 10.97 (bL2) | -10.5% | 15.10 |
| dsv4-real-2048-seq70 | 14.59 (8T) | 15.08 (8T) | 13.99 (4T) | 14.44 (bL2) | -1.1% | 14.03 |
| decode-uniform-16 | 3.16 (4T) | 3.51 (1T) | 3.07 (4T) | 3.24 (b256) | +2.4% | 2.95 |
| decode-uniform-64 | 7.25 (4T) | 7.73 (1T) | 7.06 (4T) | 7.08 (b256) | -2.4% | 6.78 |
| decode-uniform-256 | 9.76 (4T) | 10.26 (1T) | 9.52 (4T) | 9.46 (b256) | -3.1% | 9.13 |

Quick's session-to-session repeat is 0.1-1.5% on the 2048-token workloads and up to 4.3% on
the small decode batches. Cells within 1% are ties.

Full search executes best or near-best on seven of nine workloads, but plans for 0.5-50 s, so
it is an offline reference, not a runtime path. On the long/short workload it is the worst
planned arm (15.10 ms).

## The 1T problem, and what the fix actually buys

A first R1 pass showed quick choosing 1T on moe256-uniform and running 2.3x slower than 16T.
That pass was invalid: the C9g tree's `stage_window_policy.py` predated the C9g window-table
registrations (only changed files had been synced), so every planned arm ran without windows.
With the table, quick no longer picks 1T on 2048-token batches; it picks 2T on three of them and
1T on the decode batches.

Fixed 1T runs 2.8-7.7x the best width on the six 2048-token workloads and ties on the decode
batches; 2T ranges from tied to 4.4x. The fix (`planner.unreliable_widths`, below) keeps both
runnable when asked for explicitly but out of the planner's own search. Frozen acceptance - never
more than 1% slower than the unguarded quick, and on moe256-uniform at least half the gap to the
best fixed width recovered - passes: the worst case is +0.4% (active-set-32, a tie), uniform
recovers 76%. Where the gain comes from:

- **execution:** active-set-64, where unguarded quick's 2T ran 22% slower than 4T;
- **planning time:** fewer shapes scored, 0.5-1.1 ms down to 0.2-0.6 ms, which is 5-10% of a
  decode batch; on decode, 1T and 4T execute in the same time.

What remains: quick still leans narrow. On active-set-32 it picks 4T (11.50 ms) where 16T runs
9.23 ms; excluding the two narrowest widths does not remove that bias.

## Shapes: does the planner need an M split?

The two workloads the flat queue wins, with hand-built Plan V2 shapes timed beside it
(two sessions, execution only):

| moe256-long-short-bimodal | ms | vs flat |
| --- | --- | --- |
| flat queue, L2 block | 10.93 | - |
| mixed widths: longs on 5x16T, shorts on 4x4T | 11.75 | +7.5% |
| quick (16T) | 11.78 | +7.8% |
| M split, 6 slices on 6x16T | 13.49 | +23.4% |
| M split, 12 slices on 12x8T | 13.63 | +24.7% |
| full | 15.29 | +39.8% |
| M split, 2 slices on 2x48T | 17.90 | +63.7% |
| N split: each long expert alone on 96T, then the shorts | 20.20 | +84.7% |

| moe256-active-set-32 | ms | vs flat |
| --- | --- | --- |
| M split, each expert in 3, on 12x8T | 8.45 | -5.1% |
| full | 8.76 | -1.5% |
| flat queue, L2 block | 8.90 | - |
| fixed 16T | 9.06 | +1.9% |
| M split, 3 slices on 6x16T | 9.52 | +6.9% |
| quick (4T) | 11.24 | +26.3% |

On the long/short workload nothing static closes the gap: splitting M re-reads each long
expert's 12 MiB of weights per slice, and a 96-thread team per long expert is worse still. The
flat queue wins with its dynamic tail - idle threads take the remaining N blocks - which a
static plan cannot express. Mixed widths come closest (7.5%). On 32 equal experts the loss is
lane imbalance, not a critical path; an M split that balances the lanes beats the flat queue.

Route-sliced plans are not bitwise equal to the unsliced fused result (1.3-1.5e-3 relative L2,
BF16 last place), because the final merge's rounding path changes; unsplit plans are bitwise
equal. Production's bounded tail repartition with route slices carries the same caveat.

## K: KTransformers' design with this repository's GEMM

`fused_moe_bf16_tiled_kt_staged` reproduces KTransformers' forward
(`kt-kernel/operators/amx/moe_base.hpp` at `f7607c0`): per-token copy into per-expert buffers,
per-expert A pack, gate and up as separate (expert, N block) GEMM tasks with BF16 outputs, a
separate SiLU x up pass, a per-expert pack of the activation, the down GEMM, and a per-token
weighted sum, each pass a flat queue ended by a barrier; tasks are claimed one at a time from a
single atomic counter in expert-major order, which is what KTransformers' `do_work_stealing_job`
does (`block = 1`). The GEMM is this repository's plain packed direct-store kernel; weights are
packed plain with the SVE tile. It matches a torch emulation of KTransformers' rounding points
(relative L2 <= 5e-3) and the fused path within the BF16 contract (4.5e-3 on the workloads);
tested on Arm-codex (SVE-256) and C9g (SVE-128).

| workload | KT best block | KT at N_BLOCK 256 | flat queue (B) | quick (C) | K -> B | B -> C | K -> C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moe256-uniform | 17.16 (128) | 22.14 | 15.44 | 13.71 | +11.1% | +12.6% | +25.1% |
| moe256-active-set-32 | 9.55 (128) | 13.88 | 8.86 | 11.22 | +7.8% | -21.0% | -14.9% |
| moe256-active-set-64 | 10.15 (128) | 14.93 | 9.38 | 8.85 | +8.3% | +5.9% | +14.7% |
| moe256-tiered-hotspot | 10.55 (128) | 15.26 | 9.37 | 8.96 | +12.5% | +4.6% | +17.7% |
| moe256-long-short-bimodal | 16.17 (128) | 24.05 | 10.95 | 12.28 | +47.6% | -10.8% | +31.7% |
| dsv4-real-2048-seq70 | 16.47 (128) | 20.68 | 14.44 | 14.63 | +14.1% | -1.4% | +12.6% |

`K -> B` is the dataflow (same scheduler, same GEMM): 7.8-47.6%, largest on the long/short
workload. KTransformers' own N_BLOCK = 256 is 1.3-1.5x slower than 128 on this GEMM and machine.

## R2: vLLM's Arm op (A)

vLLM `03f8301` `cpu_fused_moe`, isa `neon` (its Arm BFMMLA path), weights prepacked by its own
`prepack_moe_weight`, timed in the same process as B and C on the same tensors. Its OpenMP
threads are confined to node 0 by numactl but not pinned per core (`OMP_PROC_BIND=false`); a
first run with `OMP_PROC_BIND=close` was invalid - libgomp collapsed the affinity to one CPU and
every thread ran there (741.6 ms per call against 16.3 ms). Correctness against the fused result:
relative L2 1.8-1.9e-3. Two sessions, total ms:

| workload | vLLM (A) | best flat queue (B) | quick (C) | A -> B | A -> C |
| --- | --- | --- | --- | --- | --- |
| moe256-uniform | 16.05 | 15.51 | 13.71 | +3.5% | +17.1% |
| moe256-active-set-32 | 9.00 | 8.87 | 11.51 | +1.5% | -21.8% |
| moe256-active-set-64 | 9.44 | 9.42 | 8.84 | +0.2% | +6.7% |
| moe256-tiered-hotspot | 9.39 | 9.42 | 9.01 | -0.3% | +4.2% |
| moe256-long-short-bimodal | 12.31 | 11.09 | 12.29 | +11.0% | +0.1% |
| dsv4-real-2048-seq70 | 14.16 | 14.44 | 14.63 | -2.0% | -3.2% |

(A -> X is A / X - 1: positive means X is faster.)

**vLLM's Arm op is about as fast as this repository's kernel under the same flat schedule**:
within 2% on four of six workloads, 3.5% and 11.0% slower on the other two. It beats the
production quick plan on the real DSV4 layer by 3.2% and on 32 equal experts by 21.8%, ties on
the long/short workload, and loses on uniform (17.1%) and moderately skewed batches (4-7%).
No general kernel speedup over vLLM on Arm can be claimed, and the system comparison is
workload-dependent. KTransformers' unfused dataflow on the same GEMM (K) is 12-48% slower than
the flat queue, so vLLM's dataflow is much closer to the fused one than KTransformers' is.

## What this does and does not show

- One machine, one expert shape, TP4. The ranking of the planner against the flat queue is
  workload-dependent and must be stated that way.
- The flat queue's block is the best of four per workload, chosen after the fact; its default
  (L2-derived) is that best on six of nine.
- K borrows only the GEMM; it says nothing about KTransformers on its AMX path, which does not
  exist on Arm.

Lab: `tmp/c9g_related_work_20260923/` (designs, harnesses, raw JSON, `r1nowin_*` for the invalid
first pass, `r2_invalid_bind/` for the invalid first R2).
