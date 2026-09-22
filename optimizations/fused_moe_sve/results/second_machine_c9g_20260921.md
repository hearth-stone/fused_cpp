# A second machine: C9g (Neoverse-V3, 2 NUMA nodes, 128-bit SVE)

## Status

The work was ported to `AmazonC9g192Cores` and measured there at both TP shapes. Three things
came out of it: the footprint mechanism found on `Arm-codex` transfers, and its state variable
does not depend on the machine or the shape; the best lane width does **not** transfer, which
makes "2T does not pay" a statement about `Arm-codex` rather than about the kernel; and the
machine's analytic calibration now exists, accurate to 2% in the region the planner works in.

The W8A8 path is broken on this machine, and both machines became unreachable at the end of
the session - `Arm-codex` refuses the SSH key and C9g times out - so the open items below are
waiting on access, not on analysis.

## Machine and port

ARM Neoverse-V3, 192 cores in two 96-core NUMA nodes, 96 MiB of L3 per node, 193 GB per node,
**128-bit SVE** so the packed-B tile is 8 against `Arm-codex`'s 16. Bootstrapped from bare:
`build-essential`, uv, CPython 3.12, **torch 2.8.0** (2.14 requires C++20 and the extension
builds with C++17), then `setup.py build_ext`. Details in `docs/agent_remote_execution.md`.

The port immediately exercised this month's `n_tile` automation:
`fused_moe_bf16_tiled_backend_n_tile()` reports 8 here and every geometry test adapted without
an edit - `test_moe_stage_window_plan`, `test_moe_cost_model_v2` and
`test_moe_probe_event_model` pass, 285 of them including the native planner's parity test.
`test_moe_analytic_model` fails its five shadow-window tests, which are uncommitted work on the
machines and fail on `Arm-codex` too.

## The grids: windows matter much more here, and narrow lanes win

> **The width claim in this section is withdrawn (2026-09-22).** The grid ranking below
> reproduces on a rebuilt instance, but the inference from it to real layers does not. On 18
> real layers the width search never picks 2T even when allowed, and forcing 48 two-thread
> lanes loses on 18 of 18 by a median 157%
> (`c9g_width_and_panel_20260922.md`). A homogeneous full-load grid gives every expert the same
> M and loads every lane equally, which is 2T's best case; a real layer routes unevenly and the
> makespan is pinned by the heaviest of 48 narrow lanes. E13's conclusion holds on this machine
> too, for the same reason it held on `Arm-codex`. What does transfer is the footprint
> mechanism and the window value, not the width ranking.


The 2026-09-19 grid bench with the machine and the shape lifted into environment variables
(CPU range, `n_tile`, F, widths, routes, windows), 144 experts, two sessions per shape, one
node, jemalloc never-purge. Design and rules: `tmp/c9g_calibration_20260921/design.md`.

At TP4 and M = 96 a 2T lane runs **28.0 ms on full stripes and 9.4 ms with a one-tile window**;
at TP2 it is 71.8 against 16.4. On `Arm-codex` the same contrast is about 1.5x, not 3x.

With the table's windows the best width at M = 96 is **2T at 9.44 ms**, against 9.64 (4T),
10.30 (8T), 11.90 (16T) and 15.51 (32T); 2T is the best width for every M up to 192 in both
shapes and 16T only takes over at M >= 384. Without windows the order reverses and 16T wins.
On `Arm-codex` the widths sit within about 1% of each other at M = 96 and 2T was worthless,
which is why E13 closed 2T there - **that closure does not transfer to this machine**.

## The footprint mechanism transfers; its level does not

The penalty of a full stripe against the best window, indexed by rho, the weights a 40- or
96-core LLC domain holds live over its capacity:

| rho | C9g TP4 | C9g TP2 | Arm-codex TP4 |
| --- | --- | --- | --- |
| 0.25 | 0.99 | - | - |
| 0.5 | 1.08 | 1.08 | - |
| 1.0 | 1.32 | 1.30 | - |
| 2.0 | 1.98 | 2.06 | - |
| 4.0 | 2.97 | 3.12 | - |
| 8.0 | - | 4.39 | - |
| 2.29 | - | - | 1.47 |

Two shapes whose experts differ by a factor of two in weight bytes land on one curve, so **rho
is the state variable and the lane width is not** - the same conclusion P9 reached on
`Arm-codex`, now across machines and shapes. The level is a machine property: `Arm-codex` sits
lower at the same rho. Below rho = 0.5 at large M the full stripe wins by 10-25%, which is a
window's A-rescan cost with no footprint to save.

Both shapes' window tables were composed with the unchanged adoption rule
(`tp4_table.json`, `tp2_table.json`).

## Efficiency against Arm-codex

Per expert at M = 96, each machine with its own table windows on 4T lanes: **6.4 core-ms here
against 15.84 on `Arm-codex`**, about 2.5x. TP2 is about 13% more efficient per unit of work
than TP4 - at M = 96 it does twice the work in 1.74x the time - which is the argument for one
rank per NUMA node on this machine.

## Analytic calibration

`profile_analytic_services.py` on one node (96 cores, n_tile 8, widths 1-96) measured L1d 64
KiB, **L2 2 MiB per core** (`Arm-codex` has 1.25 MiB), 96 MiB of LLC per rank and DRAM from
43.6 GB/s on one thread to 410 GB/s saturated. `profile_contention_async.py` supplied the
training rows - routes 1 to 2040 on widths 1 to 96, 96 isolated points per shape - and
`build_analytic_calibration.py` fitted the triple; `fit_by_width.py` then fitted one
`(expert_fixed, route)` pair per width on that width's own rows and wrote them into
`overheads.by_width`.

| | expert_fixed | route | stage scale | training MAPE |
| --- | --- | --- | --- | --- |
| C9g TP4 | 44.3 us | 264.6 ns | 1.0965 | 10.5%, 7.1% with by_width |
| C9g TP2 | 80.2 us | 127.0 ns | 1.0688 | 11.9%, 7.2% |
| Arm-codex v9 | 210 us at 1T, 82 us at 2T | 3373 ns at 1T | 1.0 | - |

This machine's fixed costs are an order of magnitude lower. With the small-M points in, the
residual error is structured rather than spread:

| region | TP4 | TP2 |
| --- | --- | --- |
| the planner's own region (width <= 16, M >= 24) | MAPE 2.0%, worst 10.1% | 4.5%, 23.1% |
| M >= width | 4.9%, 23.0% | 6.2%, 29.3% |
| M < width, fewer than one row per core | 13.8%, 40.6% | 15.2%, 42.4% |

Two effects remain and neither is an overhead. Where **M < threads** the physics charges every
core while most have no row, so the model overpredicts by up to 42% and a non-negative overhead
cannot subtract it; the planner does not go there on this machine. Just past a **12-row panel
boundary** (M = 16 and 24 against M = 12) the sign flips to -16% and -29%, so the model's panel
quantisation is not the kernel's - that is the next thing to look at.

The per-width pairs are a level correction, not a measurement of a hardware overhead: TP4's
rise monotonically with width (22, 19, 29, 28, 29, 42, 48, 68 us) while TP2's still contain
zeros at 8T and 16T, where the fit has nothing to separate the fixed term from the per-route
one. `Arm-codex`'s own `by_width` covers only 1T and 2T and came from a victim-lane experiment,
the opposite defect.

## The native fast planner, measured here

45 real route layers, 30 repeats each: the Python planner takes 0.703 ms in median (p90 0.974,
worst 1.157) and the C++ port 0.205 ms (p90 0.233, worst 0.242), a 3.4x speedup with a much
tighter tail, and the plans are identical on all 45 layers. Of the 0.205 ms, 0.069 ms is still
Python building the cost rows and crossing pybind.

## W8A8 is broken here

`test_moe_w8a8.py::test_w8a8_plan_v2_matches_dynamic_quantization_reference` corrupts the heap.
Minimal repro, 2 experts and 48 tokens: `free(): chunks in smallbin corrupted` at h = f = 64
with 4 threads, heap corruption at h = 128 or 256, and at 8 threads it runs but returns a
different result on the second call. An ASAN build of the whole extension reports **no
out-of-bounds access** in any of those configurations while the non-determinism persists, so
the defect is a race and the corruption is its symptom.

Four consecutive calls on the same inputs:

| team width | differing rows | differing columns | largest difference |
| --- | --- | --- | --- |
| 1 | 0 of 48 | 0 of 64 | 0 |
| 2 | 45-47 of 48 | 64 of 64 | 3.0e-4 |
| 4 | 35-48 of 48 | 64 of 64 | 7.8e-4 |

A single-thread lane is deterministic and every multi-thread lane is not, with differences a
hundred times the 3e-6 the test allows. The synchronisation to look at is `Workspace::barrier`
and the three stages it separates: the per-row input quantisation, the W13 epilogue that writes
the intermediate rows the owner owns, and the W2 stage and `store_w2`, which read all rows.
Whether `Arm-codex` is affected is untested.

## Open

- The W8A8 race, and whether it is specific to the 128-bit build.
- The 12-row panel boundary in the analytic model.
- The event model's probe curves are not ported, so the planner's contention side does not yet
  work on this machine; the isolated side does.
- Both machines are unreachable as of the end of 2026-09-21.

## Artifacts

`tmp/c9g_calibration_20260921/` (design, decision, grids, tables, service probe, training
profiles, calibrations and their reports, the by-width fitter),
`tmp/native_hot_wide_20260921/` (the planner benchmark), and the W8A8 repros
`tmp/w8a8_matrix.py` and `tmp/w8a8_diff.py`.
