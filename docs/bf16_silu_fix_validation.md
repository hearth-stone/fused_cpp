# BF16 fused W13 SiLU delivery backport

## Change

Bug fix (class B): backport the coefficient correction from research commit
`15bcaf97de92bffd99cdbbdf8e4e66d7b16a519e` to delivery base
`de5a940a7f68fbf0607081af22755f7c2f40d014`.

The cached two-row SiLU epilogue used `r * (c1 + c0 * r)` while the per-row
and small-M epilogues used `r * (c0 + c1 * r)`. Different FP32 intermediate
values could therefore round to different BF16 outputs for identical inputs
at different panel positions or in different route slices. Both paths now
use the latter expression, with `z7 = c0` and `z24 = c1`.

This delivery contains AOT kernels rather than a live Xbyak generator.
The static macro in `csrc/moe/arm/sve_bf16/kernels.S` and both checked-in AOT
files (`generated/vl128.S`, `generated/vl256.S`) are corrected together.
Each AOT file contains 64 cached row-pair sequences across exact-M9–M12
BF16/W8A16 W13 and clamped W13 kernels. The four instruction words per
sequence change as follows:

| Before | After | Correct instruction |
| --- | --- | --- |
| `0x04783302` | `0x046730e2` | `mov z2.d, z7.d` |
| `0x04783303` | `0x046730e3` | `mov z3.d, z7.d` |
| `0x65a70402` | `0x65b80402` | `fmla z2.s, p1/m, z0.s, z24.s` |
| `0x65a70423` | `0x65b80423` | `fmla z3.s, p1/m, z1.s, z24.s` |

LLVM assembly and disassembly verify these encodings. Exactly 256 instruction
words change in each AOT file; all other source lines, instruction counts,
symbol sizes and branch offsets are preserved. Kernel ABI, packed layout,
backend IDs and quick-planner configuration are unchanged. Rebuild the native
extensions to apply the change; updating Python source alone is insufficient.

## Regression coverage

- `tests/test_moe_fused_silu_e2e.py`: identical rows in 9/10/11/12/24/37-row
  calls must equal the single-row result exactly, for unclamped AOT, clamped
  AOT and unclamped static asm.
- `tests/test_fused_moe_bf16_tiled.py`: concurrent strict Plan V2 slices of
  24/20/13/7/1 rows must equal the full-expert result exactly, with the same
  three supported epilogue configurations.
- `benchmarks/bench_bf16_silu_fix.py`: deterministic single-core panel calls
  and reduced-expert DSV4 TP4 routed+shared calls, including the shared-aware
  quick planner. The benchmark records raw samples and selected geometry.

## Target validation

Validation runs on `Arm-codex-internal` in the independent directory
`/home/zhangxu/codex/bf16-silu-delivery-fix-20261009`. Existing deployments and
their local planner edits are preserved. Baseline source is the exact delivery
base above; candidate source differs by this fix, tests and benchmark only.

Both builds use GCC 13.2.0, Python 3.12, PyTorch `2.13.0+cpu`,
`FUSED_CPP_SVE_VECTOR_BITS=256` and `MAX_JOBS=16`, with the same
`setup.py build_ext --inplace` command. Tests use CPU/NUMA-local placement
(CPUs 0–7 for focused regressions, 0–39 for the complete delivery suite;
memory node 0). The full suite sets both `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` to 8: setting the latter to 1 caps PyTorch/OpenMP at one
thread and makes the two existing multi-team W8A8 tests fail with
`invalid W8A8 num_threads`. The same two cases fail on the unchanged baseline
under that restriction; this is a test-launch configuration issue.
Performance runs use core 0 for panel mode and CPUs 0–39
for TP4 rank mode, memory node 0, jemalloc with
`thp:always,oversize_threshold:2097152,background_thread:true`.

Before the fix, the 33 focused regression cases yield **30 failed, 3 passed**
on this host. This reproduces numerical mismatches through the exported
operators, rather than relying only on source inspection.

With the corrected kernels, the same focused cases yield **33 passed**
(1.69 seconds). The complete delivery suite yields **1614 passed, 1 skipped**
(45.72 seconds). The skip is the intentional native-backend-unavailable test
on a host where the native backend is available. No tolerances were relaxed.
The initial full-suite invocation yielded 1612 passed, 1 skipped and the two
W8A8 thread-limit failures described above; the corrected test-launch
configuration resolves them.

Both VL128 and VL256 generated files assemble successfully with LLVM, and
disassembly of the VL256 M12 entry verifies the corrected register operands.
The static SVE assembly also assembles successfully for AArch64. Native
runtime tests here use VL256; this record does not claim a VL128 runtime pass.

## Performance results

Five fresh-process rounds per build, alternating baseline/candidate order.
The table reports the median of per-round medians; ranges are the minimum and
maximum per-round medians. Positive change means a slowdown.

| Path | Tokens | Baseline ms (range) | Fixed ms (range) | Change |
| --- | ---: | ---: | ---: | ---: |
| Single-core AOT, clamped | 9 | 0.125975 (0.124990–0.126695) | 0.126645 (0.125650–0.126925) | +0.53% |
| Single-core AOT, clamped | 12 | 0.131735 (0.130505–0.133415) | 0.131375 (0.131040–0.132135) | −0.27% |
| Single-core AOT, clamped | 24 | 0.178255 (0.176095–0.179745) | 0.177160 (0.176530–0.179140) | −0.61% |
| Single-core AOT, clamped | 128 | 0.578145 (0.573575–0.582035) | 0.575870 (0.575065–0.578045) | −0.39% |
| Single-core asm, unclamped | 9 | 0.129245 (0.128710–0.130715) | 0.129465 (0.128960–0.130300) | +0.17% |
| Single-core asm, unclamped | 12 | 0.130410 (0.129965–0.132125) | 0.130700 (0.129915–0.131745) | +0.22% |
| Single-core asm, unclamped | 24 | 0.176810 (0.175570–0.178655) | 0.176415 (0.175390–0.178315) | −0.22% |
| Single-core asm, unclamped | 128 | 0.577430 (0.574670–0.579755) | 0.577415 (0.577085–0.577465) | −0.003% |
| TP4-rank routed+shared quick | 128 | 10.117960 (9.957240–11.618785) | 9.974205 (9.875380–10.409835) | −1.42% |
| TP4-rank routed+shared quick | 512 | 19.102645 (19.043560–19.615855) | 19.074195 (18.982370–19.272395) | −0.15% |
| TP4-rank routed+shared quick | 2048 | 69.951805 (69.703485–74.962060) | 69.679570 (69.403355–70.040565) | −0.39% |

These measurements show no material regression in the tested configurations.
The small differences are within fresh-process variability; they are not a
speedup claim. Every benchmark process also checks finite output and exact
repeatability for its last measured invocation.

Panel mode: H=512, I=64, one expert, TopK=1, one thread. TP4-rank mode:
H=4096, I=512, 20 routed experts plus one shared expert, TopK=6, 40 threads.
Both builds use the same calibration JSON. Every TP4-rank case selects
`cpp_shared_quick`, strict Plan V2, shape `(20, 20)`, early merge enabled,
with 18 strict candidates and no dynamic/tail-repartition candidates.
Backend N tile is 16. Per-expert full W13/W2 stages contain 8,388,608 and
4,194,304 bytes. Selected geometry is
`(t, w13_window_tiles, w2_window_tiles, R13, R2) = (20, 0, 0, 1, 1)`;
maximum per-thread owner windows are 524,288 and 212,992 bytes respectively.
Zero window fields mean `full_n_team_stripes`, with a potentially short final
owner stripe. The shared entrypoint includes planning, combined routing,
kernel execution and merge; weight preparation is outside timing.

All raw samples, calibration and logs remain in the remote task directory's
`results/`, and were copied locally to ignored
`tmp/bf16_silu_fix_20261009/`. `baseline_regressions.log`,
`candidate_regressions.log` and `candidate_suite_fixed_threads.log` are the
correctness evidence; `{baseline,candidate}_{panel,shared}_{jit,asm}_*.json`
(for applicable mode/implementation pairs) contain the five timing rounds.

## Reproduction

Set `SOURCE` to the selected checkout and `PY` to the shared vLLM environment's
Python. Start a fresh process for each build so native modules cannot be reused.

```bash
cd "$SOURCE"
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=16 "$PY" setup.py build_ext --inplace
PYTHONPATH="$SOURCE/src:$SOURCE" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=FALSE \
  numactl --physcpubind=0-7 --membind=0 "$PY" -m pytest -q \
  tests/test_moe_fused_silu_e2e.py tests/test_fused_moe_bf16_tiled.py \
  -k 'panel_rows_match_single_row or route_slices_match_full_experts'
PYTHONPATH="$SOURCE/src:$SOURCE" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=FALSE \
  numactl --physcpubind=0-39 --membind=0 "$PY" -m pytest -q -rs
```

For performance, use the same rank-local calibration JSON for baseline and
candidate. Alternate baseline/candidate order over five fresh-process rounds.
Panel mode uses 10 warmups and 50 samples per case; shared mode uses 5 warmups
and 20 samples per case. Compare the median of the five per-round medians and
retain their ranges to expose process-to-process variability.
The change is a correctness fix, with no expected instruction-count reduction.
The performance check screens for regressions, not a speedup claim: a
repeatable TP4-rank median slowdown above 5% requires investigation before
closing validation. Panel microseconds are interpreted with the recorded
fresh-process variability.

```bash
PYTHONPATH="$SOURCE/src:$SOURCE" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 LD_PRELOAD=/usr/lib64/libjemalloc.so \
  MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true \
  numactl --physcpubind=0 --membind=0 "$PY" \
  benchmarks/bench_bf16_silu_fix.py --mode panel --impl jit \
  --warmup 10 --samples 50 --output "$OUTPUT"
# Repeat panel mode with --impl asm for the unclamped static path.
PYTHONPATH="$SOURCE/src:$SOURCE" OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 \
  OPENBLAS_NUM_THREADS=1 LD_PRELOAD=/usr/lib64/libjemalloc.so \
  MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true \
  numactl --physcpubind=0-39 --membind=0 "$PY" \
  benchmarks/bench_bf16_silu_fix.py --mode shared --impl jit \
  --profile "$PROFILE" --warmup 5 --samples 20 --output "$OUTPUT"
```

The shared benchmark uses H=4096, I=512, 20 routed experts plus one shared
expert, routed TopK=6 and 40 threads. It represents the DSV4 TP4 rank matrix
dimensions with reduced experts and synthetic balanced routing; it does not
measure model-level serving performance or MMLU accuracy. Native runtime
validation is VL256 on this host; VL128 receives assembly/encoding checks.
