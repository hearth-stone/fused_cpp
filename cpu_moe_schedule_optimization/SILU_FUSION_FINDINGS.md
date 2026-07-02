# Fused w13-GEMM + SiLU-and-Mul (pure-asm epilogue) — Findings

## What was built
The SiLU-and-mul activation is fused into the w13 bf16 GEMM's **store epilogue**.
After the bfmmla kernel computes an 8×8 tile (interleaved as 4 gate + 4 up
columns), it computes `silu(gate)*up` at the register level and writes the bf16
`intermediate[M, F]` directly — eliminating the `gate_up[M, 2F]` fp32 buffer and
the separate activation pass.

- **Prepack**: `prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)`
  packs w13 in an interleaved layout: per 8-col N-block `[g(4b..4b+3) | u(4b..4b+3)]`
  (gate feature `f` → packed col `8*(f/4)+(f%4)`, up → `+4`). Requires `F % 8 == 0`.
- **Kernel**: `bf16gemm_k_ld_silu_poly{4,5,6}` (+ `_m{1,2,4}` tails) in
  `i8gemm/lib/bf16gemm_k.S`, generated via the existing `GEMM_BODY` macro with a
  new fused store macro (`ldc_shift=1`, `c_adv=8`). The exp is a NEON-asm port of
  `vexpq_f32_poly_impl<deg>` (range reduction + Horner + `2^n` bit-trick); silu via
  `num=g*u`, `denom=1+exp(-g)`, `out=num/denom` (fdiv).
- **Integration**: opt-in via the prepared-weights `fused_silu` flag +
  `silu_poly_degree` (default 5). Wired into the **default** per-expert path of
  `fused_moe_bf16_tiled` (activation must be `silu`); the legacy w13+activation
  path stays as fallback and numerical reference. Hierarchical/scheduled paths
  are left on the legacy path for now.

## Correctness
- 118 fused tests (kernel + e2e) pass on macOS arm64 **and** AWS Graviton (ELF
  `adrp/:lo12:` const-table path validated on real hardware).
- 573 existing MoE regression tests unchanged.
- Fused (poly5) vs baseline (`std::exp`) e2e diff is ≤ 1 bf16 ULP at the output
  magnitude (e.g. `maxdiff=0.25` at values ~32-64), i.e. pure bf16 rounding.

## Benchmark (AWS Graviton, 8 cores, `OMP_PROC_BIND=FALSE taskset -c 0-7`)
H=4096, F=512, E=8, top_k=2; median ms; speedup = baseline / fused(poly5):

| tokens | threads | base_ms | fused_p5 | speedup |
|---|---|---|---|---|
| 8    | 1 | 3.848  | 3.790  | 1.02 |
| 64   | 1 | 12.175 | 11.941 | 1.02 |
| 256  | 1 | 49.222 | 48.201 | 1.02 |
| 1024 | 1 | 182.01 | 178.20 | 1.02 |
| 1024 | 4 | 74.31  | 73.55  | 1.01 |
| 1024 | 8 | 63.14  | 62.59  | 1.01 |

poly4/5/6 are within noise of each other (the exp is cheap relative to the GEMM).

## Conclusion
The fused path delivers a **~1-2% end-to-end speedup**, exactly as the earlier
stage-breakdown predicted: the SiLU-and-mul activation is only ~2% of e2e and the
two GEMMs dominate (~85%). The win comes from removing the activation pass and the
`gate_up[M,2F]` fp32 buffer traffic. In the default (1-thread-per-expert) path
there is no inter-stage barrier to remove, so the gain is purely activation +
buffer traffic.

**This is a polish/building-block, not a major lever** — consistent with the
prior finding that post-w13 fusion is a few-percent item while the GEMMs (and MoE
scheduling) are where the real time goes. The larger remaining opportunity for the
fused epilogue is the **team (N-split) path**, where fusing would also drop the
w13→activation barrier (up to ~6-19% of the small-M/high-thread regime per the gap
analysis) — deferred, along with w13 bias and the scheduled/async paths.

## How to reproduce
```bash
# build (macOS or AWS Graviton)
python setup.py build_ext --inplace
# correctness
pytest -q tests/test_moe_fused_silu.py tests/test_moe_fused_silu_e2e.py
# benchmark (pin the process)
OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
  python cpu_moe_schedule_optimization/benchmarks/bench_fused_silu.py
```

---

# Update: N-split single-expert multithreading for the fused path

The fused path was previously wired only into the default per-expert scheduler
(one thread per expert), so when **active experts < cores** the extra cores sat
idle. This update lets a **team of G threads cooperate on a single expert** via
N-split (column split of the interleaved 2F weight), reusing the existing
hierarchical team infrastructure. The fused w13 stage computes each member's
disjoint feature-column slice and writes `intermediate` directly, collapsing the
w13 GEMM + separate activation into one stage and dropping one barrier.

- **Fused w13 is forced to kN** (N-split); the w2 stage keeps its existing
  dynamic split (kN in ~all MoE shapes). No asm / w13-kernel changes.
- Opt-in via the same `fuse_silu=True` weights + the hierarchical N-split env
  (`FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=1`, `..._N_SPLIT_CORE_BASES`,
  `..._N_SPLIT_GROUPS_PER_PARTITION`).
- Correctness: `tests/test_moe_fused_silu_nsplit.py` — the N-split assembly is
  **bit-identical** to the single-thread fused kernel (108 cases), and the
  threaded end-to-end MoE matches the non-fused baseline within tolerance (9
  cases). Green on macOS arm64 and AWS Graviton (573 regression + 235 fused).

## Benchmark (AWS Graviton, 8 cores, `OMP_PROC_BIND=FALSE taskset -c 0-7`)
H=4096, F=512. One group of G threads over each active expert (E = 8/G experts).
Median ms. `default_fused` = the previous 1-thread-per-expert fused path.

| rows/expert | G | nsplit_fused | nsplit_legacy | default_fused | fused/legacy | **nsplit vs default** |
|---|---|---|---|---|---|---|
| 128  | 8 | 2.221  | 2.241  | 25.282  | 1.009 | **11.4×** |
| 512  | 8 | 29.140 | 29.507 | 104.392 | 1.013 | **3.6×** |
| 1024 | 8 | 58.436 | 58.333 | 204.585 | 1.00  | **3.5×** |
| 1    | 8 | 0.236  | 0.248  | 0.574   | 1.049 | **2.4×** |
| 128  | 4 | 4.144  | 4.216  | 27.232  | 1.017 | **6.6×** |
| 1024 | 2 | 137.1  | 138.5  | 227.4   | 1.010 | **1.7×** |

## Conclusion
1. **Single-expert N-split multithreading is the real win** — it lets the fused
   path use otherwise-idle cores in the few-active-experts regime, giving
   **~2.4×–11×** over the previous one-thread-per-expert fused path. This is the
   valuable lever (using more cores), not the fusion arithmetic itself.
2. **Fusion on top of N-split** adds a further **~1-6%** (`fused/legacy`),
   largest at tiny M where the removed w13→activation barrier + activation pass
   is a bigger fraction. Consistent with the earlier finding that the activation
   is a small slice of e2e.
3. Under N-split the A-pack is duplicated G× (each team member repacks the full
   `M×H` A into its own buffer). This affects **both** fused and legacy equally
   (same team infra), so it cancels in `fused/legacy` — but it is a real absolute
   cost. Eliminating it (pre-pack A once into a shared read-only buffer) is the
   next opportunity, and it is specifically the N-split regime where it matters.

## How to reproduce (N-split)
```bash
OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
  python cpu_moe_schedule_optimization/benchmarks/bench_fused_silu_nsplit.py
# correctness
pytest -q tests/test_moe_fused_silu_nsplit.py
```


---

# packA 融合(gather→w13 A、w13 epilogue→w2 A)

## 动机
单专家 N-split 下,GEMM kernel 在内部把 A strided 重排进 per-thread 的
`a_reorder`(尺寸 `G × ceil8(rows) × max(w13.K_pad,w2.K_pad) × 2`,G=8、M=2048
时约 **268MB**)。这份 scratch 每次 forward 都 `malloc` + zero-init,构成单专家
8 线程 non-GEMM 的绝对大头。实测 G=8 M=2048:`scratch_alloc ≈ 96ms`(占 e2e
~64%),远超所有 GEMM 的 ~25ms。

## 做法(两部分,`FUSED_CPP_MOE_FUSED_PACKA`,默认 ON)
- **Part 1(gather → w13 packA)**:`gather_pack_a_reorder_m8` 在收集 token 时
  直接写成 reorder-m8 packed 布局,w13 用 `bf16gemm_k_ldp_silu_poly*`(读预打包
  A、跳过 kernel 内 repack)。消除行主序 `scratch.input` 中转 + 独立 pack。
- **Part 2(w13 epilogue → w2 packA)**:新增 fused_cpp 自有 asm
  `csrc/bf16gemm_silu_packc.S`(不改 i8gemm),`STORE_C_SILU_POLY*_PACKC_8`
  把融合 SiLU 的 8×4 输出直接按 reorder-m8 写成 64 连续字节;w2 用
  `bf16gemm_k_ldp`(packed-read fp32)直接读该 packed intermediate,不再 repack。
  仅当 w2 无 bias 时启用(packed-read w2 无 bias 变体),否则退化为 Part-1-only。

两部分都启用后,`a_reorder`(及行主序 `input`、非融合才用的 `gate_up`)在
fused-packa 路径**不再分配**,scratch 只剩一份 ~16MB 的共享 packed_a +
intermediate + down。

## asm 关键点(零 i8gemm 改动)
- packed-C 复用 `GEMM_BODY` 不变:M-block 步长 `LDC_B<<3 = ldc*16 =
  8*F_pad*2` 已与 reorder-m8 一致;只把 `c_adv` 从 8 改成 **64**(一个 reorder
  K-block = 32 uint16 = 64B)。存储把 8 行 pair 用 `ins v0.d[1],v1.d[0]` 合成
  q 后连续 `str`。
- `.macro`/`#define` 是按编译单元作用域,新 .S 与 `bf16gemm_k.S` 各自成 .o,
  宏名可重复无冲突;只有导出 label 用 `_packc` / `bf16gemm_k_ldp` 新名。
- reorder-m8 天生 8 行块:m8 kernel 通过 padding 到 8 处理任意 M(bit-identical,
  含 m=3/37 尾块),故 packc **只做 m8**,不需要 m1/m2/m4 尾块变体(那需要新的
  部分行 load,且现有 packed-read 尾核与 8 行布局不兼容)。

## 正确性
`tests/test_moe_packa_fusion.py`:gather_pack 对拍 gather+pack_a_reorder_m8
(36)、packc 对拍 row-major fused+pack(36)、full MoE packa on/off bit-identical
(9)。全套 MoE 测试 local + AWS Graviton **各 889 pass + 1 skip**。packa on/off
输出逐位相同(`torch.equal`)。

## 收益(AWS Graviton, 单专家, G=8, pinned, H=4096 F=512)
| M | packa OFF e2e | packa ON e2e | 加速 | scratch_alloc OFF→ON |
|---|---|---|---|---|
| 512  | 35.6ms | 7.1ms | **5.0×** | — |
| 2048 | 139.8ms | 38.4ms | **3.6×** | ~88ms → ~9.5ms |

alloc 从 ~88ms(268MB a_reorder + 清零)降到 ~9.5ms;e2e 3.6–5.0×。这是极端
"全部 token 压 1 专家"场景;真实 MoE 中 token 分散、`max_expert_rows` 小,
a_reorder 本就较小,收益按比例递减,但 packed 路径无回归(输出逐位一致)。

## 复现
```bash
# 关闭 packa 走 fallback 对比:FUSED_CPP_MOE_FUSED_PACKA=0
# alloc 分解:FUSED_CPP_MOE_STAGE_TIMING=1(默认关)
pytest -q tests/test_moe_packa_fusion.py
```


---

# packA 尾核融合(按 tail=rows%8 分派,消除小 M padding 回归)

## 动机
上面的 packA 融合里,packc/packed GEMM 把每个专家的行数一律 pad 到 8 跑 m8。
对 decode 这种"每专家 1–2 行"的小 M,padding 到 8 浪费大量 compute。实测多专家
小 M(avg 1.5 行/专家,8 线程)packa **ON 反而比 OFF 慢 26%**。

## 关键测量(AWS Graviton,单线程,GEMM-only,w13 K=4096 N=1024)
padding 到 8 跑 m8 vs 尾核只算 mr 行:
| 尾行 M | 尾核(repack) | pad→8(m8) | 说明 |
|---|---|---|---|
| 1 | 170us | 411us | 尾核 2.4× |
| 2 | 170us | 411us | m1/m2 同价(bfmmla 2 行粒度) |
| 3 | 341us(m2+m1) | 411us | pad→4(m4=224)更优 |
| 4 | 224us | 411us | m4 |
| 5 | 370us(m4+m1) | 411us | |
| 6 | 371us(m4+m2) | 411us | |
| 7 | 548us(m4+m2+m1)| 411us | pad→8 更优(尾核拆 3 次) |

## packed 世界的尾核实测(Option X:64B 步进只读前 mr 行,无 repack)
把尾核改成直接读 gather 预打包的 reorder-m8(不 repack、不需 a_reorder):
| 尾行 | packed 尾核 | pad→8 | packed/pad8 |
|---|---|---|---|
| 1 | 241us | 412us | 0.58 |
| 2 | 240us | 415us | 0.58 |
| 3 | 277us(m4) | 416us | 0.67 |
| 4 | 277us(m4) | 413us | 0.67 |
| 5 | m4+m1=508us | 415us | **1.23 → 改 pad8** |
| 6 | m4+m2=510us | 418us | **1.23 → 改 pad8** |
| 7 | pad8=412us | 415us | 0.99 |

**关键发现**:packed 世界与 repack 世界不同。packed 尾核每个子核都要重读整块 B
(w13 的 B=8MB),所以 tail 5/6 拆成 m4+m1/m4+m2(508/510us,读 B 两次)反而比
pad→8(415us,读 B 一次)慢。故 packed 的最优表是:
**{1:m1, 2:m2, 3:m4(pad4), 4:m4, 5:m8, 6:m8, 7:m8}**(与最初按 repack 推的表不同,
tail 5/6 改 pad8)。此外 packed 尾核 M=1,2(240us)比 repack(170us)略慢——因为
64B 步进 touch 的 A cache 行是紧凑布局的 8×;但 repack 需要 a_reorder(已被消除),
所以在 no-a_reorder 路径里 240us 是最优,仍比 pad8 快 1.7×。

## 做法(Option X,gather 不变)
- gather **完全不改**(仍 `gather_pack_a_reorder_m8`,尾块按 ceil8 补零)。
- 满块:m8 packc(w13)/ m8 packed(w2)。
- 尾块:新 packed-read reorder-m8 尾核(`csrc/bf16gemm_silu_packc.S`):新增 64B 步进
  cached 宏 `LOAD_A0_B0_P{2,4}` + `COMPUTE_*_P{2,4}`(读前 mr 行、A_ADDR 前进 64B 跳过
  padding 行),packc 尾 store `STORE_C_SILU_POLY*_PACKC_M{1,2,4}`(w13)、复用
  `STORE_C_{1,2,4}` fp32 rowmajor(w2);9 个 w13 尾核 + 3 个 w2 尾核,全 `a_mode=packed`。
- C++:`packc_w13_tail_dispatch` / `packed_w2_tail_dispatch` 按 tail 表分派,w13/w2 用
  **同一策略**(intermediate 写/读行数一致);`team_fused_w13_silu_packed_packc` /
  `team_w2_packed` 改为按 `rows`(非 ceil8)在 N-slice 内跑满块+尾。
- **零 repack、零 padding compute 浪费、零额外 a_reorder**;满块热路径与 gather 全程不动。

## 正确性
`tests/test_moe_packa_fusion.py`:`test_packc_tail_dispatch_matches_pad8`(M∈{1..7,9,11,
15,16,37,45}×h,f×degree=126 例)对拍 m8-pad packc bit-identical;
`test_packa_multi_expert_all_tails`(多专家每专家行数覆盖全 tail,gpp=1/8)packa on/off
`torch.equal`。全套 MoE 测试 local + AWS Graviton **各 1015 pass + 1 skip**。

## 收益(AWS Graviton, 8 线程, H=4096 F=512, packa on/off e2e 比值,<1 = packa 更快)
| 平均行/专家 | gpp | 尾核前(全 pad8) | 尾核后 |
|---|---|---|---|
| 1.5 | 1 | **1.26(慢 26%)** | **0.95(快 5%)** |
| 4   | 1 | 1.02 | 0.86 |
| 8   | 1 | 0.97 | 0.91 |
| 1.5 | 8 | 1.01 | 0.87 |
| 8   | 8 | 0.36 | 0.84 |
| 64  | 1 | 0.96 | 0.96 |
| 2048(大 M) | 1 | 0.28 | 0.28(不变,仍 3.6×) |

小 M padding 回归被彻底消除(avg 1.5 从 1.26 → 0.95),大 M 无影响。


---

# scratch buffer 首次缺页:THP(默认开)

免清零后,packed_a/down 的成本从"串行清零"变成"gather/w2 首次写时的 mmap 缺页"
(每 4KB 页一次内核陷入 + 清零)。本机基础页 4KB、THP=madvise(不自动),故大 buffer
用 4KB 页,单专家 M=2048 每次调用 ~16000 次缺页。

对 packed_a/down/intermediate `madvise(MADV_HUGEPAGE)`(默认开,`FUSED_CPP_MOE_THP=0`
可关;非 Linux / THP=never 时安全 no-op;4KB 对齐后 2MB 对齐的内部区域收敛为大页):

| M | THP off e2e / 缺页 | THP on e2e / 缺页 |
|---|---|---|
| 2048 | 31.9ms / 16128 | **27.8ms / 1085**(-13%,15×) |
| 4096 | 68.4ms / 39996 | **55.6ms / 1702**(-19%,23×) |

单专家累计:M=2048 37→32(免清零)→**27.8**(+THP);M=4096 86.6→67→**55.6**。
跨调用 buffer 池可进一步把"首次一次"也 amortize 掉(未做)。
