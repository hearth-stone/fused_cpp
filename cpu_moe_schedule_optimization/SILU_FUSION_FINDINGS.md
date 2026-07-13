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


---

# scratch 后端(malloc/THP/hugetlb)+ 全局 buffer 池

把 scratch 内存后端做成运行时可选的 `backend_allocator`(取代之前的
madvise-on-malloc),并加了一个 per-calling-thread 的持久池(grow-only、跨调用复用)。

## 后端(env 选,默认 THP)
- `FUSED_CPP_MOE_THP=0`:`operator new`(4KB 页)。
- 默认 / `FUSED_CPP_MOE_THP!=0`:匿名 `mmap`(2MB 取整)+ `madvise(MADV_HUGEPAGE)`。
- `FUSED_CPP_MOE_HUGETLB=1`:`mmap(MAP_HUGETLB)`,页大小 `FUSED_CPP_MOE_HUGETLB_MB`
  (默认 32MB),池耗尽时安全回退到同长度 THP-mmap。

### THP vs hugetlb 实测(单专家)
| M | malloc 4KB | THP 2MB | hugetlb 32MB |
|---|---|---|---|
| 2048 | 32.4ms / 15983 faults | 27.8ms / 119 | 28.3ms / 17 |
| 4096 | 68.8ms / 40351 | 55.0ms / 234 | 55.1ms / 25 |

**THP ≈ hugetlb(e2e 基本持平)**;hugetlb 缺页更少但不转化为 e2e 优势,还要预留固定
页池。1GB 权重顺序流式 GEMM 三者也完全一致(顺序访问的 TLB miss 被硬件预取隐藏,
大页无用)。结论:**THP 是默认;hugetlb 仅作 opt-in**。

## 全局 buffer 池(默认)
`HierarchicalScratchPool`:per-calling-thread、grow-only、跨调用复用,按 group_size
keying(barrier 尺寸变则重建)。首次分配付一次 mmap+首触缺页,之后调用复用 warm 页
→ **稳态零缺页**。correctness:packed_a/down 每次被全量覆盖(无需清零);intermediate
每次 memset(便宜、warm),防跨形状复用残留(虽 fused 路径 F 必为 8 的倍数、无 feature
padding,memset 作安全兜底)。

### 池收益(单专家,默认 THP+池)
| M | 首次调用缺页 | 稳态缺页/次 | e2e |
|---|---|---|---|
| 2048 | 184 | **0.0** | **26.4ms**(THP 无池 27.8) |
| 4096 | 342 | **0.0** | **52.6ms**(THP 无池 55.6) |

## 单专家 e2e 累计
- M=2048:fallback ~122 → packa 38 → 尾核 37 → 免清零 32 → THP 27.8 → **池 26.4ms**(4.6×)
- M=4096:~241 → … → THP 55.6 → **池 52.6ms**(4.6×)


---

# SVE M12 SiLU: reciprocal refinement 替换 FDIV(2026-07-10)

## 实现

在 M12 fused SiLU epilogue 中保留三种可在同一二进制内切换的除法路径:

- `FUSED_CPP_MOE_SILU_RECIP_NR=0`:精确 `fdiv`,默认。
- `=1`:`r=frecpe(d); r*=frecps(d,r); y=num*r`。
- `=2`:在 NR1 后再执行一次 `r*=frecps(d,r)`。

只替换 M12 主路径;M8/M4/M2/M1 尾块继续使用精确 `fdiv`。测试固定使用
split-W13、BF16 W2 route、poly5,并在 AWS 64-core 机器的 `0-31` 核运行。

## 性能

| 场景 | fdiv | NR1 | NR2 |
|---|---:|---:|---:|
| E8/topk6,8 expert × 4T | 18.8968ms | 19.0713ms(-0.92%) | 19.1583ms(-1.37%) |
| E64 多轮,8 expert × 4T | 20.8551ms | 20.9492ms(-0.45%) | 21.1104ms(-1.21%) |
| 单 expert,M=2048,1T | 87.2669ms | 88.0700ms(-0.91%) | 88.4186ms(-1.30%) |
| 单 expert,M=2048,4T | 22.3294ms | 22.5331ms(-0.90%) | 22.6227ms(-1.30%) |

E8 aggregate 为 `8.1823 / 8.1074 / 8.0706 TFLOP/s`。所有场景中 reciprocal
都稳定慢于 `fdiv`,不是多专家争用或线程数造成的结果。

## 精度(相对当前 fdiv)

E8/topk6 最终输出:

| 模式 | 不同元素 | 1 ULP 内 | rel-L2 |
|---|---:|---:|---:|
| NR1 | 1.0351% | 99.7152% | 0.03655% |
| NR2 | 0.03251% | 99.9913% | 0.00603% |

用 identity-W2 直接导出 SiLU BF16 intermediate,并覆盖 input/weight 多种尺度:

- NR1:约 `0.025%-0.030%` 元素变化,全部在 1 ULP 内,rel-L2
  `0.0078%-0.0096%`。
- NR2:约 `0.0001%-0.0010%` 元素变化,全部在 1 ULP 内,rel-L2
  `0.00048%-0.00194%`。

normal/scheduled/async、poly4/5/6、rows `1..23` 均通过回归。

## 结论

当前优化后的 M12 epilogue 已把两个独立 `fdiv` 相邻发射,硬件能够重叠其延迟。
NR1 将每次除法换成 4 条有依赖的浮点指令,NR2 换成 6 条,同时与 exp polynomial
争用乘法/FMA 流水线。因此 reciprocal 虽然精度足够,但没有性能价值。默认继续
使用 `fdiv`;reciprocal 仅保留为实验开关。下一项实验应通过更低阶的 minimax exp
polynomial 减少 FMA 数量,而不是继续优化除法。


# SVE hybrid tail 9-11: pad to M12 (2026-07-10)

SVE hybrid row planning now rounds a final `9..11` rows up to one M12 block.
The gather writes zero rows into the padded slots, and W13/W2 use the same M12
layout; scatter still processes only the real route count. Tails `1..8` retain
the existing M1/M2/M4/M8 dispatch.

This avoids the previous `M8 + M1/M2/M4` decomposition, which streamed the
complete B slice twice. The `rows=1..23` normal/scheduled/async regression is
bit-exact across the FP32/BF16 W2 routes and all poly4/5/6 M12 epilogues.

AWS `0-31`, single-thread streaming-distinct-expert measurements:

| routes | 9 | 10 | 11 | 12 | 21 | 22 | 23 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| us/expert | 545.9 | 545.8 | 546.4 | 548.3 | 1086.0 | 1086.9 | 1087.6 | 1088.5 |

The flat timing within each group confirms that `9..11` execute as one padded
M12 block and `21..23` as two M12 blocks.


---

# SVE M12 SiLU:更换 exp polynomial(2026-07-10)

## 候选

当前默认 poly5 是 range reduction 后对 `exp(r)` 使用五阶 Taylor polynomial,
其中 `r∈[-ln(2)/2,ln(2)/2]`。先测试已有 Taylor poly4(少 1 条 FMA),再增加三阶
near-minimax 候选(少 2 条 FMA):

```text
p(r) = p0 + r * (p1 + r * (p2 + r * p3))
p0 = 0.99992455695087079
p1 = 0.99998492863187383
p2 = 0.5050222842135581
p3 = 0.16767011875167742
```

离线 dense sweep 的 `exp(r)` 最大相对误差:

| polynomial | 最大相对误差 | 相对 poly5 的 FMA 数量 |
|---|---:|---:|
| Taylor poly5 | `3.24e-6` | 基线 |
| Taylor poly4 | `5.57e-5` | -1 |
| minimax3 | `9.97e-5` | -2 |

minimax3 通过 `FUSED_CPP_MOE_SILU_MINIMAX3=1` 实验性启用,仅覆盖 degree=5 的
M12 主块;尾块仍使用 poly5。默认关闭,range reduction 和精确 `fdiv` 均不变。

## 性能(AWS 0-31 核)

| 场景 | poly5 | Taylor poly4 | minimax3 | poly6 |
|---|---:|---:|---:|---:|
| E8/topk6,8 expert × 4T | 18.8829ms | 18.8294ms(+0.28%) | 18.8350ms(+0.25%) | 18.9122ms(-0.16%) |
| 单 expert,M=2048,1T | 86.6725ms | 86.3956ms(+0.32%) | 86.3334ms(+0.39%) | 86.7421ms(-0.08%) |
| E64 多轮,8 expert × 4T | 20.8974ms | 20.8251ms(+0.35%) | 20.7931ms(+0.50%) | 20.9299ms(-0.16%) |

E64 使用 100 次交错 run;E8 使用 80 次,T1 使用 60 次。少 1-2 条 FMA 的实际
e2e 上限只有约 `0.3%-0.5%`。

## 精度(相对 poly5)

E8/topk6 最终输出:

| polynomial | 不同元素 | 1 ULP 内 | rel-L2 |
|---|---:|---:|---:|
| Taylor poly4 | 2.3095% | 99.3837% | 0.05350% |
| minimax3 | 10.5798% | 97.0309% | 0.11885% |
| poly6 | 0.1323% | 99.9654% | 0.01293% |

identity-W2 直接观察 intermediate,覆盖 input/weight 多种尺度:

- Taylor poly4:约 `0.053%-0.068%` 元素变化,全部在 1 ULP 内,rel-L2
  `0.00064%-0.01945%`。
- minimax3:约 `0.43%-0.44%` 元素变化,全部在 1 ULP 内,rel-L2
  `0.00165%-0.03580%`。

normal/scheduled/async 和 rows `1..23` 回归通过。尚未做模型级精度评估。

## 结论

polynomial 降阶有正收益,但绝对幅度很小。minimax3 相比 Taylor poly4 在 T1/E64
只额外获得约 `0.07%-0.15%`,E8 中没有额外收益,同时最终输出误差约增至 2 倍。
因此默认继续使用 poly5。若后续愿意用约 `0.05%` rel-L2 换取约 `0.3%` 性能,
优先直接选择已有 Taylor poly4;minimax3 保留为实验模式,不建议直接设为默认。


---

# EP4 F=2048 parallel strategy (2026-07-10)

Shape: W13 `K=4096,N=4096` (32 MiB/expert), W2 `K=2048,N=4096`
(16 MiB/expert). Measurements use AWS cores `0-31`, 16 distinct expert
weights, eight consecutive experts per timed call, `skip_weighted=True`, and
30 interleaved runs per strategy. W13 split means two 16 MiB panels; W2 is not
split.

| strategy | active W13/W2 | M=1024 | M=1536 | M=2048 |
|---|---:|---:|---:|---:|
| 1x32, no split | 32/16 MiB | 46.230 ms / 8.919 T | 68.147 ms / 9.076 T | 90.718 ms / 9.090 T |
| 1x32, split | 16/16 MiB | 46.636 ms / 8.841 T | 69.388 ms / 8.913 T | 93.136 ms / 8.854 T |
| 2x16, no split | 64/32 MiB | 56.779 ms / 7.262 T | 79.867 ms / 7.744 T | 105.134 ms / 7.844 T |
| **2x16, split** | **32/32 MiB** | **46.058 ms / 8.952 T** | **67.243 ms / 9.198 T** | **89.492 ms / 9.215 T** |
| 4x8, no split | 128/64 MiB | 82.241 ms / 5.014 T | 118.532 ms / 5.218 T | 157.251 ms / 5.244 T |
| 4x8, split | 64/64 MiB | 58.460 ms / 7.053 T | 83.086 ms / 7.444 T | 108.042 ms / 7.633 T |

For two concurrent experts, splitting W13 improves the same `2x16` schedule by
`23.3%/18.8%/17.5%` at M=`1024/1536/2048`. Compared with the best serial
candidate (`1x32`, no split), however, `2x16` split is only `0.37%/1.35%/1.37%`
faster: both strategies already hold their active weight set near the measured
32 MiB optimum. `4x8` remains slower because even split W13 and unsplit W2 each
create a 64 MiB active set.

Planner implication: use `1x32` without W13 splitting when only one long expert
is ready. With at least two independent long experts, `2x16` plus two-panel W13
is the throughput choice; the gain is clear from roughly M=1536 onward and is
effectively a tie at M=1024. Do not split W13 for `1x32`, and do not use `4x8`
for uniform long routes on this 32-core partition.
