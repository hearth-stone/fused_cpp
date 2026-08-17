# SDPA 待办优化清单

> ⚠️ **维护规则**：与 `SDPA_VERSIONS.md` 配套。**做完一项**就把对应章节移到下面的「已完成」段落；**新发现**的优化点追加到 Tier 表中。每次会话开始前先扫这份文档。
>
> 路径：`fused_cpp/csrc/SDPA_TODO.md`
> 配套真理来源：`fused_cpp/csrc/SDPA_VERSIONS.md`

---

## 当前 SDPA 性能水位（baseline 用于评估增益）

按 R1-like shape（B=1, N=4, L=S=1024, E=192, Ev=128）单线程实测：

| 指标 | 值 | 占当前指令路径峰值 |
|---|---|---|
| **bf16 SDPA wall-clock GFLOPS** | **52.5–54.6**（l3kv / l3kv_packv） | **95–99%**（理论 55 → 限于 BFMMLA half-rate） |
| **fp32 SDPA wall-clock GFLOPS（baseline）** | 13.9–15.2 | 13–14% fmla peak（被 QKᵀ-fp32 单累加器卡住） |
| **fp32 SDPA wall-clock GFLOPS（qk_ublock4）** | **80.2–84.2**（l3kv / l3kv_packv） | **74–77% fmla peak**（**P0-fp32 已落地，5.3–5.5× 提升**） |
| 微内核：fp32 PV 8×8 (pquad) | 95–106 | 87–97% fmla peak (109) |
| 微内核：fp32 QKᵀ 8×8 (baseline) | 7.6–10.1 | 7–9% fmla peak ⚠️（dot product 单累加器，已留作参考） |
| 微内核：fp32 QKᵀ 8×8 (qk_ublock4) | **104.7–106.5** | **96–98% fmla peak**（设计目标已达成） |
| 微内核：bf16 PV 8×8 | 67–69 | ~63% fmla peak（受 V widen 限制） |
| 微内核：bf16 QKᵀ 8×8 (BFMMLA) | 47–49 | **86–89% BFMMLA half-rate**（指令本身受限，待 P0 BFMLALB/T 切换） |

> 数据来源：本机（Apple Silicon / aarch64）单线程，OMP_NUM_THREADS=1，`tools/bench_sdpa_versions.py` + `_C.benchmark_microkernel`，2026-05-23 测得。fp32 fmla peak 109 GFLOPS、BFMMLA half-rate peak 55 GFLOPS 来自 `CLAUDE.local.md`。

---

## 优化优先级总览

| 优先级 | 优化项 | 预期增益 | 工程量 | 风险 |
|---|---|---|---|---|
| 🔴 P0 | [QKᵀ-bf16 切 BFMLALB/T](#p0-qk%E1%B5%80-bf16-%E5%88%87%E5%88%B0-bfmlalbt-%E4%B8%BB%E8%B7%AF%E5%BE%84) | **bf16 SDPA ~1.5–1.8×↑（端到端 +50–80%）** | 小（半天） | 低（已有 bfmlalb/t 退化路径，仅切换主分支宏） |
| 🟢 P2 | [bf16 PV 也走 P-quad load + lane FMA](#p2-bf16-pv-%E4%B9%9F%E8%B5%B0-p-quad-load--lane-fma) | bf16 +3–5% | 小（1 天） | 低 |
| 🟢 P2 | [Lc_l2 / Sc_l2 tile size 扫参](#p2-lc_l2--sc_l2-tile-size-%E6%89%AB%E5%8F%82) | 0–3% | 中（脚本 + 重编多次） | 无 |
| 🟢 P2 | [vexpq_f32 多项式精度/速度调参](#p2-vexpq_f32-%E5%A4%9A%E9%A1%B9%E5%BC%8F%E7%B2%BE%E5%BA%A6%E9%80%9F%E5%BA%A6%E8%B0%83%E5%8F%82) | softmax 内 +5–10%（总 +0.5%） | 小 | 数值精度小幅变化 |
| ⚪ P3 | [Online softmax 融合到 PV](#p3-online-softmax-%E8%9E%8D%E5%90%88%E5%88%B0-pv) | +2–5% | 大（重写 microkernel） | 高 |
| ⚪ P3 | [K 预 pack（类比 V packv）](#p3-k-%E9%A2%84-pack-%E7%B1%BB%E6%AF%94-v-packv) | bf16 +2–4% / fp32 不亏 | 中 | 内存占用 +50% |
| ⚪ P3 | [INT8 量化路径](#p3-int8-%E9%87%8F%E5%8C%96%E8%B7%AF%E5%BE%84) | 4× 吞吐（435 GOPS / 109） | 极大（新路径） | 算法精度 |
| ⚪ P3 | [QKᵀ inline asm 调度](#p3-qk%E1%B5%80-inline-asm-%E8%B0%83%E5%BA%A6) | +1–2% | 大 | 维护成本高 |

> **建议执行顺序**：P0 → P1 全部 → 测试看效率到哪 → P2 选做 → P3 视需要

---

## 🔴 P0 项

### P0: QKᵀ-bf16 切到 BFMLALB/T 主路径

**问题诊断**：
- CLAUDE.local.md 明确：本机 BFMMLA = 55 GFLOPS（half-issue），BFMLALB/T = 109 GFLOPS（full-issue），**差 2×**
- 当前 `gemm_qkt_microkernel_8x8_bf16`（行 45–340）默认走 BFMMLA 主路径（`#if FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA`），仅在退化场景走 BFMLALB/T
- bf16 SDPA wall-clock 51 GFLOPS、效率 74% 也部分受 BFMMLA 限制（理论峰值只有 69 GFLOPS）

**目标**：把 BFMLALB/T 升格为主路径；BFMMLA 改为可选退化（或彻底移除）

**实现路径**：
- 检查 `gemm_qkt_microkernel_8x8_bf16` 内 `FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLALBT` 退化分支是否完整
- 默认 `FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA = 0`、`FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMLALBT = 1`
- 跑 `test_sdpa_versions_equiv` 确认数值通过（bf16 容忍度 `atol=5e-2, rtol=5e-2`）

**预期增益**：
- bf16 SDPA 理论峰值：69 → 109 GFLOPS（QKᵀ 路径峰值翻倍）
- bf16 SDPA wall-clock：51 → ~75–85 GFLOPS（按 ~70% peak 利用率估）

**数值等价性**：BFMLALB/T 累加顺序与 BFMMLA 不同，bf16 ULP 级误差，已被等价测试容忍度覆盖。

**工程量**：半天（如果退化路径已完整）；1 天（如果还要补完 BFMLALB/T 路径）

**风险**：低。指令本身已经在用了。

**先决条件**：无。

---

## 🟢 P2 项

### P2: bf16 PV 也走 P-quad load + lane FMA

**问题**：`MK_PQuad` 只重写了 fp32 PV，bf16 PV (`gemm_pv_microkernel_8x8_bf16` 行 435–505) 仍是「8 P 标量 load + vfmaq_n_f32」。bf16 PV 同样有 LSU 端的优化空间。

**目标**：复制 fp32 pquad 思路到 bf16 PV：每 4-k 段开头 8 条 `vld1q_f32(P_hat + ...)` 拿 P 行 quad（P 始终是 fp32），段内用 `vfmaq_laneq_f32`。

注意点：
- bf16 PV 的 `widen_bf16x4_to_fp32(V)` 是个 helper，per-k 调用 2 次。如果 lane-form FMA 内层一次处理 4 个 k，需要把 widen 也展开成 4 套
- 寄存器预算：16 O acc + 8 P quad + 4 V_lo + 4 V_hi + 4 widen 临时 = ~36，会溢出
- 可能需要 2-way unroll 而不是 4-way（妥协 ILP 换 reg 余量）

**预期增益**：bf16 SDPA +3–5%

**工程量**：1 天（含微内核实现 + 寄存器调试 + 单测）

**风险**：寄存器压力大，可能 spill。先做 ILP=2 版本验证可行性，再尝试 ILP=4。

**先决条件**：无。

---

### P2: Lc_l2 / Sc_l2 tile size 扫参

**问题**：`compute_tile_sizes_l3kv` 给的 tile size 是按 cache 容量启发式估算，**未必是 wall-clock 最优**。

**目标**：写一个扫参脚本，对几组 R1-like / GPT-like / Llama-like shape 跑：
- Lc_l2 ∈ {16, 24, 32, 48, 64}
- Sc_l2 ∈ {32, 48, 64, 96, 128}
- 25 组组合 × 5 个 shape = 125 次运行

输出每个组合的 wall-clock，找最优组合。如果与默认值差距 ≥ 5%，固化新规则到 `compute_tile_sizes_l3kv`。

**预期增益**：0–3%（默认值已经比较优了，容易颗粒无收）

**工程量**：中（脚本 + 重编多次 + 数据分析）

**风险**：无。

**先决条件**：完成 P0 + P1 所有项后再做（否则数据会被未优化的瓶颈干扰）。

---

### P2: vexpq_f32 多项式精度/速度调参

**问题**：`csdv` 行 79–109 的 `vexpq_f32` 用 5 阶多项式 + range reduction。当前最大误差大约 1–2 ULP，但每次 ~30 cycle。如果接受 5–10 ULP 误差，可以用 4 阶多项式或 reciprocal table 实现，~20 cycle。

**目标**：测两种 approximation 的 ULP 误差和 cycle，看是否值得。

**预期增益**：softmax 内 5–10%；总时间 +0.5–1%

**工程量**：小

**风险**：数值精度变化，需要扩大 SDPA 等价性容忍度。

**先决条件**：无，可以做也可以不做。

---

## ⚪ P3 项

### P3: Online softmax 融合到 PV

**问题**：当前每个 inner 8 行组的流程：scores 写到 scratch → row_max 扫描 → exp 写到 P_hat scratch → PV 调用读 P_hat。每个中间产物都过一遍 scratch buffer。

**激进做法**：把 softmax 的 P̂ 写出 + PV 的 P̂ 读取**融在同一寄存器周期**——softmax 计算完每个 P̂ 行立刻 broadcast 进 PV 的 lane FMA，不落 scratch。

**预期增益**：+2–5%（消掉 scratch 读写带宽）

**工程量**：大（重写 microkernel 边界）

**风险**：高。会破坏 microkernel 的封装性、影响 baseline / scalar / pquad 的复用。

**建议**：等 P0 + P1 + P2 都做完、效率仍 < 80% 时再考虑。

---

### P3: K 预 pack（类比 V packv）

**问题**：QKᵀ 路径里 K 的访问类似 V——跨 Sk 行 stride = E。理论上可以做 K-packv：把 K 重排成 `[B, N, E/4, S, 4]` 之类。

**判断**：
- bf16 路径：K 是 BFMMLA 的 B 矩阵，BFMMLA 内已经有内部 pack（vcombine_u16 重组），重 pack 收益不明显
- fp32 路径：K 在新 4×4 双向分块 microkernel（P0）下也是按行连续访问，**不需要 pack**

**结论**：除非 P0-fp32 重写后发现 K 仍然有 stride 瓶颈，否则**不做**。

**工程量**：中

**风险**：DRAM 占用翻倍，单线程负收益（与 V packv 对照）。

---

### P3: INT8 量化路径

**问题**：本机 dp4a 峰值 435 GOPS、mmla 峰值 367 GOPS，是 fp32/bf16 的 4× 吞吐。但需要：
- W4A8 / W8A8 量化训练或 PTQ
- 全新的 INT8 SDPA microkernel（dp4a 主路径）
- KV cache 量化策略

**预期增益**：4× 吞吐（如果算法精度可接受）

**工程量**：极大（新路径，估算 1–2 周）

**风险**：算法精度损失 + 校准复杂

**优先级**：模型层决定。

---

### P3: QKᵀ inline asm 调度

**问题**：编译器对 BFMMLA / 重 SIMD 的指令调度不一定最优，可能有偶发的寄存器 spill 或 false dep。

**目标**：手写 inline asm 把 QKᵀ 8×8 主路径写死最优指令序列。

**预期增益**：+1–2%

**工程量**：大

**风险**：维护成本巨大。每次升级 LLVM / ARM ABI 都要重测。

**建议**：除非发现编译器实在生成不出最优代码（性能咨询级别），否则**不做**。

---

## 已完成（参 SDPA_VERSIONS.md 的 Changelog）

- ✅ **Sparse MLA token-panel L2 cache schedule rejected** —— 2026-08-17
  - panel 4 kept per-token online-softmax order and reused each packed B chunk
    across four adjacent tokens, but improved Amazon 8C by only 1.5%/0.9% at
    1/8 threads; panel 8 regressed eight-thread latency by 1.1%
  - the existing token-major fallback was 79--82% slower; retain the current
    head-major token-first schedule and use KV shards only for query underfill
  - M5 cores 96--191: the full panel `{4,8,16}` x B-block `{64,128,256}`
    sweep also failed; paired panel8/B256 changed 2048/8192/later-sparse by
    -0.14%/+0.30%/+0.19%. PMU showed fewer 2048 L2 refills without wall-time gain
  - after score/max and packed-P fusion, fixed Sc tiles 64/128/256/512 were
    rescanned; auto remained fastest, with fixed 512 slower on 2048 and 8192
- ✅ **P0: QKᵀ-fp32 重写为 4×4 双向分块（`MK_QkUblock4` trait）** —— 2026-05-23
  - 新增 `gemm_qkt_microkernel_8x8_fp32_ublock4`：8×8 输出切 4 个 4×4 子块，每块 16 个独立 fp32 累加器外积扇出 + vpaddq 树 reduce，把 ILP 从 1 条链拉到 16 条独立 fma 链
  - 不需要 K 转置或 pre-pack；Q/K 仍按行连续 vld1q_f32 加载
  - 自动注册 `flash2_neon_cache_qk_ublock4` / `flash2_neon_l3kv_qk_ublock4` / `flash2_neon_l3kv_packv_qk_ublock4`
  - 数值上与 baseline **接近等价但不按位相同**（fp32 ULP-级误差，已被 SDPA atol=rtol=1e-4 等价测试覆盖）
  - **本机实测**（Apple Silicon 单线程，2026-05-23）：QKᵀ-fp32 8×8 单核 7.6–10.1 → **104.7–106.5 GFLOPS（10.4–14.0×，96–98% fmla peak）**；R1-like fp32 SDPA wall-clock 13.9–15.2 → **80.2–84.2 GFLOPS（5.3–5.5×，74–77% fmla peak）**；bf16 路径完全不动（1.00× 实测）
- ✅ `MK_PQuad` trait（fp32 PV 8×8 P-quad load + lane FMA）—— 2026-05-20
- ✅ `flash2_neon_l3kv_packv` SDPA 变体（V 预 pack）—— 2026-05-22
- ✅ PV outer k_off loop 合并（消掉 7×16=112 vld_O round-trip）—— 2026-05-22
- ✅ **P1 优化批次** —— 2026-05-22
  - softmax row-max 向量化（`vmaxq_f32` + `vmaxvq_f32`）
  - `is_causal` / `mask_ptr` 模板特化（`if constexpr` 编译期消去热路径分支；模板实例化 12→48）
  - mask add 内层循环向量化（`vld + vadd + vst`）
  - packv 路径下 ev_off 切换时对下一个 ev_block 起始 2 cache line 发 PLDL1KEEP
  - bf16 单线程 51 → 54–57 GFLOPS（74% → 78–83% BFMMLA-half 峰值）

---

## 怎么挑下一个做

**如果你想要最大单项提升**：选 🔴 P0：
- 关心 bf16：QKᵀ-bf16 切 BFMLALB/T（端到端 ~1.5–1.8× 提升）
- 关心 fp32：P0-fp32 重写（`MK_QkUblock4`）已在 2026-05-23 落地，**待目标机重测**确认实际增益；如目标机数据未达 ~80 GFLOPS wall-clock，回头审视 ublock4 实现的 LSU / 寄存器调度细节

**如果想要稳定渐进收益**：进 🟢 P2 微调：
- bf16 PV pquad（+3–5% bf16，先做 ILP=2 验证可行性）
- Lc_l2 / Sc_l2 tile size 扫参（先做 P0 让瓶颈移走再扫，否则数据被现有瓶颈干扰）
- vexpq_f32 多项式调参（+0.5–1%）

**当前态势**：bf16 efficiency 已达 78–83% BFMMLA-half 峰值，离指令上限只剩 ~20%。从 ROI 看下一步**优先 🔴 P0-bf16（QKᵀ-bf16 切 BFMLALB/T）**——把 bf16 指令峰值从 69 提升到 109 GFLOPS，端到端可期 75–85 GFLOPS。fp32 路径已通过 `MK_QkUblock4` 打开 QKᵀ 瓶颈，等目标机重测确认 wall-clock。
