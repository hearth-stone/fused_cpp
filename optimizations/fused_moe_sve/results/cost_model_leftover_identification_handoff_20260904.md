# CPU MoE leftover 识别交接

日期：2026-09-04
用途：交给下一位 agent。本文是 leftover / stream-count 识别链的当前 source of
truth。更早的
[cost_model_absolute_pressure_next_steps_20260903.md](./cost_model_absolute_pressure_next_steps_20260903.md)
仍保留压力拟合与消融史，但其「第一项代码工作」已过时，不要当当前任务。

对用户用中文。代码、changelog、Arm 结果 md、commit message 用英文。
`MATHEMATICAL_MODEL.md` 正文用中文。

> 2026-09-04 后续更新：用户已明确要求继续 core PMU + L3C + DDRC 来源分账，
> 因此下文“没有用户新指令就停”的条件已被满足。新 probe 与两次 joint session
> 记录在
> [arm_codex_80c_stream_pressure_pmu_20260904.md](./arm_codex_80c_stream_pressure_pmu_20260904.md)。
> 结果把饱和形状定位到 victim-visible LLC miss 与 DDR read-command queue
> latency，而不是 aggregate DRAM bytes/bandwidth；五个 cell 尚不足以识别公式，
> 所以“不读 holdout、不替换 frozen v8、不接 VND/LNS”仍有效。
> 后续 count LOCO 已完成，见
> [arm_codex_80c_stream_pressure_count_loco_20260904.md](./arm_codex_80c_stream_pressure_count_loco_20260904.md)：
> queue/LLC 单变量均未通过冻结门槛；queue 只作为同一长生命周期 process 内 paired
> counter-reset 实验的下一候选，LLC miss ratio 只作 guardrail。
> 该 paired counter-reset 实验现已完成，当前 source of truth 是
> [arm_codex_80c_stream_pressure_paired_pmu_20260904.md](./arm_codex_80c_stream_pressure_paired_pmu_20260904.md)：
> queue 是明确胜出的物理特征，但线性 LOCO 仍未通过；不要加 knee/residual。
> fixed-count request-shape 分解也已完成，见
> [arm_codex_80c_stream_pressure_request_shape_20260904.md](./arm_codex_80c_stream_pressure_request_shape_20260904.md)：
> count 8 下 wider team 稳定提高 queue 与 victim slowdown，count 4 仍不可分辨；
> 下一假设是 distinct-B count 与 requester/team-width shape 的交互，不是单独 count。
> 最小 proxy 网格也已完成，当前最终结论见
> [arm_codex_80c_stream_pressure_proxy_grid_20260904.md](./arm_codex_80c_stream_pressure_proxy_grid_20260904.md)：
> 四个 proxy 在锁定 count-6 holdout 上全部失败，`stop_absolute_model_expansion=true`；
> 后续转向 partial order、top-K recall 和 false pruning，不再开新物理项。

## 1. 当前状态

同 LLC 上 1-route 1T victim 的 leftover 税已经识别完：

**跟并发 transfer-bound 的互异 packed-B 份数走，不跟核数、页大小、W13/W2
是否拼成一块走。不要从本链加结构。**

不要再估 $(\beta,\alpha_g)$，不要在 count 曲线上拟合 $T_\mathrm{sat}$，不要读
holdout，不要替换 frozen v8，不要接入 VND/LNS。没有用户新指令就停。

## 2. 仓库与工作树

- 仓库：`/Users/zhangxu/Codes/vllm-aarch64/fused_cpp`
- HEAD：`935d643`（`docs: record calibrated MoE width holdout`），相对
  `origin/main` ahead 15
- 相关工作几乎全是 **未提交** 的；Arm 用 `bash rsync.sh` 直同步，不要从
  远程 git 推断源码状态
- **不要 commit / push**，除非用户明确要求
- **不要回退、覆盖、重排无关脏文件**
- 本链 **没有改** `analytic_model.py`。该文件在本链开始前就已经脏，保持原样

Worktree 里至少有三摊互不相关的未提交改动。只动当前任务那一摊：

| 摊 | 代表路径 | 本链 |
| --- | --- | --- |
| leftover / pressure 识别 | 下文 benches、results、`MATHEMATICAL_MODEL.md` 9.50–9.54、`manifest.yaml` 的 `planner.dram_domain_injection_probe` | 是 |
| cost-model 旧脏改动 | `cpu_moe_schedule_optimization/cost_model/analytic_model.py` | **不要动** |
| neighborhood / pairwise | `executable_plan_neighborhood.py`、`pairwise_plan_ordering.py`、narrow-merge benches 与对应 tests | **不要动** |

## 3. 硬约束

生产外基线仍是 frozen v8：

```text
bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json
SHA256: 7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3
```

- **禁止读 holdout**。SHA 锁在
  `optimizations/fused_moe_sve/benchmarks/fit_phase_reaccount_calibration.py`
  的 `LOCKED_HOLDOUT_SHA256`
- **禁止**替换 frozen v8、接入 VND/LNS、升 production schema、加经验 residual
- **禁止**再写入 contrast-only 参数 `capacity_scale=0.787`、
  `effective_gather_traffic_multiplier=22.26`
- **禁止**把 session-1 联合拟合点 `capacity_scale=0.78`、
  `effective_gather_traffic_multiplier=0.25` 当 freeze
- `effective_traffic_multiplier` 只能叫 effective coupling，不能当实测 byte
  ratio
- 改 cost-model 前读并更新 `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`
- Lab 实验登记 `optimizations/fused_moe_sve/manifest.yaml` 的
  `planner.dram_domain_injection_probe`
- phase re-accounting：**结构可留，完整 candidate 已拒绝**
- per-LLC-domain DRAM injection：**default-off**
- 本链不加结构，不改 packed ABI / kernel / Plan V2 / 默认 dispatch

打分路径：`_phase_reaccounted` + `_mode_bridges` + `_predict_spans`，
`thread_cpu_ids=tuple(range(240, 320))`。

Change class：本链是 **E Lab**。聚焦 pytest + 点名的 Arm 命令。不要声称没跑过
的测试过了。

## 4. 远程机器

- Host：`Arm-codex-internal`（备用 `Arm-codex`）
- 远程根：`/home/zhangxu/codex/fused_cpp`
- Python：`/home/zhangxu/codex/fused_cpp/.venv/bin/python`
- 同步：`bash rsync.sh`（排除 `tmp/`）
- 绑定：`taskset -c 240-319 numactl --membind=3`
- 远程写 `/tmp/...`，再 `scp` 回本地 `tmp/`（gitignore，不提交 json dump）
- HugeTLB：`HugePages_Total: 0`。**不要 sudo reserve**
- THP sysfs：`enabled=[always]`，`defrag=[madvise]`
- 最近几次扩展 SHA：
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`
  （以当次 artifact 为准）

## 5. Victim 与协议

除非结果 md 另写：

- victim：expert 0，$M=1$，1T，logical 64 / CPU304 / LLC7
- 「本地」= **same-LLC7**，不是 NUMA 本地
- LLC7 核：`280-319`；logical 48 = CPU288；logical 64 = CPU304
- 16 核 team：logical `48-63`（CPU288-303），不含 victim
- cross 对照：logical `0-15`（LLC6，CPU240-255）
- 协议：5 warmup，31 randomized paired，4 measured copies，每 sample 前一份
  disjoint isolated scrub
- 形状：hidden 4096，intermediate 512，单 expert packed W13 $=4HF=8\,\mathrm{MiB}$
- 1-route 1T isolated W13 是 **transfer-bound**；M=68 1T 是 GEMM-bound

## 6. 识别链（全部拒绝加结构）

数学模型 9.50–9.54，changelog v1.59–v1.63。

| 阶段 | 结论 | 关键数 | 结果 |
| --- | --- | --- | --- |
| phase re-account | 分阶段结构可留；完整 schema-v11 拒绝 | 不替换 v8 | [arm_codex_80c_phase_reaccount_20260903.md](./arm_codex_80c_phase_reaccount_20260903.md) |
| joint fit | $(\beta,\alpha_g)$ 不可辨识 | 0.78 / 0.25 贴下界 | [arm_codex_80c_absolute_pressure_joint_fit_20260904.md](./arm_codex_80c_absolute_pressure_joint_fit_20260904.md) |
| rank DRAM | ≈一半远程共模 | $+0.893\to+0.454\,\mathrm{ms}$ | [arm_codex_80c_rank_dram_domain_scope_20260904.md](./arm_codex_80c_rank_dram_domain_scope_20260904.md) |
| rank LLC | 剩余远程 100% | $+0.454\to 0$ | [arm_codex_80c_rank_llc_domain_scope_20260904.md](./arm_codex_80c_rank_llc_domain_scope_20260904.md) |
| victim-asymmetric | 1T W13 是 transfer-bound；own-demand 把硬件本地税一起清掉 | 无臂过三门槛 | [arm_codex_80c_victim_asymmetric_dilation_20260904.md](./arm_codex_80c_victim_asymmetric_dilation_20260904.md) |
| occupancy $M$ | leftover **随 $M$ 下降**；利用率假 | n4 $M=1/4/16/68$ $=+0.660/+0.639/+0.402/+0.290$ | [arm_codex_80c_aggressor_m_occupancy_20260904.md](./arm_codex_80c_aggressor_m_occupancy_20260904.md) |
| fill-port | 1×16T ≈ 一条 1T，不是 16 个填数口 | $+0.018/+0.014/+0.160$；签名 `inconclusive`（gap $0.146<0.20$） | [arm_codex_80c_fill_port_vs_stream_20260904.md](./arm_codex_80c_fill_port_vs_stream_20260904.md) |
| stream-count | leftover 跟互异 B 份数走，不跟线程数 | 见下表；签名 `stream_count` | [arm_codex_80c_stream_count_composition_20260904.md](./arm_codex_80c_stream_count_composition_20260904.md) |
| unified block | 一层 W13+W2 拼成一块不降 leftover | $+0.153/+0.157$，`layout_neutral` | [arm_codex_80c_unified_weight_block_20260904.md](./arm_codex_80c_unified_weight_block_20260904.md) |
| THP | 验证过的 2 MiB vs 4 KiB 不降 leftover | $+0.167/+0.162$，smaps $0$ vs $100\%$，`page_neutral` | [arm_codex_80c_weight_thp_20260904.md](./arm_codex_80c_weight_thp_20260904.md) |

stream-count 阶梯（16 线程，cores `48-63`，seed `20260912`）：

| 构图 | 互异 B 份数 | 线程 | leftover |
| --- | ---: | ---: | ---: |
| 1×16T | 1 | 16 | $+0.003$ |
| 2×8T | 2 | 16 | $+0.035$ |
| 4×4T | 4 | 16 | $+0.075$ |
| 4×1T | 4 | 4 | $+0.070$ |
| 16×1T | 16 | 16 | $+0.155$ |
| `8+8+4+1` | 4 | 21 | $+0.110$ |
| 四起点 1T | 4 | 4 | $+0.103$ |
| 21×1T | 21 | 21 | $+0.301$ |

`8+8+4+1` 的远程 $+0.042$；21×1T 远程 $+0.211$。

联合拟合失败已经拆开：rank DRAM 是约一半远程共模，rank LLC 是另一半。两者去掉
后远程预测为 0。剩余过预测来自短 transfer-bound victim 继承 68-route cohort
字节 dilation。硬件 leftover 不是 68-route 字节利用率。

## 7. 「流」不要再讲错

用户已经明确：宽队从 packed-B 的不同 N 条带启动，表面上就是多条访存流。旧比喻
「16 核啃同一份 B」不够。正确说法：

1. $M=1$ 的宽队走 **N-split**（`choose_moe_gemm_split` / `StageWindowPlan`）。
   packed-B 按 N tile 外层连续。16 核是 16 条 **不相交** 条带，每核约
   $512\,\mathrm{KiB}$，cache line 不共享。核上预取流确实是 16 条。
2. 实验里的「流」**不是** CPU 预取器流，也不是「有几个核在 load」。它是：
   **victim 窗口里有几份互不相同的 packed-B 正在往共享 LLC/DRAM 里填。**
3. 1×16T：16 条带加起来仍是 **一份** $8\,\mathrm{MiB}$ W13。leftover
   $+0.014$，贴近 1×1T $+0.018$，不是 16×1T $+0.160$。
4. 16×1T：16 份 $8\,\mathrm{MiB}$ $\approx 128\,\mathrm{MiB}$ 互异 B。
5. 4×4T（16 核、4 份 B）$\approx$ 4×1T（4 核、4 份 B）。核数差 4 倍，leftover
   不变。
6. THP 与 unified-block 都不改 leftover：它们不改变「现在有几份互异 B」。

规划含义：**数并发 transfer-bound expert，不要 `sum(team_width)`。** 这不是
公式改动，也不要为此写入 production schema。

还没有 PMU/SPE 拆「16 核填 8 MiB 的瞬时带宽」和「128 MiB 的 LLC 驱逐」。现有
阶梯已经把核数和互异 B 份数拆开。不要为了比喻加结构。

## 8. 明确不要做

- 不要再估 $(\beta,\alpha_g)$
- 不要在 n=1/2/4/8/15 曲线上拟合 $T_\mathrm{sat}$ 或 $n_\mathrm{knee}$
- 不要从 occupancy / fill-port / stream-count / unified / THP 加 default-off
  结构
- 不要读 holdout，不要经验 residual
- 不要为专家间 gap 改 `packed_stride` ABI
- 不要把 HugeTLB 写成已预留
- 不要把 `analytic_model.py` 树上的旧脏改动卷进本链
- 不要执行
  [cost_model_absolute_pressure_next_steps_20260903.md](./cost_model_absolute_pressure_next_steps_20260903.md)
  里「第一项代码工作：`--aggressor-counts 0,1,2,4,8,15`」——已经做完

`manifest.yaml` 当前 `next_decision`：THP session-1 `page_neutral`；不加结构；
不读 holdout；不换 frozen v8。

## 9. 仅当用户明确要求才做

本链科学上已闭合。下列都不是默认下一任务：

1. PMU / SPE：16 核填 $8\,\mathrm{MiB}$ vs 16 份 $8\,\mathrm{MiB}$ 的 LLC miss /
   DRAM 字节（回应用户「预取流」质疑）。
2. leftover 公式候选：只数并发 transfer-bound expert。必须独立 probe + 预声明
   门槛，先不要写进 production schema。
3. occupancy session-2 仅确认（$M=68$ 税更轻）。
4. 另一条线：TODO Step-3 width neighborhood / pairwise ordering。与 leftover
   识别独立，worktree 里也有未提交文件。

## 10. 关键命令与 artifact

统一协议绑定：

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python <bench> \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json
```

| Probe | 脚本 | seed | 本地 artifact SHA256 |
| --- | --- | ---: | --- |
| fill-port | `benchmarks/bench_fill_port_vs_stream.py` | 20260911 | `89b6de1517cb1d0e6e5f1920a0fb28659d86dbc38039da0bf50d51c3dde31676` |
| stream-count | `benchmarks/bench_stream_count_composition.py` | 20260912 | `175d50cb2aa220be3047ea1904de92927e785a9c7233e6d2af534b03200d6490` |
| unified-block | `benchmarks/bench_unified_weight_block.py` | 20260913 | `7bd2319f49178bdc46f62abb6d7de016af2ff1ba5fe88bdd927a5be47b3ebc32` |
| THP | `benchmarks/bench_weight_thp.py` | 20260914 | `da6d1883e28483619bd844b3e56017fcd3350a0abb51561e2b0a75ea4ed9c2c4` |

json 在本地 `tmp/`，不进 git。md 里已记 SHA。

最近一次 THP 全命令：

```bash
taskset -c 240-319 numactl --membind=3 \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_weight_thp.py \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --seed 20260914 \
  --trace-dir /tmp/moe_weight_thp \
  --output /tmp/moe_weight_thp_fit_20260904.json
```

## 11. 本链文件索引

Benches：

- `optimizations/fused_moe_sve/benchmarks/bench_gather_injection_overlap.py`
- `optimizations/fused_moe_sve/benchmarks/bench_aggressor_m_occupancy.py`
- `optimizations/fused_moe_sve/benchmarks/bench_fill_port_vs_stream.py`
- `optimizations/fused_moe_sve/benchmarks/bench_stream_count_composition.py`
- `optimizations/fused_moe_sve/benchmarks/bench_unified_weight_block.py`
- `optimizations/fused_moe_sve/benchmarks/bench_weight_thp.py`
- `optimizations/fused_moe_sve/benchmarks/ablate_rank_dram_domain_scope.py`
- `optimizations/fused_moe_sve/benchmarks/ablate_rank_llc_domain_scope.py`
- `optimizations/fused_moe_sve/benchmarks/ablate_victim_asymmetric_dilation.py`

文档：

- `cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md` 9.50–9.54，changelog
  v1.59–v1.63
- `cpu_moe_schedule_optimization/TODO.md` leftover 项均已勾完
- `optimizations/fused_moe_sve/manifest.yaml` →
  `planner.dram_domain_injection_probe`

聚焦测试：`tests/test_moe_{fill_port_vs_stream,stream_count_composition,unified_weight_block,weight_thp,aggressor_m_occupancy,...}.py`

## 12. 给下一位 agent 的第一句话

Leftover 识别已闭合。没有用户新指令就不要开新 probe，也不要改
`analytic_model.py`。若用户要继续，先问清是 PMU 证据、公式候选，还是另一条
neighborhood 线；三条不要混在一次改动里。
