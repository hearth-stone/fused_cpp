# CPU MoE cost model：absolute memory pressure 下一步执行单

日期：2026-09-03
用途：压力拟合与消融史。**leftover 识别已闭合**，当前交接以
[cost_model_leftover_identification_handoff_20260904.md](./cost_model_leftover_identification_handoff_20260904.md)
为准。下文「第一项代码工作」已过时，不要再执行 `--aggressor-counts` 扩展。

## 结论与当前边界

当前 phase re-accounting 与 absolute-pressure 联合拟合都已完成，结论是：

- 保留 gather、W13、W2 分阶段校准的分析结构。
- 保留按 LLC domain 表达 DRAM injection capacity 的实验结构，但继续保持 default-off。
- 拒绝完整 schema-v11 candidate，不得替换 frozen v8，也不得接入 VND/LNS。
- 拒绝 contrast-only 参数 `capacity_scale=0.787`、
  `effective_gather_traffic_multiplier=22.26`。
- 拒绝 session-1 联合拟合点 `capacity_scale=0.78`、
  `effective_gather_traffic_multiplier=0.25`：贴 $\alpha_g$ 下界、不可辨识，
  且 session-2 contrast residual 系统为正。
- 下一步不是再估 $(\beta,\alpha_g)$，也不是把 occupancy 税写成 count-cap
  或 own-demand。aggressor $M$ session-1 已经表明 same-LLC n4 税随 $M$
  **下降**（`+0.660` → `+0.290 ms`）。利用率被证伪。不要从该 sweep 加结构。
  当前唯一硬件问题是：同一 16 个 LLC7 核上，1×16T $M=1$ 对 1T victim 的税
  更接近 1 条流还是 16 个填数口。Arm session-1 已回答方向：贴近 1 条流
  （$+0.014$ vs 1T $+0.018$，16×1T $+0.160$），预声明门槛仍为 inconclusive。
  Rank DRAM 解释远程 n15 共模的约 49%（`+0.893` → `+0.454 ms`）；
  rank LLC 解释剩余 100%（`+0.454` → `0 ms`）。domain LLC 是剩余本地项，
  但随 count 增长到 n15 `+0.590 ms`，不能在 n≈4 饱和。去掉全部共享
  cache/DRAM 后模型 victim 近似 isolated（`+0.001 ms`），硬件 same-LLC
  n15 仍为 `+0.312 ms`。victim-asymmetric dilation 消融进一步证明：
  1-route 1T W13 自身是 transfer-bound，`compute_bound_skip` 与对称臂相同；
  `same_llc_peers` 只修远程；`own_demand` 把 n15 打到 `+0.001/0 ms`，
  也把硬件本地税一起清掉。没有任何臂同时满足三门槛。

联合拟合失败的核心原因已经拆开：rank DRAM 是约一半远程共模，rank LLC 是
另一半。两者去掉后，远程预测为 0。victim-asymmetric 消融证明剩余过预测
来自「短 transfer-bound victim 继承 68-route cohort 字节 dilation」。
own-demand 去掉该继承后模型 ≈ isolated。aggressor $M$ 消融进一步表明：
硬件 leftover 在 $M=1$ 的 n4 是 `+0.660 ms`，在 $M=68$ 是 `+0.290 ms`，
随 peer 变肥而下降。现有字节利用率公式识别不了这种 transfer-bound 同 LLC
争用。

完整证据见
[arm_codex_80c_phase_reaccount_20260903.md](./arm_codex_80c_phase_reaccount_20260903.md)、
[arm_codex_80c_absolute_pressure_20260904.md](./arm_codex_80c_absolute_pressure_20260904.md)、
[arm_codex_80c_absolute_pressure_joint_fit_20260904.md](./arm_codex_80c_absolute_pressure_joint_fit_20260904.md)、
[arm_codex_80c_rank_dram_domain_scope_20260904.md](./arm_codex_80c_rank_dram_domain_scope_20260904.md)
[arm_codex_80c_rank_llc_domain_scope_20260904.md](./arm_codex_80c_rank_llc_domain_scope_20260904.md)、
[arm_codex_80c_victim_asymmetric_dilation_20260904.md](./arm_codex_80c_victim_asymmetric_dilation_20260904.md)
和
[arm_codex_80c_aggressor_m_occupancy_20260904.md](./arm_codex_80c_aggressor_m_occupancy_20260904.md)、
[arm_codex_80c_fill_port_vs_stream_20260904.md](./arm_codex_80c_fill_port_vs_stream_20260904.md)、
[arm_codex_80c_stream_count_composition_20260904.md](./arm_codex_80c_stream_count_composition_20260904.md)、
[arm_codex_80c_unified_weight_block_20260904.md](./arm_codex_80c_unified_weight_block_20260904.md)、
[arm_codex_80c_weight_thp_20260904.md](./arm_codex_80c_weight_thp_20260904.md)
和
[cost_model_leftover_identification_handoff_20260904.md](./cost_model_leftover_identification_handoff_20260904.md)。

## 下一阶段的唯一目标

Aggressor $M$ session-1 已经拒绝「peer 越肥、victim 越慢」的利用率解释，也拒绝
「只跟人数字有关、与 $M$ 无关」的 occupancy。不要重跑 $(\beta,\alpha_g)$，
不要在 count 曲线上拟合 $T_\mathrm{sat}$。

当前不再从本 $M$ sweep 加结构。可选 session-2 只做确认。仍然不能读 holdout，
不能加经验 residual。fill-port vs stream session-1 已跑完：1×16T 贴近单条
1T（$+0.014$ vs $+0.018\,\mathrm{ms}$），不是 16 个填数口（16×1T $+0.160$）。
预声明仍为 inconclusive。stream-count composition session-1 已接受「数流」：
`8+8+4+1` 为 $+0.110$，贴近四起点 1T $+0.103$，不是 21×1T $+0.301$。unified
weight-block session-1 签名 `layout_neutral`：16×1T leftover split/unified
为 $+0.153/+0.157\,\mathrm{ms}$，差 $0.004$。把一层 W13+W2 放进同一连续
allocation 不能把 16 条并发专家流变成 1 条流，也不降低 leftover。weight THP
session-1 签名 `page_neutral`：smaps 验证 4 KiB AnonHugePages $=0$、THP
$100\%$，16×1T leftover small/THP $+0.167/+0.162\,\mathrm{ms}$。2 MiB 页不降低
leftover。不要从这些 probe 加结构。

## 第一项代码工作：扩展 absolute-pressure probe

基于
[bench_gather_injection_overlap.py](../benchmarks/bench_gather_injection_overlap.py)
扩展，不要先改 production planner。至少增加：

```text
--aggressor-counts 0,1,2,4,8,15
```

实现要求：

1. 保持 target 固定为 expert 0、`M=1`、1T、logical core 64 / physical
   CPU304 / LLC7。
2. 每个 aggressor 固定为 1T、`M=68`；shape 固定为 BF16 fused-SiLU、
   `H=4096`、`F=512`、backend N tile 16，W13/W2 使用 full owner-stripe
   windows。
3. same-LLC aggressor 使用 logical cores `65-79`，对应 physical CPUs
   `305-319` / LLC7。
4. cross-LLC aggressor 使用 logical cores `0-14`，对应 physical CPUs
   `240-254` / LLC6。
5. 所有 mode 必须保留同一组 tasks、routes、weights 和总工作量。未启用的
   background tasks 通过 dependency 延迟到 target lane tail 之后，不能直接删除，
   避免改变 packed-weight working set 或调度元数据。
6. `head` 保持 target 位于 lane head；`after_1` 保持 target 位于一个
   68-route task 之后。需要从 trace 验证 target 开始时 peers 实际处于哪个 phase，
   不能仅由 task 顺序推断。
7. `aggressor_count=0` 是 matched isolated control；它不应产生伪造的
   same/cross-LLC 差异。
8. rank-wide/split mode 要预先固定 placement 规则，例如前一半放 LLC6、
   后一半放 LLC7；不能根据结果临时改变分配。

同步更新
[test_moe_gather_injection_overlap.py](../../../tests/test_moe_gather_injection_overlap.py)，
至少覆盖 aggressor count 边界、CPU placement、dependency、任务总量不变、
输出 schema 和非法参数拒绝。

## 测量协议

每个 session 必须使用：

- 5 次 warmup；
- 31 个 randomized paired trace rounds；
- 4 个轮换 measured packed-weight copies；
- 每个 sample 前使用独立的第 5 个 packed copy 做严格 scrub；
- 所有 modes 在每轮内随机顺序执行；
- 两个完全独立的 sessions，使用不同 seed；
- 固定 affinity：`taskset -c 240-319 numactl --membind=3`。

不要恢复没有显式 scrub 的旧 benchmark 方法。SPE 可用于检查 cold-weight
source，但不能把当前 sampled L3-hit 比例解释成无偏 DRAM byte fraction。

每个 `(count, placement, phase)` 至少序列化：

- target absolute span；
- target gather、W13、W2 phase time；
- peer 与 target 的分阶段 overlap core-ms；
- target 开始时 active peer count，按 gather/W13/W2 分组；
- 实际发生 overlap 的 expert 数；
- 相对 matched isolated 的 paired delta、P10/P50/P90 和置信区间；
- same-LLC minus cross-LLC paired delta；
- calibration、extension、输出 artifact 的 SHA256；
- logical CPU 到 physical CPU、NUMA node、LLC domain 的映射。

在任何拟合前，先完成数据质量检查：

- 两个 session 的 delta 方向是否一致；
- target 开始时的 peer phase 是否符合预期；
- aggressor count 是否随 trace 中的 overlap count 单调变化；
- isolated control 是否稳定；
- 是否存在 affinity、LLC 映射或 trace 丢行问题；
- stage 时间必须用 stage envelope，不能把多线程记录求和成 core-ms 后当延迟。

若这些检查不通过，停止拟合，先修 probe。

## 数据切分：拟合前锁死

### 可用于拟合

- 新 absolute-pressure probe 的 session 1：参数拟合；
- 新 absolute-pressure probe 的 session 2：同族 validation，不参与参数选择；
- 已接受的 isolated phase floor/scale 结构及其已冻结参数。

建议新 artifact 名称：

```text
tmp/moe_gather_absolute_pressure_fit_<YYYYMMDD>.json
tmp/moe_gather_absolute_pressure_repeat_<YYYYMMDD>.json
```

拟合脚本必须在读数据前按 SHA256 检查 fit、validation、holdout 路径互斥，
不能仅依赖文件名。

### 永久 pure holdout：禁止调参

旧 route/context artifacts：

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_small_expert_context_scrub_20260903.json` | `61c0e929ad7a575831f972bdbb2c690df243e028062cdbb10044e66d15867e34` |
| `tmp/moe_small_expert_context_scrub_repeat_20260903.json` | `90f34a8498a2a3ad1abfdae69b21ff4b9ab97f08cef1192dcc62c23bf8c07d4f` |
| `tmp/moe_small_expert_context_scrub_m2_20260903.json` | `be6b83475360c330ba4e053c90f19fc162465228eef835661b692fac569ca1a1` |
| `tmp/moe_small_expert_context_scrub_m5_20260903.json` | `277d3896e5d04bafedbf931edba38c0ff29ea6a6b9ad542ad3ff5d78d261df74` |
| `tmp/moe_small_expert_context_scrub_m6_20260903.json` | `28cf59e56d6c281df3245fcc941e19885496ec6e7df9bb4b7f3634d59e1b4376` |
| `tmp/moe_small_expert_context_scrub_m12_20260903.json` | `81c195d34902e0d9002a59f85d0c8bccfc884a6395bacb22031aa69a5c01b0dd` |

三条 real-trace measured shortlist：

| Trace | Artifact | SHA256 |
| --- | --- | --- |
| high-skew | `tmp/pairwise_high_skew_holdout_20260902.json` | `b23ece71ffecbbbc839747ac4be679b34c02c7998a3737764aeee63964ddb67f` |
| median | `tmp/pairwise_median_holdout_20260902.json` | `fe103f49f816d930b12c021e2ca5d8966bbf0283670c61f245f3464f5d3f92d3` |
| uniformish | `tmp/pairwise_uniformish_holdout_20260902.json` | `bba234ae99508af9fc9f7026b43150e8c6e09f50026049e53f4392a4d908a8d8` |

这些数据只能在 candidate calibration 已冻结并写出 SHA256 后读取一次，不能因
holdout 结果修改参数、bucket 或 uncertainty margin。

## 模型拟合顺序

1. 固定已接受的 gather/W13/W2 phase floor/scale，不在本轮重新拟合。
2. 保持旧 `wide_team_pressure` 与
   `narrow_team_contention_correction` 为 identity/zero。
3. 不使用 whole-expert total residual 掩盖 phase residual。
4. 用新 probe 的所有绝对 slowdown 行和 locality contrast 行，联合拟合：
   - per-LLC-domain injection capacity；
   - gather offered-rate 或 gather/stream coupling。
5. 不允许只拟合 `same_llc - cross_llc`。absolute 与 contrast 应在目标函数中
   分别报告误差，避免样本数多的一类完全支配另一类。
6. 对参数做 profile/grid identifiability 检查：
   - 最优点不能贴搜索边界；
   - 参数邻域应有可辨识的 loss 曲率；
   - 不应存在相差数倍的参数组合却给出近似相同预测。
7. 只有 residual 在两个独立 session、多个 aggressor counts 上保持同号且随物理
   context 系统变化，才允许增加一个新 residual term。不得从单点反例加参数。
8. `effective_gather_traffic_multiplier` 只能称为 effective coupling，不能解释为
   实测 byte ratio。

相关实现：

- [analytic_model.py](../../../cpu_moe_schedule_optimization/cost_model/analytic_model.py)
- [fit_absolute_pressure_calibration.py](../benchmarks/fit_absolute_pressure_calibration.py)
- [ablate_rank_dram_domain_scope.py](../benchmarks/ablate_rank_dram_domain_scope.py)
- [fit_phase_reaccount_calibration.py](../benchmarks/fit_phase_reaccount_calibration.py)
- [evaluate_phase_reaccount_holdout.py](../benchmarks/evaluate_phase_reaccount_holdout.py)
- [rescore_measured_neighborhood_states.py](../benchmarks/rescore_measured_neighborhood_states.py)

如果修改 cost-model 结构、calibration schema、资源方程或 pruning 语义，必须按
仓库规则先做 impact analysis，并同步更新
[MATHEMATICAL_MODEL.md](../../../cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md)。

## Freeze-once 与最终 holdout

拟合及 session-2 validation 通过后：

1. 写出 candidate calibration；
2. 写出完整 fit/validation report；
3. 记录 candidate、输入 artifacts、extension 的 SHA256；
4. 关闭拟合路径；
5. 此后才允许读取 route/context 与 real-trace holdout。

当前生产外基线仍是 frozen v8：

```text
remote path:
bench_assets/moe_paper/arm_codex_numa3_80c_temporal/
analytic_machine_numa3_80c_narrow_merge_v8_20260903.json

SHA256:
7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3
```

必须报告以下结果：

- 25 个 route/context 点的 absolute MAPE、median、P90、maximum；
- isolated gather/W13/W2/total phase MAPE；
- hardware-resolvable pair 的 direction accuracy 与 false dominance；
- 每条 real trace 的 Spearman、measured-best top-K recall、false pruning、
  false dominance；
- model shortlist 中 hardware measured best 与 candidate 的实际 regret；
- 三条 real trace 的所有旧 measured hashes 是否被完整恢复。

上次结果仅作为拒绝线和参考基线：

| Metric | frozen v8 / accepted reference | rejected candidate |
| --- | ---: | ---: |
| 25-point absolute MAPE | 21.72% | 273.97% |
| isolated W13 MAPE | 1.46% | - |
| isolated W2 MAPE | 9.18% | - |
| isolated total MAPE | 3.69% | - |
| route resolvable-pair false dominance | - | 2 / 48 |
| real-trace false dominance | - | 16 total |

## 预先声明的通过标准

只有同时满足以下条件，才可以提出替换 frozen v8：

1. route/context absolute MAPE 严格低于 frozen v8 的 `21.72%`；
2. isolated W13、W2、total 不明显回退于 `1.46% / 9.18% / 3.69%`；
3. route/context 和三条 real trace 上的 false dominance 均为 0；
4. 三条 real trace 的 measured best 均进入 model top-8；
5. 所有 hardware-resolvable 的被剪枝关系中 false pruning 为 0；
6. candidate 选中方案相对各 trace measured best 的回退不超过 2%；
7. session-2 的 absolute 与 contrast residual 没有系统性同号偏差；
8. 参数可辨识且不贴搜索边界。

Spearman 作为诊断报告，但不能代替 top-K、false pruning 或实际 regret。

任何一项失败都应拒绝 candidate、保留 frozen v8，并记录失败机制。禁止查看
holdout 后继续微调再重复宣称 holdout。

## 明确禁止事项

- 不得把 `0.787`、`22.26`、或联合拟合点 `0.78`/`0.25` 写入可用 calibration。
- 不得把 $\alpha_g$ 搜索上界扩到 22.26，也不得再只拟合 `same_llc - cross_llc`。
- 不得重新启用旧 `wide_team_pressure` 或
  `narrow_team_contention_correction` 来快速压低 total residual。
- 不得用旧 route sweep、三条 real trace 或 aggregate pairwise report 拟合。
- 不得用 total residual 掩盖 gather、W13、W2 的分阶段误差。
- 不得把同一 LLC 内的 CPUs288-302 当作 cross-LLC 对照。
- 不得在通过上述 gate 前接入 VND/LNS 或改变默认 planner 行为。
- 不得仅因为 top-1/top-K recall 看起来好就忽略 false dominance。

## 建议执行顺序与完成定义

按以下顺序推进：

1. ~~扩展 probe CLI、输出 schema 和单元测试。~~ 完成。
2. ~~在本地跑 focused tests；同步到 Arm-codex 前按 remote-execution 文档检查路径。~~ 完成。
3. ~~在 Arm-codex 收集两个独立 session。~~ 完成。
4. ~~做数据质量检查；不合格则只修 probe，不开始拟合。~~ 完成。
5. ~~用 session 1 拟合、session 2 validation，并完成 identifiability 检查。~~
   完成并拒绝冻结。
6. ~~不要冻结当前 candidate。下一步是 rank DRAM vs domain-only 的离线消融。~~
   完成：rank DRAM 只解释约 49% 远程共模，不新增结构。
7. ~~同一锁定 DAG 上的 rank LLC vs domain LLC 消融。~~ 完成：rank LLC 解释
   剩余远程共模的 100%，domain LLC 仍不饱和；不新增结构。
8. ~~同一锁定 DAG 上的 victim-asymmetric dilation 消融。~~ 完成：own-demand
   把 n15 打到 `+0.001/0 ms`，没有任何臂同时满足三门槛；不新增结构。
9. ~~aggressor-$M$ occupancy probe 代码与 focused tests。~~ 完成：
   `bench_aggressor_m_occupancy.py`，victim $M=1$，count 0/1/4，
   aggressor $M\in\{1,4,16,68\}$。本 probe 不加结构。
10. ~~Arm-codex aggressor-$M$ session-1。~~ 完成：n4 税随 $M$ 下降，
    不新增结构。
11. ~~fill-port vs stream：同一 LLC7 16 核上对照 1×16T 与 16×1T。~~ 完成：
    same-LLC head 为 $+0.018/+0.014/+0.160\,\mathrm{ms}$，`fill_ports` 不成立，
    预声明 `one_stream` 差 $0.054\,\mathrm{ms}$ 未过门槛；不新增结构。
12. ~~stream-count composition：等线程阶梯与 `8+8+4+1` vs 4×1T vs 21×1T。~~
    完成：签名 `stream_count`；4×4T $+0.075$ 贴近 4×1T $+0.070$，mix $+0.110$
    贴近四起点 1T $+0.103$，不是 21×1T $+0.301$；不新增结构。
13. 此后才允许一次性读取 route/context 与三条 real-trace holdout。
14. 按预声明 gate 做 accept/reject 决策。
15. 只有 accept 后，才另开任务评估接入 partial-order comparator 和 VND/LNS。

最小本地验证命令：

```bash
PYTHONPATH=.:src .venv/bin/python -m pytest tests/test_moe_stream_count_composition.py
PYTHONPATH=.:src .venv/bin/python -m pytest tests/test_moe_fill_port_vs_stream.py
git diff --check
```

本任务完成时，下一位 agent 应交付：

- ~~probe 代码与 focused tests；~~ 完成。
- ~~两个新 raw artifacts 及 SHA256；~~ 完成。
- ~~数据质量报告；~~ 完成。
- ~~session-1 联合拟合与 session-2 validation / identifiability 报告；~~
  完成并拒绝冻结。
- ~~rank DRAM vs domain-only 离线消融；~~ 完成：不新增结构。
- ~~rank LLC vs domain LLC 离线消融；~~ 完成：不新增结构。
- ~~victim-asymmetric dilation 离线消融；~~ 完成：不新增结构。
- ~~aggressor-$M$ occupancy probe 与 tests；~~ 完成。
- ~~Arm-codex session-1；~~ 完成：n4 税随 $M$ 下降，不新增结构。
- ~~fill-port vs stream probe 与 Arm session；~~ 完成：1×16T 贴近 1T 而不是
  16×1T，预声明仍为 inconclusive，不新增结构。
- ~~stream-count composition probe 与 Arm session；~~ 完成：签名
  `stream_count`，`8+8+4+1` 按 4 流而不是 21 线程；不新增结构。
- ~~unified weight-block probe 与 Arm session；~~ 完成：签名
  `layout_neutral`，16×1T leftover split/unified $+0.153/+0.157\,\mathrm{ms}$；
  不新增结构。
- ~~weight THP probe 与 Arm session；~~ 完成：签名 `page_neutral`，smaps 4
  KiB/THP AnonHugePages $0/100\%$，16×1T leftover $+0.167/+0.162\,\mathrm{ms}$；
  不新增结构。
- 下一步：不要从 $M$ sweep、fill-port、stream-count、unified-block 或 THP
  probe 加结构。仍不要读 holdout。
- 明确的 accept/reject 决策；当前结构仍为 reject。
- leftover 物理图像：同 LLC 上并发 transfer-bound 流的争用，不是 68-route
  字节利用率，也不是与 $M$ 无关的占位费。

当前 worktree 已包含未提交的 cost-model、benchmark、test 与文档修改；下一位
agent 必须先运行 `git status --short`，不得覆盖或回退这些已有改动。
