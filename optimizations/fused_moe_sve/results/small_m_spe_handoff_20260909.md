# Small-M memory-pressure modeling / SPE audit handoff

本文件用于新 session 接续，不需要重跑已完成的实验来恢复上下文。
更新时间：本地 2026-09-08；最新实验目录按远端日期命名为 20260909。

## 1. 目标与当前结论

长期目标：建立有理论结构和独立实验证据的小 M 性能模型，解释不同背景
访存压力、同/异 LLC 放置下的时间变化，而不是继续堆 total residual 拟合项。
先关注供给、复用、并发和重叠的可辨识性；不能把系统总 DDR 指标直接当成
前台任务的服务延迟。

最近一项明确任务：解决 SPE 控制延迟污染，精确过滤 kernel 指令范围，检查
采样损失和 profiling 开销。

**最近这项审计已完成；长期物理模型未完成。**

- perf ACK 已移到 PREP 之前；新增原生 `PREP_GO`，执行前不再经过 Python/perf 握手。
- 精确 IP/EL0/AUX CPU/TID 过滤及已记录数据的完整性检查通过。
- 两场独立实验确认：period 1024、4096 的 SPE 开销很大且依赖 kernel/背景。
  两档均不能作为本场景的低扰动校准来源。
- 没有新增物理参数、重新拟合 calibration、改生产 kernel 或接入 planner。
- 本轮实验与分析进程均已完成，没有需要接管的活动终端 session。
- 没有提交 commit。工作区有大量既有修改与未跟踪文件，必须保留。

## 2. 为什么走到 SPE：需要保留的已有证据

优先阅读以下结果，避免重复已排除的解释：

1. [冻结分层模型](layered_measured_supply_20260908.md) 与
   [高压力独立重复](layered_high_pressure_repeat_20260908.md)：高压力区偏差
   跨 session 重复，不能仅归因于随机争抢。balanced48 持续预测偏慢约13–14%，
   balanced78 持续低估约15–16%；不要拿这些 holdout 回去重新拟合。
2. [固定总数放置对照](fixed_total_placement_20260908.md)：50 个 M12/1T 后台，
   25/25、38/12、12/38 总 DDR 带宽都约279–280 GB/s，但前台真实 W13 时间不同。
   相比25/25，38/12 两场慢15.61%/16.53%，12/38 快约28%。38/12 的 LLC miss
   量、L2 refill 量和总 DDR 带宽变化很小，仍有明显时间惩罚。
3. [HHA 来源映射](hha_source_mapping_20260908.md)：用于识别流量来源，不能将
   `rx_sccl` 解释为前台等待时间；流量计数不等于排队延迟。
4. [DDRC 时钟换算](ddrc_clock_scale_20260908.md)：转换 occupancy/read-command
   的时间尺度后，仍不能解释高压力下全部增长。它依然是控制器聚合 proxy，
   不是 foreground latency，更不能倒推一个独立测得的 MLP。

其他已完成、按需阅读的排查：

- [依赖加载服务](dependent_load_service_20260908.md)
- [load-PC 几何](load_pc_geometry_20260908.md)
- [相同 SVE 代码重定位](relocated_sve_supply_20260908.md)
- [single-K4 访存流](single_k4_stream_20260908.md)

这些是机制约束/负结果，不应重新包装成已识别的生产修正项。

冻结的 measured-feature 模型（不要与生产 v8 calibration 混为一谈）：

```text
tmp/layered_supply_model_20260908/frozen_fit.json
SHA256 f3796a0f7e815dc5c5e297b3bafdb3bb64150358daa50150f91bec6897157d86
```

该文件本地存在，handoff 时已重新核对 SHA256。其特征来自执行时测量，不是
可直接给离线邻域搜索使用的 plan-visible predictor。

## 3. 最新 SPE 审计结果

权威详细记录：[spe_control_audit_20260909.md](spe_control_audit_20260909.md)。
失败尝试与旧协议留档：[spe_capability_20260908.md](spe_capability_20260908.md)。

### 实验参数

| 项目 | 冻结设置 |
| --- | --- |
| 前台 | CPU304，M1/1T，K4096、N1024，SVE256/BF16 |
| Probe | 1=B-only；4=full-no-store；不是完整 MoE/real-trace replay |
| 背景 | condition0=isolated；950=38同LLC＋12异LLC的真实M12/1T后台 |
| CPU / memory | NUMA3；允许CPU240–319；controller240 |
| 权重 / workspace | 同进程同批 allocations；persistent预分配/触碰workspace；4-copy rotation |
| 准备 | 两个LLC各256 MiB scrub；后台启动后 nominal 5 ms lead-in |
| 模式 | period0=无perf；-1=挂载但禁用；1024、4096=活动采样 |
| SPE | `load_filter=1,jitter=1,ts_enable=1/u`；`-m 256,4096` |
| 重复 | 每组合5轮warmup＋31轮测量；每轮随机交错；同round/copy配对 |
| 独立session | session1 seed19508；session2 seed19509；两场使用相同二进制 |

系统页大小4096字节，AUX配置对应16 MiB。THP policy为always；session1整个
进程观察到3,141,632 kB AnonHugePages、无显式HugeTLB。不是每份权重的页类型证明。

两场各576个cell，共1152个：160 warmup、992 measured，其中496个活动SPE测量。

### 结果摘要

- 原生准备结束→gate的分组中位数142.05–516.06 us；最大单次1122.23 us。
  该指标包括线程唤醒，不等同于旧Python PREP→GO；不能计算一个所谓严格的
  “控制延迟加速比”。旧的207.69 ms post-PREP perf ACK已从路径中移除。
- B-only精确代码长度264字节；full-no-store为340字节。AB（probe3）元数据
  额外验证通过，长度276字节，但没有做本轮AB profiling sweep。
- 496份正式采样文件全部通过当前capture checks；113,112条kernel记录被保留，
  约98.01%的范围外记录被排除。
- 未观察到reported loss/error、非零/无法解析的AUX flags、错误CPU/TID、缺失
  kernel TOT/ISSUE、截断/孤立记录、包偏移或AUX长度不一致；decoder stderr为空。

以下为配对中位额外耗时，两列各按S1/S2列出；**不是cost model预测误差**：

| Probe / 背景 | period1024 | period4096 |
| --- | ---: | ---: |
| B-only / isolated | +45.60% / +47.64% | +16.99% / +17.49% |
| Full-no-store / isolated | +132.18% / +129.79% | +58.27% / +62.15% |
| B-only / 38+12 | +17.32% / +17.34% | +6.90% / +7.29% |
| Full-no-store / 38+12 | +48.87% / +47.06% | +22.74% / +24.82% |

挂载但禁用：各组配对中位变化-0.34%至+1.65%。活动采样的巨大变化不能由
“只是挂载perf”解释。完整绝对时间、噪声、95% bootstrap区间见详细报告和JSON。

period也影响原始分布：B-only isolated的TOT中位值，S1为149.5→69，S2为
164→68（1024→4096）。这里只报告raw值，不将其解释成ns或DRAM latency。

### 不允许越过的推断边界

- 已记录AUX完整，不等于硬件无sampling bias/collision，也不等于每条load都被记录。
- `perf report -D`仍报TIME_CONV unhandled；本方法不依赖跨cell时间戳转换。
- 旧`perf script`还提示缺少CONTEXT，不能直接信任其合成TID/time归属。
- 安装的perf提示`PID/TID switch overriding CPU`：`-C`不能作为独立过滤保障。
  实际依赖`-t`、worker固定CPU304、原始AUX identity和精确函数范围。
- TOT/ISSUE/XLAT单位与物理含义仍需验证；未知latency index6保留为unnamed/raw，
  不能直接命名为memory latency。原始包保留了其编码。
- 采样开销包含活动profiling整体干预，不是已唯一识别出的SVE端口/中断/AUX带宽成本。
- 不能统一扣掉一个固定开销百分比；不能用这批instrumented时间/latency拟合正常执行模型。
- 窄化采样窗口可能减少准备阶段采集，但不能假设它解决kernel内部采样开销。

## 4. 重要文件与实现入口

以下路径相对本地仓库根目录：

| 文件 | 用途 |
| --- | --- |
| `optimizations/fused_moe_sve/benchmarks/bench_spe_audit.py` | 当前runner；pre-arm、PREP_GO、随机交错、配对/rotation、增量JSONL |
| `optimizations/fused_moe_sve/benchmarks/analyze_spe_audit.py` | 精确IP解析、AUX字节ledger、capture_failures、配对开销和原始分布 |
| `optimizations/fused_moe_sve/benchmarks/phase_supply_native.cpp` | 新增`SPE_INFO`、`PREP_GO`；旧PREP/GO保留；`Run`可记录prepared_to_gate_ns |
| `optimizations/fused_moe_sve/benchmarks/bench_spe_cell.py` | 旧能力验证runner；当前audit复用control/stop/victim_tid等helper；不要删除 |
| `optimizations/fused_moe_sve/benchmarks/bench_phase_supply.py` | 原有协议、condition映射、正确性/背景核对helper |
| `optimizations/fused_moe_sve/benchmarks/m1_supply_probe.h` | byte-identical Lab Generator；代码长度来源 |
| `tests/test_moe_spe_audit.py`、`tests/test_moe_spe_cell.py` | 当前直接回归测试 |
| `optimizations/fused_moe_sve/manifest.yaml` | feature `measurement.spe_capability`；仍experimental，下一决策是稀疏采样审计 |

源代码新旧关系：当前`SPE_INFO`输出真实函数地址、size、code_hex、byte_match，
拒绝不支持的probe；不是扫描RET猜长度。当前runner支持probes1/3/4，不支持0。
`dual`协议只支持W13形状；检查W2必须使用原始stage协议，不要修改guard来跑通。

历史失败（均已保留，不应重试旧目录）：

- `native_cells`：event语法`/:u`错误；已改为`/u`。
- `native_cells_v2`：ACK终止符处理；现在兼容newline/NUL并保留非法回复错误。
- `native_cells_v3`：主动SIGINT导致perf返回-2，被误判失败；现在仅在自己请求
  stop时允许这个退出码，其他错误保留，文件有效性另由审计判断。
- `native_cells_v4`：能力采集成功，但post-PREP ACK严重改变背景lead-in。
  **不要将其作为正式耗时数据。**

## 5. 环境、产物和复现

本地root：`/Users/zhangxu/Codes/vllm-aarch64/fused_cpp`。
远端：`Arm-codex-internal`；root `/home/zhangxu/codex/fused_cpp`。
Python用各自项目`.venv/bin/python`，不是系统Python。

远端完整产物：

```text
tmp/spe_control_audit_20260909/
  phase_supply_native             # 已编译并验证的二进制
  *.py / *.h / phase_supply_native.cpp  # 本轮运行的Lab源文件快照
  smoke/
  session1/
  session2/
  audit_summary.json
```

每个正式session中有metadata.json、cells.jsonl、complete.json、native.stderr、
逐cell `.data`/`.perf.log`，正式采样的`.dump.txt`/`.decode.stderr`/`.decode.ok`。
新输出采用exclusive-create；不要覆盖旧目录或总结文件。若dump存在但没有
decode.ok，解析器会拒绝复用：保留失败产物，在新解码路径处理，不能伪造ok标记。

本地已取回：`tmp/spe_control_audit_20260909/audit_summary.json`、
`session1_cells.jsonl`及少量smoke验证文件。**完整原始采样主要在远端，别误认为
本地tmp已包含全部.data。** 这些大产物不应加入源码commit。

身份：

```text
HEAD c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5 + dirty Lab changes
native SHA256
489e13e96c5b244252a401d28b6820bb11847f37892ce6a77a555e0caf0b954b
unchanged production csrc/moe/arm/sve_bf16/jit_kernels.cpp SHA256
1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629
GCC 13.2.0; C++17/O3/pthread; armv8.2-a+bf16+sve; SVE256
```

详细报告有完整build、collect、analyze命令。不要照抄已完成session1/session2的
输出路径启动新run。仅恢复理解时读取现有summary即可；不要自动重编译/重跑。

本轮已有验证证据：

```sh
.venv/bin/pytest -q tests/test_moe_spe_cell.py tests/test_moe_spe_audit.py \
  tests/test_moe_phase_supply.py tests/test_moe_dual_m12.py
# 70 passed
```

Ruff、git diff --check通过。原生旧PREP/GO的W13/W2数值检查、AB元数据、非法
SPE_INFO拒绝均通过。复用这些结果要说明是已有证据；若修改相关代码则重跑。

## 6. 未完成事项和建议下一步

下列都是**未运行/未验证的下一阶段**，不要写成已有结论：

1. 先做稀疏SPE最小网格，例如period16384/65536；这只是候选值，不是已验证配置。
   保留无perf和挂载禁用两种控制，优先复用B-only/full-no-store、isolated/950，
   同进程随机配对、4-copy、scrub、两场独立session。先小规模检查，后正式重复。
2. 在看结果之前声明允许的profiling开销、样本量与稳定性标准。需要同时看
   配对开销区间、每cell有效kernel样本、跨period与跨session分布、capture failures。
   不要事后降低标准来“通过”。
3. 稀疏采样若开销仍大，或有效样本过少/分布不稳定，停止将SPE作为定量校准来源。
   不要为了堆样本立即扩大所有M/width/background组合。考虑独立的低扰动
   kernel-native PMU/服务探针路线，但该替代路线尚未实施、不能宣称一定有效。
4. 只有低扰动与采样有效性成立，再验证计数器单位/语义和按指令类型的归属。
   后续可先回到固定总数25/25与38/12反例，量化前台供给变化；不要直接跳到
   完整planner或把全局queue ratio改名为前台latency。
5. 最后才提出新的物理项：单独训练探针、冻结参数、独立session及未见数量/
   placement holdout。原有反例与冻结模型数据不能悄悄回流参与拟合。

当前没有因授权而悬置的操作：用户已授权本次远端同步/实验且已完成。
后续新session仍应确认机器连通性；不推定任何未来连接一直可用，不改变机器/
NUMA协议来绕过连接问题。交接请求本身不要求立即启动上述下一阶段。

## 7. 新 session 起步顺序与仓库约束

1. 读本文件和最新SPE详细报告，再读`AGENTS.md`及相关topic文档。
2. `git status --short`：大量dirty/untracked是用户和历史实验资产，不reset、不清理。
   未提交代码在此工作区中，不能仅checkout HEAD就恢复本轮实现。
3. 读本地audit_summary.json确认两场各248个decoded正式采样且failure列表为空；
   按需读取上一阶段固定50与分层模型报告。不要重做已完成审计。
4. 若用户要求继续实验，再按第6节执行有边界的下一步，先不改模型。

约束：中文沟通；CodeGraph先用于结构定位，rg用于文字/配置；修改前按仓库
impact-analysis/test-selector/code-review-gate流程。远端前读
`docs/agent_remote_execution.md`，性能工作读`docs/agent_benchmark_hygiene.md`。
Lab变化遵循`docs/change_policy.md`与`docs/agent_optimization_governance.md`。
若真正改变模型/资源/剪枝语义，必须同步MATHEMATICAL_MODEL.md；本轮未做该类改变。
不要擅自提交/push；不要引入生产依赖或全局构建变动。没有必须使用subagent跑
实验的限制；当前规则不允许无明确请求主动派生subagent。

一句话交接：**已经拿到可信归属、完整记录的SPE样本，但采样严重干扰小M；
下一步先判断更稀疏采样能否成为低扰动工具，而不是把现有latency拿去拟合模型。**
