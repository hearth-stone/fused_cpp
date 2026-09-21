# 单 expert 8T 完整路径对照（2026-09-10）

## 结论

两份真实输入各两轮已完成。用第一轮完整expert实测预测第二轮，总时间MAE4.918us、
MAPE1.035%；已有精确块分开模型补上相同的第一轮非GEMM开销后为15.348us、2.418%。
同M跨输入22个方向性对照MAE5.688us、MAPE1.421%。完整路径标定在本组条件更准。

但已有精确块模型在真实无竞争路径中W13/W2 MAPE仅3.446%/1.842%，远小于此前
联合trace中W13约29–33%的高估。分开测量存在小幅基线偏差，并非已证实的联合大误差主因。
本实验没有把联合竞争唯一归因到某个机制；目标起始状态、placement与联合任务历史仍不同。
实际获益来自更贴近真实路径的标定，不证明必须把阶段模型合并成不可分解的函数。

## 协议、范围与验证

E类Lab采集/诊断；生产kernel、默认planner与接受的Lab基线均不变。复用
prepare_workspace_isolated_width.isolate_bridge：目标expert移至首位，其他所有expert等其完成，
保留剩余依赖链、完整路由、输入/权重/输出对象。80线程运行时仍在，不代表纯8线程进程。
目标真实执行gather→W13/SiLU/packC→W2。early merge关闭，最终merge不计入expert包络。

机器Arm-codex-internal，目录/home/zhangxu/codex/fused_cpp，NUMA3，rank CPU240–319，
目标8T CPU312–319；H4096/F512、E256、TP4、2048tokens/TopK6，BF16输入/FP32输出。
Ntile16，全W13/W2权重8MiB/4MiB，每线程1MiB/512KiB，几何(8,0,0,1,1)。
现有release扩展复用，无新编译；其SHA与冻结校准身份匹配。
环境OMP_NUM_THREADS=1、OMP_DYNAMIC=FALSE、OMP_PROC_BIND=false、MKL_NUM_THREADS=1、
OPENBLAS_NUM_THREADS=1；worker实际宽度由native计划设为8，不能从OMP变量误读为1T。
固定预触碰workspace，未另启HugeTLB；四份权重轮换，每次216MiB scrub，5warmup+31随机配对轮，
独立seed610081/610082。profile协议与workspace身份一致。

median M=8/12/13/14/16/20/24/25/26/36/48/60；uniformish将48换40。
每个M选择真实expert，优先原计划8T，统一目标placement；原位置在frontier中保留。
源输入为tmp/planner_repaired_pressure_20260910，未改route_file、layer或权重准备方法。

四次运行各521个trace调用全部解析验证，总2084调用；其中1920次隔离目标调用，
正式测量1488个隔离目标观察（24条件×31×2），另有anchor及正确性/warmup。
四份权重的poisoned-workspace bitwise校验全部通过；原始trace另审计实际CPU312–319、
三阶段每线程顺序及early merge零事件，最终merge均在scheduled_compute结束后。
每个隔离调用均通过其他expert不得与目标重叠检查。

## 留出比较

第一轮完整expert中位数作为精确M查表预测第二轮；没有用第二轮拟合。
已有split预测在采集前冻结。以下24条件均为第二轮31次样本的中位数，误差跨条件等权。

| 方法 | W13 MAE / MAPE | W2 MAE / MAPE | GEMM和 MAE / MAPE | 完整包络 MAE / MAPE |
|---|---:|---:|---:|---:|
| 原analytic（比例尾块） | 63.390us / 19.282% | 32.449us / 19.833% | 96.463us / 19.549% | 154.789us / 28.484% |
| 已有精确块模型 | 13.253us / 3.446% | 3.410us / 1.842% | 15.908us / 2.781% | 15.348us / 2.418% |
| 完整路径第一轮标定 | 1.322us / 0.391% | 0.755us / 0.472% | 1.621us / 0.313% | 4.918us / 1.035% |

精确块模型的完整包络列额外加入第一轮测得的非GEMM包络剩余量，是公平归账诊断，
不是旧模型已能独立预测gather。analytic总时间列保持其原operator项（本组8T为0），
所以它还包括缺失gather/间隙的影响，不能与GEMM误差混为一谈。
GEMM和为每次W13/W2阶段包络之和；完整包络为最早gather开始到最晚W2结束。
非GEMM剩余是同次总包络减两个阶段包络，包含gather、到达偏斜及间隙，不称纯gather成本。
中位数不满足加法分配律，各列不能直接按中位数机械相加。

| case | 分开模型完整包络 MAE | 完整路径标定 MAE | 后者MAPE |
|---|---:|---:|---:|
| median | 13.651us | 5.033us | 1.135% |
| uniformish | 17.044us | 4.803us | 0.936% |

## 逐条件总时间与回退

单位us。“分开”已补同一第一轮非GEMM开销；“完整”是第一轮包络预测。

| case | M | 分开预测 | 完整预测 | 第二轮实测 |
|---|---:|---:|---:|---:|
| median | 8 | 222.90 | 230.94 | 222.23 |
| median | 12 | 293.62 | 290.36 | 296.47 |
| median | 13 | 461.11 | 453.17 | 451.73 |
| median | 14 | 453.95 | 446.06 | 446.60 |
| median | 16 | 471.80 | 460.09 | 456.75 |
| median | 20 | 496.76 | 489.30 | 476.48 |
| median | 24 | 545.24 | 534.34 | 539.28 |
| median | 25 | 710.52 | 678.81 | 676.82 |
| median | 26 | 704.98 | 677.12 | 683.69 |
| median | 36 | 786.76 | 771.76 | 781.74 |
| median | 48 | 1029.07 | 1014.10 | 1013.47 |
| median | 60 | 1278.60 | 1248.61 | 1251.93 |
| uniformish | 8 | 234.73 | 236.56 | 242.65 |
| uniformish | 12 | 294.21 | 293.41 | 301.18 |
| uniformish | 13 | 455.99 | 450.55 | 452.42 |
| uniformish | 14 | 465.82 | 459.22 | 456.77 |
| uniformish | 16 | 466.50 | 459.46 | 459.76 |
| uniformish | 20 | 504.52 | 491.61 | 494.22 |
| uniformish | 24 | 548.39 | 540.69 | 538.33 |
| uniformish | 25 | 713.95 | 681.80 | 684.34 |
| uniformish | 26 | 708.63 | 684.48 | 679.04 |
| uniformish | 36 | 793.62 | 788.32 | 775.99 |
| uniformish | 40 | 961.19 | 922.62 | 912.86 |
| uniformish | 60 | 1277.28 | 1248.40 | 1252.52 |

完整标定并非逐点都更好：median M8/M12/M36与uniformish M12总时间绝对误差比上述分开对照大。
M12 gather两轮median43.35→47.85us，uniformish44.10→51.38us，仍有跨轮变化。
所有条件/轮次中W13最大CV3.66%、W2最大CV5.33%、总包络最大CV18.26%（median M12第二轮），
未删除异常样本或调整统计口径；这里报告的是31次中位数预测，不保证单次us级精度。

## 迁移与决策

使用另一份输入第一轮、相同M的完整expert时间预测本输入第二轮，11个共同M双向共22点，
MAE5.688us/MAPE1.421%、最大绝对误差15.13us。该对照属于补充的同M跨输入诊断；
仍复用同一权重准备协议/物理placement，不能宣称未测M或所有真实状态泛化。

保留完整路径采集作后续无竞争标定来源，同时保留阶段计时和精确块结构以解释/插值。
不在本次自动替换planner或重搜；没有联合竞争拟合，也没有硬件计算提速声明。
此前8T联合高估应继续在固定实测无竞争基线后检查竞争增量，避免将总体残差直接回灌基线。

## 身份、产物和复现

HEAD c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5加原有dirty tree，状态保存于实验目录。
MoE扩展SHA dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f。
workspace SHA 7e0124673b4742a4747f6b4b733cb364def2f455cc5919ef13379eca4cb098db。
实际复用runner SHA ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba，
路径tmp/workspace_phase_timeline_20260907/runner/bench_bounded_order_extension.py。
冻结来源、frontier、校准与原始trace身份在frozen.json/session JSON/runtime_audit.json，未改冻结预测。

本地/远端同名产物根tmp/eight_single_expert_20260910：frontier、frozen、四份session元数据、
compact阶段数据、comparison.json逐点样本与指标、runtime_audit.json、run_remote.sh及源码快照。
原始.trace和无损.trace.gz保留在远端同目录，未删除；本地保留compact及元数据。
preparation_record.md保留此前网络阻塞状态；本次重试恢复连接后完成采集。

已执行命令：

```sh
# 远端，同目录run_remote.sh顺序执行两份输入各两轮，内含全部参数与日志路径
bash tmp/eight_single_expert_20260910/run_remote.sh
.venv/bin/python tmp/eight_single_expert_20260910/audit_runtime.py
# 本地
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_eight_single_expert.py --root tmp/eight_single_expert_20260910 --output tmp/eight_single_expert_20260910/comparison.json
.venv/bin/pytest -q tests/test_moe_eight_single_expert.py tests/test_moe_workspace_isolated_width.py
```

15 tests passed；新增分析器/相关测试Ruff通过，git diff --check通过。
实现是Lab准备/分析脚本，复用已有运行器；无生产源、默认值或公共契约变更，无提交。
